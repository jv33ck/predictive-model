#!/usr/bin/env python
"""
Build a team-level training dataset for matchup outcome modeling.

For a given season and date range, this script:

1. Uses ScheduleLeagueV2 (via nba_api_provider.get_schedule_for_date) to
   enumerate games for each date.
2. Uses LeagueGameLog (team) to build *pre-game* team context:
   - Season-to-date games played, win%, points per game, plus/minus per game.
   - Last-N-game (rolling) summary (default N=5).
3. Uses LeagueGameLog to fetch *actual game results* for targets:
   - home_pts, away_pts, margin, total_points, home_win.

Output:
- A CSV (and optional JSON) with one row per game, ready for ML training.

NOTE: This v1 uses *team-level* features only (no player/EPM features yet),
so it is leak-free with respect to future games if you only use games
with GAME_DATE < today in your date range.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import time
from requests.exceptions import RequestException

from data import nba_api_provider


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build team-level matchup training dataset from ScheduleLeagueV2 + LeagueGameLog."
    )
    parser.add_argument(
        "--season-label",
        type=str,
        required=True,
        help="Season label, e.g. '2025-26'. Used to drive nba_api season string.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (inclusive) for training window, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (inclusive) for training window, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--recent-n",
        type=int,
        default=5,
        help="Number of most recent games to use for rolling context features (default=5).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/training/team_matchups_train.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional: path to output JSON file. If empty, JSON will not be written.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers for team features from LeagueGameLog
# ---------------------------------------------------------------------------


def _ensure_game_date_as_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure GAME_DATE column is present and parsed as datetime64[ns].
    Mutates and returns df for convenience.
    """
    if "GAME_DATE" not in df.columns:
        raise ValueError("LeagueGameLog team DataFrame is missing 'GAME_DATE'.")

    # Use pandas type helper to avoid numpy DType typing issues
    if not pd.api.types.is_datetime64_any_dtype(df["GAME_DATE"]):
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def _compute_team_context(
    logs: pd.DataFrame,
    cutoff_date: dt.date,
    recent_n: int,
) -> Dict[str, float]:
    """
    Compute team context features as of *before* cutoff_date (i.e., using
    only games with GAME_DATE < cutoff_date).

    Parameters
    ----------
    logs : DataFrame
        Full LeagueGameLog rows for a single team & season.
    cutoff_date : date
        Games strictly before this date are considered "pre-game" history.
    recent_n : int
        Number of most recent games to aggregate for rolling features.

    Returns
    -------
    dict of team context features.
    """
    logs = _ensure_game_date_as_datetime(logs)

    mask_hist = logs["GAME_DATE"] < pd.Timestamp(cutoff_date)
    hist = logs.loc[mask_hist].sort_values("GAME_DATE")

    games_played = int(len(hist))
    if games_played == 0:
        # No prior history: return zeros/NaNs as a neutral baseline.
        return {
            "games_played": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "win_pct": 0.0,
            "pts_for_total": 0.0,
            "pts_for_per_game": 0.0,
            "plus_minus_total": 0.0,
            "plus_minus_per_game": 0.0,
            "recent_games": 0.0,
            "recent_wins": 0.0,
            "recent_losses": 0.0,
            "recent_plus_minus_total": 0.0,
            "recent_plus_minus_per_game": 0.0,
        }

    wins = float((hist["WL"] == "W").sum())
    losses = float((hist["WL"] == "L").sum())
    win_pct = wins / games_played if games_played > 0 else 0.0

    pts_for_total = float(hist["PTS"].sum())
    pts_for_per_game = pts_for_total / games_played if games_played > 0 else 0.0

    plus_minus_total = float(hist["PLUS_MINUS"].sum())
    plus_minus_per_game = plus_minus_total / games_played if games_played > 0 else 0.0

    # Rolling recent-N context
    recent = hist.tail(recent_n)
    recent_games = float(len(recent))
    if recent_games > 0:
        recent_wins = float((recent["WL"] == "W").sum())
        recent_losses = float((recent["WL"] == "L").sum())
        recent_plus_minus_total = float(recent["PLUS_MINUS"].sum())
        recent_plus_minus_per_game = recent_plus_minus_total / recent_games
    else:
        recent_wins = 0.0
        recent_losses = 0.0
        recent_plus_minus_total = 0.0
        recent_plus_minus_per_game = 0.0

    return {
        "games_played": float(games_played),
        "wins": wins,
        "losses": losses,
        "win_pct": win_pct,
        "pts_for_total": pts_for_total,
        "pts_for_per_game": pts_for_per_game,
        "plus_minus_total": plus_minus_total,
        "plus_minus_per_game": plus_minus_per_game,
        "recent_games": recent_games,
        "recent_wins": recent_wins,
        "recent_losses": recent_losses,
        "recent_plus_minus_total": recent_plus_minus_total,
        "recent_plus_minus_per_game": recent_plus_minus_per_game,
    }


def _get_game_result_for_team(
    logs: pd.DataFrame,
    game_id: str,
) -> Tuple[int, int, int, int]:
    """
    From a team's LeagueGameLog, pull PTS and PLUS_MINUS for a specific game_id.

    Returns
    -------
    (pts, plus_minus, is_win, is_loss) as ints.
    """
    rows = logs.loc[logs["GAME_ID"] == game_id]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly 1 row in LeagueGameLog for game_id={game_id}, "
            f"found {len(rows)}."
        )
    row = rows.iloc[0]
    pts = int(row["PTS"])
    plus_minus = int(row["PLUS_MINUS"])
    wl = str(row["WL"])
    is_win = int(wl == "W")
    is_loss = int(wl == "L")
    return pts, plus_minus, is_win, is_loss


# ---------------------------------------------------------------------------
# Schedule normalization helper
# ---------------------------------------------------------------------------


def _normalize_schedule_df(schedule_df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Normalize raw schedule data so it has GAME_ID, GAME_DATE, and
    HOME_TEAM_ABBREVIATION / VISITOR_TEAM_ABBREVIATION columns.

    This is intentionally defensive because different endpoints / versions of
    ScheduleLeagueV2 may expose slightly different column names.
    """
    if schedule_df.empty:
        return schedule_df

    df = schedule_df.copy()

    # --- GAME_DATE ---
    if "GAME_DATE" not in df.columns:
        for cand in ["GAME_DATE", "gameDate", "GAME_DATE_EST"]:
            if cand in df.columns:
                df["GAME_DATE"] = pd.to_datetime(df[cand])
                break
    else:
        # Ensure it is datetime-like
        if not pd.api.types.is_datetime64_any_dtype(df["GAME_DATE"]):
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # --- GAME_ID ---
    if "GAME_ID" not in df.columns:
        for cand in ["GAME_ID", "gameId", "GAME_ID_x", "GAME_ID_y"]:
            if cand in df.columns:
                df["GAME_ID"] = df[cand].astype(str)
                break

    # --- Team abbreviations: try direct abbreviation columns first ---
    if "HOME_TEAM_ABBREVIATION" not in df.columns:
        home_candidates = [
            c
            for c in df.columns
            if "HOME" in c.upper()
            and any(key in c.upper() for key in ["TRICODE", "ABBREVIATION", "ABBREV"])
        ]
        if home_candidates:
            df["HOME_TEAM_ABBREVIATION"] = df[home_candidates[0]].astype(str)

    if "VISITOR_TEAM_ABBREVIATION" not in df.columns:
        away_candidates = [
            c
            for c in df.columns
            if any(key in c.upper() for key in ["VISITOR", "AWAY"])
            and any(key in c.upper() for key in ["TRICODE", "ABBREVIATION", "ABBREV"])
        ]
        if away_candidates:
            df["VISITOR_TEAM_ABBREVIATION"] = df[away_candidates[0]].astype(str)

    # --- If still missing, map from team IDs using the team_id -> abbreviation map ---
    if (
        "HOME_TEAM_ABBREVIATION" not in df.columns
        or "VISITOR_TEAM_ABBREVIATION" not in df.columns
    ):
        # Try to identify home/away ID columns
        # Map numeric team IDs to abbreviations via provider helper
        team_id_map = nba_api_provider.get_team_id_map()
        home_id_col = None
        away_id_col = None
        for cand in ["HOME_TEAM_ID", "homeTeamId", "HOME_TEAM_ID_x"]:
            if cand in df.columns:
                home_id_col = cand
                break
        for cand in [
            "VISITOR_TEAM_ID",
            "AWAY_TEAM_ID",
            "awayTeamId",
            "VISITOR_TEAM_ID_x",
        ]:
            if cand in df.columns:
                away_id_col = cand
                break

        if home_id_col is not None and "HOME_TEAM_ABBREVIATION" not in df.columns:
            df["HOME_TEAM_ABBREVIATION"] = df[home_id_col].map(team_id_map)
        if away_id_col is not None and "VISITOR_TEAM_ABBREVIATION" not in df.columns:
            df["VISITOR_TEAM_ABBREVIATION"] = df[away_id_col].map(team_id_map)

        # Drop any rows where we still could not resolve abbreviations
        if (
            "HOME_TEAM_ABBREVIATION" in df.columns
            and "VISITOR_TEAM_ABBREVIATION" in df.columns
        ):
            mask_valid = (
                df["HOME_TEAM_ABBREVIATION"].notna()
                & df["VISITOR_TEAM_ABBREVIATION"].notna()
            )
            df = df.loc[mask_valid].reset_index(drop=True)

    required = {
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ABBREVIATION",
        "VISITOR_TEAM_ABBREVIATION",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Schedule DataFrame for {date_str} is missing required columns after normalization: {sorted(missing)}; "
            f"available columns: {sorted(df.columns)}"
        )

    return df


def _load_team_logs_with_retries(
    team_abbrev: str,
    season_label: str,
    max_retries: int = 3,
    delay_sec: float = 2.0,
) -> pd.DataFrame:
    """
    Robust wrapper around nba_api_provider.get_leaguegamelog_team.

    Retries a few times on network / API errors and returns a DataFrame.
    If all attempts fail, returns an empty DataFrame so callers can
    safely skip games for this team instead of crashing the whole build.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logs = nba_api_provider.get_leaguegamelog_team(
                season=season_label,
                team_abbrev=team_abbrev,
                date_from=None,
                date_to=None,
            )
            # Ensure GAME_DATE is parsed
            logs = _ensure_game_date_as_datetime(logs)
            return logs
        except RequestException as e:
            last_exc = e
            print(
                f"   ⏳ Attempt {attempt} failed for LeagueGameLog(team={team_abbrev}): {e}"
            )
        except Exception as e:
            # Catch any other unexpected endpoint/json issues the same way
            last_exc = e
            print(
                f"   ⏳ Attempt {attempt} failed for LeagueGameLog(team={team_abbrev}) "
                f"due to unexpected error: {e}"
            )

        if attempt < max_retries:
            time.sleep(delay_sec)

    print(
        f"   ⚠️ Giving up on LeagueGameLog for team {team_abbrev} "
        f"after {max_retries} attempts; games for this team will be skipped."
    )
    if last_exc is not None:
        print(f"      Last error was: {last_exc}")

    # Return empty DataFrame so caller can detect and skip safely
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main training dataset builder
# ---------------------------------------------------------------------------


def build_team_training_dataset(
    season_label: str,
    start_date: str,
    end_date: str,
    recent_n: int = 5,
) -> pd.DataFrame:
    """
    Build a team-level matchup training dataset for [start_date, end_date].

    One row per game with:
    - Features: home/away pre-game context + diffs.
    - Targets: home_pts, away_pts, margin, total_points, home_win.

    Parameters
    ----------
    season_label : str
        Season label like '2025-26'.
    start_date : str
        Inclusive start date, 'YYYY-MM-DD'.
    end_date : str
        Inclusive end date, 'YYYY-MM-DD'.
    recent_n : int
        Rolling N for recent context.

    Returns
    -------
    DataFrame with one row per game.
    """
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end_date must be >= start_date")

    print("===============================================")
    print("🏗️  Building team training dataset")
    print("===============================================")
    print(f"📆 Season label: {season_label}")
    print(f"📅 Date range:  {start} → {end}")
    print(f"📈 Recent-N:    {recent_n}")
    print("===============================================")

    # Cache LeagueGameLog per team to avoid repeated API calls
    team_logs_cache: Dict[str, pd.DataFrame] = {}

    rows: List[Dict[str, float | str]] = []

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"\n📅 Fetching schedule for {date_str} ...")
        schedule_df = nba_api_provider.get_schedule_for_date(date_str, season_label)
        if schedule_df.empty:
            print("   ℹ️ No games scheduled on this date.")
            current += dt.timedelta(days=1)
            continue

        # Normalize schedule columns so we always have GAME_ID, GAME_DATE, and
        # HOME_TEAM_ABBREVIATION / VISITOR_TEAM_ABBREVIATION, regardless of the
        # exact column names returned by the upstream endpoint.
        schedule_df = _normalize_schedule_df(schedule_df, date_str)

        # Keep only rows where the game has been played (LeagueGameLog exists for both teams).
        for _, sched_row in schedule_df.iterrows():
            game_id = str(sched_row["GAME_ID"])
            home_team = str(sched_row["HOME_TEAM_ABBREVIATION"])
            away_team = str(sched_row["VISITOR_TEAM_ABBREVIATION"])
            game_date = pd.to_datetime(sched_row["GAME_DATE"]).date()

            # Only use games whose actual date is within [start, end]
            if not (start <= game_date <= end):
                continue

            # Fetch or reuse LeagueGameLog for each team, with retries
            for team in (home_team, away_team):
                if team not in team_logs_cache:
                    print(f"   📥 Loading LeagueGameLog for team {team} ...")
                    logs = _load_team_logs_with_retries(team, season_label)
                    if logs.empty:
                        print(
                            f"   ⚠️ LeagueGameLog is empty for team {team}; "
                            "games involving this team will be skipped."
                        )
                    team_logs_cache[team] = logs

            home_logs = team_logs_cache[home_team]
            away_logs = team_logs_cache[away_team]

            # If we failed to get logs for either team, skip this game
            if home_logs.empty or away_logs.empty:
                print(
                    f"   ⚠️ Skipping game {game_id} ({away_team} @ {home_team}) "
                    "because one or both teams have no LeagueGameLog data."
                )
                continue

            # If this game isn't in both logs yet (i.e., not played / not final), skip it.
            try:
                home_pts, home_pm, home_win_flag, _ = _get_game_result_for_team(
                    home_logs, game_id
                )
                away_pts, away_pm, away_win_flag, _ = _get_game_result_for_team(
                    away_logs, game_id
                )
            except RuntimeError as e:
                print(f"   ⚠️ Skipping game {game_id} ({away_team} @ {home_team}): {e}")
                continue

            # Sanity: home_win_flag & away_win_flag should be complementary unless OT got weird.
            # We'll derive home_win from the score to be safe.
            margin = home_pts - away_pts
            total_points = home_pts + away_pts
            home_win = int(margin > 0)

            # Build pre-game team context using only games BEFORE this game date.
            home_ctx = _compute_team_context(home_logs, game_date, recent_n)
            away_ctx = _compute_team_context(away_logs, game_date, recent_n)

            # If a team had no prior games before this date, you can choose to skip
            # or keep with zero/neutral features. Here we keep them, but you might
            # later filter these out when training if desired.

            row: Dict[str, float | str] = {
                # IDs / metadata
                "game_id": game_id,
                "season": season_label,
                "game_date": game_date.isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                # Targets
                "home_pts": float(home_pts),
                "away_pts": float(away_pts),
                "margin": float(margin),
                "total_points": float(total_points),
                "home_win": float(home_win),
            }

            # Prefix home/away context
            for key, val in home_ctx.items():
                row[f"home_{key}"] = float(val)
            for key, val in away_ctx.items():
                row[f"away_{key}"] = float(val)

            # Diff features (home - away)
            diff_keys = [
                "games_played",
                "wins",
                "losses",
                "win_pct",
                "pts_for_per_game",
                "plus_minus_per_game",
                "recent_wins",
                "recent_losses",
                "recent_plus_minus_per_game",
            ]
            for key in diff_keys:
                row[f"diff_{key}"] = float(home_ctx[key] - away_ctx[key])

            rows.append(row)

        current += dt.timedelta(days=1)

    if not rows:
        raise RuntimeError(
            "No games were added to the training dataset for the given date range."
        )

    df = pd.DataFrame(rows)
    # Sort by date then game_id for readability
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    print(f"\n✅ Built training dataset with {len(df)} games.")
    return df


def main() -> None:
    args = _parse_args()
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_team_training_dataset(
        season_label=args.season_label,
        start_date=args.start_date,
        end_date=args.end_date,
        recent_n=args.recent_n,
    )

    print(f"💾 Writing training CSV to: {output_csv}")
    df.to_csv(output_csv, index=False)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Writing training JSON to: {output_json}")
        df.to_json(output_json, orient="records")

    print("\n🎉 Team training dataset export complete.")


if __name__ == "__main__":
    main()
