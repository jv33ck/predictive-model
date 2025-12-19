#!/usr/bin/env python
"""
Daily player+team modeling pipeline.

Steps:
  1) Update DB for target teams (boxscores + lineup stints)
  2) Export season-long player profiles from DB
  3) Export impact ratings for each team
  4) Export gameday player profiles (and upload to S3 if requested)
  5) Export gameday matchup features
  6) Generate team-level matchup predictions
  7) Generate gameday player-level stat predictions
  8) Combine team + player predictions into a single matchup file
     and upload to S3 (if requested)
"""

import argparse
import datetime as dt
import subprocess
import sys
from typing import List

import pandas as pd  # type: ignore[import-untyped]
from nba_api.stats.endpoints import ScheduleLeagueV2  # type: ignore[import-untyped]


DEFAULT_S3_BUCKET = "oddzup-stats-2025"
DEFAULT_S3_PREFIX = "gameday"


# Local helper: fetch NBA schedule for a date via nba_api.


def get_schedule_for_date(
    date_str: str, season_label: str | None = None
) -> pd.DataFrame:
    """Fetch NBA schedule for a single calendar date via nba_api.

    This implementation avoids passing date-specific keyword arguments to
    ScheduleLeagueV2 (which vary across nba_api versions). Instead, it
    optionally filters by season on the API side and then filters by date
    locally in pandas.
    """
    # Conservative kwargs: only pass season if provided so we work across
    # nba_api versions that may not support date_from_nullable/game_date.
    kwargs: dict[str, object] = {}
    if season_label is not None:
        kwargs["season"] = season_label

    endpoint = ScheduleLeagueV2(**kwargs)  # type: ignore[call-arg]
    df = endpoint.get_data_frames()[0]

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if df.empty:
        return df

    # Normalize and filter by the game date using common schedule columns.
    target_date = pd.to_datetime(date_str).date()
    candidate_date_cols = [
        "GAME_DATE",
        "GAME_DATE_EST",
        "GAME_DATE_LCL",
        "GAME_DATE_NBA",
        "gameDate",
    ]

    for col in candidate_date_cols:
        if col in df.columns:
            tmp = df.copy()
            tmp[col] = pd.to_datetime(tmp[col]).dt.date
            filtered = tmp[tmp[col] == target_date].copy()
            if not filtered.empty:
                return filtered

    # If we cannot find/normalize a date column, fall back to the raw
    # DataFrame; infer_teams_for_date will raise a helpful error if no teams
    # can be extracted.
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full daily OddzUp modeling pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Gameday date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--season-label",
        type=str,
        default=None,
        help=(
            "Season label used in exports and logs, e.g. '2025-26'. "
            "If omitted, it will be inferred from --date (or today)."
        ),
    )
    parser.add_argument(
        "--teams",
        type=str,
        default="",
        help="Comma-separated list of team abbreviations (e.g. 'CHI,CLE,MEM,MIN').",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=-1,
        help="Max number of games per team to process when updating DB / impact.",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="If set, upload gameday profiles and combined predictions to S3.",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=DEFAULT_S3_BUCKET,
        help="S3 bucket to upload to when --upload-s3 is set.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=DEFAULT_S3_PREFIX,
        help="S3 key prefix (e.g. 'gameday') when --upload-s3 is set.",
    )
    return parser.parse_args()


def infer_season_label(gameday_date: dt.date) -> str:
    """Infer NBA season label (e.g. '2025-26') from a game date.

    Assumes seasons start in October and span two calendar years.
    """
    year = gameday_date.year
    if gameday_date.month >= 10:
        start_year = year
    else:
        start_year = year - 1
    end_year_short = (start_year + 1) % 100
    return f"{start_year}-{end_year_short:02d}"


def infer_teams_for_date(
    gameday_date: dt.date, season_label: str | None = None
) -> List[str]:
    """Infer the set of team abbreviations playing on a given date.

    Uses the shared NBA schedule helper so behavior matches other
    scripts (e.g. check_db_coverage, matchup exports).
    """
    date_str = gameday_date.isoformat()
    try:
        raw = get_schedule_for_date(date_str, season_label=season_label)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Failed to infer today's teams from schedule; "
            "please pass --teams explicitly."
        ) from exc

    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        df = pd.DataFrame(raw)

    if df.empty:
        raise RuntimeError(
            f"No games found in schedule for {date_str}; "
            "please pass --teams explicitly."
        )

    # Try a variety of likely column names for team tricodes/abbreviations.
    candidate_home_cols = [
        "HOME_TEAM_ABBREVIATION",
        "home_team_abbreviation",
        "homeTeam_teamTricode",
        "home_team_tricode",
        "homeTeam_tricode",
    ]
    candidate_away_cols = [
        "VISITOR_TEAM_ABBREVIATION",
        "away_team_abbreviation",
        "awayTeam_teamTricode",
        "away_team_tricode",
        "awayTeam_tricode",
    ]

    home_col = next((c for c in candidate_home_cols if c in df.columns), None)
    away_col = next((c for c in candidate_away_cols if c in df.columns), None)

    if home_col is None or away_col is None:
        raise RuntimeError(
            "Could not locate home/away team abbreviation columns in schedule "
            f"for {date_str}. Available columns: {list(df.columns)}"
        )

    teams_set = set(
        df[home_col].dropna().astype(str).str.upper().tolist()
        + df[away_col].dropna().astype(str).str.upper().tolist()
    )

    teams = sorted(t for t in teams_set if t)
    if not teams:
        raise RuntimeError(
            f"Schedule for {date_str} did not contain any team abbreviations; "
            "please pass --teams explicitly."
        )

    return teams


def run_cmd(cmd: List[str]) -> None:
    print(f"\n💻 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ Command failed with code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()

    # Resolve date
    if args.date is not None:
        gameday_date = dt.date.fromisoformat(args.date)
    else:
        gameday_date = dt.date.today()

    gameday_date_str = gameday_date.isoformat()

    # Resolve season label (allow auto-infer from date)
    season_label = args.season_label or infer_season_label(gameday_date)

    # Resolve teams (explicit or inferred from schedule)
    if args.teams.strip():
        teams = [t.strip().upper() for t in args.teams.split(",") if t.strip()]
        print(f"📌 Using explicit teams from CLI: {teams}")
    else:
        print("📌 No --teams supplied; inferring teams from NBA schedule ...")
        teams = infer_teams_for_date(gameday_date, season_label=season_label)
        print(f"📌 Using inferred gameday teams: {teams}")

    teams_csv = ",".join(teams)

    # S3 info
    upload_s3 = args.upload_s3
    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix

    print("====================================")
    print("🏀 Daily Player Pipeline")
    print("====================================")
    print(f"📅 Date:          {gameday_date_str}")
    print(f"📆 Season label:  {season_label}")
    print(f"🎮 Max games:     {args.max_games}")
    print(f"☁️ Upload to S3:  {'yes' if upload_s3 else 'no'}")
    print(f"📌 Teams:         {teams}")
    print("====================================")

    # 1️⃣ DB update ------------------------------------------------------------
    print("\n------------------------------------")
    print(
        "1️⃣ [DB] Updating player stats for all target teams via update_db_today.py ..."
    )

    cmd = [
        sys.executable,
        "src/scripts/update_db_today.py",
        "--max-games",
        str(args.max_games),
        "--teams",
        teams_csv,
    ]
    run_cmd(cmd)
    print("✅ Daily DB update complete.")
    print(f"   Updated teams: {teams}")

    # 2️⃣ Season profiles from DB ---------------------------------------------
    print("\n------------------------------------")
    print("2️⃣ [Profiles] Rebuilding season profiles from DB for all target teams ...")

    cmd = [
        sys.executable,
        "src/scripts/export_profiles_from_db.py",
        "--season-label",
        season_label,
        "--team",
        teams_csv,
    ]
    run_cmd(cmd)

    # 3️⃣ Impact ratings -------------------------------------------------------
    print("\n------------------------------------")
    for team in teams:
        print(f"🏷️  Team: {team}")
        print("3️⃣ [Impact] Exporting impact ratings ...")

        cmd = [
            sys.executable,
            "src/scripts/export_impact_ratings.py",
            "--team",
            team,
            "--season-label",
            season_label,
            "--max-games",
            str(args.max_games),
        ]
        run_cmd(cmd)

    # 4️⃣ Gameday player profiles ---------------------------------------------
    print("\n------------------------------------")
    print("4️⃣ [Gameday] Exporting gameday player profiles ...")

    cmd = [
        sys.executable,
        "src/scripts/export_gameday_profiles.py",
        "--season-label",
        season_label,
        "--teams",
        teams_csv,
    ]
    if upload_s3:
        cmd.extend(
            [
                "--s3-bucket",
                s3_bucket,
                "--s3-prefix",
                s3_prefix,
            ]
        )
    run_cmd(cmd)

    # 5️⃣ Gameday matchup features --------------------------------------------
    print("\n------------------------------------")
    print("5️⃣ [Matchups] Exporting gameday matchup features ...")

    cmd = [
        sys.executable,
        "src/scripts/export_gameday_matchups.py",
        "--season-label",
        season_label,
        "--date",
        gameday_date_str,
        "--teams",
        teams_csv,
    ]
    run_cmd(cmd)

    # 6️⃣ Team-level matchup predictions --------------------------------------
    print("\n------------------------------------")
    print("6️⃣ [Team Predictions] Generating team-level matchup predictions ...")

    cmd = [
        sys.executable,
        "src/scripts/predict_team_matchups.py",
        "--season-label",
        season_label,
        "--date",
        gameday_date_str,
    ]
    run_cmd(cmd)

    # 7️⃣ Player-level stat predictions ---------------------------------------
    print("\n------------------------------------")
    print("7️⃣ [Player Predictions] Generating gameday player stat predictions ...")

    # We rely on defaults in predict_gameday_players.py for:
    #   --train-json, --models-dir, --gameday-profiles
    cmd = [
        sys.executable,
        "src/scripts/predict_gameday_players.py",
        "--date",
        gameday_date_str,
    ]
    run_cmd(cmd)

    # 8️⃣ Combined matchup + player package -----------------------------------
    print("\n------------------------------------")
    print(
        "8️⃣ [Package] Combining team+player predictions into matchup file "
        "and uploading to S3 (if enabled) ..."
    )

    cmd = [
        sys.executable,
        "src/scripts/export_gameday_predictions.py",
        "--date",
        gameday_date_str,
    ]
    if upload_s3:
        cmd.extend(
            [
                "--s3-bucket",
                s3_bucket,
                "--s3-prefix",
                s3_prefix,
            ]
        )
    run_cmd(cmd)

    # Done --------------------------------------------------------------------
    print("\n🎉 Daily pipeline completed successfully!")
    print(f"   Date:   {gameday_date_str}")
    print(f"   Season: {season_label}")
    print(f"   Teams:  {teams}")


if __name__ == "__main__":
    main()
