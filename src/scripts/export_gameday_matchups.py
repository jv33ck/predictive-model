#!/usr/bin/env python
# src/scripts/export_gameday_matchups.py
"""
Export gameday matchup feature rows for today's games (or a specified date),
using the already-built gameday player profiles plus team-level features
from LeagueGameLog.

Intended to be called from run_daily_pipeline.py after:

  1) DB updated
  2) Season profiles rebuilt
  3) Impact ratings exported
  4) Gameday player profiles exported

This script then:

  * Loads data/exports/gameday_player_profiles.json
  * Uses ScoreboardV2 to find today's games (or games for --date)
  * Optionally restricts to a subset of teams (--teams)
  * Builds one matchup feature row per game via build_matchup_features_from_profiles
  * Writes:
        data/exports/gameday_matchups_<YYYY-MM-DD>.csv
        data/exports/gameday_matchups_<YYYY-MM-DD>.json
  * Optionally uploads these files to S3 if --s3-bucket is provided
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


from features.matchup_features import build_matchup_features_from_profiles
from utils.s3_upload import upload_to_s3


import pandas as pd
from data.nba_api_provider import get_schedule_league_v2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid --date '{date_str}'. Expected format YYYY-MM-DD."
        ) from exc


def _normalize_team_list(raw_teams: Sequence[str] | None) -> List[str]:
    """
    Normalize a raw --teams argument into a clean, deduplicated list.

    Supports:
      --teams NYK SAS
      --teams NYK,SAS
      --teams NYK SAS,LAL
    """
    teams: List[str] = []
    if not raw_teams:
        return teams

    for token in raw_teams:
        for part in token.split(","):
            code = part.strip().upper()
            if code:
                teams.append(code)

    seen: set[str] = set()
    deduped: List[str] = []
    for t in teams:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _get_scheduled_games_for_date(
    season_label: str,
    date_str: str,
    teams_filter: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Use ScheduleLeagueV2 to get the list of games for a given date (and optional team filter).

    Returns a dataframe with columns that mirror the old ScoreboardV2 GameHeader:
    ['GAME_ID', 'GAME_DATE', 'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION']
    so downstream code can remain unchanged.
    """
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid date_str '{date_str}' (expected YYYY-MM-DD)."
        ) from exc

    print(
        f"📅 Fetching schedule for {date_str} via ScheduleLeagueV2 (season={season_label})..."
    )
    schedule_df = get_schedule_league_v2(season_label)
    if schedule_df.empty:
        print("⚠️ ScheduleLeagueV2 returned no rows; cannot build matchups.")
        return pd.DataFrame()

    required_cols = {
        "gameDate",
        "gameId",
        "homeTeam_teamTricode",
        "awayTeam_teamTricode",
    }
    missing = required_cols.difference(schedule_df.columns)
    if missing:
        raise RuntimeError(
            "Unexpected response from ScheduleLeagueV2; missing columns: "
            + ", ".join(sorted(missing))
        )

    df = schedule_df.copy()
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce").dt.date
    day_games = df[df["gameDate"] == target_date]

    if teams_filter:
        team_set = {t.upper() for t in teams_filter}
        # Build the mask directly on day_games to avoid index misalignment
        mask = day_games["homeTeam_teamTricode"].astype(str).str.upper().isin(
            team_set
        ) | day_games["awayTeam_teamTricode"].astype(str).str.upper().isin(team_set)
        day_games = day_games[mask]

    if day_games.empty:
        print(
            f"ℹ️ No scheduled games found in ScheduleLeagueV2 for {date_str} "
            f"with team filter={list(teams_filter) if teams_filter else None}."
        )
        return pd.DataFrame()

    # Map to the same column names the rest of the script expects from the old ScoreboardV2 path
    mapped = day_games.rename(
        columns={
            "gameId": "GAME_ID",
            "gameDate": "GAME_DATE",
            "homeTeam_teamTricode": "HOME_TEAM_ABBREVIATION",
            "awayTeam_teamTricode": "VISITOR_TEAM_ABBREVIATION",
        }
    )[
        ["GAME_ID", "GAME_DATE", "HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION"]
    ].copy()

    print(f"✅ Found {len(mapped)} scheduled games for {date_str}.")
    return mapped


@dataclass
class MatchupConfig:
    season_label: str
    date_str: str
    teams_filter: Optional[List[str]]
    profiles_path: Path
    output_dir: Path
    s3_bucket: Optional[str]
    s3_prefix: str


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _filter_games_to_teams(
    game_header: pd.DataFrame, teams_filter: Optional[Iterable[str]]
) -> pd.DataFrame:
    if game_header.empty:
        return game_header

    if not teams_filter:
        return game_header

    teams = {t.upper() for t in teams_filter}
    mask = game_header["HOME_TEAM_ABBREVIATION"].isin(teams) | game_header[
        "VISITOR_TEAM_ABBREVIATION"
    ].isin(teams)

    filtered = game_header.loc[mask].copy()
    print(
        f"🎯 Restricting games to teams {sorted(teams)}: "
        f"{len(filtered)} games found."
    )
    return filtered


def export_gameday_matchups(cfg: MatchupConfig) -> Optional[Path]:
    """
    Build and export gameday matchup feature rows.

    Returns the CSV path if files were written, or None if there were
    no qualifying games for the date/teams.
    """
    # 1) Load gameday player profiles
    if not cfg.profiles_path.exists():
        raise FileNotFoundError(
            f"Gameday profiles JSON not found at {cfg.profiles_path}. "
            "Make sure export_gameday_profiles.py has been run first."
        )

    print(f"📥 Loading gameday player profiles from: {cfg.profiles_path}")
    profiles_df = pd.read_json(cfg.profiles_path)
    print(f"   Loaded {len(profiles_df)} player rows from profiles JSON.")

    # 2) Fetch scoreboard and restrict to teams of interest
    game_header = _get_scheduled_games_for_date(
        season_label=cfg.season_label,
        date_str=cfg.date_str,
        teams_filter=cfg.teams_filter,
    )

    if game_header.empty:
        print(
            f"ℹ️ No scheduled games available for {cfg.date_str} (season={cfg.season_label}); "
            "no matchup features will be exported."
        )
        return
    # Ensure the expected columns are present now
    for col in ["GAME_ID", "HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION"]:
        if col not in game_header.columns:
            print(
                f"⚠️ Scoreboard game_header missing required column '{col}'. "
                f"Columns are: {list(game_header.columns)}. "
                "No matchup features will be exported."
            )
            return None

    game_header = _filter_games_to_teams(game_header, cfg.teams_filter)

    if game_header.empty:
        team_msg = (
            f"teams_filter={cfg.teams_filter}"
            if cfg.teams_filter is not None
            else "no team filter"
        )
        print(
            f"ℹ️ No games found for date={cfg.date_str} after applying {team_msg}; "
            "no matchup features will be exported."
        )
        return None

    # 3) Build matchup feature rows
    matchup_rows: List[pd.DataFrame] = []

    for _, row in game_header.iterrows():
        game_id = str(row["GAME_ID"])
        home_team = str(row["HOME_TEAM_ABBREVIATION"])
        away_team = str(row["VISITOR_TEAM_ABBREVIATION"])

        print(
            f"📊 Building matchup features for {away_team} @ {home_team} "
            f"(GameID {game_id}) ..."
        )

        # build_matchup_features_from_profiles already computes:
        #   - aggregated lineup features from profiles_df
        #   - team LeagueGameLog features (via nba_api)
        matchup_df = build_matchup_features_from_profiles(
            profiles_df=profiles_df,
            home_team=home_team,
            away_team=away_team,
            season_label=cfg.season_label,
        )

        # Ensure game_id is present and consistent
        if "game_id" not in matchup_df.columns:
            matchup_df = matchup_df.copy()
            matchup_df["game_id"] = f"{cfg.season_label}_{away_team}_at_{home_team}"

        matchup_rows.append(matchup_df)

    if not matchup_rows:
        print(
            "⚠️ No matchup rows were constructed despite non-empty scoreboard; "
            "no files will be written."
        )
        return None

    full_matchups = pd.concat(matchup_rows, ignore_index=True)

    # 4) Write to disk
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"gameday_matchups_{cfg.date_str}"

    csv_path = cfg.output_dir / f"{base_name}.csv"
    json_path = cfg.output_dir / f"{base_name}.json"

    full_matchups.to_csv(csv_path, index=False)
    full_matchups.to_json(json_path, orient="records")

    print(f"💾 Wrote matchup CSV to:  {csv_path}")
    print(f"💾 Wrote matchup JSON to: {json_path}")

    # 5) Optional S3 upload
    if cfg.s3_bucket:
        csv_key = f"{cfg.s3_prefix}/{base_name}.csv"
        json_key = f"{cfg.s3_prefix}/{base_name}.json"

        print(f"☁️ Uploading matchups CSV to s3://{cfg.s3_bucket}/{csv_key} ...")
        upload_to_s3(csv_path, cfg.s3_bucket, csv_key)

        print(f"☁️ Uploading matchups JSON to s3://{cfg.s3_bucket}/{json_key} ...")
        upload_to_s3(json_path, cfg.s3_bucket, json_key)

    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export gameday matchup features from gameday player profiles "
            "plus team LeagueGameLog data."
        )
    )

    parser.add_argument(
        "--season-label",
        required=True,
        help="Season label string, e.g. '2025-26'.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help=("Target date in YYYY-MM-DD format. Defaults to today's system date."),
    )
    parser.add_argument(
        "--teams",
        nargs="*",
        default=None,
        help=(
            "Optional explicit team list to filter games (e.g. NYK SAS or NYK,SAS). "
            "If omitted, all games on the date are considered."
        ),
    )
    parser.add_argument(
        "--profiles-json",
        default="data/exports/gameday_player_profiles.json",
        help="Path to the gameday profiles JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exports",
        help="Directory to write matchup CSV/JSON files.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Optional S3 bucket name for uploading matchup files.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="gameday_matchups",
        help="S3 key prefix for matchup files (if --s3-bucket is set).",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    season_label: str = args.season_label
    date_str: str = args.date or datetime.today().strftime("%Y-%m-%d")
    teams_filter = _normalize_team_list(args.teams)
    profiles_path = Path(args.profiles_json)
    output_dir = Path(args.output_dir)
    s3_bucket: Optional[str] = args.s3_bucket
    s3_prefix: str = args.s3_prefix

    print("====================================")
    print("🏀 Gameday Matchup Export")
    print("====================================")
    print(f"📆 Season:      {season_label}")
    print(f"📅 Date:        {date_str}")
    print(f"📁 Profiles:    {profiles_path}")
    print(f"📂 Output dir:  {output_dir}")
    print(f"☁️ S3 bucket:   {s3_bucket or '(none)'}")
    print(f"☁️ S3 prefix:   {s3_prefix}")
    print("====================================")

    cfg = MatchupConfig(
        season_label=season_label,
        date_str=date_str,
        teams_filter=teams_filter if teams_filter else None,
        profiles_path=profiles_path,
        output_dir=output_dir,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
    )

    result = export_gameday_matchups(cfg)

    if result is None:
        print("\nℹ️ No gameday matchups were exported (no qualifying games).")
    else:
        print("\n🎉 Gameday matchup export completed successfully.")


if __name__ == "__main__":
    main()
