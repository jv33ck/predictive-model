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

import pandas as pd

from features.matchup_features import build_matchup_features_from_profiles
from utils.s3_upload import upload_to_s3

try:
    from nba_api.stats.endpoints import ScoreboardV2
except ImportError:
    ScoreboardV2 = None  # type: ignore[assignment]


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


def _get_scoreboard_games(date_str: str) -> pd.DataFrame:
    """
    Fetch the scoreboard for a given date and ensure we have
    HOME_TEAM_ABBREVIATION and VISITOR_TEAM_ABBREVIATION columns.

    We use ScoreboardV2 and, if necessary, derive abbreviations
    from the line_score TEAM_ID/TEAM_ABBREVIATION mapping.

    If there are no games or we cannot map abbreviations, we return
    an EMPTY DataFrame instead of raising, so the caller can treat
    it as "no matchups today".
    """
    if ScoreboardV2 is None:
        raise RuntimeError(
            "nba_api is not installed or could not be imported. "
            "Install nba_api to use scoreboard-based matchup export."
        )

    dt = _parse_date(date_str)
    game_date_str = dt.strftime("%m/%d/%Y")
    print(f"📅 Fetching scoreboard for {game_date_str} ...")

    sb = ScoreboardV2(game_date=game_date_str)

    game_header = sb.game_header.get_data_frame()
    line_score = sb.line_score.get_data_frame()

    # No games at all for this date
    if game_header.empty:
        print(
            "ℹ️ Scoreboard returned no games for this date; "
            "no gameday matchups will be exported."
        )
        return game_header

    # If the modern columns already exist, just use them.
    expected_cols = ["HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION"]
    if all(col in game_header.columns for col in expected_cols):
        print("ℹ️ Scoreboard game_header already has home/away abbreviations.")
        return game_header

    # Otherwise derive abbreviations from TEAM_ID -> TEAM_ABBREVIATION mapping.
    required_game_header_cols = ["HOME_TEAM_ID", "VISITOR_TEAM_ID", "GAME_ID"]
    if not all(col in game_header.columns for col in required_game_header_cols):
        print(
            "⚠️ ScoreboardV2.game_header is missing expected team ID columns; "
            f"got columns: {list(game_header.columns)}. "
            "Treating as no games for matchup export."
        )
        return pd.DataFrame()

    if line_score.empty:
        print(
            "⚠️ ScoreboardV2.line_score returned no rows; "
            "cannot derive team abbreviations. Treating as no games."
        )
        return pd.DataFrame()

    if (
        "TEAM_ID" not in line_score.columns
        or "TEAM_ABBREVIATION" not in line_score.columns
    ):
        print(
            "⚠️ ScoreboardV2.line_score is missing TEAM_ID or TEAM_ABBREVIATION; "
            f"got columns: {list(line_score.columns)}. Treating as no games."
        )
        return pd.DataFrame()

    print(
        "ℹ️ Deriving HOME_TEAM_ABBREVIATION / VISITOR_TEAM_ABBREVIATION from line_score..."
    )

    # Build a mapping TEAM_ID -> TEAM_ABBREVIATION (global mapping is fine).
    abbr_map = (
        line_score[["TEAM_ID", "TEAM_ABBREVIATION"]]
        .drop_duplicates()
        .set_index("TEAM_ID")["TEAM_ABBREVIATION"]
        .to_dict()
    )

    if not abbr_map:
        print(
            "⚠️ TEAM_ID -> TEAM_ABBREVIATION map is empty; "
            "cannot derive abbreviations. Treating as no games."
        )
        return pd.DataFrame()

    # Map abbreviations onto game_header
    game_header = game_header.copy()
    game_header["HOME_TEAM_ABBREVIATION"] = game_header["HOME_TEAM_ID"].map(abbr_map)
    game_header["VISITOR_TEAM_ABBREVIATION"] = game_header["VISITOR_TEAM_ID"].map(
        abbr_map
    )

    # Filter out any rows where mapping failed (NaNs)
    valid_mask = (
        game_header["HOME_TEAM_ABBREVIATION"].notna()
        & game_header["VISITOR_TEAM_ABBREVIATION"].notna()
    )
    mapped = game_header.loc[valid_mask].copy()

    if mapped.empty:
        print(
            "⚠️ Failed to map HOME_TEAM_ID/VISITOR_TEAM_ID to abbreviations using line_score; "
            "no games had valid abbreviations after mapping. Treating as no games."
        )
        return pd.DataFrame()

    # Optional debug if some rows were dropped
    dropped = len(game_header) - len(mapped)
    if dropped > 0:
        print(
            f"⚠️ Dropped {dropped} game_header rows that lacked valid team abbreviations "
            "after mapping from line_score."
        )

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
    game_header = _get_scoreboard_games(cfg.date_str)

    # If scoreboard gave us nothing usable, bail out gracefully.
    if game_header.empty:
        print(
            f"ℹ️ No scoreboard games available for {cfg.date_str}; "
            "no matchup features will be exported."
        )
        return None

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
