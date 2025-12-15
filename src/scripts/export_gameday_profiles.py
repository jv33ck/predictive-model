# src/scripts/export_gameday_profiles.py
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from scripts.export_profiles_from_db import (
    load_player_game_stats,
    aggregate_season_profiles,
    _build_team_filter,
    FIELD_DOCS,
)
from utils.s3_upload import upload_to_s3
from data.nba_api_provider import (
    get_today_games_and_teams,
)  # NEW: reuse same schedule logic as update_db_today


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export game-day player profiles from player_stats.db. "
            "Profiles are season-to-date, for either the teams you specify "
            "or (by default) the teams playing today."
        )
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/player_stats.db",
        help="Path to SQLite database (default: data/player_stats.db).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/exports",
        help="Output directory for CSV/JSON (default: data/exports).",
    )
    parser.add_argument(
        "--season-label",
        type=str,
        default="2025-26",
        help="Season label to include in local filenames (e.g. 2025-26).",
    )
    parser.add_argument(
        "--teams",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of team abbreviations to include "
            "(e.g. 'NYK,OKC,ORL,SAS'). If omitted or empty, this script "
            "will automatically use the teams playing today via "
            "data.nba_api_provider.get_today_games_and_teams()."
        ),
    )
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help=(
            "Optional ISO date label (YYYY-MM-DD) for local filenames and payload "
            "(e.g. game date). If omitted, today's date is used."
        ),
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="",
        help=(
            "Optional S3 bucket name. If provided, the script will also upload "
            "CSV/JSON game-day outputs to this bucket under fixed keys:\n"
            "  gameday/gameday_player_profiles.csv\n"
            "  gameday/gameday_player_profiles.json"
        ),
    )
    return parser.parse_args()


def export_gameday_profiles(
    db_path: Path,
    output_dir: Path,
    season_label: str,
    teams_filter: Optional[List[str]],
    date_label: str,
    s3_bucket: Optional[str] = None,
) -> Path:
    """
    Export season-to-date player profiles for a subset of teams (e.g. today's matchups).

    - Reads per-game rows from player_game_stats for the requested teams.
    - Aggregates into one row per (team, player).
    - Writes CSV + JSON locally (with season/date in the filename).
    - Optionally uploads both to S3 under fixed 'gameday/...' keys, overwriting on each run.
    """
    if not teams_filter:
        raise SystemExit(
            "No teams provided to export_gameday_profiles; this should be "
            "caught earlier in main()."
        )

    print(f"📥 Loading game-day data for teams: {teams_filter}")
    df = load_player_game_stats(db_path, teams_filter=teams_filter)

    if df.empty:
        print(
            "⚠️ No data in player_game_stats for the requested teams; "
            "did you update the DB first?"
        )
        return output_dir

    profiles = aggregate_season_profiles(df)
    if profiles.empty:
        print("⚠️ Aggregation produced no profiles; nothing to export.")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Local filenames keep season + date for your own history
    base_path = output_dir / f"gameday_player_profiles_{season_label}_{date_label}"
    csv_path = f"{base_path}.csv"
    json_path = f"{base_path}.json"

    # --- CSV export (local) ---
    print(f"💾 Writing game-day profiles CSV to: {csv_path}")
    profiles.to_csv(csv_path, index=False)

    # --- JSON export (local, grouped by team) ---
    # Ensure object columns are JSON-safe strings
    object_cols = profiles.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        print(f"🔧 Cleaning string columns for JSON export: {list(object_cols)}")

        def _safe_to_str(value) -> str:
            if isinstance(value, (bytes, bytearray)):
                try:
                    return value.decode("utf-8", errors="replace")
                except Exception:
                    return value.decode("latin-1", errors="replace")
            return str(value)

        for col in object_cols:
            profiles[col] = profiles[col].map(_safe_to_str)

    # Build a team-centric payload: { season, date, meta, teams: [ {team, players[]} ] }
    teams_payload = []
    for team in sorted(profiles["team"].unique()):
        team_df = profiles[profiles["team"] == team].reset_index(drop=True)
        team_players = team_df.to_dict(orient="records")
        teams_payload.append(
            {
                "team": team,
                "players": team_players,
            }
        )

    payload = {
        "season": season_label,
        "date": date_label,
        "meta": {
            "schema_version": "1.0",
            "description": (
                "OddzUp game-day season-to-date player profiles derived from "
                "player_game_stats for the specified teams."
            ),
            "fields": FIELD_DOCS,
        },
        "teams": teams_payload,
    }

    print(f"💾 Writing game-day profiles JSON to: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # --- Optional S3 upload WITH FIXED KEYS ---
    if s3_bucket:
        # These keys DO NOT include date or season; they are overwritten each run.
        csv_key = "gameday/gameday_player_profiles.csv"
        json_key = "gameday/gameday_player_profiles.json"

        print(f"☁️ Uploading game-day CSV to s3://{s3_bucket}/{csv_key} ...")
        upload_to_s3(Path(csv_path), s3_bucket, csv_key, public=False)

        print(f"☁️ Uploading game-day JSON to s3://{s3_bucket}/{json_key} ...")
        upload_to_s3(Path(json_path), s3_bucket, json_key, public=False)

    print(
        f"\n✅ Exported {len(profiles)} game-day player profiles "
        f"for teams {sorted(profiles['team'].unique())} on {date_label}."
    )
    return base_path


def main() -> None:
    args = parse_args()

    # Resolve paths relative to project root (same pattern as export_profiles_from_db.py)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    db_path = (project_root / args.db_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    # Decide which teams to include:
    teams_arg = args.teams.strip()
    if teams_arg:
        # Explicit list from CLI
        teams_filter = _build_team_filter(teams_arg)
        if not teams_filter:
            raise SystemExit("No valid team abbreviations provided via --teams.")
        print(f"📌 Using explicit team list from --teams: {teams_filter}")
    else:
        # Auto-detect teams playing today using the SAME provider as update_db_today.py
        todays_df, teams = get_today_games_and_teams()
        if todays_df.empty or not teams:
            print("⚠️ No games found for today; nothing to export.")
            return
        teams_filter = sorted(teams)
        print(f"📅 Teams playing today (for game-day export): {teams_filter}")

    # Use provided date label or default to today's date in ISO format (for LOCAL metadata/filenames only)
    date_label = args.date or date.today().isoformat()

    export_gameday_profiles(
        db_path=db_path,
        output_dir=output_dir,
        season_label=args.season_label,
        teams_filter=teams_filter,
        date_label=date_label,
        s3_bucket=args.s3_bucket or None,
    )


if __name__ == "__main__":
    main()
