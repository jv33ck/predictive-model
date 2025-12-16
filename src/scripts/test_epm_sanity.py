# src/scripts/test_epm_sanity.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Reuse the DB loader + season aggregation from the profile exporter
from scripts.export_profiles_from_db import (
    load_player_game_stats,
    aggregate_season_profiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check for EPM v0.1: load season profiles from player_stats.db "
            "via the same aggregation used for exported player profiles, and "
            "inspect offensive/defensive/on-court ratings plus EPM."
        )
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/player_stats.db",
        help="Path to SQLite database (default: data/player_stats.db).",
    )
    parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team abbreviation to inspect (e.g. ATL, NYK).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths relative to project root (…/predictive-model)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    db_path = (project_root / args.db_path).resolve()

    team = args.team.upper()

    print(f"📥 Loading player_game_stats from {db_path} ...")
    df = load_player_game_stats(db_path=db_path, teams_filter=[team])

    if df.empty:
        print(f"⚠️ No rows found in player_game_stats for team {team}; aborting.")
        return

    print(f"📊 Aggregating season profiles for team {team} ...")
    season_df = aggregate_season_profiles(df)

    if season_df.empty:
        print("⚠️ aggregate_season_profiles returned an empty DataFrame; aborting.")
        return

    # Filter to the requested team (should already be, but keep it explicit)
    season_df = season_df[season_df["team"] == team].copy()
    if season_df.empty:
        print(f"⚠️ Season profiles contain no rows for team {team}; aborting.")
        return

    # Columns we care about for sanity checking
    cols_to_show = [
        "player_id",
        "player_name",
        "games_played",
        "minutes_played",
        "total_possessions",
        "off_rating_box",
        "def_rating_box",
        "net_rating_box",
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
        "epm_off",
        "epm_def",
        "epm_net",
    ]
    cols_to_show = [c for c in cols_to_show if c in season_df.columns]

    # Sort by EPM net (best to worst and worst to best)
    if "epm_net" not in season_df.columns:
        print(
            "⚠️ epm_net is missing from season_df; check EPM wiring in "
            "aggregate_season_profiles."
        )
        return

    top = season_df.sort_values("epm_net", ascending=False).head(15)
    bottom = season_df.sort_values("epm_net", ascending=True).head(15)

    print("\n🔎 Top 15 by epm_net:")
    print(top[cols_to_show].to_string(index=False))

    print("\n🔎 Bottom 15 by epm_net:")
    print(bottom[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
