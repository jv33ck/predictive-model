# src/scripts/test_matchup_features.py
"""
test_matchup_features.py

Quick sanity script to build and inspect matchup-level features for a single game
using the gameday player profiles JSON.

Example usage
-------------

# Assuming you've already run run_daily_pipeline.py with NYK and SAS,
# and have data/exports/gameday_player_profiles.json:
PYTHONPATH=src python src/scripts/test_matchup_features.py \
  --home-team NYK \
  --away-team SAS \
  --season-label 2025-26

You can also override the gameday profiles path if needed:

PYTHONPATH=src python src/scripts/test_matchup_features.py \
  --home-team NYK \
  --away-team SAS \
  --season-label 2025-26 \
  --gameday-json data/exports/gameday_player_profiles.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from features.matchup_features import build_matchup_features_from_profiles


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test script for building matchup-level features from gameday player profiles."
    )

    parser.add_argument(
        "--home-team",
        required=True,
        help="Home team abbreviation (e.g. NYK).",
    )
    parser.add_argument(
        "--away-team",
        required=True,
        help="Away team abbreviation (e.g. SAS).",
    )
    parser.add_argument(
        "--season-label",
        required=False,
        default=None,
        help="Optional season label (e.g. 2025-26) to tag on the matchup row.",
    )
    parser.add_argument(
        "--gameday-json",
        default="data/exports/gameday_player_profiles.json",
        help="Path to gameday_player_profiles.json (default: data/exports/gameday_player_profiles.json).",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    home_team: str = args.home_team.upper()
    away_team: str = args.away_team.upper()
    season_label: str = args.season_label
    json_path = Path(args.gameday_json)

    if not json_path.exists():
        raise FileNotFoundError(f"Could not find gameday profiles JSON at: {json_path}")

    print(f"📥 Loading gameday profiles from: {json_path}")
    profiles_df = pd.read_json(json_path, orient="records")

    print(
        f"📊 Building matchup features for {away_team} @ {home_team}"
        f"{' (' + season_label + ')' if season_label else ''} ..."
    )

    matchup_df = build_matchup_features_from_profiles(
        profiles_df,
        home_team=home_team,
        away_team=away_team,
        season_label=season_label,
        game_id=None,
    )

    print("\n✅ Matchup feature row (columns):")
    print(matchup_df.columns.tolist())

    print("\n🔍 Matchup feature values (transposed):")
    # Show as a (column, value) listing
    print(matchup_df.T)


if __name__ == "__main__":
    main()
