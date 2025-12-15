# src/scripts/test_impact_dataset.py
from __future__ import annotations

import argparse

from features.impact_dataset import build_possession_impact_for_team_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test building a possession-level impact dataset for one team."
    )
    parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team abbreviation, e.g. 'ATL' or 'OKC'.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=3,
        help="Max number of games to process (default: 3, use -1 for all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    team = args.team.strip().upper()
    max_games = None if args.max_games == -1 else args.max_games

    impact_df = build_possession_impact_for_team_season(
        team_abbrev=team,
        max_games=max_games,
    )

    if impact_df.empty:
        print("⚠️ Impact dataset is empty.")
        return

    print("\n📐 Impact dataset columns:")
    print(sorted(impact_df.columns))

    print("\n🔎 First 10 possession rows:")
    print(impact_df.head(10))

    print("\n📊 Basic counts:")
    print("Rows (possessions):", len(impact_df))
    if "game_id" in impact_df.columns:
        print("Unique games:", impact_df["game_id"].nunique())
    if "lineup_stint_index" in impact_df.columns:
        print(
            "Unique lineup stints:",
            impact_df["lineup_stint_index"].nunique(),
        )


if __name__ == "__main__":
    main()
