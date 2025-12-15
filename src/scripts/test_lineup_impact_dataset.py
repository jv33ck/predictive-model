# src/scripts/test_lineup_impact_dataset.py
from __future__ import annotations

import argparse

from features.impact_dataset import build_lineup_stint_impact_for_team_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test building a lineup-stint impact dataset for one team using "
            "possessions + lineups."
        )
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
    max_games = (
        None if args.max_games == -1 else args.max_games
    )  # <- fix this typo when you paste: max_games

    stint_df = build_lineup_stint_impact_for_team_season(
        team_abbrev=team,
        max_games=max_games,
    )

    if stint_df.empty:
        print("⚠️ Lineup-stint impact dataset is empty.")
        return

    print("\n📐 Lineup-stint impact columns:")
    print(sorted(stint_df.columns))

    print("\n🔎 First 10 lineup stints:")
    print(stint_df.head(10))

    print("\n📊 Summary:")
    print("Rows (lineup stints):", len(stint_df))
    if "game_id" in stint_df.columns:
        print("Unique games:", stint_df["game_id"].nunique())
    if "home_team" in stint_df.columns:
        print("Home teams in sample:", stint_df["home_team"].unique())
    if "away_team" in stint_df.columns:
        print("Away teams in sample:", stint_df["away_team"].unique())

    if "net_rating_home_per_100" in stint_df.columns:
        print("\n🔍 Net rating (home) summary:")
        print(stint_df["net_rating_home_per_100"].describe())


if __name__ == "__main__":
    main()
