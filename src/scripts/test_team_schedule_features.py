# src/scripts/test_team_schedule_features.py

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from features.team_schedule_features import build_team_schedule_features_df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check team schedule features built from LeagueGameLog."
    )
    parser.add_argument(
        "--season-label",
        required=True,
        help="Season label, e.g. '2025-26'.",
    )
    parser.add_argument(
        "--team",
        required=True,
        help="Team abbreviation, e.g. NYK or SAS.",
    )
    parser.add_argument(
        "--date-cutoff",
        default=None,
        help=(
            "Optional upper date bound (YYYY-MM-DD or MM/DD/YYYY). "
            "If omitted, uses all games returned by LeagueGameLog."
        ),
    )
    parser.add_argument(
        "--recent-n",
        type=int,
        default=5,
        help="Window size for 'recent form' features (default 5).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    season_label: str = args.season_label
    team: str = args.team
    date_cutoff: str | None = args.date_cutoff
    recent_n: int = args.recent_n

    print("====================================")
    print("🏀 Team Schedule Features Test")
    print("====================================")
    print(f"📆 Season:        {season_label}")
    print(f"🏷️ Team:          {team}")
    print(f"📅 Date cutoff:   {date_cutoff or '(none)'}")
    print(f"📈 Recent window: last {recent_n} games")
    print("====================================")

    features_df = build_team_schedule_features_df(
        season_label=season_label,
        team=team,
        date_cutoff=date_cutoff,
        recent_n=recent_n,
    )

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("\n📊 Feature row (columns):")
        print(features_df.columns.tolist())
        print("\n🔍 Values (transposed):")
        print(features_df.T)


if __name__ == "__main__":
    main()
