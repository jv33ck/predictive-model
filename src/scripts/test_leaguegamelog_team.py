#!/usr/bin/env python
"""
test_leaguegamelog_team.py

Quick sanity script to inspect team-level game logs from the
LeagueGameLog endpoint via our nba_api_provider helper.

Usage examples
--------------

# All team games for the 2025-26 regular season
PYTHONPATH=src python src/scripts/test_leaguegamelog_team.py \
  --season-label 2025-26

# Single team only (e.g. NYK)
PYTHONPATH=src python src/scripts/test_leaguegamelog_team.py \
  --season-label 2025-26 \
  --team NYK

# Team, restricted to a date window
PYTHONPATH=src python src/scripts/test_leaguegamelog_team.py \
  --season-label 2025-26 \
  --team SAS \
  --date-from 2025-11-01 \
  --date-to 2025-11-30
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Sequence

import pandas as pd

from data.nba_api_provider import get_leaguegamelog_team


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test helper around nba_api LeagueGameLog for team-level game logs."
    )
    parser.add_argument(
        "--season-label",
        required=True,
        help="Season label for nba_api (e.g. '2025-26').",
    )
    parser.add_argument(
        "--team",
        default=None,
        help="Optional team abbreviation (e.g. NYK, SAS). If omitted, returns all teams.",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="Optional lower bound date in YYYY-MM-DD or MM/DD/YYYY.",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="Optional upper bound date in YYYY-MM-DD or MM/DD/YYYY.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    season_label: str = args.season_label
    team_abbrev: str | None = args.team
    date_from: str | None = args.date_from
    date_to: str | None = args.date_to

    print("====================================")
    print("🏀 LeagueGameLog Team Test")
    print("====================================")
    print(f"📆 Season:      {season_label}")
    print(f"🏷️ Team:        {team_abbrev or '(ALL TEAMS)'}")
    print(f"📅 Date from:   {date_from or '(none)'}")
    print(f"📅 Date to:     {date_to or '(none)'}")
    print("====================================")

    df = get_leaguegamelog_team(
        season=season_label,
        season_type="Regular Season",
        team_abbrev=team_abbrev,
        date_from=date_from,
        date_to=date_to,
    )

    if df.empty:
        print("⚠️ No rows returned from LeagueGameLog for this query.")
        return

    print(f"✅ Retrieved {len(df)} team-game rows.")
    print("\n📊 Columns:")
    print(sorted(df.columns.tolist()))

    # Show the first few rows nicely
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("\n🔍 Head:")
        print(df.head(10))

    # Basic season summary by team if multiple teams are present
    num_teams = df["TEAM_ABBREVIATION"].nunique()
    if num_teams > 1:
        print("\n📈 Basic summary by team:")
        summary = (
            df.groupby("TEAM_ABBREVIATION")
            .agg(
                games=("GAME_ID", "nunique"),
                pts_for=("PTS", "sum"),
                plus_minus=("PLUS_MINUS", "sum"),
            )
            .sort_values("games", ascending=False)
        )
        print(summary)


if __name__ == "__main__":
    main()
