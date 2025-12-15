# src/scripts/update_db_today.py
from __future__ import annotations

import argparse
from typing import List, Optional, Set

from data.nba_api_provider import get_today_games_and_teams
from features.season_player_aggregate import build_player_profile_for_team_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update player_stats.db for teams by fetching and processing only "
            "missing games for each team."
        )
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=-1,
        help=(
            "Optional cap on number of games per team to consider "
            "(useful for dev). Use -1 for all regular-season games before today."
        ),
    )
    parser.add_argument(
        "--teams",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of team abbreviations to update, "
            "e.g. 'ORL' or 'ORL,NYK'. "
            "If omitted or empty, uses the teams playing today."
        ),
    )
    parser.add_argument(
        "--skip-games",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of game IDs to skip entirely for this run, "
            "e.g. '0022500314,0022500123'. "
            "Useful when a specific game has broken nba_api data."
        ),
    )
    return parser.parse_args()


def _parse_team_list(teams_arg: str) -> Optional[List[str]]:
    teams_arg = teams_arg.strip()
    if not teams_arg:
        return None
    parts = [t.strip().upper() for t in teams_arg.split(",") if t.strip()]
    return parts or None


def _parse_game_id_list(skip_arg: str) -> Set[str]:
    skip_arg = skip_arg.strip()
    if not skip_arg:
        return set()
    parts = [g.strip() for g in skip_arg.split(",") if g.strip()]
    return set(parts)


def main() -> None:
    args = parse_args()
    max_games = None if args.max_games == -1 else args.max_games

    explicit_teams = _parse_team_list(args.teams)
    skip_game_ids = _parse_game_id_list(args.skip_games)

    if explicit_teams:
        # ✅ Use the explicit team list, ignore today's schedule
        teams = explicit_teams
        print(f"📌 Using explicit team list from --teams: {teams}")
    else:
        # ✅ Use today's schedule to determine teams
        todays_df, teams = get_today_games_and_teams()
        if todays_df.empty or not teams:
            print("⚠️ No games found for today; nothing to update.")
            return
        print(f"📅 Teams playing today: {teams}")

    if skip_game_ids:
        print(f"⏭️ Will skip these game IDs for this run: {sorted(skip_game_ids)}")

    failed_teams: list[str] = []
    updated_teams: list[str] = []

    for team in teams:
        print(f"\n📊 Updating DB for team {team}...")

        try:
            # This call:
            # - Processes only games before today
            # - Skips games already in player_stats.db (via is_game_processed)
            # - Skips any game whose ID is in skip_game_ids
            # - Inserts per-game stats into the DB
            _ = build_player_profile_for_team_season(
                team_abbrev=team,
                max_games=max_games,
                skip_game_ids=skip_game_ids,
            )
        except RuntimeError as e:
            print(f"❌ Failed to update DB for team {team}: {e}")
            failed_teams.append(team)
            continue

        updated_teams.append(team)

    print("\n✅ Daily DB update complete.")
    if updated_teams:
        print(f"   Updated teams: {updated_teams}")
    else:
        print("   No teams were successfully updated.")

    if failed_teams:
        print("\n⚠️ The following teams failed due to API or data issues:")
        for t in failed_teams:
            print(f"  - {t}")
        print(
            "\nYou can retry just these teams with, for example:\n"
            f"  PYTHONPATH=src python src/scripts/update_db_today.py "
            f"--teams {','.join(failed_teams)} "
            f"--max-games {args.max_games}"
        )


if __name__ == "__main__":
    main()
