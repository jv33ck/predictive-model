# src/scripts/export_player_profiles.py
from __future__ import annotations

import argparse
from typing import List
from pathlib import Path
import os
import pandas as pd

from features.season_player_aggregate import build_player_profile_for_team_season

ALL_TEAMS = [
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
]


def export_player_profiles(
    teams: List[str],
    max_games: int | None = None,
    output_dir: str = "data/exports",
    season_label: str | None = None,
) -> Path | None:
    """
    Build player stat profiles for a list of teams and export them
    as CSV and JSON in a single combined table.

    Args:
        teams:
            List of team abbreviations, e.g. ["ATL", "BOS"].
        max_games:
            Optional limit on number of games per team (useful for dev/testing).
            Set to None to use all available regular-season games.
        output_dir:
            Directory where output files will be written.
        season_label:
            Optional label to include in the filename (e.g. '2025-26').
            If None, a generic label like 'current' is used.

    Returns:
        The base output path (without extension).
    """
    teams = [t.upper() for t in teams]
    season_label = season_label or "current"

    all_profiles: list[pd.DataFrame] = []
    failed_teams: list[str] = []

    for team in teams:
        print(f"\n📊 Building player profiles for team {team}...")

        try:
            profiles = build_player_profile_for_team_season(
                team_abbrev=team,
                max_games=max_games,
            )
        except RuntimeError as e:
            # Any failure inside compute/build for this team means: incomplete data.
            print(f"❌ Skipping team {team} due to data/API issue: {e}")
            failed_teams.append(team)
            continue  # move on to the next team

        if profiles.empty:
            print(f"⚠️ No profiles produced for team {team}, skipping.")
            continue

        all_profiles.append(profiles)

    if not all_profiles:
        print("❌ No team profiles were built; nothing to export.")
        # Still print failed teams info if there are any
        if failed_teams:
            print("\nTeams that failed due to API or data issues:")
            for t in failed_teams:
                print(f"  - {t}")
            print(
                "\nYou can retry just these teams with, for example:\n"
                f"  --teams {','.join(failed_teams)}"
            )
        return

    combined = pd.concat(all_profiles, ignore_index=True)

    # Resolve output_dir relative to the project root (one level above src/)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # .../predictive-model
    output_dir_path = (project_root / output_dir).resolve()

    os.makedirs(output_dir_path, exist_ok=True)

    base_path = output_dir_path / f"player_profiles_{season_label}"
    csv_path = f"{base_path}.csv"
    json_path = f"{base_path}.json"

    print(f"\n💾 Writing CSV to: {csv_path}")
    combined.to_csv(csv_path, index=False)

    print(f"💾 Writing JSON to: {json_path}")
    combined.to_json(json_path, orient="records")

    print(
        f"\n✅ Export complete. Teams: {teams}, "
        f"rows: {len(combined)}, "
        f"columns: {len(combined.columns)}"
    )
    if failed_teams:
        print(
            "\n⚠️ The following teams failed due to API or data issues "
            "and were NOT included in this export:"
        )
        for t in failed_teams:
            print(f"  - {t}")
        print(
            "\nYou can rerun the exporter just for these teams with:\n"
            f"  PYTHONPATH=src python src/scripts/export_player_profiles.py "
            f"--teams {','.join(failed_teams)} "
            f"--max-games {max_games} "
            f"--season-label {season_label}"
        )
    else:
        print("\n✅ All teams were processed successfully.")

    return base_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export player stat profiles for one or more teams."
    )
    parser.add_argument(
        "--teams",
        type=str,
        required=True,
        help=(
            "Comma-separated list of team abbreviations, e.g. 'ATL,BOS,DEN'. "
            "Use 'ALL' later once you wire in a league-wide team list."
        ),
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=-1,
        help=(
            "Maximum number of games per team to process "
            "(useful for dev; set to -1 for all available games)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/exports",
        help="Output directory for CSV/JSON exports (default: data/exports).",
    )
    parser.add_argument(
        "--season-label",
        type=str,
        default="current",
        help="Season label to embed in the filename, e.g. '2025-26'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    teams_arg = args.teams.strip()
    if teams_arg.upper() == "ALL":
        teams = ALL_TEAMS
        print(f"🌐 Using ALL NBA teams: {teams}")
    else:
        teams = [t.strip().upper() for t in teams_arg.split(",") if t.strip()]

    if not teams:
        raise SystemExit("No valid teams provided.")

    max_games = None if args.max_games == -1 else args.max_games

    export_player_profiles(
        teams=teams,
        max_games=max_games,
        output_dir=args.output_dir,
        season_label=args.season_label,
    )


if __name__ == "__main__":
    main()
