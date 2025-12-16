#!/usr/bin/env python
"""
run_daily_pipeline.py

One-command daily pipeline for refreshing all player-level artifacts
for a set of teams:

  1) Update player stats DB for all target teams (via update_db_today.py)
  2) Rebuild full-season profiles from the DB (per team)
  3) Export impact ratings (per team)
  4) Export gameday profiles (local JSON + optional S3 upload)

Usage examples
--------------

# Easiest: auto-detect teams playing today (system date), full-season impact
PYTHONPATH=src python src/scripts/run_daily_pipeline.py --season-label 2025-26

# Explicit teams (e.g., for testing SAS and NYK only)
PYTHONPATH=src python src/scripts/run_daily_pipeline.py \
  --season-label 2025-26 \
  --teams NYK SAS

# Also supported:
# PYTHONPATH=src python src/scripts/run_daily_pipeline.py \
#   --season-label 2025-26 \
#   --teams NYK,SAS
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from datetime import datetime
from typing import List, Sequence

# We rely on nba_api to infer which teams play on a given date if --teams is not provided.
try:
    from nba_api.stats.endpoints import ScoreboardV2
except ImportError:
    ScoreboardV2 = None  # type: ignore[assignment]

# Default S3 settings for gameday profile uploads
S3_BUCKET_DEFAULT = "oddzup-stats-2025"
S3_PREFIX_DEFAULT = "gameday"


def _run_cmd(args: Sequence[str]) -> None:
    """Run a subprocess command with logging and error propagation."""
    cmd_str = " ".join(args)
    print(f"\n💻 Running: {cmd_str}")
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"❌ Command failed with exit code {exc.returncode}: {cmd_str}",
            file=sys.stderr,
        )
        raise


def _get_today_ymd() -> str:
    """Return today's date in YYYY-MM-DD (local system time)."""
    return datetime.today().strftime("%Y-%m-%d")


def _get_teams_playing_on_date(date_str: str) -> List[str]:
    """
    Use nba_api ScoreboardV2 to infer which teams play on the given date.

    Parameters
    ----------
    date_str : str
        Date in 'YYYY-MM-DD'.

    Returns
    -------
    list of team abbreviations (e.g. ['SAS', 'NYK']).
    """
    if ScoreboardV2 is None:
        raise RuntimeError(
            "nba_api is not installed or could not be imported. "
            "Install nba_api or provide --teams explicitly."
        )

    # nba_api expects MM/DD/YYYY
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid --date format '{date_str}'. Expected YYYY-MM-DD."
        ) from exc

    game_date_str = parsed.strftime("%m/%d/%Y")
    print(f"📅 Resolving teams from schedule for {game_date_str} ...")

    sb = ScoreboardV2(game_date=game_date_str)
    line_score = sb.line_score.get_data_frame()

    if "TEAM_ABBREVIATION" not in line_score.columns:
        raise RuntimeError(
            "Unexpected response from ScoreboardV2; TEAM_ABBREVIATION column not found."
        )

    teams = sorted({str(t) for t in line_score["TEAM_ABBREVIATION"].unique()})
    print(f"📝 Teams detected for {date_str}: {teams}")
    return teams


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
        # Allow comma-separated inside a single token.
        for part in token.split(","):
            code = part.strip().upper()
            if code:
                teams.append(code)

    # De-duplicate while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for t in teams:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run daily pipeline for updating DB, profiles, impact, and gameday profiles."
    )

    parser.add_argument(
        "--season-label",
        required=True,
        help="Season label string, e.g. '2025-26'.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Target date in YYYY-MM-DD for schedule / gameday profiles. "
            "Defaults to today's date if not provided."
        ),
    )
    parser.add_argument(
        "--teams",
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of team abbreviations to process "
            "(e.g. SAS NYK or SAS,NYK). "
            "If omitted, teams are inferred from the NBA schedule for --date."
        ),
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=-1,
        help=(
            "Max number of games to consider when building impact ratings "
            "(passed through to export_impact_ratings.py). "
            "Use -1 (default) to use all completed games for the season."
        ),
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="(Reserved) If set, will be wired to export_gameday_profiles.py S3 options later.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    season_label: str = args.season_label
    date_str: str = args.date or _get_today_ymd()
    max_games: int = args.max_games
    upload_s3: bool = args.upload_s3  # currently unused, reserved for future S3 wiring

    print("====================================")
    print("🏀 Daily Player Pipeline")
    print("====================================")
    print(f"📅 Date:          {date_str}")
    print(f"📆 Season label:  {season_label}")
    print(f"🎮 Max games:     {max_games}")
    print(f"☁️ Upload to S3:  {'yes' if upload_s3 else 'no'}")

    # Resolve team list
    if args.teams and len(args.teams) > 0:
        teams = _normalize_team_list(args.teams)
        if not teams:
            raise RuntimeError(
                f"--teams was provided ({args.teams}) but no valid team codes were parsed."
            )
        print(f"📌 Using explicit teams from CLI: {teams}")
    else:
        # Auto-detect from NBA schedule for the given date
        teams = _get_teams_playing_on_date(date_str)
        if not teams:
            raise RuntimeError(
                f"No games found for {date_str}. "
                "Provide --teams explicitly if this is unexpected."
            )

    # 1) Update DB once for all teams via update_db_today.py
    print("\n------------------------------------")
    print(
        "1️⃣ [DB] Updating player stats for all target teams via update_db_today.py ..."
    )

    db_cmd: list[str] = [
        sys.executable,
        "src/scripts/update_db_today.py",
        "--max-games",
        str(max_games),
    ]

    # Pass teams as a single comma-separated string, e.g. "NYK,SAS"
    if teams:
        db_cmd.extend(["--teams", ",".join(teams)])

    _run_cmd(db_cmd)
    print("✅ Daily DB update complete.")
    print(f"   Updated teams: {teams}")

    # 2–3: per-team processing
    for team in teams:
        print("\n------------------------------------")
        print(f"🏷️  Team: {team}")
        print("------------------------------------")

        # 2) Rebuild season profiles from DB
        print(f"2️⃣ [Profiles] Rebuilding season profiles from DB for {team} ...")
        _run_cmd(
            [
                sys.executable,
                "src/scripts/export_profiles_from_db.py",
                "--team",
                team,
                "--season-label",
                season_label,
            ]
        )

        # 3) Export impact ratings
        print(f"3️⃣ [Impact] Exporting impact ratings for {team} ...")
        _run_cmd(
            [
                sys.executable,
                "src/scripts/export_impact_ratings.py",
                "--team",
                team,
                "--season-label",
                season_label,
                "--max-games",
                str(max_games),
            ]
        )

        print(f"✅ Finished steps 2–3 for {team}.")

    # 4) Export gameday profiles (once, for the full set of teams)
    print("\n------------------------------------")
    print("4️⃣ [Gameday] Exporting gameday player profiles ...")

    gameday_cmd: list[str] = [
        sys.executable,
        "src/scripts/export_gameday_profiles.py",
        "--season-label",
        season_label,
        "--teams",
        ",".join(teams),
    ]

    if upload_s3:
        gameday_cmd.extend(
            [
                "--s3-bucket",
                S3_BUCKET_DEFAULT,
                "--s3-prefix",
                S3_PREFIX_DEFAULT,
            ]
        )

    # NOTE: the new export_gameday_profiles.py expects --s3-bucket / --s3-prefix
    # if you want S3 upload. For now we only run it locally from this pipeline.
    # Later we can add --s3-bucket / --s3-prefix to this script and pass them through.

    _run_cmd(gameday_cmd)

    print("\n🎉 Daily pipeline completed successfully!")
    print(f"   Date:   {date_str}")
    print(f"   Season: {season_label}")
    print(f"   Teams:  {teams}")


if __name__ == "__main__":
    main()
