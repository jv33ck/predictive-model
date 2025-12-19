#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import pandas as pd

# ✅ Uses your existing nba_api wrapper
from data.nba_api_provider import get_schedule_for_date

from db.player_stats_db import get_connection  # add this import


def build_schedule_df(
    season_label: str, start_date: date, end_date: date
) -> pd.DataFrame:
    """
    Pull the official schedule from NBA stats for a date range and return
    a DataFrame with one row per game.
    """
    rows: list[pd.DataFrame] = []
    cur = start_date
    while cur <= end_date:
        print(f"📅 Fetching schedule for {cur} ...")
        sched = get_schedule_for_date(cur.isoformat(), season_label=season_label)
        if sched.empty:
            cur += timedelta(days=1)
            continue

        # Normalize expected columns
        game_id_col: str | None = None
        for cand in ["GAME_ID", "gameId", "game_id"]:
            if cand in sched.columns:
                game_id_col = cand
                break

        if game_id_col is None:
            raise RuntimeError(
                f"Schedule for {cur} missing game id column; got columns={sched.columns}"
            )

        # Some versions name the date column differently; keep it simple
        game_date_col: str | None = None
        for cand in ["GAME_DATE", "GAME_DATE_EST", "game_date"]:
            if cand in sched.columns:
                game_date_col = cand
                break

        if game_date_col is None:
            # fall back to the date we're looping over
            sched = sched.copy()
            sched["game_date"] = cur
            game_date_col = "game_date"

        sched = sched.copy()
        sched["game_date"] = pd.to_datetime(sched[game_date_col]).dt.date

        rows.append(
            sched[[game_id_col, "game_date"]]
            .drop_duplicates()
            .rename(columns={game_id_col: "game_id"})
        )
        cur += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["game_id", "game_date"])

    all_sched = pd.concat(rows, ignore_index=True)
    return all_sched.drop_duplicates(subset=["game_id"]).sort_values("game_date")


def load_db_games() -> pd.DataFrame:
    """
    Load distinct game IDs from the SQLite DB using player_stats_db.

    Reads from player_game_stats in data/player_stats.db and returns a DataFrame
    with a single column game_id (lowercase) to match schedule.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT DISTINCT game_id AS game_id FROM player_game_stats",
            conn,
        )
    finally:
        conn.close()

    if "game_id" not in df.columns:
        raise RuntimeError(
            f"Expected column 'game_id' in DB query result, got columns={df.columns}"
        )

    return df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DB coverage check script."""
    parser = argparse.ArgumentParser(
        description=(
            "Check which scheduled NBA games in a date range are missing from "
            "the local player_game_stats table."
        )
    )

    parser.add_argument(
        "--season-label",
        required=True,
        help=(
            "Season label string used for schedule calls, e.g. '2025-26'. "
            "This should match what your other scripts use."
        ),
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date for coverage check in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end-date",
        required=False,
        default=None,
        help=(
            "Optional end date for coverage check in YYYY-MM-DD format. "
            "If omitted, defaults to today."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = date.today()

    print("====================================")
    print("🔍 Checking DB coverage vs schedule")
    print("====================================")
    print(f"📆 Season:     {args.season_label}")
    print(f"📅 Date range: {start_date} → {end_date}")
    print("====================================")

    sched_df = build_schedule_df(args.season_label, start_date, end_date)
    print(f"✅ Schedule has {len(sched_df)} games in this window.")

    db_games = load_db_games()
    print(f"✅ DB has {len(db_games)} distinct game_ids total.")

    merged = sched_df.merge(
        db_games, how="left", on="game_id", indicator=True, suffixes=("", "_db")
    )

    missing = merged[merged["_merge"] == "left_only"].copy()

    print(f"\n📉 Scheduled games missing from DB: {len(missing)}")
    if not missing.empty:
        print("\n🔎 First 20 missing games:")
        print(
            missing[["game_date", "game_id"]]
            .sort_values("game_date")
            .head(20)
            .to_string(index=False)
        )

        # Also show a quick count by date so you can see if it's just a couple
        # or whole days missing.
        by_date = (
            missing.groupby("game_date")["game_id"].count().reset_index(name="missing")
        )
        print("\n📊 Missing games by date (all):")
        print(by_date.to_string(index=False))
    else:
        print("🎉 No missing scheduled games found in this window.")


if __name__ == "__main__":
    main()
