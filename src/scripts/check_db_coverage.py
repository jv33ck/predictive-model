#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import pandas as pd

# ✅ Uses your existing nba_api wrapper
from data.nba_api_provider import get_schedule_for_date

# ✅ Reuse the same DB connection pattern your DB scripts use.
# If your engine helper lives somewhere else, just change this import.
from db.session import get_engine, get_session  # type: ignore # adjust if needed


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
        sched = get_schedule_for_date(
            cur.strftime("%Y-%m-%d"), season_label=season_label
        )
        if sched.empty:
            cur += timedelta(days=1)
            continue

        # Normalize expected columns
        if "GAME_ID" not in sched.columns:
            raise RuntimeError(
                f"Schedule for {cur} missing GAME_ID column; got columns={sched.columns}"
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
            sched[["GAME_ID", "game_date"]]
            .drop_duplicates()
            .rename(columns={"GAME_ID": "game_id"})
        )
        cur += timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["game_id", "game_date"])

    all_sched = pd.concat(rows, ignore_index=True)
    return all_sched.drop_duplicates(subset=["game_id"]).sort_values("game_date")


def load_db_games() -> pd.DataFrame:
    """
    Load distinct game_ids that are present in player_game_stats.

    We don't assume a season column here – we just look at all games
    the DB knows about.
    """
    engine = get_engine()
    # This assumes a table named player_game_stats with a game_id column,
    # which is how your other scripts refer to it.
    df = pd.read_sql("SELECT DISTINCT game_id FROM player_game_stats", con=engine)
    if "game_id" not in df.columns:
        raise RuntimeError(
            f"player_game_stats query did not return a game_id column; got {df.columns}"
        )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which scheduled games are missing from the DB."
    )
    parser.add_argument(
        "--season-label",
        type=str,
        required=True,
        help="Season label, e.g. 2025-26",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-10-22",
        help="Start date (YYYY-MM-DD) for coverage check.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        help="End date (YYYY-MM-DD) for coverage check; default = today.",
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
