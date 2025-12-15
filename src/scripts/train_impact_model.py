# src/scripts/train_impact_model.py
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

from features.impact_dataset import build_lineup_stint_impact_for_team_season
from features.impact_ridge import fit_ridge_impact_model
from db.player_stats_db import load_player_game_stats_for_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a RAPM-style impact model (ridge regression) for one team "
            "based on lineup stints and possession outcomes."
        )
    )
    parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team abbreviation (e.g., 'ATL', 'OKC').",
    )
    parser.add_argument(
        "--season-label",
        type=str,
        default="2025-26",
        help="Season label used in your DB/exports (default: 2025-26).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=10,
        help="Max number of games to use for this team (default: 10, -1 = all).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Ridge regularization strength (default: 50.0).",
    )
    parser.add_argument(
        "--min-stint-possessions",
        type=int,
        default=3,
        help="Minimum possessions per stint to include (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/modeling",
        help="Directory to write impact ratings CSV (default: data/modeling).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    team = args.team.strip().upper()
    season_label = args.season_label
    max_games = None if args.max_games == -1 else args.max_games
    alpha = args.alpha
    min_stint_possessions = args.min_stint_possessions

    print(
        f"📊 Training impact model for team {team}, "
        f"season {season_label}, max_games={max_games}, alpha={alpha}, "
        f"min_stint_possessions={min_stint_possessions}..."
    )

    # 1) Build lineup-stint impact dataset for this team
    stint_df = build_lineup_stint_impact_for_team_season(
        team_abbrev=team,
        max_games=max_games,
    )
    if stint_df.empty:
        print(f"⚠️ No lineup-stint data built for {team}; aborting.")
        return

    print(f"✅ Built stint dataset with {len(stint_df)} rows.")

    # 2) Fit ridge impact model (multi-output: off / def / net)
    impact_df = fit_ridge_impact_model(
        stints_df=stint_df,
        alpha=alpha,
        min_possessions_per_stint=min_stint_possessions,
    )

    print(f"✅ Model fitted; {len(impact_df)} players with impact estimates.")

    # 3) Enrich with names from DB (player_game_stats) where possible
    print("📥 Loading player_game_stats from DB to attach player names...")
    stats_df = load_player_game_stats_for_season(season_label=season_label)

    # --- Normalize player_id from DB into a clean integer column ---
    def _normalize_player_id(value) -> int | None:
        """
        Convert whatever is in player_id (int, str, bytes) into a clean integer
        nba_api personId. If it can't be parsed, return None.
        """
        if isinstance(value, (bytes, bytearray)):
            # Decode any bytes, ignoring bad sequences
            s = value.decode("utf-8", errors="ignore")
        else:
            s = str(value)

        # Keep only digits (in case there is junk around)
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None

    # Apply normalization to the DB stats
    stats_df["player_id_int"] = stats_df["player_id"].map(_normalize_player_id)

    # Drop rows where we couldn't get a valid ID
    stats_df = stats_df.dropna(subset=["player_id_int"]).copy()
    stats_df["player_id_int"] = stats_df["player_id_int"].astype("int64")

    # Build a mapping from player_id_int -> (team, player_name)
    name_map = stats_df[["player_id_int", "team", "player_name"]].drop_duplicates(
        subset=["player_id_int"]
    )

    # --- Align impact_df's IDs to the same int type and merge from DB ---
    impact_df["player_id_int"] = impact_df["player_id"].astype("int64")

    impact_df = impact_df.merge(
        name_map,
        on="player_id_int",
        how="left",
    )

    # At this point, some rows (especially opponents whose teams you haven't
    # processed into the DB) will still have player_name = NaN.

    # --- Fallback: fetch missing names from nba_api for remaining players ---
    missing_mask = impact_df["player_name"].isna()
    missing_ids = (
        impact_df.loc[missing_mask, "player_id"]
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    if missing_ids:
        print(
            f"ℹ️ {len(missing_ids)} players missing names from DB; fetching from nba_api..."
        )

        id_to_name: dict[int, str] = {}

        for pid in missing_ids:
            try:
                resp = commonplayerinfo.CommonPlayerInfo(player_id=int(pid), timeout=10)
                info_df = resp.common_player_info.get_data_frame()
                if not info_df.empty:
                    raw_name = info_df.loc[0, "DISPLAY_FIRST_LAST"]
                    name: str = str(raw_name)
                    id_to_name[int(pid)] = name
                    print(f"   - {pid} → {name}")
            except Exception as e:
                print(f"⚠️ Failed to fetch name for player_id={pid}: {e}")
            # Be a little gentle with the stats API
            time.sleep(0.6)

        # Only fill where player_name is missing
        mask = impact_df["player_name"].isna()

        # Map player_id -> name for those rows
        impact_df.loc[mask, "player_name"] = (
            impact_df.loc[mask, "player_id"].astype("int64").map(id_to_name)
        )

    # Keep original player_id, drop helper int if you want
    impact_df = impact_df.drop(columns=["player_id_int"], errors="ignore")

    # 4) Save to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        output_dir / f"impact_ratings_{team}_{season_label.replace('/', '-')}.csv"
    )
    impact_df.to_csv(out_path, index=False)

    print(f"💾 Saved impact ratings to: {out_path}")

    # 5) Show a small preview sorted by impact_per_100
    print("\n🔎 Top 10 by impact_per_100:")
    preview_cols = [
        "player_id",
        "player_name",
        "impact_off_per_100",
        "impact_def_per_100",
        "impact_per_100",
        "exposure_stint_units",
    ]
    cols = [c for c in preview_cols if c in impact_df.columns]
    print(impact_df[cols].head(10))

    print("\n🔎 Bottom 10 by impact_per_100:")
    print(impact_df[cols].tail(10))


if __name__ == "__main__":
    main()
