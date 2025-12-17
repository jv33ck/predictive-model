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
from db.player_reference_db import (
    load_player_reference_map,
    upsert_player_reference,
)


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
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/player_stats.db",
        help="Path to SQLite DB (player_stats.db). Default: data/player_stats.db",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    team = args.team.strip().upper()
    season_label = args.season_label
    max_games = None if args.max_games == -1 else args.max_games
    alpha = args.alpha
    min_stint_possessions = args.min_stint_possessions
    db_path = Path(args.db_path)

    print(
        f"📊 Training impact model for team {team}, "
        f"season {season_label}, max_games={max_games}, alpha={alpha}, "
        f"min_stint_possessions={min_stint_possessions}..."
    )

    # 1) Build lineup-stint impact dataset for this team
    stint_df = build_lineup_stint_impact_for_team_season(
        team_abbrev=team,
        season_label=season_label,
        max_games=max_games,
        db_path=str(db_path),
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

    # 3) Attach player names as cheaply as possible:
    #    (a) from player_game_stats in DB
    #    (b) from player_reference cache
    #    (c) finally from nba_api for any truly-missing IDs, and cache them.

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

    # Build a mapping from player_id_int -> player_name from the DB
    name_map_df = stats_df[["player_id_int", "player_name"]].dropna(
        subset=["player_name"]
    )
    db_name_map: dict[int, str] = (
        name_map_df.drop_duplicates(subset=["player_id_int"])
        .set_index("player_id_int")["player_name"]
        .astype(str)
        .to_dict()
    )

    # Upsert these DB-derived names into the persistent player_reference table
    if db_name_map:
        upsert_player_reference(
            db_name_map,
            db_path=db_path,
            season_label=season_label,
            team_abbrev=team,
            source="db_player_game_stats",
        )

    # --- Align impact_df's IDs to the same int type and merge names from DB ---
    impact_df["player_id_int"] = impact_df["player_id"].astype("int64")

    # Merge from the stats_df-based name_map (first pass)
    impact_df = impact_df.merge(
        name_map_df.drop_duplicates(subset=["player_id_int"]),
        on="player_id_int",
        how="left",
        suffixes=("", "_db"),
    )

    # If there was already a player_name column in impact_df, prefer existing non-null
    if "player_name_db" in impact_df.columns:
        impact_df["player_name"] = impact_df["player_name"].combine_first(
            impact_df["player_name_db"]
        )
        impact_df = impact_df.drop(columns=["player_name_db"])

    # --- Second pass: consult player_reference cache for any remaining IDs ---
    ref_map = load_player_reference_map(db_path=db_path)
    if ref_map:
        impact_df["player_name"] = impact_df["player_name"].combine_first(
            impact_df["player_id_int"].map(ref_map)
        )

    # --- Third pass: fetch remaining missing names from nba_api, cache them ---
    missing_mask = impact_df["player_name"].isna()
    missing_ids = (
        impact_df.loc[missing_mask, "player_id_int"]
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )

    id_to_name_from_api: dict[int, str] = {}

    if missing_ids:
        print(
            f"ℹ️ {len(missing_ids)} players still missing names after DB+cache; "
            f"fetching from nba_api..."
        )

        for pid in missing_ids:
            try:
                resp = commonplayerinfo.CommonPlayerInfo(player_id=int(pid), timeout=10)
                info_df = resp.common_player_info.get_data_frame()
                if not info_df.empty:
                    raw_name = info_df.loc[0, "DISPLAY_FIRST_LAST"]
                    name: str = str(raw_name)
                    id_to_name_from_api[int(pid)] = name
                    print(f"   - {pid} → {name}")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Failed to fetch name for player_id={pid}: {e}")
            # Be a little gentle with the stats API
            time.sleep(0.6)

        # Persist these newly-fetched names into player_reference for future runs
        if id_to_name_from_api:
            upsert_player_reference(
                id_to_name_from_api,
                db_path=db_path,
                season_label=season_label,
                team_abbrev=team,
                source="nba_api_commonplayerinfo",
            )

        # Fill only where player_name is still missing
        mask = impact_df["player_name"].isna()
        impact_df.loc[mask, "player_name"] = impact_df.loc[mask, "player_id_int"].map(
            id_to_name_from_api
        )

    # Keep original player_id, drop helper int column
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
