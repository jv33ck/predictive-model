# src/scripts/export_db_raw.py
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd


def export_player_game_stats(
    db_path: Path,
    output_dir: Path,
    season_label: str = "current",
) -> Path:
    """
    Export the raw per-game, per-player stats from player_game_stats
    table into CSV and JSON.

    This does NOT hit the NBA API; it only reads from the SQLite DB.
    """
    if not db_path.exists():
        raise SystemExit(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)

    # You can customize this query later (filter by season, team, date, etc.)
    query = """
        SELECT
            game_id,
            team,
            player_id,
            player_name,
            minutes_played,
            off_possessions,
            def_possessions,
            total_possessions,
            pts,
            fgm,
            fga,
            fg3m,
            fg3a,
            ftm,
            fta,
            oreb,
            dreb,
            treb,
            ast,
            stl,
            blk,
            tov,
            pf,
            plus_minus,
            ts_pct,
            efg_pct,
            usg_pct,
            est_usg_pct,
            oreb_pct,
            dreb_pct,
            reb_pct,
            ast_pct,
            tov_ratio,
            off_rating_box,
            def_rating_box,
            net_rating_box,
            pace,
            possessions_est,
            off_rating_per_100,
            def_rating_per_100,
            net_rating_per_100
        FROM player_game_stats
    """

    print(f"📥 Reading player_game_stats from {db_path} ...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("⚠️ player_game_stats table is empty; nothing to export.")
        return output_dir

        os.makedirs(output_dir, exist_ok=True)

    base_path = output_dir / f"player_game_stats_{season_label}"
    csv_path = f"{base_path}.csv"
    json_path = f"{base_path}.json"

    print(f"💾 Writing per-game CSV to: {csv_path}")
    df.to_csv(csv_path, index=False)

    # Sanitize string columns to avoid JSON encoding issues (ujson + bad UTF-8)
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        print(f"🔧 Cleaning string columns for JSON export: {list(object_cols)}")
        for col in object_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.encode("utf-8", "replace")  # replace invalid bytes with �
                .str.decode("utf-8")
            )

    print(f"💾 Writing per-game JSON to: {json_path}")
    # force_ascii=False keeps non-ASCII chars (accents, etc.) intact
    df.to_json(json_path, orient="records", force_ascii=False)

    print(
        f"\n✅ Exported {len(df)} rows from player_game_stats to CSV/JSON "
        f"based on DB contents only."
    )
    return base_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw per-game, per-player stats from player_stats.db "
            "to CSV and JSON, without calling the NBA API."
        )
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/player_stats.db",
        help="Path to the SQLite database (default: data/player_stats.db).",
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
        help="Season label to include in the filename (default: 'current').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # .../predictive-model

    db_path = (project_root / args.db_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    export_player_game_stats(
        db_path=db_path,
        output_dir=output_dir,
        season_label=args.season_label,
    )


if __name__ == "__main__":
    main()
