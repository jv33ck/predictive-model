# src/scripts/export_profiles_from_db.py
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd
from utils.s3_upload import upload_to_s3
import json


# Human/machine-readable schema for player season profile fields
FIELD_DOCS = {
    "team": {
        "label": "Team",
        "type": "string",
        "source": "nba_api",
        "description": "Team abbreviation (e.g. 'OKC', 'NYK').",
    },
    "player_id": {
        "label": "Player ID",
        "type": "string",
        "source": "nba_api",
        "description": "NBA player ID (stringified).",
    },
    "player_name": {
        "label": "Player Name",
        "type": "string",
        "source": "nba_api",
        "description": "Full player name.",
    },
    "games_played": {
        "label": "Games Played",
        "type": "int",
        "source": "aggregate",
        "description": "Number of games with non-zero minutes in the season for this team.",
    },
    "minutes_played": {
        "label": "Minutes Played",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total minutes played across all games for this team.",
    },
    "minutes_per_game": {
        "label": "Minutes per Game",
        "type": "float",
        "source": "derived",
        "description": "Average minutes played per game (minutes_played / games_played).",
    },
    "total_off_possessions": {
        "label": "Offensive Possessions (On-Court)",
        "type": "float",
        "source": "pbp_lineup",
        "description": "Estimated number of team offensive possessions while the player was on the floor.",
    },
    "total_def_possessions": {
        "label": "Defensive Possessions (On-Court)",
        "type": "float",
        "source": "pbp_lineup",
        "description": "Estimated number of team defensive possessions while the player was on the floor.",
    },
    "total_possessions": {
        "label": "Total Possessions (On-Court)",
        "type": "float",
        "source": "pbp_lineup",
        "description": "total_off_possessions + total_def_possessions while the player was on the floor.",
    },
    "possessions_per_game": {
        "label": "Possessions per Game",
        "type": "float",
        "source": "derived",
        "description": "Average total on-court possessions per game (total_possessions / games_played).",
    },
    "pts": {
        "label": "Points",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total points scored.",
    },
    "pts_per_game": {
        "label": "Points per Game",
        "type": "float",
        "source": "derived",
        "description": "Average points per game (pts / games_played).",
    },
    "pts_per_36": {
        "label": "Points per 36 Minutes",
        "type": "float",
        "source": "derived",
        "description": "Scaled scoring rate per 36 minutes (pts / minutes_played * 36).",
    },
    "pts_per_100_poss": {
        "label": "Points per 100 Offensive Possessions",
        "type": "float",
        "source": "derived",
        "description": "Points scored per 100 team offensive possessions while on the floor (pts / total_off_possessions * 100).",
    },
    "fgm": {
        "label": "Field Goals Made",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total field goals made.",
    },
    "fga": {
        "label": "Field Goals Attempted",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total field goals attempted.",
    },
    "fg_pct": {
        "label": "Field Goal Percentage",
        "type": "float",
        "source": "derived",
        "description": "Field goal percentage (fgm / fga).",
    },
    "fg3m": {
        "label": "3-Pointers Made",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total three-point field goals made.",
    },
    "fg3a": {
        "label": "3-Pointers Attempted",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total three-point field goals attempted.",
    },
    "three_pct": {
        "label": "3-Point Percentage",
        "type": "float",
        "source": "derived",
        "description": "Three-point percentage (fg3m / fg3a).",
    },
    "ftm": {
        "label": "Free Throws Made",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total free throws made.",
    },
    "fta": {
        "label": "Free Throws Attempted",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total free throws attempted.",
    },
    "ft_pct": {
        "label": "Free Throw Percentage",
        "type": "float",
        "source": "derived",
        "description": "Free throw percentage (ftm / fta).",
    },
    "oreb": {
        "label": "Offensive Rebounds",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total offensive rebounds.",
    },
    "dreb": {
        "label": "Defensive Rebounds",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total defensive rebounds.",
    },
    "treb": {
        "label": "Total Rebounds",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total rebounds (offensive + defensive).",
    },
    "ast": {
        "label": "Assists",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total assists.",
    },
    "stl": {
        "label": "Steals",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total steals.",
    },
    "blk": {
        "label": "Blocks",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total blocks.",
    },
    "tov": {
        "label": "Turnovers",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total turnovers.",
    },
    "pf": {
        "label": "Personal Fouls",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total personal fouls.",
    },
    "plus_minus": {
        "label": "Plus/Minus (Total)",
        "type": "float",
        "source": "boxscore_traditional",
        "description": "Total on-court point differential across all games.",
    },
    "plus_minus_per_game": {
        "label": "Plus/Minus per Game",
        "type": "float",
        "source": "derived",
        "description": "Average plus/minus per game (plus_minus / games_played).",
    },
    "plus_minus_per_100_poss": {
        "label": "Plus/Minus per 100 Possessions",
        "type": "float",
        "source": "derived",
        "description": "Plus/minus per 100 total on-court possessions (plus_minus / total_possessions * 100).",
    },
    "ts_pct": {
        "label": "True Shooting Percentage",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level true shooting percentage from NBA advanced boxscore.",
    },
    "efg_pct": {
        "label": "Effective Field Goal Percentage",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level effective field goal percentage from NBA advanced boxscore.",
    },
    "usg_pct": {
        "label": "Usage Percentage",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level usage percentage from NBA advanced boxscore.",
    },
    "est_usg_pct": {
        "label": "Estimated Usage Percentage",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level estimated usage percentage from NBA advanced boxscore.",
    },
    "oreb_pct": {
        "label": "Offensive Rebound Percentage",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level offensive rebound percentage from NBA advanced boxscore.",
    },
    "dreb_pct": {
        "label": "Defensive Rebound Percentage",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level defensive rebound percentage from NBA advanced boxscore.",
    },
    "reb_pct": {
        "label": "Total Rebound Percentage",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level total rebound percentage from NBA advanced boxscore.",
    },
    "ast_pct": {
        "label": "Assist Percentage",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level assist percentage from NBA advanced boxscore.",
    },
    "tov_ratio": {
        "label": "Turnover Ratio",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level turnover ratio from NBA advanced boxscore.",
    },
    "off_rating_box": {
        "label": "Offensive Rating (Boxscore)",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of NBA advanced boxscore offensive rating (points per 100 possessions, NBA definition).",
    },
    "def_rating_box": {
        "label": "Defensive Rating (Boxscore)",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of NBA advanced boxscore defensive rating (points allowed per 100 possessions, NBA definition).",
    },
    "net_rating_box": {
        "label": "Net Rating (Boxscore)",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "off_rating_box - def_rating_box.",
    },
    "pace": {
        "label": "Pace",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Average of game-level pace from NBA advanced boxscore (possessions per 48 minutes).",
    },
    "possessions_est": {
        "label": "Estimated Possessions",
        "type": "float",
        "source": "boxscore_advanced",
        "description": "Sum of NBA advanced boxscore 'possessions' estimate for this player.",
    },
    "off_rating_per_100": {
        "label": "Offensive Rating per 100 (OddzUp PBP)",
        "type": "float",
        "source": "pbp_lineup",
        "description": "Points scored by the player's team per 100 offensive possessions while he is on the floor, from OddzUp PBP pipeline.",
    },
    "def_rating_per_100": {
        "label": "Defensive Rating per 100 (OddzUp PBP)",
        "type": "float",
        "source": "pbp_lineup",
        "description": "Points allowed by the player's team per 100 defensive possessions while he is on the floor, from OddzUp PBP pipeline.",
    },
    "net_rating_per_100": {
        "label": "Net Rating per 100 (OddzUp PBP)",
        "type": "float",
        "source": "pbp_lineup",
        "description": "off_rating_per_100 - def_rating_per_100, based on OddzUp PBP possessions.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export season-level player profiles from player_stats.db, "
            "aggregating per-game rows into one row per player."
        )
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/player_stats.db",
        help="Path to SQLite database (default: data/player_stats.db).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/exports",
        help="Output directory for CSV/JSON (default: data/exports).",
    )
    parser.add_argument(
        "--season-label",
        type=str,
        default="2025-26",
        help="Season label to include in filenames (e.g. 2025-26).",
    )
    parser.add_argument(
        "--teams",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of team abbreviations to include "
            "(e.g. 'ATL,BOS'). If empty, export all teams in the DB."
        ),
    )
    return parser.parse_args()


def _build_team_filter(teams_arg: str) -> Optional[List[str]]:
    if not teams_arg:
        return None
    parts = [t.strip().upper() for t in teams_arg.split(",") if t.strip()]
    return parts or None


def _normalize_player_id(val) -> str:
    """
    Normalize player_id from DB into a clean string.

    - If it's an 8-byte blob (or repr of one), treat as little-endian int:
        b'7\\xdb\\x18\\x00\\x00\\x00\\x00\\x00' -> "1628983"
    - If it's numeric, cast to int then str.
    - Otherwise, just str() it.
    """
    import math
    import ast

    if val is None:
        return ""

    # If it's already a string, it might be:
    #   * a clean "1628983"
    #   * a repr: "b'7\\xdb\\x18\\x00\\x00\\x00\\x00\\x00'"
    if isinstance(val, str):
        s = val.strip()

        # Looks like a bytes literal from repr()
        if (s.startswith("b'") and s.endswith("'")) or (
            s.startswith('b"') and s.endswith('"')
        ):
            try:
                b = ast.literal_eval(s)  # -> bytes
                if isinstance(b, (bytes, bytearray)) and len(b) > 0:
                    n = int.from_bytes(b, "little")
                    return str(n)
            except Exception:
                # If anything goes wrong, just fall back to raw string
                return s

        # Already just digits -> keep as-is
        if s.isdigit():
            return s

        # Fallback: leave the string
        return s

    # Real bytes/bytearray from SQLite
    if isinstance(val, (bytes, bytearray)):
        if len(val) == 0:
            return ""
        n = int.from_bytes(val, "little")
        return str(n)

    # Numeric types
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return ""
        return str(int(val))

    # Anything else, just string-ify
    return str(val)


def load_player_game_stats(
    db_path: Path,
    teams_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not db_path.exists():
        raise SystemExit(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)

    base_query = """
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

    if teams_filter:
        placeholders = ",".join("?" for _ in teams_filter)
        query = base_query + f" WHERE team IN ({placeholders})"
        print(f"📥 Loading player_game_stats for teams: {teams_filter}")
        df = pd.read_sql_query(query, conn, params=list(teams_filter))
    else:
        print("📥 Loading all rows from player_game_stats ...")
        df = pd.read_sql_query(base_query, conn)

    conn.close()

    # Normalize player_id immediately so everything downstream sees a clean string
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].apply(_normalize_player_id)

    return df


def aggregate_season_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-game rows from player_game_stats into season-level
    profiles (one row per team/player).
    """

    if df.empty:
        print("⚠️ player_game_stats query returned no rows; nothing to aggregate.")
        return df

    # Ensure numeric types where expected
    numeric_cols = [
        "minutes_played",
        "off_possessions",
        "def_possessions",
        "total_possessions",
        "pts",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "oreb",
        "dreb",
        "treb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "plus_minus",
        "ts_pct",
        "efg_pct",
        "usg_pct",
        "est_usg_pct",
        "oreb_pct",
        "dreb_pct",
        "reb_pct",
        "ast_pct",
        "tov_ratio",
        "off_rating_box",
        "def_rating_box",
        "net_rating_box",
        "pace",
        "possessions_est",
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["team", "player_id", "player_name"]

    # Aggregate across games
    grouped = (
        df.groupby(group_cols)
        .agg(
            games_played=("game_id", "nunique"),
            minutes_played=("minutes_played", "sum"),
            total_off_possessions=("off_possessions", "sum"),
            total_def_possessions=("def_possessions", "sum"),
            total_possessions=("total_possessions", "sum"),
            pts=("pts", "sum"),
            fgm=("fgm", "sum"),
            fga=("fga", "sum"),
            fg3m=("fg3m", "sum"),
            fg3a=("fg3a", "sum"),
            ftm=("ftm", "sum"),
            fta=("fta", "sum"),
            oreb=("oreb", "sum"),
            dreb=("dreb", "sum"),
            treb=("treb", "sum"),
            ast=("ast", "sum"),
            stl=("stl", "sum"),
            blk=("blk", "sum"),
            tov=("tov", "sum"),
            pf=("pf", "sum"),
            plus_minus=("plus_minus", "sum"),
            # Advanced stats: per-game averages across season
            ts_pct=("ts_pct", "mean"),
            efg_pct=("efg_pct", "mean"),
            usg_pct=("usg_pct", "mean"),
            est_usg_pct=("est_usg_pct", "mean"),
            oreb_pct=("oreb_pct", "mean"),
            dreb_pct=("dreb_pct", "mean"),
            reb_pct=("reb_pct", "mean"),
            ast_pct=("ast_pct", "mean"),
            tov_ratio=("tov_ratio", "mean"),
            off_rating_box=("off_rating_box", "mean"),
            def_rating_box=("def_rating_box", "mean"),
            net_rating_box=("net_rating_box", "mean"),
            pace=("pace", "mean"),
            possessions_est=("possessions_est", "sum"),
            off_rating_per_100=("off_rating_per_100", "mean"),
            def_rating_per_100=("def_rating_per_100", "mean"),
            net_rating_per_100=("net_rating_per_100", "mean"),
        )
        .reset_index()
    )

    # Derived season metrics
    grouped["minutes_per_game"] = grouped["minutes_played"] / grouped[
        "games_played"
    ].clip(lower=1)

    grouped["possessions_per_game"] = grouped["total_possessions"] / grouped[
        "games_played"
    ].clip(lower=1)

    grouped["pts_per_game"] = grouped["pts"] / grouped["games_played"].clip(lower=1)

    grouped["pts_per_36"] = grouped.apply(
        lambda row: (
            (row["pts"] / row["minutes_played"] * 36.0)
            if row["minutes_played"] > 0
            else 0.0
        ),
        axis=1,
    )

    grouped["pts_per_100_poss"] = grouped.apply(
        lambda row: (
            (row["pts"] / row["total_off_possessions"] * 100.0)
            if row["total_off_possessions"] > 0
            else 0.0
        ),
        axis=1,
    )

    grouped["fg_pct"] = grouped.apply(
        lambda row: row["fgm"] / row["fga"] if row["fga"] > 0 else 0.0,
        axis=1,
    )
    grouped["three_pct"] = grouped.apply(
        lambda row: row["fg3m"] / row["fg3a"] if row["fg3a"] > 0 else 0.0,
        axis=1,
    )
    grouped["ft_pct"] = grouped.apply(
        lambda row: row["ftm"] / row["fta"] if row["fta"] > 0 else 0.0,
        axis=1,
    )

    # Plus/minus per game
    grouped["plus_minus_per_game"] = grouped["plus_minus"] / grouped[
        "games_played"
    ].clip(lower=1)

    # Plus/minus per 100 possessions
    grouped["plus_minus_per_100_poss"] = grouped.apply(
        lambda row: (
            (row["plus_minus"] / row["total_possessions"] * 100.0)
            if row["total_possessions"] > 0
            else 0.0
        ),
        axis=1,
    )

    # Round for presentation (but keep numeric)
    one_decimal_cols = [
        "minutes_played",
        "minutes_per_game",
        "total_possessions",
        "possessions_per_game",
        "pts_per_game",
        "pts_per_36",
        "pts_per_100_poss",
        "plus_minus_per_game",
        "plus_minus_per_100_poss",
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
        "off_rating_box",
        "def_rating_box",
        "net_rating_box",
        "pace",
    ]

    three_decimal_cols = [
        "fg_pct",
        "three_pct",
        "ft_pct",
        "ts_pct",
        "efg_pct",
        "usg_pct",
        "est_usg_pct",
        "oreb_pct",
        "dreb_pct",
        "reb_pct",
        "ast_pct",
        "tov_ratio",
    ]

    for col in one_decimal_cols:
        if col in grouped.columns:
            grouped[col] = grouped[col].round(1)

    for col in three_decimal_cols:
        if col in grouped.columns:
            grouped[col] = grouped[col].round(3)

    # Order columns similar to your existing profile export
    profile_cols = [
        "team",
        "player_id",
        "player_name",
        "games_played",
        "minutes_played",
        "minutes_per_game",
        "total_off_possessions",
        "total_def_possessions",
        "total_possessions",
        "possessions_per_game",
        "pts",
        "pts_per_game",
        "pts_per_36",
        "pts_per_100_poss",
        "fgm",
        "fga",
        "fg_pct",
        "fg3m",
        "fg3a",
        "three_pct",
        "ftm",
        "fta",
        "ft_pct",
        "oreb",
        "dreb",
        "treb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "plus_minus",
        "plus_minus_per_game",
        "plus_minus_per_100_poss",
        "ts_pct",
        "efg_pct",
        "usg_pct",
        "est_usg_pct",
        "oreb_pct",
        "dreb_pct",
        "reb_pct",
        "ast_pct",
        "tov_ratio",
        "off_rating_box",
        "def_rating_box",
        "net_rating_box",
        "pace",
        "possessions_est",
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
    ]

    # Keep only columns that exist
    profile_cols = [c for c in profile_cols if c in grouped.columns]
    return grouped[profile_cols].copy()


def export_profiles_from_db(
    db_path: Path,
    output_dir: Path,
    season_label: str,
    teams_filter: Optional[List[str]] = None,
) -> Path:
    df = load_player_game_stats(db_path, teams_filter=teams_filter)

    if df.empty:
        print(
            "⚠️ No data in player_game_stats for the requested filter; nothing to export."
        )
        return output_dir

    profiles = aggregate_season_profiles(df)
    if profiles.empty:
        print("⚠️ Aggregation produced no profiles; nothing to export.")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    base_path = output_dir / f"player_profiles_from_db_{season_label}"
    csv_path = f"{base_path}.csv"
    json_path = f"{base_path}.json"

    print(f"💾 Writing season profiles CSV to: {csv_path}")
    profiles.to_csv(csv_path, index=False)

    # Clean string columns for JSON export (robust to non-UTF8 bytes)
    object_cols = profiles.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        print(f"🔧 Cleaning string columns for JSON export: {list(object_cols)}")

        def _safe_to_str(value) -> str:
            if isinstance(value, (bytes, bytearray)):
                try:
                    return value.decode("utf-8", errors="replace")
                except Exception:
                    return value.decode("latin-1", errors="replace")
            return str(value)

        for col in object_cols:
            profiles[col] = profiles[col].map(_safe_to_str)

    print(f"💾 Writing season profiles JSON to: {json_path}")

    # Represent each player profile as a dict
    profile_records = profiles.to_dict(orient="records")

    payload = {
        "season": season_label,
        "meta": {
            "schema_version": "1.0",
            "description": "OddzUp season-level player profiles aggregated from player_game_stats.",
            "fields": FIELD_DOCS,
        },
        "profiles": profile_records,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"\n✅ Exported {len(profiles)} player season profiles from DB "
        f"to CSV/JSON for season {season_label}."
    )
    return base_path


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # .../predictive-model

    db_path = (project_root / args.db_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    teams_filter = _build_team_filter(args.teams)

    export_profiles_from_db(
        db_path=db_path,
        output_dir=output_dir,
        season_label=args.season_label,
        teams_filter=teams_filter or None,
    )


if __name__ == "__main__":
    main()
