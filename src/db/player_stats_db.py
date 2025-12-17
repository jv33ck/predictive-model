# src/db/player_stats_db.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Tuple, Any

import pandas as pd

import json

import numpy as np

# DB path: project_root/data/player_stats.db
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "player_stats.db"


def get_connection() -> sqlite3.Connection:
    """
    Open a connection to the SQLite database.
    Creates the data directory if needed.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db() -> None:
    """
    Create tables if they do not exist.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Games table: which games we've processed
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            season TEXT NOT NULL
        )
        """
    )

    # Per-game, per-player stats (simplified for now – we can expand later)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS player_game_stats (
            game_id TEXT NOT NULL,
            team TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,

            minutes_played REAL NOT NULL,
            off_possessions REAL NOT NULL,
            def_possessions REAL NOT NULL,
            total_possessions REAL NOT NULL,

            pts REAL NOT NULL,
            fgm REAL NOT NULL,
            fga REAL NOT NULL,
            fg3m REAL NOT NULL,
            fg3a REAL NOT NULL,
            ftm REAL NOT NULL,
            fta REAL NOT NULL,
            oreb REAL NOT NULL,
            dreb REAL NOT NULL,
            treb REAL NOT NULL,
            ast REAL NOT NULL,
            stl REAL NOT NULL,
            blk REAL NOT NULL,
            tov REAL NOT NULL,
            pf REAL NOT NULL,
            plus_minus REAL NOT NULL,

            ts_pct REAL,
            efg_pct REAL,
            usg_pct REAL,
            est_usg_pct REAL,
            oreb_pct REAL,
            dreb_pct REAL,
            reb_pct REAL,
            ast_pct REAL,
            tov_ratio REAL,
            off_rating_box REAL,
            def_rating_box REAL,
            net_rating_box REAL,
            pace REAL,
            possessions_est REAL,

            off_rating_per_100 REAL,
            def_rating_per_100 REAL,
            net_rating_per_100 REAL,

            PRIMARY KEY (game_id, team, player_id)
        )
        """
    )

    conn.commit()
    conn.close()


def mark_game_processed(
    game_id: str,
    game_date: str,
    home_team: str,
    away_team: str,
    season: str = "current",
) -> None:
    """
    Insert or update a row in the games table to mark a game as processed.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO games (game_id, game_date, home_team, away_team, season)
        VALUES (?, ?, ?, ?, ?)
        """,
        (game_id, game_date, home_team, away_team, season),
    )
    conn.commit()
    conn.close()


def is_game_processed(game_id: str) -> bool:
    """
    Return True if the game_id already exists in the games table.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM games WHERE game_id = ?", (game_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def load_player_game_stats_for_season(season_label: str | None = None) -> pd.DataFrame:
    """
    Load all rows from player_game_stats as a DataFrame.

    For now the DB is effectively single-season, so `season_label` is accepted
    just for API compatibility with callers but is not used to filter rows.
    If you later add a season column to the table, you can plug it in here.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM player_game_stats", conn)
    finally:
        conn.close()
    return df


def insert_player_game_stats(per_game_df: pd.DataFrame) -> None:
    """
    Insert per-game, per-player stats into player_game_stats.
    """
    if per_game_df.empty:
        return

    expected_cols = [
        "game_id",
        "team",
        "player_id",
        "player_name",
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

    # Keep only columns that exist; missing ones will be filled with NULL
    cols_present = [c for c in expected_cols if c in per_game_df.columns]
    df = per_game_df[cols_present].copy()

    # Reindex to include all expected columns, adding missing as NaN/None
    df = df.reindex(columns=expected_cols)

    # 🔧 Fill NOT NULL numeric columns with 0.0 if missing
    not_null_numeric_cols = [
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
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
    ]

    for col in not_null_numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    conn = get_connection()
    cur = conn.cursor()

    records = df.to_records(index=False)

    placeholders = ", ".join(["?"] * len(expected_cols))
    columns_sql = ",\n            ".join(expected_cols)

    sql = f"""
        INSERT OR REPLACE INTO player_game_stats (
            {columns_sql}
        ) VALUES (
            {placeholders}
        )
    """

    cur.executemany(sql, records)

    conn.commit()
    conn.close()

    # ---------------------------------------------------------------------------


# impact_lineup_stints: cache of stint-level impact rows per team/season
#
# We store each stint row as a JSON blob (stint_json) keyed by:
#   (team, season, game_id, lineup_stint_index)
#
# This keeps the schema flexible: if the stint builder adds columns later,
# they are still captured in the JSON without needing a DB migration.
# ---------------------------------------------------------------------------

IMPACT_LINEUP_STINTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS impact_lineup_stints (
    team TEXT NOT NULL,
    season TEXT NOT NULL,
    game_id TEXT NOT NULL,
    lineup_stint_index INTEGER NOT NULL,
    stint_json TEXT NOT NULL,
    PRIMARY KEY (team, season, game_id, lineup_stint_index)
);
"""


def ensure_impact_lineup_stints_table(conn: sqlite3.Connection) -> None:
    """
    Ensure the impact_lineup_stints table exists.

    This should be called once per process before reading/writing the table.
    """
    with conn:
        conn.execute(IMPACT_LINEUP_STINTS_TABLE_SQL)


def _json_default(obj: Any) -> Any:
    """
    Helper for json.dumps to handle numpy / pandas scalar types.

    Converts numpy scalar types to native Python types so they serialize cleanly.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def upsert_impact_lineup_stints_for_team_season(
    conn: sqlite3.Connection,
    team: str,
    season: str,
    stints: pd.DataFrame,
) -> int:
    """
    Upsert stint-level rows for a given (team, season) into impact_lineup_stints.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to player_stats.db (use get_connection()).
    team : str
        Team abbreviation (e.g. 'NYK').
    season : str
        Season label (e.g. '2025-26').
    stints : pd.DataFrame
        DataFrame containing at least ['game_id', 'lineup_stint_index'] plus any
        other columns produced by the stint builder. Each row is stored as a
        JSON blob in 'stint_json'.

    Returns
    -------
    int
        Number of rows upserted.
    """
    if stints.empty:
        return 0

    if "game_id" not in stints.columns:
        raise ValueError("stints DataFrame is missing required column 'game_id'.")
    if "lineup_stint_index" not in stints.columns:
        raise ValueError(
            "stints DataFrame is missing required column 'lineup_stint_index'."
        )

    # Normalize to strings/ints for keys
    rows: list[tuple[str, str, str, int, str]] = []
    for _, row in stints.iterrows():
        game_id = str(row["game_id"])
        lineup_idx = int(row["lineup_stint_index"])
        stint_json = json.dumps(row.to_dict(), default=_json_default)
        rows.append((team, season, game_id, lineup_idx, stint_json))

    with conn:
        conn.executemany(
            """
            INSERT INTO impact_lineup_stints (
                team, season, game_id, lineup_stint_index, stint_json
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(team, season, game_id, lineup_stint_index)
            DO UPDATE SET
                stint_json = excluded.stint_json
            """,
            rows,
        )

    return len(rows)


def load_impact_lineup_stints_for_team_season(
    conn: sqlite3.Connection,
    team: str,
    season: str,
) -> pd.DataFrame:
    """
    Load all stint rows for (team, season) from impact_lineup_stints.

    Returns a DataFrame reconstructed from the stored JSON blobs.
    If no rows exist, returns an empty DataFrame.
    """
    cur = conn.execute(
        """
        SELECT stint_json
        FROM impact_lineup_stints
        WHERE team = ? AND season = ?
        ORDER BY game_id, lineup_stint_index
        """,
        (team, season),
    )

    dicts: list[dict[str, Any]] = []
    for (stint_json,) in cur.fetchall():
        try:
            d = json.loads(stint_json)
            dicts.append(d)
        except json.JSONDecodeError:
            # Skip any corrupted rows rather than blowing up the whole load
            continue

    if not dicts:
        return pd.DataFrame()

    return pd.DataFrame(dicts)
