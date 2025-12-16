# src/db/impact_ratings_db.py
"""
SQLite helpers for storing and loading player impact ratings.

This lives alongside player_stats_db.py and uses the same database file
(data/player_stats.db by default).

Table schema
------------

Table: impact_ratings

Primary key: (team, season, player_id)

Columns:
  - team                         TEXT NOT NULL
  - season                       TEXT NOT NULL
  - player_id                    INTEGER NOT NULL
  - player_name                  TEXT
  - impact_off_per_possession    REAL
  - impact_def_per_possession    REAL
  - impact_per_possession        REAL
  - impact_off_per_100           REAL
  - impact_def_per_100           REAL
  - impact_per_100               REAL
  - exposure_stint_units         REAL
  - last_updated_utc             TEXT
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Paths / connection helpers
# ---------------------------------------------------------------------------


def _default_db_path() -> Path:
    """
    Return the default SQLite DB path, aligned with the rest of the project.

    We assume this file lives at: src/db/impact_ratings_db.py
    So the project root is two levels up from here.
    """
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "player_stats.db"


def get_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """
    Open a SQLite connection to the given DB path (or default).

    Parameters
    ----------
    db_path : Optional[Path | str]
        If provided, use this path. Otherwise, use the project default.

    Returns
    -------
    sqlite3.Connection
    """
    path = Path(db_path) if db_path is not None else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    # Return rows as dict-like objects when needed
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------


def ensure_impact_table_exists(conn: sqlite3.Connection) -> None:
    """
    Create the impact_ratings table if it does not already exist.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS impact_ratings (
            team TEXT NOT NULL,
            season TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT,
            impact_off_per_possession REAL,
            impact_def_per_possession REAL,
            impact_per_possession REAL,
            impact_off_per_100 REAL,
            impact_def_per_100 REAL,
            impact_per_100 REAL,
            exposure_stint_units REAL,
            last_updated_utc TEXT,
            PRIMARY KEY (team, season, player_id)
        )
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ImpactRow:
    team: str
    season: str
    player_id: int
    player_name: Optional[str]
    impact_off_per_possession: Optional[float]
    impact_def_per_possession: Optional[float]
    impact_per_possession: Optional[float]
    impact_off_per_100: Optional[float]
    impact_def_per_100: Optional[float]
    impact_per_100: Optional[float]
    exposure_stint_units: Optional[float]
    last_updated_utc: str


def upsert_impact_ratings(
    team: str,
    season: str,
    ratings_df: pd.DataFrame,
    db_path: Optional[Path | str] = None,
) -> None:
    """
    Insert or replace impact ratings for a single team/season.

    Parameters
    ----------
    team : str
        Team abbreviation (e.g. "OKC").
    season : str
        Season label (e.g. "2025-26").
    ratings_df : pd.DataFrame
        DataFrame with at least:
          ['player_id',
           'player_name',
           'impact_off_per_possession',
           'impact_def_per_possession',
           'impact_per_possession',
           'impact_off_per_100',
           'impact_def_per_100',
           'impact_per_100',
           'exposure_stint_units']
    db_path : Optional[Path | str]
        Optional custom DB path. Defaults to the project standard.
    """
    if ratings_df.empty:
        print(f"⚠️ No impact ratings to upsert for team {team}, season {season}.")
        return

    conn = get_connection(db_path)
    try:
        ensure_impact_table_exists(conn)

        # Make sure required columns exist
        required_cols = {
            "player_id",
            "player_name",
            "impact_off_per_possession",
            "impact_def_per_possession",
            "impact_per_possession",
            "impact_off_per_100",
            "impact_def_per_100",
            "impact_per_100",
            "exposure_stint_units",
        }
        missing = [c for c in required_cols if c not in ratings_df.columns]
        if missing:
            raise ValueError(
                f"ratings_df missing required columns for upsert_impact_ratings: {missing}"
            )

        now_utc = datetime.now(timezone.utc).isoformat()

        records = []
        for _, row in ratings_df.iterrows():
            records.append(
                (
                    team,
                    season,
                    int(row["player_id"]),
                    str(row.get("player_name") or ""),
                    _safe_float(row.get("impact_off_per_possession")),
                    _safe_float(row.get("impact_def_per_possession")),
                    _safe_float(row.get("impact_per_possession")),
                    _safe_float(row.get("impact_off_per_100")),
                    _safe_float(row.get("impact_def_per_100")),
                    _safe_float(row.get("impact_per_100")),
                    _safe_float(row.get("exposure_stint_units")),
                    now_utc,
                )
            )

        conn.executemany(
            """
            INSERT INTO impact_ratings (
                team,
                season,
                player_id,
                player_name,
                impact_off_per_possession,
                impact_def_per_possession,
                impact_per_possession,
                impact_off_per_100,
                impact_def_per_100,
                impact_per_100,
                exposure_stint_units,
                last_updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team, season, player_id) DO UPDATE SET
                player_name = excluded.player_name,
                impact_off_per_possession = excluded.impact_off_per_possession,
                impact_def_per_possession = excluded.impact_def_per_possession,
                impact_per_possession = excluded.impact_per_possession,
                impact_off_per_100 = excluded.impact_off_per_100,
                impact_def_per_100 = excluded.impact_def_per_100,
                impact_per_100 = excluded.impact_per_100,
                exposure_stint_units = excluded.exposure_stint_units,
                last_updated_utc = excluded.last_updated_utc
            """,
            records,
        )
        conn.commit()
        print(
            f"💾 Upserted {len(records)} impact rows for team {team}, season {season}."
        )
    finally:
        conn.close()


def load_impact_ratings(
    db_path: Optional[Path | str] = None,
    team: Optional[str] = None,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load impact ratings from the database as a DataFrame.

    Parameters
    ----------
    db_path : Optional[Path | str]
        Optional DB path; defaults to the project-standard player_stats.db.
    team : Optional[str]
        If provided, filter to this team.
    season : Optional[str]
        If provided, filter to this season.

    Returns
    -------
    pd.DataFrame
    """
    conn = get_connection(db_path)
    try:
        ensure_impact_table_exists(conn)

        sql = "SELECT * FROM impact_ratings WHERE 1=1"
        params: list[Any] = []

        if team is not None:
            sql += " AND team = ?"
            params.append(team)

        if season is not None:
            sql += " AND season = ?"
            params.append(season)

        df = pd.read_sql_query(sql, conn, params=params or None)
        return df
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_float(value: Any | None) -> Optional[float]:
    """Convert to float if not null/NaN; otherwise return None."""
    if value is None:
        return None

    # Handle numeric types directly
    if isinstance(value, (int, float)):
        f = float(value)
    # Handle strings that might represent numbers
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
    # Anything else we treat as invalid / non-convertible
    else:
        return None

    if pd.isna(f):
        return None
    return f
