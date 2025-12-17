# src/db/player_reference_db.py
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Dict, Mapping, Iterable, Tuple

DBPath = str | Path


def _get_connection(db_path: DBPath) -> sqlite3.Connection:
    """
    Small helper to open a SQLite connection.

    Callers are responsible for closing the connection.
    """
    return sqlite3.connect(str(db_path))


def ensure_player_reference_table(conn: sqlite3.Connection) -> None:
    """
    Ensure the player_reference table exists.

    Schema:
      - player_id: integer nba_api personId (PRIMARY KEY)
      - player_name: latest known display name
      - last_seen_season: last season label where we saw this player (e.g. '2025-26')
      - last_seen_team: last team abbreviation where we saw this player (e.g. 'OKC')
      - source: where we got the name ('db', 'nba_api', etc.)
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_reference (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT NOT NULL,
            last_seen_season TEXT,
            last_seen_team TEXT,
            source TEXT
        )
        """
    )
    conn.commit()


def upsert_player_reference(
    mapping: Mapping[int, str],
    db_path: DBPath = "data/player_stats.db",
    season_label: str | None = None,
    team_abbrev: str | None = None,
    source: str | None = None,
) -> None:
    """
    Upsert (player_id -> player_name) rows into player_reference.

    - mapping: dict of {player_id_int: player_name}
    - season_label / team_abbrev / source: optional metadata for last-seen context
    """
    if not mapping:
        return

    conn = _get_connection(db_path)
    try:
        ensure_player_reference_table(conn)

        rows: list[Tuple[int, str, str | None, str | None, str | None]] = []
        for pid, name in mapping.items():
            if name is None:
                continue
            rows.append((int(pid), str(name), season_label, team_abbrev, source))

        conn.executemany(
            """
            INSERT INTO player_reference (
                player_id,
                player_name,
                last_seen_season,
                last_seen_team,
                source
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                player_name = excluded.player_name,
                last_seen_season = COALESCE(
                    excluded.last_seen_season,
                    player_reference.last_seen_season
                ),
                last_seen_team = COALESCE(
                    excluded.last_seen_team,
                    player_reference.last_seen_team
                ),
                source = COALESCE(
                    excluded.source,
                    player_reference.source
                )
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def load_player_reference_map(
    db_path: DBPath = "data/player_stats.db",
) -> Dict[int, str]:
    """
    Load a simple {player_id: player_name} mapping from player_reference.
    """
    conn = _get_connection(db_path)
    try:
        ensure_player_reference_table(conn)
        cur = conn.execute("SELECT player_id, player_name FROM player_reference")
        rows = cur.fetchall()
        result: Dict[int, str] = {}
        for pid, name in rows:
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            result[pid_int] = str(name)
        return result
    finally:
        conn.close()
