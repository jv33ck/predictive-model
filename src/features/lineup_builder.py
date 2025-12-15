# src/features/lineup_builder.py
from __future__ import annotations

from typing import Optional

import pandas as pd


def _parse_real_time_to_seconds(t: Optional[str]) -> Optional[float]:
    """
    Convert GameRotation's IN_TIME_REAL / OUT_TIME_REAL into seconds from game start.

    Handles several possible formats:

    - Numeric (int/float) or numeric string: treated directly as seconds
    - ISO-like durations: 'PT11M26.00S', 'PT0M0.00S', 'PT0H11M26.00S'
    - Clock strings: 'MM:SS' or 'HH:MM:SS'

    Returns:
        float seconds, or None if parsing fails.
    """
    # Already numeric?
    if isinstance(t, (int, float)):
        return float(t)

    if not isinstance(t, str):
        return None

    s = t.strip()
    if s == "":
        return None

    # 1) Numeric string: '123.45'
    try:
        return float(s)
    except ValueError:
        pass

    # 2) ISO-like: 'PT11M26.00S' or 'PT0H11M26.00S'
    if s.startswith("PT"):
        # Remove leading 'PT' and trailing 'S' if present
        body = s[2:]
        if body.endswith("S"):
            body = body[:-1]

        hours = 0.0
        minutes = 0.0
        seconds = 0.0

        # Optional hours: split on 'H'
        if "H" in body:
            hours_str, body = body.split("H", 1)
            try:
                hours = float(hours_str)
            except ValueError:
                hours = 0.0

        # Optional minutes: split on 'M'
        if "M" in body:
            minutes_str, sec_str = body.split("M", 1)
            try:
                minutes = float(minutes_str)
            except ValueError:
                minutes = 0.0
            try:
                seconds = float(sec_str)
            except ValueError:
                seconds = 0.0
        else:
            # No minutes, only seconds
            try:
                seconds = float(body)
            except ValueError:
                seconds = 0.0

        return hours * 3600.0 + minutes * 60.0 + seconds

    # 3) Clock-like: 'MM:SS' or 'HH:MM:SS'
    parts = s.split(":")
    try:
        if len(parts) == 2:
            # MM:SS
            minutes = int(parts[0])
            seconds = int(float(parts[1]))
            return minutes * 60.0 + seconds
        elif len(parts) == 3:
            # HH:MM:SS
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(float(parts[2]))
            return hours * 3600.0 + minutes * 60.0 + seconds
    except ValueError:
        return None

    return None


def build_player_stints_from_rotation(rotation_df: pd.DataFrame) -> pd.DataFrame:
    ...
    if rotation_df.empty:
        return pd.DataFrame()

    df = rotation_df.copy()

    # 🔧 Drop the existing 'player_id' convenience column from get_game_rotation
    # so we don't end up with two player_id columns after renaming PERSON_ID.
    if "player_id" in df.columns:
        df = df.drop(columns=["player_id"])

    # Normalize column names we care about
    rename_map = {
        "GAME_ID": "game_id",
        "TEAM_ID": "team_id",
        "TEAM_NAME": "team_name",
        "PERSON_ID": "player_id",
        "PLAYER_PTS": "player_pts",
        "PT_DIFF": "pt_diff",
        "USG_PCT": "usg_pct",
        "IN_TIME_REAL": "in_time_real",
        "OUT_TIME_REAL": "out_time_real",
    }
    df = df.rename(columns=rename_map)
    # Keep a focused subset
    keep_cols = [
        "game_id",
        "team_id",
        "team_name",
        "team_abbrev",
        "is_home",
        "player_id",
        "player_name",
        "in_time_real",
        "out_time_real",
        "player_pts",
        "pt_diff",
        "usg_pct",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Parse real-time strings into numeric seconds from game start
    df["in_seconds"] = df["in_time_real"].apply(_parse_real_time_to_seconds)
    df["out_seconds"] = df["out_time_real"].apply(_parse_real_time_to_seconds)

    # Compute stint duration where we have both times
    df["stint_duration_seconds"] = df["out_seconds"] - df["in_seconds"]

    # Keep rows, but you can filter later when analyzing
    valid_mask = df["stint_duration_seconds"].notna() & (
        df["stint_duration_seconds"] >= 0
    )
    df = df[valid_mask].reset_index(drop=True)
    return df
