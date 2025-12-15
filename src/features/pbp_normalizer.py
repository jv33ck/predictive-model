# src/features/pbp_normalizer.py
from __future__ import annotations

from typing import Optional

import pandas as pd


def _parse_clock_to_seconds(clock: Optional[str]) -> Optional[int]:
    """
    Convert clock into seconds remaining in the period.

    Supports:
    - 'MM:SS' (e.g., '11:26')
    - ISO-like 'PT11M26.00S' (as returned by PlayByPlayV3)
    """
    if not isinstance(clock, str):
        return None

    # Format 1: 'MM:SS'
    if ":" in clock and not clock.startswith("PT"):
        parts = clock.split(":")
        if len(parts) != 2:
            return None
        minutes_str, seconds_str = parts
        try:
            minutes = int(minutes_str)
            seconds = int(seconds_str)
            return minutes * 60 + seconds
        except ValueError:
            return None

    # Format 2: 'PT11M26.00S'
    if clock.startswith("PT"):
        # strip leading 'PT' and trailing 'S'
        body = clock[2:]
        if body.endswith("S"):
            body = body[:-1]

        # Now expect something like '11M26.00' or '11M26'
        if "M" in body:
            minutes_str, seconds_str = body.split("M", 1)
            try:
                minutes = int(minutes_str)
                # seconds part may be '26.00' -> float
                seconds = int(float(seconds_str))
                return minutes * 60 + seconds
            except ValueError:
                return None

    return None


def normalize_pbp_v3(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw PlayByPlayV3 DataFrame into a canonical event schema.
    """
    if pbp_df.empty:
        return pbp_df.copy()

    df = pbp_df.copy()

    # Ensure sorted so score diffs and time flow are consistent
    sort_cols = [c for c in ["gameId", "period", "actionNumber"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ---- Time handling ----
    if "clock" in df.columns:
        df["seconds_remaining"] = df["clock"].apply(_parse_clock_to_seconds)
    else:
        df["seconds_remaining"] = None

    # ---- Score handling: convert to numeric before doing diffs ----
    for col in ["scoreHome", "scoreAway"]:
        if col not in df.columns:
            df[col] = 0
        # Convert to numeric, treating '', None, etc. as NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill scores within each game, then fill any remaining NaNs with 0
    if "gameId" in df.columns:
        df[["scoreHome", "scoreAway"]] = df.groupby("gameId")[
            ["scoreHome", "scoreAway"]
        ].ffill()
    df["scoreHome"] = df["scoreHome"].fillna(0).astype(int)
    df["scoreAway"] = df["scoreAway"].fillna(0).astype(int)

    # Compute per-event scoring deltas within each game
    df["scoreHome_prev"] = df.groupby("gameId")["scoreHome"].shift(1)
    df["scoreAway_prev"] = df.groupby("gameId")["scoreAway"].shift(1)

    # For the first event of the game, assume no prior score change
    df["scoreHome_prev"] = df["scoreHome_prev"].fillna(df["scoreHome"])
    df["scoreAway_prev"] = df["scoreAway_prev"].fillna(df["scoreAway"])

    df["points_home"] = df["scoreHome"] - df["scoreHome_prev"]
    df["points_away"] = df["scoreAway"] - df["scoreAway_prev"]
    df["points"] = df["points_home"] + df["points_away"]
    df["is_scoring_event"] = df["points"] != 0

    # ---- Build canonical subset with standardized names ----
    canonical_cols = {
        "gameId": "game_id",
        "actionNumber": "event_number",
        "period": "period",
        "clock": "clock",
        "seconds_remaining": "seconds_remaining",
        "teamId": "team_id",
        "teamTricode": "team",
        "personId": "player_id",
        "playerName": "player_name",
        "shotDistance": "shot_distance",
        "shotResult": "shot_result",
        "isFieldGoal": "is_field_goal",
        "scoreHome": "score_home",
        "scoreAway": "score_away",
        "pointsTotal": "points_total",
        "description": "event_description",
        "actionType": "action_type",
        "subType": "action_subtype",
        "points_home": "points_home",
        "points_away": "points_away",
        "points": "points",
        "is_scoring_event": "is_scoring_event",
    }

    existing_input_cols = [c for c in canonical_cols.keys() if c in df.columns]
    renamed = df[existing_input_cols].rename(columns=canonical_cols)

    return renamed
