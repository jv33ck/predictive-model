# src/features/lineup_stints.py
from __future__ import annotations

from typing import List

import pandas as pd


def build_lineup_stints_for_game(stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lineup stints (5-on-5 segments) for a single game from a player-stints
    DataFrame (output of build_player_stints_from_rotation).

    Args:
        stints_df:
            DataFrame with columns including:
                - game_id
                - team_abbrev
                - is_home (bool)
                - player_id
                - in_seconds  (float, seconds from game start)
                - out_seconds (float, seconds from game start)
                - stint_duration_seconds

            Must contain only ONE game_id.

    Returns:
        DataFrame with one row per lineup stint and columns:
            - game_id
            - stint_index
            - start_seconds
            - end_seconds
            - stint_duration_seconds
            - home_team
            - away_team
            - home_player_ids : List[int]
            - away_player_ids : List[int]
    """
    if stints_df.empty:
        return pd.DataFrame()

    df = stints_df.copy()

    # --- Validate single-game assumption ---
    game_ids = df["game_id"].unique()
    if len(game_ids) != 1:
        raise ValueError(f"Expected stints_df for a single game_id, found {game_ids}.")
    game_id = game_ids[0]

    # --- Infer home/away teams ---
    home_teams = df.loc[df["is_home"] == True, "team_abbrev"].unique()
    away_teams = df.loc[df["is_home"] == False, "team_abbrev"].unique()

    if len(home_teams) != 1 or len(away_teams) != 1:
        raise ValueError(
            f"Expected exactly one home and one away team, got home={home_teams}, away={away_teams}"
        )

    home_team = home_teams[0]
    away_team = away_teams[0]

    # --- Collect time boundaries (when any player subbed in/out) ---
    boundaries_series = pd.concat(
        [df["in_seconds"], df["out_seconds"]],
        ignore_index=True,
    )

    boundaries = (
        boundaries_series.dropna()  # remove NaNs
        .unique()  # unique values
        .tolist()  # convert to Python list
    )

    boundaries = sorted(float(b) for b in boundaries)
    # No meaningful stints if we don't have at least 2 distinct times
    if len(boundaries) < 2:
        return pd.DataFrame()

    home_df = df[df["is_home"] == True].copy()
    away_df = df[df["is_home"] == False].copy()

    lineup_rows: List[dict] = []
    stint_index = 0

    # Iterate over consecutive boundary pairs [start, end)
    for i in range(len(boundaries) - 1):
        start = float(boundaries[i])
        end = float(boundaries[i + 1])

        # Skip zero-length windows, just in case
        if end <= start:
            continue

        # Find players on court at 'start' time
        # Condition: in_seconds <= start < out_seconds
        home_on = home_df[
            (home_df["in_seconds"] <= start) & (home_df["out_seconds"] > start)
        ]
        away_on = away_df[
            (away_df["in_seconds"] <= start) & (away_df["out_seconds"] > start)
        ]

        # We're only interested in proper 5-on-5 lineups
        if len(home_on) != 5 or len(away_on) != 5:
            continue

        # Build sorted lists of player IDs (for stable ordering)
        home_player_ids = sorted(home_on["player_id"].astype(int).tolist())
        away_player_ids = sorted(away_on["player_id"].astype(int).tolist())

        lineup_rows.append(
            {
                "game_id": game_id,
                "stint_index": stint_index,
                "start_seconds": start,
                "end_seconds": end,
                "stint_duration_seconds": end - start,
                "home_team": home_team,
                "away_team": away_team,
                "home_player_ids": home_player_ids,
                "away_player_ids": away_player_ids,
            }
        )
        stint_index += 1

    if not lineup_rows:
        return pd.DataFrame()

    return pd.DataFrame(lineup_rows)
