# src/features/possession_lineups.py
from __future__ import annotations

from typing import List

import pandas as pd


SECONDS_PER_PERIOD = 12 * 60  # 12-minute quarters


def _possession_time_to_rotation_seconds(
    period: int, seconds_remaining: float
) -> float:
    """
    Convert (period, seconds_remaining_in_period) into the same time scale used
    by GameRotation / lineup stints.

    - PBP / possessions: seconds_remaining = time left in the period (real seconds)
    - Game time elapsed (real seconds) = (period - 1)*720 + (720 - seconds_remaining)
    - GameRotation time is in tenths of a second, so multiply by 10.
    """
    elapsed_in_period = SECONDS_PER_PERIOD - seconds_remaining
    elapsed_game_seconds = (period - 1) * SECONDS_PER_PERIOD + elapsed_in_period
    return elapsed_game_seconds * 10.0  # tenths of a second


def attach_lineups_to_possessions(
    possessions: pd.DataFrame,
    lineup_stints: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach lineup information (5-on-5 players) to each possession.

    Args:
        possessions:
            DataFrame from build_all_possessions_for_game() (or the older
            build_scoring_possessions_for_game()), with columns like:
                - game_id
                - possession_index
                - period
                - seconds_remaining  (time left in period, real seconds)
                - offense_team
                - defense_team
                - points

        lineup_stints:
            DataFrame from build_lineup_stints_for_game(), with columns:
                - game_id
                - stint_index
                - start_seconds
                - end_seconds
                - home_team
                - away_team
                - home_player_ids (List[int])
                - away_player_ids (List[int])

    Returns:
        New DataFrame with all original possession columns plus:
            - rotation_time   (float, tenths of seconds from game start)
            - lineup_stint_index
            - lineup_start_seconds
            - lineup_end_seconds
            - home_team
            - away_team
            - home_player_ids
            - away_player_ids
            - offense_player_ids : List[int]
            - defense_player_ids : List[int]
    """
    if possessions.empty or lineup_stints.empty:
        return possessions.copy()

    sp = possessions.copy()
    ls = lineup_stints.copy()

    if "seconds_remaining" not in sp.columns:
        raise ValueError("possessions must have a 'seconds_remaining' column.")

    # Compute rotation-time (tenths of a second) for each possession
    sp["rotation_time"] = sp.apply(
        lambda row: _possession_time_to_rotation_seconds(
            int(row["period"]),
            float(row["seconds_remaining"]),
        ),
        axis=1,
    )

    result_rows: List[dict] = []

    for _, row in sp.iterrows():
        game_id = row["game_id"]
        offense_team = str(row["offense_team"]).upper()
        defense_team = str(row["defense_team"]).upper()
        t_rot = float(row["rotation_time"])

        ls_game = ls[ls["game_id"] == game_id]

        stint_matches = ls_game[
            (ls_game["start_seconds"] <= t_rot) & (ls_game["end_seconds"] > t_rot)
        ]

        if stint_matches.empty:
            enriched = row.to_dict()
            enriched.update(
                {
                    "lineup_stint_index": None,
                    "lineup_start_seconds": None,
                    "lineup_end_seconds": None,
                    "home_team": None,
                    "away_team": None,
                    "home_player_ids": None,
                    "away_player_ids": None,
                    "offense_player_ids": None,
                    "defense_player_ids": None,
                }
            )
            result_rows.append(enriched)
            continue

        stint = stint_matches.iloc[0]

        home_team = str(stint["home_team"]).upper()
        away_team = str(stint["away_team"]).upper()
        home_players = stint["home_player_ids"]
        away_players = stint["away_player_ids"]

        if offense_team == home_team and defense_team == away_team:
            offense_players = home_players
            defense_players = away_players
        elif offense_team == away_team and defense_team == home_team:
            offense_players = away_players
            defense_players = home_players
        else:
            # Team abbreviations don't line up; something's off.
            offense_players = None
            defense_players = None

        enriched = row.to_dict()
        enriched.update(
            {
                "lineup_stint_index": int(stint["stint_index"]),
                "lineup_start_seconds": float(stint["start_seconds"]),
                "lineup_end_seconds": float(stint["end_seconds"]),
                "home_team": home_team,
                "away_team": away_team,
                "home_player_ids": home_players,
                "away_player_ids": away_players,
                "offense_player_ids": offense_players,
                "defense_player_ids": defense_players,
            }
        )
        result_rows.append(enriched)

    return pd.DataFrame(result_rows)
