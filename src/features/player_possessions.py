# src/features/player_possessions.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple, Any

import pandas as pd


def aggregate_player_possession_stats(
    enriched_possessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-player possession stats from enriched scoring possessions.

    Args:
        enriched_possessions:
            DataFrame from attach_lineups_to_scoring_possessions(), with columns:
                - game_id
                - offense_team
                - defense_team
                - points
                - offense_player_ids : List[int] or None
                - defense_player_ids : List[int] or None

            (Other columns are preserved but not required here.)

    Returns:
        DataFrame with one row per (game_id, team, player_id), including:
            - game_id
            - team
            - player_id
            - off_possessions
            - def_possessions
            - total_possessions
            - off_points_for
            - def_points_against
            - net_points    (off_points_for - def_points_against)
            - off_rating_per_100_scoring  (per 100 scoring possessions)
            - def_rating_per_100_scoring
            - net_rating_per_100_scoring
    """
    if enriched_possessions.empty:
        return pd.DataFrame()

    # We'll accumulate stats in a dict keyed by (game_id, team, player_id)
    stats: Dict[Tuple[Any, Any, Any], Dict[str, float]] = defaultdict(
        lambda: {
            "off_possessions": 0.0,
            "def_possessions": 0.0,
            "off_points_for": 0.0,
            "def_points_against": 0.0,
        }
    )

    for row in enriched_possessions.itertuples():
        game_id = row.game_id
        offense_team = str(row.offense_team).upper()
        defense_team = str(row.defense_team).upper()
        points = float(getattr(row, "points", 0.0) or 0.0)

        offense_players = getattr(row, "offense_player_ids", None) or []
        defense_players = getattr(row, "defense_player_ids", None) or []

        # Offensive side: each offensive player gets an offensive possession + points_for
        for pid in offense_players:
            key = (game_id, offense_team, int(pid))
            s = stats[key]
            s["off_possessions"] += 1.0
            s["off_points_for"] += points

        # Defensive side: each defensive player gets a defensive possession + points_against
        for pid in defense_players:
            key = (game_id, defense_team, int(pid))
            s = stats[key]
            s["def_possessions"] += 1.0
            s["def_points_against"] += points

    # Convert accumulated stats into a DataFrame
    rows = []
    for (game_id, team, player_id), s in stats.items():
        off_poss = s["off_possessions"]
        def_poss = s["def_possessions"]
        total_poss = off_poss + def_poss

        off_pts = s["off_points_for"]
        def_pa = s["def_points_against"]
        net_pts = off_pts - def_pa

        # Ratings per 100 "scoring possessions" (for now — we'll refine later)
        off_rating = 100.0 * off_pts / off_poss if off_poss > 0 else 0.0
        def_rating = 100.0 * def_pa / def_poss if def_poss > 0 else 0.0
        net_rating = 100.0 * net_pts / total_poss if total_poss > 0 else 0.0

        rows.append(
            {
                "game_id": game_id,
                "team": team,
                "player_id": player_id,
                "off_possessions": off_poss,
                "def_possessions": def_poss,
                "total_possessions": total_poss,
                "off_points_for": off_pts,
                "def_points_against": def_pa,
                "net_points": net_pts,
                "off_rating_per_100": off_rating,
                "def_rating_per_100": def_rating,
                "net_rating_per_100": net_rating,
            }
        )
    return pd.DataFrame(rows)
