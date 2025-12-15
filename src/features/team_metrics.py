# src/features/team_metrics.py
from __future__ import annotations

import pandas as pd


def compute_team_game_ratings_from_scoring_possessions(
    scoring_possessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute simple team-level offensive/defensive ratings per game
    using *scoring* possessions only.

    Args:
        scoring_possessions: DataFrame from build_scoring_possessions_for_game()
        with columns including:
            - game_id
            - offense_team
            - defense_team
            - points
            - points_home
            - points_away

    Returns:
        DataFrame with one row per (game_id, team), columns:
            - game_id
            - team
            - role (e.g. 'home' or 'away' if inferable)
            - points_for
            - points_against
            - scoring_possessions_off
            - scoring_possessions_def
            - ortg_scoring_only   (per 100 scoring possessions)
            - drtg_scoring_only
            - netrtg_scoring_only
    """
    if scoring_possessions.empty:
        return pd.DataFrame()

    df = scoring_possessions.copy()

    # Offensive side: group by game & offense_team
    off_group = (
        df.groupby(["game_id", "offense_team"])
        .agg(
            points_for=("points", "sum"),
            scoring_possessions_off=("points", "count"),
        )
        .reset_index()
        .rename(columns={"offense_team": "team"})
    )

    # Defensive side: group by game & defense_team
    def_group = (
        df.groupby(["game_id", "defense_team"])
        .agg(
            points_against=("points", "sum"),
            scoring_possessions_def=("points", "count"),
        )
        .reset_index()
        .rename(columns={"defense_team": "team"})
    )

    # Merge offense & defense info
    merged = pd.merge(
        off_group,
        def_group,
        on=["game_id", "team"],
        how="outer",
    )

    # Fill any missing values with 0
    for col in [
        "points_for",
        "scoring_possessions_off",
        "points_against",
        "scoring_possessions_def",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(float)

    # Compute offensive/defensive ratings based on scoring possessions only
    # Note: these are not true ORtg/DRtg until we include *all* possessions.
    merged["ortg_scoring_only"] = merged.apply(
        lambda row: (
            100 * row["points_for"] / row["scoring_possessions_off"]
            if row["scoring_possessions_off"] > 0
            else 0.0
        ),
        axis=1,
    )
    merged["drtg_scoring_only"] = merged.apply(
        lambda row: (
            100 * row["points_against"] / row["scoring_possessions_def"]
            if row["scoring_possessions_def"] > 0
            else 0.0
        ),
        axis=1,
    )
    merged["netrtg_scoring_only"] = (
        merged["ortg_scoring_only"] - merged["drtg_scoring_only"]
    )

    return merged
