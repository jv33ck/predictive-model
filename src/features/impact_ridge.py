# src/features/impact_ridge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class RidgeDesignMatrix:
    """
    Container for the design matrix and metadata used in the RAPM ridge model.
    """

    X: np.ndarray  # shape: (n_stints, n_players)
    Y: np.ndarray  # shape: (n_stints, 3) -> [y_off, y_def, y_net]
    sample_weight: np.ndarray  # shape: (n_stints,)
    player_ids: List[int]


def _resolve_player_id_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Determine which columns to use for home/away player ID lists.

    Accepts either:
      - 'home_player_ids_norm' / 'away_player_ids_norm', or
      - 'home_player_ids'      / 'away_player_ids'

    Raises a clear error if neither pair is present.
    """
    home_col: str | None = None
    away_col: str | None = None

    if "home_player_ids_norm" in df.columns and "away_player_ids_norm" in df.columns:
        home_col = "home_player_ids_norm"
        away_col = "away_player_ids_norm"
    elif "home_player_ids" in df.columns and "away_player_ids" in df.columns:
        home_col = "home_player_ids"
        away_col = "away_player_ids"

    if home_col is None or away_col is None:
        raise ValueError(
            "Could not find player ID list columns in stints_df. "
            "Expected either ['home_player_ids_norm','away_player_ids_norm'] "
            "or ['home_player_ids','away_player_ids']. "
            f"Available columns: {list(df.columns)}"
        )

    return home_col, away_col


def _extract_unique_player_ids(
    stints_df: pd.DataFrame,
    home_col: str,
    away_col: str,
) -> List[int]:
    """
    Collect sorted unique player IDs from home/away player ID list columns.

    Expects that home_col and away_col are names of list-like columns.
    """
    player_ids_set: set[int] = set()

    for _, row in stints_df.iterrows():
        for col in (home_col, away_col):
            ids = row.get(col)
            if isinstance(ids, (list, tuple)):
                for pid in ids:
                    try:
                        player_ids_set.add(int(pid))
                    except (TypeError, ValueError):
                        continue

    return sorted(player_ids_set)


def _resolve_scoring_columns(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determine points_for, points_against, net_points arrays from the DataFrame.

    Supported patterns:

      1) Already team-centric:
         - 'points_for' and 'points_against' present
         - optional 'net_points' (computed if missing)

      2) Home/away scoring with a team-side indicator:
         - 'points_home' / 'points_away'
         - AND either:
             - 'team_is_home' boolean, OR
             - 'team', 'home_team', 'away_team' strings

      3) Aggregated home/away sums:
         - 'points_home_sum' / 'points_away_sum'
         - optional 'net_points_home' (computed if missing),
           interpreted from the home team perspective.

    Returns:
      points_for, points_against, net_points  (all np.ndarray[float])
    """
    cols = set(df.columns)

    # Case 1: direct team-centric columns
    if {"points_for", "points_against"}.issubset(cols):
        points_for = df["points_for"].to_numpy(dtype=float)
        points_against = df["points_against"].to_numpy(dtype=float)
        if "net_points" in cols:
            net_points = df["net_points"].to_numpy(dtype=float)
        else:
            net_points = points_for - points_against
        return points_for, points_against, net_points

    # Case 2: home/away with team side info
    if {"points_home", "points_away"}.issubset(cols):
        points_home = df["points_home"].to_numpy(dtype=float)
        points_away = df["points_away"].to_numpy(dtype=float)

        # Prefer explicit team_is_home if present
        if "team_is_home" in cols:
            is_home = df["team_is_home"].astype(bool).to_numpy()
        elif {"team", "home_team", "away_team"}.issubset(cols):
            # Derive team_is_home from strings: team == home_team
            is_home = (df["team"] == df["home_team"]).to_numpy()
        else:
            raise ValueError(
                "Found 'points_home' and 'points_away' but no 'team_is_home' or "
                "['team','home_team','away_team'] to interpret which side we model. "
                f"Available columns: {list(df.columns)}"
            )

        # Team-centric from their perspective
        points_for = np.where(is_home, points_home, points_away)
        points_against = np.where(is_home, points_away, points_home)
        net_points = points_for - points_against
        return points_for, points_against, net_points

    # Case 3: aggregated home/away sums with home net points
    if {"points_home_sum", "points_away_sum"}.issubset(cols):
        points_home = df["points_home_sum"].to_numpy(dtype=float)
        points_away = df["points_away_sum"].to_numpy(dtype=float)
        points_for = points_home
        points_against = points_away
        if "net_points_home" in cols:
            net_points = df["net_points_home"].to_numpy(dtype=float)
        else:
            net_points = points_for - points_against
        return points_for, points_against, net_points

    # If we get here, we don't know how to get scoring info
    raise ValueError(
        "Could not find suitable scoring columns in stints_df. "
        "Expected either ['points_for','points_against'] or "
        "['points_home','points_away', plus team side info], or "
        "['points_home_sum','points_away_sum',(optional) 'net_points_home']. "
        f"Available columns: {list(df.columns)}"
    )


def build_ridge_design_matrix_from_stints(
    stints_df: pd.DataFrame,
    min_possessions_per_stint: int = 3,
) -> RidgeDesignMatrix:
    """
    Build X, Y, and sample_weight for a RAPM-style ridge regression from lineup stints.

    Expected data in stints_df (flexible naming):

    Player IDs (one of):
      - home_player_ids_norm / away_player_ids_norm
      - home_player_ids      / away_player_ids

    Possessions (required):
      - 'possessions': float  # possessions in this stint

    Scoring (one of):
      - 'points_for' / 'points_against' (team-centric), or
      - 'points_home' / 'points_away' plus team-side info:
          - 'team_is_home' (bool), OR
          - 'team', 'home_team', 'away_team'
      - 'points_home_sum' / 'points_away_sum' (home-centric aggregates)

    Targets Y are per-possession:
      - y_off  = points_for / possessions
      - y_def  = - points_against / possessions   (sign flipped so higher is better)
      - y_net  = net_points / possessions
    """
    if "possessions" not in stints_df.columns:
        raise ValueError(
            "stints_df is missing 'possessions' column. "
            f"Available columns: {list(stints_df.columns)}"
        )

    df = stints_df.copy()

    # Resolve which cols to use for player IDs
    home_col, away_col = _resolve_player_id_columns(df)

    # Filter out very small stints
    df = df[df["possessions"] >= float(min_possessions_per_stint)].reset_index(
        drop=True
    )
    if df.empty:
        raise ValueError(
            "No stints remain after filtering by min_possessions_per_stint."
        )

    # Collect unique players from the chosen ID columns
    player_ids = _extract_unique_player_ids(df, home_col=home_col, away_col=away_col)
    if not player_ids:
        raise ValueError("No player IDs found in stints_df.")

    n_stints = len(df)
    n_players = len(player_ids)

    id_to_idx: Dict[int, int] = {pid: j for j, pid in enumerate(player_ids)}

    # Design matrix: +1 for home players, -1 for away players
    X = np.zeros((n_stints, n_players), dtype=np.float32)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        home_ids = row.get(home_col) or []
        away_ids = row.get(away_col) or []

        # Home players = +1
        for pid in home_ids:
            try:
                j = id_to_idx[int(pid)]
                X[row_idx, j] += 1.0
            except (KeyError, TypeError, ValueError):
                continue

        # Away players = -1
        for pid in away_ids:
            try:
                j = id_to_idx[int(pid)]
                X[row_idx, j] -= 1.0
            except (KeyError, TypeError, ValueError):
                continue

    # Targets / scoring
    poss = df["possessions"].to_numpy(dtype=float)

    points_for, points_against, net_points = _resolve_scoring_columns(df)

    # Per-possession targets
    denom = np.clip(poss, 1e-6, None)

    y_off = points_for / denom
    y_def_raw = points_against / denom
    # For defense: fewer points allowed is "better", so we flip the sign
    y_def = -y_def_raw
    y_net = net_points / denom

    Y = np.stack([y_off, y_def, y_net], axis=1)  # (n_stints, 3)

    # Sample weights: weight by possessions so longer stints matter more
    sample_weight = poss.astype(float)

    return RidgeDesignMatrix(
        X=X,
        Y=Y,
        sample_weight=sample_weight,
        player_ids=player_ids,
    )


def fit_ridge_impact_model(
    stints_df: pd.DataFrame,
    alpha: float = 50.0,
    min_possessions_per_stint: int = 3,
) -> pd.DataFrame:
    """
    Fit a multi-output ridge RAPM model (off/def/net) from lineup stints.

    Returns a DataFrame with one row per player and columns:
      - player_id
      - impact_off_per_possession
      - impact_def_per_possession
      - impact_per_possession (net)
      - impact_off_per_100
      - impact_def_per_100
      - impact_per_100 (net)
      - exposure_stint_units  (possession-weighted exposure)
    """
    design = build_ridge_design_matrix_from_stints(
        stints_df=stints_df,
        min_possessions_per_stint=min_possessions_per_stint,
    )

    X = design.X
    Y = design.Y
    sample_weight = design.sample_weight
    player_ids = design.player_ids

    if X.shape[0] == 0:
        raise ValueError("Design matrix X has zero rows; cannot fit model.")

    # Multi-output ridge regression: Y has 3 targets (off, def, net)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, Y, sample_weight=sample_weight)

    coef = model.coef_.astype(float)  # shape: (3, n_players)
    if coef.shape[0] != 3:
        raise RuntimeError(
            f"Expected 3 targets (off, def, net) but got coef shape {coef.shape}."
        )

    impact_off_per_poss = coef[0, :]
    impact_def_per_poss = coef[1, :]
    impact_net_per_poss = coef[2, :]

    # Possession-weighted exposure for each player
    poss = sample_weight  # already possessions per stint
    exposure = np.abs(X).T @ poss  # (n_players,)

    df_out = pd.DataFrame(
        {
            "player_id": player_ids,
            "impact_off_per_possession": impact_off_per_poss,
            "impact_def_per_possession": impact_def_per_poss,
            "impact_per_possession": impact_net_per_poss,
            "impact_off_per_100": impact_off_per_poss * 100.0,
            "impact_def_per_100": impact_def_per_poss * 100.0,
            "impact_per_100": impact_net_per_poss * 100.0,
            "exposure_stint_units": exposure,
        }
    )

    return df_out
