# src/features/matchup_features.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from features.team_schedule_features import build_team_schedule_features_df


@dataclass
class TeamAggregateFromProfiles:
    """
    Aggregated team-level stats from gameday player profiles.

    All per-player stats are aggregated using minutes_played as the weight.
    """

    team: str
    season: str

    num_players: int
    total_minutes: float

    epm_off: float
    epm_def: float
    epm_net: float

    impact_off_per_100: float
    impact_def_per_100: float
    impact_per_100: float

    off_rating_per_100: float
    def_rating_per_100: float
    net_rating_per_100: float

    pts_per_100_poss: float
    ts_pct: float
    efg_pct: float
    usg_pct: float
    three_pct: float
    ft_pct: float
    tov_ratio: float


def _weighted_mean(
    df: pd.DataFrame, value_col: str, weight_col: str = "minutes_played"
) -> float:
    """
    Compute a minutes-weighted mean of value_col.
    Returns NaN if total weight <= 0 or value_col missing.
    """
    if value_col not in df.columns or weight_col not in df.columns:
        return float("nan")

    w = pd.to_numeric(df[weight_col], errors="coerce")
    v = pd.to_numeric(df[value_col], errors="coerce")

    valid = w.notna() & v.notna()
    w = w[valid]
    v = v[valid]

    total_w = w.sum()
    if total_w <= 0:
        return float("nan")

    return float((v * w).sum() / total_w)


def _aggregate_team_from_profiles(
    profiles_df: pd.DataFrame, season_label: str, team: str
) -> TeamAggregateFromProfiles:
    """
    Aggregate gameday player profiles to a single team row for the given season/team.
    """
    team = team.upper()
    df_team = profiles_df.loc[profiles_df["team"] == team].copy()

    if df_team.empty:
        raise ValueError(f"No player rows found for team={team} in profiles_df.")

    # Coerce minutes to numeric and handle weirdness
    df_team["minutes_played"] = pd.to_numeric(
        df_team["minutes_played"], errors="coerce"
    ).fillna(0.0)

    num_players = int(df_team["player_id"].nunique())
    total_minutes = float(df_team["minutes_played"].sum())

    def wm(col: str) -> float:
        return _weighted_mean(df_team, col, "minutes_played")

    return TeamAggregateFromProfiles(
        team=team,
        season=season_label,
        num_players=num_players,
        total_minutes=total_minutes,
        epm_off=wm("epm_off"),
        epm_def=wm("epm_def"),
        epm_net=wm("epm_net"),
        impact_off_per_100=wm("impact_off_per_100"),
        impact_def_per_100=wm("impact_def_per_100"),
        impact_per_100=wm("impact_per_100"),
        off_rating_per_100=wm("off_rating_per_100"),
        def_rating_per_100=wm("def_rating_per_100"),
        net_rating_per_100=wm("net_rating_per_100"),
        pts_per_100_poss=wm("pts_per_100_poss"),
        ts_pct=wm("ts_pct"),
        efg_pct=wm("efg_pct"),
        usg_pct=wm("usg_pct"),
        three_pct=wm("three_pct"),
        ft_pct=wm("ft_pct"),
        tov_ratio=wm("tov_ratio"),
    )


def _prefix_dict(
    d: Dict[str, Any],
    prefix: str,
    strip_keys: Optional[List[str]] = None,
    strip_team_prefix: bool = True,
) -> Dict[str, Any]:
    """
    Prefix keys in a dict with `prefix`.

    - If strip_keys is provided, those keys are omitted entirely.
    - If strip_team_prefix is True, any key starting with 'team_' has that prefix removed
      before applying the new prefix. Example:
        {'team_games_played': 25} + prefix='home_' -> {'home_games_played': 25}
    """
    strip_keys = strip_keys or []
    out: Dict[str, Any] = {}

    for key, value in d.items():
        if key in strip_keys:
            continue

        base_key = key
        if strip_team_prefix and key.startswith("team_"):
            base_key = key[len("team_") :]

        out[f"{prefix}{base_key}"] = value

    return out


def build_matchup_features_from_profiles(
    profiles_df: pd.DataFrame,
    season_label: str,
    home_team: str,
    away_team: str,
    game_id: Optional[str] = None,
    date_cutoff: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a single-row matchup feature DataFrame for (away_team at home_team)
    using:

      - Gameday player profiles for both teams (minutes-weighted aggregates)
      - Team schedule features from LeagueGameLog for both teams

    Parameters
    ----------
    profiles_df : DataFrame
        Gameday player profiles (already joined with impact + EPM).
    season_label : str
        Season label, e.g. '2025-26'.
    home_team : str
        Home team abbreviation, e.g. 'NYK'.
    away_team : str
        Away team abbreviation, e.g. 'SAS'.
    game_id : str, optional
        Optional identifier for the matchup row. If None, a synthetic ID like
        '{season_label}_{away}_at_{home}' is used.
    date_cutoff : str, optional
        Optional cutoff date ('YYYY-MM-DD' or 'MM/DD/YYYY') for schedule features.
        If None, uses all LeagueGameLog games returned (i.e., up-to-date snapshot).

    Returns
    -------
    DataFrame
        A 1-row DataFrame of matchup features.
    """
    home_team = home_team.upper()
    away_team = away_team.upper()

    if game_id is None:
        game_id = f"{season_label}_{away_team}_at_{home_team}"

    # --- 1) Aggregate player profiles to team level (what we already had) ---
    home_agg = _aggregate_team_from_profiles(
        profiles_df=profiles_df, season_label=season_label, team=home_team
    )
    away_agg = _aggregate_team_from_profiles(
        profiles_df=profiles_df, season_label=season_label, team=away_team
    )

    home_agg_dict = asdict(home_agg)
    away_agg_dict = asdict(away_agg)

    # Prefix player-aggregate features (home_/away_)
    home_profile_feats = _prefix_dict(
        home_agg_dict,
        prefix="home_",
        strip_keys=["team", "season"],
        strip_team_prefix=False,
    )
    away_profile_feats = _prefix_dict(
        away_agg_dict,
        prefix="away_",
        strip_keys=["team", "season"],
        strip_team_prefix=False,
    )

    # --- 2) Add team schedule features (LeagueGameLog-based) ---
    # These are whole-team / recent-form metrics (wins, plus/minus, etc.)
    home_sched_df = build_team_schedule_features_df(
        season_label=season_label,
        team=home_team,
        date_cutoff=date_cutoff,
    )
    away_sched_df = build_team_schedule_features_df(
        season_label=season_label,
        team=away_team,
        date_cutoff=date_cutoff,
    )

    home_sched_dict_raw = home_sched_df.iloc[0].to_dict()
    away_sched_dict_raw = away_sched_df.iloc[0].to_dict()

    # Prefix schedule features as home_* / away_* and strip 'team_' prefix.
    home_sched_feats = _prefix_dict(
        home_sched_dict_raw,
        prefix="home_",
        strip_keys=["team", "season"],
        strip_team_prefix=True,
    )
    away_sched_feats = _prefix_dict(
        away_sched_dict_raw,
        prefix="away_",
        strip_keys=["team", "season"],
        strip_team_prefix=True,
    )

    # --- 3) Build diff features for player aggregates (same as before) ---
    def _get_safe(d: Dict[str, Any], key: str) -> float:
        v = d.get(key, np.nan)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    diff_feats: Dict[str, float] = {}

    profile_diff_pairs = [
        ("epm_off", "diff_epm_off"),
        ("epm_def", "diff_epm_def"),
        ("epm_net", "diff_epm_net"),
        ("impact_off_per_100", "diff_impact_off_per_100"),
        ("impact_def_per_100", "diff_impact_def_per_100"),
        ("impact_per_100", "diff_impact_per_100"),
        ("off_rating_per_100", "diff_off_rating_per_100"),
        ("def_rating_per_100", "diff_def_rating_per_100"),
        ("net_rating_per_100", "diff_net_rating_per_100"),
        ("pts_per_100_poss", "diff_pts_per_100_poss"),
        ("ts_pct", "diff_ts_pct"),
        ("efg_pct", "diff_efg_pct"),
        ("usg_pct", "diff_usg_pct"),
        ("three_pct", "diff_three_pct"),
        ("ft_pct", "diff_ft_pct"),
        ("tov_ratio", "diff_tov_ratio"),
    ]

    for base_key, diff_key in profile_diff_pairs:
        h_key = f"home_{base_key}"
        a_key = f"away_{base_key}"
        h_val = _get_safe(home_profile_feats, h_key)
        a_val = _get_safe(away_profile_feats, a_key)
        diff_feats[diff_key] = h_val - a_val

    # --- 4) Build diff features for schedule stats (home − away) ---
    # Focus on a small, high-signal subset.
    sched_diff_keys = [
        "games_played",
        "wins",
        "losses",
        "win_pct",
        "pts_for_per_game",
        "plus_minus_per_game",
        "recent_wins",
        "recent_losses",
        "recent_plus_minus_per_game",
    ]

    for base_key in sched_diff_keys:
        h_key = f"home_{base_key}"
        a_key = f"away_{base_key}"
        h_val = _get_safe(home_sched_feats, h_key)
        a_val = _get_safe(away_sched_feats, a_key)
        diff_feats[f"diff_team_{base_key}"] = h_val - a_val

    # --- 5) Assemble final row ---
    row: Dict[str, Any] = {
        "game_id": game_id,
        "season": season_label,
        "home_team": home_team,
        "away_team": away_team,
    }

    # core counts from profiles
    row["home_num_players"] = home_agg.num_players
    row["away_num_players"] = away_agg.num_players
    row["home_total_minutes"] = home_agg.total_minutes
    row["away_total_minutes"] = away_agg.total_minutes

    # merge everything
    row.update(home_profile_feats)
    row.update(away_profile_feats)
    row.update(home_sched_feats)
    row.update(away_sched_feats)
    row.update(diff_feats)

    return pd.DataFrame([row])
