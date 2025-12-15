# src/features/impact_dataset.py
from __future__ import annotations

from typing import Optional, List

import pandas as pd
import numpy as np

from data.nba_api_provider import (
    get_team_regular_season_games,
    get_game_play_by_play,
    get_game_rotation,
)
from features.pbp_normalizer import normalize_pbp_v3
from features.possession_builder import build_all_possessions_for_game
from features.lineup_builder import build_player_stints_from_rotation
from features.lineup_stints import build_lineup_stints_for_game
from features.possession_lineups import attach_lineups_to_possessions
from features.season_player_aggregate import (
    _infer_home_away_from_matchup,
    _fetch_with_retries,
)


def build_possessions_with_lineups_for_game(
    game_id: str,
    team_abbrev: str,
    matchup: str,
) -> pd.DataFrame:
    """
    Build a possession-level dataset for a single game with on-court lineups
    attached to each possession.

    This wraps the same pipeline used in compute_player_stats_for_team_season:

      - PBPv3 -> normalize -> all possessions
      - GameRotation -> player stints -> lineup stints
      - Attach lineups to possessions

    Returns one row per possession with columns like:
      - game_id, period, possession_index, offense_team, defense_team
      - points (offense points on that possession)
      - offense_player_ids, defense_player_ids
      - home_team, away_team, lineup_stint_index, etc.
    """
    team_abbrev = team_abbrev.upper()

    # Figure out which side is home/away for this matchup
    home_team, away_team = _infer_home_away_from_matchup(team_abbrev, matchup)

    # --- Fetch & normalize play-by-play ---
    pbp_raw = _fetch_with_retries(get_game_play_by_play, game_id)
    if pbp_raw.empty:
        raise RuntimeError(f"Empty PBP for game {game_id}")

    pbp_norm = normalize_pbp_v3(pbp_raw)

    # --- Build all possessions (including non-scoring) ---
    # NOTE: build_all_possessions_for_game derives game_id from the events df
    possessions_df = build_all_possessions_for_game(
        pbp_norm,
        home_team=home_team,
        away_team=away_team,
    )

    if possessions_df.empty:
        raise RuntimeError(f"No possessions built for game {game_id}")

    # --- Fetch rotation -> stints -> lineup stints ---
    rotation_df = _fetch_with_retries(get_game_rotation, game_id)
    if rotation_df.empty:
        raise RuntimeError(f"Empty GameRotation for game {game_id}")

    stints_df = build_player_stints_from_rotation(rotation_df)
    if stints_df.empty:
        raise RuntimeError(f"No player stints built for game {game_id}")

    lineup_stints_df = build_lineup_stints_for_game(stints_df)
    if lineup_stints_df.empty:
        raise RuntimeError(
            f"No lineup stints built for game {game_id}; cannot attach lineups."
        )

    # --- Attach lineups to possessions ---
    enriched = attach_lineups_to_possessions(
        possessions_df,
        lineup_stints_df,
    )

    if enriched.empty:
        raise RuntimeError(
            f"attach_lineups_to_possessions produced empty result for game {game_id}"
        )

    return enriched


def build_possession_impact_for_team_season(
    team_abbrev: str,
    max_games: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a possession-level impact dataset for all eligible regular-season games
    for a single team.

    - Uses the same schedule + game filtering rules as compute_player_stats_for_team_season:
        * Only regular-season games
        * Exclude games on/after today (PBP/rotation may be incomplete)
    - For each game:
        * Build possessions with lineups attached.
    - Concatenates all games into a single DataFrame.

    This is the raw "impact dataset" we'll later transform into:
      * stint-level rows
      * RAPM design matrix (X with +1/-1 player indicators, y as margin per 100)

    Args:
        team_abbrev:   e.g. "ATL"
        max_games:     Optional limit for dev/testing (None = all eligible games).

    Returns:
        A DataFrame with one row per possession across all processed games for this team.
    """
    team_abbrev = team_abbrev.upper()

    # Fetch full regular-season schedule for this team
    games_df = _fetch_with_retries(get_team_regular_season_games, team_abbrev)
    if games_df.empty:
        raise RuntimeError(f"No regular season games found for team {team_abbrev}.")

    # Mirror the "exclude today's games" rule from compute_player_stats_for_team_season
    from datetime import date

    today_str = date.today().strftime("%Y-%m-%d")
    games_df = games_df[games_df["GameDate"] < today_str]

    if games_df.empty:
        print(
            f"⚠️ No eligible regular season games for {team_abbrev} before {today_str}."
        )
        return pd.DataFrame()

    games_df = games_df.sort_values("GameDate")
    if max_games is not None:
        games_df = games_df.head(max_games)

    all_possessions: List[pd.DataFrame] = []

    for _, row in games_df.iterrows():
        game_id = str(row["GameID"])
        game_date = str(row["GameDate"])
        matchup = row["MATCHUP"]
        team_for_matchup = row["Team"] if "Team" in row else team_abbrev

        print(
            f"\n🎬 [Impact] Processing GameID {game_id} "
            f"({matchup}) on {game_date} for team {team_abbrev}..."
        )

        try:
            enriched = build_possessions_with_lineups_for_game(
                game_id=game_id,
                team_abbrev=team_for_matchup,
                matchup=matchup,
            )
        except RuntimeError as e:
            # For the model dataset we generally do NOT want partial/incomplete games.
            # So bubble up the error instead of silently skipping.
            raise RuntimeError(
                f"❌ Failed to build possession impact data for game {game_id}: {e}"
            ) from e

        all_possessions.append(enriched)

    if not all_possessions:
        print(
            f"⚠️ No possessions with lineups built for team {team_abbrev}; "
            "impact dataset is empty."
        )
        return pd.DataFrame()

    impact_df = pd.concat(all_possessions, ignore_index=True)

    # Make sure game_id is string and team codes are uppercased for consistency
    if "game_id" in impact_df.columns:
        impact_df["game_id"] = impact_df["game_id"].astype(str)

    for col in ["offense_team", "defense_team", "home_team", "away_team"]:
        if col in impact_df.columns:
            impact_df[col] = impact_df[col].astype(str).str.upper()

    return impact_df


def build_lineup_stint_impact_from_possessions(
    impact_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate possession-level impact data into lineup-stint level rows.

    Each row represents a *single stint* of two lineups facing each other:
      - home_team with home_player_ids
      - away_team with away_player_ids

    For each (game_id, lineup_stint_index, home/away lineups) group, we compute:
      - possessions in stint
      - points_home_sum, points_away_sum
      - net_points_home
      - offensive/defensive/net rating for the home team per 100 possessions

    This is a natural unit for RAPM-style modeling: we know which 10 players
    were on the floor together and the scoring margin over that stint.
    """
    if impact_df.empty:
        return pd.DataFrame()

    # Work on a copy so we can safely mutate types
    df = impact_df.copy()

    required_cols = [
        "game_id",
        "lineup_stint_index",
        "home_team",
        "away_team",
        "home_player_ids",
        "away_player_ids",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Impact DataFrame is missing required columns for stint aggregation: {missing}"
        )

    # --- Ensure lineup columns are hashable (tuples) for groupby ---
    def _to_tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        if pd.isna(x):
            return tuple()
        # Fallback: single value -> 1-length tuple
        return (x,)

    df["home_player_ids"] = df["home_player_ids"].apply(_to_tuple)
    df["away_player_ids"] = df["away_player_ids"].apply(_to_tuple)

    # --- Determine which columns contain the per-possession scoring info ---
    # We support a few patterns:
    #   1) Explicit home/away scoring columns.
    #   2) Generic "points" tied to offense_team vs home/away.
    if {"points_home", "points_away"}.issubset(df.columns):
        points_home_col = "points_home"
        points_away_col = "points_away"

    elif {"points_for", "points_against"}.issubset(df.columns):
        # Fallback naming if you ever used points_for/points_against instead
        points_home_col = "points_for"
        points_away_col = "points_against"

    elif {"points", "offense_team", "home_team", "away_team"}.issubset(df.columns):
        # Construct home/away scoring from a generic "points" column:
        #   - if offense_team == home_team -> points_home = points, points_away = 0
        #   - if offense_team == away_team -> points_home = 0, points_away = points
        df["points_home"] = np.where(
            df["offense_team"] == df["home_team"], df["points"], 0.0
        )
        df["points_away"] = np.where(
            df["offense_team"] == df["away_team"], df["points"], 0.0
        )
        points_home_col = "points_home"
        points_away_col = "points_away"

    else:
        # Last-resort debug: show what columns we actually have
        raise ValueError(
            "Could not find recognizable scoring columns in impact_df. "
            f"Available columns: {sorted(df.columns)}"
        )

    group_cols = [
        "game_id",
        "lineup_stint_index",
        "home_team",
        "away_team",
        "home_player_ids",
        "away_player_ids",
    ]

    grouped = (
        df.groupby(group_cols)
        .agg(
            possessions=("game_id", "size"),
            points_home_sum=(points_home_col, "sum"),
            points_away_sum=(points_away_col, "sum"),
        )
        .reset_index()
    )

    # If stint_duration_seconds is present (from lineup_stints), preserve it.
    if "stint_duration_seconds" in df.columns:
        # Take the first value per group (it should be constant across the group).
        dur = (
            df.groupby(group_cols)["stint_duration_seconds"]
            .first()
            .reset_index(drop=True)
        )
        grouped["stint_duration_seconds"] = dur

    # Net scoring margin for home team in this stint
    grouped["net_points_home"] = grouped["points_home_sum"] - grouped["points_away_sum"]

    # Ratings per 100 possessions (home perspective)
    poss = grouped["possessions"].clip(lower=1)
    grouped["off_rating_home_per_100"] = 100.0 * grouped["points_home_sum"] / poss
    grouped["def_rating_home_per_100"] = 100.0 * grouped["points_away_sum"] / poss
    grouped["net_rating_home_per_100"] = (
        grouped["off_rating_home_per_100"] - grouped["def_rating_home_per_100"]
    )

    return grouped


def build_lineup_stint_impact_for_team_season(
    team_abbrev: str,
    max_games: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper:

      1) Build the possession-level impact dataset for a team/season.
      2) Aggregate to lineup-stint level using build_lineup_stint_impact_from_possessions().

    This yields one row per (game_id, lineup_stint_index, home/away lineups),
    with possessions/points and net ratings for the home team.

    Args:
        team_abbrev: e.g. "ATL"
        max_games:   Optional limit (None = all eligible games).

    Returns:
        DataFrame with one row per lineup stint (both lineups) across all games.
    """
    team_abbrev = team_abbrev.upper()
    impact_df = build_possession_impact_for_team_season(
        team_abbrev=team_abbrev,
        max_games=max_games,
    )
    if impact_df.empty:
        return pd.DataFrame()

    stint_df = build_lineup_stint_impact_from_possessions(impact_df)
    if stint_df.empty:
        print(
            f"⚠️ No lineup stints could be aggregated for team {team_abbrev}; "
            "stint-level impact dataset is empty."
        )
    return stint_df
