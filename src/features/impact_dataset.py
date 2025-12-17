# src/features/impact_dataset.py
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
from db.player_stats_db import (
    get_connection,
    ensure_impact_lineup_stints_table,
    upsert_impact_lineup_stints_for_team_season,
    load_impact_lineup_stints_for_team_season,
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
    # Rotation fetch (patched)
    try:
        rotation_df = _fetch_with_retries(get_game_rotation, game_id)
    except Exception as exc:
        print(
            f"⚠️ [Impact] Failed to load GameRotation for game {game_id}: {exc}. "
            f"Skipping this game for possession/impact modeling."
        )
        return pd.DataFrame()
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
    season_label: Optional[str] = None,
    max_games: Optional[int] = None,
    db_path: str = "data/player_stats.db",
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Build (or incrementally extend) the lineup-stint impact dataset for a team/season,
    using the DB-backed impact_lineup_stints cache.

    High-level flow:

      1) Pull the team's regular-season schedule from stats.nba.com.
      2) Filter to games strictly before today (to avoid incomplete PBP/rotations).
      3) Open player_stats.db and ensure impact_lineup_stints exists.
      4) Load any existing stint rows for (team, season).
      5) For any *missing* games (or all games if force_recompute=True):
           - Build possessions+lineups for that game.
           - Aggregate to lineup stints via build_lineup_stint_impact_from_possessions().
           - Upsert those stints into impact_lineup_stints.
      6) Reload all stint rows for (team, season) from the DB and return as a DataFrame.

    This keeps the heavy PBP/rotation work incremental: once a game's stints have
    been built and cached, subsequent runs only process *new* games.

    Args:
        team_abbrev:
            Team abbreviation, e.g. "NYK".
        season_label:
            Season label, e.g. "2025-26". If None, we infer from the schedule's
            SEASON_ID, but all callers in the pipeline should pass this explicitly.
        max_games:
            Optional dev/testing limit. If > 0, we only consider the earliest
            max_games eligible games in the schedule.
        db_path:
            Path to the SQLite DB (default "data/player_stats.db").
        force_recompute:
            If True, we recompute and upsert all games in the schedule window,
            even if stints already exist in the DB for those games.

    Returns:
        DataFrame with one row per lineup stint across all cached games
        for (team, season).
    """
    from datetime import date

    team_abbrev = team_abbrev.upper()

    # -----------------------------
    # 1) Fetch regular-season schedule
    # -----------------------------
    games_df = _fetch_with_retries(get_team_regular_season_games, team_abbrev)
    if games_df.empty:
        raise RuntimeError(f"No regular season games found for team {team_abbrev}.")

    # Ensure we have the columns we expect
    required_cols = {"GameID", "GameDate", "MATCHUP"}
    missing = required_cols - set(games_df.columns)
    if missing:
        raise RuntimeError(
            f"Team schedule for {team_abbrev} is missing required columns: {sorted(missing)}"
        )

    # -----------------------------
    # 2) Filter to games strictly before today
    # -----------------------------
    today_str = date.today().strftime("%Y-%m-%d")
    games_df = games_df[games_df["GameDate"] < today_str].copy()

    if games_df.empty:
        print(
            f"⚠️ No eligible regular season games for {team_abbrev} before {today_str}."
        )
        return pd.DataFrame()

    games_df = games_df.sort_values("GameDate")

    # Support the "-1 means all games" pattern used by CLI
    if max_games is not None and max_games > 0:
        games_df = games_df.head(max_games)

    # -----------------------------
    # 3) Determine season label
    # -----------------------------
    if season_label is None:
        # Infer from SEASON_ID if available, else fall back to current season_id.
        # SEASON_ID is typically like 22025 for 2025-26.
        if "SEASON_ID" in games_df.columns and not games_df["SEASON_ID"].isna().all():
            # Take the first non-null season id and convert to something like "2025-26"
            season_id = str(games_df["SEASON_ID"].dropna().iloc[0])
            # "22025" -> 2025-26 (simple mapping: last 4 digits are start year)
            if len(season_id) >= 5:
                start_year = int(season_id[-4:])
                season_label = f"{start_year}-{str(start_year + 1)[-2:]}"
            else:
                season_label = "unknown"
        else:
            season_label = "unknown"

    # -----------------------------
    # 4) Open DB and ensure table / load existing stints
    # -----------------------------
    # -----------------------------
    # 4) Open DB and ensure table / load existing stints
    # -----------------------------
    conn = get_connection()  # uses the default DB path from player_stats_db
    ensure_impact_lineup_stints_table(conn)

    existing_stints = load_impact_lineup_stints_for_team_season(
        conn=conn,
        team=team_abbrev,
        season=season_label,
    )

    existing_game_ids: set[str] = set()
    if not existing_stints.empty and "game_id" in existing_stints.columns:
        existing_game_ids = set(existing_stints["game_id"].astype(str).unique())

    # -----------------------------
    # 5) Figure out which games need (re)computation
    # -----------------------------
    games_to_process: list[tuple[str, str, str, str]] = []
    # (game_id, game_date, matchup, team_for_matchup)

    for _, row in games_df.iterrows():
        game_id = str(row["GameID"])
        game_date = str(row["GameDate"])
        matchup = str(row["MATCHUP"])
        team_for_matchup = row["Team"] if "Team" in row else team_abbrev

        if not force_recompute and game_id in existing_game_ids:
            # Already cached in DB; skip heavy work.
            continue

        games_to_process.append((game_id, game_date, matchup, team_for_matchup))

    if not games_to_process and not existing_stints.empty:
        # Nothing new to compute; just return what we already have.
        print(
            f"ℹ️ [Impact] All eligible games for {team_abbrev} / {season_label} "
            "are already cached in impact_lineup_stints."
        )
        conn.close()
        # Normalize types a bit for downstream
        if "game_id" in existing_stints.columns:
            existing_stints["game_id"] = existing_stints["game_id"].astype(str)
        for col in ["home_team", "away_team"]:
            if col in existing_stints.columns:
                existing_stints[col] = existing_stints[col].astype(str).str.upper()
        return existing_stints

    # -----------------------------
    # 6) Build stints for missing games and upsert into DB
    # -----------------------------
    for game_id, game_date, matchup, team_for_matchup in games_to_process:
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
            if enriched.empty:
                print(
                    f"⚠️ [Impact] Possession+lineup dataset empty for game {game_id}; skipping."
                )
                continue

            game_stints = build_lineup_stint_impact_from_possessions(enriched)
            if game_stints.empty:
                print(
                    f"⚠️ [Impact] No lineup stints aggregated for game {game_id}; skipping."
                )
                continue

        except Exception as exc:
            # We *skip* problematic games rather than failing the entire pipeline.
            print(
                f"⚠️ [Impact] Skipping game {game_id} for {team_abbrev} due to error: {exc}"
            )
            continue

        # Persist to DB (upsert by (team, season, game_id, lineup_stint_index))
        upserted = upsert_impact_lineup_stints_for_team_season(
            conn=conn,
            team=team_abbrev,
            season=season_label,
            stints=game_stints,
        )
        print(
            f"💾 [Impact] Upserted {upserted} lineup stints for game {game_id} "
            f"into impact_lineup_stints."
        )

    # -----------------------------
    # 7) Reload all stints for (team, season) from DB and normalize
    # -----------------------------
    all_stints = load_impact_lineup_stints_for_team_season(
        conn=conn,
        team=team_abbrev,
        season=season_label,
    )
    conn.close()

    if all_stints.empty:
        print(
            f"⚠️ [Impact] No lineup stints stored in DB for {team_abbrev} / {season_label}."
        )
        return all_stints

    # Normalize a few key columns for downstream consumers (ridge, exports, etc.)
    if "game_id" in all_stints.columns:
        all_stints["game_id"] = all_stints["game_id"].astype(str)
    for col in ["home_team", "away_team"]:
        if col in all_stints.columns:
            all_stints[col] = all_stints[col].astype(str).str.upper()

    return all_stints
