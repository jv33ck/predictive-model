# src/features/season_player_aggregate.py
from __future__ import annotations

from datetime import date
from typing import Optional, List, Iterable, Set

import time

import pandas as pd
from requests.exceptions import ReadTimeout, RequestException

from data.nba_api_provider import (
    get_team_regular_season_games,
    get_game_play_by_play,
    get_game_rotation,
)
from db.player_stats_db import (
    init_db,
    insert_player_game_stats,
    mark_game_processed,
    is_game_processed,
)
from features.pbp_normalizer import normalize_pbp_v3
from features.possession_builder import build_all_possessions_for_game
from features.lineup_builder import build_player_stints_from_rotation
from features.lineup_stints import build_lineup_stints_for_game
from features.possession_lineups import attach_lineups_to_possessions
from features.player_possessions import aggregate_player_possession_stats


def _infer_home_away_from_matchup(team_abbrev: str, matchup: str) -> tuple[str, str]:
    parts = matchup.split()
    if len(parts) != 3:
        raise ValueError(f"Unexpected MATCHUP format: {matchup!r}")

    team_code, sep, opp_code = parts
    team_code = team_code.upper()
    opp_code = opp_code.upper()

    if sep == "vs.":
        home_team = team_code
        away_team = opp_code
    elif sep == "@":
        home_team = opp_code
        away_team = team_code
    else:
        raise ValueError(f"Unexpected separator in MATCHUP: {sep!r}")

    if team_abbrev.upper() not in {home_team, away_team}:
        raise ValueError(
            f"Team {team_abbrev} not in MATCHUP {matchup} -> ({home_team}, {away_team})"
        )

    return home_team, away_team


def _fetch_with_retries(
    fetch_fn, *args, max_retries: int = 3, delay_seconds: float = 2.0, **kwargs
):
    """
    Call an NBA API fetch function with simple retry logic.

    - Retries on ReadTimeout / RequestException up to max_retries times.
    - Waits delay_seconds between attempts.
    - Raises the last error if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_fn(*args, **kwargs)
        except (ReadTimeout, RequestException) as e:
            last_exc = e
            print(f"⏳ Attempt {attempt} failed for {fetch_fn.__name__}: {e}")
            if attempt < max_retries:
                time.sleep(delay_seconds)
        except Exception:
            # For non-timeout errors, don't retry silently – raise immediately.
            raise

    # If we got here, all retries failed
    raise RuntimeError(
        f"Failed to fetch via {fetch_fn.__name__} after {max_retries} attempts: {last_exc}"
    )


def compute_player_stats_for_team_season(
    team_abbrev: str,
    max_games: Optional[int] = None,
    skip_game_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Compute aggregated per-player possession stats for a team's regular season.

    Pipeline per game:
      - PBPv3 -> normalize -> all possessions
      - GameRotation -> player stints -> lineup stints
      - Attach lineups to possessions
      - Aggregate per-player possessions & points
      - Merge with traditional & advanced boxscores
      - Insert per-game stats into player_stats.db

    Then aggregate across all games processed in THIS RUN.

    IMPORTANT:
      - Games on or after today are excluded (PBP/rotation/boxscore may be incomplete).
      - Games already marked as processed in the DB (is_game_processed) are skipped.
      - Games whose IDs are in skip_game_ids are explicitly skipped.
      - If no NEW games are processed (all games already in DB or skipped), this returns an
        empty DataFrame and does NOT raise.
      - If any unprocessed, non-skipped game is missing critical data (PBP, rotation,
        boxscores, stints, etc.), this raises RuntimeError to avoid partial/incomplete stats.

    Args:
        team_abbrev:
            Team abbreviation, e.g. 'ATL'.
        max_games:
            If provided, limit to the first N regular season games (useful
            for testing so we don't hammer the API).
        skip_game_ids:
            Optional iterable of game IDs (strings) to ignore entirely for this run.

    Returns:
        DataFrame with one row per (team, player_id) for the games processed
        in THIS RUN. For daily DB updates, the caller typically ignores this
        return value and relies on the DB contents instead.
    """
    team_abbrev = team_abbrev.upper()

    # Ensure DB tables exist
    init_db()

    # Fetch full regular-season schedule for this team
    games_df = _fetch_with_retries(get_team_regular_season_games, team_abbrev)
    if games_df.empty:
        raise RuntimeError(f"No regular season games found for team {team_abbrev}.")

    # 🔒 Exclude today's games (PBP/rotation/boxscore may not be ready yet)
    today_str = date.today().strftime("%Y-%m-%d")
    games_df = games_df[games_df["GameDate"] < today_str]

    if games_df.empty:
        print(
            f"⚠️ No eligible regular season games for {team_abbrev} before {today_str}."
        )
        return pd.DataFrame()

    # Optionally limit number of games (for dev/testing)
    games_df = games_df.sort_values("GameDate")
    if max_games is not None:
        games_df = games_df.head(max_games)

    # Normalize skip list
    skip_set: Set[str] = (
        {str(g).strip() for g in skip_game_ids} if skip_game_ids else set()
    )

    all_player_stats: List[pd.DataFrame] = []
    all_player_names: List[pd.DataFrame] = []
    team_game_summaries: List[dict] = []
    all_minutes: List[pd.DataFrame] = []
    all_box_trad: List[pd.DataFrame] = []
    all_box_adv: List[pd.DataFrame] = []

    processed_any_game = False  # track whether we actually processed any NEW game

    for _, game_row in games_df.iterrows():
        game_id = str(game_row["GameID"])
        game_date = str(game_row["GameDate"])
        matchup = game_row["MATCHUP"]
        team_for_matchup = game_row["Team"] if "Team" in game_row else team_abbrev

        # Explicitly skip any game in skip_game_ids
        if game_id in skip_set:
            print(
                f"⏭️ Skipping game {game_id} on {game_date} because it is in skip_game_ids."
            )
            continue

        # Skip games already in the DB
        if is_game_processed(game_id):
            print(f"✅ Game {game_id} on {game_date} already processed; skipping.")
            continue

        print(f"\n🎬 Processing GameID {game_id} ({matchup}) for team {team_abbrev}...")

        processed_any_game = True

        try:
            home_team, away_team = _infer_home_away_from_matchup(
                team_for_matchup, matchup
            )
        except ValueError as e:
            # This is a structural issue with schedule data → hard fail
            raise RuntimeError(
                f"❌ Failed to interpret matchup for game {game_id}: {e}"
            )

        # 1) PBP -> normalized -> all possessions
        try:
            raw_pbp = _fetch_with_retries(get_game_play_by_play, game_id)
        except Exception as e:
            # Hard fail: we don't want partial season stats
            raise RuntimeError(f"❌ Failed to fetch PBP for game {game_id}: {e}") from e

        if raw_pbp.empty:
            raise RuntimeError(
                f"❌ PBP for game {game_id} is empty; aborting to avoid incomplete stats."
            )

        normalized = normalize_pbp_v3(raw_pbp)

        all_possessions = build_all_possessions_for_game(
            events_df=normalized,
            home_team=home_team,
            away_team=away_team,
        )
        if all_possessions.empty:
            raise RuntimeError(
                f"❌ No possessions built for game {game_id}; season would be incomplete."
            )

        # 2) Rotation -> stints -> lineup stints
        try:
            rotation_df = _fetch_with_retries(get_game_rotation, game_id)
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to fetch rotation for game {game_id}: {e}"
            ) from e

        if rotation_df.empty:
            raise RuntimeError(
                f"❌ Rotation data for game {game_id} is empty; aborting to avoid incomplete stats."
            )

        stints_df = build_player_stints_from_rotation(rotation_df)
        if stints_df.empty:
            raise RuntimeError(
                f"❌ No player stints built for game {game_id}; season would be incomplete."
            )

        lineup_stints_df = build_lineup_stints_for_game(stints_df)
        if lineup_stints_df.empty:
            raise RuntimeError(
                f"❌ No lineup stints built for game {game_id}; season would be incomplete."
            )

        # 2b) Compute minutes played for this team in this game
        # stints_df.stint_duration_seconds is in tenths of a second (e.g. 28800 = 48*60*10)
        stints_team = stints_df[stints_df["team_abbrev"] == team_abbrev]

        if not stints_team.empty:
            minutes_df = (
                stints_team.groupby(["team_abbrev", "player_id", "player_name"])
                .agg(
                    minutes_played=(
                        "stint_duration_seconds",
                        # convert tenths of seconds -> real minutes
                        lambda s: s.sum() / 10.0 / 60.0,
                    )
                )
                .reset_index()
                .rename(columns={"team_abbrev": "team"})
            )
            minutes_df["game_id"] = game_id
            all_minutes.append(minutes_df)
        else:
            # No stints for this team in this game; create an empty minutes_df
            minutes_df = pd.DataFrame(
                columns=[
                    "team",
                    "player_id",
                    "player_name",
                    "minutes_played",
                    "game_id",
                ]
            )

        # 3) Attach lineups to possessions
        enriched_possessions = attach_lineups_to_possessions(
            possessions=all_possessions,
            lineup_stints=lineup_stints_df,
        )

        # 3b) Compute team-level totals for this game (for on/off later)
        team_mask_off = enriched_possessions["offense_team"].str.upper() == team_abbrev
        team_mask_def = enriched_possessions["defense_team"].str.upper() == team_abbrev

        team_off_possessions = float(team_mask_off.sum())
        team_def_possessions = float(team_mask_def.sum())
        team_off_points_for = float(
            enriched_possessions.loc[team_mask_off, "points"].sum()
        )
        team_def_points_against = float(
            enriched_possessions.loc[team_mask_def, "points"].sum()
        )

        team_game_summaries.append(
            {
                "team": team_abbrev,
                "game_id": game_id,
                "team_off_possessions": team_off_possessions,
                "team_def_possessions": team_def_possessions,
                "team_total_possessions": team_off_possessions + team_def_possessions,
                "team_off_points_for": team_off_points_for,
                "team_def_points_against": team_def_points_against,
                "team_net_points": team_off_points_for - team_def_points_against,
            }
        )

        # 3c) Fetch boxscores for this game (traditional + advanced) for this team
        from data.nba_api_provider import (
            get_game_boxscore_traditional,
            get_game_boxscore_advanced,
        )

        trad_bs = _fetch_with_retries(get_game_boxscore_traditional, game_id)
        if trad_bs.empty:
            raise RuntimeError(
                f"❌ Traditional boxscore is empty for game {game_id}; aborting to avoid incomplete stats."
            )
        trad_team = trad_bs[trad_bs["teamTricode"] == team_abbrev].copy()
        if trad_team.empty:
            raise RuntimeError(
                f"❌ Traditional boxscore has no rows for team {team_abbrev} in game {game_id}."
            )
        trad_team = trad_team.rename(
            columns={
                "teamTricode": "team",
                "personId": "player_id",
            }
        )
        trad_team["game_id"] = game_id
        all_box_trad.append(trad_team)

        adv_bs = _fetch_with_retries(get_game_boxscore_advanced, game_id)
        if adv_bs.empty:
            raise RuntimeError(
                f"❌ Advanced boxscore is empty for game {game_id}; aborting to avoid incomplete stats."
            )
        adv_team = adv_bs[adv_bs["teamTricode"] == team_abbrev].copy()
        if adv_team.empty:
            raise RuntimeError(
                f"❌ Advanced boxscore has no rows for team {team_abbrev} in game {game_id}."
            )
        adv_team = adv_team.rename(
            columns={
                "teamTricode": "team",
                "personId": "player_id",
            }
        )
        adv_team["game_id"] = game_id
        all_box_adv.append(adv_team)

        # 4) Aggregate per-player for this game
        player_stats = aggregate_player_possession_stats(enriched_possessions)
        if player_stats.empty:
            raise RuntimeError(
                f"❌ No per-player stats produced for game {game_id}; "
                "season would be incomplete."
            )

        # Attach game_id for both in-memory aggregation and DB writing
        player_stats["game_id"] = game_id

        # --- Build per-game merged stats for DB ---
        trad_team_for_merge = trad_team.copy()
        adv_team_for_merge = adv_team.copy()

        trad_cols = [
            "game_id",
            "team",
            "player_id",
            "points",
            "fieldGoalsMade",
            "fieldGoalsAttempted",
            "threePointersMade",
            "threePointersAttempted",
            "freeThrowsMade",
            "freeThrowsAttempted",
            "reboundsOffensive",
            "reboundsDefensive",
            "reboundsTotal",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "foulsPersonal",
            "plusMinusPoints",
        ]
        trad_cols = [c for c in trad_cols if c in trad_team_for_merge.columns]
        trad_team_for_merge = trad_team_for_merge[trad_cols]

        adv_cols = [
            "game_id",
            "team",
            "player_id",
            "trueShootingPercentage",
            "effectiveFieldGoalPercentage",
            "usagePercentage",
            "estimatedUsagePercentage",
            "offensiveRating",
            "defensiveRating",
            "netRating",
            "offensiveReboundPercentage",
            "defensiveReboundPercentage",
            "reboundPercentage",
            "assistPercentage",
            "turnoverRatio",
            "pace",
            "possessions",
        ]
        adv_cols = [c for c in adv_cols if c in adv_team_for_merge.columns]
        adv_team_for_merge = adv_team_for_merge[adv_cols]

        # Start from player_stats (which includes possession-based ratings)
        per_game_full = player_stats.merge(
            trad_team_for_merge,
            on=["game_id", "team", "player_id"],
            how="left",
        ).merge(
            adv_team_for_merge,
            on=["game_id", "team", "player_id"],
            how="left",
        )

        # Rename boxscore columns to match DB schema expectations
        per_game_full = per_game_full.rename(
            columns={
                "points": "pts",
                "fieldGoalsMade": "fgm",
                "fieldGoalsAttempted": "fga",
                "threePointersMade": "fg3m",
                "threePointersAttempted": "fg3a",
                "freeThrowsMade": "ftm",
                "freeThrowsAttempted": "fta",
                "reboundsOffensive": "oreb",
                "reboundsDefensive": "dreb",
                "reboundsTotal": "treb",
                "assists": "ast",
                "steals": "stl",
                "blocks": "blk",
                "turnovers": "tov",
                "foulsPersonal": "pf",
                "plusMinusPoints": "plus_minus",
                "trueShootingPercentage": "ts_pct",
                "effectiveFieldGoalPercentage": "efg_pct",
                "usagePercentage": "usg_pct",
                "estimatedUsagePercentage": "est_usg_pct",
                "offensiveReboundPercentage": "oreb_pct",
                "defensiveReboundPercentage": "dreb_pct",
                "reboundPercentage": "reb_pct",
                "assistPercentage": "ast_pct",
                "turnoverRatio": "tov_ratio",
                "offensiveRating": "off_rating_box",
                "defensiveRating": "def_rating_box",
                "netRating": "net_rating_box",
                "possessions": "possessions_est",
            }
        )

        # Ensure player_name is present and non-null before DB insert
        if "player_name" not in per_game_full.columns:
            names_for_game = (
                stints_df[["player_id", "team_abbrev", "player_name"]]
                .drop_duplicates()
                .rename(columns={"team_abbrev": "team"})
            )
            per_game_full = per_game_full.merge(
                names_for_game,
                on=["team", "player_id"],
                how="left",
            )

        if "player_name" in per_game_full.columns:
            per_game_full["player_name"] = per_game_full["player_name"].fillna(
                "Unknown"
            )

        # Ensure minutes_played is present and non-null before DB insert
        if "minutes_played" not in per_game_full.columns:
            per_game_full = per_game_full.merge(
                minutes_df[["team", "player_id", "minutes_played"]],
                on=["team", "player_id"],
                how="left",
            )

        if "minutes_played" in per_game_full.columns:
            per_game_full["minutes_played"] = per_game_full["minutes_played"].fillna(
                0.0
            )

        # Insert into DB (per-game, per-player)
        insert_player_game_stats(per_game_full)

        # Mark the game as processed in the games table
        mark_game_processed(
            game_id=game_id,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team,
            season="current",
        )

        # Keep for in-memory season aggregation (this RUN only)
        all_player_stats.append(player_stats)

        # Also keep names for merge
        player_names = (
            stints_df[["player_id", "team_abbrev", "player_name"]]
            .drop_duplicates()
            .rename(columns={"team_abbrev": "team"})
        )
        all_player_names.append(player_names)

    # If we didn't process any NEW games, just return an empty DataFrame.
    # This is normal in the daily update flow when all games are already in the DB
    # or all remaining games were in skip_game_ids.
    if not processed_any_game:
        print(
            f"ℹ️ No new games to process for {team_abbrev}; "
            "all eligible games are already in the database or explicitly skipped."
        )
        return pd.DataFrame()

    if not all_player_stats:
        # This should not happen if processed_any_game is True; if it does, treat
        # as a hard error.
        raise RuntimeError(
            f"No player possession stats produced for team {team_abbrev} "
            "despite processing at least one game."
        )

    # Concatenate all games processed in THIS RUN
    all_stats_df = pd.concat(all_player_stats, ignore_index=True)
    all_names_df = (
        pd.concat(all_player_names, ignore_index=True)
        .drop_duplicates(subset=["player_id", "team"])
        .reset_index(drop=True)
    )

    # 🔧 Filter to the team we’re computing for (e.g. ATL) so opponents are excluded
    all_stats_df = all_stats_df[all_stats_df["team"] == team_abbrev]
    all_names_df = all_names_df[all_names_df["team"] == team_abbrev]

    # --- Aggregate traditional boxscore stats across games (this RUN only) ---
    if all_box_trad:
        box_trad_all = pd.concat(all_box_trad, ignore_index=True)
        box_trad_all = box_trad_all[box_trad_all["team"] == team_abbrev]

        box_trad_grouped = (
            box_trad_all.groupby(["team", "player_id"])
            .agg(
                games_with_box=("game_id", "nunique"),
                pts=("points", "sum"),
                fgm=("fieldGoalsMade", "sum"),
                fga=("fieldGoalsAttempted", "sum"),
                fg3m=("threePointersMade", "sum"),
                fg3a=("threePointersAttempted", "sum"),
                ftm=("freeThrowsMade", "sum"),
                fta=("freeThrowsAttempted", "sum"),
                oreb=("reboundsOffensive", "sum"),
                dreb=("reboundsDefensive", "sum"),
                treb=("reboundsTotal", "sum"),
                ast=("assists", "sum"),
                stl=("steals", "sum"),
                blk=("blocks", "sum"),
                tov=("turnovers", "sum"),
                pf=("foulsPersonal", "sum"),
                plus_minus=("plusMinusPoints", "sum"),
            )
            .reset_index()
        )
    else:
        box_trad_grouped = pd.DataFrame(
            columns=[
                "team",
                "player_id",
                "games_with_box",
                "pts",
                "fgm",
                "fga",
                "fg3m",
                "fg3a",
                "ftm",
                "fta",
                "oreb",
                "dreb",
                "treb",
                "ast",
                "stl",
                "blk",
                "tov",
                "pf",
                "plus_minus",
            ]
        )

    # --- Aggregate advanced boxscore stats across games (this RUN only) ---
    if all_box_adv:
        box_adv_all = pd.concat(all_box_adv, ignore_index=True)
        box_adv_all = box_adv_all[box_adv_all["team"] == team_abbrev]

        box_adv_grouped = (
            box_adv_all.groupby(["team", "player_id"])
            .agg(
                ts_pct=("trueShootingPercentage", "mean"),
                efg_pct=("effectiveFieldGoalPercentage", "mean"),
                usg_pct=("usagePercentage", "mean"),
                est_usg_pct=("estimatedUsagePercentage", "mean"),
                off_rating_box=("offensiveRating", "mean"),
                def_rating_box=("defensiveRating", "mean"),
                net_rating_box=("netRating", "mean"),
                oreb_pct=("offensiveReboundPercentage", "mean"),
                dreb_pct=("defensiveReboundPercentage", "mean"),
                reb_pct=("reboundPercentage", "mean"),
                ast_pct=("assistPercentage", "mean"),
                tov_ratio=("turnoverRatio", "mean"),
                pace=("pace", "mean"),
                poss_est=("possessions", "sum"),
            )
            .reset_index()
        )
    else:
        box_adv_grouped = pd.DataFrame(
            columns=[
                "team",
                "player_id",
                "ts_pct",
                "efg_pct",
                "usg_pct",
                "est_usg_pct",
                "off_rating_box",
                "def_rating_box",
                "net_rating_box",
                "oreb_pct",
                "dreb_pct",
                "reb_pct",
                "ast_pct",
                "tov_ratio",
                "pace",
                "poss_est",
            ]
        )

    # Aggregate minutes across games (this RUN only)
    if all_minutes:
        minutes_all = pd.concat(all_minutes, ignore_index=True)
        minutes_all = minutes_all[minutes_all["team"] == team_abbrev]

        minutes_grouped = (
            minutes_all.groupby(["team", "player_id"])
            .agg(
                minutes_played=("minutes_played", "sum"),
            )
            .reset_index()
        )
    else:
        minutes_grouped = pd.DataFrame(columns=["team", "player_id", "minutes_played"])

    # Team-level totals across games (this RUN only, for on/off splits)
    if not team_game_summaries:
        raise RuntimeError(f"No team game summaries produced for team {team_abbrev}.")

    team_games_df = pd.DataFrame(team_game_summaries)

    team_totals_df = (
        team_games_df.groupby("team")
        .agg(
            team_total_off_possessions=("team_off_possessions", "sum"),
            team_total_def_possessions=("team_def_possessions", "sum"),
            team_total_possessions=("team_total_possessions", "sum"),
            team_total_off_points_for=("team_off_points_for", "sum"),
            team_total_def_points_against=("team_def_points_against", "sum"),
            team_total_net_points=("team_net_points", "sum"),
        )
        .reset_index()
    )

    # Aggregate across games (this RUN only)
    grouped = (
        all_stats_df.groupby(["team", "player_id"])
        .agg(
            games_played=("game_id", "nunique"),
            total_off_possessions=("off_possessions", "sum"),
            total_def_possessions=("def_possessions", "sum"),
            total_possessions=("total_possessions", "sum"),
            total_off_points_for=("off_points_for", "sum"),
            total_def_points_against=("def_points_against", "sum"),
            total_net_points=("net_points", "sum"),
        )
        .reset_index()
    )

    # Compute season-level ratings per 100 possessions (for games processed in this run)
    grouped["off_rating_per_100"] = grouped.apply(
        lambda row: (
            100.0 * row["total_off_points_for"] / row["total_off_possessions"]
            if row["total_off_possessions"] > 0
            else 0.0
        ),
        axis=1,
    )
    grouped["def_rating_per_100"] = grouped.apply(
        lambda row: (
            100.0 * row["total_def_points_against"] / row["total_def_possessions"]
            if row["total_def_possessions"] > 0
            else 0.0
        ),
        axis=1,
    )
    grouped["net_rating_per_100"] = grouped.apply(
        lambda row: (
            100.0 * row["total_net_points"] / row["total_possessions"]
            if row["total_possessions"] > 0
            else 0.0
        ),
        axis=1,
    )

    # Attach team totals for on/off splits
    grouped = grouped.merge(
        team_totals_df,
        on="team",
        how="left",
    )

    # Compute off-court (team minus player-on) totals
    grouped["off_possessions_off"] = (
        grouped["team_total_off_possessions"] - grouped["total_off_possessions"]
    ).clip(lower=0.0)

    grouped["def_possessions_off"] = (
        grouped["team_total_def_possessions"] - grouped["total_def_possessions"]
    ).clip(lower=0.0)

    grouped["off_points_for_off"] = (
        grouped["team_total_off_points_for"] - grouped["total_off_points_for"]
    )

    grouped["def_points_against_off"] = (
        grouped["team_total_def_points_against"] - grouped["total_def_points_against"]
    )

    grouped["total_possessions_off"] = (
        grouped["off_possessions_off"] + grouped["def_possessions_off"]
    )

    grouped["net_points_off"] = (
        grouped["off_points_for_off"] - grouped["def_points_against_off"]
    )

    # Off-court ratings per 100 possessions
    grouped["off_rating_off_per_100"] = grouped.apply(
        lambda row: (
            100.0 * row["off_points_for_off"] / row["off_possessions_off"]
            if row["off_possessions_off"] > 0
            else 0.0
        ),
        axis=1,
    )
    grouped["def_rating_off_per_100"] = grouped.apply(
        lambda row: (
            100.0 * row["def_points_against_off"] / row["def_possessions_off"]
            if row["def_possessions_off"] > 0
            else 0.0
        ),
        axis=1,
    )
    grouped["net_rating_off_per_100"] = grouped.apply(
        lambda row: (
            100.0 * row["net_points_off"] / row["total_possessions_off"]
            if row["total_possessions_off"] > 0
            else 0.0
        ),
        axis=1,
    )

    # On/off deltas (on-court minus off-court)
    grouped["off_rating_on_minus_off"] = (
        grouped["off_rating_per_100"] - grouped["off_rating_off_per_100"]
    )
    grouped["def_rating_on_minus_off"] = (
        grouped["def_rating_per_100"] - grouped["def_rating_off_per_100"]
    )
    grouped["net_rating_on_minus_off"] = (
        grouped["net_rating_per_100"] - grouped["net_rating_off_per_100"]
    )

    # Attach minutes
    grouped = grouped.merge(
        minutes_grouped,
        on=["team", "player_id"],
        how="left",
    ).fillna({"minutes_played": 0.0})

    # Attach traditional boxscore aggregates
    grouped = grouped.merge(
        box_trad_grouped,
        on=["team", "player_id"],
        how="left",
    )

    # Attach advanced boxscore aggregates
    grouped = grouped.merge(
        box_adv_grouped,
        on=["team", "player_id"],
        how="left",
    )

    # Derived scoring metrics
    grouped["pts_per_game"] = grouped["pts"] / grouped["games_played"].clip(lower=1)

    grouped["minutes_per_game"] = grouped["minutes_played"] / grouped[
        "games_played"
    ].clip(lower=1)

    grouped["pts_per_36"] = grouped.apply(
        lambda row: (
            (row["pts"] / row["minutes_played"] * 36.0)
            if row["minutes_played"] > 0
            else 0.0
        ),
        axis=1,
    )

    grouped["pts_per_100_poss"] = grouped.apply(
        lambda row: (
            (row["pts"] / row["total_off_possessions"] * 100.0)
            if row["total_off_possessions"] > 0
            else 0.0
        ),
        axis=1,
    )

    grouped["fg_pct"] = grouped.apply(
        lambda row: row["fgm"] / row["fga"] if row["fga"] > 0 else 0.0,
        axis=1,
    )
    grouped["three_pct"] = grouped.apply(
        lambda row: row["fg3m"] / row["fg3a"] if row["fg3a"] > 0 else 0.0,
        axis=1,
    )
    grouped["ft_pct"] = grouped.apply(
        lambda row: row["ftm"] / row["fta"] if row["fta"] > 0 else 0.0,
        axis=1,
    )

    # Attach player names
    result = grouped.merge(
        all_names_df,
        on=["team", "player_id"],
        how="left",
    )

    return result


def build_player_profile_for_team_season(
    team_abbrev: str,
    max_games: Optional[int] = None,
    season_label: Optional[str] = None,
    skip_game_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper around compute_player_stats_for_team_season that
    reshapes the full season stats into a clean "player stat profile"
    suitable for downstream consumption (e.g. OddzUp backend).

    NOTE: This version does not include SportsData.io IDs; it exposes only
    NBA stats ids and the computed metrics.

    Args:
        team_abbrev:
            Team abbreviation, e.g. 'ATL'.
        max_games:
            Optional cap on games to consider (see compute_player_stats_for_team_season).
        season_label:
            Currently unused here, but kept for future compatibility.
        skip_game_ids:
            Optional iterable of game IDs to skip when building this team's profile.
    """
    # 1) Get full season stats (full precision) for this RUN
    season_df = compute_player_stats_for_team_season(
        team_abbrev=team_abbrev,
        max_games=max_games,
        skip_game_ids=skip_game_ids,
    )

    if season_df.empty:
        # No new games processed; return empty profile frame
        return season_df

    # Plus/minus per game (total +/- divided by games played)
    if "plus_minus" in season_df.columns:
        season_df["plus_minus_per_game"] = season_df["plus_minus"] / season_df[
            "games_played"
        ].clip(lower=1)

    # Derived: possessions per game
    season_df["possessions_per_game"] = season_df["total_possessions"] / season_df[
        "games_played"
    ].clip(lower=1)

    profile_df = season_df.copy()

    # 3) Define the profile columns in the order we want to expose them
    profile_cols = [
        # Identity & keys
        "team",
        "player_id",  # NBA Stats player id (personId)
        "player_name",
        # Volume
        "games_played",
        "minutes_played",
        "minutes_per_game",
        "total_possessions",
        "possessions_per_game",
        # Impact & on/off
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
        "off_rating_off_per_100",
        "def_rating_off_per_100",
        "net_rating_off_per_100",
        "net_rating_on_minus_off",
        # Boxscore volume & efficiency
        "pts",
        "pts_per_game",
        "pts_per_36",
        "pts_per_100_poss",
        "fgm",
        "fga",
        "fg_pct",
        "fg3m",
        "fg3a",
        "three_pct",
        "ftm",
        "fta",
        "ft_pct",
        "oreb",
        "dreb",
        "treb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "plus_minus",
        "plus_minus_per_game",  # average +/- per game
        # Advanced percentages & usage
        "ts_pct",
        "efg_pct",
        "usg_pct",
        "est_usg_pct",
        "oreb_pct",
        "dreb_pct",
        "reb_pct",
        "ast_pct",
        "tov_ratio",
        "off_rating_box",
        "def_rating_box",
        "net_rating_box",
        "pace",
    ]

    existing_cols = [c for c in profile_cols if c in profile_df.columns]
    profile_df = profile_df[existing_cols].copy()

    # 4) Round ONLY the appropriate fields to 1 decimal
    one_decimal_cols = [
        "minutes_played",
        "minutes_per_game",
        "total_possessions",
        "possessions_per_game",
        "pts_per_game",
        "pts_per_36",
        "pts_per_100_poss",
        "plus_minus_per_game",
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
        "off_rating_off_per_100",
        "def_rating_off_per_100",
        "net_rating_off_per_100",
        "net_rating_on_minus_off",
        "off_rating_box",
        "def_rating_box",
        "net_rating_box",
        "pace",
    ]

    cols_to_round = [c for c in one_decimal_cols if c in profile_df.columns]
    if cols_to_round:
        profile_df[cols_to_round] = profile_df[cols_to_round].astype(float).round(1)

    # 5) Round percentage-like stats to 3 decimals for cleaner output
    percent_cols = [
        "fg_pct",
        "three_pct",
        "ft_pct",
        "ts_pct",
        "efg_pct",
        "usg_pct",
        "est_usg_pct",
        "oreb_pct",
        "dreb_pct",
        "reb_pct",
        "ast_pct",
        "tov_ratio",
    ]

    perc_to_round = [c for c in percent_cols if c in profile_df.columns]
    if perc_to_round:
        profile_df[perc_to_round] = profile_df[perc_to_round].astype(float).round(3)

    return profile_df
