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
from features.player_possessions import aggregate_player_possession_stats
import pandas as pd


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


def main():
    team_abbrev = "ATL"

    print(f"📅 Fetching regular season games for {team_abbrev} in current season...")
    team_games_df = get_team_regular_season_games(team_abbrev)
    print(team_games_df[["GameID", "GameDate", "MATCHUP"]].head())

    if team_games_df.empty:
        print("\n❌ No games found; season inference may be off.")
        return

    game_row = team_games_df.iloc[0]
    game_id = str(game_row["GameID"])
    matchup = game_row["MATCHUP"]
    team_for_matchup = game_row["Team"] if "Team" in game_row else team_abbrev

    print(f"\n🎬 Using GameID {game_id} ({matchup})")

    home_team, away_team = _infer_home_away_from_matchup(team_for_matchup, matchup)
    print(f"🏠 Home team: {home_team}, 🧳 Away team: {away_team}")

    # 1) PBP -> normalized events -> ALL possessions
    raw_pbp = get_game_play_by_play(game_id)
    if raw_pbp.empty:
        print("\n❌ No PBP for this game.")
        return

    normalized = normalize_pbp_v3(raw_pbp)

    all_possessions = build_all_possessions_for_game(
        events_df=normalized,
        home_team=home_team,
        away_team=away_team,
    )

    if all_possessions.empty:
        print("\n❌ No possessions built.")
        return

    print("\n🔎 All possessions (first 10 rows):")
    print(all_possessions.head(10))

    # 2) Rotation -> player stints -> lineup stints
    rotation_df = get_game_rotation(game_id)
    if rotation_df.empty:
        print("\n❌ No rotation data for this game.")
        return

    stints_df = build_player_stints_from_rotation(rotation_df)
    if stints_df.empty:
        print("\n❌ No player stints built.")
        return

    lineup_stints_df = build_lineup_stints_for_game(stints_df)
    if lineup_stints_df.empty:
        print("\n❌ No lineup stints built.")
        return

    print("\n🔎 Lineup stints (first 5 rows):")
    print(lineup_stints_df.head())

    # 3) Attach lineups to ALL possessions
    enriched_possessions = attach_lineups_to_possessions(
        possessions=all_possessions,
        lineup_stints=lineup_stints_df,
    )

    print("\n🔎 Enriched possessions (first 10 rows):")
    cols_to_show = [
        "game_id",
        "possession_index",
        "period",
        "seconds_remaining",
        "offense_team",
        "defense_team",
        "points",
        "offense_player_ids",
        "defense_player_ids",
    ]
    existing = [c for c in cols_to_show if c in enriched_possessions.columns]
    print(enriched_possessions[existing].head(10))

    # 4) Aggregate per-player possession stats
    player_stats = aggregate_player_possession_stats(enriched_possessions)

    if player_stats.empty:
        print("\n❌ No player possession stats produced.")
        return

    print("\n📊 Raw per-player possession stats (first 15 rows):")
    print(player_stats.head(15))

    # Attach names for readability
    player_names = (
        stints_df[["player_id", "team_abbrev", "player_name"]]
        .drop_duplicates()
        .rename(columns={"team_abbrev": "team"})
    )

    player_stats_with_names = player_stats.merge(
        player_names,
        on=["player_id", "team"],
        how="left",
    )

    # Show high-possession players with ratings
    print("\n📊 Per-player possession stats with names (sorted by total_possessions):")
    cols = [
        "game_id",
        "team",
        "player_id",
        "player_name",
        "off_possessions",
        "def_possessions",
        "total_possessions",
        "off_points_for",
        "def_points_against",
        "net_points",
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
    ]
    existing_cols = [c for c in cols if c in player_stats_with_names.columns]
    print(
        player_stats_with_names[existing_cols]
        .sort_values("total_possessions", ascending=False)
        .head(20)
    )


if __name__ == "__main__":
    main()
