from data.nba_api_provider import (
    get_team_regular_season_games,
    get_game_rotation,
)
from features.lineup_builder import build_player_stints_from_rotation
from features.lineup_stints import build_lineup_stints_for_game


def main():
    team_abbrev = "ATL"  # you can change this if you want

    print(f"📅 Fetching regular season games for {team_abbrev} in current season...")
    team_games_df = get_team_regular_season_games(team_abbrev)
    print(team_games_df[["GameID", "GameDate", "MATCHUP"]].head())

    if team_games_df.empty:
        print("\n❌ No games found; season inference may be off.")
        return

    game_id = str(team_games_df.iloc[0]["GameID"])
    print(f"\n🎬 Fetching GameRotation for GameID {game_id}...")
    rotation_df = get_game_rotation(game_id)

    if rotation_df.empty:
        print("\n❌ Empty rotation_df; check API access or GameID.")
        return

    # Build per-player stints
    stints_df = build_player_stints_from_rotation(rotation_df)

    if stints_df.empty:
        print("\n❌ No player stints built; check time parsing or rotation data.")
        return

    print("\n🔎 Player stints (first 10 rows):")
    print(stints_df.head(10))

    # Build lineup stints
    lineup_stints_df = build_lineup_stints_for_game(stints_df)

    if lineup_stints_df.empty:
        print("\n⚠️ No lineup stints constructed (did not find 5v5 intervals).")
        return

    print("\n🔎 Lineup stints (first 10 rows):")
    print(lineup_stints_df.head(10))

    print("\n📐 Lineup stints columns:")
    print(lineup_stints_df.columns.tolist())

    print("\n🔍 Lineup stint duration summary (seconds):")
    print(lineup_stints_df["stint_duration_seconds"].describe())

    # Quick sanity checks
    print("\n🔍 Example home lineup IDs for first stint:")
    first = lineup_stints_df.iloc[0]
    print("Home team:", first["home_team"], "players:", first["home_player_ids"])
    print("Away team:", first["away_team"], "players:", first["away_player_ids"])


if __name__ == "__main__":
    main()
