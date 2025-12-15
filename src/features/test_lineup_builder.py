from data.nba_api_provider import (
    get_team_regular_season_games,
    get_game_rotation,
)
from features.lineup_builder import build_player_stints_from_rotation


def main():
    team_abbrev = "ATL"  # you can change this if you want

    print(f"📅 Fetching regular season games for {team_abbrev} in current season...")
    team_games_df = get_team_regular_season_games(team_abbrev)
    print(team_games_df[["GameID", "GameDate", "MATCHUP"]].head())

    if team_games_df.empty:
        print("\n❌ No games found; season inference might be off.")
        return

    game_id = str(team_games_df.iloc[0]["GameID"])
    print(f"\n🎬 Fetching GameRotation for GameID {game_id}...")
    rotation_df = get_game_rotation(game_id)

    if rotation_df.empty:
        print("\n❌ Empty rotation_df; check API access or GameID.")
        return

    print("\n🔎 Raw GameRotation sample (first 10 rows):")
    print(rotation_df.head(10))
    print("\n📐 Raw GameRotation columns:")
    print(rotation_df.columns.tolist())

    # Build player stints
    stints_df = build_player_stints_from_rotation(rotation_df)

    if stints_df.empty:
        print("\n⚠️ No stints built; check time parsing.")
        return

    print("\n🔎 Player stints (first 10 rows):")
    print(stints_df.head(10))

    print("\n📐 Player stints columns:")
    print(stints_df.columns.tolist())

    # Quick sanity checks
    print("\n🔍 Stint duration summary (seconds):")
    print(stints_df["stint_duration_seconds"].describe())


if __name__ == "__main__":
    main()
