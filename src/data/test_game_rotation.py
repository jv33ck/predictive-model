from data.nba_api_provider import get_team_regular_season_games, get_game_rotation


def main():
    # Hardcode a team for testing or reuse today's flow
    team_abbrev = "ATL"

    team_games_df = get_team_regular_season_games(team_abbrev)
    print(team_games_df[["GameID", "GameDate", "MATCHUP"]].head())

    if team_games_df.empty:
        print("No games found for this team; season inference may be off.")
        return

    game_id = str(team_games_df.iloc[0]["GameID"])
    print(f"\n🎬 Fetching GameRotation for GameID {game_id}...")

    rotation_df = get_game_rotation(game_id)
    print("\n🔎 GameRotation sample (first 10 rows):")
    print(rotation_df.head(10))

    print("\n📐 Columns:")
    print(rotation_df.columns.tolist())


if __name__ == "__main__":
    main()
