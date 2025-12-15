from data_schedule import get_teams_playing_today


def main():
    teams, games_df = get_teams_playing_today()
    print("✅ Teams playing today:")
    print(teams)
    print("\n✅ Games:")
    print(games_df[["HomeTeam", "AwayTeam", "DateTime", "GameID"]])


if __name__ == "__main__":
    main()
