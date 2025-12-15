from data.nba_api_provider import (
    get_team_regular_season_games,
    get_game_boxscore_traditional,
    get_game_boxscore_advanced,
)


def main():
    team_abbrev = "ATL"

    print(f"📅 Fetching regular season games for {team_abbrev} in current season...")
    games_df = get_team_regular_season_games(team_abbrev)
    print(games_df[["GameID", "GameDate", "MATCHUP"]].head())

    if games_df.empty:
        print("\n❌ No games found.")
        return

    game_row = games_df.iloc[0]
    game_id = str(game_row["GameID"])
    matchup = game_row["MATCHUP"]

    print(f"\n🎬 Using GameID {game_id} ({matchup}) for boxscore tests...")

    # Traditional boxscore
    trad_df = get_game_boxscore_traditional(game_id)
    if trad_df.empty:
        print("\n❌ Traditional boxscore is empty.")
    else:
        print("\n📦 Traditional boxscore columns:")
        print(trad_df.columns.tolist())
        print("\n🔎 Traditional boxscore (first 10 rows):")
        print(trad_df.head(10))

        # Sanity check: sum of points per team
        print("\n📊 Sum of points by teamTricode (traditional):")
        print(trad_df.groupby("teamTricode")["points"].sum())

    # Advanced boxscore
    adv_df = get_game_boxscore_advanced(game_id)
    if adv_df.empty:
        print("\n❌ Advanced boxscore is empty.")
    else:
        print("\n📦 Advanced boxscore columns:")
        print(adv_df.columns.tolist())
        print("\n🔎 Advanced boxscore (first 10 rows):")
        print(adv_df.head(10))

        # Sanity check: possessions / ratings for one player
        sample = adv_df.iloc[0]
        print(
            f"\n🧪 Sample advanced row: {sample['teamTricode']} {sample['firstName']} {sample['familyName']}, "
            f"possessions={sample['possessions']}, "
            f"netRating={sample['netRating']}, "
            f"trueShooting%={sample['trueShootingPercentage']}"
        )


if __name__ == "__main__":
    main()
