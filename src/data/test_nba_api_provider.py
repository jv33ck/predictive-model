from data.nba_api_provider import (
    get_today_games_and_teams,
    get_team_regular_season_games,
    get_game_play_by_play,
)
import pandas as pd


def main():
    games_df, teams = get_today_games_and_teams()
    print("✅ Today’s games from nba_api:")
    print(games_df[["GAME_ID", "HomeTeam", "AwayTeam"]])
    print("\n✅ Teams playing today:", teams)

    if not teams:
        print("\n⚠️ No games today. You can still test by hardcoding a team & season.")
        return

    test_team = teams[0]

    print(
        f"\n📅 Fetching regular season games for {test_team} in current season inferred from nba_api..."
    )
    team_games_df = get_team_regular_season_games(test_team)
    print(team_games_df[["GameID", "GameDate", "MATCHUP"]].head())

    if team_games_df.empty:
        print(
            "\n❌ Still no games found for this team in current season. Something's off with season inference."
        )
        return

    pbp_df = pd.DataFrame()
    sample_game_id = None

    for _, row in team_games_df.iterrows():
        candidate_id = row["GameID"]
        print(f"\n🎬 Trying play-by-play for GameID {candidate_id}...")
        pbp_df = get_game_play_by_play(str(candidate_id))

        if not pbp_df.empty:
            sample_game_id = candidate_id
            print(f"✅ Successfully fetched PBP for GameID {candidate_id}")
            break
        else:
            print(f"⚠️ No usable PBP data for GameID {candidate_id}, trying next...")

    if pbp_df.empty:
        print("\n❌ Could not find any game with usable PBP data for this team.")
        return

    print("\n🔎 Sample of play-by-play data:")
    print(pbp_df.head())
    print("\n🔎 Sample of PlayByPlayV3 data:")
    cols_to_show = [
        "gameId",
        "actionNumber",
        "clock",
        "period",
        "teamTricode",
        "personId",
        "playerName",
        "shotDistance",
        "shotResult",
        "isFieldGoal",
        "scoreHome",
        "scoreAway",
        "pointsTotal",
        "description",
        "actionType",
        "subType",
    ]
    existing_cols = [c for c in cols_to_show if c in pbp_df.columns]
    print(pbp_df[existing_cols].head())


if __name__ == "__main__":
    main()
