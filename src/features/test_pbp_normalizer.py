from data.nba_api_provider import (
    get_today_games_and_teams,
    get_team_regular_season_games,
    get_game_play_by_play,
)
from features.pbp_normalizer import normalize_pbp_v3
import pandas as pd


def main():
    # 1) Get today's teams
    games_df, teams = get_today_games_and_teams()
    print("✅ Today’s games from nba_api:")
    print(games_df[["GAME_ID", "HomeTeam", "AwayTeam"]])
    print("\n✅ Teams playing today:", teams)

    if not teams:
        print("\n⚠️ No games today. You can still test by hardcoding a team & game ID.")
        return

    test_team = teams[0]
    print(f"\n📅 Fetching regular season games for {test_team} in current season...")
    team_games_df = get_team_regular_season_games(test_team)
    print(team_games_df[["GameID", "GameDate", "MATCHUP"]].head())

    if team_games_df.empty:
        print("\n❌ No games found for this team; season inference might be off.")
        return

    # 2) Try games until we find one with usable PBP
    pbp_df = pd.DataFrame()
    sample_game_id = None

    for _, row in team_games_df.iterrows():
        candidate_id = row["GameID"]
        print(f"\n🎬 Trying PlayByPlayV3 for GameID {candidate_id}...")
        raw_pbp = get_game_play_by_play(str(candidate_id))

        if raw_pbp.empty:
            print(f"⚠️ No usable PBP data for GameID {candidate_id}, trying next...")
            continue

        sample_game_id = candidate_id
        pbp_df = raw_pbp
        print(f"✅ Successfully fetched raw PBP for GameID {candidate_id}")
        break

    if pbp_df.empty:
        print("\n❌ Could not find any game with usable PBP data for this team.")
        return

    # 3) Normalize
    normalized = normalize_pbp_v3(pbp_df)

    print("\n🔎 Raw PBP sample (first 5 rows):")
    print(pbp_df.head())

    print("\n🔎 Normalized PBP sample (first 10 rows):")
    print(normalized.head(10))

    print("\n📐 Normalized schema columns:")
    print(normalized.columns.tolist())

    # Optional: show scoring events only
    scoring_events = normalized[normalized["is_scoring_event"]]
    print("\n🏀 Scoring events (first 10):")
    print(scoring_events.head(10))


if __name__ == "__main__":
    main()
