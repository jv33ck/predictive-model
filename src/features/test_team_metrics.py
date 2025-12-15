from data.nba_api_provider import (
    get_today_games_and_teams,
    get_team_regular_season_games,
    get_game_play_by_play,
)
from features.pbp_normalizer import normalize_pbp_v3
from features.possession_builder import build_scoring_possessions_for_game
from features.team_metrics import compute_team_game_ratings_from_scoring_possessions
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
            f"Team {team_abbrev} not found in MATCHUP {matchup} -> ({home_team}, {away_team})"
        )

    return home_team, away_team


def main():
    games_df, teams = get_today_games_and_teams()
    print("✅ Today’s games from nba_api:")
    print(games_df[["GAME_ID", "HomeTeam", "AwayTeam"]])
    print("\n✅ Teams playing today:", teams)

    if not teams:
        print("\n⚠️ No games today. You can still test by hardcoding a team & game.")
        return

    test_team = teams[0]

    print(f"\n📅 Fetching regular season games for {test_team} in current season...")
    team_games_df = get_team_regular_season_games(test_team)
    print(team_games_df[["GameID", "GameDate", "MATCHUP"]].head())

    if team_games_df.empty:
        print("\n❌ No games found for this team; season inference might be off.")
        return

    # Try games until we get usable PBP
    pbp_df = pd.DataFrame()
    sample_row = None

    for _, row in team_games_df.iterrows():
        game_id = row["GameID"]
        print(f"\n🎬 Trying PlayByPlayV3 for GameID {game_id}...")
        raw_pbp = get_game_play_by_play(str(game_id))

        if raw_pbp.empty:
            print(f"⚠️ No usable PBP for GameID {game_id}, trying next...")
            continue

        pbp_df = raw_pbp
        sample_row = row
        print(f"✅ Successfully fetched raw PBP for GameID {game_id}")
        break

    if pbp_df.empty or sample_row is None:
        print("\n❌ Could not find any game with usable PBP data for this team.")
        return

    # Normalize events
    normalized = normalize_pbp_v3(pbp_df)

    # Infer home/away
    matchup = sample_row["MATCHUP"]
    team_abbrev = sample_row["Team"] if "Team" in sample_row else test_team
    home_team, away_team = _infer_home_away_from_matchup(team_abbrev, matchup)
    print(f"\n🏠 Home team: {home_team}, 🧳 Away team: {away_team}")

    # Build scoring possessions
    scoring_possessions = build_scoring_possessions_for_game(
        events_df=normalized,
        home_team=home_team,
        away_team=away_team,
    )

    print("\n🔎 Scoring possessions (first 10 rows):")
    print(scoring_possessions.head(10))

    # Compute team ratings from scoring possessions
    team_ratings = compute_team_game_ratings_from_scoring_possessions(
        scoring_possessions
    )

    print("\n📊 Team ratings (based on scoring possessions only):")
    print(team_ratings)

    # Quick sense-check: sum of points_for should match final score totals
    print("\n🔍 Quick check – total points_for by team:")
    print(
        team_ratings[["game_id", "team", "points_for", "points_against"]].sort_values(
            ["game_id", "team"]
        )
    )


if __name__ == "__main__":
    main()
