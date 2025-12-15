from data.nba_api_provider import (
    get_today_games_and_teams,
    get_team_regular_season_games,
    get_game_play_by_play,
)
from features.pbp_normalizer import normalize_pbp_v3
from features.possession_builder import build_scoring_possessions_for_game
import pandas as pd


def _infer_home_away_from_matchup(team_abbrev: str, matchup: str) -> tuple[str, str]:
    """
    Infer home/away from an nba_api 'MATCHUP' string and the team abbreviation.

    Examples:
        team_abbrev='ATL', matchup='ATL vs. TOR' -> home='ATL', away='TOR'
        team_abbrev='ATL', matchup='ATL @ TOR'   -> home='TOR', away='ATL'
    """
    parts = matchup.split()
    if len(parts) != 3:
        raise ValueError(f"Unexpected MATCHUP format: {matchup!r}")

    team_code, sep, opp_code = parts
    team_code = team_code.upper()
    opp_code = opp_code.upper()

    if sep == "vs.":
        # TEAM vs. OPP => TEAM is home
        home_team = team_code
        away_team = opp_code
    elif sep == "@":
        # TEAM @ OPP => TEAM is away
        home_team = opp_code
        away_team = team_code
    else:
        raise ValueError(f"Unexpected separator in MATCHUP: {sep!r}")

    # Sanity: team_abbrev should match one of them
    if team_abbrev.upper() not in {home_team, away_team}:
        raise ValueError(
            f"Team {team_abbrev} not found in MATCHUP {matchup} -> ({home_team}, {away_team})"
        )

    return home_team, away_team


def main():
    # 1) Today’s slate & teams
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

    # 2) Pick the first game we can get PBP for
    pbp_df = pd.DataFrame()
    sample_game_row = None

    for _, row in team_games_df.iterrows():
        candidate_id = row["GameID"]
        print(f"\n🎬 Trying PlayByPlayV3 for GameID {candidate_id}...")
        raw_pbp = get_game_play_by_play(str(candidate_id))

        if raw_pbp.empty:
            print(f"⚠️ No usable PBP data for GameID {candidate_id}, trying next...")
            continue

        pbp_df = raw_pbp
        sample_game_row = row
        print(f"✅ Successfully fetched raw PBP for GameID {candidate_id}")
        break

    if pbp_df.empty or sample_game_row is None:
        print("\n❌ Could not find any game with usable PBP data for this team.")
        return

    # 3) Normalize PBP
    normalized = normalize_pbp_v3(pbp_df)

    # 4) Infer home/away for this game from MATCHUP
    matchup = sample_game_row["MATCHUP"]
    team_abbrev = sample_game_row["Team"] if "Team" in sample_game_row else test_team
    home_team, away_team = _infer_home_away_from_matchup(team_abbrev, matchup)
    print(f"\n🏠 Home team: {home_team}, 🧳 Away team: {away_team}")

    # 5) Build scoring possessions
    game_id = normalized["game_id"].iloc[0]
    print(f"\n🧱 Building scoring possessions for game_id={game_id}...")

    scoring_possessions = build_scoring_possessions_for_game(
        events_df=normalized,
        home_team=home_team,
        away_team=away_team,
    )

    if scoring_possessions.empty:
        print("\n⚠️ No scoring possessions constructed (this would be unusual).")
        return

    print("\n🔎 Scoring possessions (first 10 rows):")
    print(scoring_possessions.head(10))

    print("\n📐 Scoring possessions columns:")
    print(scoring_possessions.columns.tolist())

    # Quick sanity: total points
    total_points = scoring_possessions["points"].sum()
    print(f"\n📊 Total points summed from scoring possessions: {total_points}")


if __name__ == "__main__":
    main()
