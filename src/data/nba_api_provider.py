# src/data/nba_api_provider.py
from datetime import date
from typing import Tuple, List
from requests.exceptions import ReadTimeout, RequestException

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, leaguegamelog, playbyplayv2
from nba_api.stats.static import teams as static_teams
from nba_apiv3.stats.endpoints import playbyplayv3
from nba_api.stats.endpoints import gamerotation
from nba_api.stats.endpoints import boxscoretraditionalv3, boxscoreadvancedv3


# --- Helper: map team abbreviations to NBA team IDs ---


def _infer_current_nba_api_season() -> str:
    """
    Infer the current NBA season string for nba_api (e.g., '2023-24', '2025-26')
    using today's scoreboard.

    This uses nba_api's own data, so it's guaranteed to align with LeagueGameLog.
    """
    today = date.today().strftime("%m/%d/%Y")
    sb = scoreboardv2.ScoreboardV2(game_date=today)
    games_df = sb.game_header.get_data_frame()

    if games_df.empty:
        # Fallback: if no games today, guess based on today's date
        year = date.today().year
        # NBA seasons start in the fall; if we're before August, likely still previous season
        if date.today().month < 8:
            year -= 1
        return f"{year}-{str(year + 1)[-2:]}"

    # Try to infer from SEASON or SEASON_ID if available
    if "SEASON" in games_df.columns:
        season_start_year = int(games_df["SEASON"].iloc[0])
    elif "SEASON_ID" in games_df.columns:
        # SEASON_ID is usually like 22023 (the last 4 digits are the start year)
        season_id = str(games_df["SEASON_ID"].iloc[0])
        season_start_year = int(season_id[-4:])
    else:
        # Fallback: infer from game date column if present
        if "GAME_DATE_EST" in games_df.columns:
            game_date = pd.to_datetime(games_df["GAME_DATE_EST"].iloc[0])
        else:
            game_date = pd.to_datetime(today)
        year = game_date.year
        if game_date.month < 8:  # before August => still previous season
            year -= 1
        season_start_year = year

    return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"


def _get_team_id_from_abbrev(team_abbrev: str) -> int:
    """
    Map a team abbreviation (e.g. 'BOS') to its NBA team ID using nba_api.
    """
    all_teams = static_teams.get_teams()
    for t in all_teams:
        if t["abbreviation"].upper() == team_abbrev.upper():
            return t["id"]
    raise ValueError(f"Could not find NBA team ID for abbreviation: {team_abbrev}")


def get_game_rotation(game_id: str) -> pd.DataFrame:
    """
    Fetch player rotation / stint info for a single game using the NBA Stats
    GameRotation endpoint.

    Args:
        game_id:
            10-digit NBA game ID string, e.g. '0022500082'.

    Returns:
        DataFrame with one row per player stint, combining home and away teams.
        Includes at least:
            - GAME_ID
            - TEAM_ID
            - TEAM_CITY
            - TEAM_NAME
            - PERSON_ID
            - PLAYER_FIRST
            - PLAYER_LAST
            - IN_TIME_REAL
            - OUT_TIME_REAL
            - PLAYER_PTS
            - PT_DIFF
            - USG_PCT
        Plus convenience columns:
            - is_home: bool
            - player_id, player_first, player_last, player_name
            - team_abbrev: team tricode/abbreviation (e.g. 'ATL')
    """
    # Call the NBA Stats endpoint
    rotation = gamerotation.GameRotation(
        game_id=game_id,
        league_id="00",
    )

    # Away team stints
    away_df = rotation.away_team.get_data_frame()
    away_df = away_df.copy()
    away_df["is_home"] = False

    # Home team stints
    home_df = rotation.home_team.get_data_frame()
    home_df = home_df.copy()
    home_df["is_home"] = True

    # Combine
    df = pd.concat([home_df, away_df], ignore_index=True)

    # Map TEAM_ID -> team abbreviation using static_teams
    all_teams = static_teams.get_teams()
    team_id_to_abbrev = {t["id"]: t["abbreviation"] for t in all_teams}

    df["team_abbrev"] = df["TEAM_ID"].map(team_id_to_abbrev)

    # Convenience columns
    df["player_id"] = df["PERSON_ID"]
    df["player_first"] = df["PLAYER_FIRST"]
    df["player_last"] = df["PLAYER_LAST"]

    df["player_name"] = (
        df["PLAYER_FIRST"]
        .fillna("")
        .astype(str)
        .str.cat(df["PLAYER_LAST"].fillna("").astype(str), sep=" ")
        .str.strip()
    )
    return df


# --- Public API functions (these are the ones the rest of your stack should use) ---


def get_today_games_and_teams() -> Tuple[pd.DataFrame, List[str]]:
    """
    Use nba_api to get today's NBA games and the teams playing.

    Returns:
        games_df: DataFrame of today's games (one row per game)
        teams:    sorted list of team abbreviations playing today
    """
    today = date.today().strftime("%m/%d/%Y")

    sb = scoreboardv2.ScoreboardV2(game_date=today)
    games_df = sb.game_header.get_data_frame()  # nba_api helper

    # Normalize columns we'll care about
    # GAME_ID is like '0022300001'
    # HOME_TEAM_ID / VISITOR_TEAM_ID map to team IDs, not abbrevs
    all_teams = static_teams.get_teams()
    team_id_to_abbrev = {t["id"]: t["abbreviation"] for t in all_teams}

    games_df["HomeTeam"] = games_df["HOME_TEAM_ID"].map(team_id_to_abbrev)
    games_df["AwayTeam"] = games_df["VISITOR_TEAM_ID"].map(team_id_to_abbrev)

    teams = sorted(
        set(games_df["HomeTeam"].dropna().tolist())
        | set(games_df["AwayTeam"].dropna().tolist())
    )

    return games_df, teams


def get_team_regular_season_games(
    team_abbrev: str, season: str | None = None
) -> pd.DataFrame:
    """
    Use nba_api to get all REGULAR SEASON games for a given team in a season.

    Args:
        team_abbrev: e.g. 'BOS'
        season: NBA season string for nba_api, e.g. '2023-24'.
                If None, infer current season from nba_api's scoreboard.

    Returns:
        DataFrame with at least GameID, GameDate, MATCHUP, WL
    """
    if season is None:
        season = _infer_current_nba_api_season()
        print(f"🔎 [nba_api] Using inferred season string: {season}")

    logs = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
    )
    df = logs.get_data_frames()[0]

    # Filter to the team we care about
    df = df[df["TEAM_ABBREVIATION"] == team_abbrev.upper()].copy()

    # Normalize column names
    df = df.rename(
        columns={
            "GAME_ID": "GameID",
            "GAME_DATE": "GameDate",
            "TEAM_ABBREVIATION": "Team",
        }
    )

    df = df.sort_values("GameDate").reset_index(drop=True)
    return df


import pandas as pd


def get_game_play_by_play(game_id: str) -> pd.DataFrame:
    """
    Fetch play-by-play for a single game using NBA Stats PlayByPlayV3.

    Args:
        game_id: e.g. '0022501205' (10-digit NBA game ID)

    Returns:
        DataFrame of play-by-play events with columns like:
        ['gameId', 'actionNumber', 'clock', 'period', 'teamId', 'teamTricode',
         'personId', 'playerName', 'shotDistance', 'shotResult', 'isFieldGoal',
         'scoreHome', 'scoreAway', 'pointsTotal', 'location', 'description',
         'actionType', 'subType', ...]
    """
    try:
        # v3 requires StartPeriod / EndPeriod as well (all required)
        pbp = playbyplayv3.PlayByPlayV3(
            game_id=game_id,
            start_period="1",
            end_period="10",  # safely covers all NBA periods including OT
        )

        # Depending on the library implementation, you either use:
        # df = pbp.get_data_frames()[0]
        # or:
        df = pbp.play_by_play.get_data_frame()

        return df

    except Exception as e:
        print(f"⚠️ Failed to fetch PlayByPlayV3 for GameID {game_id}: {e}")
        return pd.DataFrame()


def get_game_boxscore_traditional(game_id: str) -> pd.DataFrame:
    """
    Fetch traditional box score (per-player) for a single game using
    BoxScoreTraditionalV3.

    Returns a DataFrame with one row per player in the game, columns like:
    - gameId, teamId, teamTricode, teamSlug
    - personId, firstName, familyName, playerSlug, position, jerseyNum
    - minutes
    - fieldGoalsMade, fieldGoalsAttempted, fieldGoalsPercentage
    - threePointersMade, threePointersAttempted, threePointersPercentage
    - freeThrowsMade, freeThrowsAttempted, freeThrowsPercentage
    - reboundsOffensive, reboundsDefensive, reboundsTotal
    - assists, steals, blocks, turnovers, foulsPersonal
    - points, plusMinusPoints
    """

    try:
        b = boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=game_id,
            start_period="1",
            end_period="10",  # full game
            start_range="0",
            end_range="0",
            range_type="0",
        )

        # player_stats can be typed as Optional in nba_api, so guard it
        stats_dataset = b.player_stats
        if stats_dataset is None:
            print(
                f"⚠️ BoxScoreTraditionalV3 returned no player_stats for game {game_id}"
            )
            return pd.DataFrame()

        df = stats_dataset.get_data_frame()
        return df
    except (ReadTimeout, RequestException) as e:
        print(f"⚠️ BoxScoreTraditionalV3 timeout for game {game_id}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error fetching BoxScoreTraditionalV3 for game {game_id}: {e}")
        return pd.DataFrame()


def get_game_boxscore_advanced(game_id: str) -> pd.DataFrame:
    """
    Fetch advanced box score (per-player) for a single game using
    BoxScoreAdvancedV3.

    Returns a DataFrame with one row per player in the game, columns like:
      - gameId, teamId, teamTricode, teamSlug
      - personId, firstName, familyName, playerSlug, position, jerseyNum
      - minutes
      - estimatedOffensiveRating, offensiveRating
      - estimatedDefensiveRating, defensiveRating
      - estimatedNetRating, netRating
      - assistPercentage, assistToTurnover, assistRatio
      - offensiveReboundPercentage, defensiveReboundPercentage, reboundPercentage
      - turnoverRatio
      - effectiveFieldGoalPercentage, trueShootingPercentage
      - usagePercentage, estimatedUsagePercentage
      - estimatedPace, pace, pacePer40
      - possessions, PIE
    """
    try:
        b = boxscoreadvancedv3.BoxScoreAdvancedV3(
            game_id=game_id,
            start_period="1",
            end_period="10",  # full game
            start_range="0",
            end_range="0",
            range_type="0",
        )

        stats_dataset = b.player_stats
        if stats_dataset is None:
            print(f"⚠️ BoxScoreAdvancedV3 returned no player_stats for game {game_id}")
            return pd.DataFrame()

        df = stats_dataset.get_data_frame()
        return df
    except (ReadTimeout, RequestException) as e:
        print(f"⚠️ BoxScoreAdvancedV3 timeout for game {game_id}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error fetching BoxScoreAdvancedV3 for game {game_id}: {e}")
        return pd.DataFrame()
