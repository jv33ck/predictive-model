# src/data/nba_api_provider.py
from datetime import date, datetime
from typing import Tuple, List
from requests.exceptions import ReadTimeout, RequestException

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, leaguegamelog, playbyplayv2
from nba_api.stats.static import teams as static_teams
from nba_apiv3.stats.endpoints import playbyplayv3
from nba_api.stats.endpoints import gamerotation
from nba_api.stats.endpoints import boxscoretraditionalv3, boxscoreadvancedv3
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.stats.endpoints import ScheduleLeagueV2
from functools import lru_cache


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


@lru_cache(maxsize=1)
def get_team_id_map() -> dict[int, str]:
    """
    Return a mapping from numeric TEAM_ID -> team abbreviation (TRICODE).

    Uses nba_api.stats.static.teams and caches the result so that repeated
    calls across the pipeline are cheap and do not hit the API again.
    """
    teams_data = static_teams.get_teams()
    # Each item looks like: {'id': 1610612747, 'full_name': 'Los Angeles Lakers', 'abbreviation': 'LAL', ...}
    return {int(t["id"]): str(t["abbreviation"]) for t in teams_data}


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


def get_leaguegamelog_team(
    season: str,
    season_type: str = "Regular Season",
    team_abbrev: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    direction: str = "ASC",
    counter: int = 0,
    league_id: str = "00",
) -> pd.DataFrame:
    """
    Fetch team-level game logs from the LeagueGameLog endpoint.

    This is a *fast* way to pull one row per team-game for a season,
    including basic box score + plus/minus and game date, without
    touching play-by-play.

    Parameters
    ----------
    season : str
        NBA API season string (e.g. '2025-26').
    season_type : str, default 'Regular Season'
        One of: 'Regular Season', 'Pre Season', 'Playoffs', 'All Star', 'All-Star'.
    team_abbrev : str, optional
        If provided (e.g. 'NYK'), filter results to that team only.
        If None, return logs for all teams in the league.
    date_from : str, optional
        Lower bound date filter in 'MM/DD/YYYY'. If None, no lower bound.
    date_to : str, optional
        Upper bound date filter in 'MM/DD/YYYY'. If None, no upper bound.
    direction : str, default 'ASC'
        Sort direction, 'ASC' or 'DESC' on the `sorter` column.
    counter : int, default 0
        NBA API paging counter (usually 0 for full season).
    league_id : str, default '00'
        League ID (NBA = '00').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
         'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN',
         'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
         'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
         'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS',
         'VIDEO_AVAILABLE']
    """

    # LeagueGameLog expects MM/DD/YYYY for date filters; allow the user
    # to pass either MM/DD/YYYY directly or YYYY-MM-DD and we normalize.
    def _normalize_date(d: str | None) -> str:
        if not d:
            return ""
        d = d.strip()
        # If it's already in MM/DD/YYYY, pass through.
        if "/" in d:
            return d
        # Try to parse YYYY-MM-DD and convert.
        try:
            parsed = datetime.strptime(d, "%Y-%m-%d")
            return parsed.strftime("%m/%d/%Y")
        except ValueError:
            # If parsing fails, just return as-is and let the API complain.
            return d

    date_from_norm = _normalize_date(date_from)
    date_to_norm = _normalize_date(date_to)

    gl = LeagueGameLog(
        counter=counter,
        direction=direction,
        league_id=league_id,
        player_or_team_abbreviation="T",
        season=season,
        season_type_all_star=season_type,
        sorter="DATE",
        date_from_nullable=date_from_norm,
        date_to_nullable=date_to_norm,
    )
    df = gl.league_game_log.get_data_frame()

    # Normalize team abbreviation to uppercase strings
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str).str.upper()

    if team_abbrev:
        team_abbrev = team_abbrev.upper()
        df = df[df["TEAM_ABBREVIATION"] == team_abbrev].copy()

    return df


def get_schedule_league_v2(season: str, league_id: str = "00") -> pd.DataFrame:
    """
    Fetch the full NBA schedule for a given season via ScheduleLeagueV2.

    Parameters
    ----------
    season : str
        Season string in 'YYYY-YY' format, e.g. '2025-26'.
    league_id : str
        League identifier, default '00' for NBA.

    Returns
    -------
    pd.DataFrame
        The SeasonGames dataframe from ScheduleLeagueV2, with columns like:
        ['leagueId', 'seasonYear', 'gameDate', 'gameId', 'homeTeam_teamTricode',
         'awayTeam_teamTricode', ...].
    """
    endpoint = ScheduleLeagueV2(season=season, league_id=league_id)
    dataset = endpoint.season_games
    if dataset is None:
        print("⚠️ ScheduleLeagueV2 returned no SeasonGames dataset.")
        return pd.DataFrame()

    season_games = dataset.get_data_frame()
    return season_games


def get_schedule_for_date(
    date_str: str, season_label: str, league_id: str = "00"
) -> pd.DataFrame:
    """
    Fetch the schedule for a single calendar date via ScheduleLeagueV2.

    Parameters
    ----------
    date_str : str
        Calendar date in 'YYYY-MM-DD' format (e.g. '2025-12-17').
    season_label : str
        Season string in 'YYYY-YY' format, e.g. '2025-26'.
        This should already match nba_api's expected format.
    league_id : str
        League identifier, default '00' for NBA.

    Returns
    -------
    pd.DataFrame
        One row per scheduled game on that date, based on the SeasonGames
        dataset from ScheduleLeagueV2. Typical columns include:
          - gameId
          - gameDate
          - homeTeam_teamTricode / awayTeam_teamTricode
          - plus other metadata from ScheduleLeagueV2.
    """
    # Reuse the full-season helper and then filter by date in Python.
    season_games = get_schedule_league_v2(season=season_label, league_id=league_id)
    if season_games.empty:
        print(
            f"⚠️ ScheduleLeagueV2 returned empty SeasonGames for "
            f"season {season_label} / league_id={league_id}."
        )
        return season_games

    df = season_games.copy()

    # Normalize a canonical GAME_DATE column as datetime64[ns]
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    elif "GAME_DATE_EST" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE_EST"])
    elif "gameDate" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["gameDate"])
    else:
        # Fallback: trust the input date_str for GAME_DATE
        df["GAME_DATE"] = pd.to_datetime(date_str)

    # Compare on calendar date only (no time component) without using .dt
    target_ts = pd.to_datetime(date_str)
    start_of_day = target_ts.normalize()
    end_of_day = start_of_day + pd.Timedelta(days=1)

    mask = (df["GAME_DATE"] >= start_of_day) & (df["GAME_DATE"] < end_of_day)
    day_games = df.loc[mask].copy()
    return day_games
