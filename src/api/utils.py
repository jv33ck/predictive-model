# src/api/utils.py

from functools import lru_cache
from api.client import NBAApiClient

client = NBAApiClient()


@lru_cache(maxsize=1)
def get_current_season() -> str:
    """
    Calls SportsDataIO's CurrentSeason endpoint and returns the season year as a string.
    Caches result for the duration of the script using functools.lru_cache.
    """
    endpoint = "/scores/json/CurrentSeason"
    response = client.get(endpoint)

    if isinstance(response, dict) and "Season" in response:
        return str(response["Season"])
    else:
        raise ValueError("Could not determine current season from API response.")


@lru_cache(maxsize=1)
def get_current_season_year() -> int:
    """
    Uses SportsDataIO's CurrentSeason endpoint to return the current NBA season year.
    Example: 2023 (meaning the 2023-24 season)
    """
    endpoint = "/scores/json/CurrentSeason"
    response = client.get(endpoint)

    if isinstance(response, dict) and "Season" in response:
        return int(response["Season"])
    else:
        raise ValueError("Could not determine current season from API response.")


@lru_cache(maxsize=1)
def get_current_nba_api_season() -> str:
    """
    Returns the current season in nba_api format, e.g. '2023-24'.
    """
    year = get_current_season_year()
    return f"{year}-{str(year + 1)[-2:]}"
