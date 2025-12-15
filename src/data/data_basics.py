# src/data/data_basics.py
from api.client import NBAApiClient
import pandas as pd

client = NBAApiClient()


def fetch_teams(season: str) -> pd.DataFrame:
    """
    Fetch team metadata for a specific season.
    Season format: 'YYYYREG', 'YYYYPRE', 'YYYYPOST'
    Example: '2023REG'
    """
    endpoint = f"/scores/json/teams/{season}"
    data = client.get(endpoint)
    return pd.DataFrame(data)


def fetch_schedule(season: str) -> pd.DataFrame:
    """
    Fetch game schedule for a specific season.
    """
    endpoint = f"/scores/json/Games/{season}"
    data = client.get(endpoint)
    return pd.DataFrame(data)


def fetch_standings(season: str) -> pd.DataFrame:
    """
    Fetch team standings for a specific season.
    """
    endpoint = f"/scores/json/Standings/{season}"
    data = client.get(endpoint)
    return pd.DataFrame(data)
