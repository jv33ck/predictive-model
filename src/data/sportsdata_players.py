# src/data/sportsdata_players.py
import pandas as pd

from api.client import NBAApiClient  # same client you used for teams/schedule/etc.


def get_active_players_basic() -> pd.DataFrame:
    """
    Fetch active player basics from SportsData.io.

    Endpoint (documented):
      GET /v3/nba/scores/json/PlayersActiveBasic

    Returns:
        DataFrame with columns including:
          - PlayerID (SportsData player id)
          - Team (team abbreviation, e.g. BOS)
          - TeamID (SportsData team id)
          - FirstName, LastName
          - Position, Jersey, etc.
    """
    client = NBAApiClient()
    endpoint = "scores/json/PlayersActiveBasic"
    data = client.get(endpoint)
    return pd.DataFrame(data)
