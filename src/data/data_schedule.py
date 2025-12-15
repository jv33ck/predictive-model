# src/data/data_schedule.py
from api.client import NBAApiClient
from typing import Tuple
import pandas as pd
from datetime import date

client = NBAApiClient()


def fetch_games_by_date(target_date: str) -> pd.DataFrame:
    """
    Get all NBA games scheduled for a given date (YYYY-MM-DD).
    """
    endpoint = f"/scores/json/GamesByDate/{target_date}"
    games = client.get(endpoint)
    return pd.DataFrame(games)


def get_teams_playing_today() -> Tuple[list[str], pd.DataFrame]:
    today_str = date.today().strftime("%Y-%m-%d")
    games_df = fetch_games_by_date(today_str)

    teams = set()
    for _, row in games_df.iterrows():
        teams.add(row["HomeTeam"])
        teams.add(row["AwayTeam"])

    return sorted(list(teams)), games_df
