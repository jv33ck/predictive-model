# src/data/data_playbyplay.py
from api.client import NBAApiClient
import pandas as pd

client = NBAApiClient()


def fetch_team_schedule(season: str, team: str) -> pd.DataFrame:
    """
    Fetches only completed games ("Final") from a team's schedule for a given season.
    """
    endpoint = f"/scores/json/SchedulesBasic/{season}"
    all_games = client.get(endpoint)
    df = pd.DataFrame(all_games)

    # Filter to games involving this team
    team_games = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

    # ✅ Only include games that have been completed
    final_games = team_games[team_games["Status"].isin(["Final", "F/OT"])]

    return final_games


def fetch_play_by_play(game_id) -> dict:
    """
    Fetches final play-by-play data for a single NBA game.
    """
    endpoint = f"/pbp/json/PlayByPlayFinal/{game_id}"
    return client.get(endpoint)
