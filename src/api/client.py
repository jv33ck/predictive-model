# src/api/client.py
import os
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()


class NBAApiClient:
    def __init__(self):
        self.api_key = os.getenv("NBA_API_KEY")
        if not self.api_key:
            raise ValueError("NBA_API_KEY not found in environment.")

        self.base_url = (
            "https://api.sportsdata.io/v3/nba"  # Replace with actual base URL
        )
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        params = params or {}
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code != 200:
            raise Exception(
                f"API request failed: {response.status_code} – {response.text}"
            )

        return response.json()
