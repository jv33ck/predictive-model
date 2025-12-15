from data_basics import fetch_teams, fetch_schedule, fetch_standings
import os
from api.utils import get_current_season

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

os.makedirs(DATA_DIR, exist_ok=True)


def preview_df(df, name):
    print(f"\n✅ {name} loaded.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(5))


def main():
    season = get_current_season()

    print(f"📡 Fetching NBA metadata for season: {season}\n")

    teams_df = fetch_teams(season)
    schedule_df = fetch_schedule(season)
    standings_df = fetch_standings(season)

    preview_df(teams_df, "Teams")
    preview_df(schedule_df, "Schedule")
    preview_df(standings_df, "Standings")

    teams_df.to_csv(os.path.join(DATA_DIR, "teams.csv"), index=False)
    schedule_df.to_csv(os.path.join(DATA_DIR, "schedule.csv"), index=False)
    standings_df.to_csv(os.path.join(DATA_DIR, "standings.csv"), index=False)
    print("\n📁 Synced and saved: teams.csv, schedule.csv, standings.csv")


if __name__ == "__main__":
    main()
