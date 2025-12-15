from data_schedule import get_teams_playing_today
from data_playbyplay import fetch_team_schedule, fetch_play_by_play
import os
import json
from time import sleep
from api.utils import get_current_season


def main():
    teams, _ = get_teams_playing_today()
    season = get_current_season()  # always pull the current season

    for team in teams:
        print(f"\n📅 Fetching schedule for: {team}")
        schedule_df = fetch_team_schedule(season=season, team=team)

        team_dir = os.path.abspath(
            os.path.join(__file__, "../../", "../../data/playbyplay", team)
        )
        os.makedirs(team_dir, exist_ok=True)

        for _, row in schedule_df.iterrows():
            game_id = row["GameID"]
            output_path = os.path.join(team_dir, f"game_{game_id}.json")

            if os.path.exists(output_path):
                print(f"✅ Already downloaded: {game_id}")
                continue

            print(f"📡 Fetching PBP for GameID {game_id}...")
            try:
                pbp = fetch_play_by_play(game_id)
                with open(output_path, "w") as f:
                    json.dump(pbp, f, indent=2)
                print(f"📝 Saved: {output_path}")
                sleep(1.5)  # Be respectful of API rate limits
            except Exception as e:
                print(f"❌ Failed to fetch GameID {game_id}: {e}")


if __name__ == "__main__":
    main()
