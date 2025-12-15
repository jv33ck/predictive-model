from features.season_player_aggregate import build_player_profile_for_team_season


def main():
    team_abbrev = "ATL"
    max_games = 5

    profiles = build_player_profile_for_team_season(team_abbrev, max_games=max_games)

    print("\n📐 Profile columns:")
    print(profiles.columns.tolist())

    cols_to_show = [
        "team",
        "sportsdata_team_id",
        "player_id",
        "sportsdata_player_id",
        "player_name",
        "games_played",
        "minutes_played",
        "minutes_per_game",
        "total_possessions",
        "possessions_per_game",
        "pts_per_game",
        "pts_per_36",
        "off_rating_per_100",
        "net_rating_on_minus_off",
        "ts_pct",
        "usg_pct",
    ]
    existing = [c for c in cols_to_show if c in profiles.columns]

    print(profiles[existing].sort_values("minutes_played", ascending=False).head(10))


if __name__ == "__main__":
    main()
