from features.season_player_aggregate import compute_player_stats_for_team_season


def main():
    team_abbrev = "ATL"
    max_games = 5  # keep small for testing; set to None for full season

    print(
        f"📊 Computing season player stats for {team_abbrev} (max_games={max_games})..."
    )
    season_stats = compute_player_stats_for_team_season(
        team_abbrev=team_abbrev,
        max_games=max_games,
    )

    cols = [
        "team",
        "player_id",
        "player_name",
        "games_played",
        "total_possessions",
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
        "off_rating_off_per_100",
        "def_rating_off_per_100",
        "net_rating_off_per_100",
        "off_rating_on_minus_off",
        "def_rating_on_minus_off",
        "net_rating_on_minus_off",
    ]
    existing_cols = [c for c in cols if c in season_stats.columns]

    print(
        season_stats[existing_cols]
        .sort_values("total_possessions", ascending=False)
        .head(15)
    )

    print("\n📐 Season stats columns:")
    print(season_stats.columns.tolist())

    print("\n🔎 Season player stats (sorted by total_possessions):")
    cols = [
        "team",
        "player_id",
        "player_name",
        "games_played",
        "minutes_played",
        "minutes_per_game",
        "total_possessions",
        "pts",
        "pts_per_game",
        "pts_per_36",
        "pts_per_100_poss",
        "fgm",
        "fga",
        "fg_pct",
        "fg3m",
        "fg3a",
        "three_pct",
        "ftm",
        "fta",
        "ft_pct",
        "reb_pct",
        "ast",
        "ast_pct",
        "stl",
        "blk",
        "tov",
        "pf",
        "off_rating_per_100",
        "def_rating_per_100",
        "net_rating_per_100",
        "net_rating_on_minus_off",
        "ts_pct",
        "efg_pct",
        "usg_pct",
    ]
    existing_cols = [c for c in cols if c in season_stats.columns]

    print(
        season_stats[existing_cols]
        .sort_values("total_possessions", ascending=False)
        .head(20)
    )


if __name__ == "__main__":
    main()
