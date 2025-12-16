# src/features/team_schedule_features.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import pandas as pd

from data.nba_api_provider import get_leaguegamelog_team


@dataclass
class TeamScheduleSummary:
    """
    Compact summary of a team's schedule & results for a season
    up to an optional cutoff date.
    """

    team: str
    season: str

    games_played: int
    wins: int
    losses: int
    win_pct: float

    pts_for_total: float
    pts_for_per_game: float

    plus_minus_total: float
    plus_minus_per_game: float

    # Recent form (last N games, default N=5)
    recent_games: int
    recent_wins: int
    recent_losses: int
    recent_plus_minus_total: float
    recent_plus_minus_per_game: float


def _coerce_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Safely coerce a column to numeric, returning a Series.
    Non-convertible values become NaN.
    """
    return pd.to_numeric(df[column], errors="coerce")


def compute_team_schedule_summary(
    season_label: str,
    team: str,
    date_cutoff: Optional[str] = None,
    season_type: str = "Regular Season",
    recent_n: int = 5,
) -> TeamScheduleSummary:
    """
    Build a team schedule summary from LeagueGameLog.

    Parameters
    ----------
    season_label : str
        NBA season label, e.g. '2025-26'.
    team : str
        Team abbreviation, e.g. 'NYK'.
    date_cutoff : str, optional
        If provided, only games on or before this date are included.
        Accepts 'YYYY-MM-DD' or 'MM/DD/YYYY'.
        If None, includes all games returned by LeagueGameLog.
    season_type : str, default 'Regular Season'
        Season type for LeagueGameLog.
    recent_n : int, default 5
        Window size for "recent form" features.

    Returns
    -------
    TeamScheduleSummary
    """
    team = team.upper()

    # We use date_to to enforce an upper bound if provided.
    df = get_leaguegamelog_team(
        season=season_label,
        season_type=season_type,
        team_abbrev=team,
        date_from=None,
        date_to=date_cutoff,
    )

    if df.empty:
        raise RuntimeError(
            f"No LeagueGameLog rows found for team {team} in season {season_label} "
            f"(season_type={season_type}, date_cutoff={date_cutoff})."
        )

    # Normalize types
    df = df.copy()
    df["PTS"] = _coerce_numeric(df, "PTS")
    df["PLUS_MINUS"] = _coerce_numeric(df, "PLUS_MINUS")

    # GAME_DATE from your test is already 'YYYY-MM-DD'; still convert for safety.
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # Sort by date ascending
    df = df.sort_values("GAME_DATE")

    # Compute basic aggregates
    games_played = int(df["GAME_ID"].nunique())
    wins = int((df["WL"] == "W").sum())
    losses = int((df["WL"] == "L").sum())
    win_pct = wins / games_played if games_played > 0 else 0.0

    pts_for_total = float(df["PTS"].sum())
    pts_for_per_game = pts_for_total / games_played if games_played > 0 else 0.0

    plus_minus_total = float(df["PLUS_MINUS"].sum())
    plus_minus_per_game = plus_minus_total / games_played if games_played > 0 else 0.0

    # Recent form: last `recent_n` games
    recent_df = df.tail(recent_n)
    recent_games = int(len(recent_df))
    recent_wins = int((recent_df["WL"] == "W").sum())
    recent_losses = int((recent_df["WL"] == "L").sum())
    recent_plus_minus_total = float(recent_df["PLUS_MINUS"].sum())
    recent_plus_minus_per_game = (
        recent_plus_minus_total / recent_games if recent_games > 0 else 0.0
    )

    return TeamScheduleSummary(
        team=team,
        season=season_label,
        games_played=games_played,
        wins=wins,
        losses=losses,
        win_pct=win_pct,
        pts_for_total=pts_for_total,
        pts_for_per_game=pts_for_per_game,
        plus_minus_total=plus_minus_total,
        plus_minus_per_game=plus_minus_per_game,
        recent_games=recent_games,
        recent_wins=recent_wins,
        recent_losses=recent_losses,
        recent_plus_minus_total=recent_plus_minus_total,
        recent_plus_minus_per_game=recent_plus_minus_per_game,
    )


def build_team_schedule_features_df(
    season_label: str,
    team: str,
    date_cutoff: Optional[str] = None,
    season_type: str = "Regular Season",
    recent_n: int = 5,
) -> pd.DataFrame:
    """
    Convenience wrapper that returns a 1-row DataFrame of schedule features.

    Columns are prefixed to be easy to merge into matchup features later
    (e.g. 'team_games_played', 'team_win_pct', 'team_recent_plus_minus_per_game').
    """
    summary = compute_team_schedule_summary(
        season_label=season_label,
        team=team,
        date_cutoff=date_cutoff,
        season_type=season_type,
        recent_n=recent_n,
    )

    raw: Dict[str, Any] = asdict(summary)

    # Prefix all stats with 'team_' except team/season
    row: Dict[str, Any] = {
        "team": raw["team"],
        "season": raw["season"],
    }

    for key, value in raw.items():
        if key in ("team", "season"):
            continue
        row[f"team_{key}"] = value

    return pd.DataFrame([row])
