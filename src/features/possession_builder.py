# src/features/possession_builder.py
from __future__ import annotations

from typing import List

import pandas as pd


def build_scoring_possessions_for_game(
    events_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """
    Build a simple scoring-possession table from a normalized PBP DataFrame
    for a single game.

    Args:
        events_df: normalized PBP (output of normalize_pbp_v3) for ONE game_id
        home_team: team abbreviation for the home team (e.g. 'ATL')
        away_team: team abbreviation for the away team (e.g. 'TOR')

    Returns:
        DataFrame with one row per *scoring* possession, columns like:
        [
            'game_id', 'possession_index', 'period', 'clock', 'seconds_remaining',
            'offense_team', 'defense_team',
            'points', 'points_home', 'points_away',
            'start_score_home', 'start_score_away',
            'end_score_home', 'end_score_away',
            'action_type', 'action_subtype', 'event_number',
        ]
    """
    if events_df.empty:
        return pd.DataFrame()

    # Make sure we only have one game_id in this DF
    game_ids = events_df["game_id"].unique()
    if len(game_ids) != 1:
        raise ValueError(f"Expected events_df for a single game_id, found {game_ids}.")

    # Work only with scoring events for now
    scoring = events_df[events_df["is_scoring_event"]].copy()

    if scoring.empty:
        return pd.DataFrame()

    # Sort so possessions are in chronological order within the game
    sort_cols = [
        c
        for c in ["period", "seconds_remaining", "event_number"]
        if c in scoring.columns
    ]
    if sort_cols:
        scoring = scoring.sort_values(
            sort_cols, ascending=[True, False, True]
        ).reset_index(drop=True)

    possession_rows: List[dict] = []

    possession_index = 0
    for _, row in scoring.iterrows():
        offense_team = row["team"]

        # Skip weird events with no team or unknown team
        if not isinstance(offense_team, str):
            continue

        offense_team = offense_team.upper()
        home_team_u = home_team.upper()
        away_team_u = away_team.upper()

        if offense_team == home_team_u:
            defense_team = away_team_u
        elif offense_team == away_team_u:
            defense_team = home_team_u
        else:
            # Neutral events or bad data — skip for now
            continue

        # Compute score at start/end of this scoring possession
        # We already computed points_home / points_away in normalizer
        points_home = float(row.get("points_home", 0.0))
        points_away = float(row.get("points_away", 0.0))

        score_home_end = int(row.get("score_home", 0))
        score_away_end = int(row.get("score_away", 0))

        score_home_start = score_home_end - int(points_home)
        score_away_start = score_away_end - int(points_away)

        possession_rows.append(
            {
                "game_id": row["game_id"],
                "possession_index": possession_index,
                "period": int(row["period"]),
                "clock": row.get("clock"),
                "seconds_remaining": row.get("seconds_remaining"),
                "offense_team": offense_team,
                "defense_team": defense_team,
                "points": float(row.get("points", 0.0)),
                "points_home": points_home,
                "points_away": points_away,
                "start_score_home": score_home_start,
                "start_score_away": score_away_start,
                "end_score_home": score_home_end,
                "end_score_away": score_away_end,
                "action_type": row.get("action_type"),
                "action_subtype": row.get("action_subtype"),
                "event_number": int(row["event_number"]),
            }
        )

        possession_index += 1

    return pd.DataFrame(possession_rows)


def build_all_possessions_for_game(
    events_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """
    Build an all-possessions table (including non-scoring trips) from a
    normalized PBP DataFrame for ONE game.

    This is a first-pass possession model using simple rules:
      - A possession belongs to the team in control of the ball (inferred
        from events).
      - A possession ENDS when:
          * the offense scores (made shot or made FTs)
          * the offense commits a turnover
          * a missed shot is rebounded by the defense
          * the period changes (end of quarter)
      - Offensive rebounds KEEP the same possession (continuation).

    Args:
        events_df:
            Normalized PBP for ONE game_id (output of normalize_pbp_v3),
            with at least:
                - game_id
                - event_number
                - period
                - seconds_remaining
                - team (event team, e.g. 'ATL' or 'TOR')
                - points_home, points_away, points
                - is_scoring_event
                - action_type
                - action_subtype
        home_team:
            Home team abbreviation, e.g. 'ATL'
        away_team:
            Away team abbreviation, e.g. 'TOR'

    Returns:
        DataFrame with one row per possession, columns:
            - game_id
            - possession_index
            - period
            - seconds_remaining (at end of possession)
            - offense_team
            - defense_team
            - points
    """
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()

    # Make sure team abbreviations are upper-case
    home_team = home_team.upper()
    away_team = away_team.upper()

    # Sort events in game order: period ASC, clock DESC, then event_number ASC
    sort_cols = [
        c for c in ["period", "seconds_remaining", "event_number"] if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True, False, True]).reset_index(
            drop=True
        )

    # Helper functions to classify events
    def _get_team(row) -> str | None:
        t = row.get("team")
        return str(t).upper() if isinstance(t, str) and t.strip() else None

    def _is_shot(row) -> bool:
        # Field goal attempts
        if bool(row.get("is_field_goal")):
            return True
        sub = str(row.get("action_subtype") or "").lower()
        return "shot" in sub and "free throw" not in sub

    def _is_free_throw(row) -> bool:
        sub = str(row.get("action_subtype") or "").lower()
        return "free throw" in sub

    def _is_rebound(row) -> bool:
        at = str(row.get("action_type") or "").lower()
        sub = str(row.get("action_subtype") or "").lower()
        desc = str(row.get("event_description") or "").lower()
        return "rebound" in at or "rebound" in sub or "rebound" in desc

    def _is_turnover(row) -> bool:
        at = str(row.get("action_type") or "").lower()
        sub = str(row.get("action_subtype") or "").lower()
        desc = str(row.get("event_description") or "").lower()
        return "turnover" in at or "turnover" in sub or "turnover" in desc

    possessions: List[dict] = []

    game_ids = df["game_id"].unique()
    if len(game_ids) != 1:
        raise ValueError(f"Expected events_df for a single game, found {game_ids}.")
    game_id = game_ids[0]

    current_offense: str | None = None
    current_defense: str | None = None
    current_period: int | None = None
    current_points: float = 0.0

    # We track last shot team & result, to decide about defensive rebounds
    last_shot_team: str | None = None
    last_shot_missed: bool | None = None

    # We use the event's time as the end-of-possession time
    last_event_seconds: float | None = None
    last_event_period: int | None = None
    last_event_number: int | None = None

    def _other_team(team: str) -> str:
        team = team.upper()
        if team == home_team:
            return away_team
        elif team == away_team:
            return home_team
        return home_team  # fallback

    def _close_possession(end_row: pd.Series | None) -> None:
        nonlocal current_offense, current_defense, current_points
        nonlocal last_event_seconds, last_event_period, last_event_number

        if current_offense is None or current_defense is None:
            # Nothing to close
            current_points = 0.0
            return

        if end_row is not None:
            period = int(end_row["period"])
            sec = float(end_row.get("seconds_remaining") or 0.0)
            evt_num = int(end_row["event_number"])
        else:
            # Fallback to last tracked event info
            period = int(last_event_period or 1)
            sec = float(last_event_seconds or 0.0)
            evt_num = int(last_event_number or 0)

        possession_index = len(possessions)
        possessions.append(
            {
                "game_id": game_id,
                "possession_index": possession_index,
                "period": period,
                "seconds_remaining": sec,
                "offense_team": current_offense,
                "defense_team": current_defense,
                "points": current_points,
            }
        )

        # Reset possession state
        current_offense = None
        current_defense = None
        current_points = 0.0

    for _, row in df.iterrows():
        period = int(row["period"])
        sec_rem = row.get("seconds_remaining")
        sec_rem = float(sec_rem) if sec_rem is not None else 0.0
        evt_num = int(row["event_number"])
        team = _get_team(row)

        is_scoring = bool(row.get("is_scoring_event"))
        pts_home = float(row.get("points_home") or 0.0)
        pts_away = float(row.get("points_away") or 0.0)
        pts_total = float(row.get("points") or 0.0)

        shot = _is_shot(row)
        ft = _is_free_throw(row)
        rebound = _is_rebound(row)
        turnover = _is_turnover(row)

        # Track last event time for fallback
        last_event_seconds = sec_rem
        last_event_period = period
        last_event_number = evt_num

        # Handle period change (end of quarter closes any open possession)
        if current_period is not None and period != current_period:
            _close_possession(row)
        current_period = period

        # If no possession is active yet, start one when we see a team event
        if (
            current_offense is None
            and team is not None
            and (shot or ft or turnover or is_scoring)
        ):
            current_offense = team
            current_defense = _other_team(team)
            current_points = 0.0

        # Update last shot info
        if shot:
            last_shot_team = team
            last_shot_missed = not is_scoring

        # Free throws are trickier, but for now we treat them as part of the
        # ongoing possession and let scoring / rebounds / turnovers decide ends.
        if ft:
            # If we somehow didn't have an offense yet, assume FT shooter team
            if current_offense is None and team is not None:
                current_offense = team
                current_defense = _other_team(team)
            # Last "shot" being a FT doesn't affect rebound logic as much,
            # but we still note if it was missed.
            last_shot_team = team
            last_shot_missed = not is_scoring

        # Scoring: assign points to offense if team matches
        if is_scoring and team is not None:
            # Ensure we have an offense team
            if current_offense is None:
                current_offense = team
                current_defense = _other_team(team)

            # Determine how many points this team scored
            if team == home_team:
                pts_for_team = pts_home
            elif team == away_team:
                pts_for_team = pts_away
            else:
                pts_for_team = pts_total  # fallback

            if team == current_offense:
                current_points += pts_for_team
            else:
                # Defense scoring unexpectedly (likely a steal+fastbreak we
                # didn't detect); treat as new possession starting here.
                _close_possession(row)
                current_offense = team
                current_defense = _other_team(team)
                current_points = pts_for_team

            # End possession on scoring; next offensive event will start a new one
            _close_possession(row)
            continue

        # Turnover: end possession for the offense
        if turnover and team is not None:
            if current_offense is None:
                # Treat this as a 1-event empty possession for 'team'
                current_offense = team
                current_defense = _other_team(team)
                current_points = 0.0

            _close_possession(row)
            # Next possession will start when the new offense team does something
            continue

        # Rebounds: if last shot was missed and this rebound is by the OTHER team,
        # treat as a defensive rebound -> change of possession
        if (
            rebound
            and team is not None
            and last_shot_team is not None
            and last_shot_missed
        ):
            if team.upper() != last_shot_team.upper():
                # Defensive rebound: end last_shot_team's possession
                if current_offense is None:
                    current_offense = last_shot_team
                    current_defense = _other_team(last_shot_team)
                _close_possession(row)
                # Next possession will start when new offense acts
                # (we don't force-start it here to avoid double-counting)
                continue

        # Otherwise, continue within current possession (if any)

    # End of loop: close any open possession at last event
    if current_offense is not None and current_defense is not None:
        _close_possession(None)

    return pd.DataFrame(possessions)
