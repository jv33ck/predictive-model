#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import json

from db.player_stats_db import get_connection

# ---------- Helpers to make schema detection robust ----------


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    logical_name: str,
    required: bool = True,
) -> str | None:
    """
    Try to find a column in df whose name matches one of the candidate names.
    Returns the first match or None if not found and required=False.
    If required=True and none found, raises a helpful RuntimeError.
    """
    for cand in candidates:
        if cand in df.columns:
            return cand

    if required:
        raise RuntimeError(
            f"Could not find column for '{logical_name}'. "
            f"Tried candidates={list(candidates)}; available={list(df.columns)}"
        )
    return None


def normalize_core_columns(
    df: pd.DataFrame,
    season_label: str | None,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    """
    Normalize / standardize key columns in player_game_stats:

      - game_date (datetime64[ns])
      - game_id
      - team
      - opponent (if available)
      - is_home (if derivable)
      - player_id
      - player_name

    Also filters by season_label if a matching column exists, and by date range.
    """

    df = df.copy()

    # --- Filter by season_label if possible ---
    if season_label is not None:
        season_col = find_column(
            df,
            candidates=["season_label", "season", "SEASON", "SEASON_ID"],
            logical_name="season",
            required=False,
        )
        if season_col is not None:
            df = df[df[season_col] == season_label]

    # --- Detect / normalize date column ---
    date_col = find_column(
        df,
        candidates=["game_date", "GAME_DATE", "GAME_DATE_EST", "GAME_DATE_LCL", "date"],
        logical_name="game_date",
        required=False,
    )

    if date_col is not None:
        # Use the real date column and respect the CLI date range.
        df["game_date"] = pd.to_datetime(df[date_col]).dt.normalize()

        if start_date is not None:
            df = df[df["game_date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["game_date"] <= pd.Timestamp(end_date)]
    else:
        # Fallback: no explicit dates stored in player_game_stats.
        # We still need a stable per-player game ordering for rolling features,
        # so we create a synthetic "game_date" based on game_id order and
        # skip calendar-based filtering.
        base_ts = pd.Timestamp("2000-01-01")
        # factorize assigns 0,1,2,... in the order the game_ids appear
        order_codes, _ = pd.factorize(df["game_id"].astype(str))
        df["game_date"] = base_ts + pd.to_timedelta(order_codes, unit="D")

        # Optional: small log to make the behavior obvious when script runs.
        print(
            "⚠️ No explicit game_date column found in player_game_stats; "
            "using synthetic dates derived from game_id order and ignoring "
            "start/end date filters."
        )

    # --- game_id ---
    game_id_col = find_column(
        df,
        candidates=["game_id", "GAME_ID"],
        logical_name="game_id",
        required=True,
    )
    if game_id_col != "game_id":
        df = df.rename(columns={game_id_col: "game_id"})

    # --- team abbreviation ---
    team_col = find_column(
        df,
        candidates=["team_abbreviation", "TEAM_ABBREVIATION", "team"],
        logical_name="team",
        required=True,
    )
    if team_col != "team":
        df = df.rename(columns={team_col: "team"})

    # --- opponent abbreviation (optional) ---
    opp_col = find_column(
        df,
        candidates=[
            "opponent_abbreviation",
            "OPPONENT_TEAM_ABBREVIATION",
            "OPP_TEAM_ABBREVIATION",
            "opp_team",
            "OPPONENT",
        ],
        logical_name="opponent",
        required=False,
    )
    if opp_col is not None and opp_col != "opponent":
        df = df.rename(columns={opp_col: "opponent"})

    # --- home/away flag (optional) ---
    # Many pipelines derive this from a 'matchup' string like "CHI vs. CLE" / "CHI @ CLE".
    ha_col = find_column(
        df,
        candidates=["home_away", "HOME_AWAY", "is_home"],
        logical_name="is_home",
        required=False,
    )
    if ha_col is not None:
        # Normalize to boolean is_home
        vals = df[ha_col].astype(str).str.lower()
        df["is_home"] = vals.isin(["home", "h", "1", "true", "vs", "vs."])
    else:
        # Try to infer from a matchup string if present
        matchup_col = find_column(
            df,
            candidates=["MATCHUP", "matchup"],
            logical_name="matchup",
            required=False,
        )
        if matchup_col is not None:
            m = df[matchup_col].astype(str)
            # Convention from NBA stats: "CHI vs. CLE" (home), "CHI @ CLE" (away)
            df["is_home"] = m.str.contains("vs.", case=False)
        else:
            df["is_home"] = np.nan  # no home/away info

    # --- player id/name ---
    player_id_col = find_column(
        df,
        candidates=["player_id", "PLAYER_ID"],
        logical_name="player_id",
        required=True,
    )
    if player_id_col != "player_id":
        df = df.rename(columns={player_id_col: "player_id"})

    player_name_col = find_column(
        df,
        candidates=["player_name", "PLAYER_NAME"],
        logical_name="player_name",
        required=False,
    )
    if player_name_col is not None and player_name_col != "player_name":
        df = df.rename(columns={player_name_col: "player_name"})

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_cols: list[str],
    minutes_col: str,
    recent_n: int,
) -> pd.DataFrame:
    """
    For each player, sort games by date and add rolling / prior stats:

      - games_before (number of games before this one)
      - season-to-date averages for targets and minutes (before this game)
      - last N (recent_n) game averages for targets and minutes (before this game)
      - days_since_prev_game
    """

    df = df.copy()
    # Ensure game_date is datetime64 for diffs:
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()

    group_cols = ["player_id"]
    dfs: list[pd.DataFrame] = []

    for pid, g in df.groupby(group_cols, as_index=False, sort=False):
        g = g.sort_values("game_date").copy()

        # index within player
        g["games_before"] = np.arange(len(g), dtype=int)

        # days since previous game
        deltas = pd.to_timedelta(g["game_date"].diff())
        g["days_since_prev_game"] = deltas.dt.days.astype("float32")

        # For each target, compute season-to-date and recent-N prior averages
        for col in target_cols + [minutes_col]:
            # Make sure column exists; if not, raise clean error
            if col not in g.columns:
                raise RuntimeError(
                    f"Expected column '{col}' in player_game_stats but it was missing. "
                    f"Available columns for player_id={pid} are: {list(g.columns)}"
                )

            # shift(1) so the current game's value is NOT included
            prev_vals = g[col].shift(1)

            g[f"{col}_prev_mean_all"] = prev_vals.expanding(min_periods=1).mean()
            g[f"{col}_prev_mean_{recent_n}"] = prev_vals.rolling(
                window=recent_n, min_periods=1
            ).mean()

        dfs.append(g)

    out = pd.concat(dfs, ignore_index=True)

    # Drop games where there is no prior info at all (first game per player)
    out = out[out["games_before"] > 0].reset_index(drop=True)

    return out


# ---------- CLI + main ----------


@dataclass
class BuildPlayerTrainConfig:
    season_label: str
    start_date: date
    end_date: date
    recent_n: int
    targets: list[str]
    output_csv: str | None
    output_json: str | None


def parse_args() -> BuildPlayerTrainConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-player-per-game training dataset from player_game_stats "
            "with rolling features (season-to-date + recent-N)."
        )
    )

    parser.add_argument(
        "--season-label",
        required=True,
        help="Season label, e.g. '2025-26'. Used for filtering if a season column exists.",
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date for games to include (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="End date for games to include (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--recent-n",
        type=int,
        default=5,
        help="Number of recent games for rolling averages (default: 5).",
    )

    parser.add_argument(
        "--targets",
        type=str,
        default="pts,treb,ast",
        help=(
            "Comma-separated list of target stat columns in player_game_stats "
            "(default: 'pts,treb,ast'). 'reb' will be mapped to 'treb' if used."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to write the training dataset as CSV.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to write the training dataset as JSON (records).",
    )

    args = parser.parse_args()

    start_d = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    # Map common aliases to actual DB column names
    alias_map: dict[str, str] = {
        "reb": "treb",  # total rebounds
    }
    resolved_targets: list[str] = []
    for t in targets:
        key = t.lower()
        mapped = alias_map.get(key, t)
        # Pylance may think mapped could be None; guard explicitly.
        if mapped is None:
            mapped = t
        if mapped != t:
            print(f"ℹ️ Mapping target '{t}' to '{mapped}' based on DB schema.")
        resolved_targets.append(mapped)

    return BuildPlayerTrainConfig(
        season_label=args.season_label,
        start_date=start_d,
        end_date=end_d,
        recent_n=args.recent_n,
        targets=resolved_targets,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )


def main() -> None:
    cfg = parse_args()

    print("===============================================")
    print("🏗️  Building player training dataset")
    print("===============================================")
    print(f"📆 Season label: {cfg.season_label}")
    print(f"📅 Date range:  {cfg.start_date} → {cfg.end_date}")
    print(f"📈 Recent-N:    {cfg.recent_n}")
    print(f"🎯 Targets:     {cfg.targets}")
    print("===============================================")

    # --- Load raw player_game_stats ---
    conn = get_connection()
    try:
        raw_df = pd.read_sql_query("SELECT * FROM player_game_stats", conn)
    finally:
        conn.close()

    if raw_df.empty:
        raise RuntimeError("player_game_stats table is empty; nothing to build.")

    print(f"📥 Loaded {len(raw_df)} raw player-game rows from player_game_stats.")

    # Normalize schema & filter
    df = normalize_core_columns(
        raw_df,
        season_label=cfg.season_label,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
    )

    if df.empty:
        raise RuntimeError(
            f"No player_game_stats rows remain after filtering for "
            f"season_label={cfg.season_label}, start_date={cfg.start_date}, "
            f"end_date={cfg.end_date}."
        )

    print(f"📊 After season/date filtering: {len(df)} rows.")

    # Detect minutes column
    minutes_col = find_column(
        df,
        candidates=["minutes", "MIN", "min", "minutes_played"],
        logical_name="minutes",
        required=True,
    )
    if minutes_col is None:
        # This should not happen because required=True above, but keep a defensive check
        raise RuntimeError("Failed to detect minutes column in player_game_stats.")

    # Make sure targets exist
    missing_targets = [t for t in cfg.targets if t not in df.columns]
    if missing_targets:
        raise RuntimeError(
            "Some target columns are missing from player_game_stats.\n"
            f"  Requested targets: {cfg.targets}\n"
            f"  Missing targets:   {missing_targets}\n"
            f"  Available columns: {list(df.columns)}"
        )

    # Add rolling features
    df_features = add_rolling_features(
        df=df,
        target_cols=cfg.targets,
        minutes_col=minutes_col,
        recent_n=cfg.recent_n,
    )

    print(f"✅ Built feature dataset with {len(df_features)} rows.")
    print("   Sample columns:")
    print(sorted(df_features.columns)[:40])

    # Persist
    print(f"💾 Writing training CSV to: {cfg.output_csv}")
    df_features.to_csv(cfg.output_csv, index=False)

    print(f"💾 Writing training JSON to: {cfg.output_json}")

    # Make a copy for JSON export and normalize any datetime columns
    df_for_json = df_features.copy()
    # Make a copy for JSON export and normalize any datetime columns
    df_for_json = df_features.copy()

    # Normalize game_date to a string if it's datetime-like
    if "game_date" in df_for_json.columns and is_datetime64_any_dtype(
        df_for_json["game_date"]
    ):
        # Convert datetimes to plain YYYY-MM-DD strings in a way that's
        # friendly to static type checkers (avoid .dt.strftime).
        df_for_json["game_date"] = df_for_json["game_date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if not pd.isna(x) else None
        )

    def _coerce_for_json(obj: object) -> object:
        """Coerce values into JSON-serializable Python types."""
        # Unbox numpy scalar types
        if isinstance(obj, np.generic):
            obj = obj.item()

        # Decode any bytes/bytearray into UTF-8 strings (with replacement for bad bytes)
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="replace")
            except Exception:
                return repr(obj)

        return obj

    # Convert to plain Python objects and coerce any non-JSON-serializable
    # values (e.g., numpy scalars, bytes) into basic Python types.
    raw_records = df_for_json.to_dict(orient="records")
    records = [{k: _coerce_for_json(v) for k, v in rec.items()} for rec in raw_records]

    # cfg.output_json is declared as str | None; enforce non-None for the type checker
    if cfg.output_json is None:
        raise RuntimeError("output_json must be provided when running this script.")

    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    # Quick head print for sanity
    print("\n🔎 Sample rows:")
    print(df_features.head(10).to_string(index=False))

    print("\n🎉 Player training dataset export complete.")


if __name__ == "__main__":
    main()
