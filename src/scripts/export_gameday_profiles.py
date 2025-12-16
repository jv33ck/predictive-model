# src/scripts/export_gameday_profiles.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from data.nba_api_provider import get_today_games_and_teams
from db.impact_ratings_db import load_impact_ratings

try:
    from utils.s3_upload import upload_to_s3
except ImportError:  # pragma: no cover - optional dependency
    upload_to_s3 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_strings_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean string-like columns so that to_json doesn't choke on bad bytes.

    This mirrors the pattern we've used elsewhere: convert to str and then
    strip/ignore any invalid UTF-8 sequences.
    """
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(
            out[col]
        ):
            out[col] = (
                out[col]
                .astype(str)
                .apply(lambda s: s.encode("utf-8", "ignore").decode("utf-8", "ignore"))
            )

    return out


def _parse_team_list(teams_arg: Optional[str]) -> Optional[List[str]]:
    """
    Parse a comma-separated team list like 'OKC,NYK,ORL' into
    ['OKC', 'NYK', 'ORL'], or return None if teams_arg is falsy.
    """
    if not teams_arg:
        return None
    teams = [t.strip().upper() for t in teams_arg.split(",") if t.strip()]
    return sorted(set(teams))


def _get_gameday_teams(teams_arg: Optional[str]) -> List[str]:
    """
    Determine which teams to include for gameday:

      - If teams_arg is provided, use that explicitly.
      - Otherwise, call get_today_games_and_teams() from nba_api_provider.
    """
    manual = _parse_team_list(teams_arg)
    if manual:
        print(f"📅 Using manually supplied teams: {manual}")
        return manual

    # Auto-detect via nba_api
    try:
        raw_teams = get_today_games_and_teams()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to get teams playing today from nba_api: {exc}"
        ) from exc

    # Coerce to a plain list of strings for both runtime robustness and type checkers
    if isinstance(raw_teams, pd.Series):
        raw_list: list[str] = [str(x) for x in raw_teams.tolist()]
    else:
        raw_list = [str(x) for x in list(raw_teams)]

    # Normalize to sorted unique 3-letter codes
    teams_today = sorted({s.strip().upper() for s in raw_list if s.strip()})
    print(f"📅 Teams playing today (detected from nba_api): {teams_today}")
    return teams_today


def _load_season_profiles(
    season_label: str,
    profiles_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load full-season player profiles (with EPM etc.) from the CSV produced by
    export_profiles_from_db.py.

    By default, expects:
        data/exports/player_profiles_from_db_{season_label}.csv
    """
    if profiles_csv is None:
        profiles_path = (
            Path("data/exports") / f"player_profiles_from_db_{season_label}.csv"
        )
    else:
        profiles_path = Path(profiles_csv)

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Season profiles CSV not found at {profiles_path}. "
            "Make sure you've run export_profiles_from_db.py first."
        )

    print(f"📥 Loading season profiles from {profiles_path} ...")
    df = pd.read_csv(profiles_path)

    if "team" not in df.columns or "player_id" not in df.columns:
        raise ValueError(
            "Season profiles CSV is missing required 'team' or 'player_id' columns.\n"
            f"Columns present: {list(df.columns)}"
        )

    df["team"] = df["team"].astype(str).str.upper()
    df["player_id"] = df["player_id"].astype(str)

    # Keep everything; we assume export_profiles_from_db already cleaned / rounded
    print(f"✅ Loaded {len(df)} season-profile rows.")
    return df


def _maybe_attach_impact(
    profiles_df: pd.DataFrame,
    teams: Sequence[str],
    season_label: str,
    db_path: Path | str = Path("data/player_stats.db"),
) -> pd.DataFrame:
    """
    For each team, try to attach impact ratings from the SQLite DB via
    db.impact_ratings_db.load_impact_ratings(team, season).

    If no rows are found for a team, impact columns remain NaN for that team.
    """
    frames: List[pd.DataFrame] = []
    teams = [t.upper() for t in teams]
    db_path = Path(db_path)

    for team in teams:
        team_df = profiles_df[profiles_df["team"] == team].copy()
        if team_df.empty:
            print(f"⚠️ No season-profile rows for team {team}; skipping impact join.")
            continue

        try:
            impact_df = load_impact_ratings(
                team=team, season=season_label, db_path=db_path
            )
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Failed to load impact ratings from DB for team {team}: {exc}")
            frames.append(team_df)
            continue

        if impact_df.empty:
            print(
                f"⚠️ No impact ratings found in DB for team {team} / season {season_label}; "
                "impact columns will be NaN for this team."
            )
            frames.append(team_df)
            continue

        if "player_id" not in impact_df.columns:
            print(
                f"⚠️ Impact ratings table for {team} missing 'player_id' column; "
                f"columns present: {list(impact_df.columns)}"
            )
            frames.append(team_df)
            continue

        # Normalize player_id to string on both sides
        team_df["player_id"] = team_df["player_id"].astype(str)
        impact_df["player_id"] = impact_df["player_id"].astype(str)

        # Select only the impact columns we care about
        join_cols: List[str] = ["player_id"]
        for col in [
            "impact_off_per_100",
            "impact_def_per_100",
            "impact_per_100",
            "impact_off_per_possession",
            "impact_def_per_possession",
            "impact_per_possession",
            "exposure_stint_units",
        ]:
            if col in impact_df.columns:
                join_cols.append(col)

        impact_small = impact_df[join_cols].drop_duplicates("player_id")

        merged = team_df.merge(
            impact_small,
            on="player_id",
            how="left",
            validate="m:1",
        )
        frames.append(merged)

    if not frames:
        print("⚠️ No gameday rows after impact join; returning original profiles_df.")
        return profiles_df.copy()

    out = pd.concat(frames, ignore_index=True)
    print(f"✅ Attached impact metrics for gameday players (rows: {len(out)}).")
    return out


def export_gameday_profiles(
    season_label: str,
    teams_arg: Optional[str],
    profiles_csv: Optional[str],
    impact_dir: Path,  # kept for CLI compatibility; no longer used
    output_dir: Path,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
) -> Path:
    """
    Build gameday player profiles for all teams playing today (or a supplied
    subset of teams), including:

      - Precomputed season stats & EPM from export_profiles_from_db.py
      - Impact metrics pulled from the SQLite impact_ratings table.

    Output:
      - Local CSV + JSON
      - Optional upload to S3 under a static key so the backend can hard-code
        the endpoint.
    """
    # 1) Determine which teams we're exporting
    teams_today = _get_gameday_teams(teams_arg)
    if not teams_today:
        raise RuntimeError("No teams to export for gameday (empty team list).")

    # 2) Load full-season profiles (already has epm_off / epm_def / epm_net etc.)
    season_df = _load_season_profiles(
        season_label=season_label, profiles_csv=profiles_csv
    )

    gameday_df = season_df[season_df["team"].isin(teams_today)].copy()
    if gameday_df.empty:
        raise RuntimeError(
            f"No season-profile rows found for gameday teams: {teams_today}. "
            "Did you run export_profiles_from_db.py after updating the DB?"
        )

    print(f"📊 Gameday base profile rows: {len(gameday_df)}")

    # 3) Attach impact ratings from DB (data/player_stats.db by default)
    gameday_with_impact = _maybe_attach_impact(
        profiles_df=gameday_df,
        teams=teams_today,
        season_label=season_label,
        db_path=Path("data/player_stats.db"),
    )

    # 4) Write CSV + JSON locally (static filenames so they overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "gameday_player_profiles.csv"
    json_path = output_dir / "gameday_player_profiles.json"

    gameday_with_impact.to_csv(csv_path, index=False)
    print(f"💾 Wrote gameday CSV to: {csv_path}")

    gameday_json = _clean_strings_for_json(gameday_with_impact)
    gameday_json.to_json(json_path, orient="records", indent=2)
    print(f"💾 Wrote gameday JSON to: {json_path}")

    # 5) Optional S3 upload — static keys for backend
    if s3_bucket and upload_to_s3 is not None:
        if s3_prefix:
            prefix = s3_prefix.rstrip("/")
        else:
            # Keep it very simple/static per your requirement
            prefix = "gameday"

        csv_key = f"{prefix}/player_profiles.csv"
        json_key = f"{prefix}/player_profiles.json"

        print(f"☁️ Uploading CSV to s3://{s3_bucket}/{csv_key} ...")
        upload_to_s3(csv_path, s3_bucket, csv_key)

        print(f"☁️ Uploading JSON to s3://{s3_bucket}/{json_key} ...")
        upload_to_s3(json_path, s3_bucket, json_key)

        print("✅ S3 upload complete for gameday profiles.")
    elif s3_bucket and upload_to_s3 is None:
        print(
            "⚠️ s3_bucket was provided, but utils.s3_upload.upload_to_s3 "
            "is not available. Skipping S3 upload."
        )

    # 6) Quick console peek at a few rows so you can sanity-check fields
    print("\n🔎 Sample gameday rows (first 10):")
    preview_cols: List[str] = []
    for col in [
        "team",
        "player_id",
        "player_name",
        "games_played",
        "minutes_per_game",
        "epm_off",
        "epm_def",
        "epm_net",
        "impact_per_100",
        "exposure_stint_units",
    ]:
        if col in gameday_with_impact.columns:
            preview_cols.append(col)

    if preview_cols:
        print(gameday_with_impact[preview_cols].head(10).to_string(index=False))

    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export gameday player profiles (for teams playing today or a "
            "specified subset), merging season-long EPM with impact ratings "
            "stored in SQLite, and write CSV/JSON (optionally to S3)."
        )
    )
    parser.add_argument(
        "--season-label",
        default="2025-26",
        help="Season label, e.g. 2025-26. Default: 2025-26.",
    )
    parser.add_argument(
        "--teams",
        default=None,
        help=(
            "Optional comma-separated list of team codes, e.g. 'OKC,NYK'. "
            "If omitted, teams playing today are auto-detected via nba_api."
        ),
    )
    parser.add_argument(
        "--profiles-csv",
        default=None,
        help=(
            "Optional explicit path to season profiles CSV. If omitted, "
            "defaults to data/exports/player_profiles_from_db_{season}.csv."
        ),
    )
    parser.add_argument(
        "--impact-dir",
        default="data/impact",
        help=(
            "[Deprecated] Directory where impact_ratings_{TEAM}_{season}.csv live. "
            "Impact is now read from SQLite (impact_ratings table). "
            "Kept only for backward CLI compatibility."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/exports",
        help=(
            "Directory where gameday_player_profiles.csv/json will be written. "
            "Default: data/exports."
        ),
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Optional S3 bucket name to upload gameday profiles to.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help=(
            "Optional S3 key prefix. If omitted, defaults to 'gameday', and "
            "files are uploaded as gameday/player_profiles.csv/json."
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    season_label: str = args.season_label
    teams_arg: Optional[str] = args.teams
    profiles_csv: Optional[str] = args.profiles_csv
    impact_dir = Path(args.impact_dir)  # unused now, but kept for CLI compatibility
    output_dir = Path(args.output_dir)
    s3_bucket: Optional[str] = args.s3_bucket
    s3_prefix: Optional[str] = args.s3_prefix

    export_gameday_profiles(
        season_label=season_label,
        teams_arg=teams_arg,
        profiles_csv=profiles_csv,
        impact_dir=impact_dir,
        output_dir=output_dir,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
    )


if __name__ == "__main__":
    main()
