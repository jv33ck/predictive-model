# src/scripts/export_impact_ratings.py
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

from features.impact_dataset import build_lineup_stint_impact_for_team_season
from features.impact_ridge import fit_ridge_impact_model
from data.nba_api_provider import get_game_boxscore_traditional
from db.impact_ratings_db import upsert_impact_ratings

try:
    from utils.s3_upload import upload_to_s3
except ImportError:
    upload_to_s3 = None  # type: ignore[assignment]


# ---- Helpers ---------------------------------------------------------------


def _load_player_names_from_db(db_path: Path) -> Optional[pd.DataFrame]:
    """
    Load distinct (player_id, player_name) from player_game_stats.

    Returns:
        DataFrame with columns ['player_id', 'player_name'] or None
        if the DB does not exist or the table is missing.
    """
    if not db_path.exists():
        print(
            f"⚠️ DB not found at {db_path}; impact export will omit player_name from DB."
        )
        return None

    try:
        conn = sqlite3.connect(db_path.as_posix())
        try:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT player_id, player_name
                FROM player_game_stats
                WHERE player_name IS NOT NULL
                """,
                conn,
            )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Failed to load player names from DB: {exc}")
        return None

    if df.empty:
        print(
            "⚠️ No player_name rows found in player_game_stats; DB-based names will be NaN."
        )
        return None

    # Do *not* coerce to string here; we’ll normalize in the attach function.
    return df[["player_id", "player_name"]]


def _normalize_player_id_series(series: pd.Series) -> pd.Series:
    """
    Robustly normalize player_id values into a comparable string form.

    - Accepts ints, floats, strings, etc.
    - Strips whitespace.
    - Extracts digits, and if possible reduces things like '1630245.0' to '1630245'.
    - Returns a string of digits or None if no plausible ID can be extracted.
    """

    def _normalize_one(val: Any) -> Optional[str]:
        # Missing / NaN
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return None

        # Already a clean integer
        if isinstance(val, (int, np.integer)):
            return str(int(val))

        # Float that might represent an integer (e.g. 1630245.0)
        if isinstance(val, float):
            if float(val).is_integer():
                return str(int(val))
            # Fallback: round, but this really shouldn't happen for player ids
            return str(int(round(val)))

        # Generic object -> string, strip, and keep only digits
        s = str(val).strip()
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits or None

    return series.apply(_normalize_one)


def _build_name_map_from_boxscores(stints_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping {player_id_str -> 'First Last'} using BoxScoreTraditionalV3
    for all games in the stint dataset.

    This bypasses the DB entirely and uses the clean IDs/names from nba_api.
    """
    if "game_id" not in stints_df.columns:
        print(
            "⚠️ stints_df has no 'game_id' column; cannot build name map from boxscores."
        )
        return {}

    game_ids = sorted(stints_df["game_id"].dropna().unique().tolist())
    name_map: Dict[str, str] = {}

    for game_id in game_ids:
        try:
            box_df = get_game_boxscore_traditional(str(game_id))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Failed to fetch BoxScoreTraditionalV3 for game {game_id}: {exc}")
            continue

        if box_df is None or box_df.empty:
            continue

        # Expect personId, firstName, familyName columns per your existing provider
        for _, row in box_df.iterrows():
            pid_val = row.get("personId")
            first = str(row.get("firstName", "")).strip()
            last = str(row.get("familyName", "")).strip()
            if pid_val is None:
                continue

            # Normalize personId to digit-string
            pid_norm = "".join(ch for ch in str(pid_val) if ch.isdigit())
            if not pid_norm:
                continue

            full_name = f"{first} {last}".strip()
            if not full_name:
                continue

            # Don't overwrite an existing mapping if we already have a name
            if pid_norm not in name_map:
                name_map[pid_norm] = full_name

    print(f"ℹ️ Built name map from boxscores for {len(name_map)} players.")
    return name_map


def _attach_player_names(
    impact_df: pd.DataFrame,
    db_path: Path,
    stints_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Attach player_name to the impact DataFrame.

    Strategy:
      1. Try to use names from SQLite DB (player_game_stats).
      2. If that yields ~0 coverage, fall back to nba_api boxscores
         for all games in stints_df via BoxScoreTraditionalV3.

    We normalize player_id values into a canonical digit-string
    before joining, to handle cases like:
        - 1630245 (int) vs '1630245' (text)
        - 1630245.0 (float) vs '1630245'
        - stray whitespace or other characters.
    """
    impact = impact_df.copy()
    if impact.empty:
        impact["player_name"] = np.nan
        return impact

    # --------------------
    # 1) Try DB-based join
    # --------------------
    names_df = _load_player_names_from_db(db_path)

    if names_df is not None and not names_df.empty:
        names = names_df.copy()
        impact["player_id_norm"] = _normalize_player_id_series(impact["player_id"])
        names["player_id_norm"] = _normalize_player_id_series(names["player_id"])

        # Drop rows where we couldn't infer a plausible ID on either side
        impact_db = impact[impact["player_id_norm"].notna()].copy()
        names_db = names[names["player_id_norm"].notna()].copy()

        if not impact_db.empty and not names_db.empty:
            merged = impact_db.merge(
                names_db[["player_id_norm", "player_name"]].drop_duplicates(
                    "player_id_norm"
                ),
                on="player_id_norm",
                how="left",
                validate="m:1",  # each impact row maps to at most one name
            )

            num_with_name = merged["player_name"].notna().sum()
            print(
                f"ℹ️ DB join attached names for {num_with_name}/{len(merged)} "
                f"impact rows ({num_with_name / max(len(merged), 1):.1%} coverage)."
            )

            # If we got reasonable coverage, use this result
            if num_with_name > 0:
                # Recombine with any impact rows that had no player_id_norm
                no_norm = impact[impact["player_id_norm"].isna()].copy()
                no_norm["player_name"] = np.nan
                merged = pd.concat(
                    [merged.drop(columns=["player_id_norm"]), no_norm],
                    ignore_index=True,
                )
                return merged

            # If coverage is 0, we'll fall through to boxscore-based names below.
        else:
            print(
                "ℹ️ DB join skipped because normalized player_id sets on DB/impact "
                "were empty."
            )
    else:
        print("ℹ️ Skipping DB-based name join; no usable names loaded from DB.")

    # ---------------------------------------
    # 2) Fallback: boxscore-based name mapping
    # ---------------------------------------
    if stints_df is None or stints_df.empty:
        print(
            "⚠️ stints_df is None or empty, cannot build boxscore-based name map. "
            "Returning impact_df with player_name = NaN."
        )
        impact["player_name"] = np.nan
        return impact

    name_map = _build_name_map_from_boxscores(stints_df)
    if not name_map:
        print(
            "⚠️ Boxscore-based name map is empty; returning impact_df with "
            "player_name = NaN."
        )
        impact["player_name"] = np.nan
        return impact

    impact["player_id_norm"] = _normalize_player_id_series(impact["player_id"])
    impact["player_name"] = impact["player_id_norm"].map(name_map)

    num_with_name_box = impact["player_name"].notna().sum()
    print(
        f"ℹ️ Boxscore fallback attached names for {num_with_name_box}/{len(impact)} "
        f"impact rows ({num_with_name_box / max(len(impact), 1):.1%} coverage)."
    )

    return impact.drop(columns=["player_id_norm"])


def _clean_strings_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean string-like columns so that to_json doesn't choke on bad bytes.

    This mirrors the pattern we've used elsewhere: convert to str and then
    strip any invalid UTF-8 sequences.
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


def export_impact_ratings_for_team(
    team: str,
    season_label: str,
    max_games: int,
    alpha: float,
    min_stint_possessions: int,
    db_path: Path,
    output_dir: Path,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
) -> Path:
    """
    Build and export ridge impact ratings (RAPM-style) for a single team/season.

    Args:
        team: 3-letter team code (e.g., 'ATL', 'NYK').
        season_label: NBA season label, e.g. '2025-26'.
        max_games: Max games from that season to use (<=0 means all).
        alpha: Ridge regularization strength.
        min_stint_possessions: Minimum possessions for a stint to be included.
        db_path: Path to SQLite DB (player_stats.db).
        output_dir: Directory where CSV/JSON will be written.
        s3_bucket: Optional S3 bucket name to upload results.
        s3_prefix: Optional prefix inside the bucket, e.g. 'impact/2025-26'.

    Returns:
        Path to the written CSV file (local).
    """
    print(
        f"📊 Exporting impact ratings for team {team}, season {season_label}, "
        f"max_games={max_games}, alpha={alpha}, "
        f"min_stint_possessions={min_stint_possessions}..."
    )

    # ------------------------------------------------------------------ #
    # 1) Build stint-level impact dataset
    # ------------------------------------------------------------------ #
    stints_df = build_lineup_stint_impact_for_team_season(
        team_abbrev=team,
        max_games=max_games,
    )
    print(f"✅ Built stint dataset with {len(stints_df)} rows.")

    # ------------------------------------------------------------------ #
    # 2) Fit ridge RAPM model
    # ------------------------------------------------------------------ #
    impact_core = fit_ridge_impact_model(
        stints_df=stints_df,
        alpha=alpha,
        min_possessions_per_stint=min_stint_possessions,
    )
    print(f"✅ Fitted impact model; {len(impact_core)} players with impact estimates.")

    # ------------------------------------------------------------------ #
    # 3) Attach team/season, player names
    # ------------------------------------------------------------------ #
    impact_core = impact_core.copy()
    impact_core["team"] = team
    impact_core["season"] = season_label

    # Attach names: try DB first, then boxscore fallback
    impact_with_names = _attach_player_names(
        impact_core,
        db_path=db_path,
        stints_df=stints_df,
    )

    # Reorder columns for nicer output
    cols_order = [
        "team",
        "season",
        "player_id",
        "player_name",
        "impact_off_per_possession",
        "impact_def_per_possession",
        "impact_per_possession",
        "impact_off_per_100",
        "impact_def_per_100",
        "impact_per_100",
        "exposure_stint_units",
    ]
    cols_order = [c for c in cols_order if c in impact_with_names.columns]
    impact_final = impact_with_names[cols_order].copy()

    # Sort by impact_per_100 descending for readability
    if "impact_per_100" in impact_final.columns:
        impact_final = impact_final.sort_values(
            by="impact_per_100", ascending=False
        ).reset_index(drop=True)

    # 3b. Upsert into SQLite impact_ratings table
    try:
        upsert_impact_ratings(
            team=team,
            season=season_label,
            ratings_df=impact_with_names,
            db_path=db_path,
        )
    except Exception as e:
        # Don’t kill the export if DB upsert fails; just warn.
        print(
            f"⚠️ Failed to upsert impact ratings into DB for {team} {season_label}: {e}"
        )

    # ------------------------------------------------------------------ #
    # 4) Write CSV + JSON locally
    # ------------------------------------------------------------------ #
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"impact_ratings_{team}_{season_label}.csv"
    json_path = output_dir / f"impact_ratings_{team}_{season_label}.json"

    impact_final.to_csv(csv_path, index=False)
    print(f"💾 Wrote CSV to: {csv_path}")

    # Clean strings before JSON export to avoid UTF-8 issues
    impact_for_json = _clean_strings_for_json(impact_final)
    impact_for_json.to_json(json_path, orient="records", indent=2)
    print(f"💾 Wrote JSON to: {json_path}")

    # ------------------------------------------------------------------ #
    # 5) Optional S3 upload
    # ------------------------------------------------------------------ #
    if s3_bucket and upload_to_s3 is not None:
        # Build stable S3 keys – no date baked in so backend can hard-code path if desired
        if s3_prefix:
            prefix = s3_prefix.rstrip("/")
        else:
            prefix = f"impact/{season_label}"

        csv_key = f"{prefix}/impact_ratings_{team}_{season_label}.csv"
        json_key = f"{prefix}/impact_ratings_{team}_{season_label}.json"

        print(f"☁️ Uploading CSV to s3://{s3_bucket}/{csv_key} ...")
        upload_to_s3(csv_path, s3_bucket, csv_key)

        print(f"☁️ Uploading JSON to s3://{s3_bucket}/{json_key} ...")
        upload_to_s3(json_path, s3_bucket, json_key)

        print("✅ S3 upload complete.")

    elif s3_bucket and upload_to_s3 is None:
        print(
            "⚠️ s3_bucket was provided, but utils.s3_upload.upload_to_s3 "
            "is not available. Skipping S3 upload."
        )

    # ------------------------------------------------------------------ #
    # 6) Quick console summary (top/bottom impact)
    # ------------------------------------------------------------------ #
    if not impact_final.empty and "impact_per_100" in impact_final.columns:
        print("\n🔎 Top 10 by impact_per_100:")
        print(
            impact_final[
                ["player_id", "player_name", "impact_per_100", "exposure_stint_units"]
            ]
            .head(10)
            .to_string(index=False)
        )

        print("\n🔎 Bottom 10 by impact_per_100:")
        print(
            impact_final[
                ["player_id", "player_name", "impact_per_100", "exposure_stint_units"]
            ]
            .tail(10)
            .to_string(index=False)
        )

    return csv_path


# ---- CLI -------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export ridge impact (RAPM-style) ratings for a single team/season "
            "to CSV/JSON, optionally uploading to S3."
        )
    )
    parser.add_argument(
        "--team",
        required=True,
        help="Team code, e.g. ATL, NYK, OKC.",
    )
    parser.add_argument(
        "--season-label",
        default="2025-26",
        help="Season label, e.g. 2025-26. Default: 2025-26",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=10,
        help=(
            "Maximum games to include from that season for the RAPM dataset. "
            "Use -1 or 0 for all games. Default: 10."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Ridge regularization strength (alpha). Default: 50.0.",
    )
    parser.add_argument(
        "--min-stint-possessions",
        type=int,
        default=3,
        help=(
            "Minimum possessions for a stint to be included in the RAPM fit. "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/player_stats.db",
        help="Path to SQLite DB (player_stats.db). Default: data/player_stats.db",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/impact",
        help="Local directory to write CSV/JSON. Default: data/impact",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="Optional S3 bucket name to upload impact files.",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help=(
            "Optional S3 key prefix (e.g. 'impact/2025-26'). If omitted, "
            "defaults to 'impact/{season_label}'."
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    team = args.team.upper()
    season_label: str = args.season_label
    max_games: int = int(args.max_games)
    if max_games <= 0:
        max_games = -1

    alpha: float = float(args.alpha)
    min_stint_possessions: int = int(args.min_stint_possessions)

    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix

    export_impact_ratings_for_team(
        team=team,
        season_label=season_label,
        max_games=max_games,
        alpha=alpha,
        min_stint_possessions=min_stint_possessions,
        db_path=db_path,
        output_dir=output_dir,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
    )


if __name__ == "__main__":
    main()
