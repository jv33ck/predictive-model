#!/usr/bin/env python
import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine team-level matchup predictions and player predictions "
            "into a single per-player matchup file, and optionally upload to S3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Gameday date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--team-preds-json",
        type=str,
        default=None,
        help=(
            "Path to team matchup predictions JSON. "
            "Defaults to data/exports/gameday_team_predictions_<date>.json"
        ),
    )
    parser.add_argument(
        "--player-preds-json",
        type=str,
        default=None,
        help=(
            "Path to player predictions JSON. "
            "Defaults to data/exports/gameday_player_predictions_<date>.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Output CSV path for combined matchup+player predictions. "
            "Defaults to data/exports/gameday_matchup_predictions_<date>.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Output JSON path for combined matchup+player predictions. "
            "Defaults to data/exports/gameday_matchup_predictions_<date>.json"
        ),
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="Optional S3 bucket for upload (e.g. oddzup-stats-2025).",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="gameday",
        help="S3 key prefix, e.g. 'gameday'. Only used if --s3-bucket is set.",
    )
    return parser.parse_args()


def load_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    # Files are written with orient="records"
    return pd.read_json(path)


def build_team_side_view(team_df: pd.DataFrame, gameday_date: str) -> pd.DataFrame:
    """
    Expand team-level predictions to per-team/per-game rows.

    For each game row with home_team / away_team we create:
      - one row for home_team (is_home=True, opponent=away_team)
      - one row for away_team (is_home=False, opponent=home_team)
    """
    required = ["home_team", "away_team"]
    missing = [c for c in required if c not in team_df.columns]
    if missing:
        raise RuntimeError(
            f"Team predictions are missing required columns: {missing}. "
            "Expected at least ['home_team', 'away_team']."
        )

    records: List[Dict[str, Any]] = []

    for _, row in team_df.iterrows():
        base: Dict[str, Any] = row.to_dict()

        # Use existing game_date if present, else the CLI date
        base_game_date = base.get("game_date", gameday_date)

        home_team = str(base["home_team"])
        away_team = str(base["away_team"])

        # Home row
        home_rec = dict(base)
        home_rec["team"] = home_team
        home_rec["opponent"] = away_team
        home_rec["is_home"] = True
        home_rec["game_date"] = base_game_date

        # Away row
        away_rec = dict(base)
        away_rec["team"] = away_team
        away_rec["opponent"] = home_team
        away_rec["is_home"] = False
        away_rec["game_date"] = base_game_date

        records.append(home_rec)
        records.append(away_rec)

    team_side_df = pd.DataFrame.from_records(records)

    # Normalize team codes to string
    team_side_df["team"] = team_side_df["team"].astype(str)

    return team_side_df


def upload_to_s3(
    local_path: Path,
    bucket: Optional[str],
    key: str,
) -> None:
    if bucket is None:
        return

    try:
        import boto3  # type: ignore[import-untyped]
    except ImportError:
        print(
            "⚠️ boto3 is not installed; cannot upload to S3. "
            f"Skipping upload for {local_path}."
        )
        return

    s3 = boto3.client("s3")
    print(f"☁️ Uploading {local_path} to s3://{bucket}/{key} ...")
    try:
        s3.upload_file(str(local_path), bucket, key)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ S3 upload failed for {local_path}: {exc}")
        return

    print(f"✅ Upload complete: s3://{bucket}/{key}")


def main() -> None:
    args = parse_args()

    if args.date is not None:
        gameday_date = args.date
    else:
        gameday_date = dt.date.today().isoformat()

    print("===============================================")
    print("📦  Export gameday matchup predictions package")
    print("===============================================")
    print(f"📆 Gameday date:   {gameday_date}")

    # Resolve input paths
    if args.team_preds_json is not None:
        team_preds_path = Path(args.team_preds_json)
    else:
        team_preds_path = Path(
            f"data/exports/gameday_team_predictions_{gameday_date}.json"
        )

    if args.player_preds_json is not None:
        player_preds_path = Path(args.player_preds_json)
    else:
        player_preds_path = Path(
            f"data/exports/gameday_player_predictions_{gameday_date}.json"
        )

    # Resolve outputs
    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
    else:
        output_csv = Path(
            f"data/exports/gameday_matchup_predictions_{gameday_date}.csv"
        )

    if args.output_json is not None:
        output_json = Path(args.output_json)
    else:
        output_json = Path(
            f"data/exports/gameday_matchup_predictions_{gameday_date}.json"
        )

    print(f"📄 Team preds:     {team_preds_path}")
    print(f"📄 Player preds:   {player_preds_path}")
    print(f"📂 Output CSV:     {output_csv}")
    print(f"📂 Output JSON:    {output_json}")
    if args.s3_bucket:
        print(
            f"☁️ S3 target:      s3://{args.s3_bucket}/{args.s3_prefix}/"
            "matchup_predictions.(csv|json)"
        )
    else:
        print("☁️ S3 target:      (none)")
    print("===============================================")

    # --- Load inputs ---
    print("\n📥 Loading team predictions...")
    team_df = load_json(team_preds_path)
    print(f"   Loaded {len(team_df)} team-prediction rows.")

    print("📥 Loading player predictions...")
    player_df = load_json(player_preds_path)
    print(f"   Loaded {len(player_df)} player-prediction rows.")

    # Ensure team keys are strings
    if "team" not in player_df.columns:
        raise RuntimeError(
            "Player predictions are missing 'team' column; "
            "cannot merge with team-level predictions."
        )
    player_df["team"] = player_df["team"].astype(str)

    # --- Build per-team view from team predictions ---
    print("\n🔧 Building per-team matchup view from team predictions...")
    team_side_df = build_team_side_view(team_df, gameday_date)
    print(f"   Built {len(team_side_df)} team-side rows.")

    # --- Merge: one row per player per game, with game-level predictions attached ---
    print("\n🔗 Merging player predictions with team matchup context...")
    merged = player_df.merge(
        team_side_df,
        on="team",
        how="inner",
        suffixes=("", "_game"),
    )

    # If we dropped anything, log it
    if len(merged) < len(player_df):
        missing = len(player_df) - len(merged)
        print(
            f"⚠️ {missing} player rows had no matching team-level game row "
            "(team not in today's matchups) and were dropped."
        )

    print(f"   Combined rows: {len(merged)}")

    # --- Choose a nice column order ---
    # Player identity
    player_cols: List[str] = [
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
    ]
    player_cols = [c for c in player_cols if c in merged.columns]

    # Player predictions
    pred_cols: List[str] = [
        c for c in merged.columns if c.startswith("pred_") and c in merged.columns
    ]

    # Game context (from team predictions)
    # Keep common identifiers if present
    game_context_candidates: List[str] = [
        "game_date",
        "game_id",
        "season_label",
        "home_team",
        "away_team",
        "opponent",
        "is_home",
        "pred_margin",
        "pred_total_points",
        "pred_home_win_prob",
    ]
    game_cols = [c for c in game_context_candidates if c in merged.columns]

    # Any other remaining columns we might want to keep
    already = set(player_cols + pred_cols + game_cols)
    other_cols = [c for c in merged.columns if c not in already]

    ordered_cols = player_cols + game_cols + pred_cols + other_cols
    output_df = merged[ordered_cols].copy()

    # --- Write outputs ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Writing combined matchup+player CSV to: {output_csv}")

    # Ensure column names are unique before JSON export (required for orient='records')
    cols = list(output_df.columns)
    seen: dict[str, int] = {}
    new_cols: list[str] = []

    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_name = f"{col}__{seen[col]}"
            new_cols.append(new_name)

    if new_cols != cols:
        print("🔧 Detected duplicate column names; renaming for JSON export:")
        for old, new in zip(cols, new_cols):
            if old != new:
                print(f"   - {old!r} → {new!r}")
        output_df = output_df.copy()
        output_df.columns = new_cols

    # Now write CSV/JSON with guaranteed-unique column names
    output_df.to_csv(output_csv, index=False)

    print(f"💾 Writing combined matchup+player JSON to: {output_json}")
    output_df.to_json(output_json, orient="records")

    # --- Optional S3 upload ---
    if args.s3_bucket:
        csv_key = f"{args.s3_prefix}/matchup_predictions.csv"
        json_key = f"{args.s3_prefix}/matchup_predictions.json"
        upload_to_s3(output_csv, args.s3_bucket, csv_key)
        upload_to_s3(output_json, args.s3_bucket, json_key)

    # --- Quick sample ---
    print("\n🔎 Sample combined rows (first 10):")
    with pd.option_context("display.max_columns", None):
        print(output_df.head(10))

    print("\n🎉 Gameday matchup prediction package export complete.")


if __name__ == "__main__":
    main()
