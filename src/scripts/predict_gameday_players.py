#!/usr/bin/env python
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gameday player-level stat predictions (pts / reb / ast).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default="data/training/player_train_2025-26_oct-nov.json",
        help="Player training dataset JSON (same one used to train player models).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models/player",
        help="Directory containing trained player models and metadata.",
    )
    parser.add_argument(
        "--gameday-profiles",
        type=str,
        default="data/exports/gameday_player_profiles.json",
        help="Gameday player profiles JSON (exported by export_gameday_profiles.py).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Gameday date (YYYY-MM-DD) used for logging and default output filenames.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional explicit output CSV path for predictions.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional explicit output JSON path for predictions.",
    )
    return parser.parse_args()


CALIBRATION_PATH = Path("data/models/player/player_calibration_stats.json")


def apply_distribution_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale pred_pts / pred_treb / pred_ast so their distribution
    matches historical rotation-player totals from training.

    Uses mean/std from player_calibration_stats.json and adjusts:
        y_cal = a + b * y_raw
    so that mean/std of y_cal == mean/std of historical actuals.
    """
    if not CALIBRATION_PATH.exists():
        print(
            f"⚠️ Calibration stats not found at {CALIBRATION_PATH}; "
            "skipping distribution calibration."
        )
        return df

    with CALIBRATION_PATH.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    df = df.copy()

    for target in ["pts", "treb", "ast"]:
        col = f"pred_{target}"
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not in prediction DataFrame; skipping.")
            continue
        if target not in stats:
            print(f"⚠️ No calibration stats for '{target}'; skipping.")
            continue

        mean_actual = float(stats[target].get("mean_actual", 0.0))
        std_actual = float(stats[target].get("std_actual", 0.0))

        pred = df[col].astype(float)
        mean_pred = float(pred.mean())
        std_pred = float(pred.std(ddof=0))

        if std_pred == 0 or std_actual == 0:
            print(
                f"⚠️ Degenerate std for target '{target}' "
                f"(std_pred={std_pred:.4f}, std_actual={std_actual:.4f}); skipping."
            )
            continue

        # Solve for a, b: mean(a + b X) = mean_actual, std(a + b X) = std_actual
        b = std_actual / std_pred
        a = mean_actual - b * mean_pred

        df[col] = a + b * pred

        print(
            f"   🔧 Calibrated {col}: "
            f"mean_raw={mean_pred:.3f}, std_raw={std_pred:.3f} → "
            f"mean_cal={mean_actual:.3f}, std_cal={std_actual:.3f}"
        )

    return df


def load_metadata(models_dir: Path) -> Dict[str, Any]:
    meta_path = models_dir / "player_models_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Could not find metadata JSON at {meta_path}. "
            "Make sure train_player_models.py has been run."
        )
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_records(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both {"rows": [...]} and plain list-of-dicts
    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
    else:
        rows = data

    return pd.DataFrame(rows)


def build_player_feature_averages(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Collapse the historical training dataset into a per-player average feature vector.

    We group by (team, player_id) and take the mean of all numeric feature columns.
    This gives us a "typical" feature row per player that we can feed into the models
    for gameday predictions.
    """

    # Ensure IDs used for joining are strings
    for col in ("team", "player_id"):
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)

    # Numeric columns only; models expect numeric inputs
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # We don't include identifiers in the aggregation set
    group_keys = ["team", "player_id"]
    agg_cols = [c for c in numeric_cols if c not in group_keys]

    missing_keys = [k for k in group_keys if k not in train_df.columns]
    if missing_keys:
        raise RuntimeError(
            f"Training data is missing required columns for grouping: {missing_keys}"
        )

    grouped = (
        train_df[group_keys + agg_cols]
        .groupby(group_keys, as_index=False)[agg_cols]
        .mean()
    )

    # Some feature columns might not be numeric (e.g., if metadata includes them).
    # For any feature col not present after numeric aggregation, add as NaN.
    for col in feature_cols:
        if col not in grouped.columns and col not in group_keys:
            grouped[col] = np.nan

    return grouped


def main() -> None:
    args = parse_args()

    models_dir = Path(args.models_dir)
    train_json_path = Path(args.train_json)
    gameday_profiles_path = Path(args.gameday_profiles)

    # Resolve date string for logging / default outputs
    if args.date is not None:
        gameday_date_str = args.date
    else:
        gameday_date_str = dt.date.today().isoformat()

    default_csv = f"data/exports/gameday_player_predictions_{gameday_date_str}.csv"
    default_json = f"data/exports/gameday_player_predictions_{gameday_date_str}.json"

    output_csv = Path(args.output_csv or default_csv)
    output_json = Path(args.output_json or default_json)

    print("===============================================")
    print("📡  Gameday player predictions")
    print("===============================================")
    print(f"📆 Gameday date:   {gameday_date_str}")
    print(f"📄 Train JSON:     {train_json_path}")
    print(f"📂 Models dir:     {models_dir}")
    print(f"📄 Gameday input:  {gameday_profiles_path}")
    print(f"📂 Output CSV:     {output_csv}")
    print(f"📂 Output JSON:    {output_json}")
    print("===============================================")

    # --- Load metadata & determine feature set / targets ---
    metadata = load_metadata(models_dir)
    feature_cols = metadata.get("feature_columns")
    if not feature_cols:
        raise RuntimeError(
            "Metadata JSON does not contain 'feature_columns'. "
            "Please re-run train_player_models.py to regenerate metadata."
        )

    targets = metadata.get("targets", ["pts", "treb", "ast"])
    print(f"🎯 Targets:        {targets}")
    print(f"🔢 Feature cols:   {len(feature_cols)}")

    # --- Load training dataset (historical per-game features) ---
    print("\n📥 Loading training data for per-player averages...")
    train_df = load_json_records(train_json_path)
    print(f"   Loaded {len(train_df)} training rows with {train_df.shape[1]} columns.")

    # --- Build per-player average feature vectors ---
    print("🔧 Building per-player average feature vectors from training data...")
    player_avg = build_player_feature_averages(train_df, feature_cols)
    print(f"   Built {len(player_avg)} per-player rows from training data.")

    # --- Load gameday player list (who we want predictions for) ---
    print("\n📥 Loading gameday player profiles...")
    gameday_df = load_json_records(gameday_profiles_path)
    print(f"   Loaded {len(gameday_df)} gameday rows.")

    # Normalize join keys to string
    for col in ("team", "player_id"):
        if col in gameday_df.columns:
            gameday_df[col] = gameday_df[col].astype(str)

    # --- Join per-player averages onto today's player list ---
    print("\n🔗 Joining per-player averages onto gameday list...")
    merged = gameday_df.merge(
        player_avg,
        on=["team", "player_id"],
        how="left",
        suffixes=("", "_hist"),
    )

    # Sanity check: how many players have no historical features?
    no_hist = merged[feature_cols].isna().all(axis=1).sum()
    print(f"   Players with no historical features (all NaN): {no_hist}")

    # For any feature column still missing in merged, add as NaN
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    X = merged[feature_cols]

    # --- Load models and generate predictions ---
    print("\n🔮 Generating predictions...")
    for target in targets:
        model_path = models_dir / f"player_{target}_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected model file for target '{target}' at {model_path}"
            )

        print(f"   📦 Loading model for target='{target}' from {model_path} ...")
        model = joblib.load(model_path)

        print(f"   🚀 Predicting {target} ...")
        preds = model.predict(X)
        merged[f"pred_{target}"] = preds

    # --- Build output subset (nice, compact view) ---
    base_cols = [
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
    base_cols = [c for c in base_cols if c in merged.columns]

    pred_cols = [f"pred_{t}" for t in targets]

    out_cols = base_cols + pred_cols

    output_df = merged[out_cols].copy()

    # --- Apply distribution calibration ---
    print("\n🔧 Applying distribution calibration to predictions...")
    output_df = apply_distribution_calibration(output_df)

    # --- Post-process predictions: clamp to realistic bounds ---
    for target in targets:
        col = f"pred_{target}"
        if col in output_df.columns:
            # No negative points / rebounds / assists
            output_df[col] = output_df[col].clip(lower=0)

            # Optional: round a bit for nicer display (keep one decimal place)
            output_df[col] = output_df[col].round(1)
    # --- Save outputs ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Writing predictions CSV to: {output_csv}")
    output_df.to_csv(output_csv, index=False)

    print(f"💾 Writing predictions JSON to: {output_json}")
    output_df.to_json(output_json, orient="records")

    # --- Show a quick sample ---
    print("\n🔎 Sample predictions (first 15 rows):")
    with pd.option_context("display.max_columns", None):
        print(output_df.head(15))

    print("\n🎉 Gameday player prediction export complete.")


if __name__ == "__main__":
    main()
