#!/usr/bin/env python
"""
Walk-forward backtest for per-player box-score models.

Uses the JSON dataset produced by build_player_train_dataset.py and
simulates training on past games only, then predicting the next day's games.

Outputs:
  - A per-row CSV/JSON of predictions vs actuals, including a simple
    per-player historical-average baseline for comparison.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------


@dataclass
class BacktestPlayerConfig:
    train_json: str
    output_csv: str
    output_json: str
    min_train_games: int
    random_state: int
    targets: List[str]


def parse_args() -> BacktestPlayerConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward backtest for per-player box-score models using "
            "the JSON produced by build_player_train_dataset.py."
        )
    )

    parser.add_argument(
        "--train-json",
        required=True,
        help="Path to the player training dataset JSON (records).",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Where to write per-row backtest results as CSV.",
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="Where to write per-row backtest results as JSON.",
    )

    parser.add_argument(
        "--min-train-games",
        type=int,
        default=200,
        help=(
            "Minimum number of historical rows required before we start "
            "backtesting (default: 200)."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model training.",
    )

    parser.add_argument(
        "--targets",
        type=str,
        default="pts,treb,ast",
        help=(
            "Comma-separated list of target stat columns to model "
            "(default: 'pts,treb,ast'). 'reb' will be mapped to 'treb'."
        ),
    )

    args = parser.parse_args()

    # Parse targets and map aliases
    raw_targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    alias_map = {
        "reb": "treb",
    }

    resolved_targets: List[str] = []
    for t in raw_targets:
        key = t.lower()
        if key in alias_map:
            mapped = alias_map[key]
            if mapped != t:
                print(f"ℹ️ Mapping target '{t}' to '{mapped}' based on DB schema.")
        else:
            mapped = t
        resolved_targets.append(mapped)

    return BacktestPlayerConfig(
        train_json=args.train_json,
        output_csv=args.output_csv,
        output_json=args.output_json,
        min_train_games=args.min_train_games,
        random_state=args.random_state,
        targets=resolved_targets,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def select_feature_columns(df: pd.DataFrame, targets: List[str]) -> List[str]:
    """Select numeric / boolean feature columns, excluding target columns.

    This mirrors the logic in train_player_models.py so the backtest uses
    the same feature space as training. Columns that are entirely missing
    (all NaN) are dropped to avoid noisy imputer warnings.
    """
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    feature_cols: List[str] = []
    for col_name in numeric_cols:
        if col_name in targets:
            continue
        # Skip columns that are entirely NaN / missing
        if not df[col_name].notna().any():
            print(
                f"   ⚠️ Dropping feature '{col_name}' because it has no non-missing values."
            )
            continue
        feature_cols.append(col_name)

    if not feature_cols:
        raise RuntimeError(
            "No feature columns found after excluding targets and all-NaN columns. "
            f"Numeric columns: {numeric_cols}, targets: {targets}"
        )

    return feature_cols


def make_model_pipeline(random_state: int) -> Pipeline:
    """
    Build the per-target regression pipeline:
      - median imputer
      - RandomForestRegressor
    """
    pipeline: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------


def run_backtest(cfg: BacktestPlayerConfig) -> pd.DataFrame:
    # Load data
    print(f"📥 Loading training data from: {cfg.train_json}")
    df = pd.read_json(cfg.train_json, orient="records")
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns.")

    # Ensure targets exist
    missing = [t for t in cfg.targets if t not in df.columns]
    if missing:
        raise RuntimeError(
            "Some target columns are missing from training data.\n"
            f"  Requested: {cfg.targets}\n"
            f"  Missing:   {missing}\n"
            f"  Available: {list(df.columns)}"
        )

    # Drop rows with missing targets
    df = df.dropna(subset=cfg.targets)
    print(f"📊 After dropping rows with missing targets: {len(df)} rows remain.")

    # Ensure game_date is datetime64
    if "game_date" not in df.columns:
        raise RuntimeError(
            "Expected 'game_date' column in training data, but it was not found."
        )

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if df["game_date"].isna().any():
        raise RuntimeError(
            "Some 'game_date' values could not be parsed to datetime. "
            "Please ensure build_player_train_dataset wrote valid dates."
        )

    # Sort by date (and then game_id, player_id for deterministic ordering)
    sort_cols = ["game_date"]
    if "game_id" in df.columns:
        sort_cols.append("game_id")
    if "player_id" in df.columns:
        sort_cols.append("player_id")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Feature columns (same logic as training)
    feature_cols = select_feature_columns(df, cfg.targets)
    print("🔎 Using feature columns:")
    print(f"   {len(feature_cols)} features")

    # Walk-forward over unique dates
    game_dates = pd.to_datetime(df["game_date"], errors="coerce")

    # Extract Python date objects; Pylance's pandas stubs don't know about .dt.date,
    # so we ignore the attribute access warning here.
    date_values = game_dates.dt.date  # type: ignore[reportAttributeAccessIssue]

    # Unique calendar dates as sorted list of datetime.date
    unique_dates = sorted(pd.unique(date_values))
    print(f"📆 Found {len(unique_dates)} distinct game dates in dataset.")

    results: List[Dict[str, Any]] = []

    for current_date in unique_dates:
        # Training = all games strictly before this date
        train_mask = date_values < current_date
        test_mask = date_values == current_date

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        n_train = len(train_df)
        n_test = len(test_df)

        print(
            f"\n📅 Backtest date {current_date}: {n_test} test rows, {n_train} train rows"
        )

        if n_test == 0:
            print("   ℹ️ No games on this date; skipping.")
            continue

        if n_train < cfg.min_train_games:
            print(
                f"   ⚠️ Not enough training rows (< {cfg.min_train_games}); "
                "skipping this date."
            )
            continue

        # Build per-target models on *past* data
        models: Dict[str, Pipeline] = {}
        for target in cfg.targets:
            print(f"   🚀 Training model for target='{target}' ...")
            model = make_model_pipeline(cfg.random_state)
            X_train = train_df[feature_cols]
            y_train = train_df[target].astype(float)
            model.fit(X_train, y_train)
            models[target] = model

        # Build simple per-player historical-average baselines
        # based on *past* data only.
        baselines: Dict[str, Dict[Any, float]] = {}
        global_means: Dict[str, float] = {}
        for target in cfg.targets:
            group_means = train_df.groupby("player_id")[target].mean().to_dict()
            baselines[target] = group_means
            global_means[target] = float(train_df[target].mean())

        # Predict for this date
        X_test = test_df[feature_cols]

        for idx, row in test_df.iterrows():
            rec: Dict[str, Any] = {
                "game_date": row["game_date"].date().isoformat(),
            }
            if "game_id" in row:
                rec["game_id"] = row["game_id"]
            if "team" in row:
                rec["team"] = row["team"]
            if "player_id" in row:
                rec["player_id"] = row["player_id"]
            if "player_name" in row:
                rec["player_name"] = row["player_name"]

            for target in cfg.targets:
                # Actual
                actual_val = float(row[target])
                rec[f"actual_{target}"] = actual_val

                # Model prediction
                x_vec = X_test.loc[[idx]]
                pred_val = float(models[target].predict(x_vec)[0])
                rec[f"pred_{target}"] = pred_val
                rec[f"error_{target}"] = pred_val - actual_val
                rec[f"abs_error_{target}"] = abs(pred_val - actual_val)

                # Baseline prediction (historical mean for this player)
                pid = row.get("player_id")
                player_means = baselines[target]
                if pid in player_means:
                    baseline_val = float(player_means[pid])
                else:
                    baseline_val = global_means[target]
                rec[f"baseline_{target}"] = baseline_val
                rec[f"baseline_error_{target}"] = baseline_val - actual_val
                rec[f"baseline_abs_error_{target}"] = abs(baseline_val - actual_val)

            results.append(rec)

    if not results:
        raise RuntimeError(
            "Backtest did not produce any rows. Likely min_train_games is too high "
            "for the available data."
        )

    results_df = pd.DataFrame(results)
    print(f"\n✅ Backtest produced {len(results_df)} rows.")

    # Compute overall metrics
    print("\n📊 Backtest performance (overall):")
    metrics: Dict[str, Dict[str, float]] = {}
    for target in cfg.targets:
        actual_vals = results_df[f"actual_{target}"].astype(float).to_numpy()
        pred_vals = results_df[f"pred_{target}"].astype(float).to_numpy()
        baseline_vals = results_df[f"baseline_{target}"].astype(float).to_numpy()

        model_mae = float(mean_absolute_error(actual_vals, pred_vals))
        model_rmse = float(math.sqrt(mean_squared_error(actual_vals, pred_vals)))
        baseline_mae = float(mean_absolute_error(actual_vals, baseline_vals))
        baseline_rmse = float(math.sqrt(mean_squared_error(actual_vals, baseline_vals)))

        metrics[target] = {
            "model_mae": model_mae,
            "model_rmse": model_rmse,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
        }

        print(
            f"   🧮 {target} MAE:    model={model_mae:6.3f}  "
            f"baseline={baseline_mae:6.3f}"
        )
        print(
            f"   🧮 {target} RMSE:   model={model_rmse:6.3f} "
            f" baseline={baseline_rmse:6.3f}"
        )

    # Attach metrics as an attribute (not preserved in CSV/JSON but handy if imported)
    results_df.attrs["metrics"] = metrics

    # Save outputs
    out_csv = Path(cfg.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Writing per-row backtest results CSV to: {out_csv}")
    results_df.to_csv(out_csv, index=False)

    out_json = Path(cfg.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing per-row backtest results JSON to: {out_json}")
    records = results_df.to_dict(orient="records")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": records,
                "metrics": metrics,
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
            f,
            indent=2,
        )

    print("\n🎉 Player backtest completed.")
    return results_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()

    print("===============================================")
    print("🔁  Backtesting player-level models (walk-forward)")
    print("===============================================")
    print(f"📄 Train JSON:   {cfg.train_json}")
    print(f"📂 Output CSV:   {cfg.output_csv}")
    print(f"📂 Output JSON:  {cfg.output_json}")
    print(f"📊 Min train rows: {cfg.min_train_games}")
    print(f"🎲 Random seed:  {cfg.random_state}")
    print(f"🎯 Targets:      {cfg.targets}")
    print("===============================================")

    run_backtest(cfg)


if __name__ == "__main__":
    main()
