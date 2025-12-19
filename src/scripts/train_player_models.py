#!/usr/bin/env python
"""
Train per-player regression models for box-score stats (e.g. pts, treb, ast)
using the dataset produced by build_player_train_dataset.py.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------


@dataclass
class TrainPlayerModelsConfig:
    train_json: str
    output_dir: str
    test_size: float
    random_state: int
    targets: List[str]


def parse_args() -> TrainPlayerModelsConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train per-player regression models for box-score stats "
            "using the JSON produced by build_player_train_dataset.py."
        )
    )

    parser.add_argument(
        "--train-json",
        required=True,
        help=(
            "Path to the player training dataset JSON (records), "
            "usually produced by build_player_train_dataset.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write trained player models and metadata.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows to use for the test set (default: 0.2).",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model training.",
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
        "reb": "treb",  # map common alias to DB column
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

    return TrainPlayerModelsConfig(
        train_json=args.train_json,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        targets=resolved_targets,
    )


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------


def select_feature_columns(df: pd.DataFrame, targets: List[str]) -> List[str]:
    """
    Select numeric feature columns from df, excluding the target columns.

    We keep things simple:
      - take all numeric / bool columns
      - drop any column that is one of the targets
    """
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in targets]

    if not feature_cols:
        raise RuntimeError(
            "No feature columns found after excluding targets. "
            f"Numeric columns: {numeric_cols}, targets: {targets}"
        )

    return feature_cols


def train_regression_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple[Pipeline, dict]:
    """
    Train a RandomForest regression model for a single target column.

    Returns:
      (fitted_pipeline, metrics_dict)
    """
    if target_col not in df.columns:
        raise RuntimeError(
            f"Target column '{target_col}' not found in training DataFrame."
        )

    X = df[feature_cols]
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print(
        f"🚀 Training regression model for target='{target_col}' "
        f"on {len(X_train)} train / {len(X_test)} test rows..."
    )

    # Basic pipeline: impute missing values, then RandomForest
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

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    # Older sklearn versions don't support squared=False
    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)

    print(f"   📏 {target_col} MAE:  {mae:6.3f}")
    print(f"   📏 {target_col} RMSE: {rmse:6.3f}")

    metrics = {
        "target": target_col,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": mae,
        "rmse": rmse,
    }

    return pipeline, metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()

    print("===============================================")
    print("🤖  Training player-level models")
    print("===============================================")
    print(f"📄 Train JSON:  {cfg.train_json}")
    print(f"📂 Output dir:  {cfg.output_dir}")
    print(f"📊 Test size:   {cfg.test_size}")
    print(f"🎲 Random seed: {cfg.random_state}")
    print(f"🎯 Targets:     {cfg.targets}")
    print("===============================================")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print(f"📥 Loading training data from: {cfg.train_json}")
    df = pd.read_json(cfg.train_json, orient="records")
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns.")

    # Select feature columns
    feature_cols = select_feature_columns(df, cfg.targets)
    print("🔎 Selecting feature columns...")
    print(f"   Found {len(feature_cols)} feature columns.")

    all_metrics: dict[str, dict] = {}
    for target in cfg.targets:
        model, metrics = train_regression_model(
            df=df,
            feature_cols=feature_cols,
            target_col=target,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
        )

        model_path = output_dir / f"player_{target}_model.joblib"
        joblib.dump(model, model_path)
        print(f"💾 Saved model to: {model_path}")

        all_metrics[target] = metrics

    # Save metadata about the training run
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_json": cfg.train_json,
        "output_dir": str(output_dir),
        "test_size": cfg.test_size,
        "random_state": cfg.random_state,
        "targets": cfg.targets,
        "feature_columns": feature_cols,
        "metrics": all_metrics,
    }

    metadata_path = output_dir / "player_models_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Saved metadata JSON to: {metadata_path}")
    print("🎉 Player models trained and saved successfully.")


if __name__ == "__main__":
    main()
