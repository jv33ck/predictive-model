#!/usr/bin/env python
"""
Train team-level models for:
  - margin (spread)
  - total points
  - home win probability

Input:  team-level matchup training dataset built by build_team_train_dataset.py
Output: trained models + metadata JSON in an output directory.

Example:
PYTHONPATH=src python src/scripts/train_team_models.py \
  --train-json data/training/team_matchups_train_2025-26_oct-nov.json \
  --output-dir data/models/team \
  --test-size 0.2 \
  --random-state 42
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


TARGET_MARGIN = "margin"
TARGET_TOTAL = "total_points"
TARGET_HOME_WIN = "home_win"


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float


@dataclass
class ClassificationMetrics:
    accuracy: float
    brier: float
    roc_auc: float | None  # can be None if ROC-AUC not defined


@dataclass
class ModelMetadata:
    model_type: str
    target: str
    feature_columns: List[str]
    train_rows: int
    test_rows: int
    regression_metrics: RegressionMetrics | None = None
    classification_metrics: ClassificationMetrics | None = None
    train_json_path: str | None = None
    notes: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train team-level models (margin, total points, home win) "
        "from team matchup training dataset."
    )
    parser.add_argument(
        "--train-json",
        type=str,
        required=True,
        help="Path to training dataset JSON file produced by build_team_train_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/team",
        help="Directory to save trained models and metadata JSON (default: data/models/team)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use as validation/test split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and models (default: 42)",
    )
    return parser.parse_args()


def load_training_data(path: str) -> pd.DataFrame:
    print(f"📥 Loading training data from: {path}")
    df = pd.read_json(path, orient="records")
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns.")
    return df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Use all numeric/boolean columns except the targets and obvious leakage identifiers.
    """
    print("🔎 Selecting feature columns...")

    # Columns that should NOT be used as features
    drop_cols = {
        "game_id",
        "season",
        "game_date",
        TARGET_MARGIN,
        TARGET_TOTAL,
        TARGET_HOME_WIN,
        # If you add explicit team IDs/strings and want to one-hot later, you can handle that here.
        "home_team",
        "away_team",
        # 🔒 Label leakage: actual final scores should NOT be features
        "home_pts",
        "away_pts",
    }

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    features = [c for c in numeric_cols if c not in drop_cols]

    print(f"   Found {len(features)} feature columns.")
    return sorted(features)


def build_regression_pipeline(random_state: int) -> Pipeline:
    """
    Simple baseline: StandardScaler + GradientBoostingRegressor.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=random_state)),
        ]
    )


def build_classification_pipeline(random_state: int) -> Pipeline:
    """
    Simple baseline: StandardScaler + GradientBoostingClassifier.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=random_state)),
        ]
    )


def train_regression_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    target_name: str,
) -> Tuple[Pipeline, RegressionMetrics, int, int]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_regression_pipeline(random_state=random_state)
    print(
        f"🚀 Training regression model for target='{target_name}' "
        f"on {len(X_train)} train / {len(X_test)} test rows..."
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    # Older versions of scikit-learn don't support `squared=False`, so compute RMSE manually.
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    print(f"   📏 {target_name} MAE:  {mae:.3f}")
    print(f"   📏 {target_name} RMSE: {rmse:.3f}")

    metrics = RegressionMetrics(mae=mae, rmse=rmse)
    return model, metrics, len(X_train), len(X_test)


def train_classification_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    target_name: str,
) -> Tuple[Pipeline, ClassificationMetrics, int, int]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = build_classification_pipeline(random_state=random_state)
    print(
        f"🚀 Training classification model for target='{target_name}' "
        f"on {len(X_train)} train / {len(X_test)} test rows..."
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    brier = float(brier_score_loss(y_test, y_prob))

    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        # ROC-AUC is undefined when only one class present in y_test
        roc_auc = None

    print(f"   📊 {target_name} Accuracy: {acc:.3f}")
    print(f"   📊 {target_name} Brier:    {brier:.3f}")
    if roc_auc is not None:
        print(f"   📊 {target_name} ROC-AUC:  {roc_auc:.3f}")
    else:
        print(f"   📊 {target_name} ROC-AUC:  (undefined on this split)")

    metrics = ClassificationMetrics(accuracy=acc, brier=brier, roc_auc=roc_auc)
    return model, metrics, len(X_train), len(X_test)


def save_model(model: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Saved model to: {path}")


def save_metadata(metadata: Dict[str, ModelMetadata], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {name: asdict(meta) for name, meta in metadata.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"💾 Saved metadata JSON to: {path}")


def main() -> None:
    args = parse_args()

    print("===============================================")
    print("🤖  Training team-level models")
    print("===============================================")
    print(f"📄 Train JSON:  {args.train_json}")
    print(f"📂 Output dir:  {args.output_dir}")
    print(f"📊 Test size:   {args.test_size}")
    print(f"🎲 Random seed: {args.random_state}")
    print("===============================================")

    df = load_training_data(args.train_json)
    feature_cols = select_feature_columns(df)

    # Sanity checks on target columns
    for target in [TARGET_MARGIN, TARGET_TOTAL, TARGET_HOME_WIN]:
        if target not in df.columns:
            raise RuntimeError(
                f"Training dataset missing required target column: '{target}'"
            )

    X = df[feature_cols].copy()
    y_margin = df[TARGET_MARGIN].astype(float)
    y_total = df[TARGET_TOTAL].astype(float)
    y_home_win = df[TARGET_HOME_WIN].astype(int)

    metadata: Dict[str, ModelMetadata] = {}

    # 1) Margin model
    margin_model, margin_metrics, margin_train_n, margin_test_n = (
        train_regression_model(
            X, y_margin, args.test_size, args.random_state, TARGET_MARGIN
        )
    )
    margin_model_path = os.path.join(args.output_dir, "team_margin_model.joblib")
    save_model(margin_model, margin_model_path)
    metadata["margin_model"] = ModelMetadata(
        model_type="GradientBoostingRegressor",
        target=TARGET_MARGIN,
        feature_columns=feature_cols,
        train_rows=margin_train_n,
        test_rows=margin_test_n,
        regression_metrics=margin_metrics,
        classification_metrics=None,
        train_json_path=args.train_json,
        notes="Predicts home minus away points (spread).",
    )

    # 2) Total points model
    total_model, total_metrics, total_train_n, total_test_n = train_regression_model(
        X, y_total, args.test_size, args.random_state, TARGET_TOTAL
    )
    total_model_path = os.path.join(args.output_dir, "team_total_points_model.joblib")
    save_model(total_model, total_model_path)
    metadata["total_points_model"] = ModelMetadata(
        model_type="GradientBoostingRegressor",
        target=TARGET_TOTAL,
        feature_columns=feature_cols,
        train_rows=total_train_n,
        test_rows=total_test_n,
        regression_metrics=total_metrics,
        classification_metrics=None,
        train_json_path=args.train_json,
        notes="Predicts total game points (home + away).",
    )

    # 3) Home win probability model
    homewin_model, homewin_metrics, hw_train_n, hw_test_n = train_classification_model(
        X, y_home_win, args.test_size, args.random_state, TARGET_HOME_WIN
    )
    homewin_model_path = os.path.join(args.output_dir, "team_home_win_model.joblib")
    save_model(homewin_model, homewin_model_path)
    metadata["home_win_model"] = ModelMetadata(
        model_type="GradientBoostingClassifier",
        target=TARGET_HOME_WIN,
        feature_columns=feature_cols,
        train_rows=hw_train_n,
        test_rows=hw_test_n,
        regression_metrics=None,
        classification_metrics=homewin_metrics,
        train_json_path=args.train_json,
        notes="Predicts probability that home team wins.",
    )

    # Save combined metadata
    metadata_path = os.path.join(args.output_dir, "team_models_metadata.json")
    save_metadata(metadata, metadata_path)

    print("🎉 Team models trained and saved successfully.")


if __name__ == "__main__":
    main()
