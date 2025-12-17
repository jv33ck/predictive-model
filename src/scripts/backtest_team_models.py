#!/usr/bin/env python
"""
backtest_team_models.py

Walk-forward backtest for team-level models using the team_matchups_train_*.json
dataset produced by build_team_train_dataset.py.

For each game date in a specified window:
- Train models ONLY on games before that date.
- Predict margin, total_points, and home_win probability for games on that date.
- Compare predictions to actuals and aggregate metrics.

This simulates "what our model would have predicted on that day, using only
information available up to that point".
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    train_json: Path
    output_csv: Path
    output_json: Optional[Path]
    season_label: Optional[str]
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]
    min_train_games: int
    random_state: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> BacktestConfig:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for team-level matchup models."
    )

    parser.add_argument(
        "--train-json",
        type=str,
        required=True,
        help="Path to team training dataset JSON (from build_team_train_dataset.py).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/eval/team_backtest_results.csv",
        help="Output CSV for per-game backtest results.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output JSON for per-game backtest results.",
    )
    parser.add_argument(
        "--season-label",
        type=str,
        default=None,
        help="Optional season label for logging / filtering.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD). If omitted, uses earliest in data.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for backtest (YYYY-MM-DD). If omitted, uses latest in data.",
    )
    parser.add_argument(
        "--min-train-games",
        type=int,
        default=100,
        help="Minimum number of training games required before making predictions.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for models.",
    )

    args = parser.parse_args()

    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None

    return BacktestConfig(
        train_json=Path(args.train_json),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json) if args.output_json else None,
        season_label=args.season_label,
        start_date=start_date,
        end_date=end_date,
        min_train_games=args.min_train_games,
        random_state=args.random_state,
    )


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select numeric feature columns from the training dataset.

    We treat all numeric columns as candidate features, then drop the target columns
    and obvious identifiers/meta columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    target_cols = {"margin", "total_points", "home_win"}
    meta_cols = {
        "game_id",
        "game_timestamp",
        "season",
        "season_label",
        # sometimes dates are stored numerically; we rely on 'game_date' as datetime
    }

    feature_cols = [c for c in numeric_cols if c not in target_cols | meta_cols]
    return feature_cols


def make_regressor(random_state: int) -> RandomForestRegressor:
    """
    Construct a regression model. Keep this reasonably close in spirit to
    the train_team_models.py choices (RF + sensible defaults).
    """
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )


def make_classifier(random_state: int) -> RandomForestClassifier:
    """
    Construct a classification model for home_win.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )


def compute_rmse(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """
    Compute RMSE using mean_squared_error (no 'squared' kwarg to keep
    compatibility with older sklearn). Accepts any sequence-like numeric
    inputs and converts them to numpy arrays for type-checker friendliness.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))


# ---------------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------------


def run_backtest(cfg: BacktestConfig) -> pd.DataFrame:
    print("===============================================")
    print("🔁  Backtesting team-level models (walk-forward)")
    print("===============================================")
    print(f"📄 Train JSON:   {cfg.train_json}")
    print(f"📂 Output CSV:   {cfg.output_csv}")
    if cfg.output_json:
        print(f"📂 Output JSON:  {cfg.output_json}")
    if cfg.season_label:
        print(f"📆 Season label: {cfg.season_label}")
    if cfg.start_date or cfg.end_date:
        print(
            f"📅 Date window:  "
            f"{cfg.start_date.date() if cfg.start_date else 'min'} → "
            f"{cfg.end_date.date() if cfg.end_date else 'max'}"
        )
    print(f"📊 Min train games: {cfg.min_train_games}")
    print(f"🎲 Random seed:     {cfg.random_state}")
    print("===============================================")

    # 1) Load data
    df = pd.read_json(cfg.train_json)
    if "game_date" not in df.columns:
        raise RuntimeError("Expected 'game_date' column in training dataset JSON.")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    original_len = len(df)
    print(f"📥 Loaded {original_len} rows from training JSON.")

    # Optional season filter if season column present and season_label specified
    if cfg.season_label and "season" in df.columns:
        before = len(df)
        df = df[df["season"] == cfg.season_label].copy()
        print(f"🔎 Filtered by season='{cfg.season_label}': {before} → {len(df)} rows.")

    # 2) Apply date window
    if cfg.start_date is not None:
        df = df[df["game_date"] >= cfg.start_date].copy()
    if cfg.end_date is not None:
        df = df[df["game_date"] <= cfg.end_date].copy()

    print(f"📅 After date filtering, {len(df)} rows remain.")

    if len(df) == 0:
        raise RuntimeError("No games remain after filtering; cannot backtest.")

    # 3) Sort by date
    df = df.sort_values("game_date").reset_index(drop=True)

    # 4) Feature columns
    feature_cols = select_feature_columns(df)
    print(f"🔧 Using {len(feature_cols)} feature columns:")
    for c in feature_cols:
        print(f"   - {c}")

    # 5) Walk-forward loop
    all_dates = sorted(df["game_date"].unique())
    results: List[dict] = []

    print(f"📆 Found {len(all_dates)} distinct game dates in window.")

    for current_date in all_dates:
        date_str = current_date.date().isoformat()
        # Training: all games strictly before current_date
        train_mask = df["game_date"] < current_date
        test_mask = df["game_date"] == current_date

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        if n_test == 0:
            continue

        print(f"\n📅 Backtest date {date_str}: {n_test} games, {n_train} train rows")

        if n_train < cfg.min_train_games:
            print(
                f"   ⚠️ Not enough training games (< {cfg.min_train_games}); "
                f"skipping this date."
            )
            continue

        df_train = df.loc[train_mask]
        df_test = df.loc[test_mask]

        X_train = df_train[feature_cols].values.astype(float)
        X_test = df_test[feature_cols].values.astype(float)

        # Targets
        y_margin = df_train["margin"].values.astype(float)
        y_total = df_train["total_points"].values.astype(float)
        y_home_win = df_train["home_win"].values.astype(int)

        # Regression models
        reg_margin = make_regressor(cfg.random_state)
        reg_total = make_regressor(cfg.random_state)

        print("   🚀 Training regression models (margin, total_points)...")
        reg_margin.fit(X_train, y_margin)
        reg_total.fit(X_train, y_total)

        # Classification model (home_win), careful with class imbalance / lack of classes
        clf_home_win = None
        baseline_prob = float(y_home_win.mean()) if len(y_home_win) > 0 else 0.5
        if len(np.unique(y_home_win)) >= 2:
            clf_home_win = make_classifier(cfg.random_state)
            print("   🚀 Training classification model (home_win)...")
            clf_home_win.fit(X_train, y_home_win)
        else:
            print(
                "   ⚠️ Only one class present in y_home_win; "
                f"using baseline prob={baseline_prob:.3f} for this date."
            )

        # Predictions for each game on this date
        y_margin_true = df_test["margin"].values.astype(float)
        y_total_true = df_test["total_points"].values.astype(float)
        y_home_win_true = df_test["home_win"].values.astype(int)

        y_margin_pred = reg_margin.predict(X_test)
        y_total_pred = reg_total.predict(X_test)

        if clf_home_win is not None:
            y_home_win_prob = clf_home_win.predict_proba(X_test)[:, 1]
        else:
            y_home_win_prob = np.full(shape=len(df_test), fill_value=baseline_prob)

        # Record per-game results
        for idx, row in df_test.iterrows():
            i = df_test.index.get_loc(idx)

            results.append(
                {
                    "game_id": row.get("game_id"),
                    "game_date": row["game_date"].date().isoformat(),
                    "season": row.get("season", cfg.season_label),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "actual_margin": float(y_margin_true[i]),
                    "pred_margin": float(y_margin_pred[i]),
                    "actual_total_points": float(y_total_true[i]),
                    "pred_total_points": float(y_total_pred[i]),
                    "actual_home_win": int(y_home_win_true[i]),
                    "pred_home_win_prob": float(y_home_win_prob[i]),
                }
            )

    if not results:
        raise RuntimeError(
            "Backtest produced no results. Check filters/min_train_games."
        )

    results_df = pd.DataFrame(results)
    print(f"\n✅ Backtest produced {len(results_df)} per-game rows.")

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    print("\n📊 Backtest performance (overall):")

    # Pre-extract numeric arrays with explicit dtypes for type-checker clarity
    actual_margin = results_df["actual_margin"].to_numpy(dtype=float)
    pred_margin = results_df["pred_margin"].to_numpy(dtype=float)
    actual_total = results_df["actual_total_points"].to_numpy(dtype=float)
    pred_total = results_df["pred_total_points"].to_numpy(dtype=float)
    actual_home_win = results_df["actual_home_win"].to_numpy(dtype=int)
    pred_home_win_prob = results_df["pred_home_win_prob"].to_numpy(dtype=float)

    # Margin metrics
    margin_mae = float(np.mean(np.abs(actual_margin - pred_margin)))
    margin_rmse = compute_rmse(actual_margin, pred_margin)
    print(f"   🧮 margin MAE:  {margin_mae:.3f}")
    print(f"   🧮 margin RMSE: {margin_rmse:.3f}")

    # Total points metrics
    total_mae = float(np.mean(np.abs(actual_total - pred_total)))
    total_rmse = compute_rmse(actual_total, pred_total)
    print(f"   🧮 total_points MAE:  {total_mae:.3f}")
    print(f"   🧮 total_points RMSE: {total_rmse:.3f}")

    # Home win metrics
    try:
        pred_home_win_label = (pred_home_win_prob >= 0.5).astype(int)
        acc = float(accuracy_score(actual_home_win, pred_home_win_label))
        brier = float(brier_score_loss(actual_home_win, pred_home_win_prob))
        if np.unique(actual_home_win).size > 1:
            roc = float(roc_auc_score(actual_home_win, pred_home_win_prob))
        else:
            roc = float("nan")
    except Exception as e:  # pragma: no cover - safety
        print(f"   ⚠️ Error computing classification metrics: {e}")
        acc, brier, roc = float("nan"), float("nan"), float("nan")

    print(f"   🧮 home_win Accuracy: {acc:.3f}")
    print(f"   🧮 home_win Brier:    {brier:.3f}")
    print(f"   🧮 home_win ROC-AUC:  {roc:.3f}")

    # Attach metrics as attributes (useful if caller wants them)
    results_df.attrs["metrics"] = {
        "margin_mae": margin_mae,
        "margin_rmse": margin_rmse,
        "total_points_mae": total_mae,
        "total_points_rmse": total_rmse,
        "home_win_accuracy": acc,
        "home_win_brier": brier,
        "home_win_roc_auc": roc,
    }

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    results_df = run_backtest(cfg)

    # Ensure output directory exists
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Writing per-game backtest results to: {cfg.output_csv}")
    results_df.to_csv(cfg.output_csv, index=False)

    if cfg.output_json:
        print(f"💾 Writing per-game backtest results JSON to: {cfg.output_json}")
        results_df.to_json(cfg.output_json, orient="records")

    metrics = results_df.attrs.get("metrics", {})
    if metrics:
        print("\n📌 Summary metrics saved in DataFrame.attrs['metrics']:")
        for k, v in metrics.items():
            print(
                f"   - {k}: {v:.4f}"
                if isinstance(v, (int, float))
                else f"   - {k}: {v}"
            )

    print("\n🎉 Backtest completed.")


if __name__ == "__main__":
    main()
