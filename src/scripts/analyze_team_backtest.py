#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


def load_backtest_results(path: Path) -> pd.DataFrame:
    """
    Load per-game backtest results from a JSON file produced by backtest_team_models.py.
    """
    if not path.exists():
        raise FileNotFoundError(f"Backtest JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    required_cols = [
        "game_id",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "actual_margin",
        "pred_margin",
        "actual_total_points",
        "pred_total_points",
        "actual_home_win",
        "pred_home_win_prob",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Backtest DataFrame is missing required columns: {missing}")

    # Ensure types are sensible
    df["game_date"] = pd.to_datetime(df["game_date"])
    numeric_cols = [
        "actual_margin",
        "pred_margin",
        "actual_total_points",
        "pred_total_points",
        "actual_home_win",
        "pred_home_win_prob",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_overall_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute overall regression + classification metrics for the entire backtest window.
    """
    metrics: Dict[str, float] = {}

    # Drop rows with NaNs in relevant columns
    reg_mask = (
        df[["actual_margin", "pred_margin", "actual_total_points", "pred_total_points"]]
        .notna()
        .all(axis=1)
    )
    cls_mask = df[["actual_home_win", "pred_home_win_prob"]].notna().all(axis=1)

    reg_df = df.loc[reg_mask].copy()
    cls_df = df.loc[cls_mask].copy()

    # Regression metrics
    metrics["margin_mae"] = float(
        mean_absolute_error(reg_df["actual_margin"], reg_df["pred_margin"])
    )
    metrics["margin_rmse"] = float(
        np.sqrt(mean_squared_error(reg_df["actual_margin"], reg_df["pred_margin"]))
    )

    metrics["total_points_mae"] = float(
        mean_absolute_error(reg_df["actual_total_points"], reg_df["pred_total_points"])
    )
    metrics["total_points_rmse"] = float(
        np.sqrt(
            mean_squared_error(
                reg_df["actual_total_points"], reg_df["pred_total_points"]
            )
        )
    )

    # Classification metrics
    y_true = cls_df["actual_home_win"].astype(int)
    y_prob = cls_df["pred_home_win_prob"].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics["home_win_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["home_win_brier"] = float(brier_score_loss(y_true, y_prob))

    # ROC-AUC can fail if only one class present; guard against that
    try:
        metrics["home_win_roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["home_win_roc_auc"] = float("nan")

    return metrics


def compute_group_metrics(
    df: pd.DataFrame, group_col: str, min_games: int = 10
) -> pd.DataFrame:
    """
    Compute per-group metrics (e.g. by home_team, away_team, or game_date).

    Only groups with at least `min_games` rows are included.
    """
    rows: List[Dict[str, object]] = []

    for group_value, g in df.groupby(group_col):
        if len(g) < min_games:
            continue

        m = compute_overall_metrics(g)
        row: Dict[str, object] = {
            group_col: group_value,
            "games": len(g),
        }
        row.update(m)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("games", ascending=False)


def show_top_errors(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Return the top-N games by absolute margin error.
    """
    df = df.copy()
    df["margin_error"] = (df["pred_margin"] - df["actual_margin"]).abs()
    df["total_points_error"] = (
        df["pred_total_points"] - df["actual_total_points"]
    ).abs()
    df = df.sort_values("margin_error", ascending=False).head(n)

    # Order columns nicely for inspection
    cols = [
        "game_date",
        "home_team",
        "away_team",
        "actual_margin",
        "pred_margin",
        "margin_error",
        "actual_total_points",
        "pred_total_points",
        "total_points_error",
        "actual_home_win",
        "pred_home_win_prob",
        "game_id",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze backtest results for team-level models."
    )
    parser.add_argument(
        "--backtest-json",
        type=str,
        required=True,
        help="Path to backtest JSON file produced by backtest_team_models.py",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional path to write per-group summary CSV (e.g. by team).",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="home_team",
        choices=["home_team", "away_team", "game_date"],
        help="Column to group by for per-group metrics.",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=10,
        help="Minimum games per group to include in per-group metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backtest_path = Path(args.backtest_json)

    print("===============================================")
    print("📊  Analyzing team backtest results")
    print("===============================================")
    print(f"📄 Backtest JSON: {backtest_path}")
    print(f"📊 Group-by:      {args.group_by}")
    print(f"📊 Min games:     {args.min_games}")
    print("===============================================")

    df = load_backtest_results(backtest_path)
    print(
        f"✅ Loaded {len(df)} backtest rows spanning "
        f"{df['game_date'].min().date()} → {df['game_date'].max().date()}."
    )

    # Overall metrics
    overall = compute_overall_metrics(df)
    print("\n📈 Overall performance:")
    print(f"   🧮 margin MAE:         {overall['margin_mae']:.3f}")
    print(f"   🧮 margin RMSE:        {overall['margin_rmse']:.3f}")
    print(f"   🧮 total_points MAE:   {overall['total_points_mae']:.3f}")
    print(f"   🧮 total_points RMSE:  {overall['total_points_rmse']:.3f}")
    print(f"   🧮 home_win Accuracy:  {overall['home_win_accuracy']:.3f}")
    print(f"   🧮 home_win Brier:     {overall['home_win_brier']:.3f}")
    roc_auc = overall.get("home_win_roc_auc", float("nan"))
    if np.isfinite(roc_auc):
        print(f"   🧮 home_win ROC-AUC:   {roc_auc:.3f}")
    else:
        print("   🧮 home_win ROC-AUC:   (undefined – only one class present)")

    # Group metrics
    group_df = compute_group_metrics(
        df, group_col=args.group_by, min_games=args.min_games
    )
    if not group_df.empty:
        print(f"\n📊 Per-{args.group_by} metrics (min_games={args.min_games}):")
        # Show top 10 groups by games
        display_cols = [
            args.group_by,
            "games",
            "margin_mae",
            "margin_rmse",
            "total_points_mae",
            "total_points_rmse",
            "home_win_accuracy",
            "home_win_brier",
            "home_win_roc_auc",
        ]
        display_cols = [c for c in display_cols if c in group_df.columns]
        print(group_df[display_cols].head(10).to_string(index=False))

        if args.output_csv:
            out_path = Path(args.output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            group_df.to_csv(out_path, index=False)
            print(f"\n💾 Wrote per-{args.group_by} summary metrics to: {out_path}")
    else:
        print(
            f"\nℹ️ No groups with at least {args.min_games} games for group_by='{args.group_by}'."
        )

    # Top errors
    print("\n🔍 Top 20 games by absolute margin error:")
    top_err = show_top_errors(df, n=20)
    print(top_err.to_string(index=False))


if __name__ == "__main__":
    main()
