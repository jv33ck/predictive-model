#!/usr/bin/env python
"""
predict_team_matchups.py

Use trained team-level models to score gameday matchups and produce:
- Predicted margin (home - away)
- Predicted total points
- Home win probability
- Implied home/away scores

Intended to run AFTER export_gameday_matchups.py, using its JSON output as input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load


# --------------------------------------------------------------------
# Metadata / model loading helpers
# --------------------------------------------------------------------


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found at: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_team_models(models_dir: Path) -> Dict[str, Any]:
    """
    Load the three team-level models saved by train_team_models.py.
    """
    models = {}

    margin_path = models_dir / "team_margin_model.joblib"
    total_path = models_dir / "team_total_points_model.joblib"
    home_win_path = models_dir / "team_home_win_model.joblib"

    if not margin_path.exists():
        raise FileNotFoundError(f"Margin model not found at: {margin_path}")
    if not total_path.exists():
        raise FileNotFoundError(f"Total points model not found at: {total_path}")
    if not home_win_path.exists():
        raise FileNotFoundError(f"Home win model not found at: {home_win_path}")

    models["margin"] = joblib_load(margin_path)
    models["total_points"] = joblib_load(total_path)
    models["home_win"] = joblib_load(home_win_path)

    return models


# --------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------


_DIFF_RENAME_MAP: Dict[str, str] = {
    # Map gameday_matchups diff_team_* columns -> training diff_* feature names
    "diff_team_games_played": "diff_games_played",
    "diff_team_wins": "diff_wins",
    "diff_team_losses": "diff_losses",
    "diff_team_win_pct": "diff_win_pct",
    "diff_team_pts_for_per_game": "diff_pts_for_per_game",
    "diff_team_plus_minus_per_game": "diff_plus_minus_per_game",
    "diff_team_recent_wins": "diff_recent_wins",
    "diff_team_recent_losses": "diff_recent_losses",
    "diff_team_recent_plus_minus_per_game": "diff_recent_plus_minus_per_game",
}


def load_matchups(matchups_json: Path) -> pd.DataFrame:
    if not matchups_json.exists():
        raise FileNotFoundError(f"Matchups JSON not found at: {matchups_json}")
    df = pd.read_json(matchups_json)
    if df.empty:
        raise ValueError(f"Matchups DataFrame is empty: {matchups_json}")
    return df


def align_feature_columns(
    matchups_df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    - Renames diff_team_* columns into diff_* to match training.
    - Ensures all required feature columns are present and numeric.
    - Returns a feature matrix X with the correct column order.
    """
    df = matchups_df.copy()

    # 1) Rename diff_team_* -> diff_* where applicable
    rename_cols_present = {
        src: dst for src, dst in _DIFF_RENAME_MAP.items() if src in df.columns
    }
    if rename_cols_present:
        df = df.rename(columns=rename_cols_present)

    # 2) Check for missing required feature columns
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        sample_cols = sorted(df.columns.tolist())
        raise ValueError(
            "Matchups DataFrame is missing required feature columns.\n"
            f"  Missing: {missing}\n"
            f"  Available columns (sample): {sample_cols}"
        )

    # 3) Build X in the correct order, ensure numeric
    X = df[feature_columns].copy()

    # Force numeric dtype where possible; invalid parses become NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # Handle NaNs (simple strategy: fill with 0; we can improve later)
    if X.isna().any().any():
        X = X.fillna(0.0)

    return X


# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------


def predict_team_matchups(
    matchups_df: pd.DataFrame,
    models: Dict[str, Any],
    metadata: Dict[str, Any],
) -> pd.DataFrame:
    """
    Given a matchups DataFrame, models, and metadata, produce predictions
    and return a new DataFrame with identifiers + predictions.
    """
    margin_meta = metadata["margin_model"]
    total_meta = metadata["total_points_model"]
    home_win_meta = metadata["home_win_model"]

    margin_features = margin_meta["feature_columns"]
    total_features = total_meta["feature_columns"]
    home_win_features = home_win_meta["feature_columns"]

    # For now, we assume all three models share the same feature set.
    # If they ever diverge, we can build separate X matrices.
    feature_columns = sorted(
        set(margin_features) | set(total_features) | set(home_win_features)
    )

    X = align_feature_columns(matchups_df, feature_columns)

    # 1) Predict margin (home - away)
    margin_model = models["margin"]
    margin_pred = margin_model.predict(X).astype(float)

    # 2) Predict total points
    total_model = models["total_points"]
    total_pred = total_model.predict(X).astype(float)

    # 3) Predict home win probability
    home_win_model = models["home_win"]
    # Use proba for class "1" (home win)
    if hasattr(home_win_model, "predict_proba"):
        proba = home_win_model.predict_proba(X)
        if proba.shape[1] == 2:
            home_win_prob = proba[:, 1].astype(float)
        else:
            # Fallback: if classes are weird, just take max prob
            home_win_prob = proba.max(axis=1).astype(float)
    else:
        # Fallback: treat decision_function / predict output as score,
        # squashed into (0,1) range via simple transform.
        scores = home_win_model.decision_function(X).astype(float)
        home_win_prob = 1.0 / (1.0 + np.exp(-scores))

    # 4) Implied scores
    home_score_pred = (total_pred + margin_pred) / 2.0
    away_score_pred = (total_pred - margin_pred) / 2.0

    # Build output DataFrame with identifiers + predictions
    id_cols = []
    for c in ["game_id", "season", "date", "home_team", "away_team"]:
        if c in matchups_df.columns:
            id_cols.append(c)

    out = matchups_df[id_cols].copy()

    out["pred_margin"] = margin_pred
    out["pred_total_points"] = total_pred
    out["pred_home_win_prob"] = home_win_prob
    out["pred_home_score"] = home_score_pred
    out["pred_away_score"] = away_score_pred

    # Nice: also add implied moneyline-ish logit for debugging / future use
    with np.errstate(divide="ignore", invalid="ignore"):
        odds = np.where(
            (home_win_prob > 0) & (home_win_prob < 1),
            home_win_prob / (1.0 - home_win_prob),
            np.nan,
        )
    out["pred_home_win_odds"] = odds

    return out


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict team-level outcomes for gameday matchups "
        "using trained team models."
    )

    parser.add_argument(
        "--season-label",
        type=str,
        required=False,
        help="Season label (e.g. 2025-26). Used for logging only.",
    )
    parser.add_argument(
        "--date",
        type=str,
        required=False,
        help="Gameday date in YYYY-MM-DD. Used for logging and default paths.",
    )
    parser.add_argument(
        "--matchups-json",
        type=str,
        required=False,
        help=(
            "Path to gameday_matchups JSON file. "
            "If omitted, will default to "
            "data/exports/gameday_matchups_<date>.json"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models/team",
        help="Directory containing team_*_model.joblib files.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="data/models/team/team_models_metadata.json",
        help="Path to team_models_metadata.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=False,
        help=(
            "Output CSV path. If omitted, defaults to "
            "data/exports/gameday_team_predictions_<date>.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=False,
        help=(
            "Output JSON path. If omitted, defaults to "
            "data/exports/gameday_team_predictions_<date>.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    season_label = args.season_label or "(unknown)"
    date_str = args.date or "(unknown date)"

    print("===============================================")
    print("📡  Team matchup predictions")
    print("===============================================")
    print(f"📆 Season label: {season_label}")
    print(f"📅 Date:         {date_str}")
    print(f"📂 Models dir:   {args.models_dir}")
    print(f"📄 Metadata:     {args.metadata_json}")
    print("===============================================\n")

    models_dir = Path(args.models_dir)
    metadata_path = Path(args.metadata_json)

    # Infer default paths from date if not provided
    if args.matchups_json:
        matchups_path = Path(args.matchups_json)
    else:
        if args.date is None:
            raise ValueError(
                "Either --matchups-json must be provided, or --date must be set "
                "to infer the default path."
            )
        matchups_path = Path("data/exports") / f"gameday_matchups_{args.date}.json"

    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        if args.date is None:
            output_csv = Path("data/exports/gameday_team_predictions.csv")
        else:
            output_csv = (
                Path("data/exports") / f"gameday_team_predictions_{args.date}.csv"
            )

    if args.output_json:
        output_json = Path(args.output_json)
    else:
        if args.date is None:
            output_json = Path("data/exports/gameday_team_predictions.json")
        else:
            output_json = (
                Path("data/exports") / f"gameday_team_predictions_{args.date}.json"
            )

    print(f"📥 Loading gameday matchups from: {matchups_path}")
    matchups_df = load_matchups(matchups_path)
    print(f"   Loaded {len(matchups_df)} matchup rows.\n")

    print("📥 Loading team models and metadata...")
    metadata = load_metadata(metadata_path)
    models = load_team_models(models_dir)
    print("✅ Models and metadata loaded.\n")

    print("🔮 Generating predictions for matchups...")
    preds_df = predict_team_matchups(matchups_df, models, metadata)
    print(f"✅ Generated predictions for {len(preds_df)} games.\n")

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Writing predictions CSV to: {output_csv}")
    preds_df.to_csv(output_csv, index=False)

    print(f"💾 Writing predictions JSON to: {output_json}")
    preds_df.to_json(output_json, orient="records")

    print("\n🎉 Team matchup predictions export complete.")


if __name__ == "__main__":
    main()
