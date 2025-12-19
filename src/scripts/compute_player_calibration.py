import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute calibration stats for player projections from training data."
    )
    parser.add_argument(
        "--train-json",
        type=str,
        required=True,
        help="Path to player training JSON (from build_player_train_dataset.py)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Where to write calibration stats JSON.",
    )
    parser.add_argument(
        "--min-avg-minutes",
        type=float,
        default=15.0,
        help="Min minutes_played_prev_mean_all to treat as rotation player.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_json)
    out_path = Path(args.output_json)

    print("===============================================")
    print("📏  Computing player calibration stats")
    print("===============================================")
    print(f"📄 Train JSON:      {train_path}")
    print(f"📂 Output JSON:     {out_path}")
    print(f"⏱️ Min avg minutes: {args.min_avg_minutes}")
    print("===============================================")

    if not train_path.exists():
        raise FileNotFoundError(f"Training JSON not found: {train_path}")

    # Training JSON from build_player_train_dataset.py is a plain list of records
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} training rows.")

    # Try to define a "rotation" slice
    if "minutes_played_prev_mean_all" in df.columns:
        rot_mask = df["minutes_played_prev_mean_all"] >= args.min_avg_minutes
        source_col = "minutes_played_prev_mean_all"
    elif "minutes_played" in df.columns:
        rot_mask = df["minutes_played"] >= args.min_avg_minutes
        source_col = "minutes_played"
    else:
        print("⚠️ No minutes column found; using all rows for calibration.")
        rot_mask = pd.Series(True, index=df.index)
        source_col = None

    rot = df[rot_mask].copy()
    print(
        f"🧮 Using {len(rot)} rows for calibration "
        f"(source minutes column={source_col})"
    )

    stats: dict[str, dict[str, float]] = {}
    for target in ["pts", "treb", "ast"]:
        if target not in rot.columns:
            print(f"⚠️ Target column '{target}' not found in training data; skipping.")
            continue

        col = rot[target].astype(float)
        mean_actual = float(col.mean())
        std_actual = float(col.std(ddof=0))  # population std

        stats[target] = {
            "mean_actual": mean_actual,
            "std_actual": std_actual,
        }
        print(
            f"   {target}: mean_actual={mean_actual:.3f}, "
            f"std_actual={std_actual:.3f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n💾 Calibration stats written to: {out_path}")
    print("🎉 Done.")


if __name__ == "__main__":
    main()
