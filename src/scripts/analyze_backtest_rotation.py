import json
import numpy as np
import pandas as pd

bt_path = "data/eval/player_backtest_results_2025-26_oct-nov.json"
train_path = "data/training/player_train_2025-26_oct-nov.json"

# --- Load backtest results ---
with open(bt_path, "r", encoding="utf-8") as f:
    bt_data = json.load(f)

# Backtest JSON is usually {"rows": [...]}, but be robust to plain list
if isinstance(bt_data, dict) and "rows" in bt_data:
    bt_rows = bt_data["rows"]
else:
    bt_rows = bt_data

bt = pd.DataFrame(bt_rows)

# --- Load training data to get minutes features ---
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

# Training JSON from build_player_train_dataset is a plain list of records
if isinstance(train_data, dict) and "rows" in train_data:
    train_rows = train_data["rows"]
else:
    train_rows = train_data


train = pd.DataFrame(train_rows)

# --- Normalize key dtypes for safe merge ---
# Ensure game_id, team, player_id, and game_date are strings in both backtest and training frames
for df_norm in (bt, train):
    for col in ["game_id", "team", "player_id", "game_date"]:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col].astype(str)

# Keep only keys we need from training
train_keys = train[
    ["game_date", "team", "player_id", "minutes_played", "minutes_played_prev_mean_all"]
]

# --- Join features onto backtest rows ---
merged = bt.merge(
    train_keys,
    on=["game_date", "team", "player_id"],
    how="left",
    suffixes=("", "_feat"),
)

# Sanity: how many rows lost minutes info?
missing_minutes = merged["minutes_played_prev_mean_all"].isna().sum()
print(f"Rows missing minutes_played_prev_mean_all after merge: {missing_minutes}")

# --- Define rotation masks ---
rot_by_avg = merged["minutes_played_prev_mean_all"] >= 15
rot_by_game = merged["minutes_played"] >= 15


def summarize_slice(df: pd.DataFrame, label: str) -> None:
    print(f"\n===== {label} =====")
    print(f"Rows in slice: {len(df)}")
    for target in ["pts", "treb", "ast"]:
        mae_model = df[f"abs_error_{target}"].mean()
        mae_base = df[f"baseline_abs_error_{target}"].mean()
        rmse_model = np.sqrt((df[f"error_{target}"] ** 2).mean())
        rmse_base = np.sqrt((df[f"baseline_error_{target}"] ** 2).mean())
        print(
            f"{target:4} "
            f"MAE  model={mae_model:6.3f}  baseline={mae_base:6.3f} | "
            f"RMSE model={rmse_model:6.3f}  baseline={rmse_base:6.3f}"
        )


# Rotation defined by average minutes >= 15
summarize_slice(merged[rot_by_avg].copy(), "Rotation (avg minutes >= 15)")

# Optional: rotation defined by actual game minutes >= 15
summarize_slice(merged[rot_by_game].copy(), "Rotation (this game minutes >= 15)")
