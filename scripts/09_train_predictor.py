"""
Train an XGBoost model to predict next time-window vehicle_count or congestion_score.

v4 Upgrades:
 - Ingests enriched congestion columns: weighted_vehicle_load, load_index, rush_ratio, speed_var, weighted_speed.
 - Adds interaction features (vc_x_speed, congestion_proxy).
 - Global and per-edge safeguards for missing columns.
 - Safer edge-wise R² computation (avoids pandas deprecation behavior).
 - Saves model, scaler, and detailed metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib, json, warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

INPUT_FILE = DATA / "edges_congestion.csv"
MODEL_FILE = MODELS / "congestion_predictor.pkl"
SCALER_FILE = MODELS / "feature_scaler.pkl"
META_FILE = MODELS / "congestion_predictor_meta.json"

# ==============================
# LOAD DATA
# ==============================
if not INPUT_FILE.exists():
    print("❌ No congestion data found. Run 03_aggregate_congestion.py first.")
    raise SystemExit

df = pd.read_csv(INPUT_FILE, parse_dates=["time_window"])
if df.empty:
    print("❌ Congestion dataset is empty. Run simulation before training.")
    raise SystemExit

print(f"📊 Loaded {len(df)} records from {INPUT_FILE}")

# ==============================
# PREPROCESSING
# ==============================
df = df.sort_values(["edge_u", "edge_v", "edge_key", "time_window"]).reset_index(drop=True)

# Add numeric time index (each window = 5 min = 300s)
df["time_idx"] = ((df["time_window"] - df["time_window"].min()).dt.total_seconds() // 300).astype(int)

# Ensure enriched static/dynamic features exist; if missing, fill sensible defaults
fallbacks = {
    "length": 200.0,
    "highway": "residential",
    "weighted_vehicle_load": 0.0,
    "load_index": 0.0,
    "rush_ratio": 0.0,
    "speed_var": 0.0,
    "weighted_speed": np.nan # use avg_speed fallback later
}

for col, val in fallbacks.items():
    if col not in df.columns:
        df[col] = val

# fill weighted_speed with avg_speed if missing
if df["weighted_speed"].isna().any():
    df["weighted_speed"] = df["avg_speed"]

# highway code
df["highway_code"] = pd.Categorical(df["highway"]).codes

# ==============================
# FEATURE ENGINEERING
# ==============================
features = []
for (u, v, k), g in df.groupby(["edge_u", "edge_v", "edge_key"], sort=False):
    g = g.reset_index(drop=True)
    if len(g) < 5:
        continue

    # Basic lags & diffs for counts and speed
    g["vc_lag1"] = g["vehicle_count"].shift(1)
    g["vc_lag2"] = g["vehicle_count"].shift(2)
    g["vc_diff"] = g["vehicle_count"].diff().fillna(0)

    g["speed_lag1"] = g["avg_speed"].shift(1)
    g["speed_lag2"] = g["avg_speed"].shift(2)
    g["speed_diff"] = g["avg_speed"].diff().fillna(0)

    # Rolling trends
    g["vc_roll3"] = g["vehicle_count"].rolling(3, min_periods=1).mean()
    g["vc_roll5"] = g["vehicle_count"].rolling(5, min_periods=1).mean()
    g["speed_roll3"] = g["avg_speed"].rolling(3, min_periods=1).mean()
    g["speed_roll5"] = g["avg_speed"].rolling(5, min_periods=1).mean()

    # Ratios (relative trend indicators)
    g["vc_ratio"] = g["vehicle_count"] / (g["vc_roll3"] + 1e-6)
    g["speed_ratio"] = g["avg_speed"] / (g["speed_roll3"] + 1e-6)

    # Use enriched aggregation columns when present
    g["weighted_vehicle_load"] = g.get("weighted_vehicle_load", g["vehicle_count"]).fillna(g["vehicle_count"])
    g["load_index"] = g.get("load_index", g["weighted_vehicle_load"] / (g["vehicle_count"] + 1e-6)).fillna(0.0)
    g["rush_ratio"] = g.get("rush_ratio", 0.0).fillna(0.0)
    g["speed_var"] = g.get("speed_var", 0.0).fillna(0.0)
    g["weighted_speed"] = g.get("weighted_speed", g["avg_speed"]).fillna(g["avg_speed"])

    # Interaction features
    g["vc_x_speed"] = g["vehicle_count"] * g["avg_speed"]
    g["congestion_proxy"] = g["vc_ratio"] / (g["speed_ratio"] + 1e-6)

    # Target: next time window (vehicle_count)
    g["label"] = g["vehicle_count"].shift(-1)
    g = g.dropna()
    features.append(g)

if not features:
    print("⚠️ Not enough data for lag features — try longer simulation (≥12 hours) or increase vehicle variety.")
    raise SystemExit

df_feat = pd.concat(features, ignore_index=True)
print(f"🧩 Feature matrix built: {len(df_feat)} rows")

# ==============================
# TARGET: choose normalization strategy
# If per-edge normalization gives near-zero variance across many edges, fall back to global normalization.
# ==============================
per_edge_std = df_feat.groupby(["edge_u", "edge_v"])["label"].transform("std")
if (per_edge_std.fillna(0) < 1e-6).mean() > 0.5:
    # too many edges near-constant -> use global normalization
    print("⚠️ Many edges have near-constant target; using GLOBAL target normalization.")
    y = (df_feat["label"] - df_feat["label"].mean()) / (df_feat["label"].std() + 1e-6)
    target_mode = "global_norm"
else:
    # use per-edge normalization
    df_feat["label_norm"] = df_feat.groupby(["edge_u", "edge_v"])["label"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    y = df_feat["label_norm"]
    target_mode = "per_edge_norm"

# ==============================
# FEATURE SETUP
# ==============================
feature_cols = [
    "vehicle_count", "vc_lag1", "vc_lag2", "vc_diff", "vc_roll3", "vc_roll5",
    "avg_speed", "speed_lag1", "speed_lag2", "speed_diff", "speed_roll3", "speed_roll5",
    "vc_ratio", "speed_ratio", "length", "highway_code",
    "weighted_vehicle_load", "load_index", "rush_ratio", "speed_var", "weighted_speed",
    "vc_x_speed", "congestion_proxy"
]

# Ensure columns exist and convert to float
for c in feature_cols:
    if c not in df_feat.columns:
        df_feat[c] = 0.0

X = df_feat[feature_cols].astype(float)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chronological split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# ==============================
# MODEL TRAINING
# ==============================
print("🚀 Training XGBoost (temporal tuned)...")
model = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.015,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_lambda=8.0,
    reg_alpha=2.0,
    gamma=0.2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained — MAE: {mae:.4f}, R²: {r2:.4f} (target_mode={target_mode})")

# Edge-wise R² evaluation (safer: compute per-edge on the rows that belong to that edge)
edge_r2_list = []
grouped = df_feat.groupby(["edge_u", "edge_v", "edge_key"], sort=False)
for (eu, ev, ek), grp in grouped:
    if len(grp) < 5:
        continue
    try:
        Xg = scaler.transform(grp[feature_cols].astype(float))
        yg_true = grp["label"].values if target_mode == "global_norm" else grp["label_norm"].values
        yg_pred = model.predict(Xg)
        r = r2_score(yg_true, yg_pred)
        if np.isfinite(r):
            edge_r2_list.append(r)
    except Exception:
        continue

median_edge_r2 = float(np.median(edge_r2_list)) if edge_r2_list else 0.0
print(f"📊 Median Edge-wise R²: {median_edge_r2:.3f}")

# Feature importances
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n📈 Top Feature Importances:")
print(importances.head(12))

# ==============================
# SAVE MODEL + METADATA
# ==============================
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

meta = {
    "model_file": str(MODEL_FILE),
    "training_rows": len(df_feat),
    "features": feature_cols,
    "mae": float(mae),
    "r2": float(r2),
    "median_edge_r2": median_edge_r2,
    "top_features": importances.to_dict(),
    "target_mode": target_mode
}

with open(META_FILE, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n💾 Model saved to {MODEL_FILE}")
print(f"📏 Scaler saved to {SCALER_FILE}")
print(f"🧠 Metadata saved to {META_FILE}")
