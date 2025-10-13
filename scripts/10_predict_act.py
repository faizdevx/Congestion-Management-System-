"""
Predictive congestion control loop:
 - Load latest congestion snapshot
 - Use trained model to forecast next window's congestion
 - Save predicted snapshot for proactive route planning (inputs to 06 & 07)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
MODELS = Path("models")
MODEL_FILE = MODELS / "congestion_predictor.pkl"
INPUT_FILE = DATA / "edges_congestion.csv"
OUTPUT_FILE = DATA / "predicted_congestion_snapshot.csv"

# ==============================
# VALIDATION
# ==============================
if not MODEL_FILE.exists():
    print("❌ Model file not found. Run 09_train_predictor.py first.")
    raise SystemExit

if not INPUT_FILE.exists():
    print("❌ Missing congestion data. Run 03_aggregate_congestion.py first.")
    raise SystemExit

# ==============================
# LOAD MODEL & DATA
# ==============================
print(f"📦 Loading trained model from {MODEL_FILE}")
model = joblib.load(MODEL_FILE)

df = pd.read_csv(INPUT_FILE, parse_dates=["time_window"])
if df.empty:
    print("⚠️ No congestion data found.")
    raise SystemExit

latest_time = df["time_window"].max()
snapshot = df[df["time_window"] == latest_time].copy()
print(f"🕒 Using latest snapshot at {latest_time} with {len(snapshot)} edges.")

# ==============================
# BUILD FEATURES FOR PREDICTION
# ==============================
# Reuse the training feature pattern
snapshot["vc_lag1"] = snapshot["vehicle_count"]
snapshot["vc_lag2"] = snapshot["vehicle_count"]
snapshot["speed_lag1"] = snapshot["avg_speed"]
snapshot["speed_lag2"] = snapshot["avg_speed"]

feature_cols = ["vehicle_count", "vc_lag1", "vc_lag2", "avg_speed", "speed_lag1", "speed_lag2"]
X = snapshot[feature_cols].fillna(0).astype(float)

# ==============================
# PREDICTION
# ==============================
print("🤖 Predicting next-window congestion...")
snapshot["predicted_next_count"] = model.predict(X)

# Normalize to congestion score (proxy for congestion intensity)
mx = snapshot["predicted_next_count"].max()
mn = snapshot["predicted_next_count"].min()
snapshot["pred_congestion_score"] = (snapshot["predicted_next_count"] - mn) / (mx - mn + 1e-9)

# Next time window estimate (add +5 min to last timestamp)
next_window = latest_time + pd.Timedelta(minutes=5)
snapshot["predicted_time_window"] = next_window

# ==============================
# OUTPUT
# ==============================
snapshot.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved predicted snapshot → {OUTPUT_FILE}")
print(f"🕐 Predicted time window: {next_window}")
print(f"📈 Avg predicted vehicle count: {snapshot['predicted_next_count'].mean():.2f}")
print(f"🔥 Top predicted congested edges: {len(snapshot[snapshot['pred_congestion_score'] > 0.8])}")

# ==============================
# OPTIONAL NEXT STEPS
# ==============================
"""
To use predictions in proactive control:
  1️⃣ Replace 'edges_congestion.csv' in 06_candidate_routes.py and 07_assign_routes.py 
      with 'predicted_congestion_snapshot.csv'
  2️⃣ Rerun those scripts to generate and assign routes *before congestion happens*.
"""
