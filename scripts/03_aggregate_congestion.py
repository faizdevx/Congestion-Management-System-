"""
Aggregate per-edge congestion metrics from vehicle trace data.

v3 Upgrades:
 - Weighted vehicle impact (bus/truck contribute more to congestion).
 - Adds rush-hour factor for diurnal patterns.
 - Generates additional metrics: load_index, speed_variance, weighted_speed.
 - Handles missing/flat data gracefully.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
TRACE_FILE = DATA / "vehicle_traces_osm.csv"
OUTPUT_FILE = DATA / "edges_congestion.csv"

VEHICLE_WEIGHTS = {
    "car": 1.0,
    "bike": 0.5,
    "bus": 2.0,
    "truck": 2.5
}

RUSH_HOURS = [(8, 11), (17, 20)]  # (start_hour, end_hour)

# ==============================
# LOAD DATA
# ==============================
if not TRACE_FILE.exists():
    print(f"❌ Missing input file: {TRACE_FILE}")
    print("Run '02_simulate_traces.py' first to generate traces.")
    exit(1)

print(f"🚗 Loading data from {TRACE_FILE}...")
df = pd.read_csv(TRACE_FILE, parse_dates=["timestamp"])

# Drop rows where no edge info (vehicle not moving)
edge_pings = df.dropna(subset=["edge_u"]).copy()

# Ensure proper timestamp parsing
if not np.issubdtype(edge_pings["timestamp"].dtype, np.datetime64):
    edge_pings["timestamp"] = pd.to_datetime(edge_pings["timestamp"], errors="coerce")

# Drop invalid timestamps
edge_pings = edge_pings.dropna(subset=["timestamp"])

# ==============================
# FEATURE ENHANCEMENT
# ==============================
# Vehicle weight impact
edge_pings["veh_weight"] = edge_pings["vehicle_type"].map(VEHICLE_WEIGHTS).fillna(1.0)

# Rush-hour flag (morning/evening)
edge_pings["hour"] = edge_pings["timestamp"].dt.hour
edge_pings["rush_hour"] = edge_pings["hour"].apply(
    lambda h: any(start <= h <= end for start, end in RUSH_HOURS)
)

# Weighted speed: heavy vehicles slow more
edge_pings["weighted_speed"] = edge_pings["speed_kmph"] / edge_pings["veh_weight"]

# ==============================
# TIME WINDOW AGGREGATION
# ==============================
edge_pings["time_window"] = edge_pings["timestamp"].dt.floor("5min")

g = (
    edge_pings.groupby(["edge_u", "edge_v", "edge_key", "time_window"])
    .agg(
        vehicle_count=("vehicle_id", "nunique"),
        weighted_vehicle_load=("veh_weight", "sum"),
        pings=("vehicle_id", "count"),
        avg_speed=("speed_kmph", "mean"),
        weighted_speed=("weighted_speed", "mean"),
        speed_var=("speed_kmph", "var"),
        rush_ratio=("rush_hour", "mean")
    )
    .reset_index()
)

if len(g) == 0:
    print("⚠️ No edge pings found. Run simulation again.")
    exit(1)

# ==============================
# CONGESTION METRICS
# ==============================
# Normalize safely
g["vc_norm"] = (g["vehicle_count"] - g["vehicle_count"].min()) / (
    (g["vehicle_count"].max() - g["vehicle_count"].min()) + 1e-9
)
g["speed_inv_norm"] = 1 - (g["avg_speed"] - g["avg_speed"].min()) / (
    (g["avg_speed"].max() - g["avg_speed"].min()) + 1e-9
)
g["load_norm"] = (g["weighted_vehicle_load"] - g["weighted_vehicle_load"].min()) / (
    (g["weighted_vehicle_load"].max() - g["weighted_vehicle_load"].min()) + 1e-9
)

# Composite congestion formula
# Combines vehicle density, speed inverse, load, and rush-hour factor
g["congestion_score"] = (
    0.5 * g["vc_norm"]
    + 0.25 * g["speed_inv_norm"]
    + 0.15 * g["load_norm"]
    + 0.10 * g["rush_ratio"]
)

# Speed variance gives an instability signal (traffic turbulence)
g["speed_var"] = g["speed_var"].fillna(0)
g["load_index"] = g["weighted_vehicle_load"] / (g["vehicle_count"] + 1e-6)

# Fill missing with zeros
g = g.replace([np.inf, -np.inf], np.nan).fillna(0)

# ==============================
# SAVE OUTPUT
# ==============================
g.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved enriched congestion dataset to {OUTPUT_FILE}")
print(f"📈 Total aggregated rows: {len(g)}")
print("🏙️ Features included: vehicle_count, avg_speed, load_index, congestion_score, rush_ratio, etc.")
