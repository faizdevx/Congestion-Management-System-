"""
Simulate dispatch after assignment and visualize before/after congestion.

Outputs:
  - reports/before_after_metrics.csv
  - reports/before_map.html
  - reports/after_map.html
"""
import json
import pandas as pd
from pathlib import Path
import folium
from folium.plugins import HeatMap
import osmnx as ox
import networkx as nx

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

GRAPH_FILE = DATA / "ghaziabad_drive_graph.graphml"
TRACES_FILE = DATA / "vehicle_traces_osm.csv"
CONG_FILE = DATA / "edges_congestion.csv"
ASSIGN_FILE = DATA / "assignments.json"
HOTSPOT_FILE = DATA / "hotspots.json"

# ==============================
# VALIDATION
# ==============================
for f in [GRAPH_FILE, TRACES_FILE, CONG_FILE, ASSIGN_FILE, HOTSPOT_FILE]:
    if not f.exists():
        print(f"❌ Missing required file: {f}")
        raise SystemExit

print("📦 Loading graph and data...")

G = ox.load_graphml(str(GRAPH_FILE))
traces = pd.read_csv(TRACES_FILE, parse_dates=["timestamp"])
cong = pd.read_csv(CONG_FILE)
latest = cong["time_window"].max()
snapshot = cong[cong["time_window"] == latest].copy()

with open(HOTSPOT_FILE) as f:
    hotspots = json.load(f)
with open(ASSIGN_FILE) as f:
    assignments = json.load(f)

print(f"🕓 Using snapshot: {latest}, Hotspot groups: {len(hotspots)}, Assignments: {len(assignments)}")

# ==============================
# MAP HELPER: edge mean coordinates
# ==============================
edge_positions = (
    traces.dropna(subset=["edge_u"])
    .groupby(["edge_u", "edge_v", "edge_key"])
    .agg(lat=("lat", "mean"), lon=("lon", "mean"), vehicle_count=("vehicle_id", "nunique"))
    .reset_index()
)

# Auto-center map on trace mean
center_lat = edge_positions["lat"].mean()
center_lon = edge_positions["lon"].mean()

# ==============================
# BEFORE HEATMAP
# ==============================
heat_before = [[r.lat, r.lon, r.vehicle_count] for _, r in edge_positions.iterrows()]
m_before = folium.Map(location=[center_lat, center_lon], zoom_start=12)
HeatMap(heat_before, radius=20, blur=15, min_opacity=0.4).add_to(m_before)
m_before.save(str(REPORTS / "before_map.html"))
print("🗺️  Saved before_map.html")

# ==============================
# SIMULATE ASSIGNMENTS (AFTER)
# ==============================
load_lookup = {
    (int(r.edge_u), int(r.edge_v), int(r.edge_key)): int(r.vehicle_count)
    for r in snapshot.itertuples()
}

# Apply assignment effects (naive simulation)
for veh, a in assignments.items():
    path = a.get("path", [])
    if len(path) < 2:
        continue
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if data is None:
            continue
        key = list(data.keys())[0]
        load_lookup[(u, v, key)] = load_lookup.get((u, v, key), 0) + 1

# ==============================
# AFTER HEATMAP
# ==============================
after_points = []
for (u, v, k), count in load_lookup.items():
    tmp = edge_positions[
        (edge_positions.edge_u == u)
        & (edge_positions.edge_v == v)
        & (edge_positions.edge_key == k)
    ]
    if not tmp.empty:
        lat = float(tmp.lat.values[0])
        lon = float(tmp.lon.values[0])
        after_points.append([lat, lon, count])

m_after = folium.Map(location=[center_lat, center_lon], zoom_start=12)
HeatMap(after_points, radius=20, blur=15, min_opacity=0.4).add_to(m_after)
m_after.save(str(REPORTS / "after_map.html"))
print("🗺️  Saved after_map.html")

# ==============================
# METRICS COMPUTATION
# ==============================
before_total = int(snapshot["vehicle_count"].sum())
after_total = int(sum(load_lookup.values()))
vehicles_rerouted = len(assignments)

# Optional: compute total hotspot load reduction
hot_nodes = set()
for h in hotspots:
    hot_nodes.update(h.get("nodes", []))

hot_before = sum(
    int(r.vehicle_count)
    for r in snapshot.itertuples()
    if int(r.edge_u) in hot_nodes or int(r.edge_v) in hot_nodes
)
hot_after = sum(
    load_lookup.get((u, v, k), 0)
    for (u, v, k) in load_lookup.keys()
    if u in hot_nodes or v in hot_nodes
)
reduction_pct = (
    ((hot_before - hot_after) / hot_before * 100.0) if hot_before > 0 else 0
)

metrics = {
    "snapshot_time": str(latest),
    "total_edges": len(G.edges),
    "before_total_vehicle_count": before_total,
    "after_total_vehicle_count": after_total,
    "vehicles_rerouted": vehicles_rerouted,
    "hotspot_load_before": hot_before,
    "hotspot_load_after": hot_after,
    "hotspot_reduction_percent": round(reduction_pct, 2),
}

# ==============================
# SAVE REPORTS
# ==============================
metrics_path = REPORTS / "before_after_metrics.csv"
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"📊 Metrics saved → {metrics_path}")
print("✅ Simulation completed successfully!")
