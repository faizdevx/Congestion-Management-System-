import pandas as pd
from pathlib import Path
import networkx as nx
import osmnx as ox
import json
import numpy as np

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
CONG_FILE = DATA / "edges_congestion.csv"
OUTPUT_FILE = DATA / "hotspots.json"

# ==============================
# LOAD DATA
# ==============================
if not CONG_FILE.exists():
    print("❌ Missing congestion file. Run '03_aggregate_congestion.py' first.")
    raise SystemExit

print(f"Loading congestion data from {CONG_FILE}...")
cong = pd.read_csv(CONG_FILE)

required_cols = {"edge_u", "edge_v", "congestion_score", "time_window"}
if not required_cols.issubset(cong.columns):
    print(f"❌ Missing required columns in congestion file. Expected {required_cols}")
    raise SystemExit

cong = cong.dropna(subset=["edge_u", "edge_v", "congestion_score"]).copy()

# ==============================
# FILTER TO LATEST TIME WINDOW
# ==============================
latest = cong["time_window"].max()
snapshot = cong[cong["time_window"] == latest].copy()

if snapshot.empty:
    print("⚠️ No valid congestion snapshot found. Run aggregation again.")
    raise SystemExit

# ==============================
# DETECT HOTSPOT EDGES
# ==============================
thr = snapshot["congestion_score"].quantile(0.90)  # top 10%
hot = snapshot[snapshot["congestion_score"] >= thr].copy()

print(f"🔥 Hot edges detected: {len(hot)} (threshold = {thr:.4f})")

if hot.empty:
    print("⚠️ No hotspots found (maybe uniform traffic conditions).")
    raise SystemExit

# ==============================
# BUILD HOTSPOT GRAPH
# ==============================
H = nx.Graph()
for _, r in hot.iterrows():
    try:
        u = int(r["edge_u"])
        v = int(r["edge_v"])
        H.add_edge(u, v)
    except Exception:
        continue

if H.number_of_edges() == 0:
    print("⚠️ No valid edges added to hotspot graph.")
    raise SystemExit

components = list(nx.connected_components(H))
components.sort(key=len, reverse=True)

# ==============================
# SUMMARIZE HOTSPOTS
# ==============================
hotspots = []
for i, comp in enumerate(components):
    hotspots.append({
        "id": i,
        "nodes": list(comp),
        "size": len(comp)
    })

# ==============================
# SAVE HOTSPOT DATA
# ==============================
with open(OUTPUT_FILE, "w") as f:
    json.dump(hotspots, f, indent=2)

print(f"✅ Saved {len(hotspots)} hotspot groups to {OUTPUT_FILE}")

# ==============================
# OPTIONAL: SAVE GEOJSON (for map visualization)
# ==============================
try:
    G = ox.load_graphml(DATA / "ghaziabad_drive_graph.graphml")

    # ✅ FIX HERE: In OSMnx ≥2.x, use ox.graph_to_gdfs instead of ox.utils_graph.graph_to_gdfs
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    hot_edges = edges.merge(hot, left_on=["u", "v"], right_on=["edge_u", "edge_v"], how="inner")

    geojson_path = DATA / "hotspots.geojson"
    hot_edges.to_file(geojson_path, driver="GeoJSON")
    print(f"🌍 Saved GeoJSON for visualization: {geojson_path}")

except Exception as e:
    print(f"(Optional) GeoJSON generation skipped: {e}")
