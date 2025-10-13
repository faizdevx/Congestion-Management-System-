"""
Greedy capacity-aware route assignment.

Steps:
- Load congestion data and OSM graph.
- Estimate capacity per road type.
- For each candidate route, compute marginal benefit = (hotspot_edges_avoided / extra_time).
- Assign greedily if edge capacities allow.
"""

import json
import math
import pandas as pd
import networkx as nx
import osmnx as ox
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
GRAPH_FILE = DATA / "ghaziabad_drive_graph.graphml"
CONG_FILE = DATA / "edges_congestion.csv"
CAND_FILE = DATA / "candidates.json"
HOTSPOT_FILE = DATA / "hotspots.json"
OUT_FILE = DATA / "assignments.json"

# ==============================
# LOAD GRAPH + DATA VALIDATION
# ==============================
if not GRAPH_FILE.exists():
    print("❌ Missing graph file. Run 01_download_graph.py first.")
    raise SystemExit

if not CONG_FILE.exists():
    print("❌ Missing congestion data. Run 03_aggregate_congestion.py first.")
    raise SystemExit

if not CAND_FILE.exists():
    print("❌ Missing candidate routes. Run 06_candidate_routes.py first.")
    raise SystemExit

if not HOTSPOT_FILE.exists():
    print("❌ Missing hotspot data. Run 05_detect_hotspots.py first.")
    raise SystemExit

print(f"📦 Loading graph from {GRAPH_FILE}...")
G = ox.load_graphml(str(GRAPH_FILE))

# ==============================
# LOAD CONGESTION SNAPSHOT
# ==============================
cong = pd.read_csv(CONG_FILE)
latest = cong["time_window"].max()
snapshot = cong[cong["time_window"] == latest].copy()

print(f"📊 Using congestion snapshot from {latest} — {len(snapshot)} edges")

# Load current vehicle count per edge
load_lookup = {
    (int(r["edge_u"]), int(r["edge_v"]), int(r["edge_key"])): int(r["vehicle_count"])
    for _, r in snapshot.iterrows()
}

# ==============================
# ESTIMATE CAPACITIES BY ROAD TYPE
# ==============================
cap_default = 20
edge_cap = {}

for u, v, k, d in G.edges(keys=True, data=True):
    hw = d.get("highway", "residential")
    if isinstance(hw, list):  # Some OSM data stores as list
        hw = hw[0]
    if hw in ["motorway", "trunk"]:
        cap = 200
    elif hw in ["primary", "secondary"]:
        cap = 80
    elif hw == "tertiary":
        cap = 40
    else:
        cap = cap_default
    edge_cap[(u, v, k)] = cap

# ==============================
# LOAD CANDIDATE ROUTES
# ==============================
with open(CAND_FILE) as f:
    candidates = json.load(f)

with open(HOTSPOT_FILE) as f:
    hotspots = json.load(f)

hot_nodes = set()
for h in hotspots:
    hot_nodes.update(h["nodes"])

print(f"🔥 Loaded {len(hot_nodes)} hotspot nodes from {HOTSPOT_FILE}")

# ==============================
# HELPER FUNCTIONS
# ==============================
def edges_in_path(G, path):
    """Return list of (u, v, k) tuples for edges in the path."""
    edges = []
    for a, b in zip(path[:-1], path[1:]):
        data = G.get_edge_data(a, b)
        if not data:
            continue
        key = list(data.keys())[0]
        edges.append((a, b, key))
    return edges

# ==============================
# BUILD CANDIDATE ENTRIES
# ==============================
entries = []

for vid, info in candidates.items():
    base_time = info.get("base_time")
    orig = info.get("orig")
    dest = info.get("dest")

    if orig not in G.nodes or dest not in G.nodes:
        continue

    try:
        base_path = nx.shortest_path(G, orig, dest, weight="travel_time")
    except Exception:
        continue

    base_hot = sum(1 for n in base_path if n in hot_nodes)

    for cand in info["candidates"]:
        path = cand["path"]
        time = cand["time"]
        delta = time - base_time

        cand_hot = sum(1 for n in path if n in hot_nodes)
        benefit = max(0, base_hot - cand_hot)

        # Prevent division by zero
        if delta <= 1e-6:
            marginal = benefit / 1e-6
        else:
            marginal = benefit / delta

        entries.append({
            "veh": vid,
            "orig": orig,
            "dest": dest,
            "path": path,
            "delta": delta,
            "benefit": benefit,
            "marginal": marginal
        })

if not entries:
    print("⚠️ No candidate entries found. Run 06_candidate_routes.py first.")
    raise SystemExit

# ==============================
# GREEDY ASSIGNMENT
# ==============================
entries = sorted(entries, key=lambda x: x["marginal"], reverse=True)
projected = dict(load_lookup)
assignments = {}

print(f"🚦 Starting assignment for {len(entries)} route candidates...")

for e in entries:
    if e["benefit"] <= 0:
        continue

    path_edges = edges_in_path(G, e["path"])
    feasible = True

    # Check edge capacities
    for ed in path_edges:
        cap = edge_cap.get(ed, cap_default)
        proj = projected.get(ed, 0)
        if proj + 1 > cap:
            feasible = False
            break

    if not feasible:
        continue

    # Assign vehicle
    assignments[e["veh"]] = {
        "path": e["path"],
        "delta": e["delta"],
        "benefit": e["benefit"],
        "marginal": e["marginal"],
    }

    # Update projected load
    for ed in path_edges:
        projected[ed] = projected.get(ed, 0) + 1

# ==============================
# SAVE RESULTS
# ==============================
with open(OUT_FILE, "w") as f:
    json.dump(assignments, f, indent=2)

print(f"✅ Assignments saved to {OUT_FILE}")
print(f"📈 Total assigned vehicles: {len(assignments)}")

# Optional: Print load utilization summary
total_edges = len(projected)
overloaded = sum(1 for k, v in projected.items() if v > edge_cap.get(k, cap_default))
print(f"⚙️ Capacity utilization: {total_edges - overloaded}/{total_edges} edges within limits.")
