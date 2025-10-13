"""
For each vehicle whose current planned path intersects a hotspot,
generate candidate alternative routes (avoid hotspot, soft-penalty, limited-detour).
"""

import os
import json
import pandas as pd
import networkx as nx
import osmnx as ox
from pathlib import Path
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
GRAPH_PATH = DATA / "ghaziabad_drive_graph.graphml"
TRACES_FILE = DATA / "vehicle_traces_osm.csv"
HOTSPOT_FILE = DATA / "hotspots.json"
ROUTES_DIR = DATA / "routes"
OUT_FILE = DATA / "candidates.json"

# ==============================
# LOAD GRAPH + DATA
# ==============================
if not GRAPH_PATH.exists():
    print("❌ Graph file missing. Run 01_download_graph.py first.")
    raise SystemExit

if not HOTSPOT_FILE.exists():
    print("❌ Hotspot file missing. Run 05_detect_hotspots.py first.")
    raise SystemExit

if not TRACES_FILE.exists():
    print("❌ Vehicle traces missing. Run 02_simulate_traces.py first.")
    raise SystemExit

print(f"Loading graph from {GRAPH_PATH}...")
G = ox.load_graphml(str(GRAPH_PATH))

traces = pd.read_csv(TRACES_FILE, parse_dates=["timestamp"])
with open(HOTSPOT_FILE) as f:
    hotspots = json.load(f)

# ==============================
# EXTRACT HOTSPOT NODE SET
# ==============================
hot_nodes = set()
for h in hotspots:
    hot_nodes.update(h.get("nodes", []))

if not hot_nodes:
    print("⚠️ No hotspot nodes detected. Exiting.")
    raise SystemExit

print(f"Total hotspot nodes: {len(hot_nodes)}")

# ==============================
# BUILD VEHICLE ROUTES
# ==============================
vehicles = []
for f in ROUTES_DIR.glob("veh_*.csv"):
    df = pd.read_csv(f)
    if df.empty:
        continue

    # Drop invalid rows
    df = df.dropna(subset=["edge_u", "edge_v", "edge_key"], how="any")

    # Collect route info
    edges = list(df[["edge_u", "edge_v", "edge_key"]].itertuples(index=False, name=None))
    origin = edges[0][0] if len(edges) > 0 else None
    dest = int(df.iloc[-1]["dest_node"]) if "dest_node" in df.columns and not pd.isna(df.iloc[-1]["dest_node"]) else None

    vehicles.append({
        "vehicle_id": f.stem,
        "edges": edges,
        "origin": origin,
        "dest": dest
    })

if not vehicles:
    print("⚠️ No vehicle route files found in /data/routes. Run 02_simulate_traces.py first.")
    raise SystemExit

# ==============================
# IDENTIFY AFFECTED VEHICLES
# ==============================
affected = []
for v in vehicles:
    if v["origin"] is None or v["dest"] is None:
        continue
    if any(e[0] in hot_nodes or e[1] in hot_nodes for e in v["edges"]):
        affected.append(v)

print(f"🚗 Affected vehicles (intersecting hotspots): {len(affected)}")

# ==============================
# HELPER: SHORTEST PATH AVOIDING HOTSPOTS
# ==============================
def shortest_avoiding(G, orig, dest, forbidden_nodes):
    """
    Returns shortest path avoiding forbidden nodes by assigning large weights.
    """
    H = G.copy()
    for u, v, k, d in list(H.edges(keys=True, data=True)):
        if u in forbidden_nodes or v in forbidden_nodes:
            for key in H[u][v]:
                H[u][v][key]["travel_time"] = 1e9
    try:
        return nx.shortest_path(H, orig, dest, weight="travel_time")
    except nx.NetworkXNoPath:
        return None
    except Exception:
        return None

# ==============================
# GENERATE CANDIDATE ROUTES
# ==============================
candidates = {}

for v in tqdm(affected, desc="Generating candidate routes"):
    orig = v["origin"]
    dest = v["dest"]

    if orig not in G.nodes or dest not in G.nodes:
        continue

    try:
        base_path = nx.shortest_path(G, orig, dest, weight="travel_time")
        base_time = nx.shortest_path_length(G, orig, dest, weight="travel_time")
    except nx.NetworkXNoPath:
        continue
    except Exception:
        continue

    cand_list = []

    # Candidate 1: avoid hotspot nodes entirely
    p1 = shortest_avoiding(G, orig, dest, hot_nodes)
    if p1:
        t1 = nx.shortest_path_length(G, orig, dest, weight="travel_time")
        cand_list.append({"type": "avoid_hotspot", "path": p1, "time": t1})

    # Candidate 2: normal route (soft penalty baseline)
    try:
        p2 = nx.shortest_path(G, orig, dest, weight="travel_time")
        t2 = nx.shortest_path_length(G, orig, dest, weight="travel_time")
        cand_list.append({"type": "soft_penalty", "path": p2, "time": t2})
    except:
        pass

    # Candidate 3: limited detour (≤ +25% of base time)
    if p1:
        t1 = nx.shortest_path_length(G, orig, dest, weight="travel_time")
        if t1 <= base_time * 1.25:
            cand_list.append({"type": "limited_detour", "path": p1, "time": t1})

    # Store all candidates
    candidates[v["vehicle_id"]] = {
        "orig": orig,
        "dest": dest,
        "base_time": base_time,
        "candidates": cand_list
    }

# ==============================
# SAVE OUTPUT
# ==============================
with open(OUT_FILE, "w") as f:
    json.dump(candidates, f, indent=2, default=lambda o: list(o) if isinstance(o, set) else o)

print(f"✅ Saved {len(candidates)} candidate route sets to {OUT_FILE}")
