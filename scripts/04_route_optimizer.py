# scripts/04_route_optimizer.py
import pandas as pd
import osmnx as ox
import networkx as nx
from pathlib import Path
import math
import sys

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
GRAPH_FILE = DATA / "ghaziabad_drive_graph.graphml"
CONG_FILE = DATA / "edges_congestion.csv"

# ==============================
# LOAD GRAPH
# ==============================
if not GRAPH_FILE.exists():
    print(f"❌ Graph file not found: {GRAPH_FILE}")
    print("Run '01_download_graph.py' first to generate it.")
    sys.exit(1)

print(f"Loading graph from {GRAPH_FILE} ...")
G = ox.load_graphml(str(GRAPH_FILE))

# ==============================
# LOAD CONGESTION DATA
# ==============================
if not CONG_FILE.exists():
    print("⚠️ No congestion file found. Run '03_aggregate_congestion.py' first.")
    sys.exit(1)

cong = pd.read_csv(CONG_FILE)

# Handle potential missing or malformed timestamps
if "time_window" not in cong.columns:
    print("⚠️ Congestion data missing 'time_window' column.")
    sys.exit(1)

latest = cong["time_window"].max()
snapshot = cong[cong["time_window"] == latest].copy()

if snapshot.empty:
    print("⚠️ No valid congestion snapshot found.")
    sys.exit(1)

# ==============================
# BUILD CONGESTION LOOKUP
# ==============================
cong_lookup = {}
for _, r in snapshot.iterrows():
    try:
        key = (int(r["edge_u"]), int(r["edge_v"]), int(r["edge_key"]))
        cong_lookup[key] = float(r["congestion_score"])
    except Exception:
        continue

print(f"Loaded {len(cong_lookup)} congestion edges from latest snapshot ({latest}).")

# ==============================
# APPLY CONGESTION WEIGHTS
# ==============================
alpha = 5.0  # weight amplification factor

for u, v, k, d in G.edges(keys=True, data=True):
    base_dist_km = d.get("length", 0.0) / 1000.0
    score = cong_lookup.get((u, v, k), 0.0)
    weight = base_dist_km * (1.0 + alpha * score)
    G[u][v][k]["congestion_weight"] = weight

print("✅ Applied congestion-adjusted weights to graph edges.")

# ==============================
# COMPUTE OPTIMAL ROUTE
# ==============================
nodes = list(G.nodes)
if len(nodes) < 2:
    print("⚠️ Graph has too few nodes for routing.")
    sys.exit(1)

# Random or fixed start/end for reproducibility
orig = nodes[len(nodes) // 10]
dest = nodes[len(nodes) // 2]

print(f"Computing route from node {orig} → {dest} ...")

try:
    path = nx.shortest_path(G, source=orig, target=dest, weight="congestion_weight")
    total = nx.shortest_path_length(G, source=orig, target=dest, weight="congestion_weight")
    print(f"✅ Path found! Total cost: {total:.3f}, Nodes: {len(path)}")

    # Export coordinates for visualization
    coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
    route_df = pd.DataFrame(coords, columns=["lat", "lon"])
    route_df.to_csv(DATA / "sample_congestion_route.csv", index=False)
    print(f"✅ Saved route to {DATA / 'sample_congestion_route.csv'}")

except nx.NetworkXNoPath:
    print("❌ No path found between selected nodes (disconnected graph).")
except Exception as e:
    print(f"❌ Routing failed due to error: {e}")
