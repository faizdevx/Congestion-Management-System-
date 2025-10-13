"""
Simulate realistic vehicle trajectories over a city graph.

Upgraded v3 Features:
 - 24-hour simulation with rush-hour congestion dynamics.
 - Vehicle type diversity (car, bus, bike) with different speeds.
 - Smooth temporal variability in traffic (morning & evening peaks).
 - Generates rich data for congestion prediction.
"""

import os, random, math
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
DATA = Path("data")
DATA.mkdir(exist_ok=True)
GRAPH_FILE = DATA / "ghaziabad_drive_graph.graphml"

print("🚗 Loading road network graph...", GRAPH_FILE)
G = ox.load_graphml(str(GRAPH_FILE))

# ✅ Compatible with OSMnx 2.x API
G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)
G = ox.distance.add_edge_lengths(G)

# ==============================
# SIMULATION PARAMETERS
# ==============================
N_VEHICLES = 400              # total simulated vehicles
PING_INTERVAL_SECONDS = 15
SIM_START = datetime.datetime(2023, 11, 1, 0, 0, 0)
SIM_DURATION_HOURS = 24       # simulate full day
MAX_TRIP_KM = 30
STOP_PROB = 0.4
STOP_MEAN = 30  # seconds

# Vehicle categories — speeds in km/h (mean ± std)
VEHICLE_TYPES = {
    "car": (50, 10),
    "bike": (35, 8),
    "bus": (40, 6),
    "truck": (30, 7)
}

# ==============================
# NODE SAMPLING
# ==============================
nodes = list(G.nodes)
deg = [G.degree(n) for n in nodes]
weights = [max(1, d) for d in deg]

def sample_node():
    """Sample a node with probability proportional to degree."""
    return random.choices(nodes, weights=weights, k=1)[0]

def interpolate_geom(geom, n):
    """Evenly interpolate 'n' points along an edge geometry."""
    if geom is None:
        return []
    if not isinstance(geom, LineString):
        try:
            geom = LineString(geom)
        except Exception:
            return []
    pts = []
    for f in np.linspace(0, 1, n, endpoint=False):
        p = geom.interpolate(f, normalized=True)
        pts.append((p.y, p.x))
    return pts

def rush_hour_multiplier(t):
    """
    Simulate rush hour congestion (speed reduction factors).
    - Morning 8–11 AM: heavy traffic (0.6×)
    - Evening 5–8 PM: moderate traffic (0.7×)
    - Night (11 PM–5 AM): low traffic (1.2×)
    """
    hour = t.hour
    if 8 <= hour <= 11:
        return 0.6
    elif 17 <= hour <= 20:
        return 0.7
    elif 23 <= hour or hour <= 5:
        return 1.2
    return 1.0

# ==============================
# SIMULATION LOOP
# ==============================
rows = []
routes_dir = DATA / "routes"
routes_dir.mkdir(exist_ok=True)

for vid in tqdm(range(N_VEHICLES), desc="🚦 Simulating vehicles"):
    veh = f"veh_{vid:04d}"
    origin = sample_node()
    dest = sample_node()
    if origin == dest:
        continue

    # Assign random vehicle type
    vtype = random.choice(list(VEHICLE_TYPES.keys()))
    mean_speed, std_speed = VEHICLE_TYPES[vtype]

    try:
        path = nx.shortest_path(G, origin, dest, weight="travel_time")
    except Exception:
        continue

    edge_list = []
    total_len = 0.0
    total_time = 0.0

    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        key = list(data.keys())[0]
        ed = data[key]
        length = ed.get("length", 0.0)
        travel_time = ed.get("travel_time", max(1.0, (length / 1000.0) / 30 * 3600))
        geom = ed.get("geometry", None)
        edge_list.append((u, v, key, length, travel_time, geom))
        total_len += length
        total_time += travel_time

    if total_len / 1000.0 > MAX_TRIP_KM or total_len == 0:
        continue

    # Start randomly within the 24-hour simulation window
    window = SIM_DURATION_HOURS * 3600
    offset = random.randint(0, max(0, int(window - total_time)))
    t = SIM_START + datetime.timedelta(seconds=offset)
    route_rows = []

    for (u, v, k, length, travel_time, geom) in edge_list:
        # Determine current rush-hour multiplier
        rush_factor = rush_hour_multiplier(t)
        adjusted_speed = max(5.0, np.random.normal(mean_speed * rush_factor, std_speed))
        # Ensure travel time adjusts accordingly
        travel_time = (length / 1000.0) / adjusted_speed * 3600.0

        n_pings = max(1, int(math.ceil(travel_time / PING_INTERVAL_SECONDS)))
        pts = interpolate_geom(
            geom if geom is not None else LineString([
                (G.nodes[u]['x'], G.nodes[u]['y']),
                (G.nodes[v]['x'], G.nodes[v]['y'])
            ]),
            n_pings + 1
        )

        for lat, lon in pts[:-1]:
            row = {
                "vehicle_id": veh,
                "vehicle_type": vtype,
                "timestamp": t.isoformat(),
                "lat": lat,
                "lon": lon,
                "speed_kmph": round(adjusted_speed, 2),
                "edge_u": u,
                "edge_v": v,
                "edge_key": k,
                "dest_node": dest
            }
            rows.append(row)
            route_rows.append(row)
            t += datetime.timedelta(seconds=PING_INTERVAL_SECONDS)

        # Random stop simulation
        if random.random() < STOP_PROB:
            delay = max(0, int(np.random.normal(STOP_MEAN, STOP_MEAN * 0.25)))
            t += datetime.timedelta(seconds=delay)

    # Final destination point
    last_node = path[-1]
    rows.append({
        "vehicle_id": veh,
        "vehicle_type": vtype,
        "timestamp": t.isoformat(),
        "lat": G.nodes[last_node]['y'],
        "lon": G.nodes[last_node]['x'],
        "speed_kmph": 0.0,
        "edge_u": None,
        "edge_v": None,
        "edge_key": None,
        "dest_node": dest
    })

    pd.DataFrame(route_rows).to_csv(routes_dir / f"{veh}_route.csv", index=False)

# ==============================
# SAVE OUTPUT
# ==============================
df = pd.DataFrame(rows)
df.sort_values(["vehicle_id", "timestamp"], inplace=True)
out_path = DATA / "vehicle_traces_osm.csv"
df.to_csv(out_path, index=False)

print(f"\n✅ Simulation completed successfully.")
print(f"💾 Saved {len(df):,} trace points to {out_path}")
print(f"🕒 Duration simulated: {SIM_DURATION_HOURS} hours across {N_VEHICLES} vehicles.")
