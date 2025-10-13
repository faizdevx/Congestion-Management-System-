# scripts/01_download_graph.py
import osmnx as ox
print("Downloading graph for: Ghaziabad, India")

G = ox.graph_from_place("Ghaziabad, India", network_type="drive")
G = ox.distance.add_edge_lengths(G)

ox.save_graphml(G, "data/ghaziabad_drive_graph.graphml")
print("✅ Graph downloaded and saved successfully!")
