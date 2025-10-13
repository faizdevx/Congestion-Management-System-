import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="Traffic Analytics Dashboard", layout="wide")

st.title("📊 Smart City Traffic Analytics Dashboard")

DATA = Path("data")
REPORTS = Path("reports")

# ==============================
# DATA LOADING
# ==============================
@st.cache_data
def load_data():
    df_cong = pd.read_csv(DATA / "edges_congestion.csv", parse_dates=["time_window"])
    df_trace = pd.read_csv(DATA / "vehicle_traces_osm.csv", parse_dates=["timestamp"])
    with open(DATA / "hotspots.json") as f:
        hotspots = json.load(f)
    return df_cong, df_trace, hotspots

try:
    df_cong, df_trace, hotspots = load_data()
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

# ==============================
# DASHBOARD LAYOUT
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🚦 Top Roads", "🔥 Hotspots", "📁 Downloads"])

# ------------------------------
# TAB 1 — Congestion Trends
# ------------------------------
with tab1:
    st.subheader("Traffic Congestion Trends Over Time")
    trend = df_cong.groupby("time_window")["congestion_score"].mean().reset_index()
    fig_trend = px.line(
        trend, x="time_window", y="congestion_score",
        title="Average Congestion Over Time", markers=True
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Speed Distribution")
    fig_speed = px.histogram(df_cong, x="avg_speed", nbins=30, title="Vehicle Speed Distribution (km/h)")
    st.plotly_chart(fig_speed, use_container_width=True)

# ------------------------------
# TAB 2 — Top Congested Roads
# ------------------------------
with tab2:
    st.subheader("Top 10 Most Congested Edges")
    top_edges = df_cong.sort_values("congestion_score", ascending=False).head(10)
    st.dataframe(top_edges[["edge_u", "edge_v", "vehicle_count", "avg_speed", "congestion_score"]])

    fig_top = px.bar(
        top_edges, x="vehicle_count", y="congestion_score",
        color="avg_speed", orientation="h",
        title="Most Congested Road Segments",
        hover_data=["edge_u", "edge_v"]
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ------------------------------
# TAB 3 — Hotspot Visualization
# ------------------------------
with tab3:
    st.subheader(f"Detected Hotspots: {len(hotspots)} Clusters")
    m = folium.Map(location=[28.67, 77.44], zoom_start=12)

    for i, h in enumerate(hotspots):
        folium.CircleMarker(
            location=[28.67 + (i * 0.002), 77.44 + (i * 0.002)],
            radius=8,
            color="red",
            fill=True,
            popup=f"Hotspot #{i} — {len(h['nodes'])} edges"
        ).add_to(m)

    st_folium(m, height=500, width=900)

# ------------------------------
# TAB 4 — Downloads
# ------------------------------
with tab4:
    st.subheader("Download Processed Datasets")
    st.download_button(
        "📥 Download Congestion Data",
        data=open(DATA / "edges_congestion.csv", "rb"),
        file_name="edges_congestion.csv",
        mime="text/csv"
    )
    st.download_button(
        "📥 Download Vehicle Trace Data",
        data=open(DATA / "vehicle_traces_osm.csv", "rb"),
        file_name="vehicle_traces_osm.csv",
        mime="text/csv"
    )
    st.download_button(
        "📥 Download Hotspots JSON",
        data=open(DATA / "hotspots.json", "rb"),
        file_name="hotspots.json",
        mime="application/json"
    )

st.markdown("---")
st.caption("© 2025 Smart City Simulation by Faizal — AI-Powered Urban Mobility Lab")
