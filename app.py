# app.py
import os
import io
import zipfile
import tempfile
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless safe for servers

# ---- import the functions from your existing script ----
from routing_ors_compare import (
    load_locations, geocode_if_needed, build_distance_matrix_ors, solve_vrp,
    build_stops_delta_df, plot_stop_delta, plot_total_cost_bar, plot_route_costs,
    plot_route_distance_hist, build_stops_delta_map, add_routes_layer, write_dashboard,
    run_scenario
)

import openrouteservice as ors
import folium

st.set_page_config(page_title="Depot Comparison", page_icon="🚚", layout="wide")

@contextmanager
def cd(path):
    """Temporarily chdir into path (so all your script's relative saves go there)."""
    orig = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig)

st.title("🚚 Depot Comparison Dashboard")
st.caption("Upload your deliveries, set depots + parameters, and compare OLD vs NEW depot routes.")

with st.sidebar:
    st.subheader("OpenRouteService")
    # Prefer Streamlit secrets on Cloud; fallback to env var or user input
    default_key = os.getenv("OPENROUTESERVICE_API_KEY", "")
    default_key = st.secrets.get("OPENROUTESERVICE_API_KEY", default_key) if hasattr(st, "secrets") else default_key
    ors_key = st.text_input("OPENROUTESERVICE_API_KEY", value=default_key, type="password")

    st.subheader("Routing Inputs")
    old_warehouse = st.text_input("Old Depot (address or 'lat,lon')", value="")
    new_warehouse = st.text_input("New Depot (address or 'lat,lon')", value="")

    profile = st.selectbox(
        "Vehicle profile",
        options=["driving-hgv", "driving-car"],
        index=0
    )
    capacity = st.number_input("Truck capacity", min_value=1, value=10, step=1)
    cost_per_km = st.number_input("Cost per km", min_value=0.0, value=250.0, step=10.0)
    max_vehicles = st.number_input("Max vehicles (0 = auto)", min_value=0, value=0, step=1)
    time_limit = st.number_input("VRP time limit (sec)", min_value=5, value=20, step=5)

    st.markdown("---")
    st.caption("CSV must include: `name, demand` and either `address` or `latitude,longitude`.\n"
               "Extra columns are ignored.")

uploaded = st.file_uploader("Upload deliveries.csv", type=["csv"])
run = st.button("Run comparison", type="primary", use_container_width=True)

def pipeline(workdir: str, csv_bytes: bytes):
    # Build a client
    if not ors_key:
        raise RuntimeError("OPENROUTESERVICE_API_KEY is required.")

    client = ors.Client(key=ors_key)

    # Persist uploaded file in workdir as 'deliveries.csv'
    csv_path = os.path.join(workdir, "deliveries.csv")
    with open(csv_path, "wb") as f:
        f.write(csv_bytes)

    # Load and geocode stops
    df = load_locations(csv_path)

    stops_coords, stops_names, demands_stops = [], [], []
    for _, row in df.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        latlon = geocode_if_needed(client, row["name"], row.get("address", ""), lat, lon)
        stops_coords.append(latlon)
        stops_names.append(str(row["name"]))
        demands_stops.append(int(row["demand"]))

    # Compute old/new scenarios
    mv = None if max_vehicles == 0 else int(max_vehicles)

    res_old = run_scenario("old", old_warehouse, client, stops_coords, stops_names, demands_stops,
                           profile, int(capacity), mv, int(time_limit), float(cost_per_km))
    res_new = run_scenario("new", new_warehouse, client, stops_coords, stops_names, demands_stops,
                           profile, int(capacity), mv, int(time_limit), float(cost_per_km))

    # Overall CSV
    pd.DataFrame([
        {"scenario": "old", "total_distance_km": res_old["total_distance_km"], "total_cost": res_old["total_cost"], "vehicles_used": res_old["vehicles_used"]},
        {"scenario": "new", "total_distance_km": res_new["total_distance_km"], "total_cost": res_new["total_cost"], "vehicles_used": res_new["vehicles_used"]},
    ]).to_csv("overall_comparison.csv", index=False)

    # Charts
    plot_total_cost_bar(res_old["total_cost"], res_new["total_cost"], "comparison_costs.png")
    plot_route_costs(res_old["routes_df"], "old")
    plot_route_costs(res_new["routes_df"], "new")
    plot_route_distance_hist(res_old["routes_df"], "old")
    plot_route_distance_hist(res_new["routes_df"], "new")

    # Per-stop depot→stop deltas
    old_row0 = res_old["dist_m"][0]
    new_row0 = res_new["dist_m"][0]
    stop_delta = build_stops_delta_df(stops_names, old_row0, new_row0)
    stop_delta.to_csv("stop_distance_delta.csv", index=False)
    plot_stop_delta(stop_delta, "stop_distance_delta.png")
    build_stops_delta_map(stop_delta, stops_coords, "stops_delta_map.html")

    # Maps
    for res in (res_old, res_new):
        depot = res["depot_latlon"]
        m = folium.Map(location=[depot[0], depot[1]], zoom_start=10, control_scale=True)
        palette = ["red","orange","darkred","lightred","pink","beige"] if res["label"]=="old" else ["blue","green","darkblue","cadetblue","lightblue","darkgreen"]
        add_routes_layer(m, res["routes"], res["coords"], res["names"], client, profile,
                         f"{res['label'].upper()} Depot", palette)
        m.save(f"routes_map_{res['label']}.html")

    center_lat = (res_old["depot_latlon"][0] + res_new["depot_latlon"][0]) / 2.0
    center_lon = (res_old["depot_latlon"][1] + res_new["depot_latlon"][1]) / 2.0
    m_all = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
    add_routes_layer(m_all, res_old["routes"], res_old["coords"], res_old["names"], client, profile, "OLD Depot",
                     ["red","orange","darkred","lightred","pink","beige"])
    add_routes_layer(m_all, res_new["routes"], res_new["coords"], res_new["names"], client, profile, "NEW Depot",
                     ["blue","green","darkblue","cadetblue","lightblue","darkgreen"])
    folium.LayerControl().add_to(m_all)
    m_all.save("routes_compare_map.html")

    # Dashboard
    write_dashboard(res_old["total_cost"], res_new["total_cost"])

    return res_old, res_new

def zip_outputs(dirpath: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dirpath):
            for fn in files:
                fp = os.path.join(root, fn)
                arc = os.path.relpath(fp, dirpath)
                zf.write(fp, arc)
    mem.seek(0)
    return mem.read()

# ---------------- UI flow ----------------
if run:
    if not uploaded:
        st.error("Please upload deliveries.csv.")
    elif not old_warehouse.strip() or not new_warehouse.strip():
        st.error("Please enter both OLD and NEW depot locations.")
    elif not ors_key:
        st.error("Please provide your ORS API key.")
    else:
        with st.spinner("Running comparison..."):
            with tempfile.TemporaryDirectory() as workdir, cd(workdir):
                try:
                    res_old, res_new = pipeline(workdir, uploaded.read())

                    # KPIs
                    col1, col2, col3 = st.columns(3)
                    col1.metric("OLD total cost", f"{res_old['total_cost']:.0f}")
                    col2.metric("NEW total cost", f"{res_new['total_cost']:.0f}")
                    better = "NEW" if res_new["total_cost"] < res_old["total_cost"] else "OLD"
                    col3.metric("More cost-efficient", better)

                    # Images
                    st.subheader("Charts")
                    c1, c2 = st.columns(2)
                    c1.image("comparison_costs.png", caption="Total Cost (Old vs New)")
                    c2.image("stop_distance_delta.png", caption="Per-Stop Distance Change")

                    r1, r2 = st.columns(2)
                    r1.image("route_costs_old.png", caption="Per-Route Cost — OLD depot")
                    r2.image("route_costs_new.png", caption="Per-Route Cost — NEW depot")

                    r3, r4 = st.columns(2)
                    r3.image("route_distance_hist_old.png", caption="Route Distance Distribution — OLD")
                    r4.image("route_distance_hist_new.png", caption="Route Distance Distribution — NEW")

                    # Maps (embed HTML)
                    st.subheader("Maps")
                    with open("routes_compare_map.html", "r", encoding="utf-8") as f:
                        map_html = f.read()
                    st.components.v1.html(map_html, height=600, scrolling=True)

                    st.caption("Open individual maps:")
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.download_button("Download Combined Map (HTML)", data=open("routes_compare_map.html","rb").read(),
                                           file_name="routes_compare_map.html", mime="text/html")
                    with mcol2:
                        st.download_button("OLD Map (HTML)", data=open("routes_map_old.html","rb").read(),
                                           file_name="routes_map_old.html", mime="text/html")
                    with mcol3:
                        st.download_button("NEW Map (HTML)", data=open("routes_map_new.html","rb").read(),
                                           file_name="routes_map_new.html", mime="text/html")

                    # Data + Dashboard downloads
                    st.subheader("Downloads")
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.download_button("Dashboard (HTML)", data=open("comparison_dashboard.html","rb").read(),
                                           file_name="comparison_dashboard.html", mime="text/html")
                    with d2:
                        st.download_button("Stop Distance Delta (CSV)", data=open("stop_distance_delta.csv","rb").read(),
                                           file_name="stop_distance_delta.csv", mime="text/csv")
                    with d3:
                        st.download_button("All Outputs (ZIP)", data=zip_outputs(workdir),
                                           file_name="depot_comparison_outputs.zip", mime="application/zip")

                    st.success("Done.")

                except Exception as e:
                    st.exception(e)
