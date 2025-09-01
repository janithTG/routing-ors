#!/usr/bin/env python3
"""
routing_ors_compare.py
Compare OLD vs NEW depot (warehouse) for the SAME delivery locations and produce rich visualizations.

Outputs (in addition to previous ones):
- route_costs_old.png, route_costs_new.png
- route_distance_hist_old.png, route_distance_hist_new.png
- stop_distance_delta.csv, stop_distance_delta.png
- stops_delta_map.html
- comparison_dashboard.html
"""

import os
import math
import argparse
import time
from typing import List, Tuple, Dict

import pandas as pd
import folium
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import openrouteservice as ors
from openrouteservice import convert
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ------------------ I/O & Geocoding ------------------

def load_locations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "name" not in df.columns or "demand" not in df.columns:
        raise ValueError("CSV must include columns: name, demand (plus address or latitude/longitude).")
    for col in ["name", "address"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ["latitude", "longitude", "demand"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["demand"] = df["demand"].fillna(0).clip(lower=0).astype(int)
    return df

def geocode_if_needed(client: ors.Client, name: str, address: str, lat, lon) -> Tuple[float, float]:
    if pd.notna(lat) and pd.notna(lon):
        return float(lat), float(lon)
    query = (address or "").strip() or (name or "").strip()
    if not query:
        raise ValueError(f"No address/coords for stop '{name}'.")
    try:
        res = client.pelias_search(text=query)
    except AttributeError:
        from openrouteservice.geocode import pelias_search
        res = pelias_search(client, text=query)
    feats = (res or {}).get("features", [])
    if not feats:
        raise ValueError(f"Geocoding failed for '{name}' ({query}).")
    lng, lat = feats[0]["geometry"]["coordinates"]  # ORS: [lon,lat]
    return float(lat), float(lng)

# ------------------ ORS Matrix (batched) ------------------

def build_distance_matrix_ors(client: ors.Client, coords_latlon: List[Tuple[float,float]], profile: str) -> List[List[int]]:
    """
    NxN distances in meters using ORS Matrix API (elements limit ~3500 per request).
    coords_latlon: list of (lat,lon). ORS expects [lon,lat].
    """
    N = len(coords_latlon)
    if N < 2:
        return [[0]*N for _ in range(N)]
    coords_lonlat = [[lng, lat] for (lat, lng) in coords_latlon]
    matrix = [[0]*N for _ in range(N)]

    max_elements = 3500
    origins_per_call = max(1, max_elements // N)

    for start in range(0, N, origins_per_call):
        sources_idx = list(range(start, min(start + origins_per_call, N)))
        try:
            resp = client.distance_matrix(
                locations=coords_lonlat,
                profile=profile,
                sources=sources_idx,
                destinations=list(range(N)),
                metrics=["distance"],
                units="m",
            )
        except ors.exceptions.ApiError as e:
            raise RuntimeError(f"ORS matrix error: {e}")
        distances = resp.get("distances", [])
        if len(distances) != len(sources_idx):
            raise RuntimeError("Matrix response size mismatch.")
        for row_i, i in enumerate(sources_idx):
            row = distances[row_i]
            if len(row) != N:
                raise RuntimeError("Matrix row size mismatch.")
            for j, val in enumerate(row):
                matrix[i][j] = int(val) if val is not None else 10**9
        time.sleep(0.05)
    return matrix

# ------------------ VRP ------------------

def solve_vrp(distance_matrix_m, demands, vehicle_capacity, depot_index, max_vehicles, time_limit_sec=20):
    N = len(distance_matrix_m)
    manager = pywrapcp.RoutingIndexManager(N, max_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_index, to_index):
        i = manager.IndexToNode(from_index); j = manager.IndexToNode(to_index)
        return int(distance_matrix_m[i][j])
    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    def demand_cb(from_index):
        i = manager.IndexToNode(from_index)
        return int(demands[i])
    demand_cb = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, [int(vehicle_capacity)]*max_vehicles, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_sec)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    routes = []
    for v in range(max_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue
        nodes = []; total_m = 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            nodes.append(node)
            nxt = solution.Value(routing.NextVar(idx))
            if not routing.IsEnd(nxt):
                i = manager.IndexToNode(idx); j = manager.IndexToNode(nxt)
                total_m += distance_matrix_m[i][j]
            idx = nxt
        nodes.append(manager.IndexToNode(idx))
        routes.append({"vehicle": v, "nodes": nodes, "distance_m": int(total_m)})
    return routes

# ------------------ Maps ------------------

def fetch_leg_geometry(client: ors.Client, a_latlon, b_latlon, profile: str):
    coords_lonlat = [(a_latlon[1], a_latlon[0]), (b_latlon[1], b_latlon[0])]
    r = client.directions(coordinates=coords_lonlat, profile=profile, format="json")
    geom = r["routes"][0]["geometry"]
    decoded = convert.decode_polyline(geom)  # GeoJSON LineString [lon,lat]
    lonlat = decoded["coordinates"]
    return [[lat, lon] for lon, lat in lonlat]

def add_routes_layer(m: folium.Map, routes, coords_latlon, names, client, profile, layer_name: str, color_palette: List[str]):
    fg = folium.FeatureGroup(name=layer_name, show=True)
    depot_lat, depot_lon = coords_latlon[0]
    fg.add_child(folium.Marker([depot_lat, depot_lon], popup=f"Depot: {names[0]}", icon=folium.Icon(color="black")))
    for i, (lat, lon) in enumerate(coords_latlon[1:], start=1):
        fg.add_child(folium.Marker([lat, lon], popup=names[i], icon=folium.Icon(color="gray")))
    for r_i, r in enumerate(routes):
        color = color_palette[r_i % len(color_palette)]
        seq = r["nodes"]
        for a, b in zip(seq[:-1], seq[1:]):
            try:
                pts = fetch_leg_geometry(client, coords_latlon[a], coords_latlon[b], profile)
                fg.add_child(folium.PolyLine(pts, weight=5, opacity=0.8, color=color))
            except Exception:
                fg.add_child(folium.PolyLine([coords_latlon[a], coords_latlon[b]], weight=3, opacity=0.6, color=color, dash_array="5,10"))
        if len(seq) > 2:
            fg.add_child(folium.Marker(coords_latlon[seq[1]],
                                       popup=f"{layer_name} R{r_i+1}",
                                       icon=folium.DivIcon(html=f"<div style='color:{color};font-weight:600'>R{r_i+1}</div>")))
    m.add_child(fg)

# ------------------ Visualization helpers ------------------

def plot_total_cost_bar(old_cost, new_cost, outfile="comparison_costs.png"):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["Old Depot", "New Depot"], [old_cost, new_cost])
    ax.set_ylabel("Total Cost")
    ax.set_title("Total Cost Comparison")
    for i, v in enumerate([old_cost, new_cost]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    plt.savefig(outfile, dpi=150); plt.close(fig)

def plot_route_costs(routes_df: pd.DataFrame, label: str):
    if routes_df.empty: return
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(routes_df["route_id"].astype(str), routes_df["estimated_cost"])
    ax.set_xlabel("Route ID")
    ax.set_ylabel("Cost")
    ax.set_title(f"Per-Route Cost — {label.capitalize()} Depot")
    for x, v in zip(routes_df["route_id"].astype(str), routes_df["estimated_cost"]):
        ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    plt.savefig(f"route_costs_{label}.png", dpi=150); plt.close(fig)

def plot_route_distance_hist(routes_df: pd.DataFrame, label: str):
    if routes_df.empty: return
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(routes_df["total_distance_km"], bins=min(10, len(routes_df)), edgecolor="black")
    ax.set_xlabel("Route Distance (km)")
    ax.set_ylabel("Count")
    ax.set_title(f"Route Distance Distribution — {label.capitalize()} Depot")
    fig.tight_layout()
    plt.savefig(f"route_distance_hist_{label}.png", dpi=150); plt.close(fig)

def build_stops_delta_df(stop_names: List[str], old_depot_row_m: List[int], new_depot_row_m: List[int]) -> pd.DataFrame:
    # depot row includes depot→depot at index 0; stops start at index 1
    data = []
    for i, name in enumerate(stop_names, start=1):
        old_m = old_depot_row_m[i]
        new_m = new_depot_row_m[i]
        data.append({
            "stop_name": name,
            "old_depot_distance_km": round(old_m/1000.0, 3),
            "new_depot_distance_km": round(new_m/1000.0, 3),
            "delta_km": round((new_m - old_m)/1000.0, 3)  # negative = improvement
        })
    df = pd.DataFrame(data).sort_values("delta_km")
    df["improved"] = df["delta_km"] < 0
    return df

def plot_stop_delta(df: pd.DataFrame, outfile="stop_distance_delta.png"):
    if df.empty: return
    # Horizontal bar: most improved (negative) at top
    df_plot = df.sort_values("delta_km")
    colors = ["green" if x < 0 else "red" for x in df_plot["delta_km"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4*len(df_plot))))
    ax.barh(df_plot["stop_name"], df_plot["delta_km"], color=colors)
    ax.set_xlabel("Δ Distance to Depot (km)  [new - old]  (negative = better)")
    ax.set_title("Per-Stop Distance Change (Driving distance depot→stop)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150); plt.close(fig)

def build_stops_delta_map(df: pd.DataFrame, stops_coords: List[Tuple[float,float]], outfile="stops_delta_map.html"):
    # df is sorted but corresponds to original stops list; rebuild index by name
    name_to_idx = {name: i for i, name in enumerate(df["stop_name"].tolist())}
    # center map roughly on mean of all stops
    lat_avg = sum(lat for lat, _ in stops_coords) / len(stops_coords)
    lon_avg = sum(lon for _, lon in stops_coords) / len(stops_coords)
    m = folium.Map(location=[lat_avg, lon_avg], zoom_start=10, control_scale=True)
    for _, row in df.iterrows():
        i = name_to_idx[row["stop_name"]]
        lat, lon = stops_coords[i]
        improved = row["improved"]
        color = "green" if improved else "red"
        radius = max(6, min(18, int(abs(row["delta_km"])*2)))  # scale by delta
        popup = (f"<b>{row['stop_name']}</b><br>"
                 f"Old depot→stop: {row['old_depot_distance_km']} km<br>"
                 f"New depot→stop: {row['new_depot_distance_km']} km<br>"
                 f"Δ: {row['delta_km']} km")
        folium.CircleMarker([lat, lon], radius=radius, color=color, fill=True, fillOpacity=0.6, popup=popup).add_to(m)
    m.save(outfile)
    return outfile

def write_dashboard(old_cost, new_cost):
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Depot Comparison Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 24px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; flex: 1 1 420px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
    a.button {{ display:inline-block; margin-top:8px; padding:8px 12px; background:#0b6; color:#fff; text-decoration:none; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>Depot Comparison Dashboard</h1>
  <p><b>Total cost:</b> Old = {old_cost:.0f} &nbsp;&nbsp; New = {new_cost:.0f}</p>

  <div class="row">
    <div class="card">
      <h3>Total Cost (Old vs New)</h3>
      <img src="comparison_costs.png" alt="Total cost bar"/>
    </div>
    <div class="card">
      <h3>Per-Stop Distance Change</h3>
      <img src="stop_distance_delta.png" alt="Per-stop delta"/>
      <p><a class="button" href="stop_distance_delta.csv">Download CSV</a>
         <a class="button" href="stops_delta_map.html">Open Map</a></p>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h3>Per-Route Cost — Old Depot</h3>
      <img src="route_costs_old.png" alt="Route costs old"/>
    </div>
    <div class="card">
      <h3>Per-Route Cost — New Depot</h3>
      <img src="route_costs_new.png" alt="Route costs new"/>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h3>Route Distance Distribution — Old</h3>
      <img src="route_distance_hist_old.png" alt="Route distance hist old"/>
    </div>
    <div class="card">
      <h3>Route Distance Distribution — New</h3>
      <img src="route_distance_hist_new.png" alt="Route distance hist new"/>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h3>Routes Map (toggle layers)</h3>
      <p><a class="button" href="routes_compare_map.html">Open Combined Map</a></p>
      <p><a class="button" href="routes_map_old.html">Open Old Map</a>
         <a class="button" href="routes_map_new.html">Open New Map</a></p>
    </div>
  </div>
</body>
</html>
"""
    with open("comparison_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)

# ------------------ Scenario Runner ------------------

def run_scenario(label: str,
                 depot_addr_or_latlon: str,
                 client: ors.Client,
                 stops_coords_latlon: List[Tuple[float,float]],
                 stops_names: List[str],
                 demands_stops: List[int],
                 profile: str,
                 capacity: int,
                 max_vehicles: int,
                 time_limit: int,
                 cost_per_km: float):
    # Depot geocode/parse
    wh = depot_addr_or_latlon.strip()
    wh_lat = wh_lon = None
    if "," in wh:
        parts = [p.strip() for p in wh.split(",")]
        if len(parts) == 2:
            try:
                wh_lat = float(parts[0]); wh_lon = float(parts[1])
            except:
                wh_lat = wh_lon = None
    if wh_lat is None:
        wh_lat, wh_lon = geocode_if_needed(client, f"Depot ({label})", wh, None, None)

    coords = [(wh_lat, wh_lon)] + stops_coords_latlon
    names  = [f"Depot ({label})"] + stops_names
    demands = [0] + demands_stops

    # Distances
    dist_m = build_distance_matrix_ors(client, coords, profile=profile)

    # Vehicles
    total_demand = sum(demands)
    mv = max(1, math.ceil(total_demand / max(1, capacity))) if max_vehicles is None else max_vehicles

    # Solve
    routes = solve_vrp(dist_m, demands, capacity, depot_index=0, max_vehicles=mv, time_limit_sec=time_limit)
    if not routes:
        raise RuntimeError(f"No VRP solution for {label}. Try increasing capacity/vehicles or time_limit.")

    # Summaries
    km = lambda m: round(m/1000.0, 3)
    route_rows, stop_rows = [], []
    total_dist_km = 0.0
    for r_idx, r in enumerate(routes, start=1):
        route_km = km(r["distance_m"])
        total_dist_km += route_km
        est_cost = round(route_km * cost_per_km, 2)
        seq_names = [names[i] for i in r["nodes"]]
        route_rows.append({
            "scenario": label,
            "route_id": r_idx,
            "vehicle": r["vehicle"],
            "stops_sequence": " → ".join(seq_names),
            "total_distance_km": route_km,
            "estimated_cost": est_cost
        })
        order = 0
        for node in r["nodes"][1:-1]:
            order += 1
            stop_rows.append({
                "scenario": label,
                "route_id": r_idx,
                "vehicle": r["vehicle"],
                "visit_order": order,
                "stop_name": names[node],
                "demand": demands[node]
            })

    # Write per-scenario CSVs
    routes_df = pd.DataFrame(route_rows)
    stops_df = pd.DataFrame(stop_rows)
    routes_df.to_csv(f"routes_summary_{label}.csv", index=False)
    stops_df.to_csv(f"stops_assignment_{label}.csv", index=False)

    # Return structured result (include dist_m for depot row use)
    return {
        "label": label,
        "depot_latlon": (wh_lat, wh_lon),
        "coords": coords,
        "names": names,
        "routes": routes,
        "routes_df": routes_df,
        "stops_df": stops_df,
        "dist_m": dist_m,  # for depot→stop distances
        "total_distance_km": round(total_dist_km, 3),
        "total_cost": round(total_dist_km * cost_per_km, 2),
        "vehicles_used": len(routes)
    }

# ------------------ Main ------------------

def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Compare OLD vs NEW depot (OpenRouteService + OR-Tools) with visualizations")
    ap.add_argument("--old_warehouse", required=True, help="Old depot address (or 'lat,lon')")
    ap.add_argument("--new_warehouse", required=True, help="New depot address (or 'lat,lon')")
    ap.add_argument("--locations_csv", default="deliveries.csv", help="CSV: name,address,latitude,longitude,demand")
    ap.add_argument("--capacity", type=int, required=True, help="Truck capacity (e.g., pallets)")
    ap.add_argument("--cost_per_km", type=float, required=True, help="Cost per km")
    ap.add_argument("--profile", default="driving-hgv", help="ORS profile: driving-hgv, driving-car, etc.")
    ap.add_argument("--max_vehicles", type=int, default=None, help="Max trucks (default=ceil(total_demand/capacity))")
    ap.add_argument("--time_limit", type=int, default=20, help="VRP solver time limit (sec)")
    ap.add_argument("--api_key", default=None, help="Override ORS key; otherwise use .env")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENROUTESERVICE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTESERVICE_API_KEY not found. Put it in .env or pass --api_key.")
    client = ors.Client(key=api_key)

    # Load deliveries; geocode once
    df = load_locations(args.locations_csv)
    stops_coords, stops_names, demands_stops = [], [], []
    for _, row in df.iterrows():
        lat, lon = row.get("latitude"), row.get("longitude")
        latlon = geocode_if_needed(client, row["name"], row.get("address", ""), lat, lon)
        stops_coords.append(latlon)
        stops_names.append(row["name"])
        demands_stops.append(int(row["demand"]))

    # Run old/new scenarios
    res_old = run_scenario("old", args.old_warehouse, client, stops_coords, stops_names, demands_stops,
                           args.profile, args.capacity, args.max_vehicles, args.time_limit, args.cost_per_km)
    res_new = run_scenario("new", args.new_warehouse, client, stops_coords, stops_names, demands_stops,
                           args.profile, args.capacity, args.max_vehicles, args.time_limit, args.cost_per_km)

    # Overall comparison CSV
    comp_df = pd.DataFrame([
        {"scenario": "old", "total_distance_km": res_old["total_distance_km"], "total_cost": res_old["total_cost"], "vehicles_used": res_old["vehicles_used"]},
        {"scenario": "new", "total_distance_km": res_new["total_distance_km"], "total_cost": res_new["total_cost"], "vehicles_used": res_new["vehicles_used"]},
    ])
    comp_df.to_csv("overall_comparison.csv", index=False)

    # Charts: totals, per-route
    plot_total_cost_bar(res_old["total_cost"], res_new["total_cost"], "comparison_costs.png")
    plot_route_costs(res_old["routes_df"], "old")
    plot_route_costs(res_new["routes_df"], "new")
    plot_route_distance_hist(res_old["routes_df"], "old")
    plot_route_distance_hist(res_new["routes_df"], "new")

    # Per-stop delta using depot→stop driving distance (row 0 of matrix)
    old_row0 = res_old["dist_m"][0]
    new_row0 = res_new["dist_m"][0]
    stop_delta = build_stops_delta_df(stops_names, old_row0, new_row0)
    stop_delta.to_csv("stop_distance_delta.csv", index=False)
    plot_stop_delta(stop_delta, "stop_distance_delta.png")
    build_stops_delta_map(stop_delta, stops_coords, "stops_delta_map.html")

    # Per-scenario maps + combined map
    for res in (res_old, res_new):
        depot = res["depot_latlon"]
        m = folium.Map(location=[depot[0], depot[1]], zoom_start=10, control_scale=True)
        palette = ["red","orange","darkred","lightred","pink","beige"] if res["label"]=="old" else ["blue","green","darkblue","cadetblue","lightblue","darkgreen"]
        add_routes_layer(m, res["routes"], res["coords"], res["names"], client, args.profile, f"{res['label'].upper()} Depot", palette)
        m.save(f"routes_map_{res['label']}.html")

    center_lat = (res_old["depot_latlon"][0] + res_new["depot_latlon"][0]) / 2.0
    center_lon = (res_old["depot_latlon"][1] + res_new["depot_latlon"][1]) / 2.0
    m_all = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
    add_routes_layer(m_all, res_old["routes"], res_old["coords"], res_old["names"], client, args.profile, "OLD Depot", ["red","orange","darkred","lightred","pink","beige"])
    add_routes_layer(m_all, res_new["routes"], res_new["coords"], res_new["names"], client, args.profile, "NEW Depot", ["blue","green","darkblue","cadetblue","lightblue","darkgreen"])
    folium.LayerControl().add_to(m_all)
    m_all.save("routes_compare_map.html")

    # Dashboard HTML to tie it together
    write_dashboard(res_old["total_cost"], res_new["total_cost"])

    # Console summary
    print("\n=== Comparison Summary ===")
    print(f"OLD depot -> total_distance_km={res_old['total_distance_km']}, total_cost={res_old['total_cost']}, routes={res_old['vehicles_used']}")
    print(f"NEW depot -> total_distance_km={res_new['total_distance_km']}, total_cost={res_new['total_cost']}, routes={res_new['vehicles_used']}")
    better = "NEW" if res_new["total_cost"] < res_old["total_cost"] else "OLD"
    print(f"→ More cost-efficient: {better} depot")
    print("\nFiles created:")
    print("- routes_summary_old.csv, stops_assignment_old.csv, routes_map_old.html")
    print("- routes_summary_new.csv, stops_assignment_new.csv, routes_map_new.html")
    print("- overall_comparison.csv, comparison_costs.png, routes_compare_map.html")
    print("- route_costs_old.png / new.png, route_distance_hist_old.png / new.png")
    print("- stop_distance_delta.csv, stop_distance_delta.png, stops_delta_map.html")
    print("- comparison_dashboard.html")

if __name__ == "__main__":
    main()
