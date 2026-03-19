"""
InsideOut main pipeline: loads network and demand data, builds the MCC optimization
model, solves it via LP relaxation with L1 penalty, and exports results.
"""

import logging

import pandas as pd
import numpy as np
import matsim
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull

import Utils
import MaximumCustomerCoverage

# ── Configuration ──────────────────────────────────────────────────────────────

city_name = "singapore"

simplified_network_available = True
skeletons_available = False
n_zones = 2
l1_penalty = 1e-2
phase2_method = "greedy"  # "milp" or "greedy"

path_prefix = "/home/lorenzo/Documents/Github/proj-mt-schneider"
solutions_path = "solutions"
data_path = "data"

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    filename=f"insideout_{city_name}_{n_zones}_zones.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Network and population data loading ────────────────────────────────────────

if city_name == "Berlin":
    network_file = f"{path_prefix}/data/berlin-v5.5-network.xml.gz"
    population_file = f"{path_prefix}/data/berlin-v5.5-1pct.plans.xml.gz"
    cost_per_link_file = f"{path_prefix}/data/berlin_cost_per_link.csv"

if city_name == "singapore":
    network_file = f"{path_prefix}/data/singapore_network_combined.xml.gz"
    population_file = f"{data_path}/singapore_plans_1.2pct.xml.gz"
    cost_per_link_file = f"{path_prefix}/data/singapore_cost_per_link.csv"

plans = matsim.plan_reader_dataframe(population_file)

if simplified_network_available:
    if skeletons_available:
        link_df = pd.read_csv(f"{data_path}/skeleton_links_{city_name}_25_zones.csv")
    else:
        link_df = pd.read_csv(f"{data_path}/simplified_link_df_{city_name}.csv")

    nodes_df = pd.read_csv(f"{data_path}/simplified_node_df_{city_name}.csv")

    # Ensure largest strongly connected component
    link_df, nodes_df = Utils.get_largest_scc(link_df, nodes_df)

    nodes_df["node_id"] = nodes_df.node_id.astype(str)
    link_df["from_node"] = link_df.from_node.astype(str)
    link_df["to_node"] = link_df.to_node.astype(str)

    # Consistency check: all nodes referenced by links must exist in nodes_df
    nodes_in_nodes_df = set(nodes_df.node_id)
    link_nodes = set(link_df.from_node.to_list() + link_df.to_node.to_list())
    diff = link_nodes - nodes_in_nodes_df
    if diff:
        logger.warning(
            "Nodes in link_df but not in nodes_df: %d (link nodes: %d, node_df nodes: %d)",
            len(diff), len(link_nodes), len(nodes_in_nodes_df),
        )

else:
    network = matsim.read_network(network_file)
    cost_per_link_df = pd.read_csv(cost_per_link_file, sep=";")

    link_df = Utils.get_link_df(network)
    link_df = pd.merge(link_df, cost_per_link_df, how="left", left_on="link_id", right_on="link_id")
    link_df.rename(columns={"cost": "enabling_cost"}, inplace=True)

    # Remove public transport links
    link_df = link_df[~link_df.modes.str.contains("pt")]
    nodes_df = network.nodes

    link_df, nodes_df = Utils.get_largest_scc(link_df, nodes_df)

# ── Activity processing and coordinate transformation ──────────────────────────

activities_df = plans.activities.copy()

if city_name == "singapore" or city_name == "Dummy":
    from pyproj import Transformer

    coord_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3414")

    x, y = coord_transformer.transform(activities_df.y.values, activities_df.x.values)
    activities_df["x"] = y
    activities_df["y"] = x

    # Match each activity to the closest link by midpoint distance
    link_df["x_center"] = (link_df.x_from.values + link_df.x_to.values) / 2.0
    link_df["y_center"] = (link_df.y_from.values + link_df.y_to.values) / 2.0

    from sklearn.neighbors import KDTree

    kdt = KDTree(link_df[["x_center", "y_center"]].values, leaf_size=30, metric="euclidean")
    nearest_node = kdt.query(activities_df[["x", "y"]].values, k=1, return_distance=False)
    activities_df["link"] = link_df.iloc[nearest_node.flatten()].link_id.values

# Filter activities to links present in the network
mask = activities_df.link.isin(link_df.link_id.values)
activities_df = activities_df[mask].copy()

link_df.reset_index(drop=True, inplace=True)
activities_df.reset_index(drop=True, inplace=True)

link_df_prior = link_df.copy()

# ── Zone creation and link clustering ──────────────────────────────────────────

zones_df = Utils.k_means_zones(activities_df, n_zones)
zones_df = Utils.add_nodeId_to_zones(zones_df, nodes_df)

clustered_links_df = Utils.cluster_links(zones_df, link_df)
clustered_links_df["is_zone"] = False

# Mark links whose from_node matches a zone's nearest node
for zone_node in zones_df.nearest_node_id:
    link_idx = clustered_links_df.loc[link_df.from_node == zone_node, "is_zone"].index[0]
    clustered_links_df.loc[link_idx, "is_zone"] = True

Utils.export_to_kepler(f"network_{city_name}_{n_zones}_zones.csv", clustered_links_df, city_name)

if clustered_links_df.is_zone.sum() != n_zones:
    raise RuntimeError(
        f"Zone mapping error: {clustered_links_df.is_zone.sum()} zones mapped vs {n_zones} expected"
    )

# Compute convex hull areas per zone (for reference)
hulls = []
for zone in clustered_links_df.zone_label.unique():
    points = clustered_links_df[clustered_links_df.zone_label == zone][["x_from", "y_from"]].values
    hulls.append(ConvexHull(points))

areas_km2 = [hull.volume / (1000 * 1000) for hull in hulls]
hull_df = pd.DataFrame({"hull": hulls, "area_km2": areas_km2})

# Export station locations
zones = zones_df[["label", "x", "y"]].copy()
zones.rename(columns={"label": "station_id"}, inplace=True)
zones.to_csv(f"{solutions_path}/{city_name}_{n_zones}_stations.csv", index=False)

# ── Demand computation ─────────────────────────────────────────────────────────

demand_df = Utils.get_clustered_activities(clustered_links_df, activities_df)
demand_df = demand_df[["end_time_from", "zone_label_from", "zone_label_to", "plan_id_from"]]
demand_df.rename(
    columns={
        "end_time_from": "request_time",
        "zone_label_from": "station_id_from",
        "zone_label_to": "station_id_to",
        "plan_id_from": "matsim_plan_id",
    },
    inplace=True,
)
demand_df.sort_values("request_time", inplace=True)
demand_df["id"] = np.arange(0, len(demand_df))
demand_df[["id", "request_time", "station_id_from", "station_id_to", "matsim_plan_id"]].to_csv(
    f"{city_name}_demands_{n_zones}_stations.csv", index=False
)

cap_scale = 1.0

od_count_df = Utils.get_OD_counts(link_df, activities_df)

# Remove self-loops (same origin and destination zone)
od_count_df = od_count_df[od_count_df.zone_label_from != od_count_df.zone_label_to].copy()
od_count_df.reset_index(drop=True, inplace=True)
od_count_df["od_count"] = od_count_df.od_count.values / (24 * cap_scale)

link_df = clustered_links_df.copy()

from_to_links_per_node = Utils.get_from_and_to_links_per_node(link_df, exclude_pt=True)

# ── Scaling for numerical stability ────────────────────────────────────────────

cost_scale = 10000.0
link_df["enabling_cost"] = link_df.enabling_cost / cost_scale

cap_scale = 100.0
link_df["capacity"] = link_df.capacity.values / cap_scale
od_count_df["od_count"] = od_count_df.od_count.values / (24 * cap_scale)

budget = 10000000 / cost_scale

# Enforce minimum enabling cost per link
min_cost = 1000 / cost_scale
link_df.loc[link_df.enabling_cost < min_cost, "enabling_cost"] = min_cost

logger.info("Data setup complete, calling LP solver")

# ── Build and solve the optimization model ─────────────────────────────────────

mcc_model, served_c, flow_c_l, beta_l = MaximumCustomerCoverage.get_maximum_customer_coverage_model(
    link_df,
    od_count_df,
    zones_df,
    from_to_links_per_node,
    budget,
)

flow_sol, beta_sol, served_sol, enabled_sol = MaximumCustomerCoverage.solve_relaxed(
    model=mcc_model,
    served_c=served_c,
    flow_c_l=flow_c_l,
    beta_l=beta_l,
    n_links=len(link_df),
    n_od_flows=len(od_count_df),
    budget=budget,
    link_df=link_df,
    n_zones=n_zones,
    l1_penalty=l1_penalty,
)

# ── Post-processing and export ─────────────────────────────────────────────────

link_sol = link_df.copy()
n_links = len(link_df)

# Add per-OD passenger flow columns (rescaled)
for i in range(len(od_count_df)):
    link_sol[f"paxFlow_{i}"] = cap_scale * flow_sol[i * n_links : (i + 1) * n_links]

link_sol["rebalFlow"] = cap_scale * beta_sol

# Determine enabled links from total flow
enabled_sol = flow_sol.reshape((len(od_count_df), n_links)).sum(axis=0) + beta_sol
link_sol["enabled"] = np.where(enabled_sol > 1e-6, 1.0, 0.0)
link_sol["enabled_frac"] = enabled_sol

# Restore original enabling costs for reporting
link_sol["enabling_cost"] = link_df_prior.enabling_cost.values

# ── Solution summary ───────────────────────────────────────────────────────────

cost = (link_sol["enabled"] * link_sol.enabling_cost.values).sum()
logger.info("Cost: %s", f"{cost:,.2f}")

resume_file = f"{solutions_path}/resume_{city_name}_relaxed_zones{n_zones}.txt"
with open(resume_file, "w") as f:
    f.write(f"Cost: {cost}\n")
    f.write(f"L1 penalty: {l1_penalty}\n")

    num_enabled = np.sum(link_sol["enabled"])
    f.write(f"Number of enabled links: {num_enabled}\n")
    logger.info("Number of enabled links: %d", num_enabled)

    for threshold in [0.3, 0.5, 0.9, 0.99]:
        count = np.sum(enabled_sol >= threshold)
        f.write(f"Number of >={threshold} enabled links: {count}\n")
        logger.info("Number of >=%s enabled links: %d", threshold, count)

    total_enabled_sum = np.sum(enabled_sol[enabled_sol > 1e-6])
    f.write(f"Total enabled value sum: {total_enabled_sum:,.2f}\n")
    logger.info("Total enabled value sum: %s", f"{total_enabled_sum:,.2f}")

# Export LP relaxation enabled links to Kepler.gl
Utils.export_to_kepler(
    f"{solutions_path}/kepler_solution_{city_name}_relaxed_{n_zones}_zones.csv",
    link_sol[link_sol["enabled"] == 1],
    city_name,
)

# ── Phase 2: Cost reduction on the enabled subnetwork ──────────────────────────
# Two methods available (configured via phase2_method):
#   "milp"   — exact MILP that minimizes enabling cost with binary variables
#   "greedy" — greedy link pruning by ascending flow, re-solving LP each step

n_od = len(od_count_df)
enabled_mask = link_sol["enabled"] == 1
enabled_indices = np.where(enabled_mask.values)[0]
n_enabled = int(enabled_mask.sum())
min_demand = served_sol.sum()

logger.info(
    "Phase 2 (%s): %d enabled links (out of %d), demand floor %.4f",
    phase2_method, n_enabled, n_links, min_demand,
)

# Filter link_df to enabled-only subnetwork
enabled_link_df = link_df[enabled_mask].copy()
enabled_link_df.reset_index(drop=True, inplace=True)

# Recompute node-link adjacency on the reduced network (indices must match reset DataFrame)
enabled_from_to = Utils.get_from_and_to_links_per_node(enabled_link_df, exclude_pt=True)

# Map LP relaxation solution to reduced network indices
warm_flow = np.zeros(n_od * n_enabled)
for i in range(n_od):
    for new_j, old_j in enumerate(enabled_indices):
        warm_flow[i * n_enabled + new_j] = flow_sol[i * n_links + old_j]

warm_beta = beta_sol[enabled_indices]
warm_served = served_sol

if phase2_method == "milp":
    # Build base model (flow conservation, rebalancing, capacity) on reduced network
    mc_model, mc_served_c, mc_flow_c_l, mc_beta_l = MaximumCustomerCoverage.get_maximum_customer_coverage_model(
        enabled_link_df, od_count_df, zones_df, enabled_from_to, budget,
    )

    # Solve cost-minimization MILP with LP solution as warm start
    mc_flow_sol, mc_beta_sol, mc_served_sol, mc_enabled_sol = MaximumCustomerCoverage.solve_min_cost(
        model=mc_model,
        served_c=mc_served_c,
        flow_c_l=mc_flow_c_l,
        beta_l=mc_beta_l,
        n_links=n_enabled,
        n_od_flows=n_od,
        link_df=enabled_link_df,
        n_zones=n_zones,
        min_demand=min_demand,
        warm_start_flow=warm_flow,
        warm_start_beta=warm_beta,
        warm_start_served=warm_served,
    )

elif phase2_method == "greedy":
    # Greedy link pruning: disable links one-by-one (smallest flow first),
    # re-solving the LP each time to check feasibility.
    greedy_enabled, mc_flow_sol, mc_beta_sol, mc_served_sol = MaximumCustomerCoverage.greedy_prune(
        link_df=enabled_link_df,
        od_count_df=od_count_df,
        zones_df=zones_df,
        from_to_links_per_node=enabled_from_to,
        flow_sol=warm_flow,
        beta_sol=warm_beta,
        served_sol=warm_served,
        n_zones=n_zones,
    )
    mc_enabled_sol = greedy_enabled.astype(float)

else:
    raise ValueError(f"Unknown phase2_method: {phase2_method!r}. Use 'milp' or 'greedy'.")

# ── Phase 2 post-processing and export ─────────────────────────────────────────

mc_link_sol = enabled_link_df.copy()

for i in range(n_od):
    mc_link_sol[f"paxFlow_{i}"] = cap_scale * mc_flow_sol[i * n_enabled : (i + 1) * n_enabled]

mc_link_sol["rebalFlow"] = cap_scale * mc_beta_sol
mc_link_sol["enabled"] = mc_enabled_sol

# Restore original enabling costs (look up from link_df_prior by link_id)
mc_link_sol = mc_link_sol.merge(
    link_df_prior[["link_id", "enabling_cost"]].rename(columns={"enabling_cost": "enabling_cost_orig"}),
    on="link_id",
    how="left",
)
mc_link_sol["enabling_cost"] = mc_link_sol["enabling_cost_orig"]
mc_link_sol.drop(columns=["enabling_cost_orig"], inplace=True)

mc_cost = (mc_link_sol["enabled"] * mc_link_sol.enabling_cost.values).sum()
mc_served_total = mc_served_sol.sum()
mc_num_enabled = int(mc_enabled_sol.sum())

logger.info("Phase 2 cost: %s", f"{mc_cost:,.2f}")
logger.info("Phase 2 served demand: %.4f (floor was %.4f)", mc_served_total, min_demand)
logger.info("Phase 2 enabled links: %d (of %d in reduced network)", mc_num_enabled, n_enabled)

phase2_label = "mincost" if phase2_method == "milp" else "greedy"
resume_file_mc = f"{solutions_path}/resume_{city_name}_{phase2_label}_zones{n_zones}.txt"
with open(resume_file_mc, "w") as f:
    f.write(f"Method: {phase2_method}\n")
    f.write(f"Cost: {mc_cost}\n")
    f.write(f"Served demand: {mc_served_total}\n")
    f.write(f"Min demand floor: {min_demand}\n")
    f.write(f"Number of enabled links: {mc_num_enabled}\n")
    f.write(f"Total links in reduced network: {n_enabled}\n")

Utils.export_to_kepler(
    f"{solutions_path}/kepler_solution_{city_name}_{phase2_label}_{n_zones}_zones.csv",
    mc_link_sol[mc_link_sol["enabled"] >= 0.5],
    city_name,
)
