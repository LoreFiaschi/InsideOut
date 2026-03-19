import pandas as pd
import numpy as np
import matsim 
import Utils
import geopandas as gpd
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import MaximumCustomerCoverage

city_name = "singapore"

simplified_network_available = True
skeletons_available = False
n_zones = 2

#solver_type = "integer"; l1_penalty = 0.0
solver_type = "relaxed"; l1_penalty = 1e-2
#solver_type = "relaxed_integer"; l1_penalty = 1e-2

path_prefix = f"/home/lorenzo/Documents/Github/proj-mt-schneider"
solutions_path = "solutions"
data_path = "data"



if city_name=="Berlin":
    network_file = f"{path_prefix}/data/berlin-v5.5-network.xml.gz"
    population_file = f"{path_prefix}/data/berlin-v5.5-1pct.plans.xml.gz"
    cost_per_link_file = f"{path_prefix}/data/berlin_cost_per_link.csv"

if city_name == "singapore":
    network_file = f"{path_prefix}/data/singapore_network_combined.xml.gz"
    population_file = f"{data_path}/singapore_plans_1.2pct.xml.gz"
    cost_per_link_file = f"{path_prefix}/data/singapore_cost_per_link.csv"

plans = matsim.plan_reader_dataframe(population_file)

if simplified_network_available:
  # # load simplified network
  if skeletons_available:
    link_df = pd.read_csv(f"{data_path}/skeleton_links_{city_name}_25_zones.csv")
  else:
    link_df = pd.read_csv(f"{data_path}/simplified_link_df_{city_name}.csv")

  nodes_df = pd.read_csv(f"{data_path}/simplified_node_df_{city_name}.csv")

  # # run SSC filter (rerun to be sure)
  link_df, nodes_df = Utils.get_largest_scc(link_df, nodes_df)

  # # update link_Df and node_df
  nodes_df["node_id"] = nodes_df.node_id.astype(str)
  link_df["from_node"] = link_df.from_node.astype(str)
  link_df["to_node"] = link_df.to_node.astype(str)

  # check consistency
  nodes_in_nodes_df = set(nodes_df.node_id)
  link_nodes = link_df.from_node.to_list() + link_df.to_node.to_list()
  link_nodes = np.unique(link_nodes)
  nodes_in_link_df = set(link_nodes)
  diff = (nodes_in_link_df-nodes_in_nodes_df)
  if len(diff) != 0:
    print(f"Num Nodes in Node_df {len(nodes_in_nodes_df):,d}. Num Nodes in Link_df: {len(nodes_in_link_df):,d}. Diff: {len(diff)}")

else:
    network = matsim.read_network(network_file)

    cost_per_link_df = pd.read_csv(cost_per_link_file, sep=";")

    link_df = Utils.get_link_df(network)
    # merge with costs
    link_df = pd.merge(link_df, cost_per_link_df, how="left", left_on="link_id", right_on="link_id")
    link_df.rename(columns={"cost": "enabling_cost"}, inplace=True)

    link_df = link_df[~link_df.modes.str.contains("pt")]

    nodes_df = network.nodes

    link_df, nodes_df = Utils.get_largest_scc(link_df, nodes_df)

# # UNCOMMENT BELOW IF SIMPLIFICATION DESIRED === if done once, best to save it and reload.

#link_df.reset_index(inplace=True,drop=True)
#nodes_df.reset_index(inplace=True,drop=True)
#new_link_df, new_node_df = Utils.simplify_network(link_df, nodes_df)

#print(
#    f"New Links {len(new_link_df)}, i.e. filtered {1- len(new_link_df)/len(link_df):.2%}. New Nodes {len(new_node_df):,d}, i.e. filtered {1-len(new_node_df)/len(nodes_df):.2%}"
#)

# SAVE DATA
#new_link_df.to_csv(f"simplified_link_df_{city_name}.csv",index=False)
#new_node_df.to_csv(f"simplified_node_df_{city_name}.csv",index=False)

activities_df = plans.activities.copy()

if city_name=="singapore" or city_name=="Dummy":
    # transform coordinates
    from pyproj import Transformer

    coord_transformer = Transformer.from_crs("EPSG:4326","EPSG:3414")

    # from
    x, y = coord_transformer.transform(activities_df.y.values, activities_df.x.values)
    activities_df["x"] = y
    activities_df["y"] = x

    # match origin destination to closest link
    link_df["x_center"] = (link_df.x_from.values + link_df.x_to.values)/2.0
    link_df["y_center"] = (link_df.y_from.values + link_df.y_to.values)/2.0

    from sklearn.neighbors import KDTree
    kdt = KDTree(link_df[["x_center", "y_center"]].values, leaf_size=30, metric="euclidean")
    nearest_node = kdt.query(activities_df[["x", "y"]].values, k=1, return_distance=False)
    activities_df["link"] = link_df.iloc[nearest_node.flatten()].link_id.values

mask = activities_df.link.isin(link_df.link_id.values)
activities_df = activities_df[mask].copy()

# reset index on relevant data
link_df.reset_index(drop=True,inplace=True)
activities_df.reset_index(drop=True,inplace=True)

link_df_prior = link_df.copy()

zones_df = Utils.k_means_zones(activities_df, n_zones) # We should fix meaningful zones and then use k_monoid instead of k_means
zones_df = Utils.add_nodeId_to_zones(zones_df,nodes_df)

clustered_links_df = Utils.cluster_links(zones_df, link_df)
clustered_links_df["is_zone"] = False
# set links corresponding to zones --- choose the first option
for zone_node in zones_df.nearest_node_id:
    link_idx = clustered_links_df.loc[link_df.from_node == zone_node,"is_zone"].index[0]
    clustered_links_df.loc[link_idx,"is_zone"] = True     # TODO problem

Utils.export_to_kepler(f"network_{city_name}_{n_zones}_zones.csv",clustered_links_df,city_name)

if clustered_links_df.is_zone.sum() != n_zones:
    raise RuntimeError(f"Some issue with zones --- not all zones mapped to edges! {clustered_links_df.is_zone.sum()} vs {n_zones}  ")

hulls = []
for zone in clustered_links_df.zone_label.unique():
    points = clustered_links_df[clustered_links_df.zone_label==zone][["x_from","y_from"]].values
    hulls.append(ConvexHull(points))

areas_km2 = []
for hull in hulls:
    areas_km2.append(hull.volume/(1000*1000))

hull_df = pd.DataFrame({"hull":hulls,"area_km2":areas_km2})

# network based on stations.
zones = zones_df.copy()
zones = zones[["label", "x", "y"]]
zones.rename(columns={"label": "station_id"},inplace=True)
zones.to_csv(f"{solutions_path}/{city_name}_{n_zones}_stations.csv",index=False)

# trips based on activities
demand_df = Utils.get_clustered_activities(clustered_links_df, activities_df)
demand_df = demand_df[["end_time_from","zone_label_from","zone_label_to","plan_id_from"]]
demand_df.rename(columns={"end_time_from":"request_time","zone_label_from":"station_id_from","zone_label_to":"station_id_to","plan_id_from":"matsim_plan_id"},inplace=True)
demand_df.sort_values("request_time",inplace=True)
demand_df["id"] = np.arange(0,len(demand_df))
demand_df[["id","request_time","station_id_from","station_id_to","matsim_plan_id"]].to_csv(f"{city_name}_demands_{n_zones}_stations.csv",index=False)

cap_scale = 1.0

od_count_df = Utils.get_OD_counts(link_df, activities_df)
# self-loop filter
ij_od_mask = od_count_df.zone_label_from != od_count_df.zone_label_to
od_count_df = od_count_df[ij_od_mask].copy()
od_count_df = od_count_df.reset_index(drop=True)
od_count_df["od_count"] = od_count_df.od_count.values / (24 * cap_scale) # scale to per hour

link_df = clustered_links_df.copy()

# from to links
from_to_links_per_node = Utils.get_from_and_to_links_per_node(link_df, exclude_pt=True)

# scale
cost_scale = 10000.0  # TODO DID NOT HELP!? icreasing OD makes it go up, but not down? What else?
link_df["enabling_cost"] = link_df.enabling_cost / cost_scale


cap_scale = 100.0
link_df["capacity"] = link_df.capacity.values / cap_scale # scale cap to per day
od_count_df["od_count"] = od_count_df.od_count.values / (24 * cap_scale)

# budget
#budget = 50000000 / cost_scale  # $100m
budget = 10000000 / cost_scale  # $100m
print("data setup---calling LP func")

min_cost = 1000/cost_scale
link_df.loc[link_df.enabling_cost<min_cost,"enabling_cost"] = min_cost

## CREATE MODEL
#flow_c_l_sol, beta_c_l_sol, enabled_l_sol, served_c_sol 
mcc_model, served_c, flow_c_l, beta_l = MaximumCustomerCoverage.get_maximum_customer_coverage_model(
    link_df,
    od_count_df,
    zones_df,
    from_to_links_per_node,
    budget,
)

if solver_type == "integer":
    from MaximumCustomerCoverage import solve_with_integer_budget
    # create integer model
    flow_sol, beta_sol, served_sol, enabled_sol = solve_with_integer_budget(
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
        gurobi_warmstart=False
    )

elif solver_type == "relaxed":
    from MaximumCustomerCoverage import solve_relaxed_budget
    # create integer model
    flow_sol, beta_sol, served_sol, enabled_sol = solve_relaxed_budget(
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
        gurobi_sol=False,
    )
elif solver_type == "relaxed_integer":
    from MaximumCustomerCoverage import solve_relaxed_integer_budget
    # create integer model
    flow_sol, beta_sol, served_sol, enabled_sol = solve_relaxed_integer_budget(
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

link_sol = link_df.copy()

n_links = len(link_df)
for i in range(0, len(od_count_df)):
    link_sol[f"paxFlow_{i}"] = cap_scale * flow_sol[i * n_links : (i + 1) * n_links]

link_sol["rebalFlow"] = cap_scale * beta_sol

if solver_type == "integer":
    link_sol["enabled"] = enabled_sol
else:
    enabled_sol = flow_sol.reshape((len(od_count_df), n_links)).sum(axis=0) + beta_sol
    link_sol["enabled"] = np.where(enabled_sol > 1e-6, 1.0, 0.0)
    link_sol["enabled_frac"] = enabled_sol

link_sol["enabling_cost"] = link_df_prior.enabling_cost.values

cost = (link_sol["enabled"] * link_sol.enabling_cost.values).sum()
print(f"Cost:{cost:,.2f}")
with open(f"{solutions_path}/resume_{city_name}_{solver_type}_zones{n_zones}.txt", 'w') as f:
    f.write(f"Cost: {cost}\n")

num_enabled_links = np.sum(link_sol["enabled"])
print(f"Number of enabled links: {num_enabled_links}")
with open(f"{solutions_path}/resume_{city_name}_{solver_type}_zones{n_zones}.txt", 'a') as f:
    f.write(f"Number of enabled links: {num_enabled_links}\n")
    
#num_enabled_links = np.sum((enabled_sol > 0) & (enabled_sol <= 1e-6))
#print(f"Number of weakly enabled links: {num_enabled_links}")

num_enabled_links = np.sum(enabled_sol >= 0.3)
print(f"Number of 0.3 enabled links: {num_enabled_links}")
with open(f"{solutions_path}/resume_{city_name}_{solver_type}_zones{n_zones}.txt", 'a') as f:
    f.write(f"Number of 0.3 enabled links: {num_enabled_links}\n")

num_enabled_links = np.sum(enabled_sol >= 0.5)
print(f"Number of 0.5 enabled links: {num_enabled_links}")
with open(f"{solutions_path}/resume_{city_name}_{solver_type}_zones{n_zones}.txt", 'a') as f:
    f.write(f"Number of 0.5 enabled links: {num_enabled_links}\n")

num_enabled_links = np.sum(enabled_sol >= 0.9)
print(f"Number of 0.9 enabled links: {num_enabled_links}")
with open(f"{solutions_path}/resume_{city_name}_{solver_type}_zones{n_zones}.txt", 'a') as f:
    f.write(f"Number of 0.9 enabled links: {num_enabled_links}\n")

num_enabled_links = np.sum(enabled_sol >= 0.99)
print(f"Number of 0.99 enabled links: {num_enabled_links}")
with open(f"{solutions_path}/resume_{city_name}_{solver_type}_zones{n_zones}.txt", 'a') as f:
    f.write(f"Number of 0.99 enabled links: {num_enabled_links}\n")

if solver_type == "integer" or solver_type == "relaxed":
    with open(f"{solutions_path}/resume_{city_name}_{solver_type}_zones{n_zones}.txt", 'a') as f:
        f.write(f"L1 penalty: {l1_penalty}\n")

print(f"Total enabled value sum: {np.sum(enabled_sol[enabled_sol > 1e-6]):,.2f}")


#export to kepler
Utils.export_to_kepler(f"{solutions_path}/kepler_solution_{city_name}_{solver_type}_{n_zones}_zones.csv", link_sol[link_sol["enabled"]==1], city_name)
Utils.export_to_kepler(f"{solutions_path}/kepler_solution_{city_name}_{solver_type}_{n_zones}_zones_0_3_enabled.csv", link_sol[(enabled_sol >= 1e-6) & (enabled_sol < 0.3)], city_name)
Utils.export_to_kepler(f"{solutions_path}/kepler_solution_{city_name}_{solver_type}_{n_zones}_zones_3_5_enabled.csv", link_sol[(enabled_sol >= 0.3) & (enabled_sol < 0.5)], city_name)
Utils.export_to_kepler(f"{solutions_path}/kepler_solution_{city_name}_{solver_type}_{n_zones}_zones_5_9_enabled.csv", link_sol[(enabled_sol >= 0.5) & (enabled_sol < 0.9)], city_name)