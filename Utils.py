"""
Utility functions for network processing, spatial operations, and data preparation.

Handles MATSim network I/O, coordinate transforms, graph operations (SCC, simplification),
trip origin/destination extraction, zone clustering, and Kepler.gl export.
"""

import logging

import pandas as pd
import numpy as np
import matsim
import utm
from shapely import Polygon, Point
from tqdm import tqdm

logger = logging.getLogger(__name__)


def export_intervention(intervention_df: pd.DataFrame, file: str):
    """Export an intervention (link enabling) DataFrame to CSV."""
    intervention_df[["link_id", "enabling"]].to_csv(file, index=False, sep=";", header=["LinkID", "Enabling"])


def get_link_df(network: matsim.Network.Network):
    """Build a link DataFrame by merging network links with from/to node coordinates."""
    links = network.links.copy()
    nodes = network.nodes.copy()

    link_df = pd.merge(links, nodes, how="left", left_on="from_node", right_on="node_id")
    link_df = pd.merge(link_df, nodes, how="left", left_on="to_node", right_on="node_id", suffixes=["_from", "_to"])

    return link_df


def add_lat_lon_to_nodes_df(nodes_df: pd.DataFrame, utm_zone=32, northern=True):
    """Add WGS84 lat/lon columns to a Berlin (EPSG:31468) nodes DataFrame."""
    from pyproj import Transformer

    logger.warning("CRS for Berlin v5.5 only!")
    coord_transformer = Transformer.from_crs("EPSG:31468", "EPSG:4326")

    lat, lon = coord_transformer.transform(nodes_df.y.values, nodes_df.x.values)
    nodes_df["lat"] = lat
    nodes_df["lon"] = lon

    return nodes_df


def add_lat_lon_to_links_df(link_df: pd.DataFrame, city_name, utm_zone=32, northern=True):
    """Add WGS84 lat/lon columns to a links DataFrame, for both from and to nodes."""
    from pyproj import Transformer

    if city_name == "Berlin":
        coord_transformer = Transformer.from_crs("EPSG:31468", "EPSG:4326")
    elif city_name == "singapore":
        coord_transformer = Transformer.from_crs("EPSG:3414", "EPSG:4326")
    else:
        raise RuntimeError(f"city_name {city_name} CRS not defined!")

    lat_from, lon_from = coord_transformer.transform(link_df.y_from.values, link_df.x_from.values)
    link_df["lat_from"] = lat_from
    link_df["lon_from"] = lon_from

    lat_to, lon_to = coord_transformer.transform(link_df.y_to.values, link_df.x_to.values)
    link_df["lat_to"] = lat_to
    link_df["lon_to"] = lon_to

    return link_df


def export_to_kepler(file: str, link_df: pd.DataFrame, city_name, utm_zone=32, northern=True, interventions_df=None):
    """Export a links DataFrame with lat/lon to CSV for Kepler.gl visualization."""
    link_df = add_lat_lon_to_links_df(link_df, city_name, utm_zone, northern)

    if interventions_df is not None:
        link_df = pd.merge(link_df, interventions_df, how="left", left_on="link_id", right_on="link_id")

    link_df.to_csv(file, index=False)


def get_trip_origins(activities_df: pd.DataFrame):
    """Extract trip origins from MATSim activities (all activities except the last per plan)."""
    origins = activities_df.copy()

    # Filter out freight and car/ride interaction activities
    mask = ~(origins.type.str.contains("interaction") | origins.type.str.contains("freight"))
    origins = origins[mask].copy()

    def drop_last(group):
        return group.iloc[:-1]

    origins = origins.groupby("plan_id").apply(drop_last).reset_index(drop=True)
    return origins


def get_trip_destins(activities_df: pd.DataFrame):
    """Extract trip destinations from MATSim activities (all activities except the first per plan)."""
    destins = activities_df.copy()

    # Filter out freight and car/ride interaction activities
    mask = ~(destins.type.str.contains("interaction") | destins.type.str.contains("freight"))
    destins = destins[mask].copy()

    def drop_first(group):
        return group.iloc[1:]

    destins = destins.groupby("plan_id").apply(drop_first).reset_index(drop=True)
    return destins


def k_means_zones(activities_df: pd.DataFrame, zone_count):
    """Create station zones by k-means clustering of trip origin locations."""
    origins_df = get_trip_origins(activities_df)
    origins_df["x"] = origins_df.x.astype(float)
    origins_df["y"] = origins_df.y.astype(float)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=zone_count, random_state=0, n_init="auto").fit(origins_df[["x", "y"]].values)
    centers = kmeans.cluster_centers_
    return pd.DataFrame({"label": np.arange(0, len(centers)), "x": centers[:, 0], "y": centers[:, 1]})


def add_nodeId_to_zones(zones_df: pd.DataFrame, nodes_df: pd.DataFrame):
    """Map each zone center to the nearest non-PT network node."""
    from sklearn.neighbors import KDTree

    non_pt_mask = ~nodes_df.node_id.str.contains("pt")
    nodes_df = nodes_df[non_pt_mask]

    kdt = KDTree(nodes_df[["x", "y"]].values, leaf_size=30, metric="euclidean")
    nearest_node = kdt.query(zones_df[["x", "y"]].values, k=1, return_distance=False)
    zones_df["nearest_node_id"] = nodes_df.iloc[nearest_node.flatten()].node_id.values

    return zones_df


def cluster_links(zones_df: pd.DataFrame, link_df: pd.DataFrame):
    """Assign each link to the nearest zone based on its from-node coordinates."""
    from sklearn.neighbors import KDTree

    kdt = KDTree(zones_df[["x", "y"]].values, leaf_size=30, metric="euclidean")
    link_labels = kdt.query(link_df[["x_from", "y_from"]].values, k=1, return_distance=False)
    link_df["zone_label"] = link_labels

    return link_df


def get_clustered_activities(clustered_links_df: pd.DataFrame, activities_df: pd.DataFrame):
    """Join trip origins and destinations with their zone labels via link assignment."""
    origins_df = get_trip_origins(activities_df)
    origins_df = pd.merge(
        origins_df, clustered_links_df[["link_id", "zone_label"]], how="left", left_on="link", right_on="link_id"
    )
    destins_df = get_trip_destins(activities_df)
    destins_df = pd.merge(
        destins_df, clustered_links_df[["link_id", "zone_label"]], how="left", left_on="link", right_on="link_id"
    )

    od_df = pd.merge(origins_df, destins_df, left_index=True, right_index=True, suffixes=["_from", "_to"])

    return od_df


def get_OD_counts(clustered_links_df: pd.DataFrame, activities_df: pd.DataFrame):
    """Compute origin-destination trip counts grouped by zone pairs."""
    od_df = get_clustered_activities(clustered_links_df, activities_df)
    od_count = od_df.groupby(["zone_label_from", "zone_label_to"]).size().reset_index(name="od_count")

    return od_count


def get_from_and_to_links_per_node(links_df: pd.DataFrame, exclude_pt=True):
    """Build a DataFrame mapping each node to its outgoing and incoming link IDs and indices."""
    links_df = links_df.copy()

    if exclude_pt:
        links_df = links_df[~links_df.modes.str.contains("pt")]

    # Outgoing links per node
    from_df = links_df.groupby("from_node").apply(lambda grp: list(grp.link_id.values))
    from_df = from_df.to_frame().reset_index()
    from_df["from_idx"] = links_df.groupby("from_node").apply(lambda grp: list(grp.index.values)).values
    from_df.rename(columns={"from_node": "node_id", 0: "from_links"}, inplace=True)

    # Incoming links per node
    to_df = links_df.groupby("to_node").apply(lambda grp: list(grp.link_id.values))
    to_df = to_df.to_frame().reset_index()
    to_df["to_idx"] = links_df.groupby("to_node").apply(lambda grp: list(grp.index.values)).values
    to_df.rename(columns={"to_node": "node_id", 0: "to_links"}, inplace=True)

    from_nodes = set(from_df.node_id)
    to_nodes = set(to_df.node_id)

    diff1 = from_nodes.difference(to_nodes)
    diff2 = to_nodes.difference(from_nodes)
    logger.info("From-only nodes: %d, To-only nodes: %d", len(diff1), len(diff2))

    if diff1 or diff2:
        logger.warning("From and to node sets differ. Merge may lose nodes.")

    res = pd.merge(from_df, to_df, how="left", left_on="node_id", right_on="node_id")

    return res


def filter_network_links_df(link_df: pd.DataFrame, polygon: Polygon):
    """Return a boolean mask of links fully contained within the given polygon."""
    link_df = add_lat_lon_to_links_df(link_df)

    include_link = []
    for row in link_df.itertuples():
        fromPoint = Point(row.lon_from, row.lat_from)
        toPoint = Point(row.lon_to, row.lat_to)

        if polygon.contains(fromPoint) and polygon.contains(toPoint):
            include_link.append(True)
            continue

        include_link.append(False)

    return include_link


def filter_network_nodes_df(nodes_df: pd.DataFrame, polygon: Polygon):
    """Return a boolean mask of nodes contained within the given polygon."""
    nodes_df = add_lat_lon_to_nodes_df(nodes_df)

    include_node = []
    for row in nodes_df.itertuples():
        pt = Point(row.lon, row.lat)

        if polygon.contains(pt):
            include_node.append(True)
            continue

        include_node.append(False)

    return include_node


def get_largest_scc(link_df: pd.DataFrame, nodes_df: pd.DataFrame):
    """Filter the network to its largest strongly connected component."""
    import networkx as nx

    G = nx.from_pandas_edgelist(link_df, "from_node", "to_node", edge_attr="link_id", create_using=nx.DiGraph())

    scc = nx.strongly_connected_components(G)
    largest_scc = max(scc, key=len)
    largest_scc_subgraph = G.subgraph(largest_scc)

    scc_linkIds = [largest_scc_subgraph.edges[u, v]["link_id"] for u, v in largest_scc_subgraph.edges()]
    link_df = link_df[link_df.link_id.isin(scc_linkIds)].copy()

    scc_nodeIds = set(link_df.from_node).union(set(link_df.to_node))
    nodes_df = nodes_df[nodes_df.node_id.isin(scc_nodeIds)].copy()

    logger.info("SCC Edges: %s. SCC Nodes: %s", f"{len(link_df):,d}", f"{len(nodes_df):,d}")

    return link_df, nodes_df


def get_in_and_out_degree_per_node_df(nodes_df, link_df):
    """Compute in-degree and out-degree for each node."""
    res = pd.DataFrame({"node_id": nodes_df.node_id.unique()})

    tmp_in = link_df.groupby("to_node").from_node.nunique().reset_index(name="in_deg")
    tmp_in.rename(columns={"to_node": "node_id"}, inplace=True)
    res = pd.merge(res, tmp_in, how="left", left_on="node_id", right_on="node_id")

    tmp_out = link_df.groupby("from_node").to_node.nunique().reset_index(name="out_deg")
    tmp_out.rename(columns={"from_node": "node_id"}, inplace=True)
    res = pd.merge(res, tmp_out, how="left", left_on="node_id", right_on="node_id")

    res.fillna(0, inplace=True)

    return res


def simplify_network(link_df, node_df):
    """
    Contract degree-1 nodes (in-degree == out-degree == 1) out of the network.

    Merges consecutive links through pass-through nodes into single links,
    summing lengths and costs and taking the minimum of capacities and speeds.
    """
    in_out_df = get_in_and_out_degree_per_node_df(node_df, link_df)
    in_out_1_nodes = in_out_df[(in_out_df.in_deg == 1) & (in_out_df.out_deg == 1)].node_id.values

    new_node_df = node_df[~node_df.node_id.isin(in_out_1_nodes)].copy()

    new_link_df = link_df.copy()
    new_link_df["remove"] = False
    for node in tqdm(in_out_1_nodes, desc="Contracting degree-1 nodes"):
        in_link_mask = new_link_df.to_node == node
        out_link_mask = new_link_df.from_node == node

        new_link_df.loc[in_link_mask, "remove"] = True
        new_link_df.loc[out_link_mask, "remove"] = True

        # Take the last matching row (handles chained contractions)
        in_link = new_link_df[in_link_mask].iloc[-1]
        out_link = new_link_df[out_link_mask].iloc[-1]

        # Create merged link
        new_link_df.loc[len(new_link_df)] = {
            "length": in_link.length + out_link.length,
            "freespeed": min(in_link.freespeed, out_link.freespeed),
            "capacity": min(in_link.capacity, out_link.capacity),
            "permlanes": min(in_link.permlanes, out_link.permlanes),
            "oneway": in_link.oneway,
            "modes": in_link.modes + in_link.modes,
            "link_id": f"{in_link.link_id}_X_{out_link.link_id}",
            "from_node": in_link.from_node,
            "to_node": out_link.to_node,
            "x_from": in_link.x_from,
            "y_from": in_link.y_from,
            "node_id_from": in_link.node_id_from,
            "x_to": out_link.x_to,
            "y_to": out_link.y_to,
            "node_id_to": out_link.node_id_to,
            "enabling_cost": in_link.enabling_cost + out_link.enabling_cost,
            "lat_from": in_link.lat_from,
            "lon_from": in_link.lon_from,
            "lat_to": out_link.lat_to,
            "lon_to": out_link.lon_to,
            "remove": False,
        }

    new_link_df = new_link_df[new_link_df.remove == False].copy()
    new_link_df.drop(columns=["remove"], inplace=True)

    return new_link_df, new_node_df
