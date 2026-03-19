"""
Maximum Customer Coverage (MCC) optimization model.

Builds and solves a LP relaxation of the MILP for infrastructure-based sensor
placement on a road network. The objective is to maximize the number of served
trips subject to flow conservation, capacity, and budget constraints.

Uses an L1 penalty on the enabling variables to break LP degeneracy and push
solutions toward integrality (see thesis report, Chapter 4).
"""

import logging

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def solve_relaxed(
    model: gp.Model,
    served_c,       # gp.MVar shape (n_od,)
    flow_c_l,       # gp.MVar shape (n_od * n_links,)
    beta_l,         # gp.MVar shape (n_links,)
    n_links: int,
    n_od_flows: int,
    budget: float,
    link_df: pd.DataFrame,
    n_zones: int,
    l1_penalty: float = 0.0,
):
    """
    Solve the LP relaxation of the MCC model with an L1 penalty.

    Adds continuous enabling variables in [0, 1], linking and capacity
    constraints, budget constraint, and an L1 penalty term to the objective.
    Uses interior point method without crossover (Method=2, Crossover=0).

    Returns (flow_sol, beta_sol, served_sol, enabled_sol) as numpy arrays.
    """
    # Use interior point without crossover for LP relaxation
    model.setParam("Method", 2)
    model.setParam("Crossover", 0)

    # Add continuous enabling variables for each link
    enabled_l = model.addMVar(
        (n_links,), name="enabled_links", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0
    )

    # L1 penalty on enabling variables to encourage sparsity
    model.setObjective(model.getObjective() - l1_penalty * enabled_l.sum())
    model.update()
    logger.info("Added L1 penalty (%.1e) to objective on enabled links", l1_penalty)

    capacities = link_df.capacity.values
    link_offset_per_od = np.arange(0, n_od_flows) * n_links

    # Linking and capacity constraints for each link
    num_constraints = 0
    for l in tqdm(range(n_links), desc="Linking constraints"):
        total_flow = gp.quicksum(flow_c_l[l + link_offset_per_od]) + beta_l[l]

        # Lower bound: if there is flow, the link must be enabled
        model.addConstr(total_flow >= enabled_l[l], f"bigMLower_{l}")
        num_constraints += 1

        # Upper bound: flow on a link cannot exceed enabled fraction * capacity
        model.addConstr(total_flow <= enabled_l[l] * capacities[l], f"capFlow_{l}")
        num_constraints += 1

    logger.info("Added %d linking/capacity constraints", num_constraints)

    # Budget constraint: total enabling cost must not exceed budget
    model.addConstr(
        gp.quicksum(enabled_l * link_df.enabling_cost.values) <= budget,
        "budget_constraint",
    )
    logger.info("Budget constraint added")

    # Solve
    logger.info("Starting LP relaxation solve")
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        status_map = {
            GRB.INFEASIBLE: "Problem not feasible",
            GRB.UNBOUNDED: "Problem unbounded",
            GRB.UNDEFINED: "Problem undefined",
        }
        msg = status_map.get(model.Status, f"Solver status {model.Status}")
        raise RuntimeError(msg)

    model.write(f"mcc_model_solved_relaxed_zone{n_zones}_l1_{l1_penalty}.sol")

    flow_sol = flow_c_l.getAttr("X")
    beta_sol = beta_l.getAttr("X")
    served_sol = served_c.getAttr("X")
    enabled_sol = enabled_l.getAttr("X")

    return flow_sol, beta_sol, served_sol, enabled_sol


def get_maximum_customer_coverage_model(
    links_df: pd.DataFrame,
    od_count_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    from_to_links_per_node: pd.DataFrame,
    budget: float,
):
    """
    Build the base MCC model (without budget/enabling constraints).

    Creates variables:
      - served_c:  served trips per OD pair
      - flow_c_l:  passenger flow per OD pair per link
      - beta_l:    rebalancing (empty vehicle) flow per link

    Adds constraints:
      - Flow conservation for each OD pair at each node (Eq. 2.1.2)
      - Rebalancing flow conservation at each node (Eq. 2.1.3)
      - Demand upper bounds per OD pair (Eq. 2.1.4)
      - Arc capacity constraints (Eq. 2.1.6, without enabling variables)

    Budget and enabling constraints are added later by solve_relaxed().
    """
    env = gp.Env("gurobi.log")
    mcc_model = gp.Model("MCC", env=env)

    logger.debug("Zones:\n%s", zones_df.head())

    n_links = len(links_df)
    n_od_flows = len(od_count_df)
    logger.info("%d links and %d OD pairs", n_links, n_od_flows)

    # Decision variables
    served_c = mcc_model.addMVar((n_od_flows,), name="served", vtype=GRB.CONTINUOUS, lb=0.0)
    flow_c_l = mcc_model.addMVar((n_od_flows * n_links,), name="paxFlow", vtype=GRB.CONTINUOUS, lb=0.0)
    beta_l = mcc_model.addMVar((n_links,), name="rebalFlow", vtype=GRB.CONTINUOUS, lb=0.0)
    logger.info("Variables created: served_c(%d), flow_c_l(%d), beta_l(%d)",
                n_od_flows, n_od_flows * n_links, n_links)

    # Augment OD counts with nearest node IDs from zones
    od_count_df = pd.merge(
        od_count_df, zones_df[["label", "nearest_node_id"]],
        how="left", left_on="zone_label_from", right_on="label",
    )
    od_count_df = pd.merge(
        od_count_df, zones_df[["label", "nearest_node_id"]],
        how="left", left_on="zone_label_to", right_on="label",
        suffixes=["_from", "_to"],
    )

    from_to_links_per_node_dict = from_to_links_per_node.to_dict("records")
    od_dict = od_count_df.to_dict("records")

    # --- Passenger flow conservation constraints (Eq. 2.1.2) ---
    od_idx = 0
    num_constraints = 0
    for item in tqdm(od_dict, desc="Passenger flow conservation"):
        from_node_id = item["nearest_node_id_from"]
        to_node_id = item["nearest_node_id_to"]

        for tmp_item in tqdm(from_to_links_per_node_dict, leave=False):
            net_pax_inflow = 0
            if tmp_item["node_id"] == from_node_id:
                net_pax_inflow = served_c[od_idx]
            net_pax_outflow = 0
            if tmp_item["node_id"] == to_node_id:
                net_pax_outflow = served_c[od_idx]

            pax_from_link_idx = od_idx * n_links + np.array(tmp_item["from_idx"])
            pax_to_link_idx = od_idx * n_links + np.array(tmp_item["to_idx"])

            mcc_model.addConstr(
                gp.quicksum(flow_c_l[pax_from_link_idx]) + net_pax_outflow
                == gp.quicksum(flow_c_l[pax_to_link_idx]) + net_pax_inflow,
                f"flowConservation_pax_{od_idx}_{tmp_item['node_id']}",
            )
            num_constraints += 1

        # Demand upper bound (Eq. 2.1.4)
        mcc_model.addConstr(served_c[od_idx] <= item["od_count"], f"flow_limit[{od_idx}]")
        num_constraints += 1
        od_idx += 1

    logger.info("Added %d passenger flow constraints", num_constraints)

    # --- Rebalancing flow conservation constraints (Eq. 2.1.3) ---
    num_constraints = 0
    for tmp_item in tqdm(from_to_links_per_node_dict, desc="Rebalancing flow conservation"):
        net_pax_inflow_mask = od_count_df.nearest_node_id_from.values == tmp_item["node_id"]
        net_pax_inflow_sum = 0
        if np.sum(net_pax_inflow_mask):
            net_pax_inflow_sum = gp.quicksum(served_c[net_pax_inflow_mask])

        net_pax_outflow_mask = od_count_df.nearest_node_id_to.values == tmp_item["node_id"]
        net_pax_outflow_sum = 0
        if np.sum(net_pax_outflow_mask):
            net_pax_outflow_sum = gp.quicksum(served_c[net_pax_outflow_mask])

        mcc_model.addConstr(
            gp.quicksum(beta_l[tmp_item["to_idx"]]) + net_pax_outflow_sum
            == gp.quicksum(beta_l[tmp_item["from_idx"]]) + net_pax_inflow_sum,
            f"flowConservation_rebal_{tmp_item['node_id']}",
        )
        num_constraints += 1

    logger.info("Added %d rebalancing flow constraints", num_constraints)

    # --- Arc capacity constraints (Eq. 2.1.6, without enabling) ---
    capacities = links_df.capacity.values
    link_offset_per_od = np.arange(0, n_od_flows) * n_links
    num_constraints = 0

    for l in tqdm(range(n_links), desc="Capacity constraints"):
        mcc_model.addConstr(
            (gp.quicksum(flow_c_l[l + link_offset_per_od]) + beta_l[l]) <= capacities[l],
            f"capFlow_{l}",
        )
        num_constraints += 1

    logger.info("Added %d capacity constraints", num_constraints)

    # Objective: maximize total served trips
    mcc_model.ModelSense = GRB.MAXIMIZE
    mcc_model.setObjective(served_c.sum())
    mcc_model.write("mcc_model_new.lp")

    # Solver parameters
    mcc_model.Params.Heuristics = 0.1
    mcc_model.Params.Cuts = 2
    mcc_model.Params.Presolve = 2

    return mcc_model, served_c, flow_c_l, beta_l
