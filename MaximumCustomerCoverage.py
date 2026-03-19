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


def solve_min_cost(
    model: gp.Model,
    served_c,       # gp.MVar shape (n_od,)
    flow_c_l,       # gp.MVar shape (n_od * n_links,)
    beta_l,         # gp.MVar shape (n_links,)
    n_links: int,
    n_od_flows: int,
    link_df: pd.DataFrame,
    n_zones: int,
    min_demand: float,
    warm_start_flow: np.ndarray,
    warm_start_beta: np.ndarray,
    warm_start_served: np.ndarray,
):
    """
    Solve the cost-minimization MILP on a reduced (enabled-only) network.

    This is the second-phase problem: given that the LP relaxation identified
    which links to enable, find the minimum-cost binary enabling that still
    serves at least `min_demand` total trips.

    Differences from the LP relaxation (solve_relaxed):
      - Enabling variables are binary (0/1), not continuous.
      - Objective minimizes total enabling cost (not maximizes served trips).
      - A demand floor constraint replaces the budget constraint:
        sum(served_c) >= min_demand.
      - The LP relaxation solution is used as a warm start (incumbent).

    Returns (flow_sol, beta_sol, served_sol, enabled_sol) as numpy arrays.
    """
    # Add binary enabling variables for each link
    enabled_l = model.addMVar(
        (n_links,), name="enabled_links", vtype=GRB.BINARY
    )

    # Objective: minimize total enabling cost
    model.ModelSense = GRB.MINIMIZE
    model.setObjective(gp.quicksum(enabled_l * link_df.enabling_cost.values))
    model.update()
    logger.info("Objective set to minimize total enabling cost")

    capacities = link_df.capacity.values
    link_offset_per_od = np.arange(0, n_od_flows) * n_links

    # Linking and capacity constraints for each link
    num_constraints = 0
    for l in tqdm(range(n_links), desc="Linking constraints (min-cost)"):
        total_flow = gp.quicksum(flow_c_l[l + link_offset_per_od]) + beta_l[l]

        # Lower bound: if there is flow, the link must be enabled
        model.addConstr(total_flow >= enabled_l[l], f"bigMLower_{l}")
        num_constraints += 1

        # Upper bound: flow cannot exceed enabled * capacity
        model.addConstr(total_flow <= enabled_l[l] * capacities[l], f"capFlow_{l}")
        num_constraints += 1

    logger.info("Added %d linking/capacity constraints", num_constraints)

    # Demand floor: total served trips must be at least min_demand
    model.addConstr(served_c.sum() >= min_demand, "demand_floor")
    logger.info("Demand floor constraint added: served >= %.4f", min_demand)

    # Warm start from LP relaxation solution
    flow_c_l.setAttr("Start", warm_start_flow)
    beta_l.setAttr("Start", warm_start_beta)
    served_c.setAttr("Start", warm_start_served)
    enabled_l.setAttr("Start", np.ones(n_links))  # all links enabled in reduced network
    model.update()
    logger.info("Warm start set from LP relaxation solution")

    # Solve
    logger.info("Starting min-cost MILP solve")
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        status_map = {
            GRB.INFEASIBLE: "Problem not feasible",
            GRB.UNBOUNDED: "Problem unbounded",
            GRB.UNDEFINED: "Problem undefined",
        }
        msg = status_map.get(model.Status, f"Solver status {model.Status}")
        raise RuntimeError(msg)

    model.write(f"mcc_model_solved_mincost_zone{n_zones}.sol")

    flow_sol = flow_c_l.getAttr("X")
    beta_sol = beta_l.getAttr("X")
    served_sol = served_c.getAttr("X")
    enabled_sol = enabled_l.getAttr("X")

    return flow_sol, beta_sol, served_sol, enabled_sol


def greedy_prune(
    link_df: pd.DataFrame,
    od_count_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    from_to_links_per_node: pd.DataFrame,
    flow_sol: np.ndarray,
    beta_sol: np.ndarray,
    served_sol: np.ndarray,
    n_zones: int,
):
    """
    Greedily disable links to reduce total enabling cost.

    Starting from the enabled subnetwork and its flow solution, the algorithm
    iteratively:
      1. Sorts enabled links by total flow (ascending, cheapest to reroute first).
      2. Tentatively disables the link with the smallest flow by setting its
         capacity constraint RHS to 0.
      3. Re-solves the LP (maximize served trips with a demand floor) to check
         if flows can be rerouted through remaining links.
      4. If feasible (demand floor met): keeps the link disabled and uses the
         new flow solution for the next iteration.
      5. If infeasible: restores the link and tries the next candidate.
    The algorithm stops when a full pass over all remaining links produces no
    further pruning.

    Uses dual simplex (Method=1) for efficient warm-started re-solves, since
    each iteration only changes one constraint RHS.

    Returns (enabled_mask, flow_sol, beta_sol, served_sol) where enabled_mask
    is a boolean array over the input link_df indices.
    """
    n_links = len(link_df)
    n_od = len(od_count_df)
    min_demand = served_sol.sum()

    logger.info("Greedy pruning: %d links, demand floor %.4f", n_links, min_demand)

    # Build base model on the enabled subnetwork
    model, served_c, flow_c_l, beta_l = get_maximum_customer_coverage_model(
        link_df, od_count_df, zones_df, from_to_links_per_node, 0.0,
    )

    # Add demand floor constraint
    model.addConstr(served_c.sum() >= min_demand, "demand_floor")

    # Use dual simplex for efficient re-solves with warm start
    model.setParam("Method", 1)
    # Suppress per-solve log output (we log progress ourselves)
    model.setParam("OutputFlag", 0)
    model.update()

    # Solve once to establish a feasible basis
    model.optimize()
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        logger.warning("Initial greedy solve infeasible (status %d), returning input as-is", model.Status)
        return np.ones(n_links, dtype=bool), flow_sol, beta_sol, served_sol

    # Initialize from the solved model
    current_flow = flow_c_l.getAttr("X")
    current_beta = beta_l.getAttr("X")
    current_served = served_c.getAttr("X")
    capacities = link_df.capacity.values

    enabled = np.ones(n_links, dtype=bool)

    iteration = 0
    while True:
        iteration += 1

        # Compute total flow per enabled link
        total_flow = np.zeros(n_links)
        for l in np.where(enabled)[0]:
            pax_flow = sum(current_flow[i * n_links + l] for i in range(n_od))
            total_flow[l] = pax_flow + current_beta[l]

        # Sort enabled links by total flow ascending (try removing low-flow links first)
        candidates = np.where(enabled)[0]
        candidates = candidates[np.argsort(total_flow[candidates])]

        pruned_any = False
        for l in candidates:
            # Tentatively disable link by setting capacity to 0
            constr = model.getConstrByName(f"capFlow_{l}")
            original_rhs = constr.RHS
            constr.RHS = 0.0
            model.update()

            model.optimize()

            if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                new_served_total = served_c.getAttr("X").sum()
                if new_served_total >= min_demand - 1e-6:
                    # Pruning successful: keep link disabled, update flow solution
                    enabled[l] = False
                    current_flow = flow_c_l.getAttr("X")
                    current_beta = beta_l.getAttr("X")
                    current_served = served_c.getAttr("X")
                    remaining = int(enabled.sum())
                    cost = (enabled * link_df.enabling_cost.values).sum()
                    logger.info(
                        "Iteration %d: pruned link %d (flow=%.4f), "
                        "%d links remaining, cost=%.2f",
                        iteration, l, total_flow[l], remaining, cost,
                    )
                    pruned_any = True
                    # Restart with recomputed flows
                    break
                else:
                    # Demand not met, restore link
                    constr.RHS = original_rhs
                    model.update()
            else:
                # Infeasible, restore link
                constr.RHS = original_rhs
                model.update()

        if not pruned_any:
            logger.info(
                "Greedy pruning complete after %d iterations: %d links remaining (pruned %d)",
                iteration, int(enabled.sum()), n_links - int(enabled.sum()),
            )
            break

    return enabled, current_flow, current_beta, current_served


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
