import gurobipy as gp
from gurobipy import GRB
from gurobipy import nlfunc
import pandas as pd, numpy as np
from Utils import *
from tqdm import tqdm
import os

def solve_with_integer_budget(
    model: gp.Model,
    served_c,        # gp.MVar shape (n_od,)
    flow_c_l,        # gp.MVar shape (n_od * n_links,)
    beta_l,          # gp.MVar shape (n_links,)
    n_links,
    n_od_flows,
    budget: float,
    link_df,
    n_zones: int,
    l1_penalty: float = 0.0,
    gurobi_warmstart=False,
):
    
    if gurobi_warmstart:
        model.setParam('Method', 1)
        model.update()

    else:
        enabled_l = model.addMVar((n_links,), name="enabled_links", vtype=GRB.BINARY)
        model.setObjective(model.getObjective() - l1_penalty * enabled_l.sum())
        print("Added L1 penalty to objective on enabled links",flush=True)
        model.update()

        capacities = link_df.capacity.values  # values per hour, consider adjusting to consistent units if not done
        eps = 1e-4  # no strict inequalities, add small tolerance
        link_offset_per_od = np.arange(0, n_od_flows) * n_links

        
        num_constraints = 0
        for l in tqdm(range(0, n_links)):
            # Redundat because could be dealt with in postprocessing
            model.addConstr(
                (gp.quicksum(flow_c_l[l + link_offset_per_od]) + beta_l[l]) >= enabled_l[l], #eps - (1 - enabled_l[l]) * capacities[l],
                f"bigMLower_{l}",
            ) 
            num_constraints += 1
        
            # Redundant constraints to link enabling variables to flow variables
            model.addConstr(
                (gp.quicksum(flow_c_l[l + link_offset_per_od]) + beta_l[l]) <= enabled_l[l] * capacities[l],
                f"capFlow_{l}",
            )
            num_constraints += 1

        print(f"added {num_constraints} pax flow constraints")
        print("pax flow constraints done", flush=True)

        # Budget constraint
        model.addConstr(
            gp.quicksum(enabled_l * link_df.enabling_cost.values) <= budget,
            "budget_constraint",
        )

        print("budget constraints done", flush=True)

    print("Starting To Solve Integer MCC Model:", flush=True)
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        if model.Status == GRB.INFEASIBLE:
            raise RuntimeError("Problem not feasible!")
        if model.Status == GRB.UNBOUNDED:
            raise RuntimeError("Problem unbounded!")
        if model.Status == GRB.UNDEFINED:
            raise RuntimeError("Problem undefined!")
        raise RuntimeError(f"LP subproblem solver status {model.Status}")
    
    model.write(f"mcc_model_solved_integer_zone{n_zones}.sol")

    flow_c_l_sol = flow_c_l.getAttr("X")
    beta_c_l_sol = beta_l.getAttr("X")
    served_c_sol = served_c.getAttr("X")
    enabled_c_l_sol = enabled_l.getAttr("X")

    return flow_c_l_sol, beta_c_l_sol, served_c_sol, enabled_c_l_sol


def solve_relaxed_budget(
    model: gp.Model,
    served_c,        # gp.MVar shape (n_od,)
    flow_c_l,        # gp.MVar shape (n_od * n_links,)
    beta_l,          # gp.MVar shape (n_links,)
    n_links,
    n_od_flows,
    budget: float,
    link_df,
    n_zones: int,
    l1_penalty: float = 0.0,
    gurobi_sol=False,
):
    
    model.setParam('Method', 2)
    model.setParam('Crossover', 0)
    enabled_l = model.addMVar((n_links,), name="enabled_links", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
    model.setObjective(model.getObjective() - l1_penalty * enabled_l.sum())
    model.update()
    print("Added L1 penalty to objective on enabled links",flush=True)

    capacities = link_df.capacity.values  # values per hour, consider adjusting to consistent units if not done
    eps = 1e-4  # no strict inequalities, add small tolerance
    link_offset_per_od = np.arange(0, n_od_flows) * n_links

    
    num_constraints = 0
    for l in tqdm(range(0, n_links)):
        # Redundat because could be dealt with in postprocessing
        model.addConstr(
            (gp.quicksum(flow_c_l[l + link_offset_per_od]) + beta_l[l]) >= enabled_l[l], #eps - (1 - enabled_l[l]) * capacities[l],
            f"bigMLower_{l}",
        ) 
        num_constraints += 1
    
        # Redundant constraints to link enabling variables to flow variables
        model.addConstr(
            (gp.quicksum(flow_c_l[l + link_offset_per_od]) + beta_l[l]) <= enabled_l[l] * capacities[l],
            f"capFlow_{l}",
        )
        num_constraints += 1

    print(f"added {num_constraints} pax flow constraints")
    print("pax flow constraints done", flush=True)

    # Budget constraint
    model.addConstr(
        gp.quicksum(enabled_l * link_df.enabling_cost.values) <= budget,
        "budget_constraint",
    )

    print("budget constraints done", flush=True)

    print("Starting To Solve relaxed MCC Model:", flush=True)
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        if model.Status == GRB.INFEASIBLE:
            raise RuntimeError("Problem not feasible!")
        if model.Status == GRB.UNBOUNDED:
            raise RuntimeError("Problem unbounded!")
        if model.Status == GRB.UNDEFINED:
            raise RuntimeError("Problem undefined!")
        raise RuntimeError(f"LP subproblem solver status {model.Status}")

    if gurobi_sol:
        return flow_c_l, beta_l, served_c, enabled_l
    
    model.write(f"mcc_model_solved_relaxed_zone{n_zones}_l1_{l1_penalty}.sol")

    flow_c_l_sol = flow_c_l.getAttr("X")
    beta_c_l_sol = beta_l.getAttr("X")
    served_c_sol = served_c.getAttr("X")
    enabled_c_l_sol = enabled_l.getAttr("X")

    return flow_c_l_sol, beta_c_l_sol, served_c_sol, enabled_c_l_sol

def solve_relaxed_integer_budget(
    model: gp.Model,
    served_c,        # gp.MVar shape (n_od,)
    flow_c_l,        # gp.MVar shape (n_od * n_links,)
    beta_l,          # gp.MVar shape (n_links,)
    n_links,
    n_od_flows,
    budget: float,
    link_df,
    n_zones: int,
    l1_penalty: float = 0.0,
):
    
    print("Launch relaxed solver")
    flow_c_l, beta_l, served_c, enabled_l = solve_relaxed_budget(
        model=model,
        served_c=served_c,
        flow_c_l=flow_c_l,
        beta_l=beta_l,
        n_links=n_links,
        n_od_flows=n_od_flows,
        budget=budget,
        link_df=link_df,
        n_zones=n_zones,
        l1_penalty=l1_penalty,
        gurobi_sol=True,
    )

    print("Custom crossover starts...")

    indexes_to_zero = np.where(enabled_l.getAttr("X") <= 1e-6)[0]
    indexes_to_one = np.where(enabled_l.getAttr("X") > 1e-6)[0]

    f = flow_c_l.getAttr("X")

    for i in range(n_od_flows):
        for j in indexes_to_zero:
            f[i*n_od_flows + j] = 0.0

    flow_c_l.setAttr("Start", f)
    beta_l[indexes_to_zero].setAttr("Start", np.zeros_like(indexes_to_zero)) # = 0.0

    enabled_l[indexes_to_one].setAttr("Start", np.ones_like(indexes_to_one))# = 1
    enabled_l[indexes_to_zero].setAttr("Start", np.zeros_like(indexes_to_zero))# = 0

    enabled_l.vtype = GRB.BINARY

    model.update()

    print("Custom crossover done.", flush=True)

    print("Launch integer solver")

    flow_sol, beta_sol, served_sol, enabled_sol = solve_with_integer_budget(
        model=model,
        served_c=served_c,
        flow_c_l=flow_c_l,
        beta_l=beta_l,
        n_links=n_links,
        n_od_flows=n_od_flows,
        budget=budget,
        link_df=link_df,
        n_zones=n_zones,
        l1_penalty=l1_penalty,
        gurobi_warmstart=True,
    )

    return flow_sol, beta_sol, served_sol, enabled_sol

def get_maximum_customer_coverage_model(
    links_df: pd.DataFrame,
    od_count_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    from_to_links_per_node: pd.DataFrame,
    budget: float,
):
    #options = {
    #    "WLSACCESSID": os.getenv("GRB_WLSACCESSID"),
    #    "WLSSECRET": os.getenv("GRB_WLSSECRET"),
    #    "LICENSEID": 2731669,#int(os.getenv("GRB_LICENSEID")),
    #}
    env = gp.Env("gurobi.log")#, params=options)
    mcc_model = gp.Model("MCC", env=env)
    
    print("zones: ")
    print(zones_df.head())

    # Create Variables
    n_links = len(links_df)
    n_od_flows = len(od_count_df)

    # flow of commodity (i.e. i,j pair) on link (l)
    print(f"{n_links} links and {n_od_flows} od pairs")
    served_c = mcc_model.addMVar((n_od_flows,), name="served", vtype=GRB.CONTINUOUS, lb=0.0)
    print("served_c done", flush=True)
    flow_c_l = mcc_model.addMVar((n_od_flows * n_links,), name="paxFlow", vtype=GRB.CONTINUOUS, lb=0.0)
    print("flow_c_l done", flush=True)
    beta_l = mcc_model.addMVar((n_links,), name="rebalFlow", vtype=GRB.CONTINUOUS, lb=0.0)
    print("beta_l done", flush=True)

    print("variables done", flush=True)

    # augment od_count_df with nearest node ids
    od_count_df = pd.merge(
        od_count_df, zones_df[["label", "nearest_node_id"]], how="left", left_on="zone_label_from", right_on="label"
    )
    od_count_df = pd.merge(
        od_count_df,
        zones_df[["label", "nearest_node_id"]],
        how="left",
        left_on="zone_label_to",
        right_on="label",
        suffixes=["_from", "_to"],
    )

    # passenger flow constraints (5a)
    from_to_links_per_node_dict = from_to_links_per_node.to_dict("records")
    od_dict = od_count_df.to_dict("records")
    od_idx = 0
    num_constraints = 0
    for item in tqdm(od_dict):

        from_node_id = item["nearest_node_id_from"]
        to_node_id = item["nearest_node_id_to"]
        for tmp_item in tqdm(from_to_links_per_node_dict):
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

        # flow limit (5e)
        mcc_model.addConstr(served_c[od_idx] <= item["od_count"], f"flow_limit[{od_idx}]")
        num_constraints += 1

        od_idx += 1

    print(f"added {num_constraints} pax flow constraints")
    print("pax flow constraints done", flush=True)
    num_constraints = 0

    # REBALANCING FLOWS (5c)
    for tmp_item in tqdm(from_to_links_per_node_dict):

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

    print(f"added {num_constraints} rebalancing flow constraints")
    print("rebal flow constraints done", flush=True)
    num_constraints = 0

    # (5d), (5f), (5g) -- NOTE: removed Big-M constraints
    capacities = links_df.capacity.values  # values per hour
    eps = 1e-4
    link_offset_per_od = np.arange(0, n_od_flows) * n_links

    for l in tqdm(range(0, n_links)):
        mcc_model.addConstr(
            (gp.quicksum(flow_c_l[l + link_offset_per_od]) + beta_l[l]) <= capacities[l],
            f"capFlow_{l}",
        )
        num_constraints += 1
    print(f"added {num_constraints} arc cap and enabled street constraints")
    print("arc cap & enabled streets, constraints done", flush=True)
    # 5e is higher up.

    # budget constraint replaced by nonlinear function; model built without it here.
    print("budget constraint will be handled by the solver type", flush=True)

    mcc_model.ModelSense = GRB.MAXIMIZE
    mcc_model.setObjective(served_c.sum())
    mcc_model.write("mcc_model_new.lp")

    # set some solver params (tune as needed)
    mcc_model.Params.Heuristics = 0.1
    mcc_model.Params.Cuts = 2
    mcc_model.Params.Presolve = 2

    return mcc_model, served_c, flow_c_l, beta_l
"""
    # Run CCP-style linearization loop to handle nonlinear budget
    flow_sol, beta_sol, served_sol = solve_with_linearized_budget_fast(
        model=mcc_model,
        served_c=served_c,
        flow_c_l=flow_c_l,
        beta_l=beta_l,
        links_df=links_df,
        n_links=n_links,
        n_od_flows=n_od_flows,
        budget=budget,
        max_outer_iter=100,
        tol_x=1e-6,
        trust_delta=0.2,
        slack_penalty=1e6,
    )

    # write final model and extract variable values
    mcc_model.write("mcc_model_solved.lp")
    flow_c_l_sol = flow_c_l.getAttr("X")
    beta_c_l_sol = beta_l.getAttr("X")
    served_c_sol = served_c.getAttr("X") """

    #return flow_c_l_sol, beta_c_l_sol, served_c_sol