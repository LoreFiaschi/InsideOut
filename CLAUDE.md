# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InsideOut optimizes the placement of sensor infrastructure on city streets for autonomous driving. Instead of equipping each vehicle with sensors ("inside-out" approach), sensors are placed on road infrastructure and shared. The system determines which streets should receive infrastructure to maximize served passenger trips at minimum cost.

The pipeline runs a two-phase optimization on a city road network (currently Singapore, ~38k nodes, ~78k arcs). Graph pruning via "skeletons" can optionally reduce the network size.

Reference: `agentic/docs/report.pdf` — semester thesis describing the math and algorithms in detail.

## Running

```bash
python main.py
```

Configuration is done by editing variables at the top of `main.py`:
- `city_name`: `"singapore"` or `"Berlin"`
- `n_zones`: number of stations (e.g., 2, 5, 10, 40)
- `l1_penalty`: L1 regularization strength for the Phase 1 LP relaxation
- `phase2_method`: `"milp"` (exact cost minimization) or `"greedy"` (iterative link pruning)
- `simplified_network_available` / `skeletons_available`: toggle pre-processed network input

Logs are written to `insideout_{city}_{n_zones}_zones.log`.

### Dependencies

- **Gurobi** (with valid license) — the LP/MILP solver
- `gurobipy`, `pandas`, `numpy`, `matsim`, `geopandas`, `shapely`, `scipy`, `scikit-learn`, `pyproj`, `tqdm`, `networkx`, `utm`

### External Data

Data files (gitignored) live in `data/` and an external path (`path_prefix` in `main.py`, currently pointing to `proj-mt-schneider`):
- MATSim network XML files (`.xml.gz`)
- MATSim population/plans files
- Cost-per-link CSVs
- Pre-simplified network CSVs (`simplified_link_df_singapore.csv`, `simplified_node_df_singapore.csv`)

## Architecture

Three Python files, no package structure:

### `main.py` — Pipeline orchestrator
1. Loads network and MATSim plans data
2. Filters to largest strongly connected component (SCC)
3. Clusters trip origins into `n_zones` stations via k-means
4. Builds OD (origin-destination) demand matrix
5. Builds the base Gurobi model via `get_maximum_customer_coverage_model()`
6. **Phase 1**: Solves via `solve_relaxed()` — LP relaxation with L1 penalty, maximizes served trips under budget constraint
7. **Phase 2** (dispatched by `phase2_method`):
   - `"milp"`: Rebuilds the model on the enabled subnetwork, solves via `solve_min_cost()` — binary enabling variables, minimizes cost subject to demand floor, warm-started from Phase 1
   - `"greedy"`: Calls `greedy_prune()` — iteratively disables links by ascending flow, re-solving the LP each step to verify feasibility
8. Exports both solutions to CSV for Kepler.gl visualization

### `MaximumCustomerCoverage.py` — Gurobi optimization model
- `get_maximum_customer_coverage_model()` builds the base model without budget/enabling constraints: flow conservation per OD pair per node (Eq. 2.1.2), rebalancing flow conservation (Eq. 2.1.3), demand upper bounds (Eq. 2.1.4), capacity constraints (Eq. 2.1.6). Returns the model plus variable handles (`served_c`, `flow_c_l`, `beta_l`). Called once per phase (on different networks).
- `solve_relaxed()` (Phase 1) adds continuous `enabled_l` in [0,1], linking/capacity constraints, budget constraint, and L1 penalty. Objective: maximize served trips. Uses interior point (Method=2, Crossover=0).
- `solve_min_cost()` (Phase 2, MILP) adds binary `enabled_l`, linking/capacity constraints, and demand floor constraint (`sum(served_c) >= min_demand`). Objective: minimize total enabling cost. Sets LP solution as warm start incumbent.
- `greedy_prune()` (Phase 2, greedy) builds a model on the enabled subnetwork with a demand floor, then iteratively disables links (smallest flow first) by setting their capacity constraint RHS to 0. Uses dual simplex (Method=1) for efficient warm-started re-solves. Stops when no more links can be pruned without violating the demand floor.

### `Utils.py` — Data processing
Network loading, coordinate transforms (EPSG:3414 ↔ WGS84 for Singapore, EPSG:31468 ↔ WGS84 for Berlin), SCC extraction via NetworkX, degree-1 node contraction, trip origin/destination extraction from MATSim plans, k-means zone creation, OD count aggregation, Kepler.gl CSV export.

## Key Concepts

- **Two-phase solve**: Phase 1 (LP relaxation) quickly identifies a good set of enabled links; Phase 2 reduces cost on that subnetwork. The Phase 1 solution provides the demand floor and (for MILP) a warm start.
- **Phase 2 methods**: `"milp"` gives an exact minimum-cost solution but can be slow on large networks. `"greedy"` is faster (each iteration is one LP re-solve via warm-started dual simplex) but produces a heuristic solution. Both guarantee the demand floor is met.
- **Greedy pruning**: sorts enabled links by total flow ascending, disables the smallest-flow link, re-solves the LP to reroute flow. After each successful prune, recomputes flows and re-sorts. Stops when a full pass finds no removable link.
- **Stations/Zones**: AV pickup/dropoff locations, placed via k-means clustering of trip origins. Each zone maps to a nearest network node.
- **Enabling cost (`enabling_cost`)**: cost to install infrastructure on a street segment. In Phase 1 it is subject to a budget constraint; in Phase 2 it is the objective to minimize (MILP) or reduce (greedy).
- **Flow variables**: `flow_c_l` = passenger flow per OD pair per link; `beta_l` = rebalancing (empty vehicle) flow per link; `served_c` = trips served per OD pair.
- **Scaling**: costs are divided by `cost_scale` (10000), capacities by `cap_scale` (100), OD counts by `24 * cap_scale` for numerical stability.
- **L1 penalty**: added to Phase 1 objective to break LP degeneracy and push `enabled_l` toward 0/1 (see thesis Chapter 4).
- **Reduced network**: Between phases, `link_df` is filtered to enabled links and `from_to_links_per_node` is recomputed. The flow solution is remapped from full-network to reduced-network indices.

## Output

Solutions are written to `solutions/`:
- `kepler_solution_*_relaxed_*_zones.csv` — Phase 1 enabled links with lat/lon for Kepler.gl
- `kepler_solution_*_{mincost,greedy}_*_zones.csv` — Phase 2 enabled links with lat/lon for Kepler.gl
- `resume_*_relaxed_zones*.txt` — Phase 1 summary: cost, L1 penalty, enabled link counts at thresholds
- `resume_*_{mincost,greedy}_zones*.txt` — Phase 2 summary: method, cost, served demand, demand floor, enabled link counts
- `*_stations.csv` — station locations
- `.sol` and `.lp` files — Gurobi model artifacts (gitignored)
