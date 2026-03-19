# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InsideOut optimizes the placement of sensor infrastructure on city streets for autonomous driving. Instead of equipping each vehicle with sensors ("inside-out" approach), sensors are placed on road infrastructure and shared. Given a limited municipal budget, the system determines which streets should receive infrastructure to maximize served passenger trips.

The optimization is formulated as a Mixed-Integer Linear Program (MILP) over a city road network (currently Singapore, ~38k nodes, ~78k arcs). Two scalability strategies are used: graph pruning via "skeletons" and LP relaxation with an L1 penalty to break degeneracy.

Reference: `agentic/docs/report.pdf` — semester thesis describing the math and algorithms in detail.

## Running

```bash
python main.py
```

Configuration is done by editing variables at the top of `main.py`:
- `city_name`: `"singapore"` or `"Berlin"`
- `n_zones`: number of stations (e.g., 2, 5, 10, 40)
- `l1_penalty`: L1 regularization strength for the LP relaxation solver
- `simplified_network_available` / `skeletons_available`: toggle pre-processed network input

### Dependencies

- **Gurobi** (with valid license) — the MILP/LP solver
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
5. Calls `MaximumCustomerCoverage.get_maximum_customer_coverage_model()` to build the base Gurobi model
6. **Phase 1**: Solves via `solve_relaxed()` (LP relaxation with L1 penalty) — maximizes served trips under budget
7. **Phase 2**: Filters to enabled-only subnetwork, rebuilds the model, solves via `solve_min_cost()` — minimizes enabling cost while maintaining at least the LP-relaxation served demand
8. Exports both solutions to CSV for Kepler.gl visualization

### `MaximumCustomerCoverage.py` — Gurobi optimization model
- `get_maximum_customer_coverage_model()` builds the base LP without budget/enabling constraints: flow conservation per OD pair per node, rebalancing flow conservation, capacity constraints. Returns the model plus variable handles (`served_c`, `flow_c_l`, `beta_l`).
- `solve_relaxed()` (Phase 1) adds continuous `enabled_l` variables in [0,1], linking/capacity constraints, budget constraint, and L1 penalty. Uses interior point (Method=2, Crossover=0).
- `solve_min_cost()` (Phase 2) adds binary `enabled_l` variables, minimizes total enabling cost subject to a demand floor (>= LP served demand). Uses LP solution as warm start.

### `Utils.py` — Data processing
Network loading, coordinate transforms (EPSG:3414 ↔ WGS84 for Singapore, EPSG:31468 ↔ WGS84 for Berlin), SCC extraction via NetworkX, degree-1 node contraction, trip origin/destination extraction from MATSim plans, k-means zone creation, OD count aggregation, Kepler.gl CSV export.

## Key Concepts

- **Stations/Zones**: AV pickup/dropoff locations, placed via k-means clustering of trip origins. Each zone maps to a nearest network node.
- **Enabling cost (`enabling_cost`)**: cost to install infrastructure on a street segment; subject to a total budget constraint.
- **Flow variables**: `flow_c_l` = passenger flow per OD pair per link; `beta_l` = rebalancing (empty vehicle) flow per link; `served_c` = trips served per OD pair.
- **Scaling**: costs are divided by `cost_scale` (10000), capacities by `cap_scale` (100), OD counts by `24 * cap_scale` for numerical stability.
- **L1 penalty**: added to objective to break LP degeneracy and push `enabled_l` toward 0/1 (see thesis Chapter 4).

## Output

Solutions are written to `solutions/`:
- `kepler_solution_*.csv` — enabled links with lat/lon for Kepler.gl map visualization
- `resume_*.txt` — cost summary and enabled link counts at various thresholds
- `*_stations.csv` — station locations
- `.sol` and `.lp` files — Gurobi model artifacts (gitignored)
