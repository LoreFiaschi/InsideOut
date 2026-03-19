# InsideOut

Optimal placement of sensor infrastructure on city streets for autonomous driving.

Instead of equipping each autonomous vehicle with expensive onboard sensors, the *inside-out* approach places perception sensors on road infrastructure. A municipality builds and maintains the sensor network under a limited budget; autonomous mobility providers pay a fee to use it. This reduces per-vehicle costs, eliminates sensor redundancy across the fleet, and improves safety by enabling vehicles to "see around corners."

This repository solves the key optimization problem: **given a budget, which streets should receive sensor infrastructure to maximize the number of served passenger trips, at minimum cost?**

## Two-phase optimization

The pipeline solves two optimization problems in sequence:

### Phase 1 — LP relaxation (max served trips)

A linear program with continuous enabling variables in [0, 1] maximizes total served passenger trips subject to a budget constraint. An L1 penalty on the enabling variables breaks degeneracy and pushes the solution toward integrality. This phase runs fast via Gurobi's interior point method and identifies which links should carry flow.

### Phase 2 — Cost reduction (min enabling cost)

The network is filtered to only the links enabled in Phase 1. On this reduced network, the goal is to disable unnecessary links to reduce total infrastructure cost while maintaining at least the same total served demand. Two methods are available (configured via `phase2_method`):

- **`"milp"`** — A mixed-integer linear program with binary enabling variables minimizes total infrastructure cost exactly. The Phase 1 solution is used as a warm start (incumbent). Gives the optimal solution but can be slow on large networks.

- **`"greedy"`** — A greedy heuristic that iteratively disables links, starting from the one carrying the least flow. After each removal, the LP is re-solved (via warm-started dual simplex) to reroute flow through the remaining links. If the demand floor cannot be met, the link is kept. The algorithm stops when a full pass over all remaining links finds no more to prune. Faster than MILP but produces a heuristic (not necessarily optimal) solution.

Both phases share the same underlying flow model: passenger flow conservation per origin-destination pair, rebalancing (empty vehicle) flow conservation, demand upper bounds, and arc capacity constraints. See `agentic/docs/report.pdf` for the full mathematical formulation.

## Setup

### Prerequisites

- Python 3.10+
- [Gurobi](https://www.gurobi.com/) with a valid license
- [MATSim](https://www.matsim.org/) network and population data

### Python dependencies

```
gurobipy
pandas
numpy
matsim
geopandas
shapely
scipy
scikit-learn
pyproj
tqdm
networkx
utm
```

### Data

Data files are not included in the repository (gitignored). Place them as follows:

- `data/` — Pre-simplified network CSVs (`simplified_link_df_singapore.csv`, `simplified_node_df_singapore.csv`), MATSim population plans (`.xml.gz`), and optionally skeleton link CSVs.
- An external directory (configured via `path_prefix` in `main.py`) containing the full MATSim network XML and cost-per-link CSVs.

## Usage

Edit the configuration variables at the top of `main.py`:

| Variable | Description | Default |
|---|---|---|
| `city_name` | City network to use | `"singapore"` |
| `n_zones` | Number of AV stations | `2` |
| `l1_penalty` | L1 regularization strength (Phase 1) | `1e-2` |
| `phase2_method` | Phase 2 method: `"milp"` or `"greedy"` | `"greedy"` |
| `simplified_network_available` | Use pre-simplified network | `True` |
| `skeletons_available` | Use skeleton-pruned network | `False` |

Then run:

```bash
python main.py
```

Logs are written to `insideout_{city}_{n_zones}_zones.log`.

## Output

Results are written to `solutions/`:

| File | Contents |
|---|---|
| `kepler_solution_*_relaxed_*_zones.csv` | Phase 1: enabled links from LP relaxation |
| `kepler_solution_*_{mincost,greedy}_*_zones.csv` | Phase 2: enabled links after cost reduction |
| `resume_*_relaxed_zones*.txt` | Phase 1 summary: cost, L1 penalty, enabled link counts at various thresholds |
| `resume_*_{mincost,greedy}_zones*.txt` | Phase 2 summary: method, cost, served demand, demand floor, enabled link counts |
| `*_stations.csv` | Station (zone center) locations |

All output CSVs include lat/lon columns for visualization in [Kepler.gl](https://kepler.gl/). Gurobi model artifacts (`.sol`, `.lp`) are written to the project root (gitignored).

## Project structure

```
main.py                      Pipeline orchestrator (data loading, two-phase solve, export)
MaximumCustomerCoverage.py   Gurobi optimization models (LP relaxation, MILP, greedy pruning)
Utils.py                     Network processing, spatial ops, data preparation
data/                        Input data (gitignored)
solutions/                   Output solutions (gitignored)
agentic/docs/report.pdf      Semester thesis with full problem formulation
```

## Pipeline steps

1. **Load network** — Reads MATSim network and population plans. Optionally loads a pre-simplified or skeleton-pruned network.
2. **Preprocess** — Filters to the largest strongly connected component. Transforms coordinates (EPSG:3414 for Singapore, EPSG:31468 for Berlin).
3. **Create zones** — Clusters trip origins into `n_zones` stations via k-means. Assigns each link and activity to the nearest zone.
4. **Build demand** — Computes origin-destination trip counts between zone pairs, filtering self-loops.
5. **Scale** — Divides costs, capacities, and demand by scaling factors for numerical stability.
6. **Phase 1: LP relaxation** — Builds the base flow model on the full network. Adds continuous enabling variables, linking constraints, budget constraint, and L1 penalty. Maximizes served trips via interior point method.
7. **Phase 2: Cost reduction** — Filters the network to only Phase 1 enabled links. Depending on `phase2_method`:
   - *MILP*: Rebuilds the flow model, adds binary enabling variables and a demand floor (served >= Phase 1 total), minimizes cost with Phase 1 warm start.
   - *Greedy*: Builds the flow model with a demand floor, then iteratively disables links (smallest flow first), re-solving the LP via dual simplex each step. Stops when no more links can be pruned.
8. **Export** — Writes both solutions to CSV for Kepler.gl visualization, along with summary statistics.

## References

- I. Khomutovskiy, *Optimization Methods for Infrastructure-Based Sensor Placement in Autonomous Driving*, Semester Thesis, ETH Zurich, December 2025.
- M. Schneider, *Profitability analysis of the inside-out approach* (prior work on economic viability).
- G. Cuccorese, *MILP formulations for infrastructure sensor placement* (prior work on optimization formulations).

## License

This project is part of Prof. Dr. Emilio Frazzoli's research group at the Institute for Dynamic Systems and Control (IDSC), ETH Zurich, in collaboration with SBB AG and Siemens Mobility AG.
