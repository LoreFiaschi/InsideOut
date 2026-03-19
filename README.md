# InsideOut

Optimal placement of sensor infrastructure on city streets for autonomous driving.

Instead of equipping each autonomous vehicle with expensive onboard sensors, the *inside-out* approach places perception sensors on road infrastructure. A municipality builds and maintains the sensor network under a limited budget; autonomous mobility providers pay a fee to use it. This reduces per-vehicle costs, eliminates sensor redundancy across the fleet, and improves safety by enabling vehicles to "see around corners."

This repository solves the key optimization problem: **given a budget, which streets should receive sensor infrastructure to maximize the number of served passenger trips?**

The problem is formulated as a Mixed-Integer Linear Program (MILP) and solved via LP relaxation with an L1 penalty to break degeneracy and encourage near-integer solutions. Graph pruning via *skeletons* can optionally reduce the network size before optimization. See `agentic/docs/report.pdf` for the full mathematical formulation and algorithmic details.

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
| `l1_penalty` | L1 regularization strength | `1e-2` |
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
| `kepler_solution_*_zones.csv` | Enabled links with lat/lon for [Kepler.gl](https://kepler.gl/) visualization |
| `resume_*_zones*.txt` | Cost summary, L1 penalty, enabled link counts at various thresholds |
| `*_stations.csv` | Station (zone center) locations |

Gurobi model artifacts (`.sol`, `.lp`) are written to the project root (gitignored).

## Project structure

```
main.py                      Pipeline orchestrator
MaximumCustomerCoverage.py   Gurobi optimization model (LP relaxation + L1 penalty)
Utils.py                     Network processing, spatial ops, data preparation
data/                        Input data (gitignored)
solutions/                   Output solutions (gitignored)
agentic/docs/report.pdf      Semester thesis with full problem formulation
```

## How it works

1. **Load network** — Reads MATSim network and population plans. Optionally loads a pre-simplified or skeleton-pruned network.
2. **Preprocess** — Filters to the largest strongly connected component. Transforms coordinates (EPSG:3414 for Singapore, EPSG:31468 for Berlin).
3. **Create zones** — Clusters trip origins into `n_zones` stations via k-means. Assigns each link and activity to the nearest zone.
4. **Build demand** — Computes origin-destination trip counts between zone pairs, filtering self-loops.
5. **Scale** — Divides costs, capacities, and demand by scaling factors for numerical stability.
6. **Build model** — Constructs the LP with flow conservation constraints (passenger + rebalancing), demand bounds, and capacity limits.
7. **Solve** — Adds continuous enabling variables, linking constraints, budget constraint, and L1 penalty. Solves via Gurobi's interior point method.
8. **Export** — Writes enabled-link solutions to CSV for Kepler.gl map visualization.

## References

- I. Khomutovskiy, *Optimization Methods for Infrastructure-Based Sensor Placement in Autonomous Driving*, Semester Thesis, ETH Zurich, December 2025.
- M. Schneider, *Profitability analysis of the inside-out approach* (prior work on economic viability).
- G. Cuccorese, *MILP formulations for infrastructure sensor placement* (prior work on optimization formulations).

## License

This project is part of Prof. Dr. Emilio Frazzoli's research group at the Institute for Dynamic Systems and Control (IDSC), ETH Zurich, in collaboration with SBB AG and Siemens Mobility AG.
