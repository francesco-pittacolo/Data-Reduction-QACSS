# Data Reduction via Query-Aware Community-based Subset Selection (QACSS)

This project studies **data reduction / subset selection**: given a large dataset `D` and a workload of past queries `Q`, select a much smaller subset `D' ⊆ D` that preserves the query answers of the original dataset as well as possible.

Four selection strategies are implemented and compared:

| Method | Name | Idea |
|--------|------|------|
| **A** | Exact brute-force | Exhaustively evaluates every candidate subset of the required size and keeps the best one. Optimal but only feasible for very small candidate pools. |
| **B** | IndepDF | Assigns each element an independent relevance score based on how often it appears in the query workload, then keeps the top-scoring elements. Very fast, no optimality guarantee in general. |
| **C** | Shapley value | Estimates each element's contribution to the objective via its (exact or Monte-Carlo approximated) Shapley value. Captures interactions between elements but is computationally expensive. |
| **D** | QACSS (proposed) | Combines IndepDF relevance scores with **community detection** (Louvain) over a query co-occurrence graph. The selection budget is allocated across communities proportionally to their total relevance, and an optional **Maximum Marginal Relevance (MMR)** step adds embedding-space diversity within each community. |

The core research question is when structural information from the query workload (communities) helps beyond purely relevance-based selection (IndepDF), and how an optional diversity term (MMR) trades off against relevance. The full write-up, formal problem definition, algorithm pseudocode, complexity analysis and experimental results are in [`Data_Reduction_QACSS.pdf`](Data_Reduction_QACSS.pdf).

## Objective function

The quality of a candidate subset `D'` is measured by a **data forgetting objective**: for each query `q`, how well the answer set restricted to `D'` covers the original answer set, using cosine similarity between embeddings, averaged over the query workload. This is implemented in `data_reduction/evaluation.py`.

## Repository structure

```
.
├── README.md                    # This file
├── main.py                      # CLI entry point
├── config.py                    # Central experiment configuration
├── generate_synthetic_dataset.py# Generates the synthetic benchmark (100k items, 5k queries)
├── data/
│   ├── photos/                  # Real-world benchmark (41,620 photos / 443 queries)
│   │   └── queries.csv
│   └── synthetic_dataset/       # Synthetic benchmark (100k items / 5k queries)
│       └── synthethic_queries.csv
├── data_reduction/
│   ├── methods/
│   │   ├── method_a.py          # Exact brute-force selection
│   │   ├── method_b.py          # IndepDF relevance-based selection
│   │   ├── method_c.py          # Shapley-value based selection (exact / Monte Carlo)
│   │   └── method_d.py          # QACSS: community-based selection + MMR
│   ├── pool.py                  # Candidate-pool ("restricted D") construction & diagnostics
│   ├── similarity.py            # Cosine similarity + lazy pairwise similarity cache
│   ├── evaluation.py            # Objective function f(D') and query-answer utilities
│   ├── diversity.py             # Embedding-space diversity metric
│   ├── io_utils.py              # Dataset/query loading, results saving
│   └── runner.py                # Orchestration: loads data, builds D, runs a method, saves results
├── experiments/
│   ├── run_b_d_comparison.py    # Full-dataset comparison of Method B vs Method D across budgets
│   └── sensitivity_method_d.py  # Sensitivity analysis of Method D w.r.t. alpha / expansion / budget
├── results/                     # JSON/CSV outputs from experiment runs (checked-in examples included)
└── Data_Reduction_QACSS.pdf     # Full paper: problem definition, algorithm, experiments, discussion
```

## Installation

Requires Python 3.9+.

```bash
pip install numpy networkx python-louvain matplotlib
```

- `numpy` — vectorized similarity/embedding operations
- `networkx` — query co-occurrence graph (Method D)
- `python-louvain` (imported as `community`) — Louvain community detection (Method D)
- `matplotlib` — only needed for `experiments/run_b_d_comparison.py`

## Data format

Each dataset consists of two CSV files:

- **Embeddings file** (e.g. `photos.csv`): line `i` (1-based) is the embedding vector of element with id `i`, as comma-separated floats.
- **Queries file** (e.g. `queries.csv`): each line is a comma-separated list of element ids returned by that query.

### Downloading the real-world photo dataset

The embeddings file for the real-world benchmark (`data/photos/photos.csv`, 41,620 photo embeddings) is **not included in this repository** because of its size. Only `data/photos/queries.csv` is checked in.

Download it from Google Drive and place it alongside the queries file:

1. Download the zip from: https://drive.google.com/file/d/1YjCt-RZUyEHslqmA3yJHJi-Tk6SNFlbP/view
2. Unzip it and locate `photos.csv`.
3. Move/copy it into `data/photos/`, so you end up with:

```
data/photos/
├── photos.csv     # <- from the downloaded zip
└── queries.csv    # already in the repo
```

`config.py`'s `dataset_path` (`data/photos/photos.csv`) already points here by default, so no further changes are needed once the file is in place.

To regenerate the synthetic benchmark (100,000 items, 512-dim embeddings, 100 latent clusters, 5,000 queries):

```bash
python generate_synthetic_dataset.py
```

This writes `synthetic_dataset/photos.csv` and `synthetic_dataset/queries.csv` in the working directory. Point `config.py`'s `dataset_path` / `queries_path` at the desired dataset before running experiments.

## Configuration

All experiment parameters are centralized in [`config.py`](config.py):

```python
"""Central configuration for the data-reduction experiments."""

CONFIG = {
    "dataset_path": r"data/photos/photos.csv",
    "queries_path": r"data/photos/queries.csv",
    "seed": 42,
    "quiet": False,

    "save_results": True,
    "results_dir": "results",  

    "run": {
        "a": True,
        "b": True,
        "c": True,
        "d": True,
    },

    # --- candidate universe D ---
    # "first"         -> first pool_size ids in the dataset
    # "random"        -> random sample of pool_size ids
    # "ids"           -> explicit ids listed below
    # "queried"       -> random sample from ids appearing in queries
    # "query_union"   -> union of ids from random queries
    # "top_frequency" -> most frequent ids in queries
    "pool": {
        "strategy": "top_frequency",
        "size": 18,
        "ids": [
            30618, 13589, 12473, 26916, 37771,
            11696, 5257, 33523, 2201, 5519,
            5342, 23778, 3975, 11802, 17505,
        ],
        "keep_size": 7,
    },

    "full": {
        "keep_size": 416
    },


    # --- methods ---
    "methods": {
        "a": {
            "mode": "pool",
        },
        "b": {
            "mode": "pool",
        },
        "c": {
            "mode": "pool",
        },
        "d": {
            "mode": "pool",
            "alpha":  1,
            "expansion": 2 # default is 2
        }
    },
}
```

Key options:

- `pool.strategy` selects how the candidate universe `D` is built for pool-mode experiments (see below for details on each option).
- `methods.<x>.mode`: `"pool"` restricts `D` to the candidate pool above; `"full"` runs the method over the entire dataset (only Methods B and D are practical at full scale).
- `methods.d.alpha`: the MMR trade-off parameter — `alpha = 1` disables diversity re-ranking and reduces to pure relevance-within-community selection; `alpha = 0` maximizes embedding diversity within each community's budget.
- `methods.d.expansion`: candidate-list expansion factor used before MMR re-ranking within a community (keeps only the top `expansion * budget` relevance-ranked candidates before running MMR, for efficiency).

### Pool construction strategies

| Strategy | What it does |
|---|---|
| `"first"` | Takes the first `pool.size` ids in the dataset, in id order. Simplest option, no relation to the query workload. |
| `"random"` | Draws a random sample of `pool.size` ids from the whole dataset (seeded by `config.seed`). May include ids that never appear in any query. |
| `"ids"` | Uses the explicit list given in `pool.ids` as-is. Raises an error if any listed id is missing from the dataset. |
| `"queried"` | Randomly samples `pool.size` ids, but only from ids that appear in at least one query — guarantees every pool member is query-relevant, without controlling for query overlap between members. |
| `"query_union"` | Shuffles the queries, then greedily accumulates the union of their ids until `pool.size` is reached. Tends to include ids that co-occur within the same queries. |
| `"top_frequency"` | Takes the `pool.size` ids with the highest occurrence count across the query log. This is the strategy used in the paper's restricted-pool experiments (18-element pool), since it guarantees a small but non-trivial, query-relevant candidate universe for Methods A and C, which are only tractable at small scale. |

A random pool was intentionally **not** used for the restricted-pool experiments: it can pull in ids with very low or zero query frequency, producing a pool where most queries don't intersect `D` at all — `f(D')` would then be close to 0 for every method and the comparison would be uninformative. `top_frequency` avoids this by construction.

## Usage

Run every enabled method (as configured in `config.py`) on the configured dataset:

```bash
python main.py --mode all
# or, using the short form:
python main.py -m all
```

Run a single method with `--mode`/`-m`:

```bash
python main.py -m a   # exact brute-force (small pools only)
python main.py -m b   # IndepDF
python main.py -m c   # Shapley value
python main.py -m d   # QACSS (community-based + optional MMR)
```

Each run prints a diagnostic report (query coverage of the candidate pool, per-method progress) followed by the final objective value `f(D')`, embedding diversity, and execution time. When `save_results` is enabled, results are written as JSON under `results/<dataset>_run_<timestamp>/`.

### Experiment scripts

- **`experiments/run_b_d_comparison.py`** — reproduces the full-dataset comparison between Method B and Method D across selection budgets of 1%, 2%, 5%, 10% and 20% of the dataset, saving results to `results/full_dataset_results_real.json`.
- **`experiments/sensitivity_method_d.py`** — sweeps one of `alpha`, `expansion`, or `budget` (set via the `MODE` constant at the top of the file) to study Method D's sensitivity, saving a CSV to `results/sensitivity_<mode>_<dataset>.csv`.

## Results

The experimental results — objective value, embedding diversity, and execution time for each method across selection budgets, the comparison against theoretical complexity, and the discussion of when community-based selection helps or doesn't — are reported in full in [`Data_Reduction_QACSS.pdf`](Data_Reduction_QACSS.pdf). Raw run outputs (JSON/CSV) are kept under `results/` for reference.

## References

The IndepDF baseline (Method B) and the photo benchmark dataset are taken from:

> Rico Velgias et al. 2026. *Stochastic Submodular Data Forgetting.* In Proceedings of the ACM SIGMOD International Conference on Management of Data.

Community detection uses the Louvain algorithm:

> Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. 2008. *Fast unfolding of communities in large networks.* Journal of Statistical Mechanics: Theory and Experiment 2008, 10 (2008), P10008.