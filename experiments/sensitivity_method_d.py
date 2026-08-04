import os
import sys
import time
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from data_reduction.methods import method_d
from data_reduction.io_utils import load_dataset, load_queries
from data_reduction.similarity import build_sim_cache
from data_reduction.evaluation import compute_query_probs, f, clear_value_cache, filter_queries
from data_reduction.diversity import embedding_diversity

"""
The experiment can be run in three modes:
    - "alpha": vary alpha while keeping expansion and budget fixed;
    - "expansion": vary the candidate expansion factor while keeping alpha and budget fixed;
    - "budget": vary the selection budget while keeping alpha and expansion fixed.
"""
MODE = "alpha"

#ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
EXPANSION_VALUES = [1, 1.5, 2, 3, 5]
#BUDGET_VALUES = [416, 832, 2081, 4162, 8324]
BUDGET_VALUES = [1000, 2000, 5000, 10000, 20000]

FIXED_ALPHA = 1
FIXED_EXPANSION = 2
FIXED_BUDGET = 2000

"""
When ALL_BUDGETS=True, each parameter configuration
is evaluated across all budgets listed in BUDGET_VALUES.
"""
ALL_BUDGETS = True

def main():
    cfg = CONFIG

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings = load_dataset(os.path.join(ROOT_DIR, cfg["dataset_path"]))
    queries = load_queries(os.path.join(ROOT_DIR, cfg["queries_path"]))
    dataset_name = os.path.splitext(os.path.basename(cfg["dataset_path"]))[0]
    
    D = list(embeddings)
    print(list(embeddings.keys())[:5])
    print(D[:5])
    queries = filter_queries(queries, D)    
    query_probs = compute_query_probs(queries)

    print("=" * 70)
    print("METHOD D - SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("Dataset size:", len(D))
    print("Mode:", MODE)

    if MODE == "alpha":
        values = ALPHA_VALUES
        print("Expansion:", FIXED_EXPANSION)
        print("Budget:", FIXED_BUDGET)

    elif MODE == "expansion":
        values = EXPANSION_VALUES
        print("Alpha:", FIXED_ALPHA)
        print("Budget:", FIXED_BUDGET)

    elif MODE == "budget":
        values = BUDGET_VALUES
        print("Alpha:", FIXED_ALPHA)
        print("Expansion:", FIXED_EXPANSION)

    else:
        raise ValueError("MODE must be alpha, expansion or budget")

    print("\nBuilding similarity cache...")
    sim_cache = build_sim_cache(embeddings)

    results = []

    for value in values:
        if MODE == "alpha":
            alpha = value
            expansion = FIXED_EXPANSION
        elif MODE == "expansion":
            alpha = FIXED_ALPHA
            expansion = value
        else:
            alpha = FIXED_ALPHA
            expansion = FIXED_EXPANSION

        budgets = BUDGET_VALUES if ALL_BUDGETS else [FIXED_BUDGET]

        for budget in budgets:
            print("\n" + "=" * 70)
            print(f"alpha={alpha} expansion={expansion} budget={budget}")
            print("=" * 70)

            start = time.perf_counter()

            selected = method_d.run(D, embeddings, queries, budget, alpha=alpha, expansion=expansion, seed=CONFIG["seed"])

            clear_value_cache()
            score = f(selected, D, queries, sim_cache, query_probs)
            diversity = embedding_diversity(selected, embeddings)

            elapsed = time.perf_counter() - start

            results.append((value, budget, score, diversity, elapsed))

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    name = {"alpha": "Alpha", "expansion": "Expansion", "budget": "Budget"}[MODE]

    if ALL_BUDGETS:

        current_value = None

        for value, budget, score, diversity, elapsed in results:

            if value != current_value:
                current_value = value

                print("\n" + "=" * 70)
                if MODE == "alpha":
                    print(f"Alpha = {value:.2f}")
                elif MODE == "expansion":
                    print(f"Expansion = {value}")
                else:
                    print(f"Budget = {value}")

                print("=" * 70)
                print("{:>10} {:>12} {:>12} {:>12}".format(
                    "Budget", "f(D')", "Diversity", "Time"))
                print("-" * 70)

            print(f"{budget:>10} "
                f"{score:>12.6f} "
                f"{diversity:>12.6f} "
                f"{elapsed:>11.2f}s ")

    else:

        print("{:>10} {:>12} {:>12} {:>12}".format(
            name, "f(D')", "Diversity", "Time"))
        print("-" * 70)

        for value, _, score, diversity, elapsed in results:
            value = f"{value:.2f}" if MODE == "alpha" else str(value)
            print(f"{value:>10} "
                f"{score:>12.6f} "
                f"{diversity:>12.6f} "
                f"{elapsed:>11.2f}s ")

    out_path = f"results/sensitivity_{MODE}_{dataset_name}.csv"
    with open(out_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["value", "budget", "score", "diversity", "elapsed"])
        writer.writerows(results)
    print(f"Saved sensitivity results to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()