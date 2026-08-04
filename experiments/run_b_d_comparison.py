import copy
import json
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from data_reduction.runner import run_method
from data_reduction.io_utils import load_dataset, load_queries


BUDGETS = [416, 832, 2081, 4162, 8324]
#BUDGETS = [1000, 2000, 5000, 10000, 20000]
PERCENTAGES = [1, 2, 5, 10, 20]
OUTPUT_FILE = "full_dataset_results_real.json"

def main():

    OUTPUT_FILE = os.path.join("results", OUTPUT_FILE)
    embeddings = load_dataset(CONFIG["dataset_path"])
    queries = load_queries(CONFIG["queries_path"])

    results = []


    for budget, percentage in zip(BUDGETS, PERCENTAGES):

        print("\n" + "=" * 70)
        print(f"BUDGET {budget} ({percentage}%)")
        print("=" * 70)


        for method in ["b", "d"]:

            cfg = copy.deepcopy(CONFIG)

            cfg["methods"][method]["mode"] = "full"
            cfg["full"]["keep_size"] = budget

            if method == "d":
                cfg["methods"][method]["alpha"] = 1


            selected, score, diversity, runtime = run_method(
                method,
                cfg,
                embeddings,
                queries
            )


            print(
                f"{method.upper()} -> "
                f"score={score:.6f}, "
                f"diversity={diversity:.6f}, "
                f"time={runtime:.6f}s"
            )


            result = {
                "method": method.upper(),
                "budget": budget,
                "percentage": percentage,
            }

            if method == "d":
                result["alpha"] = cfg["methods"][method].get("alpha", None)

            result.update({
                "score": score,
                "diversity": diversity,
                "runtime": runtime
            })

            results.append(result)


    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    main()