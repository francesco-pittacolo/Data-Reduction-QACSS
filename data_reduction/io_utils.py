import csv
import json
import os
from datetime import datetime
import numpy as np


def load_dataset(path):
    """
    Load dataset embeddings from a CSV file where line i (1-based) is the
    embedding vector of element with id i.

    Returns:
        dict[int, np.ndarray]: element id -> embedding vector
    """
    embeddings = {}
    with open(path, "r", newline="") as fh:
        for idx, row in enumerate(csv.reader(fh), start=1):
            if row:
                embeddings[idx] = np.array([float(x) for x in row], dtype=np.float64)
    return embeddings


def load_queries(path):
    """
    Load the query log from a CSV file where each line is a comma-separated
    list of item ids returned by that query.

    Returns:
        list[list[int]]
    """
    queries = []
    with open(path, "r", newline="") as fh:
        for row in csv.reader(fh):
            row = [int(x.strip()) for x in row if x.strip()]
            if row:
                queries.append(row)
    return queries

def save_run_results(cfg, method_name, budget, result, dataset, run_dir=None):
    if not cfg["save_results"]:
        return

    if run_dir is None:
        folder = cfg["results_dir"]
    else:
        folder = os.path.join(cfg["results_dir"], run_dir)

    os.makedirs(folder, exist_ok=True)

    filename = f"{dataset}_method_{method_name}_B{budget}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    path = os.path.join(folder, filename)

    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results: {os.path.abspath(path)}")
