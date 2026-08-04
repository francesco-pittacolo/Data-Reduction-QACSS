"""Orchestration layer: load data, choose D and run one method."""

import statistics
import time
import os
from datetime import datetime
from .evaluation import clear_value_cache, compute_query_probs, f, filter_queries
from .io_utils import load_dataset, load_queries, save_run_results
from .methods import method_a, method_b, method_c, method_d
from .pool import build_pool, diagnose_pool_relevance, print_coverage_diagnostic
from .similarity import build_sim_cache
from .diversity import embedding_diversity

def build_dataset(method_cfg, cfg, embeddings, queries):
    dataset = method_cfg["mode"]

    if dataset == "full":
        D = list(embeddings.keys())
        keep_size = cfg["full"]["keep_size"]
    elif dataset == "pool":
        pool_cfg = cfg["pool"]
        D = build_pool(
            embeddings,
            pool_cfg["strategy"],
            pool_cfg["size"],
            cfg["seed"],
            pool_cfg["ids"],
            queries=queries,
        )
        keep_size = pool_cfg["keep_size"]
    else:
        raise ValueError("dataset must be 'pool' or 'full'")

    if keep_size <= 0 or keep_size >= len(D):
        raise ValueError(f"keep_size={keep_size} must be between 1 and {len(D) - 1}")

    return D, keep_size


def evaluate(selected, D, queries, embeddings, query_probs):
    clear_value_cache()
    sim_cache = build_sim_cache(embeddings)
    return f(selected, D, queries, sim_cache, query_probs)


def run_method(method_name, cfg, embeddings=None, queries=None, run_dir=None):
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if embeddings is None:
        embeddings = load_dataset(os.path.join(ROOT_DIR, cfg["dataset_path"]))
    if queries is None:
        queries = load_queries(os.path.join(ROOT_DIR, cfg["queries_path"]))
        
    method_cfg = cfg["methods"][method_name]
    D, keep_size = build_dataset(method_cfg, cfg, embeddings, queries)
    print("DEBUG")
    print("method:", method_name)
    print("mode:", method_cfg["mode"])
    print("D size:", len(D))
    print("keep_size:", keep_size)
    print("queries before filtering:", len(queries))

    queries = filter_queries(queries, D)

    print("queries after filtering:", len(queries))
    print("first queries:", queries[:3])
    if method_cfg["mode"] == "pool" and not cfg["quiet"] and not cfg.get("_pool_diagnostic_done", False):
        print_coverage_diagnostic(D, queries)
        diagnose_pool_relevance(D, queries)
        cfg["_pool_diagnostic_done"] = True

    queries = filter_queries(queries, D)
    query_probs = compute_query_probs(queries)

    print("\n" + "=" * 60)
    print(f"RUN METHOD {method_name.upper()}")
    print("=" * 60)
    print(f"Dataset : {method_cfg['mode']}")
    print(f"D size  : {len(D)}")
    print(f"Budget  : {keep_size}")

    start = time.perf_counter()

    if method_name == "a":
        selected, _ = method_a.run(
            D,
            queries,
            embeddings,
            keep_size,
            query_probs=query_probs,
            quiet=cfg["quiet"],
            progress_every=500,
            top_k_print=20,
        )

    elif method_name == "b":
        selected, _ = method_b.run(
            D,
            keep_size,
            queries,
            quiet=cfg["quiet"]
        )

    elif method_name == "c":
        sim_cache = build_sim_cache(embeddings)
        clear_value_cache()

        selected, _ = method_c.run(
            D,
            queries,
            sim_cache,
            keep_size,
            query_probs=query_probs,
            quiet=cfg["quiet"],
        )

    elif method_name == "d":
        selected = method_d.run(
            D,
            embeddings,
            queries,
            keep_size,
            alpha=method_cfg["alpha"],
            seed = cfg["seed"]
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")

    method_time = time.perf_counter() - start

    score = evaluate(
        selected,
        D,
        queries,
        embeddings,
        query_probs
    )

    diversity = embedding_diversity(
        selected,
        embeddings
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Method : {method_name.upper()}")
    print(f"|D'|   : {len(selected)}")
    print(f"Diversity: {diversity:.6f}")
    print(f"f(D')  : {score:.6f}")
    
    dataset_name = os.path.splitext(os.path.basename(cfg["dataset_path"]))[0]
    result = {
        "dataset": dataset_name,
        "method": method_name.upper(),
        "mode": method_cfg["mode"],
        "D_size": len(D),
        "budget": keep_size,
        "score": score,
        "diversity": diversity,
        "method_time_seconds": method_time,
        "parameters": {
            k: v for k, v in method_cfg.items()
            if k != "mode"
        },
        "selected_ids": selected,
    }

    save_run_results(cfg, method_name, keep_size, result, dataset_name, run_dir)

    return selected, score, diversity, method_time

def run_all(cfg):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    embeddings = load_dataset(os.path.join(ROOT_DIR, cfg["dataset_path"]))
    queries = load_queries(os.path.join(ROOT_DIR, cfg["queries_path"]))

    dataset = os.path.splitext(os.path.basename(cfg["dataset_path"]))[0]
    run_dir = f"{dataset}_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    results = {}

    for method_name, enabled in cfg["run"].items():
        if not enabled:
            continue

        selected, score, diversity, method_time = run_method(
            method_name, cfg, embeddings, queries, run_dir=run_dir
        )

        mode = cfg["methods"][method_name]["mode"]

        if mode == "full":
            D_size = len(embeddings)
        else:
            D_size = cfg["pool"]["size"]

        results[method_name.upper()] = {
            "mode": mode,
            "D_size": D_size,
            "D_prime_size": len(selected),
            "score": score,
            "diversity": diversity,
            "time": method_time,
        }

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        "{:<10}{:<10}{:<10}{:<10}{:<15}{:<15}{:<10}".format(
        "Method",
        "Mode",
        "|D|",
        "|D'|",
        "f(D')",
        "Diversity",
        "Time(s)"
        )
    )

    print("-" * 90)

    for method, r in sorted(results.items(), key=lambda x: x[1]["score"], reverse=True):
        print(
            f"{method:<10}"
            f"{r['mode']:<10}"
            f"{r['D_size']:<10}"
            f"{r['D_prime_size']:<10}"
            f"{r['score']:<15.6f}"
            f"{r['diversity']:<15.6f}"
            f"{r['time']:<10.4f}"
        )

    return results