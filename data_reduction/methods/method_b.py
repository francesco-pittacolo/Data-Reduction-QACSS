"""
Method B: IndepDF relevance-based selection.

Ranks items using query frequency weights and selects the highest-scoring
items without using embeddings.
Reference: "Stochastic Submodular Data Forgetting", SIGMOD 2026 (https://velgias.github.io/docs/RicoSV26.pdf)
"""

import time

import numpy as np


def compute_weight(D, queries):
    """
    w(d) = sum_q P(q) * 1/|q|, assuming uniform P(q) = 1/|Q|.

    Returns:
        dict[int, float]: item id (restricted to D) -> weight
    """
    weights = {d: 0.0 for d in D}
    n_queries = len(queries)
    if n_queries == 0:
        return weights

    query_weight = 1.0 / n_queries
    for q in queries:
        answer_set = [d for d in q if d in weights]
        if not answer_set:
            continue
        per_element_weight = query_weight / len(q)
        for d in answer_set:
            weights[d] += per_element_weight

    return weights


def indep_df(D, budget, queries):
    """
    Select the `budget` highest-weight ids from D.

    Returns:
        (selected, weights) where selected is a list of ids sorted by
        descending weight, and weights is the dict from compute_weight.
    """
    weights = compute_weight(D, queries)
    selected = sorted(D, key=lambda d: weights[d], reverse=True)[:budget]
    return selected, weights


def run(D, budget, queries, quiet=False):
    """Run IndepDF over a (typically small) candidate universe D."""
    start = time.perf_counter_ns()
    selected, weights = indep_df(D, budget, queries)
    elapsed = time.perf_counter_ns() - start

    if not quiet:
        print("=" * 60)
        print("METHOD B - IndepDF")
        print("=" * 60)
        print(f"\nCandidate universe ({len(D)}):")
        #print(D[:10])
        print("\nSelected subset:")
        if len(selected) > 20:
            print("(" + ", ".join(map(str, selected[:20])) + ", ...)")
        else:
            print(selected)
        print("\nWeights:")
        print("-" * 40)
        for d in selected[:10]:
            print(f"id={d:6d} weight={weights[d]:.8f}")
        print(f"\nExecution time: {elapsed / 1_000_000:.6f} ms")

    return selected, weights


def run_full_dataset(n_items, keep_ratio, queries, quiet=False, top_n_print=20):
    """
    Run IndepDF over ALL item ids 1..n_items (no embeddings needed), keeping
    `keep_ratio` fraction of them. Uses a vectorized numpy implementation
    since the full catalogue can be large.
    """
    start = time.time()

    weights = np.zeros(n_items)
    n_queries = len(queries)
    for q in queries:
        query_weight = 1.0 / n_queries
        per_element_weight = query_weight / len(q)
        # ids in queries.csv are 1-based
        np.add.at(weights, np.array(q) - 1, per_element_weight)

    budget = int(n_items * keep_ratio)
    top_indices = np.argpartition(weights, -budget)[-budget:]
    top_indices = top_indices[np.argsort(weights[top_indices])[::-1]]
    selected = (top_indices + 1).tolist()

    elapsed = time.time() - start

    if not quiet:
        print("=" * 60)
        print("METHOD B - IndepDF (full dataset)")
        print("=" * 60)
        print(f"Items: {n_items}")
        print(f"Queries: {n_queries}")
        print(f"Budget: {budget}")
        print(f"\nSelected items (top {top_n_print}):")
        if len(selected) > top_n_print:
            print("(" + ", ".join(map(str, selected[:top_n_print])) + ", ...)")
        else:
            print(selected)
        print("\nTop weights:")
        print("-" * 40)
        for idx in selected[:top_n_print]:
            print(f"id={idx:6d} weight={weights[idx - 1]:.8f}")
        print(f"\nExecution time: {elapsed * 1000:.3f} ms")

    return selected, weights
