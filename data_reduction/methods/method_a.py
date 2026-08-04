"""
Method A: Exact brute-force subset selection.

Enumerates all subsets of a fixed size and returns the one maximizing f(D').
Only practical for small candidate pools due to exponential complexity.

Reference: "Stochastic Submodular Data Forgetting", SIGMOD 2026 (https://velgias.github.io/docs/RicoSV26.pdf)
"""

import itertools
import math
import time

from ..similarity import build_sim_cache
from ..evaluation import f
from ..pool import format_time


def best_subset_bruteforce(D, queries, embeddings, keep_size, query_probs=None,
                            progress_every=200, top_k_print=10, quiet=False):
    """
    Exhaustively evaluate f(D') for every D' subset of D with |D'| = keep_size,
    and return the best one.

    Returns:
        (best_subset, best_score, all_scores)
    """
    D = list(D)
    n = len(D)
    k = keep_size
    total_subsets = math.comb(n, k)

    if not quiet:
        print("=" * 70)
        print("METHOD A: Brute-force exact subset search")
        print("=" * 70)
        print(f"Candidate universe D   : {D}")
        print(f"|D| = {n}, keep k = {k}, delete = {n - k}")
        print(f"Embedding dimension    : {embeddings[D[0]].shape[0]}")
        print(f"Number of queries |Q|  : {len(queries)}")
        print(f"Subsets to evaluate    : C({n},{k}) = {total_subsets:,}")
        print("-" * 70)

    t_cache_start = time.perf_counter()
    sim_cache = build_sim_cache(embeddings)
    t_cache = time.perf_counter() - t_cache_start
    if not quiet:
        print(f"Pairwise similarity cache built in {t_cache * 1000:.2f} ms "
              f"({len(sim_cache)} pairs)")

    best_score = -float("inf")
    best_subset = None
    all_scores = []

    t_search_start = time.perf_counter()
    for i, D_prime in enumerate(itertools.combinations(D, k), start=1):
        score = f(D_prime, D, queries, sim_cache, query_probs)
        all_scores.append((D_prime, score))
        if score > best_score:
            best_score = score
            best_subset = D_prime

        if not quiet and progress_every and i % progress_every == 0:
            elapsed = time.perf_counter() - t_search_start
            rate = i / elapsed if elapsed > 0 else float("inf")
            eta = (total_subsets - i) / rate if rate > 0 else float("inf")
            print(f"  [{i:,}/{total_subsets:,}] best so far f={best_score:.6f} "
                  f"| {rate:.1f} subsets/s | ETA {format_time(eta)}")

    t_search = time.perf_counter() - t_search_start

    if not quiet:
        print("-" * 70)
        if t_search > 0:
            print(f"Search finished in {t_search:.3f}s "
                  f"({total_subsets / t_search:.1f} subsets/s average)")
        else:
            print("Search finished (near-instant)")

        print(f"\nTop {min(top_k_print, len(all_scores))} subsets by f(D'):")
        for subset, score in sorted(all_scores, key=lambda x: -x[1])[:top_k_print]:
            deleted = [d for d in D if d not in subset]
            print(f"  KEEP {subset}  DELETE {deleted}  f(D') = {score:.6f}")

        print("\n" + "=" * 70)
        print(f"BEST subset to KEEP : {best_subset}")
        print(f"BEST f(D')          : {best_score:.6f}")
        print(f"Items to DELETE    : {[d for d in D if d not in best_subset]}")
        print("=" * 70)

    return best_subset, best_score, all_scores


def run(D, queries, embeddings, keep_size, query_probs=None, quiet=False,
        progress_every=500, top_k_print=10):
    """Convenience wrapper returning just (best_subset, best_score)."""
    best_subset, best_score, _ = best_subset_bruteforce(
        D, queries, embeddings, keep_size,
        query_probs=query_probs, progress_every=progress_every,
        top_k_print=top_k_print, quiet=quiet,
    )
    return best_subset, best_score
