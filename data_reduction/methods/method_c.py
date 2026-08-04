"""
Method C: Shapley value based selection.

Ranks elements by their estimated Shapley contribution and selects the most
valuable ones using exact or Monte Carlo estimation.
"""

import itertools
import math
import random
import time

from ..evaluation import clear_value_cache, f



def shapley_value_exact(item, D, queries, sim_cache, query_probs=None,
                         progress_every=0, progress_label=""):
    """
    Exact Shapley value of a single item over candidate universe D.
    """
    n = len(D)
    others = [x for x in D if x != item]
    phi = 0.0

    total_subsets = 2 ** (n - 1) if n > 0 else 0
    done = 0
    t_start = time.time()

    for r in range(n):
        weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
        for subset in itertools.combinations(others, r):
            marginal = (
                f(list(subset) + [item], D, queries, sim_cache, query_probs)
                - f(subset, D, queries, sim_cache, query_probs)
            )
            phi += weight * marginal

            done += 1
            if progress_every and done % progress_every == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else float("inf")
                eta = (total_subsets - done) / rate if rate > 0 else float("inf")
                print(f"    {progress_label}[{done:,}/{total_subsets:,}] subsets evaluated "
                      f"| {rate:.1f} subsets/s | ETA {eta:.1f}s")

    return phi


def choose_samples(n):
    """
    Choose the number of Monte Carlo permutations according to the size
    of the candidate universe.

    Larger datasets use fewer samples to limit computation time.
    """
    if n <= 50:
        return 2000
    elif n <= 300:
        return 1000
    else:
        return 500
    
def shapley_values_approx(D, queries, sim_cache, query_probs=None,
                           n_samples=2000, seed=42, progress_every=0):
    """
    Monte Carlo permutation-sampling estimate of the Shapley value for every
    id in D *simultaneously* -- O(n_samples * n) calls to f() total, instead
    of O(n * 2^(n-1)).

    For each sampled random ordering of D, walk through it left to right and
    credit each item with f(coalition-so-far) - f(coalition-before-it): its
    marginal contribution at the position it happened to land in that
    ordering. Averaging over many random orderings estimates the exact
    Shapley value.

    Returns:
        dict[int, float]: item id -> estimated phi
    """
    rng = random.Random(seed)
    D = list(D)
    n = len(D)
    phi = {d: 0.0 for d in D}

    for s in range(1, n_samples + 1):
        perm = D[:]
        rng.shuffle(perm)

        prev_value = 0.0  # f(empty set) = 0
        coalition = []
        for d in perm:
            coalition.append(d)
            new_value = f(coalition, D, queries, sim_cache, query_probs)
            phi[d] += new_value - prev_value
            prev_value = new_value

        if progress_every and s % progress_every == 0:
            print(f"  [{s:,}/{n_samples:,}] permutation samples done")

    for d in phi:
        phi[d] /= n_samples

    return phi


def best_set_by_shapley(shapley, keep_size):
    """Return the `keep_size` ids with the highest Shapley value."""
    ranking = sorted(shapley, key=shapley.get, reverse=True)
    return ranking[:keep_size]


def run(D, queries, sim_cache, keep_size, query_probs=None, quiet=False,
        max_exact_size=18, n_samples=None, seed=42, method="auto"):
    """
    Compute Shapley values for every id in D, then select the top `keep_size`.

    method:
        "auto"  -> exact if len(D) <= max_exact_size, else approximate
        "exact" -> force exact (can be very slow for large D)
        "approx"-> force the Monte Carlo approximation

    Returns:
        (best_subset, shapley) where shapley is dict[id, phi]
    """
    start = time.time()
    clear_value_cache()
    n = len(D)
    
        
    use_exact = method == "exact" or (method == "auto" and n <= max_exact_size)

    if not quiet:
        print("=" * 60)
        print(f"METHOD C - {'EXACT' if use_exact else 'APPROXIMATE (Monte Carlo)'} SHAPLEY")
        print("=" * 60)
        print(f"Queries: {len(queries)}")
        print(f"Keep size: {keep_size}")

    if use_exact:
        exact_calls = n * (2 ** (n - 1)) if n > 0 else 0
        subsets_per_item = 2 ** (n - 1) if n > 0 else 0
        
        # Print a progress line roughly 5 times per item's computation
        item_progress_every = max(subsets_per_item // 5, 1) if not quiet else 0

        if not quiet:
            print(f"\nComputing Shapley values exactly ({exact_calls:,} f() evaluations, "
                  f"{subsets_per_item:,} subsets per item)...")
        shapley = {}
        for i, d in enumerate(D, 1):
            t = time.time()
            shapley[d] = shapley_value_exact(
                d, D, queries, sim_cache, query_probs,
                progress_every=item_progress_every,
                progress_label=f"item {i}/{n} id={d}: ",
            )
            if not quiet:
                print(f"{i}/{n} id={d} phi={shapley[d]:.8f} time={time.time() - t:.2f}s")
    else:
        if n_samples is None:
            n_samples = choose_samples(n)
            
        if not quiet:
            approx_calls = n_samples * n
            print(f"\n|D|={n} is above max_exact_size={max_exact_size}: exact Shapley would need "
                  f"{n * (2 ** max(n - 1, 0)):,} f() evaluations, which is impractical.")
            print(f"Estimating via {n_samples:,} random permutations instead "
                  f"(~{approx_calls:,} f() evaluations)...")
        shapley = shapley_values_approx(
            D, queries, sim_cache, query_probs=query_probs,
            n_samples=n_samples, seed=seed,
            progress_every=max(n_samples // 10, 1) if not quiet else 0,
        )
        if not quiet:
            print("\nEstimated Shapley values:")
            for d in D:
                print(f"id={d:6d} phi~={shapley[d]:.8f}")

    if not quiet:
        print("\nRanking Shapley:")
        print("-" * 40)
        for d in sorted(D, key=lambda x: shapley[x], reverse=True):
            print(f"id={d:6d} phi={shapley[d]:.8f}")

    best = best_set_by_shapley(shapley, keep_size)
    shapley_score = sum(shapley[d] for d in best)
    coalition_value = f(best, D, queries, sim_cache, query_probs)

    if not quiet:
        print("\n" + "=" * 60)
        print(f"BEST SET SIZE {keep_size}")
        print("-" * 40)
        print(f"Subset      : {best}")
        print(f"Sum Shapley : {shapley_score:.8f}")
        print(f"Value f(S)  : {coalition_value:.8f}")
        print(f"\nTotal execution time: {time.time() - start:.4f}s")

    return best, shapley