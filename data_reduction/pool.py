"""Candidate-pool construction and diagnostics."""

import random
from collections import Counter


def build_pool(embeddings, strategy, pool_size, seed, explicit_ids, queries=None):
    all_ids = list(embeddings.keys())

    if strategy == "ids":
        if not explicit_ids:
            raise ValueError("pool_strategy='ids' requires pool_ids")
        missing = [item_id for item_id in explicit_ids if item_id not in embeddings]
        if missing:
            raise ValueError(f"Item ids not found in dataset .csv: {missing}")
        return list(explicit_ids)

    if strategy == "first":
        return all_ids[:pool_size]

    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(all_ids, min(pool_size, len(all_ids))))

    if strategy == "queried":
        if not queries:
            raise ValueError("pool_strategy='queried' requires queries")
        queried_ids = sorted({d for q in queries for d in q if d in embeddings})
        if not queried_ids:
            raise ValueError("No query ids overlap with items.csv")
        rng = random.Random(seed)
        return sorted(rng.sample(queried_ids, min(pool_size, len(queried_ids))))

    if strategy == "query_union":
        if not queries:
            raise ValueError("pool_strategy='query_union' requires queries")
        valid_queries = [q for q in queries if any(d in embeddings for d in q)]
        if not valid_queries:
            raise ValueError("No query ids overlap with items.csv")

        rng = random.Random(seed)
        rng.shuffle(valid_queries)
        pool = []
        pool_set = set()
        for q in valid_queries:
            for d in q:
                if d in embeddings and d not in pool_set:
                    pool.append(d)
                    pool_set.add(d)
            if len(pool) >= pool_size:
                break
        return sorted(pool[:pool_size])

    if strategy == "top_frequency":
        if not queries:
            raise ValueError("pool_strategy='top_frequency' requires queries")
        counter = Counter(d for q in queries for d in q if d in embeddings)
        return [d for d, _ in counter.most_common(pool_size)]

    raise ValueError(f"Unknown pool strategy: {strategy}")


def print_coverage_diagnostic(D, queries, max_print=20):
    D_set = set(D)
    overlaps = [(i, q, [d for d in q if d in D_set]) for i, q in enumerate(queries)]
    touching = [(i, q, overlap) for i, q, overlap in overlaps if overlap]

    print("\nQuery coverage diagnostic:")
    print(f"{'query#':>7}  {'|q|':>4}  {'|q n D|':>8}  overlap ids")

    for i, q, overlap in touching[:max_print]:
        print(f"{i:>7}  {len(q):>4}  {len(overlap):>8}  {overlap}")

    if len(touching) > max_print:
        print(f"... ({len(touching) - max_print} more queries omitted)")

    print(f"-> {len(touching)}/{len(queries)} queries touch the pool at all.")

    multi_hit = sum(len(overlap) >= 2 for _, _, overlap in touching)

    if not touching:
        print("WARNING: no query overlaps this pool; f(D') is always 0.")
    elif len(touching) <= 2:
        print("WARNING: very few queries touch this pool; expect flat/tied scores.")
    elif multi_hit == 0:
        print("WARNING: every touching query has exactly one pool member; expect many ties.")
    else:
        print(f"{multi_hit}/{len(touching)} touching queries have >=2 pool members.")

    print()


def diagnose_pool_relevance(D, queries, max_print=20):
    D_set = set(D)
    touching_queries = 0
    referenced = set()

    for q in queries:
        hit = D_set.intersection(q)
        if hit:
            touching_queries += 1
            referenced.update(hit)

    never_referenced = D_set - referenced

    print("-" * 70)
    print("Pool relevance diagnostic:")
    print(f"  Queries that intersect D               : {touching_queries} / {len(queries)}")
    print(f"  Pool members ever returned by a query : {len(referenced)} / {len(D)}")

    if referenced:
        print(f"    examples -> {sorted(referenced)[:max_print]}")

    if never_referenced:
        print(f"  Pool members never returned by a query: {len(never_referenced)} / {len(D)}")
        print(f"    examples -> {sorted(never_referenced)[:max_print]}")

    print("-" * 70)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f} h"
    return f"{hours / 24:.2f} days"
