"""
Query frequency analysis: find which item ids show up most often in the
query log. Handy for picking a discriminating candidate pool D to feed into
Methods A/C (see utils.py's "queried" / "query_union" strategies for
alternatives that also guarantee co-occurrence).
"""

from collections import Counter


def compute_item_frequency(queries):
    """Count how many times each item id appears across all queries."""
    counter = Counter()
    for q in queries:
        counter.update(q)
    return counter


def select_top_items(counter, k):
    """Return the k most frequent (item_id, frequency) pairs."""
    return counter.most_common(k)


def query_coverage(pool_ids, queries, quiet=False):
    """
    Report how many queries contain at least one item from `pool_ids`, and
    the average overlap size among those that do.
    """
    pool = set(pool_ids)
    touching = 0
    total_overlap = 0

    if not quiet:
        print("\nQuery coverage:")
        print("-" * 60)

    for i, q in enumerate(queries):
        overlap = list(pool.intersection(q))
        if overlap:
            touching += 1
            total_overlap += len(overlap)
            if not quiet:
                print(f"query {i:3d} | |q|={len(q):3d} | overlap={len(overlap):2d} | {overlap}")

    if not quiet:
        print("-" * 60)
        print(f"Queries touching pool: {touching}/{len(queries)} "
              f"({100 * touching / len(queries):.2f}%)")
        if touching:
            print(f"Average overlap size: {total_overlap / touching:.2f}")

    return touching, total_overlap


def run(queries, pool_size, quiet=False):
    """
    Find the `pool_size` most frequent item ids in the query log and report
    their query coverage. Returns the list of pool ids.
    """
    counter = compute_item_frequency(queries)
    top = select_top_items(counter, pool_size)

    if not quiet:
        print(f"Loaded {len(queries)} queries")
        print("\nMost frequent items:")
        print("-" * 60)
        for rank, (item_id, freq) in enumerate(top, start=1):
            print(f"{rank:3d}) id={item_id:6d} frequency={freq}")

    pool_ids = [x[0] for x in top]

    if not quiet:
        print("\n" + "=" * 40)
        print("POOL IDS")
        print("=" * 40)
        print(pool_ids)

    query_coverage(pool_ids, queries, quiet=quiet)
    return pool_ids
