"""Objective function shared by every data-reduction method."""

_VALUE_CACHE = {}


def clear_value_cache():
    _VALUE_CACHE.clear()

def filter_queries(queries, D):
    D_set = set(D)
    filtered = []

    for q in queries:
        q_filtered = list(set(q).intersection(D_set))
        if q_filtered:
            filtered.append(q_filtered)

    return filtered

def restrict_query_to_set(query, id_set):
    return list(set(query).intersection(id_set))


def S(qD, qD_prime, sim_cache):
    if not qD or not qD_prime:
        return 0.0

    total = 0.0
    for d in qD:
        total += max(sim_cache[(d, d_prime)] for d_prime in qD_prime)
    return total / len(qD)


def compute_query_probs(queries):
    n_queries = len(queries)
    if n_queries == 0:
        return []

    return [1.0 / n_queries] * n_queries



def f(D_prime, D, queries, sim_cache, query_probs=None):
    key = frozenset(D_prime)
    if key in _VALUE_CACHE:
        return _VALUE_CACHE[key]

    if not queries:
        return 0.0
    if query_probs is None:
        query_probs = compute_query_probs(queries)

    D_set = set(D)
    D_prime_set = set(D_prime)
    expected_value = 0.0

    for q, probability in zip(queries, query_probs):
        qD = restrict_query_to_set(q, D_set)
        qD_prime = restrict_query_to_set(q, D_prime_set)
        expected_value += probability * S(qD, qD_prime, sim_cache)

    _VALUE_CACHE[key] = expected_value
    return expected_value
