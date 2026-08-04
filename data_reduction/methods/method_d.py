"""
Method D: Community-based selection with MMR diversity.

Combines IndepDF relevance weights, query co-occurrence communities, and
embedding-based diversity to select a representative subset of items.
"""

import time
import numpy as np
import networkx as nx
from collections import defaultdict
from community import community_louvain
from .method_b import compute_weight



def build_item_graph(queries, D):
    G = nx.Graph()
    D = set(D)

    for q in queries:
        q = [p for p in set(q) if p in D]

        for p in q:
            G.add_node(p)

        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                a, b = q[i], q[j]

                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    return G

# Communities
def compute_communities(G, seed):
    partition = community_louvain.best_partition(G, weight="weight", random_state=seed)

    communities = defaultdict(list)

    for node, c in partition.items():
        communities[c].append(node)

    return communities


# Budget
def allocate_budget(weights, communities, budget):
    importance = {}
    total = 0

    for c, items in communities.items():
        value = sum(weights[p] for p in items)
        importance[c] = value
        total += value

    if total == 0:
        total = 1

    budgets = {}
    remaining = budget

    for c in importance:
        b = int(budget * importance[c] / total)
        budgets[c] = b
        remaining -= b

    ordered = sorted(importance, key=importance.get, reverse=True)

    i = 0
    while remaining > 0:
        budgets[ordered[i]] += 1
        remaining -= 1
        i += 1

        if i == len(ordered):
            i = 0

    return budgets


def normalize(x):
    x = np.asarray(x)

    if np.max(x) == np.min(x):
        return np.ones(len(x))

    return (x - x.min()) / (x.max() - x.min())


def compute_candidate_limit(n_items, budget, expansion=2):
    return min(n_items, max(budget, int(expansion * budget)))


# MMR
def mmr_selection(items, embeddings, weights, budget, alpha=0.8, expansion=2):

    if alpha >= 1.0:
        return sorted(items, key=lambda x: weights[x], reverse=True)[:budget]

    if len(items) <= budget:
        return items

    items = sorted(items, key=lambda x: weights[x], reverse=True)
    items = items[:compute_candidate_limit(len(items), budget, expansion)]

    X = np.array([embeddings[p] for p in items], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-10

    relevance = normalize(np.array([weights[p] for p in items]))

    n = len(items)
    selected = []
    available = np.ones(n, dtype=bool)
    max_similarity = np.zeros(n, dtype=np.float32)

    for _ in range(min(budget, n)):
        diversity = 1 - max_similarity
        scores = alpha * relevance + (1-alpha) * diversity
        scores[~available] = -np.inf

        idx = np.argmax(scores)
        selected.append(idx)
        available[idx] = False

        similarity = X @ X[idx]
        max_similarity = np.maximum(max_similarity, similarity)

    return [items[i] for i in selected]


# Method D
def run(D, embeddings, queries, keep_size, alpha=0.8, expansion=2, seed = None):

    start = time.time()

    print("=" * 60)
    print("METHOD D - COMMUNITY + MMR DIVERSITY")
    print("=" * 60)

    ids = list(D)

    weights = compute_weight(D, queries)
    print("Building graph...")
    G = build_item_graph(queries, D)

    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    print("Detecting communities...")
    communities = compute_communities(G, seed)

    print("Communities:", len(communities))
    
    budgets = allocate_budget(weights, communities, keep_size)
    selected = []

    for c, items in communities.items():

        budget = budgets[c]

        if budget == 0:
            continue

        items = [i for i in items if i in D]

        if len(items) == 0:
            continue

        selected.extend(
            mmr_selection(
                items,
                embeddings,
                weights,
                budget,
                alpha,
                expansion
            )
        )

    if len(selected) < keep_size:

        remaining = [p for p in ids if p not in selected]
        remaining.sort(key=lambda x: weights[x], reverse=True)

        selected.extend(remaining[:keep_size-len(selected)])

    elif len(selected) > keep_size:
        selected = selected[:keep_size]

    print("\nSelected:")

    if len(selected) > 20:
        print("(" + ", ".join(map(str, selected[:20])) + ", ...)")
    else:
        print(selected)

    print("Kept:", len(selected))
    print("Time:", time.time()-start, "seconds")

    return selected