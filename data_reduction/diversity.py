import numpy as np

def embedding_diversity(selected, embeddings):
    """
    Average pairwise cosine distance.

    Higher value = more diverse selection.

    Args:
        selected:
            list of item ids

        embeddings:
            dict:
                item_id -> embedding vector

    Returns:
        float
    """
    if len(selected) <= 1:
        return 0.0

    X = np.array([embeddings[i] for i in selected], dtype=np.float32)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-10)

    sim = X @ X.T
    dist = (1 - sim) / 2

    n = len(selected)
    diversity = np.sum(dist[np.triu_indices(n, k=1)]) / (n * (n - 1) / 2)

    return float(diversity)