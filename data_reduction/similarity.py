"""Cosine similarity and the pairwise similarity cache used by every method."""

def cosine_sim(v1, v2):
    import numpy as np

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denom == 0:
        return 0.0

    cos = float(np.dot(v1, v2) / denom)

    return (cos + 1) / 2


def build_sim_cache(embeddings):
    """
    Lazy cache for pairwise cosine similarities.
    """
    class LazyCache(dict):
        def __missing__(self, key):
            i, j = key
            value = cosine_sim(embeddings[i], embeddings[j])
            self[key] = value
            return value

    return LazyCache()
