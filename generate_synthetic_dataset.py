import csv
import os
import numpy as np

N_ITEMS = 100000
EMBEDDING_DIM = 512
N_CLUSTERS = 100
N_QUERIES = 5000

QUERY_SIZE_MIN = 20
QUERY_SIZE_MAX = 100

OUTPUT_DIR = "synthetic_dataset"
SEED = 42

EMBEDDING_FILE = "photos.csv"
QUERY_FILE = "queries.csv"

def generate_dataset():
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("GENERATING SYNTHETIC DATASET")
    print("=" * 60)
    print("D size:", N_ITEMS)
    print("Embedding dimension:", EMBEDDING_DIM)
    print("Latent clusters:", N_CLUSTERS)
    print("Queries:", N_QUERIES)

    print("\nGenerating embeddings...")

    centers = np.random.normal(0, 5, (N_CLUSTERS, EMBEDDING_DIM))

    embeddings = np.zeros((N_ITEMS, EMBEDDING_DIM), dtype=np.float32)
    item_clusters = np.zeros(N_ITEMS, dtype=np.int32)

    for i in range(N_ITEMS):
        cluster = np.random.randint(0, N_CLUSTERS)
        emb = centers[cluster] + np.random.normal(0, 1, EMBEDDING_DIM)
        emb /= np.linalg.norm(emb)

        embeddings[i] = emb
        item_clusters[i] = cluster

    embedding_path = os.path.join(OUTPUT_DIR, EMBEDDING_FILE)

    print("\nWriting:", embedding_path)

    with open(embedding_path, "w", newline="") as f:
        writer = csv.writer(f)
        for emb in embeddings:
            writer.writerow(emb.tolist())

    print("Building clusters...")

    clusters = {}

    for idx, cluster in enumerate(item_clusters):
        clusters.setdefault(cluster, []).append(idx + 1)

    print("Generating queries...")

    queries = []

    for _ in range(N_QUERIES):
        cluster = np.random.randint(0, N_CLUSTERS)

        available = clusters[cluster]

        size = np.random.randint(
            QUERY_SIZE_MIN,
            QUERY_SIZE_MAX + 1
        )

        size = min(size, len(available))

        query = np.random.choice(
            available,
            size=size,
            replace=False
        )

        queries.append(query.tolist())

    queries_path = os.path.join(OUTPUT_DIR, "queries.csv")

    print("Writing:", queries_path)

    with open(queries_path, "w", newline="") as f:
        writer = csv.writer(f)
        for q in queries:
            writer.writerow(q)

    print("\nDONE")
    print("-" * 28)
    print("Embedding path :", embedding_path)
    print("Queries path:", queries_path)


if __name__ == "__main__":
    generate_dataset()