import json
import numpy as np
import os
from models.embeddings import embed_text

NODES_PATH = "data/nodes.json"

# Load nodes once at startup
try:
    # Absolute path resolution
    if not os.path.isabs(NODES_PATH):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(module_dir)
        NODES_PATH = os.path.join(project_root, "data", "nodes.json")

    with open(NODES_PATH, "r") as f:
        NODES = json.load(f)

    # Preload embeddings
    EMB_ARRAY = np.array([node["embedding"] for node in NODES])
    NODE_IDS = [node["id"] for node in NODES]

except Exception as e:
    import traceback
    print("ERROR loading nodes in vector_search:", e)
    traceback.print_exc()
    NODES = []
    EMB_ARRAY = np.array([])
    NODE_IDS = []


# ------------------------------
# COSINE SIMILARITY (safe)
# ------------------------------
def cosine_vec_matrix(matrix, vector):
    """
    Computes cosine similarity between:
      - matrix: shape (N, D)
      - vector: shape (D,)
    Returns vector of length N.
    """
    vector = np.array(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)

    denom = matrix_norms * vector_norm
    denom = np.where(denom == 0, 1e-9, denom)  # avoids NaN

    return (matrix @ vector) / denom


# ------------------------------
# MAIN SEARCH FUNCTION
# ------------------------------
def search_vector(query_text, top_k=5, metadata_filter=None):
    """
    Perform vector search.

    New features:
    ✔ metadata filtering: metadata_filter=("key","value")
    ✔ safe top_k (supports k > dataset size)
    ✔ returns consistent vector_score
    """
    if len(NODES) == 0:
        return []

    # 1. Embed query
    q_embed = embed_text(query_text)

    # 2. Apply metadata filter if provided
    filtered_nodes = NODES
    filtered_embs = EMB_ARRAY

    if metadata_filter:
        key, value = metadata_filter

        filtered_nodes = [
            n for n in NODES
            if n.get("metadata", {}).get(key) == value
        ]

        if not filtered_nodes:
            return []

        filtered_embs = np.array([n["embedding"] for n in filtered_nodes])

    # 3. Similarity computation (fast, vectorized)
    sims = cosine_vec_matrix(filtered_embs, q_embed)

    # 4. Get top_k sorted results
    top_k = min(top_k, len(filtered_nodes))
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        node = filtered_nodes[idx]

        results.append({
            "id": node["id"],
            "name": node.get("name", ""),
            "vector_score": float(sims[idx]),  # IMPORTANT for hybrid TC compliance
        })

    return results

