import json
import os
import numpy as np

from models.embeddings import embed_text
from search.vector_search import search_vector
from graph.graph_search import bfs, graph_closeness


# ----------- Load nodes so we can return names -----------

NODES_PATH = "data/nodes.json"

try:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    NODES_PATH = os.path.join(project_root, "data", "nodes.json")

    with open(NODES_PATH, "r") as f:
        NODES = json.load(f)

except:
    NODES = []


# ---------------- HYBRID SEARCH IMPLEMENTATION ---------------- #

def hybrid_search(
    query_text,
    top_k=5,
    focus_id=None,
    bfs_depth=2,
    alpha=0.7,
):
    """
    Hybrid search:
    - true cosine vector_similarity (no normalization)
    - hop-distance BFS graph_score
    - weighted combination
    """

    vector_weight = alpha
    graph_weight = 1 - alpha

    # 1) Vector search — get TOP-K candidates (true cosine scores)
    vector_results = search_vector(query_text, top_k=top_k)

    if not vector_results:
        return []

    # Convert search_vector output: rename "vector_score" → "score" for internal use
    vec_scores = {r["id"]: r["vector_score"] for r in vector_results}

    # 2) Determine graph anchor
    # If none provided, anchor is the best vector match
    if focus_id is None and vector_results:
        focus_id = vector_results[0]["id"]

    # BFS graph traversal
    visited = bfs(focus_id, max_depth=bfs_depth)

    # Compute graph scores
    graph_scores = {
        node_id: graph_closeness(node_id, visited)
        for node_id in vec_scores.keys()
    }

    # 3) Merge scores using weighted formula
    merged_results = []

    for node_id in vec_scores.keys():
        v = vec_scores[node_id]
        g = graph_scores.get(node_id, 0.0)

        final_score = vector_weight * v + graph_weight * g

        node_name = next((n["name"] for n in NODES if n["id"] == node_id), None)
        node_text = next((n["text"] for n in NODES if n["id"] == node_id), None)
        
        # Find hop distance from visited list (or None if not in graph)
        hop = next((item["hop"] for item in visited if item["id"] == node_id), None)

        merged_results.append({
            "id": node_id,
            "name": node_name,
            "text": node_text,
            "vector_score": float(v),
            "graph_score": float(g),
            "final_score": float(final_score),
            "hop": hop,
        })

    # 4) Sort by final_score
    merged_results.sort(key=lambda x: x["final_score"], reverse=True)

    return merged_results[:top_k]

