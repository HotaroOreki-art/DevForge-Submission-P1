import json
import os
from collections import deque

# ------------------ PATH RESOLUTION (SAFE) ------------------ #

EDGES_PATH = "data/edges.json"

try:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    EDGES_PATH = os.path.join(project_root, "data", "edges.json")

    with open(EDGES_PATH, "r") as f:
        EDGES = json.load(f)

except Exception as e:
    import traceback
    print("ERROR loading edges in graph_search:", e)
    traceback.print_exc()
    EDGES = []

# ------------------ BUILD RICH ADJACENCY LIST ------------------ #
# We preserve edge type + weight (needed for test cases)

ADJ = {}   # node_id -> list of {target, type, weight}

try:
    for e in EDGES:
        src, tgt = e["source"], e["target"]
        rel = e.get("type", "related_to")
        wt = e.get("weight", 1.0)

        # forward
        ADJ.setdefault(src, []).append({
            "target": tgt,
            "type": rel,
            "weight": wt
        })

        # backward (make graph undirected for BFS)
        ADJ.setdefault(tgt, []).append({
            "target": src,
            "type": rel,
            "weight": wt
        })

except Exception as e:
    print("ERROR building rich adjacency list:", e)
    ADJ = {}

# ------------------ BFS (NEW, TEST-CASE COMPLIANT) ------------------ #

def bfs(start_id: int, max_depth: int = 2, allowed_types=None):
    """
    RETURNS full traversal list with hop distances + edge metadata:
      [
        { "id": 12, "hop": 1, "edge": "related_to", "weight": 1.0 },
        { "id": 58, "hop": 2, "edge": "references", "weight": 0.6 }
      ]

    Features:
      ✔ depth limit
      ✔ relationship-type filtering
      ✔ cycle-safe
      ✔ returns metadata required by hybrid search tests
    """

    if allowed_types is None:
        allowed_types = None  # meaning allow all

    if start_id not in ADJ:
        return []

    queue = deque([(start_id, 0, None, None)])  
    visited = {start_id}
    results = []

    while queue:
        node, depth, edge_type, weight = queue.popleft()

        # Skip the root node
        if depth > 0:
            results.append({
                "id": node,
                "hop": depth,
                "edge": edge_type,
                "weight": weight
            })

        if depth == max_depth:
            continue

        # Expand neighbors
        for e in ADJ.get(node, []):
            if allowed_types and e["type"] not in allowed_types:
                continue

            nxt = e["target"]

            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1, e["type"], e.get("weight", 1.0)))

    return results


# ------------------ GRAPH CLOSENESS SCORE (REMains SAME) ------------------ #

def graph_closeness(node_id, bfs_results):
    """
    Weighted graph proximity score.

    bfs_results is a list of dicts from bfs():
      { "id": ..., "hop": ..., "edge": ..., "weight": ... }

    Scoring:
      base = 1 / (1 + hop)
      score = base * edge_weight

    So:
      - same hop, higher weight -> higher score
      - farther nodes get lower score
    """
    for r in bfs_results:
        if r["id"] == node_id:
            hop = r["hop"]
            wt = r.get("weight", 1.0)
            base = 1.0 / (1 + hop)
            return base * wt

    return 0.0

