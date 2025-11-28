from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import os

# ---- Local imports ----
from search.vector_search import search_vector
from graph.graph_search import bfs, graph_closeness
from search.hybrid_search import hybrid_search
from graph.nlp.relation_extractor import extract_relations
from models.embeddings import embed_text


# ============================================================
#                APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="Hybrid Retrieval System",
    description="Vector + Graph Native Search Engine",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#                VECTOR SEARCH
# ============================================================

@app.get("/search/vector")
def search_vector_get(query: str = Query(...), top_k: int = 5):
    return {"results": search_vector(query, top_k)}

@app.post("/search/vector")
def search_vector_post(body: dict):
    q = body.get("query_text")
    k = body.get("top_k", 5)
    return {"results": search_vector(q, k)}


# ============================================================
#                GRAPH SEARCH
# ============================================================

@app.get("/search/graph")
def graph_search(
    start_id: int,
    depth: int = 2,
    rel_type: str | None = None
):
    """
    Graph traversal with optional relationship-type filtering.

    Example:
      /search/graph?start_id=10&depth=2&rel_type=author_of
    """
    try:
        allowed = [rel_type] if rel_type else None
        visited = bfs(start_id, max_depth=depth, allowed_types=allowed)
        return {
            "start_id": start_id,
            "depth": depth,
            "rel_type": rel_type,
            "visited": visited
        }
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )



# ============================================================
#                HYBRID SEARCH
# ============================================================

@app.get("/search/hybrid")
def hybrid_get(
    query: str,
    top_k: int = 5,
    focus_id: int | None = None,
    depth: int = 2,
    alpha: float = 0.7
):
    return {
        "results": hybrid_search(
            query_text=query,
            top_k=top_k,
            focus_id=focus_id,
            bfs_depth=depth,
            alpha=alpha
        )
    }


@app.post("/search/hybrid")
def hybrid_post(body: dict):
    try:
        print(f"DEBUG: Hybrid POST received body: {body}")
        return {
            "results": hybrid_search(
                query_text=body.get("query_text"),
                top_k=body.get("top_k", 5),
                focus_id=body.get("focus_id"),
                bfs_depth=body.get("bfs_depth", 2),
                alpha=body.get("alpha", 0.7)
            )
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"ERROR in hybrid_post: {e}")
        print(tb)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": tb
            }
        )


# ============================================================
#                CRUD  (NODES + EDGES)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NODES_FILE = os.path.join(DATA_DIR, "nodes.json")
EDGES_FILE = os.path.join(DATA_DIR, "edges.json")


# ----------------------- Models -----------------------

class NodeCreate(BaseModel):
    name: str
    text: str
    metadata: dict = {}

class NodeUpdate(BaseModel):
    name: str | None = None
    text: str | None = None
    metadata: dict | None = None

class EdgeCreate(BaseModel):
    source: int
    target: int
    type: str = "related_to"
    weight: float = 1.0

class EdgeUpdate(BaseModel):
    type: str | None = None
    weight: float | None = None
   


# -------------------- Helpers -------------------------

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
#                NODE CRUD
# ============================================================

@app.post("/nodes", status_code=201)
def create_node(node: NodeCreate):
    nodes = load(NODES_FILE)
    edges = load(EDGES_FILE)

    new_id = max([n["id"] for n in nodes], default=0) + 1

    new_node = {
        "id": new_id,
        "name": node.name,
        "text": node.text,
        "metadata": node.metadata,
        "embedding": embed_text(node.text)
    }
    nodes.append(new_node)

    # ---- Auto NLP edges ----
    triples = extract_relations(node.text)
    created_edges = []

    for t in triples:
        subj = t["subject"].lower()
        obj = t["object"].lower()

        subj_node = next((n for n in nodes if n["name"].lower() == subj), None)
        obj_node = next((n for n in nodes if n["name"].lower() == obj), None)

        if subj_node and obj_node:
            eid = max([e["id"] for e in edges], default=0) + 1
            edge = {
                "id": eid,
                "source": subj_node["id"],
                "target": obj_node["id"],
                "type": t["relation"],
                "weight": 1.0
            }
            edges.append(edge)
            created_edges.append(edge)

    save(NODES_FILE, nodes)
    save(EDGES_FILE, edges)

    return JSONResponse(
        status_code=201,
        content={
            "status": "created",
            "node": new_node,
            "auto_edges_created": created_edges
        }
    )


@app.get("/nodes/{id}", status_code=200)
def read_node(id: int):
    nodes = load(NODES_FILE)
    edges = load(EDGES_FILE)

    node = next((n for n in nodes if n["id"] == id), None)
    if not node:
        raise HTTPException(404, "Node not found")

    connected = [e for e in edges if e["source"] == id or e["target"] == id]

    return {"node": node, "edges": connected}


@app.put("/nodes/{id}", status_code=200)
def update_node(id: int, upd: NodeUpdate):
    nodes = load(NODES_FILE)

    for n in nodes:
        if n["id"] == id:

            if upd.name is not None:
                n["name"] = upd.name

            if upd.text is not None:
                n["text"] = upd.text
                n["embedding"] = embed_text(upd.text)

            if upd.metadata is not None:
                n["metadata"] = upd.metadata

            save(NODES_FILE, nodes)
            return {"status": "updated", "node": n}

    raise HTTPException(404, "Node not found")


@app.delete("/nodes/{id}", status_code=204)
def delete_node(id: int):
    nodes = load(NODES_FILE)
    edges = load(EDGES_FILE)

    new_nodes = [n for n in nodes if n["id"] != id]
    removed_edges = [e for e in edges if e["source"] == id or e["target"] == id]
    new_edges = [e for e in edges if e not in removed_edges]

    if len(new_nodes) == len(nodes):
        raise HTTPException(404, "Node not found")

    save(NODES_FILE, new_nodes)
    save(EDGES_FILE, new_edges)

    return JSONResponse(status_code=204, content=None)


# ============================================================
#                EDGE CRUD
# ============================================================

@app.post("/edges", status_code=201)
def create_edge(edge: EdgeCreate):
    edges = load(EDGES_FILE)
    new_id = max([e["id"] for e in edges], default=0) + 1

    new_edge = {
        "id": new_id,
        "source": edge.source,
        "target": edge.target,
        "type": edge.type,
        "weight": edge.weight,
    }

    edges.append(new_edge)
    save(EDGES_FILE, edges)

    return {"status": "created", "edge": new_edge}


@app.get("/edges/{id}", status_code=200)
def read_edge(id: int):
    edges = load(EDGES_FILE)
    edge = next((e for e in edges if e["id"] == id), None)
    if not edge:
        raise HTTPException(404, "Edge not found")
    return edge

@app.put("/edges/{id}", status_code=200)
def update_edge(id: int, upd: EdgeUpdate):
    """
    Update edge type and/or weight.
    Required by TC-API-05 (Relationship CRUD with weight update).
    """
    edges = load(EDGES_FILE)

    for e in edges:
        if e["id"] == id:
            if upd.type is not None:
                e["type"] = upd.type
            if upd.weight is not None:
                e["weight"] = upd.weight

            save(EDGES_FILE, edges)
            return {"status": "updated", "edge": e}

    raise HTTPException(status_code=404, detail="Edge not found")



@app.delete("/edges/{id}", status_code=204)
def delete_edge(id: int):
    edges = load(EDGES_FILE)
    new_edges = [e for e in edges if e["id"] != id]

    if len(new_edges) == len(edges):
        raise HTTPException(404, "Edge not found")

    save(EDGES_FILE, new_edges)
    return JSONResponse(status_code=204, content=None)




