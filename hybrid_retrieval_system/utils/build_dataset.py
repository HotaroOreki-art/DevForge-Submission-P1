import pandas as pd
import json
from models.embeddings import embed_text
from graph.nlp.relation_extractor import extract_relations


print("DEBUG: Script STARTED")

# Path to CSV
CSV_PATH = "data/pokemon.csv"

# Output files
NODES_OUT = "data/nodes.json"
EDGES_OUT = "data/edges.json"

def build_text(row):
    """Builds a descriptive text for embedding."""
    parts = []

    name = row["name"]
    t1 = row["type1"]
    t2 = row.get("type2", None)
    clas = row.get("classification", "")
    gen = row.get("generation", "")
    hp = row.get("hp", "")
    atk = row.get("attack", "")
    dfs = row.get("defense", "")

    if t2 and not pd.isna(t2):
        type_str = f"{t1} and {t2} type"
    else:
        type_str = f"{t1} type"

    parts.append(f"{name} is a {type_str} Pokémon.")
    parts.append(f"It is classified as {clas}.")
    parts.append(f"It belongs to generation {gen}.")
    parts.append(f"It has stats: HP {hp}, Attack {atk}, Defense {dfs}.")

    return " ".join(parts)

def load_pokemon():
    df = pd.read_csv(CSV_PATH)

    nodes = []
    edges = []
    adjacency = {}

    # Sort by pokedex number to detect evolutions
    df = df.sort_values("pokedex_number")

    previous_row = None

    for _, row in df.iterrows():
        node_id = int(row["pokedex_number"])
        text = build_text(row)
        embedding = embed_text(text)

        node = {
            "id": node_id,
            "name": row["name"],
            "text": text,
            "metadata": {
                "type1": row["type1"],
                "type2": row.get("type2", None),
                "generation": row.get("generation", None)
            },
            "embedding": embedding
        }

        nodes.append(node)
        adjacency[node_id] = []

        # Evolution edge detection (simple: +1 rule)
        if previous_row is not None:
            prev_id = int(previous_row["pokedex_number"])
            if node_id == prev_id + 1:
                # create evolution edge
                edges.append({
                    "source": prev_id,
                    "target": node_id,
                    "type": "evolves_into"
                })
                adjacency[prev_id].append(node_id)
                adjacency[node_id].append(prev_id)

        previous_row = row
                # -------------------------------
        # NLP RELATION EDGES (universal)
        # -------------------------------
        nlp_triples = extract_relations(text)

        for triple in nlp_triples:
            subj = triple["subject"]
            obj = triple["object"]
            relation = triple["relation"]

            # Find matching nodes by name
            subject_node = next((n for n in nodes if n["name"].lower() == subj.lower()), None)
            object_node = next((n for n in nodes if n["name"].lower() == obj.lower()), None)

            if subject_node and object_node:
                edges.append({
                    "source": subject_node["id"],
                    "target": object_node["id"],
                    "type": relation,
                    "weight": 1.0
                })

                adjacency[subject_node["id"]].append(object_node["id"])
                adjacency[object_node["id"]].append(subject_node["id"])



    # Save nodes
    with open(NODES_OUT, "w") as f:
        json.dump(nodes, f, indent=2)

    # Save edges
    with open(EDGES_OUT, "w") as f:
        json.dump(edges, f, indent=2)

    print("Dataset built successfully!")
    print(f"Nodes saved to {NODES_OUT}")
    print(f"Edges saved to {EDGES_OUT}")

if __name__ == "__main__":
    load_pokemon()
