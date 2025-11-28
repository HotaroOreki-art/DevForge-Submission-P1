Hybrid Vector + Graph Retrieval Engine Devfolio Hackathon Submission Team: Catalyst Crew 🚀 📌 Overview

This project implements a Vector + Graph Native Database, built completely from scratch — without using any existing vector DB or graph DB solutions.

Our system supports semantic vector search, graph traversal, and a hybrid retrieval model that intelligently merges both signals.

We also built:

A complete CRUD API for nodes & edges

Automatic NLP-based relationship extraction for unstructured text

A dark-mode Web UI for live testing

A reference dataset (Pokémon) + real-world custom data examples (Albert Einstein, Eiffel Tower)

Everything runs 100% locally, offline, and in real time.

🧠 Key Features 🔹 1. Vector Search (Semantic Retrieval)

Embeds text with a 384-dimensional transformer encoder.

Computes cosine similarity against stored embeddings.

Supports metadata-based filtering.

🔹 2. Graph Search (Knowledge Traversal)

BFS traversal up to configurable depth.

Supports cycles, multi-type relationships, weighted traversal.

Graph score = 1 / (1 + hop_distance).

🔹 3. Hybrid Search (Vector + Graph Merge)

Weighted scoring:

final_score = α * vector_score + (1 - α) * graph_score

Allows:

Pure vector search (α=1.0),

Pure graph search (α=0.0),

Balanced hybrid (e.g., α=0.7).

🔹 4. Unstructured Data Support

Paste any raw text → the system:

Embeds it,

Stores it as a node,

Extracts relations (subject–verb–object),

Auto-creates edges between known concepts.

🔹 5. NLP-Based Automatic Edge Generation

Example input:

“Albert Einstein developed the theory of relativity.”

System extracts:

Subject: Albert Einstein

Relation: developed

Object: Theory of Relativity

Creates:

Einstein —developed→ Theory of Relativity

Works for any domain: places, notes, research papers, wiki text, etc.

🔹 6. Fully Local Dark-Mode UI

A single HTML file (ui/index.html) that interacts with the backend:

Vector Search tab

Graph Search tab

Hybrid Search tab

Node CRUD

Edge CRUD

Built for speed and clarity.

📦 Reference Dataset Included

Your project ships with a ready-made dataset:

🟡 Pokémon Dataset (Reference Example)

Stored in:

data/nodes.json data/edges.json

Used only for demonstration:

Vector similarity between Pokémon descriptions

Graph relations (evolutions, types, etc.)

Great for showing hybrid improvements

🌍 Custom Example Nodes Added

We also added real-world knowledge nodes to prove generality:

🗼 The Eiffel Tower

Node contains:

Description of the tower

Metadata (category: landmark)

Embedding vector

Automatically generated relationships (e.g., Paris, Gustave Eiffel)

👨‍🔬 Albert Einstein

Sample text:

“Albert Einstein developed the theory of relativity and won the Nobel Prize in Physics.”

System automatically creates edges:

Einstein —developed→ Relativity

Einstein —won→ Nobel Prize

This demonstrates: ✔ Unstructured text ingestion ✔ Automatic graph construction ✔ Semantic + graph hybrid relevance

🚀 How to Run (Local Setup)

Install dependencies pip install -r requirements.txt

Start backend server uvicorn main:app --reload

Server runs at:

http://127.0.0.1:8000

Launch the Web UI
Open this file in your browser:

ui/index.html

That's it.

🧪 Demo Instructions (For Judges)

Follow these steps to verify the system:

A. Vector Search Test

In UI → "Vector Search" tab Query:

electric fast pokemon

Expected:

Pikachu and Raichu appear at top.

Try general data:

famous scientist

Expected:

Albert Einstein shows up because of semantic relevance.

B. Graph Search Test

In "Graph Search" tab, set:

start_id = 803 (Eiffel Tower)

depth = 2

Expected:

Returns nodes connected to Eiffel Tower (e.g., Paris, Gustave Eiffel).

C. Hybrid Search Test

Search:

physics nobel prize

with:

α = 0.6

graph depth = 2

Expected:

Einstein ranked high because:

Vector similarity (physics concepts)

Graph proximity (relations like “won Nobel Prize”).

D. CRUD Test Create Node

POST via UI:

Name: Paris Text: Paris is the capital of France. It contains the Eiffel Tower. Metadata: {"category":"city"}

Expected:

Node created

Auto-generated edge: Paris —contains→ Eiffel Tower

Create Edge source = 803 (Eiffel Tower) target = 1010 (Paris) type = located_in weight = 1.0

Delete Node

Delete ID = Paris Edges referencing Paris are automatically removed.

🧪 Hackathon Test Case Coverage (Confirmed)

This project fully satisfies all required test cases from Devfolio, including:

Vector-only top-k

Graph-only BFS

Hybrid merge

Cascade delete

Update node + re-embed

Unstructured text ingestion

Automatic relationship extraction

Weighted graph traversal

Endpoint correctness & error handling

Everything is fully local and no external DBs are used.

🎯 Why This Submission Stands Out

Entire database engine is custom-built

Clean architecture with modular design

Hybrid scoring shows superior relevance over vector-only systems

NLP relationship extraction transforms unstructured text into a knowledge graph

Beautiful dark-mode UI

Production-ready API structure

👥 Team Catalyst Crew

Built with passion, innovation, and absolutely no reliance on pre-built vector DBs/grapDBs. A fully original, scratch-built hybrid AI retrieval engine.
