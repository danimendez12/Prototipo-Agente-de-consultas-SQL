"""
Stage 3: navigation graph construction.
Nodes = tables (with their enriched descriptions and columns).
Edges = foreign-key relationships with cardinality many_to_one.
We use networkx in memory, as defined in the architecture for moderate-sized schemas
(without needing a dedicated graph database).
"""
import json
import sys
import pickle
from pathlib import Path

import networkx as nx
from sentence_transformers import SentenceTransformer

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.project_paths import resolve_artifact_path


def build_graph(catalog: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    print(f"Building graph from catalog (n={len(catalog)} tables)...")

    for table_name, info in catalog.items():
        if info["status"] != "active":
            continue
        g.add_node(
            table_name,
            description=info["description"],
            columns=info["columns"],
            row_count=info["row_count"],
            example_questions=info.get("example_questions", []),
        )

    for table_name, info in catalog.items():
        if info["status"] != "active":
            continue
        for fk in info["foreign_keys"]:
            g.add_edge(
                table_name,
                fk["to_table"],
                via=fk["from_column"],
                cardinality="many_to_one",
            )

    return g


def node_to_text(table_name: str, node_data: dict) -> str:
    """Concatenates name + description + columns for indexing (plain text)."""
    col_names = ", ".join(c["name"] for c in node_data["columns"])
    col_descs = " ".join(
        c.get("description", "") for c in node_data["columns"] if c.get("description")
    )
    return f"{table_name}. {node_data['description']} Columns: {col_names}. {col_descs}"


def build_embedding_index(graph: nx.DiGraph, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> dict:
    """Precompute the schema embeddings so Explorer can reuse them across runs."""
    model = SentenceTransformer(model_name)

    table_names = list(graph.nodes())
    table_texts = [node_to_text(t, graph.nodes[t]) for t in table_names]
    table_embeddings = model.encode(table_texts, normalize_embeddings=True)

    col_texts, col_to_table = [], []
    for t in table_names:
        for col in graph.nodes[t]["columns"]:
            desc = col.get("description", "")
            text = f"{t}.{col['name']}: {desc}" if desc else f"{t}.{col['name']}"
            col_texts.append(text)
            col_to_table.append(t)
    col_embeddings = model.encode(col_texts, normalize_embeddings=True) if col_texts else None

    example_texts, example_to_table = [], []
    for t in table_names:
        for eq in graph.nodes[t].get("example_questions", []):
            example_texts.append(eq)
            example_to_table.append(t)
    example_embeddings = model.encode(example_texts, normalize_embeddings=True) if example_texts else None

    return {
        "table_names": table_names,
        "table_embeddings": table_embeddings,
        "col_texts": col_texts,
        "col_to_table": col_to_table,
        "col_embeddings": col_embeddings,
        "example_texts": example_texts,
        "example_to_table": example_to_table,
        "example_embeddings": example_embeddings,
    }


if __name__ == "__main__":
    input_path = resolve_artifact_path("catalog_enriched_llm.json")
    output_path = resolve_artifact_path("graph.pkl")

    with open(input_path, encoding="utf-8") as f:
        catalog = json.load(f)

    graph = build_graph(catalog)
    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print("\nEdges (relationships):")
    for u, v, data in graph.edges(data=True):
        print(f"  {u} --[{data['via']}]--> {v}")

    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"\nGraph saved -> {output_path}")