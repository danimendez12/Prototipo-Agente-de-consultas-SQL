"""
Stage 3: navigation graph construction.
Nodes = tables (with their enriched descriptions and columns).
Edges = foreign-key relationships with cardinality many_to_one.
We use networkx in memory, as defined in the architecture for moderate-sized schemas
(without needing a dedicated graph database).
"""
import json
import sys
import networkx as nx
import pickle
from pathlib import Path

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