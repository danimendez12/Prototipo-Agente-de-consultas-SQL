"""
Stage 4 (indexed) + Stage 5 (Explorer).

Combines three similarity signals per table by taking the maximum of each:
  1. Full-table embedding (table name + description + columns)
  2. Individual-column embedding (useful for field-specific questions)
  3. Example-question embedding (question-to-question often works better than
     question-to-description)

Variable top-k: instead of always returning exactly `top_k` tables,
we cut by absolute and relative thresholds relative to the best score to avoid
forcing filler tables when the question only needs 1-2.

Filtered graph expansion: a neighbor is added only when its own similarity score
exceeds `neighbor_min_score`, not unconditionally.
"""
import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer

from src.build_graph import node_to_text
from src.project_paths import resolve_graph_path


class Explorer:
    def __init__(self, graph: nx.DiGraph, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.graph = graph
        self.table_names = list(graph.nodes())
        self.model = SentenceTransformer(model_name)

        # Signal 1: full table
        table_texts = [node_to_text(t, graph.nodes[t]) for t in self.table_names]
        self.table_embeddings = self.model.encode(table_texts, normalize_embeddings=True)

        # Signal 2: per-column
        self.col_texts, self.col_to_table = [], []
        for t in self.table_names:
            for col in graph.nodes[t]["columns"]:
                desc = col.get("description", "")
                text = f"{t}.{col['name']}: {desc}" if desc else f"{t}.{col['name']}"
                self.col_texts.append(text)
                self.col_to_table.append(t)
        self.col_embeddings = (
            self.model.encode(self.col_texts, normalize_embeddings=True)
            if self.col_texts else None
        )

        # Signal 3: example questions
        self.example_texts, self.example_to_table = [], []
        for t in self.table_names:
            for eq in graph.nodes[t].get("example_questions", []):
                self.example_texts.append(eq)
                self.example_to_table.append(t)
        self.example_embeddings = (
            self.model.encode(self.example_texts, normalize_embeddings=True)
            if self.example_texts else None
        )

    def _combined_scores(self, question: str) -> dict:
        q_emb = self.model.encode([question], normalize_embeddings=True)

        table_sims = (self.table_embeddings @ q_emb.T).flatten()
        scores = {t: table_sims[i] for i, t in enumerate(self.table_names)}

        if self.col_embeddings is not None:
            col_sims = (self.col_embeddings @ q_emb.T).flatten()
            for table, s in zip(self.col_to_table, col_sims):
                scores[table] = max(scores[table], s)

        if self.example_embeddings is not None:
            ex_sims = (self.example_embeddings @ q_emb.T).flatten()
            for table, s in zip(self.example_to_table, ex_sims):
                scores[table] = max(scores[table], s)

        return scores

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.55,
        relative_gap: float = 0.85,
        expand_hops: int = 1,
        neighbor_relative_gap: float = 0.7,
        neighbor_floor_ceiling: float = 0.5,
    ) -> dict:
        scores = self._combined_scores(question)
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        if not ranked or ranked[0][1] < min_score:
            top_tables = [ranked[0][0]] if ranked else []
        else:
            best_score = ranked[0][1]
            top_tables = [
                t for t, s in ranked[:top_k]
                if s >= min_score and s >= best_score * relative_gap
            ]

        best_score = ranked[0][1] if ranked else 0
        # The ceiling prevents the threshold from becoming unrealistically strict when the
        # main table has a very high confidence score.
        neighbor_floor = min(best_score * neighbor_relative_gap, neighbor_floor_ceiling)

        expanded = set(top_tables)
        for table in top_tables:
            for _ in range(expand_hops):
                for neighbor in list(self.graph.successors(table)) + list(self.graph.predecessors(table)):
                    if scores.get(neighbor, 0) >= neighbor_floor:
                        expanded.add(neighbor)

        context_package = {
            "question": question,
            "retrieved_by_similarity": top_tables,
            "expanded_with_graph": sorted(expanded),
            "scores": {t: round(float(s), 3) for t, s in ranked},
            "tables": {},
        }
        for t in expanded:
            data = self.graph.nodes[t]
            context_package["tables"][t] = {
                "description": data["description"],
                "columns": [c["name"] for c in data["columns"]],
                "foreign_keys": [
                    {"via": d["via"], "to": v}
                    for _, v, d in self.graph.out_edges(t, data=True)
                ],
            }
        return context_package


Explorador = Explorer


if __name__ == "__main__":
    graph_path = resolve_graph_path()
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    explorer = Explorer(graph)
    for q in [
        "Which are the top 5 music genres by songs sold?",
        "Which employee has the most assigned customers?",
        "List songs from the largest playlist",
    ]:
        ctx = explorer.retrieve(q)
        print(f"\n{q}")
        print(f"  similarity: {ctx['retrieved_by_similarity']}")
        print(f"  expanded: {ctx['expanded_with_graph']}")