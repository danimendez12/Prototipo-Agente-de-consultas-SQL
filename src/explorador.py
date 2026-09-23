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
import json
import os
import pickle

import networkx as nx
from sentence_transformers import SentenceTransformer

from src.build_graph import node_to_text
from src.project_paths import resolve_artifact_path, resolve_graph_path


class Explorer:
    def __init__(self, graph: nx.DiGraph, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.graph = graph
        self.model = SentenceTransformer(model_name)
        self.table_names = list(graph.nodes())

        graph_path = resolve_graph_path()
        cache_path = resolve_artifact_path("embeddings.pkl")
        cache_is_fresh = cache_path.exists() and graph_path.exists() and cache_path.stat().st_mtime >= graph_path.stat().st_mtime

        if cache_is_fresh:
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                self.table_names = payload.get("table_names", self.table_names)
                self.table_embeddings = payload.get("table_embeddings")
                self.col_embeddings = payload.get("col_embeddings")
                self.col_to_table = payload.get("col_to_table", [])
                self.example_embeddings = payload.get("example_embeddings")
                self.example_to_table = payload.get("example_to_table", [])
                self.col_texts = payload.get("col_texts", [])
                self.example_texts = payload.get("example_texts", [])
                if self.table_embeddings is not None:
                    return
            except Exception:
                pass

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

        payload = {
            "table_names": self.table_names,
            "table_embeddings": self.table_embeddings,
            "col_texts": self.col_texts,
            "col_to_table": self.col_to_table,
            "col_embeddings": self.col_embeddings,
            "example_texts": self.example_texts,
            "example_to_table": self.example_to_table,
            "example_embeddings": self.example_embeddings,
        }
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception:
            pass

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

    def _load_best_retrieval_params(self) -> dict:
        path = resolve_artifact_path("best_retrieval_params.json")
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _apply_thresholds(
        self,
        scores: dict,
        top_k=None,
        min_score=None,
        relative_gap=None,
        expand_hops=None,
        neighbor_relative_gap=None,
        neighbor_floor_ceiling=None,
    ) -> tuple[list, set]:
        params = self._load_best_retrieval_params()
        top_k = top_k if top_k is not None else params.get("top_k", 5)
        min_score = min_score if min_score is not None else params.get("min_score", 0.55)
        relative_gap = relative_gap if relative_gap is not None else params.get("relative_gap", 0.85)
        expand_hops = expand_hops if expand_hops is not None else params.get("expand_hops", 1)
        neighbor_relative_gap = neighbor_relative_gap if neighbor_relative_gap is not None else params.get("neighbor_relative_gap", 0.7)
        neighbor_floor_ceiling = neighbor_floor_ceiling if neighbor_floor_ceiling is not None else params.get("neighbor_floor_ceiling", 0.5)

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
        neighbor_floor = min(best_score * neighbor_relative_gap, neighbor_floor_ceiling)

        expanded = set(top_tables)
        for table in top_tables:
            for _ in range(expand_hops):
                for neighbor in list(self.graph.successors(table)) + list(self.graph.predecessors(table)):
                    if scores.get(neighbor, 0) >= neighbor_floor:
                        expanded.add(neighbor)
        return top_tables, expanded

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
        relative_gap: float | None = None,
        expand_hops: int | None = None,
        neighbor_relative_gap: float | None = None,
        neighbor_floor_ceiling: float | None = None,
    ) -> dict:
        scores = self._combined_scores(question)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_tables, expanded = self._apply_thresholds(
            scores,
            top_k=top_k,
            min_score=min_score,
            relative_gap=relative_gap,
            expand_hops=expand_hops,
            neighbor_relative_gap=neighbor_relative_gap,
            neighbor_floor_ceiling=neighbor_floor_ceiling,
        )

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