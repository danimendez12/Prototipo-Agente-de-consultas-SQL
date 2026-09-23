"""Business logic shared by the Explorer agent adapters."""


def search_tables(explorer, query: str, top_n: int = 6) -> list:
    """Return the highest-scoring schema tables for a natural-language query."""
    scores = explorer._combined_scores(query)
    ranked = sorted(scores.items(), key=lambda item: -item[1])[:top_n]
    return [
        {
            "table": table,
            "score": round(float(score), 3),
            "description": explorer.graph.nodes[table]["description"],
            "columns": list(explorer.graph.nodes[table].get("columns", [])),
        }
        for table, score in ranked
    ]


def table_neighbors(explorer, table: str) -> dict:
    """Return the tables connected to ``table`` by a foreign-key edge."""
    if table not in explorer.graph:
        return {"error": f"table '{table}' does not exist in the schema"}
    neighbors = list(explorer.graph.successors(table)) + list(
        explorer.graph.predecessors(table)
    )
    return {"neighbors": neighbors}
