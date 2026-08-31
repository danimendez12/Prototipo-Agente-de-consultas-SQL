"""
Explorer agent — wraps the retrieval tool (Explorer, purely vector-based + graph,
already tuned) with LLM reasoning.

Difference from explorer.py:
- explorer.py: fixed pipeline (embeddings -> relative threshold -> graph expansion with limits).
  No stage decides anything; it only applies rules.
- AgentExplorer: the LLM decides dynamically what to search, whether to reformulate the query,
  whether to inspect graph neighbors, and which tables to include in the final answer — the kind
  of judgment a fixed threshold cannot provide (remember the case where "Artist vs Invoice" are
  almost tied on score: a number cannot reason about that nuance, an LLM can).

Cost and latency: each question now incurs real API token costs and takes seconds instead of
milliseconds (several Claude round-trips instead of a local operation). This is a deliberate
trade-off, not a bug.
"""
import json
from anthropic import Anthropic

client = Anthropic()

SEARCH_TOOL = {
    "name": "search_tables",
    "description": (
        "Searches the schema for tables semantically related to a text query. "
        "Returns the most relevant tables with their similarity score and description."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search (the original question or a reformulation/synonym)",
            },
            "top_n": {"type": "integer", "description": "How many results to return", "default": 6},
        },
        "required": ["query"],
    },
}

NEIGHBORS_TOOL = {
    "name": "table_neighbors",
    "description": (
        "Returns the tables directly connected by a foreign key to a given table. "
        "Useful for identifying the join tables needed when a semantic search alone misses the "
        "many-to-many relationship or other necessary joins."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"table": {"type": "string"}},
        "required": ["table"],
    },
}

FINAL_TOOL = {
    "name": "deliver_final_tables",
    "description": (
        "Delivers the final, definitive set of tables needed to answer the user's question. "
        "Call this tool only once, when you are confident in your answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tables": {"type": "array", "items": {"type": "string"}},
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why these tables are needed and others are not",
            },
        },
        "required": ["tables", "reasoning"],
    },
}

SYSTEM_PROMPT = """You are the Explorer agent for a database query system.

Your job: given a natural-language question, determine EXACTLY which database tables are
required to answer it — neither more nor fewer. This list is later given to a SQL Generator:
one extra table can produce an unnecessary JOIN, while a missing table makes the question
impossible to answer.

Available tools:
- search_tables: semantic search. You may call it multiple times with different reformulations if
the first search does not produce convincing results.
- table_neighbors: use it when the question likely requires a JOIN and you want to confirm the
  intermediate table that connects the relevant entities.
- deliver_final_tables: your final answer. Call it once, when you have sufficient evidence.

Be efficient: do not call search_tables more than 2-3 times per question.
If two candidate tables have very similar scores, use your judgment about which one is truly
relevant for the meaning of the question, not just the numeric score — that is why you exist, a
similarity score does not distinguish semantic nuances as well as a human LLM can."""


class AgentExplorer:
    def __init__(self, explorer, model="claude-sonnet-5", max_tool_calls=6):
        self.explorer = explorer
        self.model = model
        self.max_tool_calls = max_tool_calls

    def _execute_tool(self, name, tool_input):
        legacy_name_map = {
            "buscar_tablas": "search_tables",
            "vecinos_de_tabla": "table_neighbors",
            "entregar_tablas_finales": "deliver_final_tables",
        }
        resolved_name = legacy_name_map.get(name, name)

        if resolved_name == "search_tables":
            scores = self.explorer._combined_scores(tool_input["query"])
            ranked = sorted(scores.items(), key=lambda x: -x[1])[: tool_input.get("top_n", 6)]
            return [
                {
                    "table": t,
                    "score": round(float(s), 3),
                    "description": self.explorer.graph.nodes[t]["description"],
                }
                for t, s in ranked
            ]
        elif resolved_name == "table_neighbors":
            table = tool_input["table"]
            if table not in self.explorer.graph:
                return {"error": f"table '{table}' does not exist in the schema"}
            neighbors = list(self.explorer.graph.successors(table)) + list(
                self.explorer.graph.predecessors(table)
            )
            return {"neighbors": neighbors}
        else:
            return {"error": f"unknown tool: {name}"}

    def retrieve(self, question: str, verbose: bool = False) -> dict:
        messages = [{"role": "user", "content": question}]
        tool_calls_made = 0
        trace = []
        usage_total = {"input_tokens": 0, "output_tokens": 0}

        while tool_calls_made < self.max_tool_calls:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL, NEIGHBORS_TOOL, FINAL_TOOL],
                messages=messages,
            )
            usage_total["input_tokens"] += response.usage.input_tokens
            usage_total["output_tokens"] += response.usage.output_tokens

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in tool_use_blocks:
                if block.name in ("deliver_final_tables", "entregar_tablas_finales"):
                    print("\n[Explorer] Final tables delivered to the SQL generator:")
                    for table in block.input["tables"]:
                        print(f"  - {table}")
                    print(f"  Reasoning: {block.input['reasoning']}")
                    return {
                        "tables": block.input["tables"],
                        "reasoning": block.input["reasoning"],
                        "tool_calls": tool_calls_made,
                        "usage": usage_total,
                        "trace": trace,
                    }

                result = self._execute_tool(block.name, block.input)
                trace.append({"tool": block.name, "input": block.input, "result": result})

                if block.name in ("search_tables", "buscar_tablas"):
                    print("\n[Explorer] Tables received from the explorer:")
                    for item in result:
                        print(
                            f"  - {item['table']}: score={item['score']} | "
                            f"description={item['description'][:100]}"
                        )
                elif block.name in ("table_neighbors", "vecinos_de_tabla"):
                    print(f"\n[Explorer] Neighbors of '{block.input['table']}': {result.get('neighbors', [])}")

                if verbose:
                    print(f"  [{block.name}] {block.input} -> {result}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                tool_calls_made += 1

            messages.append({"role": "user", "content": tool_results})

        return {
            "tables": [],
            "reasoning": "tool call limit reached without a final answer",
            "tool_calls": tool_calls_made,
            "usage": usage_total,
            "trace": trace,
        }


AgentExplorador = AgentExplorer


if __name__ == "__main__":
    import pickle
    from src.explorador import Explorador
    from src.project_paths import resolve_graph_path

    with open(resolve_graph_path(), "rb") as f:
        graph = pickle.load(f)
    explorer = Explorador(graph)
    agent = AgentExplorer(explorer)

    for q in [
        "¿Cuánto dinero se ha generado por cada género musical?",
        "¿Qué clientes tienen como representante a un empleado que fue contratado antes de 2003?",
    ]:
        print(f"\n❓ {q}")
        result = agent.retrieve(q, verbose=True)
        print(f"  Tables: {result['tables']}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Tool calls: {result['tool_calls']}  |  Tokens: {result['usage']}")