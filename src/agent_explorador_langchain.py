"""
Explorer agent via LangChain + open-source model (gpt-oss-120b on Groq).

Same architecture as agent_explorador.py (vector search + graph as tools,
LLM decides what to call and when to stop), but using LangChain as the
orchestration layer and an open-weight model instead of Claude.

Installation:
    pip install langchain langchain-groq langchain-core
    (for Ollama instead of Groq: pip install langchain-ollama, then run
     `ollama pull llama3.1` locally first)

Configuration:
    export GROQ_API_KEY="..."   (free at https://console.groq.com)
"""
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_attempt
from src.services.agent_services import is_rate_limit_error, invoke_with_backoff, _try_salvage_from_error, build_llm




@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)



def build_tools(explorer):
    """Wraps the already-tuned retrieval tool as LangChain tools."""

    @tool
    def search_tables(query: str, top_n: int = 6) -> list:
        """Searches the schema for tables semantically related to the query."""
        scores = explorer._combined_scores(query)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

        return [
            {
                "table": t,
                "score": round(float(s), 3),
                "description": explorer.graph.nodes[t]["description"],
                "columns": list(explorer.graph.nodes[t].get("columns", [])),
            }
            for t, s in ranked
        ]

    @tool
    def table_neighbors(table: str) -> dict:
        """Returns the tables directly connected by a foreign key to a given table."""
        if table not in explorer.graph:
            return {"error": f"table '{table}' does not exist in the schema"}
        neighbors = list(explorer.graph.successors(table)) + list(
            explorer.graph.predecessors(table)
        )
        return {"neighbors": neighbors}

    @tool
    def deliver_final_tables(tables: list, reasoning: str) -> str:
        """MANDATORY FINAL ACTION.

        Once you have finished analyzing the question, you MUST call this tool to deliver the final tables.

        Do not respond with free text.
        Do not call any tool named commentary.
        The exact tool name is deliver_final_tables.
        """
        return "delivered"

    return [search_tables, table_neighbors, deliver_final_tables]


SYSTEM_PROMPT = """You are the Explorer agent for a database query system.

Your job: given a natural-language question, determine EXACTLY which database tables are
required to answer it — neither more nor fewer.

=== CRITICAL INSTRUCTIONS ===
1. ALWAYS start by calling search_tables() with the user's question.
2. Analyze the results: do you need more tables? Does it require a JOIN?
3. If you need JOINs, use table_neighbors() to discover them.
4. Once you have the definitive list, call deliver_final_tables() EXACTLY ONCE.
5. NEVER answer the user directly without calling a tool first.

=== AVAILABLE TOOLS ===
- search_tables(query, top_n): Semantic search. You may call it more than once with different
  reformulations if the first search does not provide convincing results.

- table_neighbors(table): Returns tables connected by foreign keys. Use it when you suspect the
  question requires a JOIN.

- deliver_final_tables(tables, reasoning): Your final answer. It must be called exactly once at the end.

4. deliver_final_tables is the ONLY valid way to finish.
5. DO NOT answer with free text.
6. DO NOT invent tool names.
7. DO NOT use "commentary" as a tool.
8. Do not include SQL. Only select the tables needed.

The final call must have exactly this conceptual structure:

deliver_final_tables(
    tables=[...],
    reasoning="..."
)
"""


class AgentExplorerLC:
    def __init__(self, explorer, provider="groq", model_name="openai/gpt-oss-120b", max_iterations=8):
        self.explorer = explorer
        self.tools = build_tools(explorer)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm = build_llm(provider, model_name).bind_tools(self.tools)
        self.max_iterations = max_iterations

    def retrieve(self, question: str, verbose: bool = False) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
        iterations = 0
        trace = []

        while iterations < self.max_iterations:
            try:
                ai_msg = invoke_with_backoff(self.llm, messages)
            except Exception as e:
                salvaged = _try_salvage_from_error(e)
                if salvaged:
                    print(f"  [recovery] the model misformatted the response; recovered from error: {salvaged}")
                    return {
                        "tables": salvaged["tables"],
                        "reasoning": salvaged["reasoning"],
                        "tool_calls": iterations,
                        "trace": trace,
                    }
                messages.append(HumanMessage(
                    content=(
                        "Your last response was rejected because it used an invalid tool name. "
                        "The ONLY valid tool names are: search_tables, table_neighbors, deliver_final_tables. "
                        "Do not invent names such as 'json' or 'commentary'. Please try again."
                    )
                ))
                iterations += 1
                continue

            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                break

            for tc in ai_msg.tool_calls:
                if tc["name"] == "deliver_final_tables":
                    print("\n[Explorer] Final tables delivered to the SQL generator:")
                    for table in tc["args"]["tables"]:
                        print(f"  - {table}")
                    print(f"  Reasoning: {tc['args']['reasoning']}")
                    return {
                        "tables": tc["args"]["tables"],
                        "reasoning": tc["args"]["reasoning"],
                        "tool_calls": iterations,
                        "trace": trace,
                    }

                tool_fn = self.tools_by_name[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                trace.append({"tool": tc["name"], "input": tc["args"], "result": result})

                if tc["name"] == "search_tables":
                    print("\n[Explorer] Tables received from the explorer:")
                    for item in result:
                        print(
                            f"  - {item['table']}: score={item['score']} | "
                            f"description={item['description'][:100]}"
                        )
                elif tc["name"] == "table_neighbors":
                    print(f"\n[Explorer] Neighbors of '{tc['args']['table']}': {result.get('neighbors', [])}")

                if verbose:
                    print(f"  [{tc['name']}] {tc['args']} -> {result}")

                messages.append(
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"])
                )
                iterations += 1

        return {
            "tables": [],
            "reasoning": "iteration limit reached without a final answer",
            "tool_calls": iterations,
            "trace": trace,
        }


if __name__ == "__main__":
    import pickle
    from src.explorador import Explorador
    from src.project_paths import resolve_graph_path

    try:
        graph_path = resolve_graph_path()
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        explorer = Explorador(graph)
        agent = AgentExplorerLC(
            explorer,
            provider="groq",
            model_name="openai/gpt-oss-120b"
        )

        for q in [
            "How much revenue has been generated per music genre?",
            "Which customers have a representative employee hired before 2003?",
        ]:
            print(f"\n❓ {q}")
            result = agent.retrieve(q, verbose=True)
            print(f"  Tables: {result['tables']}")
            print(f"  Reasoning: {result['reasoning']}")
            print(f"  Tool calls: {result['tool_calls']}")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)