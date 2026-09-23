"""Orchestration for the natural-language database query pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.agent_ejecutor import AgentEjecutor
from src.agent_explorador_langchain import AgentExplorerLC
from src.agent_generator_langchain import SQLQueryGeneratorAgent
from src.explorador import Explorador
from src.project_paths import resolve_graph_path
from src.services.security_check import evaluate_query


class QueryPipeline:
    """Runs exploration, SQL generation, validation, and execution in order."""

    def __init__(
        self,
        explorer: Explorador,
        explorer_agent: AgentExplorerLC,
        generator: SQLQueryGeneratorAgent,
        executor: AgentEjecutor,
    ):
        self.explorer = explorer
        self.explorer_agent = explorer_agent
        self.generator = generator
        self.executor = executor

    @classmethod
    def from_defaults(
        cls,
        provider: str = "groq",
        explorer_model: str = "openai/gpt-oss-120b",
        generator_model: str = "openai/gpt-oss-120b",
        executor_model: str = "openai/gpt-oss-20b",
        graph_path: Path | None = None,
    ) -> "QueryPipeline":
        path = graph_path or resolve_graph_path()
        with path.open("rb") as graph_file:
            graph = pickle.load(graph_file)
        explorer = Explorador(graph)
        return cls(
            explorer=explorer,
            explorer_agent=AgentExplorerLC(explorer, provider, explorer_model),
            generator=SQLQueryGeneratorAgent(provider, generator_model),
            executor=AgentEjecutor(provider=provider, model_name=executor_model),
        )

    def run(self, question: str) -> dict:
        question = question.strip()
        if not question:
            return {"success": False, "stage": "input", "error": "Escribe una pregunta."}

        exploration = self.explorer_agent.retrieve(question)
        selected_tables = exploration.get("tables", [])
        selected_context = self._schema_for_tables(selected_tables)
        if not selected_context:
            return {
                "success": False,
                "stage": "exploration",
                "error": exploration.get("reasoning", "No se encontraron tablas relevantes."),
                "exploration": exploration,
            }

        generated = self.generator.generate_sql_query(
            question, selected_context, exploration.get("reasoning", "")
        )
        if not generated.get("success"):
            return {
                "success": False,
                "stage": "generation",
                "error": generated.get("error", "No se pudo generar la consulta."),
                "exploration": exploration,
                "generation": generated,
            }

        sql = generated["sql"]
        valid, validation_error = evaluate_query(sql)
        if not valid:
            return {
                "success": False,
                "stage": "validation",
                "error": validation_error,
                "exploration": exploration,
                "generation": generated,
            }

        execution = self.executor.run(question, sql)
        return {
            "success": execution.get("success", False),
            "stage": "execution" if execution.get("success") else "execution",
            "error": execution.get("error"),
            "exploration": exploration,
            "generation": generated,
            "execution": execution,
        }

    def _schema_for_tables(self, table_names: list[str]) -> dict:
        """Build complete generator context from the graph, not thresholded retrieval."""
        context = {}
        for table in table_names:
            if table not in self.explorer.graph:
                continue
            data = self.explorer.graph.nodes[table]
            context[table] = {
                "description": data.get("description", ""),
                "columns": [column["name"] for column in data.get("columns", [])],
                "foreign_keys": [
                    {"via": details["via"], "to": target}
                    for _, target, details in self.explorer.graph.out_edges(table, data=True)
                ],
            }
        return context
