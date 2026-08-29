"""
Agente Explorador — envuelve la herramienta de retrieval (Explorador,
puramente vectorial + grafo, ya afinada) con razonamiento de un LLM.

Diferencia con explorador.py:
- explorador.py: pipeline FIJO (embeddings -> umbral relativo -> expansión
  por grafo con techo). Ningún paso "decide" nada, solo aplica reglas.
- AgentExplorador: el LLM decide dinámicamente qué buscar, si reformular
  la búsqueda, si consultar vecinos del grafo, y qué tablas incluir en
  la respuesta final — el tipo de juicio que un umbral fijo no puede dar
  (recuerda el caso "Artist vs Invoice casi empatados en score": un
  número no puede razonar sobre eso, un LLM sí).

Costo y latencia: cada pregunta ahora cuesta tokens reales de API y
tarda segundos, no milisegundos (varias idas y vueltas a Claude en vez
de una operación local). Esto es un trade-off deliberado, no un error.
"""
import json
from anthropic import Anthropic

client = Anthropic()

SEARCH_TOOL = {
    "name": "buscar_tablas",
    "description": (
        "Busca tablas del esquema semánticamente relacionadas con un texto. "
        "Devuelve las tablas más relevantes con su score de similitud y descripción."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Texto a buscar (la pregunta original o una reformulación/sinónimo)",
            },
            "top_n": {"type": "integer", "description": "Cuántos resultados devolver", "default": 6},
        },
        "required": ["query"],
    },
}

NEIGHBORS_TOOL = {
    "name": "vecinos_de_tabla",
    "description": (
        "Devuelve las tablas directamente conectadas por foreign key a una tabla dada. "
        "Útil para encontrar tablas de unión necesarias en un JOIN que la búsqueda "
        "semántica pura no detecta bien (ej. tablas muchos-a-muchos)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"table": {"type": "string"}},
        "required": ["table"],
    },
}

FINAL_TOOL = {
    "name": "entregar_tablas_finales",
    "description": (
        "Entrega la lista final y definitiva de tablas necesarias para responder "
        "la pregunta del usuario. Llama a esta herramienta UNA SOLA VEZ, cuando "
        "ya estés seguro de tu respuesta."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tablas": {"type": "array", "items": {"type": "string"}},
            "razonamiento": {
                "type": "string",
                "description": "Breve explicación de por qué estas tablas y no otras",
            },
        },
        "required": ["tablas", "razonamiento"],
    },
}

SYSTEM_PROMPT = """Eres el agente Explorador de un sistema de consultas a bases de datos.

Tu trabajo: dado una pregunta en lenguaje natural, determinar EXACTAMENTE
qué tablas de la base de datos son necesarias para responderla — ni de más
ni de menos. Esta lista se le entregará después a un Generador de SQL: una
tabla de más puede hacerlo generar un JOIN innecesario, y una tabla de
menos hace imposible responder la pregunta.

Herramientas disponibles:
- buscar_tablas: búsqueda semántica. Puedes llamarla más de una vez con
  reformulaciones distintas si la primera búsqueda no te da resultados
  convincentes (por ejemplo, si la pregunta usa una palabra que quizás no
  coincide con el vocabulario de las descripciones del esquema).
- vecinos_de_tabla: úsala cuando sospeches que la pregunta requiere un JOIN
  y quieras confirmar qué tabla intermedia lo conecta.
- entregar_tablas_finales: tu respuesta final. Llámala una sola vez, cuando
  ya tengas suficiente evidencia.

Sé eficiente: no llames a buscar_tablas más de 2-3 veces por pregunta.
Si dos tablas candidatas tienen scores muy parecidos, usa tu criterio sobre
cuál es realmente relevante para el SIGNIFICADO de la pregunta, no solo el
score numérico — para eso existes, un puntaje de similitud no distingue
matices de significado tan bien como tú."""


class AgentExplorador:
    def __init__(self, explorador, model="claude-sonnet-5", max_tool_calls=6):
        self.explorador = explorador  # instancia de explorador.py, ya indexada
        self.model = model
        self.max_tool_calls = max_tool_calls

    def _execute_tool(self, name, tool_input):
        if name == "buscar_tablas":
            scores = self.explorador._combined_scores(tool_input["query"])
            ranked = sorted(scores.items(), key=lambda x: -x[1])[: tool_input.get("top_n", 6)]
            return [
                {
                    "tabla": t,
                    "score": round(float(s), 3),
                    "descripcion": self.explorador.graph.nodes[t]["description"],
                }
                for t, s in ranked
            ]
        elif name == "vecinos_de_tabla":
            table = tool_input["table"]
            if table not in self.explorador.graph:
                return {"error": f"tabla '{table}' no existe en el esquema"}
            vecinos = list(self.explorador.graph.successors(table)) + list(
                self.explorador.graph.predecessors(table)
            )
            return {"vecinos": vecinos}
        else:
            return {"error": f"herramienta desconocida: {name}"}

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
                break  # el modelo respondió sin usar ninguna tool (no debería pasar)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in tool_use_blocks:
                if block.name == "entregar_tablas_finales":
                    print("\n[Explorador] Tablas finales entregadas al Generador:")
                    for table in block.input["tablas"]:
                        print(f"  - {table}")
                    print(f"  Razonamiento: {block.input['razonamiento']}")
                    return {
                        "tables": block.input["tablas"],
                        "reasoning": block.input["razonamiento"],
                        "tool_calls": tool_calls_made,
                        "usage": usage_total,
                        "trace": trace,
                    }

                result = self._execute_tool(block.name, block.input)
                trace.append({"tool": block.name, "input": block.input, "result": result})

                if block.name == "buscar_tablas":
                    print("\n[Explorador] Tablas recibidas del explorador:")
                    for item in result:
                        print(
                            f"  - {item['tabla']}: score={item['score']} | "
                            f"descripcion={item['descripcion'][:100]}"
                        )
                elif block.name == "vecinos_de_tabla":
                    print(f"\n[Explorador] Vecinos de '{block.input['table']}': {result.get('vecinos', [])}")

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
            "reasoning": "límite de tool calls alcanzado sin respuesta final",
            "tool_calls": tool_calls_made,
            "usage": usage_total,
            "trace": trace,
        }


if __name__ == "__main__":
    import pickle
    from src.explorador import Explorador
    from src.project_paths import resolve_graph_path

    with open(resolve_graph_path(), "rb") as f:
        graph = pickle.load(f)
    explorador = Explorador(graph)
    agent = AgentExplorador(explorador)

    for q in [
        "¿Cuánto dinero se ha generado por cada género musical?",
        "¿Qué clientes tienen como representante a un empleado que fue contratado antes de 2003?",
    ]:
        print(f"\n❓ {q}")
        result = agent.retrieve(q, verbose=True)
        print(f"  Tablas: {result['tables']}")
        print(f"  Razonamiento: {result['reasoning']}")
        print(f"  Tool calls: {result['tool_calls']}  |  Tokens: {result['usage']}")