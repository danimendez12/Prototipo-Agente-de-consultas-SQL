"""
Agente Explorador vía LangChain + modelo open-source (gpt-oss-120b en Groq).

Misma arquitectura que agent_explorador.py (búsqueda vectorial + grafo como
tools, el LLM decide qué llamar y cuándo detenerse) pero usando LangChain
como capa de orquestación y un modelo de pesos abiertos en vez de Claude.


Instalación:
    pip install langchain langchain-groq langchain-core
    (para Ollama en vez de Groq: pip install langchain-ollama, y correr
     `ollama pull llama3.1` localmente primero)

Configuración:
    export GROQ_API_KEY="..."   (gratis en https://console.groq.com)
"""
import json
import os
import sys
import re
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_attempt


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))



def is_rate_limit_error(exception):
    return "rate_limit" in str(exception).lower() or "429" in str(exception)

@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def invoke_with_backoff(llm, messages):
    return llm.invoke(messages)

def _try_salvage_from_error(error) -> dict | None:
    """
    Cuando el modelo envuelve la respuesta correcta en un nombre de tool
    inventado (ej. 'json'), Groq rechaza la llamada con 400 pero el JSON
    generado sigue viniendo en el cuerpo del error. Lo recuperamos de ahí
    en vez de descartar una respuesta que en el fondo era correcta.
    """
    text = str(error)
    match = re.search(r"'failed_generation':\s*'(.*?)'\}$", text, re.DOTALL)
    if not match:
        return None
    try:
        raw = match.group(1).encode().decode("unicode_escape")
        payload = json.loads(raw)
        args = payload.get("arguments", {})
        if "tablas" in args and "razonamiento" in args:
            return args
    except Exception:
        return None
    return None


def build_llm(provider: str = "groq", model_name: str = "openai/gpt-oss-120b"):
    """
    Punto único de cambio de proveedor/modelo. Todo lo demás en este
    archivo funciona igual sin importar cuál se elija aquí, porque
    LangChain normaliza la interfaz de tool calling entre proveedores.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq
        resolved_key = os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Falta GROQ_API_KEY. Exporta la variable antes de ejecutar este script:\n"
                "  export GROQ_API_KEY='tu_clave_aqui'\n"
                "  python3 src/agent_explorador_langchain.py"
            )
        return ChatGroq(model=model_name, temperature=0, api_key=resolved_key)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)
    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    else:
        raise ValueError(f"Proveedor desconocido: {provider}")


def build_tools(explorador):
    """Envuelve la herramienta de retrieval ya afinada como tools de LangChain."""

    @tool
    def buscar_tablas(query: str, top_n: int = 6) -> list:
        """Busca tablas del esquema semánticamente relacionadas..."""

        scores = explorador._combined_scores(query)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

        return [
            {
                "tabla": t,
                "score": round(float(s), 3),
                "descripcion": explorador.graph.nodes[t]["description"],
                "columnas": list(explorador.graph.nodes[t].get("columns", [])),
            }
            for t, s in ranked
        ]

    @tool
    def vecinos_de_tabla(table: str) -> dict:
        """Devuelve las tablas directamente conectadas por foreign key a una
        tabla dada. Útil para encontrar tablas de unión necesarias en un JOIN."""
        if table not in explorador.graph:
            return {"error": f"tabla '{table}' no existe en el esquema"}
        vecinos = list(explorador.graph.successors(table)) + list(
            explorador.graph.predecessors(table)
        )
        return {"vecinos": vecinos}

    @tool
    def entregar_tablas_finales(tablas: list, razonamiento: str) -> str:
        """ÚLTIMA ACCIÓN OBLIGATORIA.

        Cuando hayas terminado de analizar la pregunta, DEBES llamar esta
        herramienta para entregar las tablas finales.

        NO respondas con texto.
        NO llames ninguna herramienta llamada commentary.
        El nombre exacto de esta herramienta es entregar_tablas_finales.
        """
        return "entregado"

    return [buscar_tablas, vecinos_de_tabla, entregar_tablas_finales]


SYSTEM_PROMPT = """Eres el agente Explorador de un sistema de consultas a bases de datos.

Tu trabajo: dado una pregunta en lenguaje natural, determinar EXACTAMENTE
qué tablas de la base de datos son necesarias para responderla — ni de más
ni de menos.

=== INSTRUCCIONES CRÍTICAS ===
1. SIEMPRE comienza llamando a buscar_tablas() con la pregunta del usuario.
2. Analiza los resultados: ¿necesitas más tablas? ¿Requiere JOIN?
3. Si necesitas JOINs, usa vecinos_de_tabla() para descubrirlas.
4. Cuando tengas la lista DEFINITIVA, llama a entregar_tablas_finales() UNA SOLA VEZ.
5. NUNCA respóndas directamente al usuario sin llamar a un tool primero.

=== HERRAMIENTAS DISPONIBLES ===
- buscar_tablas(query, top_n): Búsqueda semántica. Puedes llamarla más de una vez
  con reformulaciones distintas si la primera búsqueda no da resultados convincentes.
  
- vecinos_de_tabla(table): Devuelve tablas conectadas por foreign key.
  Úsala cuando sospeches que la pregunta requiere un JOIN.
  
- entregar_tablas_finales(tablas, razonamiento): Tu respuesta final.
  DEBE ser llamada exactamente UNA sola vez al final.

4. entregar_tablas_finales es la ÚNICA forma válida de terminar.
5. NO respondas con texto libre.
6. NO inventes nombres de herramientas.
7. NO utilices "commentary" como herramienta.
8. No incluyas SQL. Solo selecciona las tablas necesarias.

La llamada final debe tener exactamente esta estructura conceptual:

entregar_tablas_finales(
    tablas=[...],
    razonamiento="..."
)
"""


class AgentExploradorLC:
    def __init__(self, explorador, provider="groq", model_name="openai/gpt-oss-120b", max_iterations=8):
        self.explorador = explorador
        self.tools = build_tools(explorador)
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
                    print(f"  [rescate] el modelo envolvió mal la respuesta, recuperada del error: {salvaged}")
                    return {
                        "tables": salvaged["tablas"],
                        "reasoning": salvaged["razonamiento"],
                        "tool_calls": iterations,
                        "trace": trace,
                    }
                # no se pudo rescatar: le avisamos al modelo del error exacto y reintentamos
                messages.append(HumanMessage(
                    content=(
                        "Tu última respuesta fue rechazada por usar un nombre de "
                        "herramienta inválido. Las ÚNICAS herramientas válidas son: "
                        "buscar_tablas, vecinos_de_tabla, entregar_tablas_finales. "
                        "No inventes otros nombres como 'json' o 'commentary'. Intenta de nuevo."
                    )
                ))
                iterations += 1
                continue

            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                break  # el modelo respondió sin tool call (no debería pasar)

            for tc in ai_msg.tool_calls:
                if tc["name"] == "entregar_tablas_finales":
                    print("\n[Explorador] Tablas finales entregadas al Generador:")
                    for table in tc["args"]["tablas"]:
                        print(f"  - {table}")
                    print(f"  Razonamiento: {tc['args']['razonamiento']}")
                    return {
                        "tables": tc["args"]["tablas"],
                        "reasoning": tc["args"]["razonamiento"],
                        "tool_calls": iterations,
                        "trace": trace,
                    }

                tool_fn = self.tools_by_name[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                trace.append({"tool": tc["name"], "input": tc["args"], "result": result})

                if tc["name"] == "buscar_tablas":
                    print("\n[Explorador] Tablas recibidas del explorador:")
                    for item in result:
                        print(
                            f"  - {item['tabla']}: score={item['score']} | "
                            f"descripcion={item['descripcion'][:100]}"
                        )
                elif tc["name"] == "vecinos_de_tabla":
                    print(f"\n[Explorador] Vecinos de '{tc['args']['table']}': {result.get('vecinos', [])}")

                if verbose:
                    print(f"  [{tc['name']}] {tc['args']} -> {result}")

                messages.append(
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"])
                )
                iterations += 1

        return {
            "tables": [],
            "reasoning": "límite de iteraciones alcanzado sin respuesta final",
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
        explorador = Explorador(graph)
        agent = AgentExploradorLC(
            explorador,
            provider="groq",
            model_name="openai/gpt-oss-120b"
        )  # default: Groq + openai/gpt-oss-120b

        for q in [
            "¿Cuánto dinero se ha generado por cada género musical?",
            "¿Qué clientes tienen como representante a un empleado que fue contratado antes de 2003?",
        ]:
            print(f"\n❓ {q}")
            result = agent.retrieve(q, verbose=True)
            print(f"  Tablas: {result['tables']}")
            print(f"  Razonamiento: {result['reasoning']}")
            print(f"  Tool calls: {result['tool_calls']}")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)