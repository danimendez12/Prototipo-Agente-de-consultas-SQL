"""
Agente Generador de SQL — versión con salida estructurada.

Cambio clave frente a la primera versión: en vez de pedir "devuelve solo
el SQL en texto plano" (ambiguo cuando el modelo decide declinar y
devuelve un mensaje de error en su lugar), forzamos una única tool de
entrega con dos campos mutuamente excluyentes: sql o error. Esto elimina
la ambigüedad de "¿esto que me llegó es SQL real o es el modelo rindiéndose
en texto libre?" — el mismo patrón que ya usamos en el Explorador con
entregar_tablas_finales.
"""
import json
import re
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src.services.agent_services import invoke_with_backoff, build_llm


SYSTEM_PROMPT = """Eres un generador de consultas SQL de solo lectura.

Recibes una pregunta en lenguaje natural, una lista de tablas ya
seleccionadas por otro agente, y su razonamiento de por qué esas tablas
son necesarias.

Reglas estrictas:
- Genera EXCLUSIVAMENTE sentencias SELECT. Nunca INSERT, UPDATE, DELETE,
  DROP, ALTER, ni ninguna otra operación de escritura.
- Usa SOLO las tablas y columnas de la lista entregada. No asumas la
  existencia de tablas o columnas que no te dieron explícitamente.
- Si la pregunta no se puede responder con las tablas disponibles, no
  inventes una consulta — usa el campo "error" para explicar por qué.
- Nunca envuelvas el SQL en bloques de markdown ni agregues explicación
  fuera de la herramienta de entrega.

Debes entregar tu respuesta llamando SIEMPRE a la herramienta entregar_sql."""


@tool
def entregar_sql(sql: str = "", error: str = "") -> str:
    """Entrega el resultado final del Generador. Llena EXACTAMENTE uno de
    los dos campos: 'sql' si lograste construir la consulta, o 'error' si
    no es posible con las tablas dadas. Nunca llenes ambos ni dejes ambos vacíos."""
    return "entregado"


def _clean_sql(text: str) -> str:
    """Red de seguridad por si el modelo igual envuelve el SQL en markdown."""
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


class SQLQueryGeneratorAgent:
    def __init__(self, provider="groq", model_name="openai/gpt-oss-120b"):
        self.provider = provider
        self.model_name = model_name
        self.llm = build_llm(provider, model_name).bind_tools([entregar_sql], tool_choice="entregar_sql")

    def generate_sql_query(self, user_query: str, tables_context: dict, reasoning: str) -> dict:
        """
        tables_context: dict de {nombre_tabla: {"description":..., "columns": [...], "foreign_keys": [...]}}
        — el mismo formato que ya arma Explorador.retrieve() en context_package["tables"].
        Pasar solo nombres de tabla (sin columnas) hace que el modelo tenga
        que ADIVINAR nombres de columna, lo que produce SQL con columnas
        inexistentes aunque la lógica de negocio sea correcta.
        """
        schema_lines = []
        for table, info in tables_context.items():
            cols = ", ".join(info.get("columns", []))
            fks = "; ".join(f"{fk['via']} -> {fk['to']}" for fk in info.get("foreign_keys", []))
            line = f"- {table}({cols})"
            if fks:
                line += f"  [FKs: {fks}]"
            schema_lines.append(line)
        schema_text = "\n".join(schema_lines)

        prompt = f"""Pregunta del usuario: {user_query}

Esquema de las tablas seleccionadas por el agente Explorador:
{schema_text}

Razonamiento del Explorador: {reasoning}

IMPORTANTE: usa EXACTAMENTE los nombres de columna listados arriba. No
asumas convenciones de nombres genéricas (como snake_case) si no coinciden
con lo mostrado."""

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

        try:
            response = invoke_with_backoff(self.llm, messages)
        except Exception as e:
            return {"success": False, "sql": "", "error": f"Error de API: {e}"}

        if not response.tool_calls:
            # Red de seguridad: si el modelo igual respondió en texto libre
            # (raro con tool_choice forzado, pero ocurre con algunos modelos
            # open-weight), intenta rescatar un SELECT del texto plano antes
            # de declarar la pregunta como fallida sin más.
            match = re.search(r"(SELECT\b.*?;)", response.content, re.IGNORECASE | re.DOTALL)
            if match:
                return {"success": True, "sql": _clean_sql(match.group(1)), "error": ""}
            return {"success": False, "sql": "", "error": "El modelo no usó la herramienta entregar_sql"}

        args = response.tool_calls[0]["args"]
        error = args.get("error", "").strip()
        sql = _clean_sql(args.get("sql", ""))

        if error:
            return {"success": False, "sql": "", "error": error}
        if not sql:
            return {"success": False, "sql": "", "error": "El modelo no entregó SQL ni error"}
        return {"success": True, "sql": sql, "error": ""}


if __name__ == "__main__":
    agent = SQLQueryGeneratorAgent(provider="groq", model_name="openai/gpt-oss-120b")
    result = agent.generate_sql_query(
        "¿Cuánto dinero se ha generado por cada género musical?",
        ["Genre", "Track", "InvoiceLine"],
        "Se necesitan estas tablas para relacionar género, canción y venta.",
    )
    print(result)