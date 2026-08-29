"""
Etapa 2 (real): Enriquecimiento semántico vía LLM.

Este archivo usa LangChain y el helper build_llm() para mantener un único
punto flexible de configuración del proveedor/modelo (Groq, Ollama, etc.),
sin mezclar la API directa de Anthropic con la interfaz de tool calling de
LangChain.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

from langchain_core.tools import tool

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.project_paths import resolve_artifact_path, resolve_db_path


def build_llm(provider: str = "groq", model_name: str = "openai/gpt-oss-20b"):
    """Punto único de configuración del proveedor/modelo."""
    if provider == "groq":
        from langchain_groq import ChatGroq
        resolved_key = os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "Falta GROQ_API_KEY. Exporta la variable antes de ejecutar este script:\n"
                "  export GROQ_API_KEY='tu_clave_aqui'\n"
                "  python3 src/enrich_with_LLM_langchain.py"
            )
        return ChatGroq(model=model_name, temperature=0, api_key=resolved_key)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0)
    raise ValueError(f"Proveedor desconocido: {provider}")


@tool
def describir_tabla(
    table_name: str,
    columns: list,
    foreign_keys: list,
    sample_rows: list,
) -> dict:
    """Genera una descripción semántica para una tabla dada."""
    return {
        "table_name": table_name,
        "table_description": "Descripción generada por el modelo",
        "column_descriptions": {},
        "example_questions": [],
    }


@tool
def describir_esquema_completo(tables: list) -> dict:
    """Genera la descripción semántica para todas las tablas del esquema."""
    return {"tables": tables}


def get_sample_rows(db_path: str, table: str, n: int = 3) -> list:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM '{table}' LIMIT {n}")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows


def enrich_table_with_llm(table_name: str, table_info: dict, sample_rows: list, llm=None) -> dict:
    llm = llm or build_llm()
    prompt = f"""Analiza esta tabla de una base de datos y genera su documentación semántica.

Tabla: {table_name}
Columnas: {json.dumps(table_info['columns'], ensure_ascii=False)}
Relaciones (foreign keys): {json.dumps(table_info['foreign_keys'], ensure_ascii=False)}
Muestra de filas reales:
{json.dumps(sample_rows, ensure_ascii=False, indent=2)}

IMPORTANTE: si esta tabla podría confundirse con otra tabla similar del
mismo dominio (por ejemplo, dos tablas que ambas mencionan "cliente" o
"factura"), incluye en table_description qué la distingue explícitamente.
No repitas la misma palabra clave en las example_questions de tablas
distintas si esa palabra no es realmente específica de esta tabla —
evita frases genéricas que podrían aplicar a cualquier tabla del dominio."""

    model = llm.bind_tools([describir_tabla])
    response = model.invoke(prompt)

    if getattr(response, "tool_calls", None):
        args = response.tool_calls[0]["args"]
        return {
            "table_name": table_name,
            "table_description": args.get("table_description", ""),
            "column_descriptions": args.get("column_descriptions", {}),
            "example_questions": args.get("example_questions", []),
        }

    if isinstance(response.content, str):
        try:
            parsed = json.loads(response.content)
            if isinstance(parsed, dict):
                return {
                    "table_name": table_name,
                    "table_description": parsed.get("table_description", ""),
                    "column_descriptions": parsed.get("column_descriptions", {}),
                    "example_questions": parsed.get("example_questions", []),
                }
        except Exception:
            pass

    raise RuntimeError(f"El modelo no devolvió una respuesta utilizable para {table_name}")


def enrich_all_at_once(
    catalog: dict,
    db_path: str,
    provider: str = "groq",
    model_name: str = "openai/gpt-oss-20b",
) -> dict:
    """Genera y devuelve descripciones para todas las tablas usando LangChain + tool calling."""
    llm = build_llm(provider=provider, model_name=model_name)
    model = llm.bind_tools([describir_esquema_completo])

    schema_overview = []
    for table_name, info in catalog.items():
        schema_overview.append({
            "table": table_name,
            "columns": info["columns"],
            "foreign_keys": info["foreign_keys"],
            "sample_rows": get_sample_rows(db_path, table_name, n=2),
        })

    prompt = f"""Analiza este esquema de base de datos COMPLETO y genera la
documentación semántica de cada tabla.

Esquema completo ({len(schema_overview)} tablas):
{json.dumps(schema_overview, ensure_ascii=False, indent=2)}

INSTRUCCIONES CRÍTICAS:
- Tienes visibilidad de TODAS las tablas al mismo tiempo. Úsala: si dos
  tablas podrían confundirse entre sí, la descripción de CADA UNA debe
  decir explícitamente qué la distingue de la otra.
- Las preguntas de ejemplo de una tabla NO deben usar palabras clave que
  también aparezcan igual de bien en preguntas de ejemplo de otra tabla.
- Genera la salida para las {len(schema_overview)} tablas, en el mismo
  orden en que aparecen arriba."""

    response = model.invoke(prompt)

    if getattr(response, "tool_calls", None):
        tool_args = response.tool_calls[0]["args"]
        results = tool_args.get("tables", [])
    else:
        content = response.content if hasattr(response, "content") else str(response)
        try:
            payload = json.loads(content)
            results = payload.get("tables", [])
        except Exception as exc:
            raise RuntimeError(f"El modelo no devolvió un JSON de tool call usable: {exc}") from exc

    results_by_table = {item["table_name"]: item for item in results}

    for table_name, table_info in catalog.items():
        result = results_by_table.get(table_name)
        if not result:
            print(f"  ADVERTENCIA: el LLM no generó descripción para {table_name}")
            continue
        table_info["description"] = result["table_description"]
        table_info["example_questions"] = result.get("example_questions", [])
        table_info["status"] = "pending_review"
        col_descs = result.get("column_descriptions", {})
        for col in table_info["columns"]:
            if col["name"] in col_descs:
                col["description"] = col_descs[col["name"]]

    return catalog


def approve_all(catalog: dict) -> dict:
    for table_info in catalog.values():
        table_info["status"] = "active"
    return catalog


if __name__ == "__main__":
    try:
        input_path = resolve_artifact_path("catalog_raw.json")
        db_path = resolve_db_path("chinook.db")
        output_path = resolve_artifact_path("catalog_enriched_llm.json")

        with open(input_path, encoding="utf-8") as f:
            catalog = json.load(f)

        catalog = enrich_all_at_once(catalog, str(db_path))

        print("\n" + "=" * 70)
        print("DESCRIPCIONES GENERADAS (pending_review) — revísalas antes de aprobar")
        print("=" * 70)
        for t, info in catalog.items():
            print(f"\n[{info['status']}] {t}")
            print(f"  Descripción: {info['description']}")
            print(f"  Preguntas de ejemplo: {info['example_questions']}")

        respuesta = input("\n¿Aprobar todas y guardar? (s/n): ")
        if respuesta.lower() == "s":
            catalog = approve_all(catalog)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
            print(f"Guardado -> {output_path}")
        else:
            print("No se guardó. Ajusta el prompt y vuelve a correr.")
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)