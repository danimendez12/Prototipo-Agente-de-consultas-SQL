"""
Etapa 2 (real): Enriquecimiento semántico vía LLM.

Reemplaza el diccionario DESCRIPTIONS escrito a mano por generación
real: para cada tabla, le pasamos a Claude su nombre, columnas, tipos,
y una muestra de filas, y le pedimos una descripción de la tabla,
descripciones de columnas ambiguas, y preguntas de ejemplo que un
usuario típico haría sobre esa tabla — usando tool use para forzar
un schema de salida confiable.

El resultado sigue entrando a estado `pending_review`, tal como en la
versión manual — la diferencia es quién lo genera, no el flujo de
aprobación.

Requiere: pip install anthropic
          export ANTHROPIC_API_KEY=...
"""
import sqlite3
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from anthropic import Anthropic
from src.project_paths import resolve_artifact_path, resolve_db_path

client = Anthropic()  

ENRICH_TOOL = {
    "name": "describir_tabla",
    "description": "Genera la descripción semántica de una tabla de base de datos",
    "input_schema": {
        "type": "object",
        "properties": {
            "table_description": {
                "type": "string",
                "description": "1-2 frases explicando qué representa esta tabla. Si se confunde fácilmente con otra tabla del mismo dominio, menciona EXPLICITAMENTE qué la distingue."
            },
            "column_descriptions": {
                "type": "object",
                "description": "Mapa de nombre_de_columna -> descripción corta, SOLO para columnas cuyo propósito no sea obvio por el nombre (ids, flags, fechas ambiguas, campos técnicos).",
                "additionalProperties": {"type": "string"}
            },
            "example_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 preguntas en lenguaje natural, en español, que un usuario de negocio típico haría y que requerirían esta tabla para responderse."
            }
        },
        "required": ["table_description", "column_descriptions", "example_questions"]
    }
}


def get_sample_rows(db_path: str, table: str, n: int = 3) -> list:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM '{table}' LIMIT {n}")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows


def enrich_table_with_llm(table_name: str, table_info: dict, sample_rows: list) -> dict:
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

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        tools=[ENRICH_TOOL],
        tool_choice={"type": "tool", "name": "describir_tabla"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise RuntimeError(f"El modelo no devolvió la tool use esperada para {table_name}")


ENRICH_ALL_TOOL = {
    "name": "describir_esquema_completo",
    "description": "Genera la descripción semántica de todas las tablas de un esquema a la vez, permitiendo distinguir tablas que se confunden entre sí",
    "input_schema": {
        "type": "object",
        "properties": {
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string"},
                        "table_description": {
                            "type": "string",
                            "description": "1-2 frases explicando qué representa esta tabla. Si se confunde fácilmente con OTRA tabla de este mismo esquema, menciona EXPLICITAMENTE qué la distingue (ej. 'a diferencia de X, esta tabla...')."
                        },
                        "column_descriptions": {
                            "type": "object",
                            "additionalProperties": {"type": "string"}
                        },
                        "example_questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "2-3 preguntas en español que un usuario haría y que requieren ESTA tabla específicamente."
                        }
                    },
                    "required": ["table_name", "table_description", "column_descriptions", "example_questions"]
                }
            }
        },
        "required": ["tables"]
    }
}


def enrich_all_at_once(catalog: dict, db_path: str) -> dict:
    """
    Una sola llamada al LLM con el esquema COMPLETO, para que el modelo
    pueda ver todas las tablas a la vez y escribir descripciones que se
    distingan activamente entre sí — algo imposible si cada tabla se
    describe en aislamiento (el problema que causó el resultado bajo
    de la versión anterior de este script).
    """
    schema_overview = []
    for table_name, info in catalog.items():
        samples = get_sample_rows(db_path, table_name, n=2)
        schema_overview.append({
            "table": table_name,
            "columns": info["columns"],
            "foreign_keys": info["foreign_keys"],
            "sample_rows": samples,
        })

    prompt = f"""Analiza este esquema de base de datos COMPLETO y genera la
documentación semántica de cada tabla.

Esquema completo ({len(schema_overview)} tablas):
{json.dumps(schema_overview, ensure_ascii=False, indent=2)}

INSTRUCCIONES CRÍTICAS:
- Tienes visibilidad de TODAS las tablas al mismo tiempo. Úsala: si dos
  tablas podrían confundirse entre sí (ej. ambas mencionan "cliente" o
  "factura" o "canción"), la descripción de CADA UNA debe decir
  explícitamente qué la distingue de la otra.
- Las preguntas de ejemplo de una tabla NO deben usar palabras clave que
  también aparezcan igual de bien en preguntas de ejemplo de otra tabla,
  a menos que esa palabra sea realmente exclusiva de esta tabla en el
  dominio. Evita frases genéricas del dominio completo.
- Genera la salida para las {len(schema_overview)} tablas, en el mismo
  orden en que aparecen arriba."""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4000,
        tools=[ENRICH_ALL_TOOL],
        tool_choice={"type": "tool", "name": "describir_esquema_completo"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use":
            results_by_table = {t["table_name"]: t for t in block.input["tables"]}
            break
    else:
        raise RuntimeError("El modelo no devolvió la tool use esperada")

    for table_name, table_info in catalog.items():
        result = results_by_table.get(table_name)
        if not result:
            print(f"  ADVERTENCIA: el LLM no generó descripción para {table_name}")
            continue
        table_info["description"] = result["table_description"]
        table_info["example_questions"] = result["example_questions"]
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