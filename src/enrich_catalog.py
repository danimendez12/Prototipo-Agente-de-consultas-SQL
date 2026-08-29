"""
Etapa 2: Enriquecimiento semántico.

En producción, este paso llama a un LLM por cada tabla/columna ambigua,
usando su nombre, tipo y una muestra de filas anonimizadas, y el resultado
entra a una cola `pending_review` hasta que un humano lo aprueba (ver la
propuesta de arquitectura, sección "Capa semántica ligera").

Para este MVP, con solo 11 tablas, las descripciones fueron escritas
razonando directamente sobre el esquema y una muestra de filas real
(ver conversación) — cumpliendo el mismo rol que la llamada a LLM
cumpliría en producción. Quedan marcadas como pending_review y se
"aprueban" en la función approve_all() para simular ese paso del flujo.
"""
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.project_paths import resolve_artifact_path

# Descripciones semánticas por tabla y por columna ambigua.
# Este diccionario es el output que en producción generaría el agente
# Enriquecedor (LLM) tabla por tabla.
DESCRIPTIONS = {
    "Artist": {
        "table": "Artistas musicales o bandas. Solo el nombre del artista, sin sus álbumes ni canciones directamente.",
        "columns": {},
        "example_questions": [
            "¿Qué artistas tienen más de 5 álbumes?",
            "¿Cuáles son los artistas más populares?",
        ],
    },
    "Album": {
        "table": "Álbumes musicales, cada uno pertenece a un solo artista. No contiene las canciones en sí, solo el título del álbum.",
        "columns": {},
        "example_questions": [
            "¿Cuántos álbumes tiene cada artista?",
            "¿Cuál es el álbum con más canciones?",
        ],
    },
    "Track": {
        "table": "Canciones individuales disponibles para la venta, cada una pertenece a un álbum, tiene un género y un formato de audio.",
        "columns": {
            "Milliseconds": "Duración de la canción en milisegundos",
            "UnitPrice": "Precio de venta de la canción individual",
            "Composer": "Nombre del compositor de la canción",
        },
        "example_questions": [
            "¿Cuáles son las canciones más largas en duración?",
            "¿Cuál es la canción más cara?",
            "¿Quién compuso esta canción?",
        ],
    },
    "Genre": {
        "table": "Géneros musicales (Rock, Jazz, Classical, etc.), usados para clasificar canciones. No contiene información de ventas.",
        "columns": {},
        "example_questions": [
            "¿Cuáles son los géneros más populares?",
            "¿Cuántas canciones hay de cada género?",
        ],
    },
    "MediaType": {
        "table": "Formato técnico del archivo de audio de una canción (MPEG, AAC, etc.), no tiene relación con el precio ni las ventas.",
        "columns": {},
        "example_questions": [
            "¿Cuántas canciones tiene cada tipo de formato de audio?",
            "¿Qué formatos de archivo se usan?",
        ],
    },
    "Playlist": {
        "table": "Listas de reproducción creadas por usuarios, agrupan canciones. Solo el nombre de la lista, no las canciones que contiene.",
        "columns": {},
        "example_questions": [
            "¿Cuántas playlists existen?",
            "¿Cuál es el nombre de cada lista de reproducción?",
        ],
    },
    "PlaylistTrack": {
        "table": "Tabla de unión que asocia canciones con listas de reproducción (relación muchos a muchos entre Playlist y Track). Necesaria para saber qué canciones contiene cada playlist.",
        "columns": {},
        "example_questions": [
            "Lista las canciones de la playlist más grande",
            "¿En cuántas playlists aparece esta canción?",
        ],
    },
    "Customer": {
        "table": "Clientes externos que compran música (NO son empleados de la empresa). Incluye datos de contacto y el empleado de soporte asignado.",
        "columns": {
            "SupportRepId": "Empleado de soporte asignado a este cliente",
        },
        "example_questions": [
            "¿Cuál es el cliente que más ha gastado en total?",
            "¿En qué país viven más clientes?",
        ],
    },
    "Employee": {
        "table": "Empleados internos de la empresa (NO son clientes). Incluye representantes de soporte y su jerarquía de jefes directos.",
        "columns": {
            "ReportsTo": "Id del empleado que es jefe directo de este empleado (jerarquía interna)",
            "Title": "Puesto o cargo del empleado",
        },
        "example_questions": [
            "¿Qué empleado tiene más clientes asignados?",
            "¿Quién es el jefe directo de cada empleado?",
        ],
    },
    "Invoice": {
        "table": "Facturas de compra generadas por un cliente, con el monto TOTAL y la fecha. No incluye el detalle de qué canciones se compraron (eso está en InvoiceLine).",
        "columns": {
            "Total": "Monto total de la factura",
        },
        "example_questions": [
            "¿Cuál es el total facturado por país?",
            "¿Cuántas facturas se generaron el mes pasado?",
        ],
    },
    "InvoiceLine": {
        "table": "Detalle línea por línea de qué canción específica se vendió, a qué precio y en qué cantidad, dentro de una factura. Es la tabla clave para preguntas sobre canciones vendidas, ingresos por producto o ventas por género.",
        "columns": {
            "UnitPrice": "Precio unitario de la canción en el momento de la compra",
            "Quantity": "Cantidad comprada de esa canción en esa línea",
        },
        "example_questions": [
            "¿Cuáles son las canciones más vendidas?",
            "¿Qué género tiene más canciones vendidas?",
            "¿Cuántas unidades se vendieron de cada producto?",
        ],
    },
}


def enrich(catalog: dict) -> dict:
    for table_name, table_info in catalog.items():
        desc = DESCRIPTIONS.get(table_name, {})
        table_info["description"] = desc.get("table", f"Tabla {table_name} (sin descripción generada)")
        table_info["example_questions"] = desc.get("example_questions", [])
        table_info["status"] = "pending_review"
        col_descs = desc.get("columns", {})
        for col in table_info["columns"]:
            if col["name"] in col_descs:
                col["description"] = col_descs[col["name"]]
    return catalog


def approve_all(catalog: dict) -> dict:
    """Simula la aprobación humana de la cola de revisión (paso de PR/merge)."""
    for table_info in catalog.values():
        table_info["status"] = "active"
    return catalog


if __name__ == "__main__":
    input_path = resolve_artifact_path("catalog_raw.json")
    output_path = resolve_artifact_path("catalog_enriched.json")

    with open(input_path, encoding="utf-8") as f:
        catalog = json.load(f)

    catalog = enrich(catalog)
    print("Descripciones generadas (pending_review):")
    for t, info in catalog.items():
        print(f"  [{info['status']}] {t}: {info['description']}")

    catalog = approve_all(catalog)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"\nCatálogo enriquecido y aprobado -> {output_path}")