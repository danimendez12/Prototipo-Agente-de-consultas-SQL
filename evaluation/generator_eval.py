"""
Evalúa el Generador de SQL EN AISLAMIENTO del Explorador.

Idea clave: en vez de encadenar Explorador -> Generador (lo que mezcla
los errores de ambos y gasta cupo de rate limit de dos modelos por
pregunta), usamos las `expected_tables` ya anotadas en EVAL_SET como si
el Explorador hubiera acertado perfectamente. Así, si el Generador falla
aquí, el problema es 100% del Generador — no una consecuencia de que el
Explorador le haya entregado tablas de más o de menos.

Tres métricas, de más básica a más exigente:
  1. tasa_declinado: ¿el modelo se rindió (campo error) en vez de intentar?
  2. tasa_sintaxis_valida: ¿lo que generó es un SELECT sintácticamente válido?
  3. tasa_ejecuta_sin_error: ¿la consulta corre contra la BD real sin
     errores de columnas/tablas inexistentes? (NO verifica que el
     resultado sea el correcto — eso requeriría SQL de referencia por
     pregunta, un paso posterior si quieres ir más profundo)
"""
import os
import sys
from pathlib import Path

from numpy import random

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import sqlite3
import sqlglot
import statistics
from datetime import datetime

from src.agent_generator_langchain import SQLQueryGeneratorAgent
from evaluation.eval_set import EVAL_SET

try:
    from src.project_paths import resolve_results_path
except ImportError:
    def resolve_results_path(name):
        return name  # fallback simple si no existe ese helper en tu proyecto

import pickle
from src.project_paths import resolve_graph_path

DB_PATH = os.getenv("CHINOOK_DB_PATH", "data/chinook.db")


def build_generator_eval_set(graph, n=None):
    """Reutiliza pregunta + expected_tables de EVAL_SET, pero arma el
    CONTEXTO COMPLETO (columnas, FKs) desde el grafo — igual que lo haría
    el Explorador real. Pasar solo nombres de tabla, sin columnas, hace
    que el modelo adivine nombres de columna y genere SQL con columnas
    inexistentes aunque la lógica sea correcta (esto es justo lo que
    causó el único fallo de la corrida anterior)."""
    n = n or 5
    eval_subset = random.sample(EVAL_SET, min(n, len(EVAL_SET)))
    cases = []
    for case in eval_subset:
        tables = sorted(case["expected_tables"])
        tables_context = {}
        for t in tables:
            if t not in graph:
                continue
            data = graph.nodes[t]
            tables_context[t] = {
                "description": data.get("description", ""),
                "columns": [c["name"] for c in data.get("columns", [])],
                "foreign_keys": [
                    {"via": d["via"], "to": v}
                    for _, v, d in graph.out_edges(t, data=True)
                ],
            }
        cases.append({
            "question": case["question"],
            "tables_context": tables_context,
            "reasoning": f"Estas tablas ({', '.join(tables)}) son necesarias para responder la pregunta.",
        })
    return cases


def is_select_only(sql: str) -> bool:
    if not sql:
        return False
    try:
        parsed = sqlglot.parse_one(sql)
    except Exception:
        return False
    return parsed.key.lower() == "select"


def try_execute(sql: str, db_path: str):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        cur.fetchmany(5)
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)


def run_generator_eval(provider="groq", model_name="openai/gpt-oss-120b", n=None, db_path=DB_PATH):
    graph_path = resolve_graph_path()
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    generator = SQLQueryGeneratorAgent(provider=provider, model_name=model_name)
    cases = build_generator_eval_set(graph, n)

    results = []
    for case in cases:
        response = generator.generate_sql_query(case["question"], case["tables_context"], case["reasoning"])

        syntax_ok = is_select_only(response["sql"]) if response["success"] else False
        exec_ok, exec_error = try_execute(response["sql"], db_path) if syntax_ok else (False, "")

        results.append({
            "question": case["question"],
            "tables_given": list(case["tables_context"].keys()),
            "declined": not response["success"],
            "declined_reason": response["error"],
            "sql": response["sql"],
            "syntax_valid": syntax_ok,
            "executes_ok": exec_ok,
            "exec_error": exec_error,
        })

        if not response["success"]:
            status = "🚫"
        elif exec_ok:
            status = "✅"
        elif syntax_ok:
            status = "⚠️ "
        else:
            status = "❌"
        print(f"{status} {case['question'][:60]:<60} sintaxis={syntax_ok} ejecuta={exec_ok}")
        if response["success"] and not exec_ok:
            print(f"    SQL: {response['sql']}")
            print(f"    Error: {exec_error}")
        elif not response["success"]:
            print(f"    Declinado: {response['error']}")

    n_total = len(results)
    summary = {
        "n_preguntas": n_total,
        "modelo": f"{provider}/{model_name}",
        "tasa_declinado": round(sum(r["declined"] for r in results) / n_total, 3),
        "tasa_sintaxis_valida": round(sum(r["syntax_valid"] for r in results) / n_total, 3),
        "tasa_ejecuta_sin_error": round(sum(r["executes_ok"] for r in results) / n_total, 3),
    }

    print("\n" + "=" * 70)
    print("📊 RESUMEN — GENERADOR")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(resolve_results_path("metricas.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Etiqueta: GENERADOR ({provider}/{model_name})\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")
    print("\nMétricas agregadas -> metricas.txt")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    run_generator_eval(provider=args.provider, model_name=args.model, n=args.n, db_path=args.db)