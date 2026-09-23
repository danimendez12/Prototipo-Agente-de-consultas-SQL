"""
Evalúa el pipeline COMPLETO: Explorador -> Generador, encadenados de
verdad (a diferencia de generator_eval.py, que usa las tablas "oracle"
de EVAL_SET). Aquí el Generador recibe EXACTAMENTE lo que el Explorador
real decidió — si el Explorador se equivoca o alucina una tabla, eso se
propaga al Generador, tal como pasaría en producción.

Permite modelos distintos por rol, lo cual además resuelve el problema
de rate limit compartido: si Explorador y Generador usan modelos
diferentes, cada uno tiene su propio cupo de peticiones/día en Groq.
"""
import os
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pickle
import time
import sqlite3
import sqlglot
import statistics
from datetime import datetime

from src.explorador import Explorador
from src.agent_explorador_langchain import AgentExplorerLC
from src.agent_generator_langchain import SQLQueryGeneratorAgent
from evaluation.eval_set import EVAL_SET
from src.services.security_check import evaluate_query

try:
    from src.project_paths import resolve_graph_path, resolve_results_path
except ImportError:
    def resolve_graph_path():
        return "artifacts/graph.pkl"

    def resolve_results_path(name):
        return name

DB_PATH = os.getenv("CHINOOK_DB_PATH", "data/chinook.db")


def build_tables_context(graph, table_names):
    """Arma el contexto completo (columnas, FKs) para las tablas que el
    Explorador REALMENTE devolvió. Si el Explorador alucinó un nombre de
    tabla que no existe en el grafo, se omite aquí (y se reporta aparte
    como 'tablas alucinadas' — un fallo del Explorador, no del Generador)."""
    context = {}
    for t in table_names:
        if t not in graph:
            continue
        data = graph.nodes[t]
        context[t] = {
            "description": data.get("description", ""),
            "columns": [c["name"] for c in data.get("columns", [])],
            "foreign_keys": [
                {"via": d["via"], "to": v}
                for _, v, d in graph.out_edges(t, data=True)
            ],
        }
    return context


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


def precision_recall_f1(retrieved, expected):
    if not retrieved:
        return 0.0, 0.0, 0.0
    tp = len(retrieved & expected)
    precision = tp / len(retrieved)
    recall = tp / len(expected) if expected else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def run_pipeline_eval(
    explorer_provider="groq",
    explorer_model="openai/gpt-oss-20b",
    generator_provider="groq",
    generator_model="openai/gpt-oss-20b",
    n=None,
    db_path=DB_PATH,
    sleep_between=7.0,
):
    graph_path = resolve_graph_path()
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    explorer = Explorador(graph)
    explorer_agent = AgentExplorerLC(explorer, provider=explorer_provider, model_name=explorer_model)
    generator_agent = SQLQueryGeneratorAgent(provider=generator_provider, model_name=generator_model)


    eval_subset = random.sample(EVAL_SET, min(n, len(EVAL_SET)))
    results = []

    for case in eval_subset:
        question = case["question"]
        expected_tables = case["expected_tables"]

        # --- Stage 1: Explorer ---
        t0 = time.perf_counter()
        try:
            explorer_result = explorer_agent.retrieve(question)
        except Exception as e:
            print(f"❌ [Explorer] error in '{question[:50]}...': {e}")
            time.sleep(sleep_between)
            continue
        explorer_latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_tables = set(explorer_result["tables"])
        p, r, f1 = precision_recall_f1(retrieved_tables, expected_tables)
        hallucinated = [t for t in retrieved_tables if t not in graph]

        time.sleep(sleep_between)

        # --- Stage 2: Generator (receives Explorer's REAL output) ---
        tables_context = build_tables_context(graph, retrieved_tables)
        t0 = time.perf_counter()
        gen_response = generator_agent.generate_sql_query(
            question, tables_context, explorer_result.get("reasoning", "")
        )
        generator_latency_ms = (time.perf_counter() - t0) * 1000

        # --- Stage 3: Security check and execution ---

        syntax_ok = is_select_only(gen_response["sql"]) if gen_response["success"] else False

        security_ok, security_explanation = evaluate_query(gen_response["sql"]) if gen_response["success"] else (False, "Query generation failed.")
        if not security_ok:
            exec_ok = False
            exec_error = f"Security issue: {security_explanation}"
        elif syntax_ok:
            exec_ok, exec_error = try_execute(gen_response["sql"], db_path)
        else:
            exec_ok = False
            exec_error = "SQL isn't a valid SELECT statement."

        pipeline_success = gen_response["success"] and syntax_ok and security_ok and exec_ok

        results.append({
            "question": question,
            "explorer_precision": round(p, 2),
            "explorer_recall": round(r, 2),
            "explorer_f1": round(f1, 2),
            "explorer_exact_match": retrieved_tables == expected_tables,
            "explorer_hallucinated_tables": hallucinated,
            "explorer_latency_ms": round(explorer_latency_ms, 1),
            "generator_declined": not gen_response["success"],
            "generator_syntax_valid": syntax_ok,
            "generator_executes_ok": exec_ok,
            "generator_latency_ms": round(generator_latency_ms, 1),
            "sql": gen_response["sql"],
            "pipeline_success": pipeline_success,
        })

        status = "✅" if pipeline_success else "❌"
        print(f"{status} {question[:55]:<55} Explorador(F1={f1:.2f}) -> Generador(sintaxis={syntax_ok}, ejecuta={exec_ok})")
        if not pipeline_success:
            if hallucinated:
                reason = f"Explorador alucinó tabla(s): {', '.join(hallucinated)}"
            elif gen_response["error"]:
                reason = f"Generador: {gen_response['error']}"
            elif exec_error:
                reason = f"Ejecución: {exec_error}"
            else:
                reason = "SQL no válido"
            print(f"    Motivo: {reason}")

        time.sleep(sleep_between)

    n_total = len(results)
    if n_total == 0:
        print("Sin resultados.")
        return results

    summary = {
        "n_preguntas": n_total,
        "modelo_explorador": f"{explorer_provider}/{explorer_model}",
        "modelo_generador": f"{generator_provider}/{generator_model}",
        "explorador_f1_promedio": round(statistics.mean(r["explorer_f1"] for r in results), 3),
        "explorador_exact_match_rate": round(sum(r["explorer_exact_match"] for r in results) / n_total, 3),
        "generador_tasa_declinado": round(sum(r["generator_declined"] for r in results) / n_total, 3),
        "generador_tasa_ejecuta_ok": round(sum(r["generator_executes_ok"] for r in results) / n_total, 3),
        "pipeline_exito_end_to_end": round(sum(r["pipeline_success"] for r in results) / n_total, 3),
        "latencia_total_promedio_ms": round(
            statistics.mean(r["explorer_latency_ms"] + r["generator_latency_ms"] for r in results), 1
        ),
    }

    print("\n" + "=" * 70)
    print("📊 SUMMARY — COMPLETE PIPELINE (Explorer + Generator)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(resolve_results_path("metricas.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"Label: PIPELINE E2E (Explorer={explorer_provider}/{explorer_model}, "
            f"Generator={generator_provider}/{generator_model})\n"
        )
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")
    print("\nMetrics added to -> metricas.txt")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--explorer-provider", default="groq")
    parser.add_argument("--explorer-model", required=True)
    parser.add_argument("--generator-provider", default="groq")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--db", default=DB_PATH)
    args = parser.parse_args()

    run_pipeline_eval(
        explorer_provider=args.explorer_provider,
        explorer_model=args.explorer_model,
        generator_provider=args.generator_provider,
        generator_model=args.generator_model,
        n=args.n,
        db_path=args.db,
        sleep_between=args.sleep,
    )