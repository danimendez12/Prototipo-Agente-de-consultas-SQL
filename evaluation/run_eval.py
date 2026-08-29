"""
Corre el Explorador contra el set de evaluación y calcula:
- Precision: de las tablas recuperadas, cuántas eran realmente necesarias
- Recall: de las tablas necesarias, cuántas se recuperaron
- Exact match: el conjunto recuperado es idéntico al esperado
- Latencia por consulta

Se mide tanto sobre "retrieved_by_similarity" (solo TF-IDF) como sobre
"expanded_with_graph" (TF-IDF + expansión de 1 salto), para poder
argumentar cuánto aporta la expansión por grafo frente a similitud pura.
"""
import sys
import os
import pickle
import time
import json
import statistics
from datetime import datetime

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from src.explorador import Explorador
from src.project_paths import resolve_graph_path, resolve_results_path
from evaluation.eval_set import EVAL_SET


def precision_recall(retrieved: set, expected: set) -> tuple:
    if not retrieved:
        return 0.0, 0.0
    true_positives = len(retrieved & expected)
    precision = true_positives / len(retrieved)
    recall = true_positives / len(expected) if expected else 1.0
    return precision, recall


def run_eval():
    graph_path = resolve_graph_path()
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    explorador = Explorador(graph)

    results = []
    for case in EVAL_SET:
        expected = case["expected_tables"]

        t0 = time.perf_counter()
        ctx = explorador.retrieve(case["question"], top_k=3, expand_hops=1)
        latency_ms = (time.perf_counter() - t0) * 1000

        sim_only = set(ctx["retrieved_by_similarity"])
        expanded = set(ctx["expanded_with_graph"])

        p_sim, r_sim = precision_recall(sim_only, expected)
        p_exp, r_exp = precision_recall(expanded, expected)

        results.append({
            "question": case["question"],
            "expected": sorted(expected),
            "sim_only": sorted(sim_only),
            "expanded": sorted(expanded),
            "precision_sim": round(p_sim, 2),
            "recall_sim": round(r_sim, 2),
            "precision_expanded": round(p_exp, 2),
            "recall_expanded": round(r_exp, 2),
            "exact_match_expanded": expanded == expected,
            "latency_ms": round(latency_ms, 2),
        })

    return results


def summarize(results):
    n = len(results)
    avg = lambda key: round(statistics.mean(r[key] for r in results), 3)
    exact_matches = sum(1 for r in results if r["exact_match_expanded"])

    summary = {
        "n_preguntas": n,
        "precision_promedio_solo_similitud": avg("precision_sim"),
        "recall_promedio_solo_similitud": avg("recall_sim"),
        "precision_promedio_con_grafo": avg("precision_expanded"),
        "recall_promedio_con_grafo": avg("recall_expanded"),
        "exact_match_rate": round(exact_matches / n, 3),
        "latencia_promedio_ms": avg("latency_ms"),
    }
    return summary


if __name__ == "__main__":
    results = run_eval()
    summary = summarize(results)

    print("=" * 70)
    print("RESULTADOS POR PREGUNTA")
    print("=" * 70)
    for r in results:
        print(f"\n❓ {r['question']}")
        print(f"   Esperado:        {r['expected']}")
        print(f"   Solo similitud:  {r['sim_only']}  (P={r['precision_sim']}, R={r['recall_sim']})")
        print(f"   Con expansión:   {r['expanded']}  (P={r['precision_expanded']}, R={r['recall_expanded']})")
        print(f"   Exact match: {'✅' if r['exact_match_expanded'] else '❌'}  |  Latencia: {r['latency_ms']} ms")

    metricas_path = resolve_results_path("metricas.txt")
    with open(metricas_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Resumen de resultados de evaluación:\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")

    print("\n" + "=" * 70)
    print("RESUMEN GENERAL")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    results_path = resolve_results_path("eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados -> {results_path}")
