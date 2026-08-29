"""
Evalúa AgentExplorador contra EVAL_SET — mismas métricas de siempre
(precision/recall/exact_match) más las nuevas que importan para un
agente real: costo en tokens y número de tool calls por pregunta.

ADVERTENCIA DE COSTO: a diferencia de run_eval.py (gratis, local), esto
hace llamadas reales a la API por cada pregunta. Con 40 preguntas y
hasta 3-4 tool calls cada una, son ~100-150 llamadas a Claude. Corre
primero con un subconjunto pequeño (ver --n abajo) antes del set completo.
"""
import os
import sys
import pickle
import time
import sys
import statistics
from datetime import datetime

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from src.explorador import Explorador
from src.agent_explorador import AgentExplorador
from src.project_paths import resolve_graph_path, resolve_results_path
from evaluation.eval_set import EVAL_SET

# Precio de Claude Sonnet 5: $2/MTok input, $10/MTok output (agosto 2026)
PRICE_INPUT_PER_TOK = 2 / 1_000_000
PRICE_OUTPUT_PER_TOK = 10 / 1_000_000


def precision_recall(retrieved, expected):
    if not retrieved:
        return 0.0, 0.0
    tp = len(retrieved & expected)
    return tp / len(retrieved), (tp / len(expected) if expected else 1.0)


def run_agent_eval(n_questions=None):
    graph_path = resolve_graph_path()
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    explorador = Explorador(graph)
    agent = AgentExplorador(explorador)

    eval_subset = EVAL_SET[:n_questions] if n_questions else EVAL_SET

    results = []
    total_cost = 0.0
    for case in eval_subset:
        t0 = time.perf_counter()
        r = agent.retrieve(case["question"])
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved = set(r["tables"])
        expected = case["expected_tables"]
        p, rec = precision_recall(retrieved, expected)

        cost = (
            r["usage"]["input_tokens"] * PRICE_INPUT_PER_TOK
            + r["usage"]["output_tokens"] * PRICE_OUTPUT_PER_TOK
        )
        total_cost += cost
        
        # Extraer qué tools se usaron del trace
        tools_used = [t["name"] for t in r.get("trace", [])]
        tools_sequence = " → ".join(tools_used) if tools_used else "(ninguno)"

        results.append({
            "question": case["question"],
            "expected": sorted(expected),
            "retrieved": sorted(retrieved),
            "precision": round(p, 2),
            "recall": round(rec, 2),
            "exact_match": retrieved == expected,
            "tool_calls": r["tool_calls"],
            "tools_used": tools_used,
            "tools_sequence": tools_sequence,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost, 5),
        })

        status = "✅" if retrieved == expected else "❌"
        print(f"{status} {case['question'][:60]:<60} P={p:.2f} R={rec:.2f}")
        print(f"  🔧 Tools: {tools_sequence}")
        print(f"  ⏱️  {latency_ms:.0f}ms | Cost: ${cost:.5f}")

    n = len(results)
    
    # Estadísticas de tools usados
    all_tools_used = []
    for r in results:
        all_tools_used.extend(r.get("tools_used", []))
    
    from collections import Counter
    tools_frequency = Counter(all_tools_used)
    most_common_tools = dict(tools_frequency.most_common(5))
    
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    print(f"  n_preguntas: {n}")
    print(f"  precision_promedio: {round(statistics.mean(r['precision'] for r in results), 3)}")
    print(f"  recall_promedio: {round(statistics.mean(r['recall'] for r in results), 3)}")
    print(f"  exact_match_rate: {round(sum(r['exact_match'] for r in results) / n, 3)}")
    print(f"  tool_calls_promedio: {round(statistics.mean(r['tool_calls'] for r in results), 2)}")
    print(f"  latencia_promedio_ms: {round(statistics.mean(r['latency_ms'] for r in results), 1)}")
    print(f"  costo_total_usd: ${round(total_cost, 4)}")
    print(f"  costo_promedio_por_pregunta_usd: ${round(total_cost / n, 5)}")
    
    print("\n  🔧 Tools usados (frecuencia):")
    for tool_name, count in most_common_tools.items():
        pct = round(count / len(all_tools_used) * 100, 1) if all_tools_used else 0
        print(f"    - {tool_name}: {count} veces ({pct}%)")

    return results


if __name__ == "__main__":
    import sys

    # python3 agent_eval.py        -> corre las 40 preguntas completas
    # python3 agent_eval.py 5      -> corre solo las primeras 5 (para probar barato/rápido)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    results = run_agent_eval(n_questions=n)

    n_total = len(results)
    summary = {
        "n_preguntas": n_total,
        "precision_promedio": round(statistics.mean(r["precision"] for r in results), 3),
        "recall_promedio": round(statistics.mean(r["recall"] for r in results), 3),
        "exact_match_rate": round(sum(r["exact_match"] for r in results) / n_total, 3),
        "tool_calls_promedio": round(statistics.mean(r["tool_calls"] for r in results), 2),
        "latencia_promedio_ms": round(statistics.mean(r["latency_ms"] for r in results), 1),
        "costo_total_usd": round(sum(r["cost_usd"] for r in results), 4),
    }

    # Calcular métricas de tiempo adicionales
    latencias = [r["latency_ms"] for r in results]
    latencia_min = min(latencias) if latencias else 0
    latencia_max = max(latencias) if latencias else 0
    latencia_std = round(statistics.stdev(latencias), 1) if len(latencias) > 1 else 0
    latencia_total = round(sum(latencias), 1)
    
    metricas_path = resolve_results_path("metricas.txt")
    with open(metricas_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"MEDICIÓN: Evaluación de agente + búsqueda vectorial\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Etiqueta: AGENTE (tool use) - n={n_total}\n")
        f.write(f"\n📊 MÉTRICAS DE CALIDAD:\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")
        
        f.write(f"\n⏱️  MÉTRICAS DE TIEMPO (ms):\n")
        f.write(f"  latencia_promedio_ms: {summary['latencia_promedio_ms']}\n")
        f.write(f"  latencia_min_ms: {latencia_min}\n")
        f.write(f"  latencia_max_ms: {latencia_max}\n")
        f.write(f"  latencia_std_dev_ms: {latencia_std}\n")
        f.write(f"  latencia_total_ms: {latencia_total}\n")
        
        f.write(f"\n📝 DETALLES POR CONSULTA:\n")
        for i, r in enumerate(results, 1):
            status = "✅" if r["exact_match"] else "❌"
            f.write(f"  {i}. {status} P={r['precision']:.2f} R={r['recall']:.2f} | ")
            f.write(f"{r['latency_ms']:.1f}ms | tools={r['tool_calls']} | ${r['cost_usd']}\n")
            f.write(f"     Q: {r['question'][:70]}\n")
    
    print(f"\n✅ Métricas agregadas al historial -> metricas.txt")