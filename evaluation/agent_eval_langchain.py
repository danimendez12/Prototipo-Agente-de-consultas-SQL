"""
Evalúa AgentExploradorLC (LangChain + Llama 3.3 70B vía Groq) contra
EVAL_SET, con las mismas métricas que agent_eval.py, para comparar
directamente: Claude (cerrado, tool use directo) vs. Llama (abierto,
vía LangChain).

Precio de Llama 3.3 70B Versatile en Groq (agosto 2026): $0.59/$0.79
por millón de tokens input/output — bastante más barato que Sonnet 5.
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
from src.agent_explorador_langchain import AgentExploradorLC
from src.project_paths import resolve_graph_path, resolve_results_path
from evaluation.eval_set import EVAL_SET

PRICE_INPUT_PER_TOK = 0.59 / 1_000_000
PRICE_OUTPUT_PER_TOK = 0.79 / 1_000_000


def precision_recall_f1(retrieved, expected):
    if not retrieved:
        return 0.0, 0.0, 0.0

    tp = len(retrieved & expected)

    precision = tp / len(retrieved)
    recall = tp / len(expected) if expected else 1.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1



def run_agent_eval_lc(n_questions=None, provider="groq", model_name="openai/gpt-oss-120b"):
    graph_path = resolve_graph_path()
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    explorador = Explorador(graph)
    agent = AgentExploradorLC(explorador, provider=provider, model_name=model_name)

    import random

    n = n_questions or 5
    eval_subset = random.sample(EVAL_SET, min(n, len(EVAL_SET)))

    results = []
    for case in eval_subset:
        t0 = time.perf_counter()
        try:
            r = agent.retrieve(case["question"])
        except Exception as e:
            print(f"  ERROR en '{case['question'][:50]}...': {e}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved = set(r["tables"])
        expected = case["expected_tables"]
        p, rec, f1 = precision_recall_f1(retrieved, expected)
        
        # Extraer qué tools se usaron del trace
        tools_used = [t["tool"] for t in r.get("trace", [])]
        tools_sequence = " → ".join(tools_used) if tools_used else "(ninguno)"

        results.append({
            "question": case["question"],
            "expected": sorted(expected),
            "retrieved": sorted(retrieved),
            "f1": round(f1, 2),
            "precision": round(p, 2),
            "recall": round(rec, 2),
            "exact_match": retrieved == expected,
            "tool_calls": r["tool_calls"],
            "tools_used": tools_used,
            "tools_sequence": tools_sequence,
            "latency_ms": round(latency_ms, 1),
        })

        status = "✅" if retrieved == expected else "❌"
        print(f"{status} {case['question'][:60]:<60} P={p:.2f} R={rec:.2f} F1={f1:.2f}")
        print(f"  🔍 Recuperadas: {sorted(retrieved)}")
        print(f"  📊 Esperadas:   {sorted(expected)}")
        print(f"  🔧 Tools usados: {tools_sequence}")
        print(f"  ⏱️  Latencia: {latency_ms:.0f}ms | Tool calls: {r['tool_calls']}")

    n = len(results)
    if n == 0:
        print("Sin resultados (todas las preguntas fallaron).")
        return results

    # Estadísticas de tools usados
    all_tools_used = []
    for r in results:
        all_tools_used.extend(r.get("tools_used", []))
    
    from collections import Counter
    tools_frequency = Counter(all_tools_used)
    most_common_tools = dict(tools_frequency.most_common(5))

    summary = {
        "n_preguntas": n,
        "modelo": f"{provider}/{model_name}",
        "precision_promedio": round(statistics.mean(r["precision"] for r in results), 3),
        "f1_promedio": round(statistics.mean(r["f1"] for r in results), 3),
        "recall_promedio": round(statistics.mean(r["recall"] for r in results), 3),
        "exact_match_rate": round(sum(r["exact_match"] for r in results) / n, 3),
        "tool_calls_promedio": round(statistics.mean(r["tool_calls"] for r in results), 2),
        "latencia_promedio_ms": round(statistics.mean(r["latency_ms"] for r in results), 1),
        "tools_usados_frecuencia": most_common_tools,
    }

    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    for k, v in summary.items():
        if k != "tools_usados_frecuencia":
            print(f"  {k}: {v}")
    
    print("\n  🔧 Tools usados (frecuencia):")
    for tool_name, count in most_common_tools.items():
        pct = round(count / len(all_tools_used) * 100, 1) if all_tools_used else 0
        print(f"    - {tool_name}: {count} veces ({pct}%)")

    with open(resolve_results_path("metricas.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Etiqueta: AGENTE LangChain ({provider}/{model_name})\n")
        for k, v in summary.items():
            if k != "tools_usados_frecuencia":
                f.write(f"  {k}: {v}\n")
        f.write(f"  Tools usados (frecuencia):\n")
        for tool_name, count in most_common_tools.items():
            pct = round(count / len(all_tools_used) * 100, 1) if all_tools_used else 0
            f.write(f"    - {tool_name}: {count} veces ({pct}%)\n")
    print("\nMétricas agregadas -> metricas.txt")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=None)

    args = parser.parse_args()

    run_agent_eval_lc(
        n_questions=args.n,
        provider=args.provider,
        model_name=args.model
    )