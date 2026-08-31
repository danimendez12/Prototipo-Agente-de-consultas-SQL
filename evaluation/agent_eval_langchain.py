"""
Evaluates AgentExploradorLC (LangChain + Llama 3.3 70B via Groq) against EVAL_SET,
using the same metrics as agent_eval.py, to compare directly: Claude (closed, direct tool use)
vs. Llama (open, via LangChain).

Llama 3.3 70B Versatile pricing on Groq (August 2026): $0.59/$0.79 per million input/output
 tokens — notably cheaper than Sonnet 5.
"""
import os
import sys
import pickle
import time
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
    explorer = Explorador(graph)
    agent = AgentExploradorLC(explorer, provider=provider, model_name=model_name)

    import random

    n = n_questions or 5
    eval_subset = random.sample(EVAL_SET, min(n, len(EVAL_SET)))

    results = []
    for case in eval_subset:
        time.sleep(5)
        t0 = time.perf_counter()
        try:
            r = agent.retrieve(case["question"])
        except Exception as e:
            print(f"  ERROR in '{case['question'][:50]}...': {e}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved = set(r["tables"])
        expected = case["expected_tables"]
        p, rec, f1 = precision_recall_f1(retrieved, expected)

        tools_used = [t["tool"] for t in r.get("trace", [])]
        tools_sequence = " → ".join(tools_used) if tools_used else "(none)"

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
        print(f"  🔍 Retrieved: {sorted(retrieved)}")
        print(f"  📊 Expected:   {sorted(expected)}")
        print(f"  🔧 Tools used: {tools_sequence}")
        print(f"  ⏱️  Latency: {latency_ms:.0f}ms | Tool calls: {r['tool_calls']}")

    n = len(results)
    if n == 0:
        print("No results (all questions failed).")
        return results

    all_tools_used = []
    for r in results:
        all_tools_used.extend(r.get("tools_used", []))

    from collections import Counter
    tools_frequency = Counter(all_tools_used)
    most_common_tools = dict(tools_frequency.most_common(5))

    summary = {
        "n_questions": n,
        "model": f"{provider}/{model_name}",
        "average_precision": round(statistics.mean(r["precision"] for r in results), 3),
        "average_f1": round(statistics.mean(r["f1"] for r in results), 3),
        "average_recall": round(statistics.mean(r["recall"] for r in results), 3),
        "exact_match_rate": round(sum(r["exact_match"] for r in results) / n, 3),
        "average_tool_calls": round(statistics.mean(r["tool_calls"] for r in results), 2),
        "average_latency_ms": round(statistics.mean(r["latency_ms"] for r in results), 1),
        "tools_used_frequency": most_common_tools,
    }

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        if k != "tools_used_frequency":
            print(f"  {k}: {v}")

    print("\n  🔧 Tools used (frequency):")
    for tool_name, count in most_common_tools.items():
        pct = round(count / len(all_tools_used) * 100, 1) if all_tools_used else 0
        print(f"    - {tool_name}: {count} times ({pct}%)")

    with open(resolve_results_path("metricas.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Label: AGENT LangChain ({provider}/{model_name})\n")
        for k, v in summary.items():
            if k != "tools_used_frequency":
                f.write(f"  {k}: {v}\n")
        f.write(f"  Tools used (frequency):\n")
        for tool_name, count in most_common_tools.items():
            pct = round(count / len(all_tools_used) * 100, 1) if all_tools_used else 0
            f.write(f"    - {tool_name}: {count} times ({pct}%)\n")
    print("\nMetrics added -> metricas.txt")

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