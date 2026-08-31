"""
Evaluates AgentExplorador against EVAL_SET — same metrics as before
(precision/recall/exact_match) plus the new ones that matter for a real agent:
API token cost and number of tool calls per question.

COST WARNING: unlike run_eval.py (free, local), this script makes real API calls for each question.
With 40 questions and up to 3-4 tool calls each, that's roughly 100-150 Claude calls. Run a small
subset first (see --n below) before the full set.
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
from src.agent_explorador import AgentExplorador
from src.project_paths import resolve_graph_path, resolve_results_path
from evaluation.eval_set import EVAL_SET

# Claude Sonnet 5 pricing: $2/MTok input, $10/MTok output (August 2026)
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
    explorer = Explorador(graph)
    agent = AgentExplorador(explorer)

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

        tools_used = [t["name"] for t in r.get("trace", [])]
        tools_sequence = " → ".join(tools_used) if tools_used else "(none)"

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

    all_tools_used = []
    for r in results:
        all_tools_used.extend(r.get("tools_used", []))

    from collections import Counter
    tools_frequency = Counter(all_tools_used)
    most_common_tools = dict(tools_frequency.most_common(5))

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"  n_questions: {n}")
    print(f"  average_precision: {round(statistics.mean(r['precision'] for r in results), 3)}")
    print(f"  average_recall: {round(statistics.mean(r['recall'] for r in results), 3)}")
    print(f"  exact_match_rate: {round(sum(r['exact_match'] for r in results) / n, 3)}")
    print(f"  average_tool_calls: {round(statistics.mean(r['tool_calls'] for r in results), 2)}")
    print(f"  average_latency_ms: {round(statistics.mean(r['latency_ms'] for r in results), 1)}")
    print(f"  total_cost_usd: ${round(total_cost, 4)}")
    print(f"  average_cost_per_question_usd: ${round(total_cost / n, 5)}")

    print("\n  🔧 Tools used (frequency):")
    for tool_name, count in most_common_tools.items():
        pct = round(count / len(all_tools_used) * 100, 1) if all_tools_used else 0
        print(f"    - {tool_name}: {count} times ({pct}%)")

    return results


if __name__ == "__main__":
    # python3 agent_eval.py        -> runs all 40 questions
    # python3 agent_eval.py 5      -> runs only the first 5 (cheaper/faster smoke test)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    results = run_agent_eval(n_questions=n)

    n_total = len(results)
    summary = {
        "n_questions": n_total,
        "average_precision": round(statistics.mean(r["precision"] for r in results), 3),
        "average_recall": round(statistics.mean(r["recall"] for r in results), 3),
        "exact_match_rate": round(sum(r["exact_match"] for r in results) / n_total, 3),
        "average_tool_calls": round(statistics.mean(r["tool_calls"] for r in results), 2),
        "average_latency_ms": round(statistics.mean(r["latency_ms"] for r in results), 1),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
    }

    latencies = [r["latency_ms"] for r in results]
    latency_min = min(latencies) if latencies else 0
    latency_max = max(latencies) if latencies else 0
    latency_std = round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0
    latency_total = round(sum(latencies), 1)

    metrics_path = resolve_results_path("metricas.txt")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"MEASUREMENT: Agent evaluation + vector search\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Label: AGENT (tool use) - n={n_total}\n")
        f.write(f"\n📊 QUALITY METRICS:\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")

        f.write(f"\n⏱️  TIME METRICS (ms):\n")
        f.write(f"  average_latency_ms: {summary['average_latency_ms']}\n")
        f.write(f"  latency_min_ms: {latency_min}\n")
        f.write(f"  latency_max_ms: {latency_max}\n")
        f.write(f"  latency_std_dev_ms: {latency_std}\n")
        f.write(f"  latency_total_ms: {latency_total}\n")

        f.write(f"\n📝 DETAILS BY QUESTION:\n")
        for i, r in enumerate(results, 1):
            status = "✅" if r["exact_match"] else "❌"
            f.write(f"  {i}. {status} P={r['precision']:.2f} R={r['recall']:.2f} | ")
            f.write(f"{r['latency_ms']:.1f}ms | tools={r['tool_calls']} | ${r['cost_usd']}\n")
            f.write(f"     Q: {r['question'][:70]}\n")

    print(f"\n✅ Metrics added to the history -> metricas.txt")