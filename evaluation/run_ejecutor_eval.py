"""
Evaluates the Ejecutor agent (stage 7) against ejecutor_eval_set.py.

Measures exactly the things that matter for this stage, per project priorities:

  SEGURIDAD:
    - security_block_rate: fraction of ADVERSARIAL_QUERIES that were correctly rejected
      by the executor's own read-only check (target: 1.0 — any leak here is a defense-in-
      depth failure, not just a quality regression).

  CORRECCIÓN:
    - row_count_match_rate: fraction of VALID_QUERIES where the executor returned exactly
      the expected number of rows.
    - answer_contains_expected_rate: fraction of VALID_QUERIES (with non-empty
      expected_substrings) whose natural-language answer actually mentions the expected
      value(s) — a cheap proxy for "the LLM summary reflects the real data" without a
      second LLM-as-judge call.

  EFICIENCIA (costo/latencia, relevante al usar modelos open-source/gratuitos):
    - average_latency_ms, total tokens in/out, and estimated USD cost using Groq's
      published rate for openai/gpt-oss-20b (input $0.10 / output $0.50 per 1M tokens,
      as of Sep 2026 — update PRICE_* below if you change GROQ_MODEL_NAME).

Empty-result cases are excluded from the cost total by design: the Ejecutor skips the LLM
call entirely when row_count == 0, so they correctly contribute $0 and are not a bug.
"""
import os
import sys
import statistics
from datetime import datetime

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.agent_ejecutor import AgentEjecutor
from src.project_paths import resolve_results_path
from src.services.pricing import estimate_cost
from evaluation.ejecutor_eval_set import VALID_QUERIES, ADVERSARIAL_QUERIES

GROQ_MODEL_NAME = "openai/gpt-oss-20b"


def _contains_all(answer: str, substrings: list) -> bool:
    answer_lower = answer.lower()
    return all(s.lower() in answer_lower for s in substrings)


def evaluate_valid_queries(agent: AgentEjecutor) -> list:
    results = []
    for case in VALID_QUERIES:
        r = agent.run(case["question"], case["sql"])

        if not r["success"]:
            results.append({
                "name": case["name"],
                "question": case["question"],
                "ok": False,
                "reason": f"unexpected failure: [{r['error_type']}] {r['error']}",
                "row_count_match": False,
                "answer_ok": False,
                "latency_ms": 0.0,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "answer": None,
            })
            print(f"❌ {case['name']:<28} EXECUTION FAILED: {r['error']}")
            continue

        row_count_match = r["row_count"] == case["expected_row_count"]
        answer_ok = (
            _contains_all(r["answer"], case["expected_substrings"])
            if case["expected_substrings"]
            else True
        )

        results.append({
            "name": case["name"],
            "question": case["question"],
            "ok": row_count_match and answer_ok,
            "reason": None,
            "row_count_match": row_count_match,
            "answer_ok": answer_ok,
            "latency_ms": r["elapsed_ms"],
            "usage": r["usage"],
            "answer": r["answer"],
        })

        status = "✅" if (row_count_match and answer_ok) else "⚠️"
        print(f"{status} {case['name']:<28} rows={r['row_count']}/{case['expected_row_count']} "
              f"| {r['elapsed_ms']:.0f}ms | \"{r['answer'][:80]}\"")

    return results


def evaluate_adversarial_queries(agent: AgentEjecutor) -> list:
    results = []
    for case in ADVERSARIAL_QUERIES:
        r = agent.run(question="(adversarial probe, no NL question needed)", sql=case["sql"])
        blocked = (not r["success"]) and r["error_type"] == "security"

        results.append({
            "name": case["name"],
            "sql": case["sql"],
            "blocked": blocked,
            "actual_error_type": None if r["success"] else r["error_type"],
        })

        status = "✅ blocked" if blocked else "🚨 NOT BLOCKED"
        print(f"{status:<14} {case['name']:<28} {case['sql'][:60]}")

    return results


def summarize(valid_results: list, adversarial_results: list) -> dict:
    n_valid = len(valid_results)
    n_adv = len(adversarial_results)

    executed = [r for r in valid_results if r["reason"] is None]
    latencies = [r["latency_ms"] for r in executed if r["latency_ms"] > 0]

    total_input_tok = sum(r["usage"]["input_tokens"] for r in executed)
    total_output_tok = sum(r["usage"]["output_tokens"] for r in executed)
    total_cost = estimate_cost("groq/openai/gpt-oss-20b", total_input_tok, total_output_tok)

    return {
        "n_valid_queries": n_valid,
        "execution_success_rate": round(sum(r["reason"] is None for r in valid_results) / n_valid, 3),
        "row_count_match_rate": round(sum(r["row_count_match"] for r in valid_results) / n_valid, 3),
        "answer_contains_expected_rate": round(sum(r["answer_ok"] for r in valid_results) / n_valid, 3),
        "average_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "n_adversarial_queries": n_adv,
        "security_block_rate": round(sum(r["blocked"] for r in adversarial_results) / n_adv, 3),
        "total_input_tokens": total_input_tok,
        "total_output_tokens": total_output_tok,
        "total_cost_usd": round(total_cost, 6),
        "model": f"groq/{GROQ_MODEL_NAME}",
    }


def run():
    agent = AgentEjecutor(provider="groq", model_name=GROQ_MODEL_NAME)

    print("=" * 70)
    print("VALID QUERIES")
    print("=" * 70)
    valid_results = evaluate_valid_queries(agent)

    print("\n" + "=" * 70)
    print("ADVERSARIAL QUERIES (must all be blocked)")
    print("=" * 70)
    adversarial_results = evaluate_adversarial_queries(agent)

    summary = summarize(valid_results, adversarial_results)

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if summary["security_block_rate"] < 1.0:
        print("\n🚨 WARNING: not every adversarial query was blocked. This is a security "
              "regression, not a quality one — investigate before deploying.")

    metrics_path = resolve_results_path("metricas.txt")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"MEASUREMENT: Ejecutor evaluation (stage 7)\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Label: EJECUTOR ({summary['model']})\n\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n  Adversarial details:\n")
        for r in adversarial_results:
            mark = "✅" if r["blocked"] else "🚨 NOT BLOCKED"
            f.write(f"    {mark} {r['name']}\n")
    print(f"\n✅ Metrics added to the history -> {metrics_path}")

    results_path = resolve_results_path("ejecutor_eval_results.json")
    import json
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "valid_results": valid_results,
            "adversarial_results": adversarial_results,
            "summary": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved -> {results_path}")

    return summary


if __name__ == "__main__":
    run()