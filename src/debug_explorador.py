# debug_explorador.py
import pickle
from src.explorador import Explorador
from src.project_paths import resolve_graph_path
from evaluation.eval_set import EVAL_SET

graph_path = resolve_graph_path()
with open(graph_path, "rb") as f:
    graph = pickle.load(f)
explorer = Explorador(graph)


def debug_question(question, expected_tables, top_k=5):
    """Detailed retrieval debug for a question."""
    print(f"\n{'='*70}")
    print(f"❓ Question: {question}")
    print('='*70)

    result = explorer.retrieve(question, top_k=top_k, min_score=0.0)

    print("\n📊 Tables by SIMILARITY (top-k):")
    retrieved = result["retrieved_by_similarity"]
    scores = result["scores"]
    for table in retrieved:
        score = scores.get(table, 0)
        mark = "✅" if table in expected_tables else "❌"
        print(f"  {mark} {table:20s} {score:.3f}")

    expanded = result["expanded_with_graph"]
    if len(expanded) > len(retrieved):
        extra_from_graph = set(expanded) - set(retrieved)
        print(f"\n🔗 Expanded via graph: {sorted(extra_from_graph)}")

    print("\n🏆 FULL RANKING (all tables):")
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    for pos, (table, score) in enumerate(sorted_scores, 1):
        mark = "✅" if table in expected_tables else "  "
        status = ""
        if table in retrieved:
            status = " [RETRIEVED]"
        elif table in expanded:
            status = " [EXPANDED]"
        print(f"  {pos:2d}. {mark} {table:20s} {score:.3f}{status}")

    all_retrieved = set(expanded)
    expected_set = set(expected_tables)
    match = all_retrieved & expected_set
    missing = expected_set - all_retrieved
    extra = all_retrieved - expected_set

    print(f"\n📈 Evaluation:")
    print(f"   Expected:    {sorted(expected_tables)}")
    print(f"   Retrieved:   {expanded}")
    print(f"   ✅ Found:    {sorted(match)}")
    if missing:
        print(f"   ❌ Missing:  {sorted(missing)}")
    if extra:
        print(f"   ⚠️  Extra:    {sorted(extra)}")

    recall = len(match) / len(expected_set) if expected_set else 0
    precision = len(match) / len(all_retrieved) if all_retrieved else 0
    print(f"   Recall:  {recall:.1%} | Precision: {precision:.1%}")


print("\n" + "🔍 FULL EVALUATION".center(70, "="))
for case in EVAL_SET:
    debug_question(case["question"], case["expected_tables"], top_k=3)