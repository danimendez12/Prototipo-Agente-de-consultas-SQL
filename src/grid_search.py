"""
Grid search over the retrieve() hyperparameters:
  min_score, relative_gap, neighbor_relative_gap, neighbor_floor_ceiling

Instead of tuning one value at a time by hand (which already showed inconsistent behavior because
of parameter interactions), we test all reasonable combinations and report the best ones according
to exact_match_rate and F1 (precision/recall balance) over the expansion set, which is what is
actually passed to the SQL generator.
"""
import os
import sys
import pickle
import itertools
import statistics

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.explorador import Explorador
from src.project_paths import resolve_graph_path
from evaluation.eval_set import EVAL_SET


def precision_recall(retrieved: set, expected: set) -> tuple:
    if not retrieved:
        return 0.0, 0.0
    tp = len(retrieved & expected)
    precision = tp / len(retrieved)
    recall = tp / len(expected) if expected else 1.0
    return precision, recall


def f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_config(explorador, params):
    precisions, recalls, f1s, exact_matches = [], [], [], []
    for case in EVAL_SET:
        ctx = explorador.retrieve(case["question"], **params)
        expanded = set(ctx["expanded_with_graph"])
        p, r = precision_recall(expanded, case["expected_tables"])
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1(p, r))
        exact_matches.append(expanded == case["expected_tables"])

    return {
        "params": params,
        "precision": round(statistics.mean(precisions), 3),
        "recall": round(statistics.mean(recalls), 3),
        "f1": round(statistics.mean(f1s), 3),
        "exact_match_rate": round(sum(exact_matches) / len(exact_matches), 3),
    }


def run_grid_search():
    graph_path = resolve_graph_path()
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    explorador = Explorador(graph)

    grid = {
        "min_score": [0.35, 0.45, 0.5, 0.55],
        "relative_gap": [0.6, 0.75, 0.85, 0.9],
        "neighbor_relative_gap": [0.4, 0.5, 0.6, 0.7, 0.8],
        "neighbor_floor_ceiling": [0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"Testing {len(combos)} combinations...")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        result = evaluate_config(explorador, params)
        results.append(result)

    # Sort by F1 first (precision/recall balance), then exact_match as a tiebreaker.
    results.sort(key=lambda r: (r["f1"], r["exact_match_rate"]), reverse=True)

    print("\nTop 10 configurations by F1:")
    print(f"{'F1':>6} {'P':>6} {'R':>6} {'Exact%':>8}  Params")
    for r in results[:10]:
        print(f"{r['f1']:>6} {r['precision']:>6} {r['recall']:>6} {r['exact_match_rate']*100:>7.1f}%  {r['params']}")

    return results


if __name__ == "__main__":
    results = run_grid_search()