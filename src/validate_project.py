#!/usr/bin/env python3
"""
Project validator: verifies that all files, imports, and configuration are correct.
Run from the project root:
    python src/validate_project.py
"""
import sys
import os
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.project_paths import (
    PROJECT_ROOT,
    DATA_DIR,
    ARTIFACTS_DIR,
    RESULTS_DIR,
    resolve_db_path,
    resolve_artifact_path,
    resolve_results_path,
    resolve_graph_path,
)


def validate_directory_structure():
    """Checks whether the directory structure is correct."""
    print("\n" + "=" * 70)
    print("1️⃣  VALIDATING DIRECTORY STRUCTURE")
    print("=" * 70)

    required_dirs = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "src/": PROJECT_ROOT / "src",
        "evaluation/": PROJECT_ROOT / "evaluation",
        "artifacts/": ARTIFACTS_DIR,
        "data/": DATA_DIR,
        "results/": RESULTS_DIR,
    }

    all_ok = True
    for name, path in required_dirs.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name:20s} {path}")
        if not exists and name not in ["artifacts/", "results/"]:
            all_ok = False

    return all_ok


def validate_required_files():
    """Checks whether the required files exist."""
    print("\n" + "=" * 70)
    print("2️⃣  VALIDATING REQUIRED FILES")
    print("=" * 70)

    required_files = {
        "data/chinook.db": resolve_db_path("chinook.db"),
        "src/project_paths.py": PROJECT_ROOT / "src" / "project_paths.py",
        "src/explorador.py": PROJECT_ROOT / "src" / "explorador.py",
        "src/extract_catalog.py": PROJECT_ROOT / "src" / "extract_catalog.py",
        "src/build_graph.py": PROJECT_ROOT / "src" / "build_graph.py",
        "src/agent_explorador_langchain.py": PROJECT_ROOT / "src" / "agent_explorador_langchain.py",
        "evaluation/eval_set.py": PROJECT_ROOT / "evaluation" / "eval_set.py",
        "evaluation/run_eval.py": PROJECT_ROOT / "evaluation" / "run_eval.py",
    }

    all_ok = True
    for name, path in required_files.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name:40s}")
        if not exists:
            all_ok = False

    return all_ok


def validate_imports():
    """Checks whether key imports work."""
    print("\n" + "=" * 70)
    print("3️⃣  VALIDATING IMPORTS")
    print("=" * 70)

    tests = [
        ("src.project_paths", ["PROJECT_ROOT", "resolve_artifact_path"]),
        ("src.explorador", ["Explorador"]),
        ("src.build_graph", ["build_graph"]),
        ("src.extract_catalog", ["extract_catalog"]),
        ("src.agent_explorador_langchain", ["AgentExploradorLC", "build_llm"]),
        ("evaluation.eval_set", ["EVAL_SET"]),
    ]

    all_ok = True
    for module_name, items in tests:
        try:
            module = __import__(module_name, fromlist=items)
            for item in items:
                if not hasattr(module, item):
                    print(f"  ❌ {module_name}.{item}")
                    all_ok = False
                else:
                    print(f"  ✅ {module_name}.{item}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            all_ok = False

    return all_ok


def validate_pipeline_state():
    """Checks the current state of the pipeline."""
    print("\n" + "=" * 70)
    print("4️⃣  PIPELINE STATUS")
    print("=" * 70)

    artifacts = {
        "catalog_raw.json": resolve_artifact_path("catalog_raw.json"),
        "catalog_enriched_llm.json": resolve_artifact_path("catalog_enriched_llm.json"),
        "graph.pkl": resolve_graph_path(),
    }

    for name, path in artifacts.items():
        exists = path.exists()
        status = "✅" if exists else "⏳"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if exists else "(not created yet)"
        print(f"  {status} {name:30s} {size}")

    enriched_exists = any(resolve_artifact_path(f).exists() for f in [
        "catalog_enriched_llm.json",
        "catalog_enriched.json",
    ])

    return enriched_exists


def validate_configuration():
    """Checks environment-variable configuration."""
    print("\n" + "=" * 70)
    print("5️⃣  CONFIGURATION")
    print("=" * 70)

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        status = "✅"
        msg = f"(configured, {len(groq_key)} chars)"
    else:
        status = "⏳"
        msg = "(not set - will fallback to Ollama if using LangChain)"

    print(f"  {status} GROQ_API_KEY {msg}")

    return True


def main():
    print("\n🔍 PROJECT AUDIT")
    print("=" * 70)

    results = []
    results.append(("Directory structure", validate_directory_structure()))
    results.append(("Required files", validate_required_files()))
    results.append(("Imports", validate_imports()))
    results.append(("Pipeline status", validate_pipeline_state()))
    results.append(("Configuration", validate_configuration()))

    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:10s} {name}")

    all_ok = all(result for _, result in results)

    print("\n" + "=" * 70)
    if all_ok:
        print("✅ PROJECT VALIDATED - Ready to run")
        print("\nNext steps:")
        print("  1. python src/extract_catalog.py")
        print("  2. python src/enrich_with_LLM_langchain.py")
        print("  3. python src/build_graph.py")
        print("  4. python evaluation/run_eval.py")
    else:
        print("❌ PROBLEMS DETECTED - Review above")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
