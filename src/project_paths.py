from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_db_path(filename: str = "chinook.db") -> Path:
    return ensure_dir(DATA_DIR) / filename


def resolve_artifact_path(filename: str) -> Path:
    return ensure_dir(ARTIFACTS_DIR) / filename


def resolve_results_path(filename: str) -> Path:
    return ensure_dir(RESULTS_DIR) / filename


def resolve_graph_path() -> Path:
    candidates = [
        resolve_artifact_path("graph.pkl"),
        PROJECT_ROOT / "graph.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return resolve_artifact_path("graph.pkl")
