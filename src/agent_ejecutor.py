"""
Stage 7: Ejecutor (executor) agent.

Receives an already-validated SQL query (stage 6, Validador) and the original user
question, executes the query against the real database, and returns the result both as
raw rows and as a natural-language answer.

Two priorities drove every decision here, per project constraints:

1) SEGURIDAD E INTEGRIDAD (defense in depth)
   - Never trust that the previous stage (Validador) ran, or ran correctly. This module
     re-validates that the SQL is a single, plain read-only statement before touching the
     database (`validate_readonly`).
   - The database connection itself is opened in SQLite's engine-level read-only mode
     (`mode=ro`), not just "we promise not to write." Even a query that slipped past every
     text-level check cannot mutate data on a mode=ro connection.
   - Hard `timeout_s` (busy_timeout) and hard `max_rows` cap, both enforced independently
     of anything the Generador/Validador stages may or may not have injected into the SQL
     itself (e.g. a missing LIMIT).

2) EFICIENCIA / CONSUMO DE TOKENS (open-source models are not free of cost or GPU time)
   - The natural-language summary step is the only LLM call in this stage, and it is
     deliberately fed a *bounded, compact* summary of the result (row count + column names
     + at most LLM_SAMPLE_ROWS sample rows) instead of the full result set. A query capped
     at `max_rows` (e.g. 200) could still blow up the prompt if dumped verbatim; the sample
     is independent of max_rows so token usage stays flat regardless of how big the allowed
     result window is.
   - If the result set is empty, we skip the LLM call entirely and answer deterministically
     — no reason to spend tokens stating "there are no results."
"""
import os
import re
import sys
import sqlite3
import time
from pathlib import Path

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.project_paths import resolve_db_path
from src.services.agent_services import build_llm


class SecurityError(Exception):
    """Raised when a query fails the executor's own read-only safety check."""


# Overlaps on purpose with the Validador's deterministic layer (stage 6). Redundancy here
# is the point: this is the last line of defense before a query touches real data.
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)

DEFAULT_TIMEOUT_S = 5
DEFAULT_MAX_ROWS = 200
LLM_SAMPLE_ROWS = 10  # rows actually shown to the LLM — independent of max_rows


def _strip_sql_literals(sql: str) -> str:
    """Remove quoted strings and comments before scanning for forbidden keywords.

    This avoids false positives such as "SELECT * FROM Track WHERE Name = 'DELETE me'",
    while still catching write keywords used in actual SQL syntax or after comments."""
    result = []
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "-" and i + 1 < len(sql) and sql[i + 1] == "-":
            i += 2
            while i < len(sql) and sql[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and i + 1 < len(sql) and sql[i + 1] == "*":
            i += 2
            while i + 1 < len(sql) and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2 if i + 1 < len(sql) else 1
            continue
        if ch in "'\"":
            quote = ch
            result.append(" ")
            i += 1
            while i < len(sql):
                if sql[i] == "\\":
                    i += 2
                    continue
                if sql[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def validate_readonly(sql: str) -> None:
    """Defense-in-depth check: only a single, plain SELECT (or WITH ... SELECT) is allowed."""
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1].strip()

    if not stripped:
        raise SecurityError("Empty query.")

    if ";" in stripped:
        raise SecurityError("Multiple statements are not allowed.")

    if not stripped.upper().startswith(("SELECT", "WITH")):
        raise SecurityError("Only SELECT (or WITH ... SELECT) statements are allowed.")

    if FORBIDDEN_KEYWORDS.search(_strip_sql_literals(stripped)):
        raise SecurityError("The query contains a forbidden write/DDL keyword.")


def _install_query_timeout(conn: sqlite3.Connection, timeout_s: float) -> None:
    """Abort long-running SQLite statements by checking a real deadline in the progress handler."""
    deadline = time.perf_counter() + timeout_s

    def handler() -> int:
        return 1 if time.perf_counter() > deadline else 0

    conn.set_progress_handler(handler, 1000)


def _readonly_connection(db_path: str, timeout_s: int) -> sqlite3.Connection:
    """Opens the DB in SQLite's own read-only mode (mode=ro) — an engine-level guarantee,
    not just an application convention. Even a query that passed validate_readonly()
    incorrectly cannot write through this connection."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
    conn.row_factory = sqlite3.Row
    _install_query_timeout(conn, timeout_s)
    return conn


def execute_query(
    sql: str,
    db_path: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict:
    """Executes a read-only SQL query with hard limits.

    Returns {columns, rows, row_count, truncated, elapsed_ms}.
    Raises SecurityError if the query fails the read-only check, or sqlite3.Error on
    timeout / genuine SQL errors (both are expected, recoverable outcomes for the caller).
    """
    validate_readonly(sql)
    db_path = db_path or str(resolve_db_path())

    conn = _readonly_connection(db_path, timeout_s)
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_s * 1000)}")
        cursor = conn.cursor()

        t0 = time.perf_counter()
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise sqlite3.OperationalError("interrupted") from exc
            raise
        columns = [d[0] for d in cursor.description] if cursor.description else []

        # Fetch one row beyond the cap to detect truncation without loading the full result.
        rows = cursor.fetchmany(max_rows + 1)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        return {
            "columns": columns,
            "rows": [dict(r) for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
            "elapsed_ms": round(elapsed_ms, 1),
        }
    finally:
        conn.close()


SUMMARY_PROMPT_TEMPLATE = """You are a data assistant. A user asked a question, a read-only \
SQL query was already executed against the database, and below is a compact summary of the \
result. Answer the user's question in natural language ({language}), in 1-3 sentences, using \
ONLY the data given below. Do not mention SQL, tables, or column names literally unless the \
question is explicitly about the schema. Be direct and specific (name the actual values).

Question: {question}

Result columns: {columns}
Row count returned: {row_count}{truncated_note}
Sample rows (up to {sample_n}):
{sample}
"""


def _format_sample(rows: list, n: int) -> str:
    sample = rows[:n]
    if not sample:
        return "(no rows)"
    return "\n".join(", ".join(f"{k}={v}" for k, v in r.items()) for r in sample)


def summarize_result(
    question: str,
    result: dict,
    llm=None,
    provider: str = "groq",
    model_name: str = "openai/gpt-oss-20b",
    language: str = "Spanish",
) -> tuple:
    """Produces the final natural-language answer from a bounded sample of the result —
    never the full result set — to keep token usage flat and predictable.

    Returns (answer, usage), where usage is {"input_tokens": int, "output_tokens": int}
    (zeros when the result was empty, since we skip the LLM call entirely in that case).
    """
    zero_usage = {"input_tokens": 0, "output_tokens": 0}
    if result["row_count"] == 0:
        answer = (
            "No se encontraron resultados para esa consulta."
            if language == "Spanish"
            else "No results were found for that query."
        )
        return answer, zero_usage

    llm = llm or build_llm(provider=provider, model_name=model_name)

    truncated_note = (
        "\nNote: the result was truncated at the row limit; more rows may exist in the "
        "database beyond what is shown here." if result["truncated"] else ""
    )

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        language=language,
        question=question,
        columns=", ".join(result["columns"]),
        row_count=result["row_count"],
        truncated_note=truncated_note,
        sample_n=LLM_SAMPLE_ROWS,
        sample=_format_sample(result["rows"], LLM_SAMPLE_ROWS),
    )

    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    answer = content.strip() if isinstance(content, str) else str(content).strip()

    # langchain-core standardizes usage_metadata on AIMessage across providers (Groq,
    # Ollama, OpenAI-compatible). Fall back to zeros if a given integration doesn't set it
    # (e.g. some local Ollama models) rather than failing the whole call over cost bookkeeping.
    raw_usage = getattr(response, "usage_metadata", None) or {}
    usage = {
        "input_tokens": raw_usage.get("input_tokens", 0),
        "output_tokens": raw_usage.get("output_tokens", 0),
    }
    return answer, usage


class AgentEjecutor:
    """Stage 7 agent: executes a validated SQL query and returns a natural-language answer.

    One LLM instance is built once per agent (not per call) to avoid re-establishing the
    provider client on every question — another small efficiency choice given free/open
    providers are rate-limited.
    """

    def __init__(
        self,
        db_path: str | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        provider: str = "groq",
        model_name: str = "openai/gpt-oss-20b",
    ):
        self.db_path = db_path or str(resolve_db_path())
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._llm = build_llm(provider=provider, model_name=model_name)

    def run(self, question: str, sql: str, language: str = "Spanish") -> dict:
        t0 = time.perf_counter()
        try:
            result = execute_query(
                sql, db_path=self.db_path, max_rows=self.max_rows, timeout_s=self.timeout_s
            )
        except SecurityError as e:
            return {
                "success": False,
                "error_type": "security",
                "error": str(e),
                "answer": None,
                "question": question,
                "sql": sql,
            }
        except sqlite3.Error as e:
            error_type = "timeout" if "interrupted" in str(e).lower() else "sql_error"
            return {
                "success": False,
                "error_type": error_type,
                "error": str(e),
                "answer": None,
                "question": question,
                "sql": sql,
            }

        answer, usage = summarize_result(question, result, llm=self._llm, language=language)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "success": True,
            "question": question,
            "sql": sql,
            "answer": answer,
            "row_count": result["row_count"],
            "truncated": result["truncated"],
            "columns": result["columns"],
            "rows": result["rows"],
            "usage": usage,
            "elapsed_ms": elapsed_ms,
        }


AgentExecutor = AgentEjecutor  # alias, mirroring Explorer/Explorador in the rest of the codebase


if __name__ == "__main__":
    # Manual smoke test against chinook.db. Requires GROQ_API_KEY unless you swap provider.
    agent = AgentEjecutor()

    cases = [
        {
            "question": "¿Cuál es el total facturado por país?",
            "sql": (
                "SELECT BillingCountry, ROUND(SUM(Total), 2) AS total "
                "FROM Invoice GROUP BY BillingCountry ORDER BY total DESC LIMIT 5"
            ),
        },
        {
            "question": "Borra todas las facturas",
            "sql": "DELETE FROM Invoice",
        },
    ]

    for case in cases:
        print(f"\n❓ {case['question']}")
        result = agent.run(case["question"], case["sql"])
        if result["success"]:
            print(f"  ✅ {result['answer']}")
            print(f"  ⏱️  {result['elapsed_ms']}ms | rows={result['row_count']} | truncated={result['truncated']}")
        else:
            print(f"  ⛔ [{result['error_type']}] {result['error']}")