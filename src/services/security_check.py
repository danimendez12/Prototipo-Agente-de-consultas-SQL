"""
Evaluates SQL queries made by a model, checks for security issues, only accepts queries that are safe to run, and only accepts SELECT queries.
Returns the results of the evaluation (True if safe, False otherwise) in case of a security issue, or if the query is not a SELECT query retrieves an explanation of why is not safe.
"""

import re
from typing import Tuple

import sqlglot


def is_select_query(query: str) -> bool:
    """Check if the query is a single safe SELECT/WITH statement."""
    if not query or not query.strip():
        return False

    try:
        parsed = sqlglot.parse_one(query.strip(), read="sqlite")
    except Exception:
        return False

    if parsed is None:
        return False

    return type(parsed).__name__.lower() == "select"


def has_dangerous_keywords(query: str) -> bool:
    """Check for dangerous SQL keywords that should not be in SELECT queries."""
    dangerous_keywords = [
        r"\bDROP\b",
        r"\bDELETE\b",
        r"\bTRUNCATE\b",
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bALTER\b",
        r"\bCREATE\b",
        r"\bEXEC\b",
        r"\bEXECUTE\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bMERGE\b",
        r"\bCALL\b",
        r"\bUNION\b\s+\bSELECT\b",
    ]

    for keyword in dangerous_keywords:
        if re.search(keyword, query, re.IGNORECASE):
            return True
    return False


def has_sql_injection_patterns(query: str) -> bool:
    """Check for common SQL injection patterns and dangerous query patterns."""
    injection_patterns = [
        r"--\s*",
        r"/\*.*?\*/",
        r";\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|CALL|MERGE)\b",
        r"\bOR\b\s*1\s*=\s*1\b",
        r"\bAND\b\s*1\s*=\s*1\b",
        r"'\s*OR\s*'1'\s*=\s*'1'",
        r"'\s*OR\s*1\s*=\s*1",
        r"\bUNION\b\s+\bSELECT\b",
        r"\bxp_\b",
        r"\bsp_\b",
    ]

    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
            return True
    return False


def evaluate_query(query: str) -> Tuple[bool, str]:
    """
    Evaluate if a SQL query is safe to execute.

    Args:
        query: The SQL query to evaluate.

    Returns:
        A tuple of (is_safe: bool, explanation: str).
        If safe, explanation is an empty string.
        If unsafe, explanation describes why the query is not safe.
    """
    if not query or not query.strip():
        return False, "Query is empty."

    query = query.strip()

    try:
        statements = sqlglot.parse(query, read="sqlite")
    except Exception:
        return False, "Query is not valid SQL."

    if len(statements) != 1:
        return False, "Query contains multiple statements and is not allowed."

    if not is_select_query(query):
        return False, "Only SELECT queries are allowed. This query contains other operations."

    if has_dangerous_keywords(query):
        return False, "Query contains dangerous SQL keywords that are not allowed."

    if has_sql_injection_patterns(query):
        return False, "Query contains potential SQL injection patterns."

    return True, ""

