"""
Stage 2 (real): semantic enrichment via LLM.

This replaces the handwritten DESCRIPTIONS dictionary with real generation: for each table, we pass
Claude the table name, columns, types, and a sample of rows, then ask for a table description,
ambiguous column descriptions, and example questions a typical user would ask — using tool use to
force a reliable output schema.

The result still enters the `pending_review` state just like the manual version — the difference is
who generates it, not the approval flow.

Requires: pip install anthropic
          export ANTHROPIC_API_KEY=...
"""
import sqlite3
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from anthropic import Anthropic
from src.project_paths import resolve_artifact_path, resolve_db_path

client = Anthropic()

ENRICH_TOOL = {
    "name": "describe_table",
    "description": "Generates the semantic description of a database table",
    "input_schema": {
        "type": "object",
        "properties": {
            "table_description": {
                "type": "string",
                "description": "1-2 sentences explaining what this table represents. If it is easily confused with another table in the same domain, mention EXPLICITLY what distinguishes it."
            },
            "column_descriptions": {
                "type": "object",
                "description": "Map of column_name -> short description, only for columns whose purpose is not obvious from the name (IDs, flags, ambiguous dates, technical fields).",
                "additionalProperties": {"type": "string"}
            },
            "example_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 natural-language questions a typical business user would ask and that would require this table to answer."
            }
        },
        "required": ["table_description", "column_descriptions", "example_questions"]
    }
}


def get_sample_rows(db_path: str, table: str, n: int = 3) -> list:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM '{table}' LIMIT {n}")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows


def enrich_table_with_llm(table_name: str, table_info: dict, sample_rows: list) -> dict:
    prompt = f"""Analyze this database table and generate its semantic documentation.

Table: {table_name}
Columns: {json.dumps(table_info['columns'], ensure_ascii=False)}
Relationships (foreign keys): {json.dumps(table_info['foreign_keys'], ensure_ascii=False)}
Sample rows:
{json.dumps(sample_rows, ensure_ascii=False, indent=2)}

IMPORTANT: if this table could be confused with a similar table in the same domain, include in
`table_description` what clearly distinguishes it.
Do not repeat the same keyword in example_questions for different tables unless it is genuinely
specific to that table — avoid generic phrases that could apply to any table in the domain."""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        tools=[ENRICH_TOOL],
        tool_choice={"type": "tool", "name": "describe_table"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise RuntimeError(f"The model did not return the expected tool call for {table_name}")


ENRICH_ALL_TOOL = {
    "name": "describe_full_schema",
    "description": "Generates the semantic description of all tables in a schema at once, allowing tables that could be confused with each other to be differentiated",
    "input_schema": {
        "type": "object",
        "properties": {
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string"},
                        "table_description": {
                            "type": "string",
                            "description": "1-2 sentences explaining what this table represents. If it is easily confused with ANOTHER table in this same schema, explicitly mention what distinguishes it (e.g. 'unlike X, this table...')."
                        },
                        "column_descriptions": {
                            "type": "object",
                            "additionalProperties": {"type": "string"}
                        },
                        "example_questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "2-3 questions a user would ask and that specifically require THIS table."
                        }
                    },
                    "required": ["table_name", "table_description", "column_descriptions", "example_questions"]
                }
            }
        },
        "required": ["tables"]
    }
}


def enrich_all_at_once(catalog: dict, db_path: str) -> dict:
    """
    A single LLM call with the full schema, so the model can see all tables at once and write
    descriptions that actively distinguish one another — something impossible when each table is
    described in isolation (the problem that caused the weak result in the previous version of this script).
    """
    schema_overview = []
    for table_name, info in catalog.items():
        samples = get_sample_rows(db_path, table_name, n=2)
        schema_overview.append({
            "table": table_name,
            "columns": info["columns"],
            "foreign_keys": info["foreign_keys"],
            "sample_rows": samples,
        })

    prompt = f"""Analyze this complete database schema and generate the semantic
documentation for each table.

Full schema ({len(schema_overview)} tables):
{json.dumps(schema_overview, ensure_ascii=False, indent=2)}

CRITICAL INSTRUCTIONS:
- You have visibility of ALL tables at once. Use it: if two tables could be confused with each other,
  the description of EACH ONE must explicitly explain what distinguishes it from the other.
- The example questions for one table must not reuse keywords that appear equally well in another
  table's example questions unless that word is truly unique to this table in the domain.
- Generate output for the {len(schema_overview)} tables in the same order they appear above."""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4000,
        tools=[ENRICH_ALL_TOOL],
        tool_choice={"type": "tool", "name": "describe_full_schema"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use":
            results_by_table = {t["table_name"]: t for t in block.input["tables"]}
            break
    else:
        raise RuntimeError("The model did not return the expected tool call")

    for table_name, table_info in catalog.items():
        result = results_by_table.get(table_name)
        if not result:
            print(f"  WARNING: the LLM did not generate a description for {table_name}")
            continue
        table_info["description"] = result["table_description"]
        table_info["example_questions"] = result["example_questions"]
        table_info["status"] = "pending_review"
        col_descs = result.get("column_descriptions", {})
        for col in table_info["columns"]:
            if col["name"] in col_descs:
                col["description"] = col_descs[col["name"]]

    return catalog


def approve_all(catalog: dict) -> dict:
    for table_info in catalog.values():
        table_info["status"] = "active"
    return catalog


if __name__ == "__main__":
    input_path = resolve_artifact_path("catalog_raw.json")
    db_path = resolve_db_path("chinook.db")
    output_path = resolve_artifact_path("catalog_enriched_llm.json")

    with open(input_path, encoding="utf-8") as f:
        catalog = json.load(f)

    catalog = enrich_all_at_once(catalog, str(db_path))

    print("\n" + "=" * 70)
    print("GENERATED DESCRIPTIONS (pending_review) — review them before approving")
    print("=" * 70)
    for t, info in catalog.items():
        print(f"\n[{info['status']}] {t}")
        print(f"  Description: {info['description']}")
        print(f"  Example questions: {info['example_questions']}")

    response = input("\nApprove all and save? (y/n): ")
    if response.lower() == "y":
        catalog = approve_all(catalog)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {output_path}")
    else:
        print("Not saved. Adjust the prompt and rerun.")