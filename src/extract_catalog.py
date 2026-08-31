"""
Stage 1: technical catalog extraction.
Deterministic — does not use an LLM. It reads the metadata SQLite already exposes
(PRAGMA table_info, foreign_key_list).
"""
import sqlite3
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.project_paths import resolve_db_path, resolve_artifact_path


def extract_catalog(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [r[0] for r in cur.fetchall()]

    catalog = {}
    for table in table_names:
        cur.execute(f"PRAGMA table_info('{table}')")
        columns = []
        for cid, name, ctype, notnull, dflt, pk in cur.fetchall():
            columns.append({
                "name": name,
                "type": ctype,
                "is_pk": bool(pk),
            })

        cur.execute(f"PRAGMA foreign_key_list('{table}')")
        fks = []
        for row in cur.fetchall():
            # row: id, seq, table, from, to, on_update, on_delete, match
            fks.append({
                "from_column": row[3],
                "to_table": row[2],
                "to_column": row[4],
            })
        for fk in fks:
            for col in columns:
                if col["name"] == fk["from_column"]:
                    col["is_fk"] = True
                    col["references"] = f"{fk['to_table']}.{fk['to_column']}"

        cur.execute(f"SELECT COUNT(*) FROM '{table}'")
        row_count = cur.fetchone()[0]

        catalog[table] = {
            "table": table,
            "columns": columns,
            "foreign_keys": fks,
            "row_count": row_count,
            "description": None,
        }

    conn.close()
    return catalog


if __name__ == "__main__":
    db_path = resolve_db_path("chinook.db")
    output_path = resolve_artifact_path("catalog_raw.json")
    catalog = extract_catalog(str(db_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Catalog extracted: {len(catalog)} tables -> {output_path}")
    for t, info in catalog.items():
        print(f"  {t}: {len(info['columns'])} columns, {len(info['foreign_keys'])} FKs, {info['row_count']} rows")
