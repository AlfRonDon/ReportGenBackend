from __future__ import annotations

import sqlite3
from pathlib import Path


def get_parent_child_info(db_path: Path) -> dict:
    with sqlite3.connect(str(db_path)) as con:
        cur = con.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        tables = [r[0] for r in cur.fetchall()]

        cols: dict[str, list[str]] = {}
        for t in tables:
            cur.execute(f"PRAGMA table_info('{t}')")
            cols[t] = [row[1] for row in cur.fetchall()]

        preferred_child, preferred_parent = "batch_lines", "batches"
        if preferred_child in tables and preferred_parent in tables:
            child, parent = preferred_child, preferred_parent
        else:
            child = parent = None
            for t in tables:
                cur.execute(f"PRAGMA foreign_key_list('{t}')")
                rows = cur.fetchall()
                if rows:
                    child = t
                    parent = rows[0][2]
                    break

    if not child or not parent:
        raise RuntimeError("Could not determine parent/child tables.")

    child_cols = cols.get(child, [])
    parent_cols = cols.get(parent, [])
    common = sorted(set(child_cols).intersection(parent_cols))

    return {
        "child table": child,
        "parent table": parent,
        "child_columns": child_cols,
        "parent_columns": parent_cols,
        "common_names": common,
    }


def qident(name: str) -> str:
    import re
    _ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    return name if _ident_re.match(name) else f'"{name.replace("\"", "\"\"")}"'

