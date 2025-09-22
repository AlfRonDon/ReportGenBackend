from __future__ import annotations

import base64
import json
import re
import sqlite3
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from app.core.config import settings
from app.services.db_stage import qident

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _client():
    if settings.MOCK_OPENAI or not settings.OPENAI_API_KEY or OpenAI is None:
        return None
    return OpenAI()


def extract_first_batch_block(html_text: str) -> str:
    m = re.search(r'(?is)<section\s+class=["\']batch-block["\']\s*>(.*?)</section>', html_text)
    if not m:
        raise ValueError("No <section class='batch-block'> found in template.")
    inner = m.group(1)
    return '<section class="batch-block">' + inner + '</section>'


def html_without_batch_blocks(html_text: str) -> str:
    pat = re.compile(r'(?is)\s*<section\s+class=["\']batch-block["\']\s*>.*?</section>\s*')
    return pat.sub("", html_text)


def _b64_pdf_page_first(pdf_path: Path, dpi: int = 400) -> str:
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        pix = doc.load_page(0).get_pixmap(dpi=dpi)
        return base64.b64encode(pix.tobytes("png")).decode("ascii")


def llm_pick_with_chat_completions(html_scope: str, catalog: List[str], model: str) -> Dict[str, str]:
    if settings.MOCK_OPENAI:
        # simple deterministic mapping
        return {"Batch No": catalog[0] if catalog else "UNRESOLVED"}
    client = _client()
    prompt = (
        "Task:\nOnly use the HTML SCOPE provided below (the batch block). Ignore content outside it.\n\n"
        "Collect visible header/label texts that correspond to data fields and map EACH to exactly ONE database column from the allow-list CATALOG.\n"
        "Choose strictly from CATALOG (fully-qualified 'table.column'). If no mapping exists, set UNRESOLVED.\n\n"
        "Inputs:\n[HTML SCOPE]\n" + html_scope + "\n\n[CATALOG]\n" + json.dumps(catalog, ensure_ascii=False) + "\n\n"
        "Return strict JSON ONLY mapping '<header>' => 'table.column' or 'UNRESOLVED'."
    )
    resp = client.chat.completions.create(  # type: ignore
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    raw_text = (resp.choices[0].message.content or "").strip()  # type: ignore
    try:
        mapping = json.loads(raw_text)
        if not isinstance(mapping, dict):
            raise ValueError
    except Exception:
        m2 = re.search(r"\{[\s\S]*\}", raw_text)
        if not m2:
            raise RuntimeError("Could not parse JSON mapping from model output.")
        mapping = json.loads(m2.group(0))
        if not isinstance(mapping, dict):
            raise RuntimeError("Parsed JSON is not an object.")
    return mapping


def approval_errors(mapping: Dict[str, str], unresolved_token: str = "UNRESOLVED") -> List[Dict[str, str]]:
    rev: Dict[str, List[str]] = defaultdict(list)
    errors: List[Dict[str, str]] = []
    for label, choice in mapping.items():
        if choice == unresolved_token:
            errors.append({"label": label, "issue": unresolved_token})
        else:
            rev[choice].append(label)
    for colid, labels in rev.items():
        if len(labels) > 1:
            errors.append({"label": "; ".join(labels), "issue": f"Duplicate mapping to {colid}"})
    return errors


# token substitution helpers (outside style/script)
STYLE_OR_SCRIPT_RE = re.compile(r"(?is)(<style\b[^>]*>.*?</style>|<script\b[^>]*>.*?</script>)")


def _apply_outside_styles_scripts(html_in: str, transform_fn: Callable[[str], str]) -> str:
    parts = STYLE_OR_SCRIPT_RE.split(html_in)
    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = transform_fn(parts[i])
    return "".join(parts)


def _sub_token_text(text: str, token: str, val: str) -> str:
    pat = re.compile(r"(\{\{\s*" + re.escape(token) + r"\s*\}\}|\{\s*" + re.escape(token) + r"\s*\})")
    return pat.sub(val, text)


def sub_token(html_in: str, token: str, val: str) -> str:
    return _apply_outside_styles_scripts(html_in, lambda txt: _sub_token_text(txt, token, val))


def _blank_known_tokens_text(text: str, tokens: Iterable[str]) -> str:
    for t in tokens:
        text = re.sub(r"\{\{\s*" + re.escape(t) + r"\s*\}\}", "", text)
        text = re.sub(r"\{\s*" + re.escape(t) + r"\s*\}", "", text)
    return text


def blank_known_tokens(html_in: str, tokens: Iterable[str]) -> str:
    return _apply_outside_styles_scripts(html_in, lambda txt: _blank_known_tokens_text(txt, tokens))


def best_rows_tbody(inner_html: str, allowed_tokens: set) -> Tuple[Optional[re.Match], Optional[str]]:
    tbodys = list(re.finditer(r'(?is)<tbody\b[^>]*>(.*?)</tbody>', inner_html))
    best: Tuple[Optional[re.Match], Optional[str], int] = (None, None, -1)
    for m in tbodys:
        tin = m.group(1)
        hits = 0
        for trm in re.finditer(r"(?is)<tr\b[^>]*>.*?</tr>", tin):
            tr_html = trm.group(0)
            toks = re.findall(r"\{\{\s*([^}\n]+?)\s*\}\}|\{\s*([^}\n]+?)\s*\}", tr_html)
            flat = [a.strip() if a else b.strip() for (a, b) in toks]
            hits += sum(1 for t in flat if t in allowed_tokens)
        if hits > best[2]:
            best = (m, tin, hits)
    if best[0] is not None:
        return best[0], best[1]
    return (tbodys[0], tbodys[0].group(1)) if tbodys else (None, None)


def find_row_template(tbody_inner: str, allowed_tokens: set) -> Tuple[Optional[str], Optional[Tuple[int, int]], List[str]]:
    for m in re.finditer(r"(?is)<tr\b[^>]*>.*?</tr>", tbody_inner):
        tr_html = m.group(0)
        toks = re.findall(r"\{\{\s*([^}\n]+?)\s*\}\}|\{\s*([^}\n]+?)\s*\}", tr_html)
        flat: List[str] = []
        for a, b in toks:
            if a:
                flat.append(a.strip())
            if b:
                flat.append(b.strip())
        flat = [t for t in flat if t in allowed_tokens]
        if flat:
            return tr_html, (m.start(0), m.end(0)), sorted(set(flat), key=len, reverse=True)
    return None, None, []


def majority_table_for_tokens(tokens: Iterable[str], mapping: Dict[str, str]) -> Optional[str]:
    tbls: List[str] = []
    for t in tokens:
        tc = mapping.get(t, "")
        if "." in tc:
            tbls.append(tc.split(".", 1)[0])
    return Counter(tbls).most_common(1)[0][0] if tbls else None


def _parse_key_cols(key_spec: str) -> list[str]:
    return [c.strip() for c in str(key_spec).split(",") if c and c.strip()]


def _key_expr(cols: list[str]) -> str:
    parts = [f"COALESCE(CAST({qident(c)} AS TEXT),'')" for c in cols]
    if not parts:
        return "''"
    expr = parts[0]
    for p in parts[1:]:
        expr = f"{expr} || '|' || {p}"
    return expr


def _split_bid(bid: str, n: int) -> list[str]:
    parts = str(bid).split("|")
    if len(parts) != n:
        raise ValueError(f"Composite key mismatch: expected {n} parts, got {len(parts)} in {bid!r}")
    return parts


def _looks_like_composite_id(x: str, n: int) -> bool:
    return isinstance(x, str) and x.count("|") == (n - 1)


def _discover_batch_ids(db_path: Path, parent_table: str, child_table: str, pcols: list[str], ccols: list[str], parent_date: str, child_date: str, start_date: str, end_date: str) -> List[str]:
    with sqlite3.connect(str(db_path)) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        if len(pcols) == 1:
            parent_sql = f"SELECT DISTINCT {qident(pcols[0])} AS bid FROM {qident(parent_table)} WHERE {qident(parent_date)} BETWEEN ? AND ?"
        else:
            parent_sql = f"SELECT DISTINCT {_key_expr(pcols)} AS bid FROM {qident(parent_table)} WHERE {qident(parent_date)} BETWEEN ? AND ?"

        if len(ccols) == 1:
            child_sql = f"SELECT DISTINCT {qident(ccols[0])} AS bid FROM {qident(child_table)} WHERE {qident(child_date)} BETWEEN ? AND ?"
        else:
            child_sql = f"SELECT DISTINCT {_key_expr(ccols)} AS bid FROM {qident(child_table)} WHERE {qident(child_date)} BETWEEN ? AND ?"

        parent_ids = [r["bid"] for r in cur.execute(parent_sql, (start_date, end_date))]
        child_ids = [r["bid"] for r in cur.execute(child_sql, (start_date, end_date))]
        all_ids = sorted({str(x) for x in (parent_ids + child_ids)})

        if len(all_ids) <= 1:
            if len(pcols) == 1:
                p_all = f"SELECT DISTINCT {qident(pcols[0])} AS bid FROM {qident(parent_table)}"
            else:
                p_all = f"SELECT DISTINCT {_key_expr(pcols)} AS bid FROM {qident(parent_table)}"
            if len(ccols) == 1:
                c_all = f"SELECT DISTINCT {qident(ccols[0])} AS bid FROM {qident(child_table)}"
            else:
                c_all = f"SELECT DISTINCT {_key_expr(ccols)} AS bid FROM {qident(child_table)}"
            parent_ids = [r["bid"] for r in cur.execute(p_all)]
            child_ids = [r["bid"] for r in cur.execute(c_all)]
            all_ids = sorted({str(x) for x in (parent_ids + child_ids)})
        return all_ids


def build_contract_with_llm(schema_info: dict, batch_block_html: str, pdf_path: Path, model: str) -> Dict[str, Any]:
    if settings.MOCK_OPENAI:
        # Minimal contract for tests
        child = schema_info["child table"]
        parent = schema_info["parent table"]
        pcols = schema_info["parent_columns"][0] if schema_info["parent_columns"] else "id"
        ccols = schema_info["child_columns"][0] if schema_info["child_columns"] else "id"
        return {
            "mapping": {"name": f"{child}.{ccols}", "set": f"{child}.{ccols}"},
            "join": {"parent_table": parent, "parent_key": pcols, "child_table": child, "child_key": ccols},
            "date_columns": {parent: pcols, child: ccols},
            "header_tokens": [],
            "row_tokens": ["name", "set"],
            "totals": {},
            "row_order": ["ROWID"],
            "literals": {},
        }
    client = _client()
    image_b64 = _b64_pdf_page_first(pdf_path)
    prompt = f"""
You are given three inputs:

[SCHEMA_INFO]
{schema_info}

[HTML_BLOCK]
{batch_block_html}

[PDF_IMAGE]
(see attached image)

Goal:
From SCHEMA_INFO, HTML_BLOCK (one batch section), and the PDF image, return everything needed to fill the batch details (header + row template + per-batch totals).

Return STRICT JSON ONLY with these keys:
{{
  "mapping": {{ "<token>": "table.column" }},
  "join": {{ "parent_table": "...", "parent_key": "...", "child_table": "...", "child_key": "..." }},
  "date_columns": {{ "<table>": "<date_or_timestamp_col>" }},
  "header_tokens": ["<token>"],
  "row_tokens": ["<token>"],
  "totals": {{ "<token>": "table.column" }},
  "row_order": ["<primary_order_col>", "ROWID"],
  "literals": {{ "<token>": "<verbatim text from PDF if not DB-backed>" }}
}}

Rules:
- Use ONLY tables/columns present in SCHEMA_INFO. Do NOT invent.
- Tokens must match EXACTLY as they appear in HTML_BLOCK (without {{ }}).
- date_columns must include entries for every table referenced and MUST include both join.parent_table and join.child_table.
- row_order[0] must be a column from join.child_table; keep "ROWID" fallback.
- mapping values must be simple "table.column" identifiers.
- literals is ONLY for fixed header text visible in PDF image that is NOT DB-backed.
- Do NOT include tokens that are not present in HTML_BLOCK. Avoid duplicates.

Output ONLY JSON, no prose.
"""
    user_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ]
    resp = client.chat.completions.create(  # type: ignore
        model=model,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = resp.choices[0].message.content  # type: ignore
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw or "")
        if not m:
            raise RuntimeError("Could not parse contract JSON from model output.")
        return json.loads(m.group(0))


def fill_document(obj_contract: dict, db_path: Path, template_path: Path, start_date: str, end_date: str, out_html_path: Path) -> str:
    html = template_path.read_text(encoding="utf-8")

    # Extract one prototype batch-block and build shell
    _block_re = re.compile(r'(?is)<section\b[^>]*class\s*=\s*["\'][^"\']*\bbatch-block\b[^"\']*["\'][^>]*>.*?</section>')
    all_blocks = list(_block_re.finditer(html))
    if not all_blocks:
        raise ValueError("Could not find any <section class='batch-block'> blocks.")
    prototype_block = all_blocks[0].group(0).strip()
    start0 = all_blocks[0].start()
    end_last = all_blocks[-1].end()
    shell_prefix = html[:start0] + "<!-- BEGIN:BATCH (auto) -->"
    shell_suffix = "<!-- END:BATCH (auto) -->" + html[end_last:]

    # Unpack contract
    PLACEHOLDER_TO_COL = obj_contract["mapping"]
    JOIN = obj_contract["join"]
    DATE_COLUMNS = obj_contract["date_columns"]
    HEADER_TOKENS = obj_contract["header_tokens"]
    ROW_TOKENS = obj_contract["row_tokens"]
    TOTALS = obj_contract.get("totals", {})
    ROW_ORDER = obj_contract.get("row_order", ["ROWID"]) or ["ROWID"]
    LITERALS = obj_contract.get("literals", {})

    parent_table = JOIN["parent_table"]
    parent_key = JOIN["parent_key"]
    child_table = JOIN["child_table"]
    child_key = JOIN["child_key"]
    parent_date = DATE_COLUMNS[parent_table]
    child_date = DATE_COLUMNS[child_table]
    order_col = ROW_ORDER[0] if ROW_ORDER else "ROWID"

    pcols = _parse_key_cols(parent_key)
    ccols = _parse_key_cols(child_key)

    # Discover batch ids (auto if not provided)
    BATCH_IDS = _discover_batch_ids(db_path, parent_table, child_table, pcols, ccols, parent_date, child_date, start_date, end_date)

    # Pre-compute minimal column sets
    header_cols = sorted({PLACEHOLDER_TO_COL[t].split(".", 1)[1] for t in HEADER_TOKENS if t in PLACEHOLDER_TO_COL})
    row_cols = sorted({PLACEHOLDER_TO_COL[t].split(".", 1)[1] for t in ROW_TOKENS if t in PLACEHOLDER_TO_COL})
    tot_cols = sorted({(TOTALS.get(t) or PLACEHOLDER_TO_COL[t]).split(".", 1)[1] for t in TOTALS.keys() if (t in TOTALS or t in PLACEHOLDER_TO_COL)})

    rendered_blocks: List[str] = []
    for batch_id in BATCH_IDS:
        block_html = prototype_block

        # (a) Header fill (parent row)
        if header_cols:
            if len(pcols) == 1:
                sql = (
                    f"SELECT {', '.join(qident(c) for c in header_cols)} "
                    f"FROM {qident(parent_table)} WHERE {qident(pcols[0])} = ? AND {qident(parent_date)} BETWEEN ? AND ? LIMIT 1"
                )
                hdr_params = (batch_id, start_date, end_date)
            else:
                where = " AND ".join([f"{qident(c)} = ?" for c in pcols])
                sql = (
                    f"SELECT {', '.join(qident(c) for c in header_cols)} FROM {qident(parent_table)} "
                    f"WHERE {where} AND {qident(parent_date)} BETWEEN ? AND ? LIMIT 1"
                )
                hdr_parts = _split_bid(batch_id, len(pcols))
                hdr_params = (*hdr_parts, start_date, end_date)
            con = sqlite3.connect(str(db_path)); con.row_factory = sqlite3.Row
            cur = con.cursor(); cur.execute(sql, hdr_params)
            row = cur.fetchone(); con.close()
            if row:
                r = dict(row)
                for t in HEADER_TOKENS:
                    if t in PLACEHOLDER_TO_COL:
                        col = PLACEHOLDER_TO_COL[t].split(".", 1)[1]
                        val = r.get(col, "")
                        block_html = sub_token(block_html, t, "" if val is None else str(val))

        # (b) Row repeater (child rows)
        allowed_row_tokens = {t for t in PLACEHOLDER_TO_COL.keys() if t not in TOTALS} - set(HEADER_TOKENS)
        tbody_m, tbody_inner = best_rows_tbody(block_html, allowed_row_tokens)
        if tbody_m and tbody_inner:
            row_template, row_span, row_tokens_in_template = find_row_template(tbody_inner, allowed_row_tokens)
            if row_template and row_tokens_in_template:
                row_cols_needed = sorted({PLACEHOLDER_TO_COL[t].split(".", 1)[1] for t in row_tokens_in_template})
                if order_col.upper() != "ROWID" and order_col not in row_cols_needed:
                    row_cols_needed.append(order_col)
                order_clause = "ORDER BY ROWID" if order_col.upper() == "ROWID" else f"ORDER BY {qident(order_col)}, ROWID"
                if len(ccols) == 1:
                    sql = (
                        f"SELECT {', '.join(qident(c) for c in row_cols_needed)} FROM {qident(child_table)} "
                        f"WHERE {qident(ccols[0])} = ? AND {qident(child_date)} BETWEEN ? AND ? {order_clause}"
                    )
                    row_params = (batch_id, start_date, end_date)
                else:
                    where = " AND ".join([f"{qident(c)} = ?" for c in ccols])
                    sql = (
                        f"SELECT {', '.join(qident(c) for c in row_cols_needed)} FROM {qident(child_table)} "
                        f"WHERE {where} AND {qident(child_date)} BETWEEN ? AND ? {order_clause}"
                    )
                    row_parts = _split_bid(batch_id, len(ccols))
                    row_params = (*row_parts, start_date, end_date)
                con = sqlite3.connect(str(db_path)); con.row_factory = sqlite3.Row
                cur = con.cursor(); cur.execute(sql, row_params)
                rows = [dict(r) for r in cur.fetchall()]
                con.close()
                if not rows:
                    maj_table = majority_table_for_tokens(row_tokens_in_template, PLACEHOLDER_TO_COL)
                    if maj_table:
                        date_col = obj_contract.get("date_columns", {}).get(maj_table)
                        if date_col:
                            cols_needed = sorted({PLACEHOLDER_TO_COL[t].split(".", 1)[1] for t in row_tokens_in_template})
                            if date_col not in cols_needed:
                                cols_needed.append(date_col)
                            sql_fb = (
                                f"SELECT {', '.join(qident(c) for c in cols_needed)} FROM {qident(maj_table)} "
                                f"WHERE {qident(date_col)} BETWEEN ? AND ? ORDER BY {qident(date_col)} ASC, ROWID ASC"
                            )
                            con = sqlite3.connect(str(db_path)); con.row_factory = sqlite3.Row
                            cur = con.cursor(); cur.execute(sql_fb, (start_date, end_date))
                            rows = [dict(r) for r in cur.fetchall()]
                            con.close()
                parts: List[str] = []
                for r in rows:
                    tr = row_template
                    for t in row_tokens_in_template:
                        col = PLACEHOLDER_TO_COL[t].split(".", 1)[1]
                        tr = sub_token(tr, t, "" if r.get(col) is None else str(r.get(col)))
                    parts.append(tr)
                new_tbody_inner = tbody_inner[:row_span[0]] + "\n".join(parts) + tbody_inner[row_span[1]:]
                block_html = block_html[:tbody_m.start(1)] + new_tbody_inner + block_html[tbody_m.end(1):]

        # (c) Totals (child SUM)
        if tot_cols:
            exprs = ", ".join([f"COALESCE(SUM({qident(c)}),0) AS {qident(c)}" for c in tot_cols])
            if len(ccols) == 1:
                sql = (
                    f"SELECT {exprs} FROM {qident(child_table)} WHERE {qident(ccols[0])} = ? AND {qident(child_date)} BETWEEN ? AND ?"
                )
                tot_params = (batch_id, start_date, end_date)
            else:
                where = " AND ".join([f"{qident(c)} = ?" for c in ccols])
                sql = (
                    f"SELECT {exprs} FROM {qident(child_table)} WHERE {where} AND {qident(child_date)} BETWEEN ? AND ?"
                )
                tot_parts = _split_bid(batch_id, len(ccols))
                tot_params = (*tot_parts, start_date, end_date)
            con = sqlite3.connect(str(db_path)); con.row_factory = sqlite3.Row
            cur = con.cursor(); cur.execute(sql, tot_params)
            sums = dict(cur.fetchone() or {})
            con.close()
            for token, target in TOTALS.items():
                target = TOTALS.get(token) or PLACEHOLDER_TO_COL[token]
                col = target.split(".", 1)[1]
                v = sums.get(col, 0)
                try:
                    fv = float(v)
                    s = str(int(fv)) if fv.is_integer() else str(fv)
                except Exception:
                    s = "0"
                block_html = sub_token(block_html, token, s)

        rendered_blocks.append(block_html)

    html_multi = shell_prefix + "\n".join(rendered_blocks) + shell_suffix
    for t, s in LITERALS.items():
        html_multi = sub_token(html_multi, t, s)
    ALL_KNOWN_TOKENS = set(HEADER_TOKENS) | set(ROW_TOKENS) | set(TOTALS.keys()) | set(LITERALS.keys())
    html_multi = blank_known_tokens(html_multi, ALL_KNOWN_TOKENS)

    out_html_path.write_text(html_multi, encoding="utf-8")
    return html_multi

