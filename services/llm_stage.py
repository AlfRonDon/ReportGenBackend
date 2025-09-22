from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _client():
    if settings.MOCK_OPENAI or not settings.OPENAI_API_KEY or OpenAI is None:
        return None
    return OpenAI()


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:html|HTML|json|JSON)?\s*([\s\S]*?)```", text)
    return m.group(1).strip() if m else text.strip()


def _mock_json_schema() -> Dict[str, Any]:
    return {"scalars": {}, "blocks": {"rows": ["sl", "name", "set", "ach", "err", "errp"]}, "notes": "mock"}


def request_schema_for_page(page_png: Path, model: str) -> Dict[str, Any]:
    if settings.MOCK_OPENAI:
        return _mock_json_schema()
    client = _client()
    prompt = (
        "Infer a placeholder schema for this PDF page. Identify dynamic fields and repeating blocks. "
        "Return ONLY compact JSON with keys: { 'scalars': { ... }, 'blocks': { 'rows': ['sl','name','set','ach','err','errp'] }, 'notes': '...' }. "
        "Do not generate HTML in this step."
    )
    resp = client.chat.completions.create(  # type: ignore
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(page_png)}"}},
            ]
        }]
    )
    txt = resp.choices[0].message.content.strip()  # type: ignore
    body = strip_code_fences(txt)
    try:
        return json.loads(body)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", body)
        return json.loads(m.group(0)) if m else {"scalars": {}, "blocks": {}, "notes": "(parse_error)"}


def request_initial_html(page_png: Path, schema_json: Dict[str, Any], model: str) -> str:
    if settings.MOCK_OPENAI:
        # Minimal valid HTML with a batch-block marker for later stages
        return (
            "<!doctype html><html><head><meta charset='utf-8'><style>body{font:12px Arial}</style></head>"
            "<body><section class='batch-block'><table><tbody><tr><td>{name}</td><td>{set}</td></tr></tbody></table></section></body></html>"
        )
    client = _client()
    prompt = (
        "Produce a COMPLETE, self-contained HTML document (<!DOCTYPE html>) with inline <style>. "
        "It must visually photocopy the given PDF page image as closely as possible. "
        "Mirror fonts, spacing, borders, alignment, and table layouts. "
        "Tables must use border-collapse, 1px borders, and table-layout: fixed for neat alignment.\n\n"
        "SCHEMA USAGE\n"
        "- Use provided schema for tokens; include a single prototype row inside a tbody within <section class=\"batch-block\">.\n"
        "OUTPUT RULES\n"
        "- Return RAW HTML only (no markdown fences)."
    )
    schema_str = json.dumps(schema_json, ensure_ascii=False, separators=(",", ":"))
    resp = client.chat.completions.create(  # type: ignore
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "SCHEMA:\n" + schema_str},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(page_png)}"}},
            ]
        }]
    )
    return strip_code_fences(resp.choices[0].message.content)  # type: ignore


def request_fix_html(schema_json: Dict[str, Any], ref_png: Path, render_png: Path, current_html: str, ssim_value: float, model: str) -> str:
    if settings.MOCK_OPENAI:
        # Return current html (no change) for deterministic test
        return current_html
    client = _client()
    prompt = (
        f"Compare these images: REFERENCE (PDF page) vs RENDER (current HTML). SSIM={ssim_value:.4f}.\n"
        "Goal: refine the provided HTML/CSS so the render becomes a near-perfect PHOTOCOPY of the reference.\n\n"
        "STRICT RULES\n- Do NOT rename, add, remove, or move SCHEMA placeholders; keep tokens unchanged.\n"
        "VISUAL MATCHING\n- Identify and correct discrepancies across layout, borders, typography, alignment.\n"
        "OUTPUT\n- Return FULL HTML with inline <style> only. No markdown.\n"
    )
    schema_str = json.dumps(schema_json, ensure_ascii=False)
    resp = client.chat.completions.create(  # type: ignore
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "SCHEMA:\n" + schema_str},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(ref_png)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(render_png)}"}},
                {"type": "text", "text": current_html},
            ]
        }]
    )
    return strip_code_fences(resp.choices[0].message.content)  # type: ignore

