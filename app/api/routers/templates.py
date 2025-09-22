from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import TemplateCreateResponse, TemplateMeta
from app.services.compare import compare_images
from app.services.llm_stage import request_fix_html, request_initial_html, request_schema_for_page
from app.services.pdf_stage import pdf_to_pngs, render_html_to_png


router = APIRouter(prefix="/templates", tags=["templates"])


def _templates_root() -> Path:
    root = settings.BASE_ARTIFACTS_DIR / "templates"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _template_dir(template_id: str, *, ensure_exists: bool = True) -> Path:
    path = _templates_root() / template_id
    if ensure_exists and not path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return path


def _meta_path(template_id: str) -> Path:
    return _template_dir(template_id) / "meta.json"


def _load_meta(template_id: str) -> Dict:
    meta_file = _meta_path(template_id)
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Template metadata not found")
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _write_meta(template_id: str, data: Dict) -> TemplateMeta:
    meta_file = _meta_path(template_id)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return TemplateMeta(**data)


def _safe_filename(name: str, fallback: str) -> str:
    keep = [c if c.isalnum() or c in {"-", "_", "."} else "_" for c in name]
    candidate = "".join(keep).strip("._")
    return candidate or fallback


def _generate_id(prefix: str = "tpl", length: int = 8) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    return prefix + "_" + "".join(secrets.choice(alphabet) for _ in range(length))


def _normalize_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return sorted({part.strip().lower() for part in raw.split(",") if part.strip()})


async def _render_preview(final_html: Path, preview_path: Path) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    await render_html_to_png(final_html, preview_path)


@router.get("", response_model=List[TemplateMeta])
async def list_templates() -> List[TemplateMeta]:
    templates: List[TemplateMeta] = []
    root = _templates_root()
    for meta_file in root.glob("*/meta.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            templates.append(TemplateMeta(**data))
        except Exception:
            continue
    templates.sort(key=lambda item: item.updated_at, reverse=True)
    return templates


@router.post("", response_model=TemplateCreateResponse)
async def create_template(
    file: UploadFile = File(...),
    name: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
    tags: Optional[str] = Form(default=None),
) -> TemplateCreateResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name")
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in {"pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF templates are supported currently")

    template_id = _generate_id()
    tpl_dir = _template_dir(template_id, ensure_exists=False)
    tpl_dir.mkdir(parents=True, exist_ok=True)
    original_dir = tpl_dir / "original"
    schema_dir = tpl_dir / "schema"
    html_dir = tpl_dir / "html"
    render_dir = tpl_dir / "render"
    diff_dir = tpl_dir / "diff"
    original_dir.mkdir(exist_ok=True)
    schema_dir.mkdir(exist_ok=True)
    html_dir.mkdir(exist_ok=True)
    render_dir.mkdir(exist_ok=True)
    diff_dir.mkdir(exist_ok=True)

    safe_name = _safe_filename(file.filename, f"{template_id}.pdf")
    original_path = original_dir / safe_name
    try:
        original_bytes = await file.read()
        original_path.write_bytes(original_bytes)
    finally:
        await file.close()

    try:
        png_dir = tpl_dir / "png"
        ref_png = pdf_to_pngs(original_path, png_dir, settings.PDF_DPI)[0]
        schema = request_schema_for_page(ref_png, model=settings.OPENAI_MODEL)
        schema_path = schema_dir / "schema_p1.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

        html_v1 = request_initial_html(ref_png, schema, model=settings.OPENAI_MODEL)
        curr_html_path = html_dir / "template_p1_v1.html"
        curr_html_path.write_text(html_v1, encoding="utf-8")

        iterations = max(1, settings.REFINE_ITERS)
        for i in range(iterations):
            render_png = render_dir / f"render_p1_v{i+1}.png"
            diff_png = diff_dir / f"diff_p1_v{i+1}.png"
            await render_html_to_png(curr_html_path, render_png)
            score = compare_images(ref_png, render_png, diff_png)
            refined_html = request_fix_html(
                schema,
                ref_png,
                render_png,
                curr_html_path.read_text(encoding="utf-8"),
                score,
                model=settings.OPENAI_MODEL,
            )
            curr_html_path = html_dir / f"template_p1_v{i+2}.html"
            curr_html_path.write_text(refined_html, encoding="utf-8")

        preview_png = tpl_dir / "preview" / "render.png"
        await _render_preview(curr_html_path, preview_png)
    except Exception as exc:
        shutil.rmtree(tpl_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Template processing failed: {exc}") from exc

    timestamp = datetime.utcnow().isoformat()
    meta = {
        "id": template_id,
        "name": name.strip() if name and name.strip() else Path(file.filename).stem,
        "description": description.strip() if description else "",
        "tags": _normalize_tags(tags),
        "status": "draft",
        "sourceType": "pdf",
        "created_at": timestamp,
        "updated_at": timestamp,
        "preview_url": f"/templates/{template_id}/preview.png",
    }
    _write_meta(template_id, meta)

    return TemplateCreateResponse(id=template_id, status="draft", preview_url=meta["preview_url"])


@router.post("/{template_id}/approve", response_model=TemplateMeta)
async def approve_template(template_id: str, payload: dict | None = Body(default=None)) -> TemplateMeta:
    meta = _load_meta(template_id)
    if payload:
        name = payload.get('name')
        if name:
            meta['name'] = str(name).strip()
        if 'description' in payload:
            meta['description'] = (payload.get('description') or '').strip()
        if 'tags' in payload:
            raw_tags = payload.get('tags')
            if isinstance(raw_tags, list):
                meta['tags'] = sorted({str(t).strip().lower() for t in raw_tags if str(t).strip()})
            elif raw_tags is not None:
                meta['tags'] = _normalize_tags(str(raw_tags))
    meta['status'] = 'approved'
    meta['updated_at'] = datetime.utcnow().isoformat()
    return _write_meta(template_id, meta)


@router.get("/{template_id}/preview.png")
async def get_preview(template_id: str):
    path = _template_dir(template_id) / "preview" / "render.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(str(path), media_type="image/png")


@router.delete("/{template_id}")
async def delete_template(template_id: str):
    tpl_dir = _template_dir(template_id)
    shutil.rmtree(tpl_dir, ignore_errors=True)
    return {"id": template_id, "deleted": True}
