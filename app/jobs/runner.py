from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict

from app.core.config import settings
from app.core.job_store import store
from app.core.logging import logger
from app.models.schemas import JobConfig
from app.services.pdf_stage import pdf_to_pngs, render_html_to_png
from app.services.llm_stage import request_schema_for_page, request_initial_html, request_fix_html
from app.services.compare import compare_images
from app.services.db_stage import get_parent_child_info
from app.services.fill_stage import (
    extract_first_batch_block,
    html_without_batch_blocks,
    llm_pick_with_chat_completions,
    approval_errors,
    build_contract_with_llm,
    fill_document,
)
from app.services.export_stage import html_to_pdf_async


def _check_cancel(job_id: str):
    st = store.get(job_id)
    if st and st.cancel_requested:
        raise asyncio.CancelledError


async def run_full_pipeline(job: JobConfig, emit: Callable[[str, int, str], None]) -> None:
    job_dir = Path(settings.BASE_ARTIFACTS_DIR) / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. pdf_to_png
        emit("pdf_to_png", 10, "Extracting first page PNG...")
        input_pdf = Path(job.input_pdf) if job.input_pdf else None
        assert input_pdf and input_pdf.exists(), "input.pdf not found"
        png_dir = job_dir / "png"
        png_dir.mkdir(exist_ok=True)
        ref_pngs = pdf_to_pngs(input_pdf, png_dir, dpi=settings.PDF_DPI)
        ref_png = ref_pngs[0]

        _check_cancel(job.job_id)
        # 2. schema
        emit("schema", 15, "Requesting schema for page...")
        schema = request_schema_for_page(ref_png, model=job.model)
        (job_dir / "schema").mkdir(exist_ok=True)
        (job_dir / "schema" / "schema_p1.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

        _check_cancel(job.job_id)
        # 3. initial_html
        emit("initial_html", 20, "Generating initial HTML template...")
        html_v1 = request_initial_html(ref_png, schema, model=job.model)
        (job_dir / "html").mkdir(exist_ok=True)
        curr_html_path = job_dir / "html" / "template_p1_v1.html"
        curr_html_path.write_text(html_v1, encoding="utf-8")

        # 4. refine loop
        iterations = max(1, job.iterations)
        for i in range(1, iterations + 1):
            stage = f"refine_v{i}"
            emit(stage, 20 + int((i / max(iterations, 1)) * 40), f"Refine pass {i}...")
            render_png = job_dir / "png" / f"render_p1_v{i}.png"
            diff_png = job_dir / "diff" / f"diff_p1_v{i}.png"
            diff_png.parent.mkdir(exist_ok=True)
            await render_html_to_png(curr_html_path, render_png)
            ssim_score = compare_images(ref_png, render_png, diff_png)
            next_html = request_fix_html(schema, ref_png, render_png, curr_html_path.read_text(encoding="utf-8"), ssim_score, model=job.model)
            curr_html_path = curr_html_path.with_name(f"template_p1_v{i+1}.html")
            curr_html_path.write_text(next_html, encoding="utf-8")
            if i == 1:
                emit(stage, 20 + int((i / max(iterations, 1)) * 40), f"Refine pass {i}", render_v1_png=str(render_png.resolve()), diff_v1_png=str(diff_png.resolve()))
            _check_cancel(job.job_id)

        # 5. db_discover
        _check_cancel(job.job_id)
        emit("db_discover", 62, "Discovering parent/child schema...")
        db_path = Path(job.db_url.replace("sqlite:///", "").replace("file:", ""))
        schema_info = get_parent_child_info(db_path)
        (job_dir / "schema" / "schema_info.json").write_text(json.dumps(schema_info, indent=2), encoding="utf-8")

        # 6. mapping (labels)
        _check_cancel(job.job_id)
        emit("mapping", 70, "LLM mapping of labels to columns...")
        full_html = curr_html_path.read_text(encoding="utf-8")
        html_scope = html_without_batch_blocks(full_html)
        catalog = [
            *[f"{schema_info['parent table']}.{c}" for c in schema_info["parent_columns"]],
            *[f"{schema_info['child table']}.{c}" for c in schema_info["child_columns"]],
        ]
        mapping = llm_pick_with_chat_completions(html_scope, catalog, model=job.model)
        (job_dir / "schema" / "mapping_pdf_labels.json").write_text(json.dumps({"mapping": mapping}, indent=2, ensure_ascii=False), encoding="utf-8")
        errs = approval_errors(mapping)
        if errs:
            raise RuntimeError("Mapping approval failed: " + "; ".join(f"{e['issue']}: {e['label']}" for e in errs))

        # Build batch-block HTML to feed contract LLM
        batch_block_html = extract_first_batch_block(full_html)
        (job_dir / "html" / "batch_block.html").write_text(batch_block_html, encoding="utf-8")

        # Contract for fill
        _check_cancel(job.job_id)
        contract = build_contract_with_llm(schema_info, batch_block_html, input_pdf, model=job.model)
        (job_dir / "schema" / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

        # 7. fill
        _check_cancel(job.job_id)
        emit("fill", 85, "Filling document with DB data...")
        filled_html_path = job_dir / "filled" / "report_filled.html"
        filled_html_path.parent.mkdir(exist_ok=True)
        fill_document(contract, db_path, curr_html_path, job.start_date, job.end_date, filled_html_path)

        # 8. export_pdf
        _check_cancel(job.job_id)
        emit("export_pdf", 95, "Exporting filled HTML to PDF...")
        final_pdf_path = job_dir / "pdf" / "report_filled_new.pdf"
        final_pdf_path.parent.mkdir(exist_ok=True)
        await html_to_pdf_async(filled_html_path, final_pdf_path)

        artifacts = {
            "schema_json": str((job_dir / 'schema' / 'schema_p1.json').resolve()),
            "initial_html": str((job_dir / 'html' / 'template_p1_v1.html').resolve()),
            "final_html": str(filled_html_path.resolve()),
            "final_pdf": str(final_pdf_path.resolve()),
        }
        emit("export_pdf", 100, "Done", **artifacts)
    except asyncio.CancelledError:
        emit("canceled", 0, "Job canceled")
        raise
    except Exception as e:
        logger.exception("Job failed: %s", e)
        emit("failed", 0, f"Job failed: {e}")
