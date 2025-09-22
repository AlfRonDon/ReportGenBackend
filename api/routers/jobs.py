from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.job_store import JobState, store
from app.core.logging import logger
from app.jobs.runner import run_full_pipeline
from app.models.schemas import CreateJobRequest, CreateJobResponse, JobConfig, JobStatus


router = APIRouter(prefix="/jobs", tags=["jobs"])


def _emit(job_id: str):
    def emit(stage: str, progress: int, message: str = "", **artifacts: str):
        state = store.get(job_id)
        if not state:
            return
        status = state.status
        if stage in {"failed", "canceled"}:
            status = stage
        else:
            status = "running"
        existing = state.artifacts or {}
        existing.update(artifacts)
        store.update(job_id, status=status, stage=stage, progress=progress, message=message, artifacts=existing)
        # persist to state.json is handled by store
    return emit


@router.post("", response_model=CreateJobResponse)
async def create_job(
    background: BackgroundTasks,
    pdf: Optional[UploadFile] = File(default=None),
    body: Optional[str] = Form(default=None),
    req_json: Optional[CreateJobRequest] = None,
):
    # Accept either multipart with pdf+body, or application/json
    if req_json is None:
        if body is None:
            raise HTTPException(status_code=400, detail="Missing payload: provide JSON or multipart 'body'")
        try:
            data = json.loads(body)
            req = CreateJobRequest(**data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
    else:
        req = req_json

    job_id = str(uuid.uuid4())
    job_dir = settings.BASE_ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save input PDF
    input_pdf = job_dir / "input.pdf"
    if pdf is not None:
        content = await pdf.read()
        input_pdf.write_bytes(content)
    else:
        # Allow server-side path via data
        if hasattr(req, "pdf_url") and req.pdf_url:  # type: ignore[attr-defined]
            src = Path(req.pdf_url)
            if not src.exists():
                raise HTTPException(status_code=400, detail="pdf_url not found")
            input_pdf.write_bytes(src.read_bytes())
        else:
            # For tests without real PDF, create tiny blank PDF when mocks enabled
            if settings.MOCK_RENDER:
                import fitz
                doc = fitz.open()
                doc.new_page(); doc.save(str(input_pdf)); doc.close()
            else:
                raise HTTPException(status_code=400, detail="No PDF provided")

    job = JobConfig(
        job_id=job_id,
        db_url=req.db_url,
        start_date=str(req.start_date),
        end_date=str(req.end_date),
        iterations=req.iterations,
        model=req.model,
        input_pdf=str(input_pdf),
    )

    # Initialize job state
    store.put(JobState(job_id=job_id, status="queued", stage="queued", progress=0, message="queued", artifacts={}))

    # Background task
    async def runner():
        emit = _emit(job_id)
        try:
            await run_full_pipeline(job, emit)
            st = store.get(job_id)
            if st and st.status not in {"failed", "canceled"}:
                store.update(job_id, status="succeeded", progress=100, stage="export_pdf", message="Done")
        except asyncio.CancelledError:
            store.update(job_id, status="canceled", message="Canceled")
        except Exception as e:
            logger.exception("Pipeline crashed: %s", e)
            store.update(job_id, status="failed", message=str(e))

    # Launch task without blocking
    background.add_task(runner)

    return CreateJobResponse(job_id=job_id, status="queued")


@router.get("/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    st = store.get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=st.job_id,
        status=st.status, stage=st.stage, progress=st.progress,
        message=st.message, artifacts=st.artifacts or {},
    )


@router.get("/{job_id}/artifacts")
async def list_artifacts(job_id: str):
    job_dir = settings.BASE_ARTIFACTS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    items = []
    for path in job_dir.rglob("*"):
        if path.is_file():
            kind = "other"
            name = path.name
            if name.endswith(".json"):
                kind = "json"
            elif name.endswith(".html"):
                kind = "html"
            elif name.endswith(".png"):
                kind = "png"
            elif name.endswith(".pdf"):
                kind = "pdf"
            items.append({
                "name": name,
                "kind": kind,
                "size": path.stat().st_size,
                "created_at": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat(),
                "url": str(path.resolve()),
            })
    return items


@router.get("/{job_id}/html")
async def get_final_html(job_id: str):
    path = settings.BASE_ARTIFACTS_DIR / job_id / "filled" / "report_filled.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="HTML not found")
    return FileResponse(str(path))


@router.get("/{job_id}/pdf")
async def get_final_pdf(job_id: str):
    path = settings.BASE_ARTIFACTS_DIR / job_id / "pdf" / "report_filled_new.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(path))


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    st = store.get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    store.request_cancel(job_id)
    return {"job_id": job_id, "status": st.status, "message": "cancel requested"}

