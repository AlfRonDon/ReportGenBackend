from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.routers.jobs import queue_job_from_path
from app.core.config import settings
from app.core.job_store import store
from app.models.schemas import (
    CreateJobRequest,
    RunCreateRequest,
    RunJobSnapshot,
    RunRecord,
    RunStatusResponse,
)


router = APIRouter(prefix="/runs", tags=["runs"])


def _runs_root() -> Path:
    root = settings.BASE_ARTIFACTS_DIR / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_path(run_id: str) -> Path:
    return _runs_root() / f"{run_id}.json"


def _save_run(record: RunRecord) -> None:
    path = _run_path(record.id)
    path.write_text(json.dumps(record.dict(), indent=2), encoding="utf-8")


def _generate_run_id(prefix: str = "run", length: int = 8) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    return prefix + "_" + "".join(secrets.choice(alphabet) for _ in range(length))


def _templates_root() -> Path:
    root = settings.BASE_ARTIFACTS_DIR / "templates"
    if not root.exists():
        raise HTTPException(status_code=400, detail="No templates have been created yet")
    return root


def _load_template_meta(template_id: str) -> Dict:
    meta_path = _templates_root() / template_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _template_pdf_path(template_id: str) -> Path:
    tpl_dir = _templates_root() / template_id
    if not tpl_dir.exists():
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    original_dir = tpl_dir / "original"
    if not original_dir.exists():
        raise HTTPException(status_code=400, detail=f"Template {template_id} is missing original assets")
    for candidate in original_dir.glob("*.pdf"):
        return candidate
    raise HTTPException(status_code=400, detail=f"Template {template_id} does not have a PDF source")


def _load_run(run_id: str) -> RunRecord:
    path = _run_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord(**data)


def _aggregate_status(record: RunRecord) -> Tuple[RunRecord, RunStatusResponse]:
    snapshots: List[RunJobSnapshot] = []
    statuses: List[str] = []
    progresses: List[int] = []
    for job_id in record.job_ids:
        st = store.get(job_id)
        if st:
            statuses.append(st.status)
            progresses.append(st.progress)
            snapshots.append(
                RunJobSnapshot(
                    job_id=st.job_id,
                    status=st.status,
                    stage=st.stage,
                    progress=st.progress,
                    message=st.message,
                )
            )
        else:
            statuses.append("queued")
            progresses.append(0)
            snapshots.append(
                RunJobSnapshot(
                    job_id=job_id,
                    status="queued",
                    stage=None,
                    progress=0,
                    message="pending",
                )
            )

    if not statuses:
        agg_status = record.status
        progress_value = record.progress
    else:
        has_running = any(s in {"running"} for s in statuses)
        has_queued = any(s == "queued" for s in statuses)
        has_failed = any(s == "failed" for s in statuses)
        has_succeeded = any(s == "succeeded" for s in statuses)
        has_canceled = any(s == "canceled" for s in statuses)

        if all(s == "queued" for s in statuses):
            agg_status = "queued"
        elif has_running or has_queued:
            agg_status = "running"
        elif has_failed and has_succeeded:
            agg_status = "partial"
        elif has_failed:
            agg_status = "failed"
        elif has_canceled and has_succeeded:
            agg_status = "partial"
        elif has_canceled and not has_succeeded:
            agg_status = "canceled"
        elif all(s == "succeeded" for s in statuses):
            agg_status = "complete"
        else:
            agg_status = record.status

        progress_value = sum(progresses) / len(progresses)

    updated_record = record.copy()
    updated_record.status = agg_status
    updated_record.progress = progress_value
    updated_record.updated_at = datetime.utcnow().isoformat()
    _save_run(updated_record)

    response = RunStatusResponse(
        id=updated_record.id,
        name=updated_record.name,
        template_ids=updated_record.template_ids,
        job_ids=updated_record.job_ids,
        status=agg_status,
        progress=progress_value,
        created_at=updated_record.created_at,
        updated_at=updated_record.updated_at,
        jobs=snapshots,
    )
    return updated_record, response


@router.post("", response_model=RunStatusResponse)
async def create_run(req: RunCreateRequest, background: BackgroundTasks) -> RunStatusResponse:
    if not req.template_ids:
        raise HTTPException(status_code=400, detail="template_ids cannot be empty")

    job_ids: List[str] = []
    for template_id in req.template_ids:
        meta = _load_template_meta(template_id)
        if meta.get("status") != "approved":
            raise HTTPException(status_code=400, detail=f"Template {template_id} is not approved")
        pdf_path = _template_pdf_path(template_id)
        job_request = CreateJobRequest(
            db_url=req.db_url,
            start_date=req.start_date,
            end_date=req.end_date,
            iterations=req.iterations,
            model=req.model,
        )
        response = await queue_job_from_path(background, job_request, pdf_path)
        job_ids.append(response.job_id)

    run_id = _generate_run_id()
    timestamp = datetime.utcnow().isoformat()
    record = RunRecord(
        id=run_id,
        name=req.name or f"Run {timestamp[:19]}",
        template_ids=req.template_ids,
        job_ids=job_ids,
        status="queued",
        progress=0.0,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={
            "db_url": req.db_url,
            "start_date": str(req.start_date),
            "end_date": str(req.end_date),
            "iterations": req.iterations,
            "model": req.model,
        },
    )
    _save_run(record)
    _, status = _aggregate_status(record)
    return status


@router.get("", response_model=List[RunStatusResponse])
async def list_runs() -> List[RunStatusResponse]:
    responses: List[RunStatusResponse] = []
    for path in sorted(_runs_root().glob("*.json")):
        record = RunRecord(**json.loads(path.read_text(encoding="utf-8")))
        _, status = _aggregate_status(record)
        responses.append(status)
    responses.sort(key=lambda r: r.updated_at, reverse=True)
    return responses


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str) -> RunStatusResponse:
    record = _load_run(run_id)
    _, status = _aggregate_status(record)
    return status


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str):
    record = _load_run(run_id)
    for job_id in record.job_ids:
        store.request_cancel(job_id)
    record.status = "canceled"
    record.updated_at = datetime.utcnow().isoformat()
    _save_run(record)
    return {"id": run_id, "status": "canceled"}
