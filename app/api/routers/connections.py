from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse, unquote

from fastapi import APIRouter, HTTPException

from app.models.schemas import ConnectionTestRequest, ConnectionTestResponse
from app.services.db_stage import get_parent_child_info


router = APIRouter(prefix="/connections", tags=["connections"])


def _normalize_sqlite(req: ConnectionTestRequest) -> Tuple[str, Path]:
    if req.db_url:
        raw_url = req.db_url.strip()
        parsed = urlparse(raw_url)
        if parsed.scheme != "sqlite":
            raise HTTPException(status_code=400, detail="Only sqlite URLs are supported for now")
        netloc = parsed.netloc or ""
        path_part = unquote(parsed.path or "")
        if netloc and path_part:
            raw_path = f"{netloc}{path_part}"
        elif netloc:
            raw_path = netloc
        else:
            raw_path = path_part
        if not raw_path:
            raise HTTPException(status_code=400, detail="sqlite URL is missing a file path")
        if os.name == "nt":
            if raw_path.startswith("/") and len(raw_path) > 2 and raw_path[1].isalpha() and raw_path[2] == ":":
                raw_path = raw_path.lstrip("/")
        if raw_path.startswith("//"):
            raw_path = raw_path[1:]
        db_url = raw_url
        path = Path(raw_path)
    else:
        if (req.db_type or "").lower() != "sqlite" or not req.path:
            raise HTTPException(status_code=400, detail="Provide either db_url or { db_type: 'sqlite', path }")
        path = Path(req.path)
        db_url = f"sqlite:///{path}"
    if not path.is_absolute():
        path = path.resolve()
    return db_url, path


@router.post("/test", response_model=ConnectionTestResponse)
async def test_connection(request: ConnectionTestRequest) -> ConnectionTestResponse:
    db_url, db_path = _normalize_sqlite(request)
    start = time.perf_counter()
    if not db_path.exists():
        raise HTTPException(status_code=502, detail=f"Database not found: {db_path}")
    try:
        info = get_parent_child_info(db_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to inspect database: {exc}") from exc
    latency = int((time.perf_counter() - start) * 1000)
    details = (
        f"Connected to sqlite database with parent '{info.get('parent table')}' "
        f"and child '{info.get('child table')}'."
    )
    return ConnectionTestResponse(status="connected", details=details, latencyMs=latency)