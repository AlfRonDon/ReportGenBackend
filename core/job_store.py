from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings


JobStatusStr = str


@dataclass
class JobState:
    job_id: str
    status: JobStatusStr = "queued"
    stage: str | None = None
    progress: int = 0
    message: str | None = None
    artifacts: Dict[str, str] | None = None
    created_at: str = datetime.utcnow().isoformat()
    updated_at: str = datetime.utcnow().isoformat()
    cancel_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        return settings.BASE_ARTIFACTS_DIR / job_id

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "state.json"

    def put(self, job: JobState) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._persist(job)

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            st = self._jobs.get(job_id)
        if st is None:
            # Try load from disk
            path = self.state_path(job_id)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    st = JobState(**data)
                    with self._lock:
                        self._jobs[job_id] = st
                except Exception:
                    return None
        return st

    def update(self, job_id: str, **kwargs: Any) -> Optional[JobState]:
        with self._lock:
            st = self._jobs.get(job_id)
            if not st:
                return None
            for k, v in kwargs.items():
                setattr(st, k, v)
            st.updated_at = datetime.utcnow().isoformat()
            self._persist(st)
            return st

    def list(self) -> Dict[str, JobState]:
        with self._lock:
            return dict(self._jobs)

    def request_cancel(self, job_id: str) -> None:
        self.update(job_id, cancel_requested=True)

    def _persist(self, job: JobState) -> None:
        job_dir = self.job_dir(job.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_path(job.job_id)
        path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")


store = JobStore()

