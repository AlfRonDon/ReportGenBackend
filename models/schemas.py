from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


JobStatusLiteral = Literal["queued", "running", "succeeded", "failed", "canceled"]


class CreateJobRequest(BaseModel):
    db_url: str
    start_date: date
    end_date: date
    iterations: int = Field(default=1, ge=1, le=10)
    model: str = Field(default="gpt-5")


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatusLiteral


class ArtifactMeta(BaseModel):
    name: str
    kind: str
    size: int
    created_at: str
    url: str


class JobStatus(BaseModel):
    job_id: str
    status: JobStatusLiteral
    stage: Optional[str] = None
    progress: int = 0
    message: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)


class JobConfig(BaseModel):
    job_id: str
    db_url: str
    start_date: str
    end_date: str
    iterations: int = 1
    model: str = "gpt-5"
    input_pdf: Optional[str] = None

