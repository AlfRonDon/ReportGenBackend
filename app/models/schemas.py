from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


JobStatusLiteral = Literal["queued", "running", "succeeded", "failed", "canceled"]


class CreateJobRequest(BaseModel):
    db_url: str
    start_date: date
    end_date: date
    iterations: int = Field(default=1, ge=1, le=10)
    model: str = Field(default="gpt-5")
    pdf_url: Optional[str] = None


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


class ConnectionTestRequest(BaseModel):
    db_url: Optional[str] = None
    db_type: Optional[str] = None
    path: Optional[str] = None


class ConnectionTestResponse(BaseModel):
    status: Literal["connected"]
    details: str
    latencyMs: int


class TemplateMeta(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    status: Literal["draft", "approved"]
    sourceType: Literal["pdf", "excel"]
    created_at: str
    updated_at: str
    preview_url: Optional[str] = None


class TemplateCreateResponse(BaseModel):
    id: str
    status: Literal["draft", "approved"]
    preview_url: str


class RunCreateRequest(BaseModel):
    template_ids: List[str]
    db_url: str
    start_date: date
    end_date: date
    iterations: int = Field(default=1, ge=1, le=10)
    model: str = Field(default="gpt-5")
    name: Optional[str] = None


RunStatusLiteral = Literal["queued", "running", "complete", "failed", "canceled", "partial"]


class RunRecord(BaseModel):
    id: str
    name: Optional[str] = None
    template_ids: List[str]
    job_ids: List[str]
    status: RunStatusLiteral
    progress: float = 0.0
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunJobSnapshot(BaseModel):
    job_id: str
    status: JobStatusLiteral
    stage: Optional[str] = None
    progress: int
    message: Optional[str] = None


class RunStatusResponse(BaseModel):
    id: str
    name: Optional[str] = None
    template_ids: List[str]
    job_ids: List[str]
    status: RunStatusLiteral
    progress: float
    created_at: str
    updated_at: str
    jobs: List[RunJobSnapshot] = Field(default_factory=list)