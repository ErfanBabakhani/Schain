"""Validated project settings; pipeline options retain their existing meanings."""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tier: int = Field(default=3, ge=0, le=10)
    max_documents_per_entity: int = Field(default=5, ge=1, le=100)
    max_documents_per_run: int = Field(default=10, ge=1, le=1000)
    max_total_documents: int = Field(default=100, ge=1, le=100000)
    llm_max_concurrency: int = Field(default=1, ge=1, le=8)
    retrieval_concurrency: int = Field(default=2, ge=1, le=8)
    max_minutes: float = Field(default=30, gt=0, le=1440, allow_inf_nan=False)


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Name
    seeds: list[Name] = Field(min_length=1, max_length=100)
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def unique_seeds(self):
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique")
        return self


RunStatus = Literal["queued", "running", "pausing", "paused", "completed", "failed"]


class RunResponse(BaseModel):
    id: str
    kind: Literal["expand", "ingest"]
    status: RunStatus
    created_at: str
    finished_at: str | None
    summary: dict[str, Any] | None
    error: str | None


class ProjectResponse(ProjectSettings):
    id: str
    status: Literal["ready", "queued", "running", "pausing", "paused", "completed", "failed"]
    runs: list[RunResponse]
    statistics: dict[str, Any]
    errors: list[dict[str, Any]]
    graph_revision: str | None


class GraphResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    pagination: dict[str, int]


class DocumentResponse(BaseModel):
    doc_id: str
    source_url: str | None
    status: Literal["PENDING", "PROCESSING", "EXTRACTED", "RESOLVED", "COMMITTED", "FAILED"]
    error: str | None
    created_at: str
    updated_at: str
    origin_entity: str | None
    tier: int | None
    title: str | None
    stage: str | None
    run_id: str | None
