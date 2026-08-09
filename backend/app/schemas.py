from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ServerStatus(BaseModel):
    server_url: str
    origin: str
    reachable: bool
    checks: list[dict[str, Any]]
    server_command: str


class ExtractResponse(BaseModel):
    job_id: str
    filename: str
    page_count: int = Field(ge=0)
    source_type: str | None = None
    confidence: int = Field(ge=0, le=100)
    elapsed_seconds: float
    pages: list[dict[str, Any]]
    data: dict[str, Any]
    download_url: str
