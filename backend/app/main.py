from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .pipeline import (
    OCRPipelineError,
    estimate_quality_score,
    ocr_config_from_env,
    probe_server,
    scan_invoice_file,
)
from .schemas import ExtractResponse, ServerStatus


BACKEND_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = BACKEND_ROOT / "storage"
UPLOAD_ROOT = STORAGE_ROOT / "uploads"
RESULT_ROOT = STORAGE_ROOT / "results"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


app = FastAPI(
    title="invoiceOCR API",
    description="Local FastAPI wrapper for the Vintern GGUF invoice OCR pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)


@app.on_event("startup")
def ensure_storage() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str | None) -> str:
    filename = filename or "invoice.pdf"
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename).strip(" .")
    return safe or "invoice.pdf"


async def save_upload(file: UploadFile, destination: Path) -> int:
    size = 0
    with destination.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            output.write(chunk)
    return size


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=ServerStatus)
async def status() -> dict:
    config = ocr_config_from_env()
    return await run_in_threadpool(probe_server, config.server_url, 5)


@app.post("/api/invoices/extract", response_model=ExtractResponse)
async def extract_invoice(file: UploadFile = File(...)) -> dict:
    original_name = sanitize_filename(file.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ PDF hoặc ảnh hóa đơn (.pdf, .png, .jpg, .webp, .tif).",
        )

    job_id = uuid4().hex
    job_upload_dir = UPLOAD_ROOT / job_id
    job_result_dir = RESULT_ROOT / job_id
    image_dir = job_result_dir / "images"
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_result_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_upload_dir / original_name
    await save_upload(file, input_path)

    config = ocr_config_from_env()
    started_at = time.perf_counter()
    try:
        pages = await run_in_threadpool(
            scan_invoice_file,
            input_path,
            image_dir,
            config,
        )
    except OCRPipelineError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - surfaced to UI
        raise HTTPException(status_code=500, detail=f"OCR thất bại: {error}") from error

    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    first_data = pages[0].get("data", {}) if pages else {}
    source_type = first_data.get("metadata", {}).get("source_type") if pages else None
    confidence = estimate_quality_score(pages)

    result_payload = {
        "job_id": job_id,
        "filename": original_name,
        "page_count": len(pages),
        "source_type": source_type,
        "confidence": confidence,
        "elapsed_seconds": elapsed_seconds,
        "pages": pages,
        "data": first_data,
        "download_url": f"/api/invoices/{job_id}/download",
    }

    result_path = job_result_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(result_payload, result_file, ensure_ascii=False, indent=2)

    return result_payload


@app.get("/api/invoices/{job_id}/download")
def download_result(job_id: str) -> FileResponse:
    result_path = RESULT_ROOT / job_id / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả OCR.")
    return FileResponse(
        result_path,
        media_type="application/json",
        filename=f"invoice-ocr-{job_id}.json",
    )


@app.get("/api/invoices/{job_id}", response_model=ExtractResponse)
def get_result(job_id: str) -> dict:
    result_path = RESULT_ROOT / job_id / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả OCR.")
    with result_path.open("r", encoding="utf-8") as result_file:
        return json.load(result_file)
