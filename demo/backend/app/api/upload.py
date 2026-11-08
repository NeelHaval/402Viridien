"""POST /upload."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile
from fastapi import File as FormFile

from app.api import deps
from app.api.errors import ApiError
from app.api.schemas import UploadResponse
from app.config import Settings
from app.services.ingest import IngestService, ModelUnavailable
from app.tools.errors import CorruptFile, TooLarge, ToolError, UnsupportedFile

router = APIRouter()
COPY_CHUNK = 1024 * 1024
BBOX_MAX_BYTES = 1024 * 1024


def _save(upload: UploadFile, dest: Path, limit: int) -> None:
    """Stream an upload to disk, refusing anything larger than `limit` bytes."""
    size = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(COPY_CHUNK):
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise ApiError(413, "too_large", f"file is larger than {limit // (1024 * 1024)} MB")
            out.write(chunk)


def _tool_status(exc: ToolError) -> int:
    """HTTP status for a rejected file."""
    return 415 if isinstance(exc, UnsupportedFile) else 413 if isinstance(exc, TooLarge) else 400


@router.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = FormFile(...), bbox_json: UploadFile | None = FormFile(None),
           settings: Settings = Depends(deps.settings), ingest: IngestService = Depends(deps.ingest)) -> UploadResponse:
    """Accept a scan (and optional bbox JSON), queue it, return its id at once."""
    bbox = None
    if bbox_json is not None and bbox_json.filename:
        raw = bbox_json.file.read(BBOX_MAX_BYTES + 1)
        try:
            bbox = json.loads(raw) if len(raw) <= BBOX_MAX_BYTES else None
        except ValueError:
            bbox = None
        if not isinstance(bbox, dict):
            raise ApiError(400, "invalid_bbox", "bbox_json must be a JSON object smaller than 1 MB")
    tmp = settings.scans_dir / f".incoming-{uuid.uuid4().hex}{Path(file.filename or 'scan').suffix.lower()}"
    _save(file, tmp, settings.max_upload_bytes)
    try:
        submitted = ingest.submit(tmp, file.filename or "upload", bbox)
    except ModelUnavailable as exc:
        tmp.unlink(missing_ok=True)
        raise ApiError(503, "model_unavailable", f"scans cannot be processed: {exc}") from exc
    except ToolError as exc:
        tmp.unlink(missing_ok=True)
        raise ApiError(_tool_status(exc), exc.code, exc.message) from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return UploadResponse(id=submitted.id, status=submitted.status)
