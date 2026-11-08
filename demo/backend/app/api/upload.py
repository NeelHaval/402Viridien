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
from app.tools.errors import InputConflict, QueueBusy, TooLarge, ToolError, UnsupportedFile

router = APIRouter()
COPY_CHUNK = 1024 * 1024
BBOX_MAX_BYTES = 1024 * 1024


def _save(upload: UploadFile, dest: Path, limit: int) -> None:
    """Stream an upload to disk, refusing anything larger than `limit` bytes."""
    size = 0
    created = False
    try:
        with dest.open("xb") as out:
            created = True
            while chunk := upload.file.read(COPY_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise ApiError(413, "too_large", f"file is larger than {limit // (1024 * 1024)} MB")
                out.write(chunk)
    except BaseException:
        # Never remove a pre-existing file if exclusive creation failed.
        if created:
            dest.unlink(missing_ok=True)
        raise


def _tool_status(exc: ToolError) -> int:
    """HTTP status for a rejected file."""
    if isinstance(exc, InputConflict):
        return 409
    if isinstance(exc, QueueBusy):
        return 503
    return 415 if isinstance(exc, UnsupportedFile) else 413 if isinstance(exc, TooLarge) else 400


def _reject_nonfinite(value: str):
    raise ValueError("nonfinite JSON number")


def _read_bbox(raw: bytes) -> dict:
    """Reject malformed metadata before it can enter the background job queue."""
    try:
        if len(raw) > BBOX_MAX_BYTES:
            raise ValueError("oversized bounding-box metadata")
        bbox = json.loads(raw, parse_constant=_reject_nonfinite)
        if not isinstance(bbox, dict):
            raise ValueError("bounding-box metadata must be an object")
        boxes = bbox.get("bounding_boxes", bbox)
        if not isinstance(boxes, dict):
            raise ValueError("bounding_boxes must be an object")
        for name in ("depth_column", "interpreted_lithology_column", "lithology_legend"):
            if name not in boxes:
                continue
            box = boxes[name]
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(value) is not int or value < 0 for value in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError(f"{name} needs four nonnegative integer coordinates with positive area")
        return bbox
    except (ValueError, RecursionError) as exc:
        raise ApiError(400, "invalid_bbox", "bbox_json must contain valid bounding boxes in a JSON object no larger than 1 MB") from exc


@router.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = FormFile(...), bbox_json: UploadFile | None = FormFile(None),
           settings: Settings = Depends(deps.settings), ingest: IngestService = Depends(deps.ingest)) -> UploadResponse:
    """Accept a scan (and optional bbox JSON), queue it, return its id at once."""
    bbox = None
    if bbox_json is not None and bbox_json.filename:
        raw = bbox_json.file.read(BBOX_MAX_BYTES + 1)
        bbox = _read_bbox(raw)
    # Image type is sniffed from bytes. Never put a client-supplied suffix in
    # a filesystem path (notably Windows alternate-data-stream syntax).
    tmp = settings.scans_dir / f".incoming-{uuid.uuid4().hex}"
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
