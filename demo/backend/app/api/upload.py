"""POST /upload."""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi import File as FormFile

from app.api import deps
from app.api.errors import ApiError
from app.api.schemas import UploadResponse
from app.config import Settings
from app.services.ingest import IngestService, ModelUnavailable
from app.timing import Trace, attach, span
from app.tools.errors import CorruptFile, TooLarge, ToolError, UnsupportedFile
from app.tools.read_upload import read_upload

router = APIRouter()
COPY_CHUNK = 1024 * 1024
BBOX_MAX_BYTES = 1024 * 1024
BOX_NAMES = ("depth_column", "interpreted_lithology_column", "lithology_legend")


def _bbox_boxes(bbox: dict) -> dict:
    """Validate the boxes used by preprocessing; preserve other annotation fields."""
    boxes = bbox.get("bounding_boxes", bbox)
    if not isinstance(boxes, dict):
        raise ApiError(400, "invalid_bbox", "bounding_boxes must be an object")
    for name in BOX_NAMES:
        if name not in boxes:
            continue
        box = boxes[name]
        if not isinstance(box, list) or len(box) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or
            (isinstance(v, float) and not math.isfinite(v)) for v in box
        ):
            raise ApiError(400, "invalid_bbox", f"{name} must contain four finite numeric coordinates")
        x0, y0, x1, y1 = box
        if x0 < 0 or y0 < 0 or int(x0) >= int(x1) or int(y0) >= int(y1):
            raise ApiError(400, "invalid_bbox", f"{name} must have nonnegative, ordered bounds with positive size")
    return boxes


def _check_bbox_image(boxes: dict, path: Path) -> None:
    """Check against the actual image dimensions without running OCR."""
    if not any(name in boxes for name in BOX_NAMES):
        return
    scan = read_upload(str(path))
    try:
        for name in BOX_NAMES:
            if name in boxes:
                _, _, x1, y1 = boxes[name]
                if x1 > scan.width or y1 > scan.height:
                    raise ApiError(400, "invalid_bbox", f"{name} must fit inside the image dimensions")
    finally:
        scan.close()


def _save(upload: UploadFile, dest: Path, limit: int) -> None:
    """Stream an upload to disk, refusing anything larger than `limit` bytes."""
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := upload.file.read(COPY_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise ApiError(413, "too_large", f"file is larger than {limit // (1024 * 1024)} MB")
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def _tool_status(exc: ToolError) -> int:
    """HTTP status for a rejected file."""
    return 415 if isinstance(exc, UnsupportedFile) else 413 if isinstance(exc, TooLarge) else 400


@router.post("/upload", response_model=UploadResponse)
def upload(request: Request, file: UploadFile = FormFile(...), bbox_json: UploadFile | None = FormFile(None),
           settings: Settings = Depends(deps.settings), ingest: IngestService = Depends(deps.ingest)) -> UploadResponse:
    """Accept a scan (and optional bbox JSON), queue it, return its id at once."""
    t_request = getattr(request.state, "t_request", time.perf_counter())
    trace = Trace(origin=t_request)  # time zero is the moment the request arrived
    trace.add("http.receive_multipart", t_request, time.perf_counter() - t_request)  # body read and spooled before the handler runs
    bbox = None
    if bbox_json is not None and bbox_json.filename:
        raw = bbox_json.file.read(BBOX_MAX_BYTES + 1)
        try:
            bbox = json.loads(raw) if len(raw) <= BBOX_MAX_BYTES else None
            # Reject non-finite values in all metadata, not only known boxes.
            json.dumps(bbox, allow_nan=False)
        except (ValueError, UnicodeDecodeError, RecursionError):
            bbox = None
        if not isinstance(bbox, dict):
            raise ApiError(400, "invalid_bbox", "bbox_json must be a JSON object smaller than 1 MB")
        boxes = _bbox_boxes(bbox)
    tmp = settings.scans_dir / f".incoming-{uuid.uuid4().hex}.upload"
    with attach(trace), span("upload.save"):
        _save(file, tmp, settings.max_upload_bytes)
    try:
        if bbox is not None:
            _check_bbox_image(boxes, tmp)
        submitted = ingest.submit(tmp, file.filename or "upload", bbox, trace=trace)
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
