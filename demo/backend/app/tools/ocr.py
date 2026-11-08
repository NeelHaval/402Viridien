"""Tool 2: OCR with RapidOCR (primary) and Tesseract (fallback), windowed and time-boxed."""
from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from app.tools.errors import OcrUnavailable

ENGINE_ORDER = ("rapidocr", "tesseract")
# Thread layout of the shared engine (legend + depth OCR). Overridable by environment for measurements.
MAX_WORKERS = int(os.environ.get("OCR_WORKERS", "4"))     # windows OCR'd at once
REC_THREADS = int(os.environ.get("OCR_REC_THREADS", "1"))  # recognition batches run at once inside one window
INTRA_THREADS = int(os.environ.get("OCR_INTRA", "2"))      # onnxruntime intra-op threads per session (0 = its default); 2/4/1 was the fastest layout measured on the 8-core M3
REC_BATCH = 6
GRACE_S = 0.15
WINDOW_PX = 1500
OVERLAP_PX = 120
TARGET_TEXT_PX = 35.0
MIN_SCALE, MAX_SCALE = 0.25, 3.0
BW_MAX_SCALE = 2.0
SCALE_DEADBAND = (0.85, 1.15)
OK_TEXT_PX = (22.0, 60.0)
MAX_PROC_PIXELS = 12_000_000
MIN_SIDE_PX = 6
STRETCH_PERCENTILES = (1.0, 99.0)
MIN_STRETCH_RANGE = 30
LOW_CONF_MEAN = 0.6
DEDUPE_IOU = 0.3
INK_ROW_FRACTION = 0.005
TEXT_HEIGHT_RANGE = (4, 200)
FALLBACK_TEXT_FRACTION = 1 / 50
TESSERACT_SINGLE_LINE_MAX_PX = 80
RAPID_DET_LIMIT = 1600
INSTALL_HINT = "pip install rapidocr-onnxruntime (preferred) or install the Tesseract binary and pip install pytesseract"
CONFUSABLE_CHARS = "OolI|SB"
DIGIT_CONFUSIONS = str.maketrans(CONFUSABLE_CHARS, "0011158")
NUMERIC_PUNCT = ".,-:"


@dataclass
class OcrLine:
    """One recognised text line with confidence and an (x0, y0, x1, y1) box in the input frame."""

    text: str
    conf: float
    box: tuple[float, float, float, float]


@dataclass
class OcrResult:
    """OCR output for one region."""

    lines: list[OcrLine] = field(default_factory=list)
    engine: str = ""
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    rotation: int = 0


class _Engine(Protocol):
    """Interface every OCR backend implements."""

    name: str

    def run(self, gray: np.ndarray, deadline: float | None = None) -> tuple[list[tuple[str, float, tuple[float, float, float, float]]], bool]:
        """Recognise text; return ((text, conf 0..1, box) list, truncated_by_deadline)."""


class _RapidEngine:
    """RapidOCR (PaddleOCR models on ONNX Runtime, CPU)."""

    name = "rapidocr"

    def __init__(self, intra_threads: int | None = None, rec_threads: int = REC_THREADS) -> None:
        """Load the ONNX models once, without the angle classifier (rotation is handled by callers).

        `intra_threads` fixes onnxruntime's intra-op thread count for every session of this engine (None = its default).
        """
        import rapidocr_onnxruntime.utils as rapid_utils
        from rapidocr_onnxruntime import RapidOCR
        self._rec_threads = rec_threads
        original = rapid_utils.SessionOptions
        if intra_threads:
            class _Options(original):  # type: ignore[misc, valid-type]
                def __init__(self) -> None:
                    super().__init__()
                    self.intra_op_num_threads, self.inter_op_num_threads = intra_threads, 1
            rapid_utils.SessionOptions = _Options
        try:
            self._ocr = RapidOCR(use_angle_cls=False, width_height_ratio=-1, min_height=0, det_model_path=None,
                                 det_limit_type="max", det_limit_side_len=RAPID_DET_LIMIT, text_score=0.0)
        finally:
            rapid_utils.SessionOptions = original

    def run(self, gray: np.ndarray, deadline: float | None = None):
        """Detect once, then recognise lines in reading order across threads until the deadline."""
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        boxes, _ = self._ocr.text_detector(bgr)
        if boxes is None or len(boxes) == 0:
            return [], False
        boxes = self._ocr.sorted_boxes(boxes)
        crops = self._ocr.get_crop_img_list(bgr, boxes)
        batches = [(i, crops[i:i + REC_BATCH]) for i in range(0, len(crops), REC_BATCH)]
        with ThreadPoolExecutor(self._rec_threads) as pool:
            parts = list(pool.map(lambda b: self._recognise(b, deadline), batches))
        out, truncated = [], False
        for (start, chunk), rec in zip(batches, parts):
            truncated |= rec is None
            for k, (text, conf) in enumerate(rec or []):
                xs, ys = boxes[start + k][:, 0], boxes[start + k][:, 1]
                out.append((str(text), float(conf), (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))))
        return out, truncated

    def detect_lines(self, gray: np.ndarray) -> tuple[list[tuple[float, float, float, float]], list[np.ndarray]]:
        """Detect text lines once; returns their boxes and crops in reading order (top to bottom, left to right)."""
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        boxes, _ = self._ocr.text_detector(bgr)
        if boxes is None or len(boxes) == 0:
            return [], []
        boxes = self._ocr.sorted_boxes(boxes)
        crops = self._ocr.get_crop_img_list(bgr, boxes)
        return [(float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 0].max()), float(b[:, 1].max())) for b in boxes], crops

    def recognise(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognise crops in ONE batched call (the recogniser sorts them by width, which wastes the least padding)."""
        if not crops:
            return []
        rec, _ = self._ocr.text_recognizer(crops)
        return [(str(t), float(c)) for t, c in rec]

    def _recognise(self, batch: tuple[int, list[np.ndarray]], deadline: float | None):
        """Recognise one batch of crops, or return None when the deadline has passed."""
        if deadline is not None and time.perf_counter() > deadline:
            return None
        rec, _ = self._ocr.text_recognizer(batch[1])
        return rec


class _TesseractEngine:
    """Tesseract via pytesseract (psm 6 for blocks, psm 7 for single lines)."""

    name = "tesseract"

    def __init__(self) -> None:
        """Verify the pytesseract module and the tesseract binary exist."""
        import pytesseract
        if shutil.which("tesseract") is None:
            raise ImportError("tesseract binary not found")
        self._tess = pytesseract

    def run(self, gray: np.ndarray, deadline: float | None = None):
        """Run Tesseract and group words into lines (the deadline is not enforced mid-call)."""
        psm = 7 if gray.shape[0] <= TESSERACT_SINGLE_LINE_MAX_PX else 6
        data = self._tess.image_to_data(gray, config=f"--psm {psm}", output_type=self._tess.Output.DICT)
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, text in enumerate(data["text"]):
            if text.strip() and float(data["conf"][i]) >= 0:
                groups.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(i)
        return [self._line(data, ids) for ids in groups.values()], False

    @staticmethod
    def _line(data: dict, ids: list[int]) -> tuple[str, float, tuple[float, float, float, float]]:
        """Merge word indices of one Tesseract line."""
        text = " ".join(data["text"][i].strip() for i in ids)
        conf = sum(float(data["conf"][i]) for i in ids) / len(ids) / 100.0
        x0 = min(data["left"][i] for i in ids)
        y0 = min(data["top"][i] for i in ids)
        x1 = max(data["left"][i] + data["width"][i] for i in ids)
        y1 = max(data["top"][i] + data["height"][i] for i in ids)
        return text, conf, (float(x0), float(y0), float(x1), float(y1))


_ENGINE: _Engine | None = None
_LOCATE_ENGINE: _Engine | None = None
LOCATE_THREADS = int(os.environ.get("OCR_LOCATE", "4"))
_ENGINE_LOCK = threading.Lock()
_POOL: ThreadPoolExecutor | None = None


def get_engine() -> _Engine:
    """Return the lazily created engine singleton or raise OcrUnavailable."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = _create_engine()
        return _ENGINE


def get_locate_engine() -> _Engine:
    """A second engine with its own onnxruntime sessions (4 threads) used only for the header location, so it never
    waits for, or competes with, legend and depth OCR. Falls back to the shared engine when RapidOCR is missing."""
    global _LOCATE_ENGINE
    with _ENGINE_LOCK:
        if _LOCATE_ENGINE is None:
            try:
                _LOCATE_ENGINE = _RapidEngine(intra_threads=LOCATE_THREADS, rec_threads=1)
            except (ImportError, OSError, RuntimeError, ValueError):
                _LOCATE_ENGINE = None
        engine = _LOCATE_ENGINE
    return engine if engine is not None else get_engine()


def _create_engine() -> _Engine:
    """Try each engine in ENGINE_ORDER and return the first that loads."""
    makers = {"rapidocr": lambda: _RapidEngine(intra_threads=INTRA_THREADS or None), "tesseract": _TesseractEngine}
    for name in ENGINE_ORDER:
        try:
            return makers[name]()
        except (ImportError, OSError, RuntimeError, ValueError, KeyError):
            continue
    raise OcrUnavailable(f"no OCR engine is available. Install with: {INSTALL_HINT}", "ocr_unavailable")


def _pool() -> ThreadPoolExecutor:
    """Return the shared OCR thread pool, created on first use."""
    global _POOL
    with _ENGINE_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(MAX_WORKERS, thread_name_prefix="ocr")
        return _POOL


def warmup() -> None:
    """Load every engine and run one tiny recognition on each so the first request is fast."""
    img = np.full((80, 320), 255, np.uint8)
    cv2.putText(img, "0123 456", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    get_engine().run(img, None)
    get_locate_engine().run(img, None)


def repair_digits(text: str) -> str:
    """Fix common OCR letter/digit confusions inside tokens that are mostly numeric."""
    return " ".join(_repair_token(tok) for tok in text.split(" "))


def _repair_token(token: str) -> str:
    """Translate confusable letters only when the token is numeric-like."""
    core = [c for c in token if c not in NUMERIC_PUNCT]
    digits = sum(c.isdigit() for c in core)
    confusable = sum(c in CONFUSABLE_CHARS or c.isdigit() for c in core)
    if digits == 0 or confusable != len(core) or digits * 2 < len(core):
        return token
    return token.translate(DIGIT_CONFUSIONS)


def _is_bilevel(gray: np.ndarray) -> bool:
    """True when a sampled subset of pixels holds at most two distinct values."""
    return np.unique(gray[::7, ::7]).size <= 2


def _estimate_text_height(gray: np.ndarray) -> float:
    """Estimate text line height from runs of inked rows; fall back to a width-based guess."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows = (ink.sum(axis=1) / 255.0) > INK_ROW_FRACTION * gray.shape[1]
    edges = np.diff(np.r_[0, rows.astype(np.int8), 0])
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    runs = runs[(runs >= TEXT_HEIGHT_RANGE[0]) & (runs <= TEXT_HEIGHT_RANGE[1])]
    return float(np.median(runs)) if runs.size else max(gray.shape[1] * FALLBACK_TEXT_FRACTION, 1.0)


def _stretch_contrast(gray: np.ndarray) -> np.ndarray:
    """Linear percentile contrast stretch; low-range images are left alone."""
    lo, hi = np.percentile(gray[::4, ::4], STRETCH_PERCENTILES)
    if hi - lo < MIN_STRETCH_RANGE:
        return gray
    lut = np.clip((np.arange(256) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.LUT(gray, lut)


def _choose_scale(gray: np.ndarray, text_height: float | None, bilevel: bool) -> float:
    """Pick the resize factor that brings text to about TARGET_TEXT_PX tall."""
    height = text_height if text_height else _estimate_text_height(gray)
    if OK_TEXT_PX[0] <= height <= OK_TEXT_PX[1]:
        return 1.0
    scale = float(np.clip(TARGET_TEXT_PX / max(height, 1.0), MIN_SCALE, BW_MAX_SCALE if bilevel else MAX_SCALE))
    pixels = gray.shape[0] * gray.shape[1] * scale * scale
    if pixels > MAX_PROC_PIXELS:
        scale *= (MAX_PROC_PIXELS / pixels) ** 0.5
    return 1.0 if SCALE_DEADBAND[0] <= scale <= SCALE_DEADBAND[1] else scale


def _prepare(gray: np.ndarray, text_height: float | None) -> tuple[np.ndarray, float]:
    """Contrast-stretch and rescale a gray image for recognition; return (image, scale)."""
    bilevel = _is_bilevel(gray)
    scale = _choose_scale(gray, text_height, bilevel)
    if not bilevel:
        gray = _stretch_contrast(gray)
    if scale != 1.0:
        interp = cv2.INTER_NEAREST if bilevel and scale > 1 else (cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    return np.ascontiguousarray(gray), scale


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert any uint8/bool image to 2-D uint8 gray."""
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if img.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
    return img


def _window_starts(length: int) -> list[int]:
    """Start offsets of overlapping windows covering `length` pixels along the long axis."""
    if length <= WINDOW_PX:
        return [0]
    step = WINDOW_PX - OVERLAP_PX
    starts = list(range(0, length - WINDOW_PX, step))
    return starts + [length - WINDOW_PX]


def unrotate_box(box: tuple[float, float, float, float], k: int, h: int, w: int) -> tuple[float, float, float, float]:
    """Map a box from an image rotated by np.rot90(k) back to the original (h, w) frame."""
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    mapped = [_unrotate_point(x, y, k, h, w) for x, y in corners]
    xs, ys = [m[0] for m in mapped], [m[1] for m in mapped]
    return min(xs), min(ys), max(xs), max(ys)


def _unrotate_point(xr: float, yr: float, k: int, h: int, w: int) -> tuple[float, float]:
    """Map one point from np.rot90(img, k) coordinates to the original image (k=1: x=W-yr, y=xr)."""
    k %= 4
    if k == 1:
        return w - yr, xr
    if k == 2:
        return w - xr, h - yr
    if k == 3:
        return yr, h - xr
    return xr, yr


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection-over-union of two boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(lines: list[OcrLine]) -> list[OcrLine]:
    """Drop duplicate lines produced by overlapping windows, keeping the more confident one."""
    kept: list[OcrLine] = []
    for line in sorted(lines, key=lambda ln: -ln.conf):
        if not any(_iou(line.box, k.box) > DEDUPE_IOU and _texts_match(line.text, k.text) for k in kept):
            kept.append(line)
    return sorted(kept, key=lambda ln: (round(ln.box[1]), ln.box[0]))


def _texts_match(a: str, b: str) -> bool:
    """True when two strings are equal or one contains the other (clipped at a window edge)."""
    a, b = a.strip(), b.strip()
    return a == b or a in b or b in a


def _run_window(engine: _Engine, window: np.ndarray, offset: int, horizontal: bool, deadline: float):
    """OCR one window and shift its boxes by the window offset along the long axis."""
    out = []
    found, truncated = engine.run(window, deadline)
    for text, conf, (x0, y0, x1, y1) in found:
        shift = (offset, 0, offset, 0) if horizontal else (0, offset, 0, offset)
        out.append((text, conf, (x0 + shift[0], y0 + shift[1], x1 + shift[2], y1 + shift[3])))
    return out, truncated


def _collect(futures: list[Future], deadline: float, warnings: list[str]) -> list:
    """Gather window results; windows stop themselves at the deadline, stragglers get a short grace."""
    done, pending = wait(futures, timeout=max(0.0, deadline + GRACE_S - time.perf_counter()))
    for fut in pending:
        fut.cancel()
    items, truncated = [], bool(pending)
    for fut in futures:
        if fut in done and not fut.exception():
            found, cut = fut.result()
            items += found
            truncated |= cut
    if truncated:
        warnings.append("ocr_time_budget_hit")
    return items


def _finish(raw: list, scale: float, k: int, shape: tuple[int, int], min_conf: float) -> list[OcrLine]:
    """Convert raw (text, conf, box) tuples to OcrLines in the original frame."""
    h, w = shape
    lines = []
    for text, conf, box in raw:
        if conf < min_conf or not text.strip():
            continue
        scaled = tuple(v / scale for v in box)
        lines.append(OcrLine(text.strip(), conf, unrotate_box(scaled, k, h, w)))
    return _dedupe(lines)


def ocr_region(img: np.ndarray, *, rotate: int = 0, min_conf: float = 0.5, time_budget_s: float = 1.0,
               target_height: float | None = None, engine: _Engine | None = None, inline: bool = False) -> OcrResult:
    """OCR an image region; `rotate` is np.rot90 quarter turns, `target_height` the known text height in px."""
    t0 = time.perf_counter()
    result = OcrResult(rotation=rotate % 4)
    if img is None or img.size == 0:
        result.warnings.append("empty_image")
        return _timed(result, t0)
    gray = _to_gray(img)
    h, w = gray.shape
    k = rotate % 4
    if k:
        gray = np.rot90(gray, k)
    if min(gray.shape) < MIN_SIDE_PX:
        result.warnings.append("image_too_small")
        return _timed(result, t0)
    if int(gray.min()) == int(gray.max()):
        return _timed(result, t0)
    engine = engine or get_engine()
    result.engine = engine.name
    proc, scale = _prepare(gray, target_height)
    raw = _run_windows(engine, proc, t0 + time_budget_s, result.warnings, inline)
    result.lines = _finish(raw, scale, k, (h, w), min_conf)
    if result.lines and float(np.mean([ln.conf for ln in result.lines])) < LOW_CONF_MEAN:
        result.warnings.append("low_ocr_confidence")
    return _timed(result, t0)


def _run_windows(engine: _Engine, proc: np.ndarray, deadline: float, warnings: list[str], inline: bool = False) -> list:
    """Split along the long axis, OCR windows in the pool (or one after another in this thread when `inline`)."""
    horizontal = proc.shape[1] >= proc.shape[0]
    length = proc.shape[1] if horizontal else proc.shape[0]
    if inline:
        items = []
        for start in _window_starts(length):
            win = proc[:, start:start + WINDOW_PX] if horizontal else proc[start:start + WINDOW_PX]
            found, cut = _run_window(engine, np.ascontiguousarray(win), start, horizontal, deadline)
            items += found
            if cut:
                warnings.append("ocr_time_budget_hit")
        return items
    pool = _pool()
    futures = []
    for start in _window_starts(length):
        win = proc[:, start:start + WINDOW_PX] if horizontal else proc[start:start + WINDOW_PX]
        futures.append(pool.submit(_run_window, engine, np.ascontiguousarray(win), start, horizontal, deadline))
    return _collect(futures, deadline, warnings)


def _timed(result: OcrResult, t0: float) -> OcrResult:
    """Stamp elapsed time on a result."""
    result.elapsed_s = time.perf_counter() - t0
    return result


def digit_score(result: OcrResult) -> float:
    """Sum of confidence-weighted digit characters across all lines."""
    return sum(ln.conf * sum(c.isdigit() for c in repair_digits(ln.text)) for ln in result.lines)


def ocr_rotated_strip(img: np.ndarray, *, rotations: tuple[int, ...] = (1, 3), min_conf: float = 0.5,
                      time_budget_s: float = 1.0, target_height: float | None = None) -> OcrResult:
    """OCR a strip with rotated text by trying 90 and 270 degrees and keeping the one with more digits."""
    t0 = time.perf_counter()
    best: OcrResult | None = None
    for i, k in enumerate(rotations):
        left = time_budget_s - (time.perf_counter() - t0)
        if i and left <= 0:
            best.warnings.append("ocr_time_budget_hit")
            break
        res = ocr_region(img, rotate=k, min_conf=min_conf, time_budget_s=max(left, 0.05) / (len(rotations) - i),
                         target_height=target_height)
        if best is None or digit_score(res) > digit_score(best):
            best = res
    best.elapsed_s = time.perf_counter() - t0
    return best
