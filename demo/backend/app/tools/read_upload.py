"""Tool 1: safely open an uploaded scan and expose random-access crops."""
from __future__ import annotations

import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from tools.errors import CorruptFile, TooLarge, ToolError, UnsupportedFile

MAX_FILE_BYTES = 500 * 1024 * 1024
MAX_PIXELS = 400_000_000
TIFF_MAX_PIXELS = 2_000_000_000
ASPECT_WARN = 150.0
BAND_ROWS = 2048
DECODE_WORKERS = 4
BIG_SEGMENT_BYTES = 32 * 1024 * 1024
MAX_PREFIX_CACHE_BYTES = 1_500_000_000
PREFIX_GROWTH = 2.0
BW_INK_THRESHOLD = 192
LARGE_FULL_DECODE_PIXELS = 50_000_000
MAGIC_TIFF = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
MAGIC_JPEG = b"\xff\xd8\xff"
MAGIC_PDF = b"%PDF"
LZW_CODE = 5
DEFLATE_CODES = (8, 32946)
NO_COMPRESSION = 1


@dataclass
class UploadedScan:
    """A scan opened for random-access reading."""

    path: str
    width: int
    height: int
    mode: str
    dpi: float | None
    reader: "_BaseReader"
    warnings: list[str] = field(default_factory=list)

    def close(self) -> None:
        """Release file handles and cached buffers."""
        self.reader.close()


def sniff_type(head: bytes) -> str:
    """Return 'tiff', 'png', 'jpeg', 'pdf' or '' from leading magic bytes."""
    if head.startswith(MAGIC_TIFF):
        return "tiff"
    if head.startswith(MAGIC_PNG):
        return "png"
    if head.startswith(MAGIC_JPEG):
        return "jpeg"
    if head.startswith(MAGIC_PDF):
        return "pdf"
    return ""


def downscale(img: np.ndarray, scale: float, out_h: int, out_w: int, bw: bool) -> np.ndarray:
    """Resize with area averaging; keep bilevel images strictly 0/255 and ink-preserving."""
    if out_h <= 0 or out_w <= 0:
        return np.zeros((0, 0), np.uint8)
    out = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if bw:
        out = np.where(out < BW_INK_THRESHOLD, 0, 255).astype(np.uint8)
    return out


class _BaseReader:
    """Shared clipping, banding and scaling logic for all readers."""

    width = 0
    height = 0
    bw = False

    def crop(self, y0: int, y1: int, x0: int = 0, x1: int | None = None, scale: float = 1.0) -> np.ndarray:
        """Return rows y0..y1 and columns x0..x1 as uint8 gray or RGB, scaled by `scale` <= 1."""
        x1 = self.width if x1 is None else x1
        y0, y1 = max(0, int(y0)), min(self.height, int(y1))
        x0, x1 = max(0, int(x0)), min(self.width, int(x1))
        scale = min(1.0, max(float(scale), 1e-3))
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0), np.uint8)
        out_w = max(1, round((x1 - x0) * scale))
        total_h = max(1, round((y1 - y0) * scale))
        self._prepare(y0, y1)
        if scale == 1.0:
            return self._assemble(y0, y1, x0, x1)
        parts: list[np.ndarray] = []
        for ya in range(y0, y1, BAND_ROWS):
            yb = min(y1, ya + BAND_ROWS)
            oa, ob = round((ya - y0) * scale), round((yb - y0) * scale)
            rows = min(ob, total_h) - oa
            if rows > 0:
                parts.append(downscale(self._assemble(ya, yb, x0, x1), scale, rows, out_w, self.bw))
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), np.uint8)

    def _assemble(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Read a full-resolution region in bands to bound peak memory."""
        if y1 - y0 <= BAND_ROWS:
            return self._read_region(y0, y1, x0, x1)
        bands = [self._read_region(ya, min(y1, ya + BAND_ROWS), x0, x1) for ya in range(y0, y1, BAND_ROWS)]
        return np.concatenate(bands, axis=0)

    def _prepare(self, y0: int, y1: int) -> None:
        """Hook to pre-decode data for rows y0..y1 before banded reads (default: nothing)."""

    def _read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Read one full-resolution region (implemented by subclasses)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (default: nothing to release)."""


class TiffReader(_BaseReader):
    """Region reader for TIFF files that decodes only the strips or tiles it needs."""

    def __init__(self, path: str, tif, page) -> None:
        """Capture page geometry and keep the TIFF handle open for segment reads."""
        self._tif, self._page, self._path = tif, page, path
        self.width, self.height = int(page.imagewidth), int(page.imagelength)
        self.samples = int(page.samplesperpixel)
        self.bps = int(page.bitspersample)
        self.bw = self.bps == 1
        self.compression = int(page.compression)
        self.predictor = int(page.predictor)
        self.tiled = bool(page.is_tiled)
        self.offsets = [int(v) for v in page.dataoffsets]
        self.counts = [int(v) for v in page.databytecounts]
        self.photometric = page.photometric.name
        self.miniswhite = self.photometric == "MINISWHITE"
        self.palette = self._build_palette(page)
        self._lock = threading.Lock()
        self._prefix: np.ndarray | None = None
        self._pool = ThreadPoolExecutor(DECODE_WORKERS)
        self._check_supported(page)
        self._single_big = self._is_single_big_segment()

    @staticmethod
    def _build_palette(page) -> np.ndarray | None:
        """Return an (N, 3) uint8 palette for palette images, else None."""
        if page.photometric.name != "PALETTE" or page.colormap is None:
            return None
        return (np.asarray(page.colormap, dtype=np.uint16) >> 8).astype(np.uint8).T

    def _check_supported(self, page) -> None:
        """Reject layouts and sample formats this reader cannot decode."""
        if self.photometric not in ("MINISBLACK", "MINISWHITE", "RGB", "PALETTE"):
            raise UnsupportedFile(f"TIFF photometric '{self.photometric}' is not supported", "unsupported_photometric")
        if self.bps not in (1, 8, 16):
            raise UnsupportedFile(f"{self.bps}-bit TIFF samples are not supported", "unsupported_bit_depth")
        if page.dtype.kind not in "uib":
            raise UnsupportedFile("floating-point TIFFs are not supported", "unsupported_sample_format")
        if self.samples > 1 and int(page.planarconfig) != 1:
            raise UnsupportedFile("planar-separate TIFFs are not supported", "unsupported_planar")

    def _is_single_big_segment(self) -> bool:
        """True for a huge single-strip file that must be read as a decoded prefix."""
        if self.tiled or self.bps == 1 or len(self.offsets) != 1:
            return False
        rows_bytes = self.width * self.samples * (self.bps // 8) * self.height
        return rows_bytes > BIG_SEGMENT_BYTES and self.predictor in (1, 2) and self._prefix_codec_ok()

    def _prefix_codec_ok(self) -> bool:
        """True when the compression allows decoding only the first N bytes."""
        if self.bps != 8:
            return False
        return self.compression in (LZW_CODE, NO_COMPRESSION, *DEFLATE_CODES)

    def _read_raw(self, index: int) -> bytes:
        """Read the compressed bytes of one segment under the file lock."""
        with self._lock:
            fh = self._tif.filehandle
            fh.seek(self.offsets[index])
            return fh.read(self.counts[index])

    def _decode_segment(self, index: int):
        """Decode one strip or tile into (array, start_y, start_x)."""
        seg, idx, _ = self._page.decode(self._read_raw(index), index)
        arr = seg[0]
        return arr, int(idx[2]), int(idx[3])

    def _segment_indices(self, y0: int, y1: int, x0: int, x1: int) -> list[int]:
        """List the strip or tile indices that intersect a region."""
        page = self._page
        if not self.tiled:
            rps = int(page.rowsperstrip)
            return list(range(y0 // rps, (y1 - 1) // rps + 1))
        tw, tl = int(page.tilewidth), int(page.tilelength)
        per_row = -(-self.width // tw)
        return [ty * per_row + tx for ty in range(y0 // tl, (y1 - 1) // tl + 1)
                for tx in range(x0 // tw, (x1 - 1) // tw + 1)]

    def _read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Decode the needed segments and paste them into the output region."""
        if self._single_big:
            return self._normalize(self._prefix_rows(y0, y1)[:, x0:x1])
        indices = self._segment_indices(y0, y1, x0, x1)
        decoded = list(self._pool.map(self._decode_segment, indices)) if len(indices) > 1 else [self._decode_segment(indices[0])]
        shape = (y1 - y0, x1 - x0) + ((3,) if self._is_color() else ())
        out = np.full(shape, 255, np.uint8)
        for arr, sy, sx in decoded:
            self._paste(out, self._normalize(arr), sy, sx, y0, x0)
        return out

    def _is_color(self) -> bool:
        """True when output pixels have three channels."""
        return self.palette is not None or self.photometric == "RGB"

    @staticmethod
    def _paste(out: np.ndarray, seg: np.ndarray, sy: int, sx: int, y0: int, x0: int) -> None:
        """Copy the overlap of a decoded segment into the output region."""
        ya, xa = max(sy, y0), max(sx, x0)
        yb, xb = min(sy + seg.shape[0], y0 + out.shape[0]), min(sx + seg.shape[1], x0 + out.shape[1])
        if yb > ya and xb > xa:
            out[ya - y0:yb - y0, xa - x0:xb - x0] = seg[ya - sy:yb - sy, xa - sx:xb - sx]

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Convert a decoded segment to uint8 gray (H, W) or RGB (H, W, 3), white paper = 255."""
        if self.bps == 1:
            arr = (arr != 0).astype(np.uint8) * 255
        elif arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        arr = arr.astype(np.uint8, copy=False)
        if self.palette is not None:
            return self.palette[arr[..., 0]]
        arr = arr[..., :3] if self.photometric == "RGB" else arr[..., 0]
        return 255 - arr if self.miniswhite else arr

    def _prepare(self, y0: int, y1: int) -> None:
        """Decode the whole requested prefix once so banded reads do not re-decode it."""
        if self._single_big and (self._prefix is None or self._prefix.shape[0] < y1):
            self._prefix = self._decode_prefix(y1, self.width * self.samples)

    def _prefix_rows(self, y0: int, y1: int) -> np.ndarray:
        """Return rows y0..y1 of a single-strip file, decoding only a cached prefix."""
        row_bytes = self.width * self.samples
        have = 0 if self._prefix is None else self._prefix.shape[0]
        if y1 > have:
            want = min(self.height, max(y1, int(have * PREFIX_GROWTH)))
            self._prefix = self._decode_prefix(want, row_bytes)
        rows = self._prefix[y0:y1]
        if self._prefix.nbytes > MAX_PREFIX_CACHE_BYTES:
            rows = rows.copy()
            self._prefix = None
        return rows

    def _decode_prefix(self, rows: int, row_bytes: int) -> np.ndarray:
        """Decode the first `rows` rows of the single strip into a (rows, W, samples) array."""
        import imagecodecs
        raw = self._read_raw(0)
        need = rows * row_bytes
        if self.compression == LZW_CODE:
            data = imagecodecs.lzw_decode(raw, out=need)
        elif self.compression == NO_COMPRESSION:
            data = raw[:need]
        else:
            data = zlib.decompressobj().decompress(raw, need)
        buf = np.frombuffer(data, np.uint8)
        got = buf.size // row_bytes
        if got < min(rows, self.height):
            raise CorruptFile("TIFF strip data ended early", "truncated")
        arr = buf[:got * row_bytes].reshape(got, self.width, self.samples)
        if self.predictor == 2:
            arr = np.cumsum(arr, axis=1, dtype=np.uint8)
        return arr

    def close(self) -> None:
        """Close the TIFF handle, decode pool and prefix cache."""
        self._pool.shutdown(wait=False)
        self._prefix = None
        self._tif.close()


class PilReader(_BaseReader):
    """Reader for PNG and JPEG. These formats cannot be decoded by region, so the
    image is decoded once on first use, converted to 8-bit and cached."""

    def __init__(self, path: str, image) -> None:
        """Remember the PIL image; decoding is deferred until the first crop."""
        self._path, self._image = path, image
        self.width, self.height = image.size
        self.bw = image.mode == "1"
        self._pixels: np.ndarray | None = None
        self._lock = threading.Lock()

    def _load(self) -> np.ndarray:
        """Decode the image once, converting every PIL mode to uint8 gray or RGB."""
        with self._lock:
            if self._pixels is None:
                self._pixels = self._decode()
            return self._pixels

    def _decode(self) -> np.ndarray:
        """Convert the PIL image to a numpy array, mapping each mode to 8-bit."""
        img = self._image
        try:
            img.load()
            mode = img.mode
            if mode == "1":
                return np.asarray(img, dtype=np.uint8) * 255 if img.getextrema()[1] <= 1 else np.asarray(img.convert("L"))
            if mode in ("I;16", "I;16L", "I;16B", "I"):
                return (np.asarray(img).astype(np.uint32) >> 8).clip(0, 255).astype(np.uint8)
            if mode in ("L", "LA", "La"):
                return np.asarray(img.convert("L"))
            return np.asarray(img.convert("RGB"))
        except (OSError, ValueError, SyntaxError) as exc:
            raise CorruptFile(f"image data could not be decoded: {exc}", "truncated") from exc

    def _read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Slice the cached decoded array."""
        return self._load()[y0:y1, x0:x1]

    def close(self) -> None:
        """Drop the decoded pixels and close the file."""
        self._pixels = None
        self._image.close()


def _check_path(path: str) -> Path:
    """Validate existence, file type, symlink target and size limits."""
    p = Path(path)
    if not p.exists():
        raise UnsupportedFile(f"file not found: {path}", "file_not_found")
    if p.is_symlink() and p.resolve().parent != p.absolute().parent.resolve():
        raise UnsupportedFile("symlinks pointing outside the upload folder are not allowed", "symlink_rejected")
    if not p.is_file():
        raise UnsupportedFile("path is not a regular file", "not_a_file")
    size = p.stat().st_size
    if size == 0:
        raise CorruptFile("file is empty", "empty_file")
    if size > MAX_FILE_BYTES:
        raise TooLarge(f"file is {size / 1e6:.0f} MB; the limit is {MAX_FILE_BYTES // 2**20} MB", "file_too_large")
    return p


def _check_dimensions(width: int, height: int, warnings: list[str], max_pixels: int = MAX_PIXELS) -> None:
    """Reject zero or oversized images and warn on absurd aspect ratios."""
    if width <= 0 or height <= 0:
        raise CorruptFile("image has zero width or height", "zero_size")
    if width * height > max_pixels:
        raise TooLarge(f"image has {width * height / 1e6:.0f} megapixels; the limit is {max_pixels // 10**6}", "image_too_large")
    if max(width / height, height / width) > ASPECT_WARN:
        warnings.append("absurd_aspect_ratio")


def _tiff_dpi(page) -> float | None:
    """Read dpi from TIFF resolution tags, converting centimetres to inches."""
    xres, unit = page.tags.get("XResolution"), page.tags.get("ResolutionUnit")
    if xres is None:
        return None
    num, den = xres.value
    value = num / den if den else 0.0
    if value <= 0:
        return None
    return value * 2.54 if unit is not None and unit.value == 3 else float(value)


def _check_truncation(page, size: int) -> None:
    """Raise CorruptFile when strip or tile data extends past the end of the file."""
    ends = [o + c for o, c in zip(page.dataoffsets, page.databytecounts)]
    if not ends or max(ends) > size:
        raise CorruptFile("TIFF data is truncated", "truncated")


def _open_tiff(p: Path) -> UploadedScan:
    """Open a TIFF with tifffile and wrap page 0 in a TiffReader."""
    import tifffile
    warnings: list[str] = []
    tif = tifffile.TiffFile(str(p))
    try:
        if len(tif.pages) == 0:
            raise CorruptFile("TIFF has no pages", "no_pages")
        if len(tif.pages) > 1:
            warnings.append("multi_page_tiff_using_page_0")
        page = tif.pages[0]
        _check_dimensions(int(page.imagewidth), int(page.imagelength), warnings, TIFF_MAX_PIXELS)
        _check_truncation(page, p.stat().st_size)
        reader = TiffReader(str(p), tif, page)
    except BaseException:
        tif.close()
        raise
    mode = "bw" if reader.bw else "rgb"
    return UploadedScan(str(p), reader.width, reader.height, mode, _tiff_dpi(page), reader, warnings)


def _open_pil(p: Path, kind: str) -> UploadedScan:
    """Open a PNG or JPEG with PIL, enforcing the pixel cap before any decode."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2
    warnings: list[str] = []
    try:
        img = Image.open(str(p))
        width, height = img.size
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise CorruptFile(f"{kind.upper()} could not be opened: {exc}", "unreadable") from exc
    _check_dimensions(width, height, warnings)
    if width * height > LARGE_FULL_DECODE_PIXELS:
        warnings.append("full_decode_required")
    dpi = img.info.get("dpi")
    dpi_value = float(dpi[0]) if dpi and dpi[0] else None
    reader = PilReader(str(p), img)
    return UploadedScan(str(p), width, height, "bw" if reader.bw else "rgb", dpi_value, reader, warnings)


def read_upload(path: str) -> UploadedScan:
    """Validate and open an uploaded scan without decoding the full image."""
    p = _check_path(path)
    with open(p, "rb") as fh:
        kind = sniff_type(fh.read(16))
    if kind == "pdf":
        raise UnsupportedFile("PDF files are not supported; export the scan as TIFF, PNG or JPEG", "pdf_unsupported")
    if not kind:
        raise UnsupportedFile("file content is not a TIFF, PNG or JPEG image", "unsupported_type")
    try:
        return _open_tiff(p) if kind == "tiff" else _open_pil(p, kind)
    except ToolError:
        raise
    except Exception as exc:
        raise CorruptFile(f"{kind.upper()} file is corrupt or unreadable: {exc}", "corrupt") from exc
