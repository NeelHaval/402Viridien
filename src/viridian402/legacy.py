"""Explicit adapter for the team's frozen S11 FFT model, with automatic legend OCR.

Original snapshot: silmoon04/hacktheburgh26@663721e72e3fa2ba496987c13b66610e52d1c8f3.
No label files or precomputed answer caches are vendored. Each invocation uses
an empty disposable OCR cache. See docs/model-interface.md for limits.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile
from functools import lru_cache

from PIL import Image


def _ocr_engine(factory):
    engine=factory(intra_op_num_threads=2,inter_op_num_threads=2,
                   det_model_path=None,det_limit_side_len=3200,det_limit_type='max')
    if not hasattr(engine,'text_rec'):
        if not hasattr(engine,'text_recognizer'):
            raise RuntimeError('RapidOCR lacks a supported batch recognition interface')
        engine.text_rec=engine.text_recognizer
    return engine


def predict(request: dict, config: dict) -> list[dict]:
    # Imported only when requested, so the standard CPU harness stays small.
    legacy_root = Path(__file__).parent / "_legacy"
    sys.path.insert(0, str(legacy_root))
    try:
        from harness import io, legend, legend_v2, depth
        from strategies import s11_fft_legend
    except ImportError as error:
        raise RuntimeError("Install legacy dependencies: python -m pip install -e '.[legacy]'") from error
    # RapidOCR 1.2.x requires model_path in any detector-options override.
    # Keep the frozen model source unchanged and adapt that constructor here.
    from rapidocr_onnxruntime import RapidOCR
    legend._ocr = lru_cache(maxsize=1)(lambda: _ocr_engine(RapidOCR))
    annotation = request["annotation"]
    if isinstance(annotation, (str, Path)):
        annotation = json.loads(Path(annotation).read_text(encoding="utf-8"))
    boxes = annotation.get("bounding_boxes", annotation)
    with Image.open(request["image"]) as header:
        # Pillow's fallback can decode the complete original page on .crop().
        # Conservative byte guard includes decoder storage and RGB conversion.
        channels = max(4, len(header.getbands()) * 4)
        estimate = header.width * header.height * channels
        if estimate > int(config.get("max_decode_bytes", 1_200_000_000)):
            raise ValueError("Legacy TIFF decode exceeds configured byte budget; use a streaming model")
    with tempfile.TemporaryDirectory(prefix="viridian-s11-") as temporary:
        io.ROOT = Path(temporary)
        legend_v2.CACHE = io.ROOT / "out" / "_cache" / "legend_v2"
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(request["image"])
        log = io.Log(image, boxes, hashlib.sha256(request["id"].encode()).hexdigest()[:24])
        try:
            from .data import read_rgb
            # Reuse the bounded reader rather than legacy's unguarded fallback.
            io._crop = lambda opened, box: Image.fromarray(read_rgb(
                request["image"], box,
                max_window_pixels=int(config.get("max_window_pixels", 12_000_000)),
                max_decode_bytes=int(config.get("max_decode_bytes", 1_200_000_000))))
            if request.get("legend"):
                entries = []
                for item in request["legend"]:
                    box = item.get("swatch_box", item.get("box"))
                    if box is None:
                        raise ValueError("Supplied legend requires full-image swatch boxes")
                    patch = io.read_rgb(log, box)
                    entries.append(legend.LegendEntry(int(item["id"]), item["label"], box,
                                                       patch, tuple(patch.mean(axis=(0, 1))),
                                                       bool(item.get("is_lithology", True))))
            else:
                parser = legend_v2 if int(config.get("legend_version", 2)) == 2 else legend
                entries = parser.parse(log)
            rows, _ = s11_fft_legend.run(log, tuple(request["window"]), {"legend": entries})
            depth_mode = config.get("depth", "skip")
            if depth_mode not in ("skip", "ocr"):
                raise ValueError("depth must be skip or ocr")
            if depth_mode == "ocr":
                calibration = depth.calibrate(log)
                unit = config.get("image_depth_unit")
                if unit not in ("ft", "m"):
                    raise ValueError("OCR depth needs explicit image_depth_unit ft or m")
                factor = 1 / 0.3048 if unit == "m" else 1
                for row in rows:
                    row.update(top_depth=round(calibration.to_depth(row["top_y"]) * factor, 3),
                               bottom_depth=round(calibration.to_depth(row["bottom_y"]) * factor, 3),
                               depth_unit="ft", depth_datum=config.get("depth_datum", "unknown"))
            return rows
        finally:
            log.close()
