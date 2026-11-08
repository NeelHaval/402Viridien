"""Benedek d39919f rule + detected legend + full visual lithology matcher.

The core functions are copied with an AST allowlist from the exact notebook.
Only input/output access and the existing OCR cache path are adapted here.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from harness import io, depth
from . import s24_benedek_core as core

NAME = 'S24 Benedek d39919f exact notebook inference'
NEEDS_GPU = False
PAID_API = None
SOURCE_COMMIT = 'd39919f1bbcf254b357a99eba1cf0845c391b135'


def ocr_assets():
    """Return paths and hashes only; do not download absent OCR weights."""
    roots = [Path.home()/'.EasyOCR'/'model', io.ROOT/'out'/'_cache'/'easyocr'/'model']
    for root in roots:
        paths = [root/name for name in ('craft_mlt_25k.pth', 'english_g2.pth')]
        if all(p.is_file() for p in paths):
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    raise RuntimeError('S24 requires cached EasyOCR craft_mlt_25k.pth and english_g2.pth; no OCR substitution or download is allowed')


@lru_cache(maxsize=1)
def _reader():
    import easyocr
    assets = ocr_assets()
    model_dir = str(Path(next(iter(assets))).parent)
    return easyocr.Reader(['en'], gpu=False, verbose=False,
                          model_storage_directory=model_dir, download_enabled=False)


class _Image:
    def __init__(self, log):
        self.log = log

    def crop(self, box):
        return io._crop(self.log, tuple(map(int, box)))


def run(log, window, ctx):
    """Run original detector, OCR, splitting, naming and merging, then depth."""
    y0, y1 = map(int, window)
    bounds = log.boxes['interpreted_lithology_column']
    y0, y1 = max(y0, int(bounds[1])), min(y1, int(bounds[3]))
    if y1 <= y0:
        return []
    core.torch.set_num_threads(2)
    core.read_swatch_labels.reader = _reader()
    image = _Image(log)
    legend_box = tuple(map(int, log.boxes['lithology_legend']))
    boxes = core.detect_swatches(image, legend_box)
    names = core.read_swatch_labels(image, boxes, legend_box)
    legend = core.pd.DataFrame([
        {'legend_id': i, 'lithology': name or f'legend_{i}', 'box': box}
        for i, (box, name) in enumerate(zip(boxes, names))
    ], columns=['legend_id', 'lithology', 'box'])
    column = io.read_rgb(log, 'interpreted_lithology_column', y0, y1)
    peaks = core.ink_boundaries(core.column_features(column))
    segments = core.to_table(peaks, len(column), y0)
    _, layers = core.lithology_layers(segments, column, y0, legend, image, split=True)
    calib = ctx.get('calib')
    if calib is None:
        calib = depth.calibrate(log)
    rows = []
    for row in layers.itertuples():
        # The notebook uses -1 for blank, which is not a rock interval.
        if int(row.legend_id) < 0:
            continue
        result = dict(top_y=int(row.y_top), bottom_y=int(row.y_bottom),
                      lithology=str(row.lithology), cluster_id=int(row.legend_id))
        if calib is not None:
            result.update(top_depth=round(calib.to_depth(row.y_top), 1),
                          bottom_depth=round(calib.to_depth(row.y_bottom), 1))
        rows.append(result)
    return rows
