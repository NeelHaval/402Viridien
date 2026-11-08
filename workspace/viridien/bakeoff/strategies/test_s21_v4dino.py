"""Smoke checks for S21; uses cached model and synthetic image pixels only."""
import json
import os
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from harness import io
from strategies import scoreboard_v4 as base
from strategies import s21_v4dino as s


def main():
    # Distinct native pixels must tile, with no interpolation or rescaling.
    small = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    tiled = s.canvas(small)
    assert tiled.shape == (224, 224, 3)
    assert np.array_equal(tiled[:2, :4], np.tile(small, (1, 2, 1)))
    wide = np.broadcast_to(np.arange(240, dtype=np.uint8)[None, :, None], (1, 240, 3))
    assert np.array_equal(s.canvas(wide)[0, :, 0], np.arange(224, dtype=np.uint8))
    assert os.environ['HF_HUB_OFFLINE'] == '1'
    # Find an existing native legend, then make a small in-memory column.
    cache = base._native()
    name = next(name for name, entries in cache.items() if len(entries) >= 2)
    classifier = base.make_classifier(SimpleNamespace(name=name), {})
    labels, images = s._swatches(classifier, name)
    assert labels and len(images) == 3 * (len(labels) // 3)
    image = images[1]
    column = np.tile(image, ((288 + image.shape[0] - 1) // image.shape[0],
                             (96 + image.shape[1] - 1) // image.shape[1], 1))[:288, :96]
    log = io.Log(Image.fromarray(column), {'interpreted_lithology_column': [0, 0, 96, 288]}, name)
    ctx = {'calib': SimpleNamespace(to_depth=lambda y: y * 0.1), 'device': 'cuda'}
    started = time.perf_counter()
    rows = s.run(log, (16, 272), ctx)
    elapsed = time.perf_counter() - started
    diag = s.run.last_diagnostics
    assert diag['dino_swatch_vectors'] == len(images)
    assert diag['dino_chunk_vectors'] > 0 and diag['model']['weights_sha256']
    assert all(16 <= r['top_y'] < r['bottom_y'] <= 272 for r in rows)
    assert all(rows[i]['bottom_y'] <= rows[i + 1]['top_y'] for i in range(len(rows) - 1))
    assert diag['model']['device'].lower().find('4080') >= 0
    # Uncached log is not repaired with parser, and stays Unknown.
    absent = io.Log(Image.fromarray(column), {'interpreted_lithology_column': [0, 0, 96, 288]}, '__missing_native__')
    missing_rows = s.run(absent, (16, 272), ctx)
    assert not s.run.last_diagnostics['native_cache_present']
    assert all(r['lithology'] == 'Unknown' for r in missing_rows)
    print(json.dumps({'status': 'PASS', 'rows': len(rows), 'chunks': diag['dino_chunk_vectors'],
                      'swatch_vectors': diag['dino_swatch_vectors'], 'seconds_cold': round(elapsed, 2),
                      'model': diag['model'], 'missing_cache_rows': len(missing_rows)}), flush=True)

if __name__ == '__main__':
    main()
