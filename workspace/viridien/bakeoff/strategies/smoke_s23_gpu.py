"""Small GPU smoke with actual D-582 swatches and an in-memory column."""
import json
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from harness import io
from strategies import scoreboard_v4 as base
from strategies import s23_v4dino_identity as strategy


def main():
    cache = base._native()
    name = next(n for n in cache if 'd-582' in n.lower() and 'log06' in n.lower())
    source = next(a for label, a in cache[name] if label == 'Shale')
    col = np.tile(source, ((288 + source.shape[0] - 1) // source.shape[0],
                           (96 + source.shape[1] - 1) // source.shape[1], 1))[:288, :96]
    log = io.Log(Image.fromarray(col), {'interpreted_lithology_column': [0, 0, 96, 288]}, name)
    ctx = {'calib': SimpleNamespace(to_depth=lambda y: y * .1), 'device': 'cuda'}
    start = time.perf_counter()
    rows = strategy.run(log, (16, 272), ctx)
    elapsed = time.perf_counter() - start
    diag = strategy.run.last_diagnostics
    labels = {label for label, _ in cache[name]}
    assert rows and all(r['lithology'] in labels | {'Unknown'} for r in rows), rows
    assert diag['dino_chunk_vectors'] > 0 and diag['model']['weights_sha256']
    assert diag['legend_identity'] == 'exact printed caption'
    print(json.dumps({'status': 'PASS', 'seconds_cold': round(elapsed, 2),
                      'rows': len(rows), 'labels': [r['lithology'] for r in rows],
                      'chunks': diag['dino_chunk_vectors'], 'model': diag['model']}), flush=True)

if __name__ == '__main__':
    main()
