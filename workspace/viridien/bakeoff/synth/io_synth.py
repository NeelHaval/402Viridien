"""harness.io-compatible fake-log loader and a reversible in-process adapter."""
from contextlib import contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch

from PIL import Image
from harness import io as real_io
from .statistics import ROOT, dump, read_json

HERE = Path(__file__).resolve().parent
FAKE_ROOT = HERE / 'data/fake'
Log = real_io.Log
read_rgb = real_io.read_rgb
load_rows = real_io.load_rows
save_rows = real_io.save_rows
FIELDS = real_io.FIELDS
_ACTIVE_ROOT = FAKE_ROOT
_ACTIVE_POOL = None
_ACTIVE_VARIANT = 'image'


def list_logs(root=None, pool=None):
    root = Path(root or _ACTIVE_ROOT)
    pool = pool if pool is not None else _ACTIVE_POOL
    return [p.parent.name for p in sorted(root.glob('*/meta.json'))
            if pool in (None, 'all') or read_json(p)['pool'] == pool]


def open_log(name, root=None, variant=None):
    root = Path(root or _ACTIVE_ROOT).resolve()
    folder = (root / name).resolve()
    if folder.parent != root:
        raise ValueError('Invalid fake log name')
    meta = read_json(folder / 'meta.json')
    if meta is None:
        raise FileNotFoundError('Unknown fake log')
    variant = variant or _ACTIVE_VARIANT
    filename = 'image_clean.png' if variant == 'clean' else meta['image_file']
    data = read_json(folder / 'boxes.json')
    log = Log(Image.open(folder / filename), data['bounding_boxes'], name)
    # The real loader probes filenames as TIFF. Fake PNGs and small LZW TIFFs
    # use Pillow directly; no full real page is decoded by this adapter.
    log._strip_backend = False
    return log


@contextmanager
def patched(root=FAKE_ROOT, pool='test', variant='image', runtime=HERE):
    """Patch existing modules, including imported io aliases, without edits.

    ROOT moves to synth/ so harness.run output, depth/legend caches and strategy
    caches stay here. Its strategy lookup still resolves the original modules.
    Donor parser results and known linear calibration are warm caches: OCR and
    depth-estimation quality are deliberately outside this synthetic score.
    Not thread-safe: use around an in-process evaluation, never concurrent runs.
    """
    global _ACTIVE_ROOT, _ACTIVE_POOL, _ACTIVE_VARIANT
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    if not runtime.is_relative_to(HERE):
        raise ValueError('Runtime writes must stay in synth/')
    from harness import run, depth, legend_v2
    import importlib
    names = list_logs(root, pool)
    for name in names:
        folder = root / name
        for source in (folder / 'parser_cache').glob('*/*'):
            target = runtime / 'out/_cache' / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != source.read_bytes():
                shutil.copyfile(source, target)
        meta = read_json(folder / 'meta.json')
        for engine in ('rapidocr', 'easyocr'):
            dump(runtime / 'out/_cache/depth' / (name + '.' + engine + '.json'),
                 dict(version=depth.VERSION, calibration=meta['calibration'], seconds=0, engine=engine, points=[]))
    dump(runtime / 'eval_set.json', [dict(log=n, y0=0, y1=read_json(root / n / 'meta.json')['height'],
                                         tier='synthetic') for n in names])

    def strategy(name):
        if name == '_dummy':
            return importlib.import_module('harness._dummy')
        matches = sorted((ROOT / 'strategies').glob(name + '*.py'))
        if len(matches) != 1:
            raise ValueError('Strategy must resolve to exactly one module')
        return importlib.import_module('strategies.' + matches[0].stem)

    original_truth = run._truth

    def truth(name, source, window):
        if name in names:
            return load_rows(root / name / 'truth.csv')
        return original_truth(name, source, window)

    old = (_ACTIVE_ROOT, _ACTIVE_POOL, _ACTIVE_VARIANT)
    _ACTIVE_ROOT, _ACTIVE_POOL, _ACTIVE_VARIANT = root, pool, variant
    try:
        with patch.object(real_io, 'ROOT', runtime), \
             patch.object(real_io, 'list_logs', list_logs), \
             patch.object(real_io, 'open_log', open_log), \
             patch.object(real_io, 'read_rgb', read_rgb), \
             patch.object(legend_v2, 'CACHE', runtime / 'out/_cache/legend_v2'), \
             patch.object(run, '_strategy', strategy), patch.object(run, '_truth', truth):
            yield
    finally:
        _ACTIVE_ROOT, _ACTIVE_POOL, _ACTIVE_VARIANT = old


patch_harness = patched
