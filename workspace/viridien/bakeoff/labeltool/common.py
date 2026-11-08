"""Shared paths, isolated harness caches and interval operations."""
import csv
import importlib
import json
import shutil
import sys
import threading
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HOME = ROOT / 'labeltool'
DATA = HOME / 'data'
CACHE = HOME / 'cache'
HUMAN = ROOT / 'gt' / 'human'
FIELDS = ['top_depth', 'bottom_depth', 'lithology', 'top_y', 'bottom_y', 'cluster_id']
GOLD = '0000000009_d-582_0000020435-0001_log06_fe_5in_md'
_OCR_LOCAL = threading.local()


def isolated_ocr():
    """Each worker owns its OCR wrapper; ONNX workers never share mutable state."""
    if not hasattr(_OCR_LOCAL, 'engine'):
        from rapidocr_onnxruntime import RapidOCR
        _OCR_LOCAL.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2,
                                    det_limit_side_len=3200, det_limit_type='max')
    return _OCR_LOCAL.engine


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def load_rows(path):
    if not Path(path).exists():
        return []
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []  # Another job may be atomically replacing a source CSV.
    parsed=[]
    for row in rows:
        try:
            top,bottom=int(float(row['top_y'])),int(float(row['bottom_y']))
            if bottom>top and row.get('lithology') is not None:
                parsed.append(dict(row,top_y=top,bottom_y=bottom))
        except (KeyError,TypeError,ValueError):
            continue  # Ignore an unfinished final row until the next source refresh.
    return parsed


def normalize(label):
    value = ''.join(str(label).casefold().split())
    return {'sandandsandstone': 'sandstone'}.get(value, value)


def family(name):
    if 'd-582' in name: return 'D-582 1-bit grid'
    if 'MC765' in name: return 'MC765 colour grid'
    if 'mc761' in name.lower(): return 'MC761 mineral/lithology'
    if any(k in name for k in ('Rio_', 'Critter', 'Callaway')): return "Texas 'Rock Types' box"
    if any(k in name for k in ('bp_gc743', 'gb506', 'hadrian', 'gc743_dc141')): return 'Geoservices grid'
    if any(k in name for k in ('Allegheny', 'Shell_', 'FREEMAN', 'Trail')): return 'One-off layout'
    return 'Gulf 3-panel'


def harness_for(name):
    """All harness writes land inside labeltool, including imported v2 globals."""
    from harness import io, depth, legend as v1
    target = CACHE / 'harness'
    io.ROOT = target
    v1._ocr = isolated_ocr
    for kind in ('legend', 'legend_v2', 'depth'):
        src = ROOT / 'out' / '_cache' / kind
        dst = target / 'out' / '_cache' / kind
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.glob(name + '.*'):
            if not (dst / path.name).exists():
                shutil.copyfile(path, dst / path.name)
    if (ROOT / 'harness' / 'legend_v2.py').exists():
        importlib.invalidate_caches()
        legend = importlib.import_module('harness.legend_v2')
        legend.CACHE = target / 'out' / '_cache' / 'legend_v2'
    else:
        legend = importlib.import_module('harness.legend')
    return io, legend, depth


def external_sources(name):
    """Discover current model outputs by manifest, summary, or vocabulary log."""
    found = {}
    manifest = read_json(ROOT / 'labelling' / 'out' / 'manifest.json', {})
    folders = ROOT / 'labelling' / 'out'
    for folder in sorted(folders.iterdir()) if folders.exists() else []:
        if not folder.is_dir(): continue
        meta = manifest.get(folder.name, {})
        for file in ('summary.json', 'vocab.json'):
            data = read_json(folder / file, {})
            if data.get('log'): meta = data
        if meta.get('log') != name: continue
        for source in ('consensus', 'S', 'T'):
            path = folder / (source + '.csv')
            if path.exists(): found['labelling/' + folder.name + '/' + source] = load_rows(path)
    return found


def suggestions_for(sources, top, bottom):
    result = []
    for source, rows in sources.items():
        votes = {}
        disputed = False
        for row in rows:
            overlap = max(0, min(bottom, row['bottom_y']) - max(top, row['top_y']))
            if overlap:
                key = row.get('lithology', 'Unknown')
                votes[key] = votes.get(key, 0) + overlap
                disputed |= str(row.get('agreed', 'true')).lower() in ('false', '0')
                disputed |= str(row.get('disputed', 'false')).lower() in ('true', '1')
        for label, n in sorted(votes.items(), key=lambda p: -p[1]):
            result.append(dict(source=source, label=label, coverage=round(n/(bottom-top), 4),
                               tier='low' if disputed or len(votes)>1 else 'medium',
                               reason='Disputed consensus' if disputed else 'Overlapping machine interval',
                               disputed=disputed, truth=source == 'gold'))
    return result


def tier_for(suggestions, d1, margin, has_swatches):
    labels = {normalize(s['label']) for s in suggestions if not s.get('truth')}
    if not has_swatches: return 'low', 'Legend has no parsed lithology swatches'
    if d1 is None: return 'low', 'Hatch-only, grey, blank or ambiguous colour'
    if d1 > 25: return 'low', 'Colour not in the legend (distance > 25)'
    if len(labels) > 1: return 'low', 'Sources disagree'
    if any(s.get('disputed') for s in suggestions): return 'low', 'Disputed consensus'
    if d1 <= 12 and margin >= 15: return 'high', 'Close colour fit; separated swatches; all available sources agree'
    return 'medium', 'Colour fits, but fit is weaker or a rival swatch is near'


def overlay(rows, row):
    """Last decision wins; trim older spans before inserting a new span."""
    top, bottom = row['top_y'], row['bottom_y']
    result = []
    for old in rows:
        if old['bottom_y'] <= top or old['top_y'] >= bottom:
            result.append(old)
        else:
            if old['top_y'] < top: result.append(dict(old, bottom_y=top))
            if old['bottom_y'] > bottom: result.append(dict(old, top_y=bottom))
    result.append(dict(row))
    return sorted(result, key=lambda r: r['top_y'])
