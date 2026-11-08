"""Native interval-safe mining, offline DINOv2 ranking, and budgeted JEV checks."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
import json
import sqlite3
import threading
import time
from types import SimpleNamespace
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw
from harness import io
from synth.common import (ROOT, HOME, DATA, CACHE, ANALYSIS, read_json, write_json,
                          norm, local_file, relative, tile, font)
from synth.patterns import target
from synth.colour import profile, colour_gate, contrastive, PaletteMatcher, lab

TRUST = {'gold': 1.0, 'consensus': .9, 'prefill/JEV': .75, 'colour/high': .55, 'embedding': .3}
PROMPT = ('Which legend rock best matches the QUERY PATCH? Compare colour, hatch orientation, '
          'density and shapes. Choose only from the supplied rock vocabulary.')
_LEGENDS = {}
_PANELS = {}
_PANEL_LOCK = threading.Lock()


def load_csv(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [dict(r, top_y=int(float(r['top_y'])), bottom_y=int(float(r['bottom_y']))) for r in csv.DictReader(f)]


def source_intervals():
    """Do not mistake silver suggestions or disputed consensus for trusted seeds."""
    result = defaultdict(list)
    for path in sorted((ROOT / 'gt' / 'gold').glob('*.csv')):
        result[path.stem].extend(dict(r, source='gold') for r in load_csv(path))
    manifest = read_json(ROOT / 'labelling' / 'out' / 'manifest.json', {})
    for path in sorted((ROOT / 'labelling' / 'out').glob('*/consensus.csv')):
        meta = read_json(path.parent / 'summary.json', {})
        name = meta.get('log') or read_json(path.parent / 'vocab.json', {}).get('log') or manifest.get(path.parent.name, {}).get('log')
        if name:
            result[name].extend(dict(r, source='consensus') for r in load_csv(path)
                                if str(r.get('agreed', '')).lower() in ('true', '1')
                                and str(r.get('disputed', 'false')).lower() not in ('true', '1'))
    seen = set()
    for filename in ('prefill.jsonl', 'queue.jsonl'):
        path = ROOT / 'labeltool' / 'data' / filename
        if not path.exists(): continue
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                for s in row.get('suggestions', []):
                    source = s.get('source', '').lower()
                    if row.get('mock') or s.get('mock') or s.get('disputed'): continue
                    if source.startswith('jev/') and ('cluster' in source or 'interval' in source): kind = 'prefill/JEV'
                    elif ('colour' in source or 'color' in source) and s.get('tier') == 'high': kind = 'colour/high'
                    else: continue
                    key = row['log'], row['top_y'], row['bottom_y'], norm(s['label']), kind
                    if key in seen: continue
                    seen.add(key)
                    result[row['log']].append(dict(top_y=row['top_y'], bottom_y=row['bottom_y'],
                                                  lithology=s['label'], source=kind))
    return result


def repeat_height(rgb):
    """First strong autocorrelation peak across native vertical scan lines."""
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    h, w = gray.shape
    if h < 12: return 0
    periods = []
    for x in np.linspace(w * .15, max(w * .15, w * .85 - 1), min(12, w)).astype(int):
        v = gray[:, x] - gray[:, x].mean()
        if v.std() < 4: continue
        f = np.fft.rfft(v, 2 * h)
        ac = np.fft.irfft(f * f.conj())[:h] / (v @ v + 1e-9)
        for lag in range(3, min(201, h // 2)):
            if ac[lag] > .3 and ac[lag] >= ac[lag-1] and ac[lag] >= ac[lag+1]:
                periods.append(lag)
                break
    return int(np.median(periods)) if periods else 0


def crop_windows(top, bottom, height, stride, margin=8):
    start, end = int(top) + margin, int(bottom) - margin - height
    if end < start: return []
    positions = list(range(start, end + 1, max(1, int(stride))))
    if end != positions[-1]: positions.append(end)
    return [(y, y + height) for y in positions]


def quality(rgb):
    """Reject empty fills, crossing rules, and rows of glyph-sized components."""
    h, w = rgb.shape[:2]
    inner = rgb[:, max(1, w//12):max(2, w-w//12)]
    if not inner.size: return False
    ink = np.min(inner, axis=2) < 235
    if ink.mean() < .012: return False
    dark = np.max(inner, axis=2) < 95
    if np.any(dark.mean(axis=1) > .88): return False
    vertical = cv2.morphologyEx(dark.astype('uint8'), cv2.MORPH_OPEN, np.ones((max(16, h*3//4), 1), np.uint8))
    if vertical.mean() > .07: return False
    count, _, stats, centers = cv2.connectedComponentsWithStats(dark.astype('uint8'), 8)
    glyphs = [(s, c) for s, c in zip(stats[1:], centers[1:])
              if 4 <= s[2] <= 18 and 6 <= s[3] <= 24 and s[4] >= 9 and .15 <= s[2]/s[3] <= 1.4]
    if len(glyphs) >= 4:
        # A text line has several similar-height components on one baseline,
        # with mostly blank rows above and below it. Repeating hatch is retained.
        ys = np.array([c[1] for s, c in glyphs])
        if any(np.sum(abs(ys-y) < 3) >= 4 for y in ys) and np.mean(ink.mean(axis=1) < .02) > .45:
            return False
    return True


def detected_intervals(rgb, offset):
    """Conservative search beds: horizontal rules, blank transitions, colour jumps."""
    inner = rgb[:, max(1, rgb.shape[1]//12):max(2, rgb.shape[1]*11//12)]
    profile = inner.astype(np.float32).mean(axis=1)
    smooth = cv2.blur(profile, (1, 9))
    ink = (inner.min(axis=2) < 235).mean(axis=1)
    rules = (inner.max(axis=2) < 95).mean(axis=1) > .85
    changes = np.linalg.norm(smooth[5:] - smooth[:-5], axis=1) > 55
    marks = np.flatnonzero(rules).tolist() + (np.flatnonzero(changes)+2).tolist()
    marks += (np.flatnonzero(np.diff((ink > .012).astype(int)))+1).tolist()
    cuts = [0]
    for y in sorted(set(marks)):
        if y-cuts[-1] >= 4: cuts.append(int(y))
    if cuts[-1] != len(rgb): cuts.append(len(rgb))
    return [(a+offset, b+offset) for a, b in zip(cuts, cuts[1:])]


class Embedder:
    """Bench's DINOv2-small, full-query resize, persistent content-hash cache."""
    def __init__(self, cache=CACHE / 'embeddings.sqlite', backend=None):
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache)
        self.db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB)')
        self.backend = backend

    def __call__(self, images):
        images = [np.asarray(im, dtype=np.uint8) for im in images]
        keys = [hashlib.sha256(b'dinov2-small-full224-v1' + str(im.shape).encode() + im.tobytes()).hexdigest() for im in images]
        found, missing = {}, {}
        for key, im in zip(keys, images):
            if key in found or key in missing: continue
            row = self.db.execute('SELECT vector FROM embeddings WHERE key=?', (key,)).fetchone()
            if row: found[key] = np.frombuffer(row[0], dtype=np.float32)
            else: missing[key] = im
        if missing:
            if self.backend is None:
                import torch
                from strategies.s05_dinov2 import _embed
                torch.set_num_threads(2)
                self.backend = lambda ims: _embed(ims, 'cuda' if torch.cuda.is_available() else 'cpu')
            items = list(missing.items())
            for start in range(0, len(items), 96):
                batch = items[start:start+96]
                vectors = self.backend([cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA) for _, im in batch])
                for (key, _), v in zip(batch, vectors):
                    v = np.asarray(v, np.float32)
                    v = v / max(np.linalg.norm(v), 1e-8)
                    found[key] = v
                    self.db.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?)', (key, v.tobytes()))
            self.db.commit()
        return np.stack([found[key] for key in keys])

    def close(self): self.db.close()


def diverse(candidates, limit=40, threshold=.97):
    """Final-set per-log cap: 25% with >= 4 logs, else an even split across the logs present.

    First dedupe within logs, then choose a feasible size and prefer underfilled
    logs to prevent a duplicate in a common log starving all rarer logs.
    """
    groups = defaultdict(list)
    ordered = sorted(candidates, key=lambda c: (-c.get('score', 0), c['log'], c['top_y']))
    for c in ordered:
        if not any(float(c['_embedding'] @ x['_embedding']) > threshold for x in groups[c['log']]):
            groups[c['log']].append(c)
    for n in range(min(limit, sum(map(len, groups.values()))), 0, -1):
        # 25% per log when >= 4 logs contribute; with fewer logs (one-off legends) spread evenly instead of returning nothing
        cap = n // 4 if len(groups) >= 4 else -(-n // len(groups))
        if sum(min(cap, len(v)) for v in groups.values()) < n: continue
        chosen, counts = [], Counter()
        # Round-robin by score among logs, then fill remaining slots. Global
        # deduplication is checked during selection, including across logs.
        for turn in range(cap):
            for log in sorted(groups, key=lambda k: (-groups[k][0].get('score', 0), k)):
                if counts[log] > turn: continue
                for c in groups[log]:
                    if any(c is x for x in chosen): continue
                    if any(float(c['_embedding'] @ x['_embedding']) > threshold for x in chosen): continue
                    chosen.append(c); counts[log] += 1
                    break
                if len(chosen) == n: return sorted(chosen, key=lambda c: -c.get('score', 0))
    return []


def candidate_score(c):
    if 'margin' in c: return c['margin']
    check = c.get('verification') or {}
    agree = check.get('agreement')
    verification = (.5 + .5 * check.get('probability', 0)) if agree else (0 if agree is False else .25)
    return .4 * TRUST[c['source']] + .4 * verification + .2 * max(0, c.get('similarity', 0))


def verification_panel(pattern, candidate, patterns):
    """Use the benchmark's exact gallery/choice panel and prompt, one query."""
    from classify.data import panel
    rocks = sorted((p for p in patterns if p['legend_type'] == pattern['legend_type']), key=lambda p: p['pid'])
    method = 'jev_gallery' if len(rocks) <= 15 else 'jev_choice'
    vocabulary = [p['label'] for p in rocks]
    entries = []
    for p in rocks:
        e = next((s for s in p['swatches'] if s.get('log') == candidate['log']), p['swatches'][0])
        with Image.open(local_file(e.get('native_file', e['file']))) as sw:
            entries.append(SimpleNamespace(label=p['label'], swatch=np.asarray(sw.convert('RGB'))))
    case = dict(entries=entries, vocab=vocabulary)
    if method == 'jev_choice':
        # Wide search may supply a column from a different legend signature.
        # The panel must still show the target pattern's complete native legend.
        name = candidate['log'] if candidate['log'] in pattern['logs'] else pattern['swatches'][0]['log']
        if name not in _LEGENDS:
            meta = read_json(ROOT / 'out' / '_cache' / 'legend_v2' / (name+'.json'), {})
            boxes = {norm(e['label']): e['swatch_box'] for e in meta.get('entries', []) if e.get('is_lithology')}
            if not all(norm(p['label']) in boxes for p in rocks):
                raise ValueError('Choice verification requires cached native legend coordinates')
            log = io.open_log(name)
            try:
                image = Image.fromarray(io.read_rgb(log, 'lithology_legend'))
                _LEGENDS[name] = image, list(log.boxes['lithology_legend']), boxes
            finally: log.close()
        image, box, boxes = _LEGENDS[name]
        for e in entries: e.swatch_box = boxes[norm(e.label)]
        case.update(legend=image, info=dict(legend_box=box))
    with Image.open(local_file(candidate['file'])) as im:
        query = np.asarray(im.convert('RGB'))
    # The benchmark's one-query panel has a fixed query rectangle. Cache its
    # rendered legend, preserving every pixel (and hence every existing key).
    digest = hashlib.sha256(json.dumps([method, vocabulary]).encode())
    if method == 'jev_choice':
        digest.update(case['legend'].tobytes())
        digest.update(json.dumps([case['info']['legend_box'], [e.swatch_box for e in entries]]).encode())
    else:
        for e in entries:
            digest.update(str(e.swatch.shape).encode()); digest.update(e.swatch.tobytes())
    panel_key = digest.hexdigest()
    with _PANEL_LOCK:
        if panel_key not in _PANELS:
            _PANELS[panel_key] = panel(case, [{'rgb': np.full((1, 1, 3), 255, np.uint8)}],
                                       'gallery' if method == 'jev_gallery' else 'choice')
        result = _PANELS[panel_key].copy()
    image = Image.fromarray(query)
    image.thumbnail((270, 240), Image.Resampling.NEAREST)
    result.paste(image, (745, 65))
    return result, vocabulary, method


class Verifier:
    def __init__(self, max_calls=3000, workers=8, client_factory=None, cache=CACHE / 'jev', sleeper=time.sleep):
        self.max_calls, self.workers = max_calls, workers
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.calls = 0
        self.stopped = False
        self.consecutive_failures = 0
        self.failures = 0
        self.sleeper = sleeper
        if client_factory is None:
            from strategies._jev import Client
            client_factory = lambda: Client(timeout=35, retries=0)
        self.client_factory = client_factory

    def request_spec(self, pattern, c, patterns):
        image, vocab, method = verification_panel(pattern, c, patterns)
        key = hashlib.sha256(b'synth-verify-v1' + image.tobytes() +
                             json.dumps([method, vocab, PROMPT]).encode()).hexdigest()
        return image, vocab, method, key

    def verify(self, pattern, c, patterns, enabled=True):
        image, vocab, method, key = self.request_spec(pattern, c, patterns)
        path = self.cache / (key + '.json')
        # Serialize cache lookup and budget reservation, so duplicated content
        # across patterns is never billed twice, even with multiple workers.
        with self.lock:
            cached = read_json(path)
            if cached is not None and cached.get('status') == 'ok':
                return dict(cached, cache_hit=True,
                            agreement=norm(cached['label']) == norm(pattern['label']) if cached.get('label') else None)
            if not enabled: return dict(status='dry_run', key=key, method=method)
            if self.stopped: return dict(status='circuit_open', key=key, method=method)
            if self.calls >= self.max_calls: return dict(status='budget_exhausted', key=key, method=method)
            if not hasattr(self.local, 'client'): self.local.client = self.client_factory()
            client = self.local.client
        # Three total attempts. Delay after failed attempts: 2/5 seconds before
        # retry, 10 seconds cooling off an exhausted request. Every attempt is
        # budgeted and contributes to the shared consecutive-failure circuit.
        for attempt, delay in enumerate((2, 5, 10)):
            with self.lock:
                if self.stopped: return dict(status='circuit_open', key=key, method=method)
                if self.calls >= self.max_calls: return dict(status='budget_exhausted', key=key, method=method)
                self.calls += 1
            record = dict(status='unavailable', key=key, method=method)
            try:
                from strategies._jev import image_asset
                if hasattr(client, 'key') and not client.key: raise ValueError('Missing key')
                response = client.ask({'task': PROMPT, 'image': 'benchmark_image'},
                                      {'benchmark_image': image_asset(image)},
                                      {'rock': dict(type='choice', instructions=PROMPT,
                                                    criteria={str(i): v for i, v in enumerate(vocab)})})
                probs = (response or {}).get('answers', {}).get('rock', {}).get('probabilities', {})
                probabilities = np.array([float(probs.get(str(i), 0)) for i in range(len(vocab))])
                if np.isfinite(probabilities).all() and (probabilities >= 0).all() and probabilities.sum() > 0:
                    probabilities /= probabilities.sum(); best = int(probabilities.argmax())
                    record = dict(status='ok', key=key, method=method, label=vocab[best],
                                  probability=float(probabilities[best]), agreement=norm(vocab[best]) == norm(pattern['label']))
            except Exception:
                pass  # Never serialize credentials or HTTP exception text.
            with self.lock:
                if record['status'] == 'ok':
                    self.consecutive_failures = 0
                    write_json(path, record)
                    break
                self.failures += 1
                self.consecutive_failures += 1
                if self.consecutive_failures >= 8: self.stopped = True
                stopped = self.stopped
            if stopped: break
            self.sleeper(delay)
        return dict(record, cache_hit=False)

    def run(self, tasks, patterns):
        # Group identical content before scheduling, so the cache/budget is exact.
        groups = defaultdict(list)
        for p, c in tasks:
            groups[self.request_spec(p, c, patterns)[3]].append((p, c))
        def one(group):
            p, c = group[0]
            record = self.verify(p, c, patterns)
            for p, c in group:
                c['verification'] = dict(record, agreement=(norm(record['label']) == norm(p['label'])) if record.get('label') else None)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(one, groups.values()))


def clean_candidate(c):
    return {k: v for k, v in c.items() if not k.startswith('_')}


def calibrate(selected, sources):
    """Estimate a family tint only from gold/agreed crops, across >=2 labels.

    A robust residual and bounded displacement prevent unrelated label/scan
    errors from becoming permission to accept arbitrary colours.
    """
    samples = defaultdict(list)
    for p in selected:
        for e in p['swatches']:
            rows = [r for r in sources[e['log']] if r['source'] in ('gold', 'consensus')
                    and norm(r['lithology']) == norm(p['label']) and r['bottom_y']-r['top_y'] >= 64][:3]
            if not rows: continue
            with Image.open(local_file(e.get('native_file', e['file']))) as im:
                reference = profile(np.asarray(im.convert('RGB')))
            if reference['binary'] or reference['main'] is None: continue
            log = io.open_log(e['log'])
            try:
                for r in rows:
                    a, b = r['top_y']+8, min(r['bottom_y']-8, r['top_y']+200)
                    crop = profile(io.read_rgb(log, 'interpreted_lithology_column', a, b))
                    if crop['main'] is not None:
                        samples[p['family']].append((p['label'], crop['main']-reference['main'], reference, crop))
            finally: log.close()
    result = {}
    for fam in sorted({p['family'] for p in selected}):
        rows = samples[fam]
        shift = np.median([r[1] for r in rows], axis=0) if rows else np.zeros(3)
        residual = float(np.median([np.linalg.norm(r[1]-shift) for r in rows])) if rows else None
        supported = len(rows) >= 5 and len({r[0] for r in rows}) >= 2 and residual <= 5 and np.linalg.norm(shift) <= 25
        applied = shift if supported else np.zeros(3)
        accepted = sum(colour_gate(r[2], r[3], applied)[0] for r in rows)
        result[fam] = dict(samples=len(rows), measured_shift_lab=shift.tolist(), residual_deltaE=residual,
                           shift_lab=shift.tolist() if supported else [0., 0., 0.],
                           confirmed_gate_passed=accepted, confirmed_gate_pass_rate=accepted/len(rows) if rows else None,
                           status='confirmed shift' if supported else 'insufficient consistent confirmed colour matches')
    return result


def scan_identity(selected, sources, budgets):
    digest = hashlib.sha256()
    for name in ('mine.py', 'colour.py'):
        digest.update((HOME/name).read_bytes())
    lookalikes = DATA/'lookalikes.json'
    if lookalikes.exists(): digest.update(lookalikes.read_bytes())
    identity = [{k: p[k] for k in ('pid', 'family', 'legend_type', 'logs', 'swatches')} for p in selected]
    digest.update(json.dumps([identity, sources, budgets], sort_keys=True).encode())
    files = {local_file(e.get('native_file', e['file'])) for p in selected for e in p['swatches']}
    images = io._images()
    for name in images:
        files.add(images[name])
        files.add(io.DATASET/'bbox_annotation'/(name+'.json'))
    for path in sorted(files):
        stat = path.stat()
        digest.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return digest.hexdigest()


def save_scan(path, identity, pools, stats, visited, errors):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [c for pool in pools.values() for c in pool]
    vectors = np.stack([c['_embedding'] for c in rows]) if rows else np.empty((0, 0), np.float32)
    np.savez_compressed(path.with_suffix('.npz'), vectors=vectors)
    write_json(path, dict(identity=identity, pools={pid: [clean_candidate(c) for c in pool] for pid, pool in pools.items()},
                          stats={pid: dict(counts) for pid, counts in stats.items()}, visited=visited, errors=errors))


def refine_pools(pools, selected, stats, calibration):
    """Conservative clean-bed checks after discovery; never replenish a quota.

    Tiny positive margins proved unstable in visual calibration. Four-row
    colour bands catch internal contacts and coloured annotation rules that
    can pass a whole-crop fill test. Reference width outliers catch label text
    mistakenly parsed as a swatch, without dropping the catalogue entry.
    """
    refs, widths = {}, defaultdict(list)
    for p in selected:
        e = p['swatches'][0]
        with Image.open(local_file(e.get('native_file', e['file']))) as im:
            refs[p['pid']] = profile(np.asarray(im.convert('RGB')))
            widths[p['legend_type']].append(im.width)
    bylog = defaultdict(list)
    result = defaultdict(list)
    for p in selected:
        pid = p['pid']
        e = p['swatches'][0]
        with Image.open(local_file(e.get('native_file', e['file']))) as im:
            invalid = im.width > 2.5 * np.median(widths[p['legend_type']])
        p['reference_width_outlier'] = bool(invalid)
        for c in pools[pid]:
            if invalid:
                stats[pid]['reference_rejected'] += 1
            elif not p['family'].startswith('D-582') and c['margin'] < .015:
                stats[pid]['margin_floor_rejected'] += 1
            else:
                stats[pid]['after_margin'] += 1
                bylog[c['log']].append((p, c))
    for name, rows in sorted(bylog.items()):
        log = io.open_log(name)
        try:
            y0 = int(log.boxes['interpreted_lithology_column'][1])
            rgb = io.read_rgb(log, 'interpreted_lithology_column')
            for p, c in rows:
                pid = p['pid']
                reference = refs[pid]
                palette = reference['palette'] + np.asarray(calibration[p['family']]['shift_lab'])
                clean = True
                if not p['family'].startswith('D-582') and len(palette):
                    ref_nonink = reference['labs'][:, 0] >= 25
                    ref_dist = np.linalg.norm(reference['labs'][:, None]-palette[None], axis=2).min(axis=1)
                    outside = reference['weights'][ref_nonink & (ref_dist > 15)].sum() / max(reference['weights'][ref_nonink].sum(), 1e-9)
                    allowed = min(.5, float(outside)+.15)
                    crop = rgb[c['top_y']-y0:c['bottom_y']-y0]
                    pixels = lab(crop).reshape(crop.shape)
                    nonink = np.max(crop, axis=2) >= 60
                    distance = np.linalg.norm(pixels[:, :, None]-palette[None, None], axis=3).min(axis=2)
                    for a in range(0, len(crop)-3, 2):
                        mask = nonink[a:a+4]
                        if mask.sum() >= 8 and np.mean(distance[a:a+4][mask] > 15) > allowed:
                            clean = False; break
                if clean:
                    stats[pid]['eligible'] += 1
                    result[pid].append(c)
                else: stats[pid]['clean_rejected'] += 1
        finally: log.close()
    print(f'Clean-bed refinement: {sum(map(len, result.values()))} retained possibilities', flush=True)
    return result


def mine(catalog, families=None, dry_run=False, no_jev=False, max_calls=3000, workers=8, resume_scan=False):
    selected = [p for p in catalog['patterns'] if not families or any(f.lower() in p['family'].lower() for f in families)]
    if not selected: raise ValueError('No matching families')
    sources = source_intervals()
    with (ANALYSIS / 'volume_budget.csv').open(encoding='utf-8-sig', newline='') as f:
        budgets = {r['family']: r for r in csv.DictReader(f)}
    bylog = defaultdict(list)
    pools = defaultdict(list)
    stats = defaultdict(Counter)
    print('Calibrating fill colours from confirmed intervals...', flush=True)
    calibration = calibrate(selected, sources)
    print('Colour calibration complete; scanning all compatible logs.', flush=True)
    twins = {pid: {r['pid'] for r in rows} for pid, rows in read_json(DATA / 'lookalikes.json', {}).items()}
    # Search every log of the family, regardless of its exact legend signature.
    # Also include one-offs with the same normalized legend vocabulary.
    all_logs = sorted({name for p in catalog['patterns'] for name in p['logs']})
    log_families = {name: p['family'] for p in catalog['patterns'] for name in p['logs']}
    log_labels = {name: {norm(q['label']) for q in catalog['patterns'] if name in q['logs']} for name in all_logs}
    for p in selected:
        labels = {norm(q['label']) for q in catalog['patterns'] if q['legend_type'] == p['legend_type']}
        for name in all_logs:
            if log_families[name] == p['family'] or (log_families[name] == 'One-off layout' and log_labels[name] == labels):
                bylog[name].append(p)
    checkpoint = CACHE/'mining_scan.json'
    identity = scan_identity(selected, sources, budgets) if resume_scan else None
    saved = read_json(checkpoint, {}) if resume_scan else {}
    restored = saved.get('identity') == identity and identity is not None and checkpoint.with_suffix('.npz').exists()
    embed = Embedder()
    visited, errors = [], []
    if restored:
        with np.load(checkpoint.with_suffix('.npz'), allow_pickle=False) as archive:
            vectors = iter(archive['vectors'])
            pools = defaultdict(list, {pid: [dict(c, _embedding=next(vectors)) for c in rows]
                                       for pid, rows in saved['pools'].items()})
        stats = defaultdict(Counter, {pid: Counter(counts) for pid, counts in saved['stats'].items()})
        visited, errors = saved['visited'], saved['errors']
        bylog = {}
        print(f'Resuming validated scan of {len(visited)} logs.', flush=True)
    try:
        for name, pats in sorted(bylog.items()):
            log = None
            try:
                print(f'Scanning {name[:65]} ({len(pats)} patterns)', flush=True)
                log = io.open_log(name)
                x0, y0, x1, y1 = map(int, log.boxes['interpreted_lithology_column'])
                rgb = io.read_rgb(log, 'interpreted_lithology_column')
                intervals = detected_intervals(rgb, y0)
                # Known truth contacts also constrain unlabeled search windows.
                cuts = sorted({y for r in sources[name] if r['source'] in ('gold', 'consensus')
                               for y in (r['top_y'], r['bottom_y']) if y0 < y < y1})
                intervals = [(a, b) for top, bottom in intervals
                             for a, b in zip([top]+[y for y in cuts if top < y < bottom],
                                             [y for y in cuts if top < y < bottom]+[bottom])]
                budget = budgets[pats[0]['family']]
                stride = max(4, round(float(budget['bed_px']) / 2))
                pattern_info = {}
                for p in pats:
                    e = next((s for s in p['swatches'] if s['log'] == name), p['swatches'][0])
                    with Image.open(local_file(e.get('native_file', e['file']))) as image:
                        swatch = image.convert('RGB')
                    period = repeat_height(np.asarray(swatch)) if e.get('native_file') else 0
                    period = period or round(float(budget['period_med']))
                    height = max(48, 3 * period)
                    pattern_info[p['pid']] = (period, height, swatch, profile(np.asarray(swatch)))
                family_binary = {fam: all(info[3]['binary'] for pid, info in pattern_info.items()
                                         if next(p for p in pats if p['pid'] == pid)['family'] == fam)
                                 for fam in {p['family'] for p in pats}}
                for p in pats:
                    pattern_info[p['pid']][3]['binary'] = family_binary[p['family']]
                matcher = PaletteMatcher([pattern_info[p['pid']][3] for p in pats],
                                         [calibration[p['family']]['shift_lab'] for p in pats])
                # A crop shared by multiple candidate patterns is embedded once.
                plans = defaultdict(dict)
                for p in pats:
                    period, height, swatch, _ = pattern_info[p['pid']]
                    for top, bottom in intervals:
                        for a, b in crop_windows(top, bottom, height, stride):
                            plans[a, b][p['pid']] = dict(source='embedding', interval_top=top, interval_bottom=bottom)
                    for r in sources[name]:
                        if norm(r['lithology']) != norm(p['label']): continue
                        top, bottom = max(y0, r['top_y']), min(y1, r['bottom_y'])
                        for a, b in crop_windows(top, bottom, height, stride):
                            old = plans[a, b].get(p['pid'])
                            if old is None or TRUST[r['source']] > TRUST[old['source']]:
                                plans[a, b][p['pid']] = dict(source=r['source'], interval_top=top, interval_bottom=bottom)
                good, colour_scores, shares = [], {}, {}
                pat_by_id = {p['pid']: p for p in pats}
                # Gate before embedding; keep rival colour scores even when a
                # rival fails its gate, so contrastive competition stays honest.
                for a, b in sorted(plans):
                    for pid in plans[a, b]: stats[pid]['stretches'] += 1
                    if not quality(rgb[a-y0:b-y0]): continue
                    crop_profile = profile(rgb[a-y0:b-y0])
                    scores, passed = {}, {}
                    matches, fill_shares, similarities = matcher(crop_profile)
                    for p, ok, share, score in zip(pats, matches, fill_shares, similarities):
                        pid = p['pid']
                        scores[pid] = float(score)
                        if pid in plans[a, b]:
                            stats[pid]['clean'] += 1
                            if ok:
                                stats[pid]['colour'] += 1
                                passed[pid] = plans[a, b][pid]
                                shares[a, b, pid] = float(share)
                    plans[a, b] = passed
                    if passed:
                        good.append((a, b)); colour_scores[a, b] = scores
                log_pools = defaultdict(list)
                refs = {p['pid']: embed([np.asarray(tile(pattern_info[p['pid']][2], x1-x0, pattern_info[p['pid']][1]))])[0] for p in pats}
                for start in range(0, len(good), 96):
                    batch = good[start:start+96]
                    vectors = embed([rgb[a-y0:b-y0] for a, b in batch])
                    for (a, b), v in zip(batch, vectors):
                        scores = {pid: .5 * float(v @ ref) + .5 * colour_scores[a, b][pid] for pid, ref in refs.items()}
                        for pid, plan in plans[a, b].items():
                            p = pat_by_id[pid]
                            rivals = {q['pid']: scores[q['pid']] for q in pats if q['legend_type'] == p['legend_type']}
                            margin = contrastive(rivals, pid, twins.get(pid, ()))
                            if margin <= 0: continue
                            stats[pid]['margin'] += 1
                            c = dict(plan, log=name, top_y=a, bottom_y=b, col_w=x1-x0, h=b-a,
                                     repeat_height=pattern_info[pid][0], _embedding=v,
                                     similarity=scores[pid], colour_share=shares[a, b, pid],
                                     margin=margin, twin=bool(twins.get(pid)))
                            c['score'] = candidate_score(c)
                            log_pools[pid].append(c)
                for p in pats:
                    candidates = sorted(log_pools[p['pid']], key=lambda c: -c['score'])
                    # Bound memory after full-log sampling. Reserve known-good
                    # seeds and diverse alternatives before cross-log selection.
                    retained = []
                    for c in candidates:
                        if any(float(c['_embedding'] @ x['_embedding']) > .97 for x in retained): continue
                        retained.append(c)
                        if len(retained) >= 80: break
                    pools[p['pid']].extend(retained)
                visited.append(name)
                print(f'[{len(visited)}/{len(bylog)}] {name[:65]}: {len(good)} clean stretches', flush=True)
            except Exception as exc:
                errors.append(dict(log=name, error=type(exc).__name__))
                print(f'ERROR {name[:65]}: {type(exc).__name__}', flush=True)
            finally:
                if log: log.close()
    finally: embed.close()
    if resume_scan and not restored and not errors:
        save_scan(checkpoint, identity, pools, stats, visited, errors)
        print('Saved completed scan checkpoint before export/verification.', flush=True)
    pools = refine_pools(pools, selected, stats, calibration)
    maxima = defaultdict(float)
    for p in catalog['patterns']: maxima[p['family']] = max(maxima[p['family']], p['share'])
    shortlisted = {}
    for p in selected:
        pool = pools[p['pid']]
        if pool:
            vectors = np.stack([c['_embedding'] for c in pool])
            p['variety'] = float(np.clip(1 - np.linalg.norm(vectors.mean(axis=0))**2, 0, 1))
        else: p['variety'] = 0
        p['n_target'] = target(p['share'], maxima[p['family']], p['variety'])
        p['raw_candidate_count'] = len(pool)
        p['candidate_logs'] = len({c['log'] for c in pool})
        shortlist = diverse(pool, min(40, p['n_target'] + 10))
        p['preverify_count'] = len(shortlist)
        p['status'] = 'mined' if shortlist else ('diversity_cap_infeasible' if pool else 'no_candidates')
        shortlisted[p['pid']] = shortlist
    # Export only native crops, and only for candidates reaching verification.
    bylog = defaultdict(list)
    for p in selected:
        for c in shortlisted[p['pid']]: bylog[c['log']].append(c)
    for name, candidates in sorted(bylog.items()):
        log = io.open_log(name)
        try:
            for c in candidates:
                crop = io.read_rgb(log, 'interpreted_lithology_column', c['top_y'], c['bottom_y'])
                key = hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest()
                path = DATA / 'crops' / (key + '.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists(): Image.fromarray(crop).save(path)
                c.update(file=relative(path), content_hash=key)
        finally: log.close()
    tasks = [(p, c) for p in selected for c in shortlisted[p['pid']]]
    verifier = Verifier(max_calls=max_calls, workers=workers)
    specs = {verifier.request_spec(p, c, catalog['patterns'])[3] for p, c in tasks}
    uncached = sum(read_json(verifier.cache / (key+'.json'), {}).get('status') != 'ok' for key in specs)
    if not dry_run and not no_jev: verifier.run(tasks, catalog['patterns'])
    else:
        for p, c in tasks:
            c['verification'] = verifier.verify(p, c, catalog['patterns'], enabled=False)
            if no_jev and c['verification']['status'] == 'dry_run': c['verification']['status'] = 'disabled'
    output = read_json(DATA / 'candidates.json', {})
    for p in selected:
        for c in shortlisted[p['pid']]: c['score'] = candidate_score(c)
        final = diverse(shortlisted[p['pid']], min(30, p['n_target']))
        p['candidate_count'] = len(final)
        output[p['pid']] = [clean_candidate(c) for c in final]
    catalog['no_candidates'] = [p['pid'] for p in catalog['patterns'] if not p['candidate_count']]
    catalog['no_raw_candidates'] = [p['pid'] for p in catalog['patterns'] if p.get('raw_candidate_count') == 0]
    # Keep discovery coordinates for sparse legend types even when the strict
    # diversity constraint cannot produce a nonempty review set.
    raw = read_json(DATA / 'pools.json', {})
    raw.update({p['pid']: [clean_candidate(c) for c in pools[p['pid']]] for p in selected})
    write_json(DATA / 'pools.json', raw)
    pattern_report = {}
    for p in selected:
        counts = stats[p['pid']]
        reasons = []
        if p['candidate_count'] < p['n_target']:
            if counts['colour'] < p['n_target']: reasons.append('no colour match')
            if counts['margin'] < counts['colour']: reasons.append('margin')
            if counts['margin_floor_rejected'] and 'margin' not in reasons: reasons.append('margin')
            if counts['margin'] >= p['n_target'] or counts['clean'] < p['n_target']: reasons.append('too few clean stretches (including diversity/deduplication)')
            if (counts['clean_rejected'] or counts['reference_rejected']) and not any(r.startswith('too few clean') for r in reasons):
                reasons.append('too few clean stretches (mixed bands or invalid native swatch)')
            if p['candidate_count'] == 30 and p['n_target'] > 30: reasons = ['review cap (30)']
        pattern_report[p['pid']] = dict({key: counts[key] for key in ('stretches','clean','colour','margin','margin_floor_rejected','after_margin','clean_rejected','reference_rejected','eligible')},
                                         target=p['n_target'], candidates=p['candidate_count'], shortfall=reasons)
    family_report = {}
    for fam in sorted({p['family'] for p in selected}):
        ps = [p for p in selected if p['family'] == fam]
        clean = sum(stats[p['pid']]['clean'] for p in ps)
        passed = sum(stats[p['pid']]['colour'] for p in ps)
        family_report[fam] = dict(patterns=len(ps), candidates=sum(p['candidate_count'] for p in ps),
                                zero_candidates=sum(not p['candidate_count'] for p in ps),
                                gate_tested=clean, gate_passed=passed, gate_pass_rate=passed / max(clean, 1))
        print(f'{fam}: colour gate {passed}/{clean} ({passed/max(clean,1):.1%}); {family_report[fam]["candidates"]} candidates', flush=True)
    report = dict(dry_run=dry_run, no_jev=no_jev, scan_resumed=restored, requested_logs=len(visited)+len(errors), logs=len(visited), errors=errors,
                  patterns=len(selected), preverify_candidates=len(tasks), unique_verifications=len(specs),
                  uncached_verifications=uncached, dry_run_jev_calls=min(max_calls, uncached), actual_jev_calls=verifier.calls,
                  sheet_candidates=sum(len(output[p['pid']]) for p in selected),
                  circuit_open=verifier.stopped, consecutive_failures=verifier.consecutive_failures,
                  verification_failures=verifier.failures, calibration=calibration,
                  colour_thresholds=dict(deltaE=15, palette_min_share=.08, main_share_ratio=.5, minimum_margin=.015,
                                         clean_band_rows=4, band_outside_slack=.15),
                  pattern_counts=pattern_report, families=family_report)
    write_json(DATA / 'candidates.json', output)
    write_json(DATA / 'catalog.json', catalog)
    write_json(DATA / 'mining_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+')
    parser.add_argument('--max-calls', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true', help='Run real offline mining; plan but do not make JEV calls')
    parser.add_argument('--no-jev', action='store_true')
    parser.add_argument('--resume-scan', action='store_true', help='Reuse a validated completed scan; save one when absent')
    args = parser.parse_args()
    if args.max_calls < 0 or args.workers < 1: parser.error('Require max-calls >= 0 and workers >= 1')
    catalog = read_json(DATA / 'catalog.json')
    if not catalog: parser.error('Run synth/patterns.py first')
    report = mine(catalog, **vars(args))
    print(json.dumps(report, indent=2))
    if report['errors']: sys.exit(1)


if __name__ == '__main__': main()
