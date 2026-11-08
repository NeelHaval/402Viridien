"""Read-only legend atlas. All writes are confined to this directory."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from urllib.parse import quote, unquote, urlparse
from urllib.request import urlopen

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import model
from genlab.swatch import remove_frame

DATA = ROOT / 'data/genlab'
CACHE = HERE / 'cache'
SHOTS = HERE / 'shots'
MATCH = .98
DICE = .95
COLOUR = 8
NOVELTY = .01
VERSION = 4
STATUSES = ['Done', 'Partly done', 'In progress', 'To do', 'Held out', 'Skipped']
URLS = {'board': 'http://127.0.0.1:8782/board.json',
        'lab': 'http://127.0.0.1:8781/state.json',
        'legend': 'http://127.0.0.1:8771/api/legend?v=2',
        'mix': 'http://127.0.0.1:8771/api/mix', 'cat': 'http://127.0.0.1:8771/api/cat'}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {} if default is None else default


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    for n in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(.05 * (n + 1))
    raise OSError('Could not replace ' + str(path))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:24]


def stamp(path):
    try:
        s = path.stat()
        return [s.st_mtime_ns, s.st_size]
    except OSError:
        return None


def norm(s):
    return ''.join(str(s or '').lower().split())


def image_url(path):
    p = Path(path).resolve()
    p.relative_to(DATA.resolve())
    return '/img/' + quote(p.relative_to(DATA).as_posix(), safe='/') + '?v=' + str(p.stat().st_mtime_ns)


def source_file(did, name):
    if not name:
        return None
    p = Path(name)
    choices = [p] if p.is_absolute() else [DATA / p, DATA / 'proposals' / did / p]
    for p in choices:
        p = p.resolve()
        if p.is_relative_to(DATA.resolve()) and p.is_file() and p.suffix.lower() == '.png':
            return p
    return None


def generatable(member):
    return member.get('training_allowed') is True and member.get('state') not in ('Not lithology', 'Skipped (not a rock fill)')


def symbol_status(members, versions):
    train = [m for m in members if generatable(m)]
    if not any(m.get('training_allowed') is True for m in members):
        return 'Held out'
    if not train:
        return 'Skipped'
    in_lab = {v['key'] for v in versions if not v.get('waiting')}
    complete = lambda m: m['key'] in in_lab or m.get('state') in ('In the lab', 'Approved, waiting for the lab')
    has_lab = bool(in_lab)
    if has_lab and all(complete(m) for m in train):
        return 'Done'
    if any(m.get('live') or m.get('state') in ('Being made', 'Being reviewed') for m in train):
        return 'In progress'
    if has_lab:
        return 'Partly done'
    return 'To do'


def next_action(members, versions, status, real):
    train = [m for m in members if generatable(m)]
    in_lab = {v['key'] for v in versions if not v.get('waiting')}
    missing = [m for m in train if m['key'] not in in_lab and m['state'] != 'Approved, waiting for the lab']
    if status == 'Held out':
        return 'Test-well only: held out on purpose; ' + ('a generated version from an identical training symbol exists.' if versions else 'display only, never generate from these members.')
    if status == 'Skipped':
        return 'Not lithology or not a rock fill: skipped.'
    if status == 'Done':
        return f"Done: in the lab for {sum(m['key'] in in_lab for m in train)} of {len(train)} training entries; the rest are approved and waiting." if any(m['key'] not in in_lab for m in train) else f'Done: all {len(train)} training entries are in the lab.'
    running = [m for m in train if m.get('live') or m['state'] in ('Being made', 'Being reviewed')]
    if running:
        return 'Retry running in the sweep (' + ', '.join(sorted({m.get('job') or m['short_doc'] for m in running})) + ').'
    failed = [m for m in missing if m['state'] == 'Failed']
    rejected = [m for m in missing if m['state'] in ('Rejected', 'Unsure')]
    owner_rejects = [m for m in train if m['key'] not in in_lab and m.get('owner_verdict', {}).get('verdict') == 'reject']
    prefix = f"In the lab for {sum(m['key'] in in_lab for m in train)} of {len(train)} training entries; " if in_lab else ''
    if failed:
        return prefix + 'failed: ' + '; '.join(f"{m['short_doc']} #{m['legend_id']}: {m.get('note') or 'no reason supplied'}" for m in failed) + '. Retry these entries.'
    if owner_rejects:
        return prefix + 'Owner rejection blocks lab import: ' + '; '.join(f"{m['doc_id'][:8]} #{m['legend_id']}: {m['owner_verdict'].get('note') or 'rejected by owner'}" for m in owner_rejects) + ' Review the owner rejection before retrying or importing.'
    if rejected:
        return prefix + 'rejected or unsure: ' + '; '.join(f"{m['short_doc']} #{m['legend_id']}: {m.get('note') or m['state']}" for m in rejected) + '. Review before retrying.'
    if versions and not in_lab:
        return 'Made proposals are waiting for current-round approval or lab import; review their verdicts.'
    if not real:
        return prefix + 'Needs real examples: label 2 boxes of this in the Pattern Labeller.'
    return prefix + f"Not started: make and review the {len(missing)} remaining training entries."


def signature(rgb):
    from scipy.ndimage import binary_dilation
    a = np.asarray(Image.fromarray(rgb).resize((64, 40), Image.Resampling.NEAREST)).astype(np.float32)
    light = rgb[rgb.max(2) >= 80]
    bg = np.median(light, axis=0) if len(light) else np.zeros(3)
    ink = a.max(2) < 128
    return dict(a=a, fft=np.fft.rfft2(a, axes=(0, 1)), bg=bg, ink=ink,
                expanded=binary_dilation(ink), amount=float(ink.mean()))


def pixel_similarity(a, b):
    """Best cyclic RGB least-squares phase, then exact colour and ink predicates."""
    colour = float(np.linalg.norm(a['bg'] - b['bg']))
    if colour > COLOUR:
        return 0., 0., colour, 1.
    corr = np.fft.irfft2((np.conj(a['fft']) * b['fft']).sum(2), s=(40, 64))
    dy, dx = np.unravel_index(corr.argmax(), corr.shape)
    shifted = np.roll(b['a'], (-int(dy), -int(dx)), axis=(0, 1))
    hit = float((np.linalg.norm(a['a'] - shifted, axis=2) < 35).mean())
    bi = shifted.max(2) < 128
    total = int(a['ink'].sum() + bi.sum())
    dice = 2 * int((a['ink'] & bi).sum()) / total if total else 1.
    be = np.roll(b['expanded'], (-int(dy), -int(dx)), axis=(0, 1))
    novelty = max(float((a['ink'] & ~be).sum()) / max(1, int(a['ink'].sum())),
                  float((bi & ~a['expanded']).sum()) / max(1, int(bi.sum())))
    return hit, dice, colour, novelty


def deduplicate(rows, owner=None, pair_cache=None):
    """Representative grouping avoids transitive chains of look-alikes."""
    owner = owner or {}
    pair_cache = pair_cache if pair_cache is not None else {}
    sigs = {r['key']: signature(r['rgb']) for r in rows if r.get('rgb') is not None}
    groups, pairs = [], []
    # Owner groups nominate first comparisons, but never decide a merge.
    for row in sorted(rows, key=lambda r: (str(owner.get(r['key'], '~')), r['key'])):
        if row['key'] not in sigs:
            groups.append([row]); continue
        sig = sigs[row['key']]
        candidates = sorted((g for g in groups if g[0]['key'] in sigs), key=lambda g: owner.get(g[0]['key']) != owner.get(row['key']))
        for group in candidates:
            rep = group[0]
            other = sigs[rep['key']]
            if np.linalg.norm(sig['bg'] - other['bg']) > COLOUR or abs(sig['amount'] - other['amount']) > .055:
                continue
            key = ':'.join(sorted([row.get('swatch_hash', row['key']), rep.get('swatch_hash', rep['key'])]))
            if key not in pair_cache or len(pair_cache[key]) != 4:
                pair_cache[key] = list(pixel_similarity(sig, other))
            hit, dice, colour, novelty = pair_cache[key]
            accepted = hit >= MATCH and dice >= DICE and colour <= COLOUR and novelty <= NOVELTY
            if hit >= .82 and dice >= .65:
                pairs.append(dict(a=rep['key'], b=row['key'], match=hit, dice=dice, colour=colour, novelty=novelty, accepted=accepted))
            if accepted:
                group.append(row); break
        else:
            groups.append([row])
    return groups, pairs


class CropCache:
    """Windowed TIFF segments only, including CCITT via an isolated strip TIFF."""
    def __init__(self):
        self.docs = {}
        self.pending = defaultdict(list)

    def doc(self, did):
        if did not in self.docs:
            p = ROOT / 'data/reviews' / (did + '.json')
            if not p.exists():
                p = ROOT / 'data/seeds' / (did + '.json')
            self.docs[did] = read(p)
        return self.docs[did]

    def crop(self, did, box, swatch=False):
        doc = self.doc(did)
        box = list(map(int, box))
        original = list(box)
        if not swatch:
            box[3] = min(box[3], box[1] + 160)
            if box[2] - box[0] > 180:
                mid = (box[0] + box[2]) // 2
                box[0], box[2] = mid - 90, mid + 90
        model.Store.validate_box(doc, box)
        source = model.DATASET / 'images' / doc['image_name']
        key = digest([VERSION, did, box, stamp(source), swatch])
        out = CACHE / ('swatch-' if swatch else 'real-')
        out = out.with_name(out.name + key + '.png')
        if not out.exists():
            cached = DATA / 'crops' / did / ('%d_%d_%d_%d.png' % tuple(original))
            # Existing lab crops are read-only; freshness includes the TIFF mtime.
            if cached.exists() and cached.stat().st_mtime_ns >= source.stat().st_mtime_ns:
                with Image.open(cached) as im:
                    rgb = np.array(im.convert('RGB').crop((box[0]-original[0], box[1]-original[1], box[2]-original[0], box[3]-original[1])))
            elif not swatch:
                self.pending[did].append((source, box, out))
                return dict(url='/img/atlas/cache/' + out.name, size=[box[2]-box[0], box[3]-box[1]], doc_id=did, box=box, original_box=original)
            else:
                rgb = self.window(source, box)
            if swatch:
                rgb, _ = remove_frame(rgb)
            Image.fromarray(rgb).save(out)
        with Image.open(out) as im:
            size = list(im.size)
        return dict(url=image_url(out), size=size, doc_id=did, box=box, original_box=original)

    @staticmethod
    def window(path, box):
        return CropCache.scan_windows(path, [box])[0]

    def flush(self):
        for did, items in self.pending.items():
            unique = {str(out): (source, box, out) for source, box, out in items}
            values = list(unique.values())
            arrays = self.scan_windows(values[0][0], [v[1] for v in values])
            for (_, _, out), rgb in zip(values, arrays):
                Image.fromarray(rgb).save(out)
        self.pending.clear()

    @staticmethod
    def scan_windows(path, boxes):
        """libtiff scanline streaming: a giant single strip is never an RGB page."""
        import ctypes
        import tifffile
        dll = Path(sys.prefix) / 'Library/bin/tiff.dll'
        lib = ctypes.CDLL(str(dll))
        lib.TIFFOpen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.TIFFOpen.restype = ctypes.c_void_p
        lib.TIFFScanlineSize64.argtypes = [ctypes.c_void_p]
        lib.TIFFScanlineSize64.restype = ctypes.c_uint64
        lib.TIFFReadScanline.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16]
        lib.TIFFReadScanline.restype = ctypes.c_int
        lib.TIFFClose.argtypes = [ctypes.c_void_p]
        outputs = [np.empty((b[3]-b[1], b[2]-b[0], 3), np.uint8) for b in boxes]
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if page.is_tiled or page.planarconfig != 1:
                raise ValueError('Unsupported TIFF window layout; no whole-page fallback')
            handle = lib.TIFFOpen(str(path).encode(), b'r')
            if not handle:
                raise OSError('Cannot open windowed TIFF')
            try:
                size = lib.TIFFScanlineSize64(handle)
                if size > 8*1024*1024:
                    raise ValueError('Scanline exceeds memory budget')
                buf = ctypes.create_string_buffer(size)
                ranges = defaultdict(list)
                for i, b in enumerate(boxes):
                    for strip in range(b[1]//page.rowsperstrip, (b[3]-1)//page.rowsperstrip+1):
                        ranges[strip].append(i)
                for strip, indexes in sorted(ranges.items()):
                    end = max(min(boxes[i][3], (strip+1)*page.rowsperstrip) for i in indexes)
                    for y in range(strip*page.rowsperstrip, end):
                        if lib.TIFFReadScanline(handle, buf, y, 0) < 0:
                            raise OSError('TIFF scanline decode failed')
                        active = [i for i in indexes if boxes[i][1] <= y < boxes[i][3]]
                        if not active:
                            continue
                        raw = np.frombuffer(buf, np.uint8)
                        if page.bitspersample == 1:
                            row = np.unpackbits(raw)[:page.imagewidth]*255
                            if page.photometric == 0:
                                row = 255-row
                            row = np.repeat(row[:, None], 3, axis=1)
                        elif page.photometric == 3:
                            row = (page.colormap >> 8).astype(np.uint8).T[raw[:page.imagewidth]]
                        else:
                            row = raw[:page.imagewidth*page.samplesperpixel].reshape(page.imagewidth, page.samplesperpixel)
                            if page.samplesperpixel == 1:
                                row = np.repeat(row, 3, axis=1)
                            row = row[:, :3]
                        for i in active:
                            b = boxes[i]
                            outputs[i][y-b[1]] = row[b[0]:b[2]]
            finally:
                lib.TIFFClose(handle)
        return outputs


class Atlas:
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        SHOTS.mkdir(parents=True, exist_ok=True)
        self.snapshot = read(CACHE / 'atlas.json')
        self.sources = {}
        self.health = read(CACHE / 'health.json')
        self.lock = threading.Lock()
        self.rebuild_lock = threading.Lock()
        self.crops = CropCache()
        self.doc_cache = read(CACHE / 'documents.json')
        self.pair_cache = read(CACHE / 'pairs.json')
        self.errors = []

    def fetch(self, name):
        try:
            with urlopen(URLS[name], timeout=7) as response:
                value = json.load(response)
            if not isinstance(value, dict) or (name in ('board', 'lab') and not isinstance(value.get('entries' if name == 'board' else 'cards'), list)):
                raise ValueError('Unexpected source format')
            self.sources[name] = value
            save(CACHE / ('source-' + name + '.json'), value)
            self.health[name] = dict(online=True, last_good=time.time(), offline_since=None)
        except Exception as exc:
            old = self.health.get(name, {})
            self.health[name] = dict(online=False, last_good=old.get('last_good'), offline_since=old.get('offline_since') or time.time(), error=str(exc))
            if name not in self.sources:
                self.sources[name] = read(CACHE / ('source-' + name + '.json'))
        return self.sources[name]

    def proposal_versions(self, member, card):
        did, lid = member['doc_id'], str(member['legend_id'])
        kind = 'unlisted' if lid.startswith('u') else 'legend'
        number = lid[1:] if kind == 'unlisted' else lid
        path = DATA / 'proposals' / did / f'{kind}-{number}.json'
        p = read(path)
        versions = []
        if card:
            v = {k: card.get(k) for k in ('key', 'doc_id', 'legend_id', 'label', 'source', 'tile_source', 'tile_scores', 'realism', 'verdict', 'kind', 'proposal_round')}
            v['waiting'] = False
            for field in ('tile', 'repeat_preview'):
                f = source_file(did, card.get(field))
                v[field] = image_url(f) if f else None
            v['samples'] = [image_url(f) for s in card.get('samples', []) if (f := source_file(did, s))]
            versions.append(v)
        tile = source_file(did, p.get('tile_png'))
        if p and tile:
            rnd = p.get('round')
            verdict = read(path.parent / f'{kind}-review-r{rnd}.json').get(lid, {})
            # A proposal remains visible alongside the lab version unless that
            # exact current round has already entered the lab as agent-made.
            imported = card and card.get('source') in ('agent-made', 'unlisted') and card.get('proposal_round') == rnd
            if not imported:
                versions.append(dict(key=member['key'], doc_id=did, legend_id=member['legend_id'], label=member['label'],
                                     source='unlisted' if kind == 'unlisted' else 'agent-made', tile_source=p.get('method') or 'agent proposal',
                                     tile=image_url(tile), repeat_preview=None, samples=[], realism=p.get('realism'),
                                     verdict=verdict, proposal_round=rnd, waiting=True, proposal_status=p.get('status'), kind=p.get('method')))
        return versions

    def compile_member(self, entry, card, candidates, owners, guesses, jobs):
        m = dict(entry)
        m['owner_verdict'] = read(DATA / 'verdicts' / (m['doc_id'] + '.json')).get(str(m['legend_id']), {})
        m['job'] = next((j['job'] for j in jobs if entry['doc_id'][:10] in j.get('job', '') and j.get('state') == 'running'), '')
        versions = self.proposal_versions(m, card)
        for v in versions:
            v['owner_verdict'] = m['owner_verdict']
        c = candidates.get(str(m['legend_id']), {})
        swatch = None
        if c.get('swatch_box'):
            try:
                swatch = self.crops.crop(m['doc_id'], c['swatch_box'], True)
            except Exception as exc:
                self.errors.append(f"{m['key']} swatch: {exc}")
        examples = []
        owners = sorted(owners, key=lambda r: (bool(r.get('quality_issues')), r.get('quality') != 'accepted'))
        real_inputs = [(m['doc_id'], r['box_xyxy'], 'owner', r.get('split'), r.get('quality_issues', [])) for r in owners]
        if card and card.get('evidence_kind') != 'swatch' and card.get('source') != 'swatch':
            real_did = card.get('real_doc_id') or m['doc_id']
            # Some lab cards store the legend as their fallback 'real'.
            doc = self.crops.doc(real_did)
            col = doc.get('boxes', {}).get('interpreted_lithology_column', [])
            for b in card.get('real', []):
                if len(col) == 4 and b[0] < col[2] and b[2] > col[0] and b[1] >= col[1]:
                    real_inputs.append((real_did, b, 'lab real', None, []))
        if not real_inputs:
            doc = self.crops.doc(m['doc_id'])
            col = doc.get('boxes', {}).get('interpreted_lithology_column')
            if col:
                for g in guesses[:6]:
                    a, b = max(col[1], int(g['top_y'])), min(col[3], int(g['bottom_y']))
                    if b > a:
                        real_inputs.append((m['doc_id'], [col[0], a, col[2], b], 'machine guess', None, []))
        seen = set()
        for did, box, label, split, quality_issues in real_inputs:
            key = (did, tuple(box))
            if key in seen:
                continue
            seen.add(key)
            try:
                im = self.crops.crop(did, box)
                im.update(label=label, split=split, quality_issues=quality_issues)
                examples.append(im)
            except Exception as exc:
                self.errors.append(f"{m['key']} real: {exc}")
            if len(examples) >= 6:
                break
        return dict(member=m, swatch=swatch, examples=examples, versions=versions)

    def rebuild(self):
        if not self.rebuild_lock.acquire(False):
            return
        started = time.perf_counter()
        try:
            self.errors = []
            with ThreadPoolExecutor(max_workers=3) as pool:
                board, lab, legend, mix, cat = list(pool.map(self.fetch, URLS))
            save(CACHE / 'health.json', self.health)
            if not board.get('entries'):
                return
            entries = board['entries'] + board.get('unlisted', [])
            cards = {c['key']: c for c in lab.get('cards', [])}
            owners = defaultdict(list)
            normalized = read(ROOT / 'data/normalized/current.json')
            for record in normalized.get('records', []):
                if record.get('accepted', True) and record.get('quality') != 'rejected':
                    for target in record.get('targets', []):
                        owners[f"{record['document_id']}:{target['legend_id']}"].append(record)
            rows, guesses, doc_rows = Counter(), defaultdict(list), Counter()
            names = {d['name']: d['doc_id'] for d in board.get('documents', [])}
            label_keys = defaultdict(list)
            for e in board['entries']:
                label_keys[(e['doc_id'], norm(e['label']))].append(e['key'])
            prefill = ROOT.parent / 'bakeoff/labeltool/data/prefill.jsonl'
            total_rows = 0
            if prefill.exists():
                with prefill.open(encoding='utf-8') as stream:
                    for line in stream:
                        try:
                            r = json.loads(line)
                            amount = max(0, int(r['bottom_y']) - int(r['top_y']))
                            total_rows += amount
                            if r.get('log') in names:
                                doc_rows[names[r['log']]] += amount
                            for key in label_keys.get((names.get(r.get('log')), norm(r.get('suggestion'))), []):
                                rows[key] += amount / len(label_keys[(names[r['log']], norm(r['suggestion']))])
                                if len(guesses[key]) < 6:
                                    guesses[key].append(r)
                        except (ValueError, KeyError):
                            continue
            categories = {}
            category_order = {}
            for i, c in enumerate(mix.get('categories', [])):
                category_order[c['cat']] = i
                for label in c.get('labels', []):
                    categories[norm(label)] = c['cat']
            for c in cat.get('categories', []):
                for label in c.get('labels', []):
                    categories.setdefault(norm(label['label']), c['name'])
            owner_groups = {}
            for g in legend.get('groups', []):
                for item in g.get('items', []):
                    for name, lid, _ in item.get('members', []):
                        if name in names:
                            owner_groups[f'{names[name]}:{lid}'] = item.get('file')
            by_doc = defaultdict(list)
            for e in entries:
                by_doc[e['doc_id']].append(e)
            compiled, changed = [], []
            for did, doc_entries in by_doc.items():
                cp = DATA / 'candidates' / (did + '.json')
                proposals = sorted((DATA / 'proposals' / did).glob('*'))
                doc_cards = [cards[e['key']] for e in doc_entries if e['key'] in cards]
                files = [source_file(did, c.get(field)) for c in doc_cards for field in ('tile', 'repeat_preview')]
                doc = self.crops.doc(did)
                source = model.DATASET / 'images' / doc.get('image_name', '')
                fp = digest([VERSION, 3, doc_entries, doc_cards, [[p.name, stamp(p)] for p in proposals if p.is_file()],
                             stamp(DATA / 'verdicts' / (did + '.json')),
                             stamp(cp), stamp(source), [owners[e['key']] for e in doc_entries], [guesses[e['key']] for e in doc_entries],
                             [[str(f), stamp(f)] for f in files if f]])
                old = self.doc_cache.get(did, {})
                if old.get('fingerprint') != fp:
                    candidates = {str(c['legend_id']): c for c in read(cp).get('entries', [])}
                    values = [self.compile_member(e, cards.get(e['key']), candidates, owners[e['key']], guesses[e['key']], board.get('live_jobs', [])) for e in doc_entries]
                    self.doc_cache[did] = dict(fingerprint=fp, rows=values)
                    changed.append(did)
                compiled.extend(self.doc_cache[did]['rows'])
            self.crops.flush()
            grouping_rows = []
            for c in compiled:
                e = c['member']
                rgb = None
                if c['swatch']:
                    path = DATA / unquote(urlparse(c['swatch']['url']).path.removeprefix('/img/'))
                    with Image.open(path) as im:
                        rgb = np.array(im.convert('RGB'))
                grouping_rows.append(dict(key=e['key'], rgb=rgb, compiled=c,
                                          swatch_hash=digest([c['swatch']['url'] if c['swatch'] else e['key']])))
            groups, pairs = deduplicate(grouping_rows, owner_groups, self.pair_cache)
            symbols = []
            for group in groups:
                members = [r['compiled']['member'] for r in group]
                versions = [v for r in group for v in r['compiled']['versions']]
                examples = [e for r in group for e in r['compiled']['examples']]
                # A symbol with any human evidence never displays machine guesses.
                if any(e['label'] == 'owner' for e in examples):
                    examples = [e for e in examples if e['label'] != 'machine guess']
                examples.sort(key=lambda e: ({'owner': 0, 'lab real': 1, 'machine guess': 2}[e['label']], bool(e.get('quality_issues'))))
                seen = set()
                real = []
                for e in examples:
                    if e['url'] not in seen:
                        real.append(e); seen.add(e['url'])
                    if len(real) == 6:
                        break
                names_in_group = sorted({m['label'] for m in members}, key=str.casefold)
                cats = list(dict.fromkeys(categories.get(norm(n), 'Unlisted' if str(members[0]['legend_id']).startswith('u') else 'Other') for n in names_in_group))
                status = symbol_status(members, versions)
                share = sum(rows[m['key']] for m in members) / max(1, total_rows)
                unlisted = str(members[0]['legend_id']).startswith('u')
                document_share = members[0].get('share_of_column_rows') if unlisted else None
                if document_share is not None:
                    share = float(document_share) * doc_rows[members[0]['doc_id']] / max(1, total_rows)
                swatch = group[0]['compiled']['swatch']
                accent = '#687078'
                if group[0]['rgb'] is not None:
                    bg = signature(group[0]['rgb'])['bg'].astype(int)
                    accent = '#%02x%02x%02x' % tuple(bg)
                symbol = dict(id='s-' + digest(sorted(m['key'] for m in members))[:12], names=names_in_group,
                              category=' / '.join(cats), categories=cats, row_share=share, estimated_rows=round(share*total_rows),
                              document_row_share=document_share,
                              share_basis='fleet document estimate weighted by document rows; overlaps legend guesses' if unlisted else 'saved machine guesses',
                              status=status, members=members, versions=versions, real=real, swatch=swatch, accent=accent,
                              logs=len({m['doc_id'] for m in members}), entries=len(members),
                              generatable=sum(generatable(m) for m in members),
                              sources=sorted({v.get('source') or 'unknown' for v in versions}),
                              kind='unlisted' if str(members[0]['legend_id']).startswith('u') else 'legend')
                symbol['todo'] = next_action(members, versions, status, real)
                symbol['todo_kind'] = 'failed' if any(m['state'] == 'Failed' for m in members) else 'rejected or unsure' if any(m['state'] in ('Unsure', 'Rejected') or m.get('owner_verdict', {}).get('verdict') in ('reject', 'unsure') for m in members) else 'not started'
                symbols.append(symbol)
            # Variants of the same printed name sit together, with the largest
            # share in each name block first. To-do has its own strict share sort.
            name_share = defaultdict(float)
            for s in symbols:
                name_share[norm(s['names'][0])] += s['row_share']
            symbols.sort(key=lambda s: (min(category_order.get(c, 999) for c in s['categories']), -name_share[norm(s['names'][0])], norm(s['names'][0]), -s['row_share']))
            counts = Counter(s['status'] for s in symbols)
            seconds = round(time.perf_counter() - started, 3)
            history = (self.snapshot.get('build_history', []) + [dict(at=time.time(), seconds=seconds, changed_docs=len(changed))])[-30:]
            result = dict(at=time.time(), symbols=symbols, documents=board.get('documents', []), health=self.health,
                          counts={k: counts[k] for k in STATUSES}, total_symbols=len(symbols),
                          legend_entries=len(board['entries']), unlisted_entries=len(board.get('unlisted', [])),
                          merged_entries=sum(max(0, s['entries']-1) for s in symbols),
                          coverage=lab.get('coverage', {}), live_jobs=board.get('live_jobs', []),
                          dedupe=dict(match=MATCH, ink_dice=DICE, colour_distance=COLOUR, rgb_distance=35, novelty=NOVELTY,
                                      common_scale=[64, 40], owner_representatives=len(set(owner_groups.values())),
                                      comparisons=len(self.pair_cache), accepted_pairs=sum(p['accepted'] for p in pairs)),
                          build_seconds=seconds, changed_documents=changed, build_history=history,
                          errors=self.errors, row_basis='Estimated from saved machine guesses; shares use all column rows, not category totals.')
            save(CACHE / 'documents.json', self.doc_cache)
            save(CACHE / 'pairs.json', self.pair_cache)
            save(CACHE / 'dedupe-pairs.json', pairs)
            save(CACHE / 'atlas.json', result)
            self.borderline(grouping_rows, pairs)
            with self.lock:
                self.snapshot = result
            print(f"Atlas build {seconds:.3f}s; {len(changed)} changed docs; {len(symbols)} symbols; {dict(counts)}; crop errors {len(self.errors)}", flush=True)
        except Exception:
            traceback.print_exc()
        finally:
            self.rebuild_lock.release()

    def borderline(self, rows, pairs):
        # Preserve the calibration artifact if no pixels changed.
        fp = digest([MATCH, DICE, COLOUR, NOVELTY, sorted((r['key'], r['swatch_hash']) for r in rows)])
        if read(CACHE / 'borderline.json').get('fingerprint') == fp:
            return
        lookup = {r['key']: r for r in rows}
        around = sorted(pairs, key=lambda p: abs(p['match']-MATCH)+abs(p['dice']-DICE))
        selected = ([p for p in around if p['accepted']][:8] + [p for p in around if not p['accepted']][:10])
        # Include the diagnostic pair that exposed added dolomite diagonals.
        diagnostic = [p for p in pairs if .97 < p['match'] < .98 and
                      'dolomit' in lookup[p['b']]['compiled']['member']['label'].lower() and
                      lookup[p['a']]['compiled']['member']['label'].lower() == 'limestone']
        selected += diagnostic[:2]
        sheet = Image.new('RGB', (1120, 110*len(selected)+55), 'white')
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
        draw.text((12, 10), f'Pixel >= {MATCH}; ink Dice >= {DICE}; background RGB <= {COLOUR}; novel ink <= {NOVELTY}. Native crops at 2x.', font=font, fill='black')
        for i, p in enumerate(selected):
            y = 50+i*110
            for j, key in enumerate((p['a'], p['b'])):
                r = lookup[key]
                im = Image.fromarray(r['rgb'])
                im = im.resize((im.width*2, im.height*2), Image.Resampling.NEAREST)
                im.thumbnail((290, 76), Image.Resampling.NEAREST)
                x = 12+j*350
                sheet.paste(im, (x, y))
                draw.text((x, y+78), key + ' ' + r['compiled']['member']['label'][:19], font=font, fill='black')
            draw.text((725, y+15), f"{'MERGE' if p['accepted'] else 'SEPARATE'}  pixel {p['match']:.4f}", font=font, fill='#174e39' if p['accepted'] else '#973e28')
            draw.text((725, y+40), f"ink {p['dice']:.4f}  colour {p['colour']:.2f}  novel {p['novelty']:.4f}", font=font, fill='black')
        sheet.save(SHOTS / 'dedupe-borderline.png')
        save(CACHE / 'borderline.json', dict(fingerprint=fp, pairs=selected))

    def run(self):
        while True:
            started = time.monotonic()
            self.rebuild()
            time.sleep(max(1, 180 - (time.monotonic() - started)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        route = unquote(urlparse(self.path).path)
        try:
            if route in ('/', '/index.html'):
                return self.send((HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if route == '/atlas.json':
                with self.server.atlas.lock:
                    value = dict(self.server.atlas.snapshot)
                value['health'] = self.server.atlas.health
                return self.send(json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
            if route.startswith('/img/'):
                target = (DATA / route[5:]).resolve()
                allowed = [CACHE, DATA / 'generated', DATA / 'proposals']
                if target.suffix.lower() == '.png' and any(target.is_relative_to(p.resolve()) for p in allowed) and target.is_file():
                    return self.send(target.read_bytes(), 'image/png')
            self.send(b'Not found', 'text/plain', 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self.send(str(exc).encode(), 'text/plain', 500)

    def send(self, body, content_type, code=200):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'public, max-age=180' if content_type == 'image/png' else 'no-store')
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--port', type=int, default=8783)
    args = parser.parse_args()
    atlas = Atlas()
    if args.build:
        atlas.rebuild()
    else:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
        server.atlas = atlas
        threading.Thread(target=atlas.run, daemon=True).start()
        print(f'Legend atlas http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()
