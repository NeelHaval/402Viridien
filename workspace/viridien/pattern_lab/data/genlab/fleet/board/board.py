"""Read-only fleet progress board. Run through start-board.ps1."""
from __future__ import annotations
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse, quote, unquote
from urllib.request import urlopen

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image

DATA = ROOT / 'data/genlab'
FLEET = DATA / 'fleet'
PROPOSALS = DATA / 'proposals'
LAB = 'http://127.0.0.1:8781'
STATES = ['In the lab', 'Approved, waiting for the lab', 'Being made', 'Being reviewed',
          'Unsure', 'Rejected', 'Failed', 'Skipped (not a rock fill)', 'Queued',
          'Not lithology', 'Test well (held out)', 'Not started']
TERMINAL = {'done', 'failed', 'cancelled', 'canceled', 'error', 'completed'}


def read_dict(path):
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def boxes(value):
    if not isinstance(value, list):
        return []
    return [list(b) for b in value if isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in b)
            and b[2] > b[0] and b[3] > b[1]]


def text(value):
    return value if isinstance(value, str) else ''


def crop_url(doc, box):
    return '/crop?' + urlencode(dict(doc=doc, box=','.join(str(int(v)) for v in box)))


class Board:
    def __init__(self):
        self.lock = threading.Lock()
        self.crop_lock = threading.Lock()
        self.lab = read_dict(HERE / 'last-lab.json')
        self.lab_ok = False
        self.lab_at = (HERE / 'last-lab.json').stat().st_mtime if self.lab else 0
        self.jobs = read_dict(HERE / 'last-jobs.json')
        self.jobs_at = (HERE / 'last-jobs.json').stat().st_mtime if self.jobs else 0
        self.jobs_error = None
        self.snapshot = None
        self.images = None
        self.documents = {}
        self.image_sizes = {}
        self.progress_lock = threading.Lock()
        self.progress = read_dict(HERE / 'last-progress.json')

    def track(self):
        """Five-minute tracker cadence, independent of lab polling and HTTP reads."""
        time.sleep(2)  # Allow the HTTP listener to serve the tracker's board.json read.
        while True:
            try:
                result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(DATA / 'tracker.py'), '--json'],
                    capture_output=True, text=True, encoding='utf-8', timeout=150,
                    env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode:
                    raise ValueError(result.stderr[-500:])
                current = json.loads(result.stdout)
                points = []
                for line in (DATA / 'tracker.jsonl').read_text(encoding='utf-8').splitlines():
                    try:
                        row = json.loads(line)
                        points.append({k: row.get(k) for k in ('at', 'schema', 'pct', 'built')})
                    except ValueError:
                        pass
                payload = dict(current=current, history=points, label_next=label_next(self.get(), self.lab),
                               error=None, refresh_seconds=300)
                (HERE / 'last-progress.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
                with self.progress_lock:
                    self.progress = payload
            except Exception as exc:
                with self.progress_lock:
                    self.progress = dict(self.progress, error=str(exc), refresh_seconds=300)
            time.sleep(300)

    def get_progress(self):
        with self.progress_lock:
            return self.progress or dict(error=None, current=None, history=[], label_next=[], refresh_seconds=300)

    def poll(self):
        """Slow/network work stays off the HTTP request threads."""
        while True:
            try:
                with urlopen(LAB + '/state.json', timeout=4) as response:
                    lab = json.load(response)
                if not isinstance(lab, dict) or not isinstance(lab.get('cards'), list):
                    raise ValueError('Unexpected lab state')
                with self.lock:
                    self.lab, self.lab_ok, self.lab_at = lab, True, time.time()
                (HERE / 'last-lab.json').write_text(json.dumps(lab), encoding='utf-8')
            except Exception:
                with self.lock:
                    self.lab_ok = False
            if time.time() - self.jobs_at >= 30:
                self.jobs_at = time.time()
                try:
                    script = Path.home() / '.claude/skills/sol-fleet/scripts/sol_jobs.py'
                    result = subprocess.run([sys.executable, '-X', 'utf8', str(script),
                                             'status', '--prefix', 'fx-'],
                                            capture_output=True, text=True, encoding='utf-8',
                                            env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                                            timeout=25, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    if result.returncode:
                        raise ValueError('Job status command failed')
                    jobs = {}
                    for line in result.stdout.splitlines():
                        if line.startswith('SOLJOB '):
                            try:
                                job = json.loads(line[7:])
                                if isinstance(job, dict) and isinstance(job.get('job'), str):
                                    jobs[job['job']] = {k: job.get(k) for k in ('job', 'state', 'model', 'elapsed_s')}
                            except ValueError:
                                pass
                    with self.lock:
                        self.jobs, self.jobs_error = jobs, None
                    (HERE / 'last-jobs.json').write_text(json.dumps(jobs), encoding='utf-8')
                except Exception as exc:
                    self.jobs_error = str(exc)
            time.sleep(5)

    def get(self):
        with self.lock:
            if self.snapshot and time.time() - self.snapshot['at'] < 10:
                return self.snapshot
            self.snapshot = self.build()
            return self.snapshot

    def build(self):
        units = read_dict(FLEET / 'state.json')
        progress = read_dict(FLEET / 'progress.json')
        cards = {}
        for card in self.lab.get('cards', []):
            if isinstance(card, dict) and isinstance(card.get('doc_id'), str):
                lid = card.get('legend_id')
                key = f"{card['doc_id']}:{lid}"
                cards[key] = card
        proposals = {}
        reviews = {}
        for path in sorted(PROPOSALS.glob('*/*.json')):
            match = re.fullmatch(r'(legend|unlisted)-(\d+)\.json', path.name)
            if not match:
                continue
            p = read_dict(path)
            kind, n = match.groups()
            lid = n if kind == 'legend' else 'u' + n
            if str(p.get('legend_id' if kind == 'legend' else 'unlisted_id')) != lid:
                continue
            if p.get('doc_id', path.parent.name) != path.parent.name:
                continue
            rnd = p.get('round')
            if not isinstance(rnd, int) or rnd < 1:
                continue
            review_path = path.parent / f'{kind}-review-r{rnd}.json'
            if review_path not in reviews:
                reviews[review_path] = read_dict(review_path)
            verdict = reviews[review_path].get(lid, {})
            proposals[f'{path.parent.name}:{lid}'] = (p, verdict if isinstance(verdict, dict) else {})

        documents, entries, unlisted = [], [], []
        for path in sorted((DATA / 'candidates').glob('*.json')):
            doc = read_dict(path)
            did = doc.get('doc_id')
            if not isinstance(did, str) or path.stem != did or not isinstance(doc.get('entries'), list):
                continue
            work = read_dict(FLEET / 'work' / f'{did}.json')
            work_entries = work.get('entries', [])
            work_ids = {str(e['legend_id']) for e in work_entries
                        if isinstance(e, dict) and 'legend_id' in e} if isinstance(work_entries, list) else set()
            name = text(doc.get('name')) or did
            documents.append(dict(doc_id=did, name=name, training_allowed=doc.get('training_allowed') is not False))
            for entry in doc['entries']:
                if not isinstance(entry, dict) or not isinstance(entry.get('legend_id'), (str, int)):
                    continue
                entries.append(self.entry(did, name, doc, entry, 'legend', units, work_ids, cards, proposals))
            keys = {k for k in proposals if k.startswith(did + ':u')} | {k for k in cards if k.startswith(did + ':u')}
            for key in sorted(keys):
                lid = key.split(':', 1)[1]
                p = proposals.get(key, ({}, {}))[0]
                entry = dict(legend_id=lid, label=p.get('description') or cards.get(key, {}).get('label') or lid)
                unlisted.append(self.entry(did, name, doc, entry, 'unlisted', units, {lid}, cards, proposals))
        counts = Counter(e['state'] for e in entries)
        train_counts = Counter(e['state'] for e in entries if e['training_allowed'])
        live = [j for j in self.jobs.values() if isinstance(j, dict) and j.get('state') not in TERMINAL]
        return dict(at=time.time(), states=STATES, entries=entries, unlisted=unlisted, documents=documents,
                    total=len(entries), training_total=sum(train_counts.values()), document_count=len(documents),
                    counts={s: counts[s] for s in STATES}, training_counts={s: train_counts[s] for s in STATES},
                    unlisted_found=len(unlisted), unlisted_approved=sum(e['state'] in STATES[:2] for e in unlisted),
                    live_jobs=live, units_done=sum(isinstance(u, dict) and u.get('phase') == 'done' for u in units.values()),
                    units_total=len(units), progress=progress, lab_online=self.lab_ok, lab_last_good=self.lab_at,
                    jobs_at=self.jobs_at, jobs_error=self.jobs_error)

    def entry(self, did, name, doc, e, kind, units, work_ids, cards, proposals):
        lid = str(e['legend_id'])
        key = f'{did}:{lid}'
        card = cards.get(key, {})
        p, review = proposals.get(key, ({}, {}))
        unit = units.get(f'{kind}:{did}', {})
        unit = unit if isinstance(unit, dict) else {}
        job = self.jobs.get(unit.get('job'), {})
        job = job if isinstance(job, dict) else {}
        live = bool(job) and job.get('state') not in TERMINAL and unit.get('phase') in ('make', 'review') and lid in work_ids
        verdict = review.get('verdict')
        note = text(review.get('note')) or text(p.get('notes')) or text(p.get('note'))
        if card:
            state = STATES[0]
            cv = card.get('verdict')
            note = (text(cv.get('note')) if isinstance(cv, dict) else '') or note
        elif p.get('status') == 'made' and verdict == 'confident':
            state = STATES[1]
        elif live:
            state = STATES[2 if unit.get('phase') == 'make' else 3]
        elif verdict in ('unsure', 'reject', 'rejected'):
            state = 'Unsure' if verdict == 'unsure' else 'Rejected'
        elif p.get('status') in ('failed', 'skip', 'skipped'):
            state = 'Failed' if p.get('status') == 'failed' else 'Skipped (not a rock fill)'
        elif lid in work_ids and not p:
            state = 'Queued'
        elif e.get('tier') == 'not-lithology':
            state = 'Not lithology'
        elif doc.get('training_allowed') is False:
            state = 'Test well (held out)'
        else:
            state = 'Not started'
        source = text(card.get('source')) or ('unlisted' if kind == 'unlisted' else 'agent-made' if p else '')
        method = text(card.get('method')) or text(p.get('method'))
        chips = {'human': 'from your boxes', 'machine-verified': 'machine-verified', 'borrowed': 'borrowed',
                 'swatch': 'approved swatch', 'unlisted': 'not in legend'}
        chip = chips.get(source, f'agent-made ({method or "pending"})' if source == 'agent-made' else source or 'awaiting work')
        real_boxes = boxes(card.get('real')) or boxes(p.get('column_boxes'))
        swatch = boxes([e.get('swatch_box')])
        real_doc = text(card.get('real_doc_id')) or did
        real = [crop_url(real_doc, b) for b in real_boxes[:3]] or [crop_url(did, b) for b in swatch]
        samples = card.get('samples')
        strips = [LAB + '/out/' + quote(s, safe='/') for s in samples[:2] if isinstance(s, str)] if isinstance(samples, list) else []
        tile = text(p.get('tile_png'))
        tile_url = None
        if tile:
            try:
                target = proposal_file(did, tile)
                stamp = target.stat().st_mtime_ns
                tile_url = '/tile?' + urlencode(dict(doc=did, f=tile, v=stamp))
            except (OSError, ValueError):
                pass
        if not strips and tile_url:
            strips = [tile_url, tile_url + '&offset=1']
        strip_sizes = []
        for url in strips:
            if url.startswith('/tile?'):
                strip_sizes.append([64, 240])
                continue
            path = (DATA / url.split('/out/', 1)[-1]).resolve()
            try:
                stamp = path.stat().st_mtime_ns
                cached_size = self.image_sizes.get(str(path))
                if not cached_size or cached_size[0] != stamp:
                    with Image.open(path) as image:
                        cached_size = (stamp, list(image.size))
                    self.image_sizes[str(path)] = cached_size
                strip_sizes.append(cached_size[1])
            except (OSError, ValueError):
                strip_sizes.append([64, 240])
        is_live = state in STATES[2:4]
        real_kind = 'column crops' if real_boxes else 'legend swatch'
        if real_doc == did and real_boxes == swatch or card.get('evidence_kind') == 'swatch' or source == 'swatch':
            real_kind = 'legend swatch'
        return dict(key=key, doc_id=did, doc_name=name, short_doc=name[:52], legend_id=e['legend_id'],
                    label=text(e.get('label')) or lid, normalized=''.join(text(e.get('label')).casefold().split()),
                    state=state, source=source, source_chip=chip, method=method, note=note or text(e.get('reason')),
                    round=unit.get('round') if is_live else p.get('round'), model=text(job.get('model')) if is_live else '',
                    live=is_live, training_allowed=doc.get('training_allowed') is not False,
                    real=real, real_kind=real_kind,
                    swatch=[crop_url(did, b) for b in swatch], strips=strips, strip_sizes=strip_sizes, tile=tile_url,
                    description=text(p.get('description')), looks_like=text(p.get('looks_like')),
                    share_of_column_rows=p.get('share_of_column_rows'), proposal_status=p.get('status'))

    def crop(self, did, box):
        """Use the lab first; fallback never writes to the shared crop cache."""
        try:
            with urlopen(LAB + crop_url(did, box), timeout=3) as response:
                blob = response.read()
                if blob.startswith(b'\x89PNG\r\n\x1a\n'):
                    return blob
        except Exception:
            pass
        with self.crop_lock:
            if self.images is None:
                from model import Images
                self.images = Images()
            if did not in self.documents:
                from model import Store
                self.documents[did] = Store().get(did)
            cached = DATA / 'crops' / did / ('%d_%d_%d_%d.png' % tuple(box))
            if cached.is_file():
                return cached.read_bytes()
            return png(self.images.crop(self.documents[did], box))


def proposal_file(doc, filename):
    if not re.fullmatch(r'[0-9a-f]{16}', doc):
        raise ValueError('Invalid document')
    path = (PROPOSALS / doc / filename).resolve()
    if PROPOSALS.resolve() not in path.parents or path.suffix.lower() != '.png' or not path.is_file():
        raise ValueError('Invalid proposal image')
    return path


def normalized(value):
    return re.sub(r'[^\w]', '', str(value or '').casefold()).replace('_', '')


def label_next(board, lab):
    """Machine suggestions are navigation aids, never counted as owner truth."""
    labels = {normalized(r.get('normalized_label', r.get('label'))): r
              for r in lab.get('coverage', {}).get('labels', [])}
    entries = [e for e in board.get('entries', []) if e.get('training_allowed')
               and e.get('state') in ('Failed', 'Rejected', 'Unsure')]
    wanted = {(e['doc_id'], normalized(e['label'])) for e in entries}
    doc_names = {}
    for path in (DATA / 'candidates').glob('*.json'):
        d = read_dict(path)
        # Prefill.log uses candidate.name, usually without its extension.
        for name in (d.get('name', ''), Path(d.get('name', '')).stem):
            doc_names[normalized(name)] = d.get('doc_id', path.stem)
    best = {}
    prefill = ROOT.parent / 'bakeoff/labeltool/data/prefill.jsonl'
    try:
        with prefill.open(encoding='utf-8') as source:
            for line in source:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                did = doc_names.get(normalized(row.get('log')))
                top, bottom = row.get('top_y'), row.get('bottom_y')
                if not isinstance(top, (int, float)) or not isinstance(bottom, (int, float)) or bottom <= top:
                    continue
                suggestions = list(row.get('suggestions', []))
                suggestions.append(dict(label=row.get('suggestion'), tier=row.get('tier'), source=row.get('best_source')))
                for suggestion in suggestions:
                    key = (did, normalized(suggestion.get('label')))
                    if key not in wanted:
                        continue
                    confidence = suggestion.get('confidence', suggestion.get('probability', 0)) or 0
                    tier = suggestion.get('tier', row.get('tier'))
                    rank = (not row.get('disputed'), {'high': 3, 'medium': 2, 'low': 1}.get(tier, 0), confidence, bottom-top)
                    if key not in best or rank > best[key][0]:
                        interval = {k: row.get(k) for k in ('id', 'top_y', 'bottom_y', 'depth', 'top_depth', 'bottom_depth', 'page') if row.get(k) is not None}
                        interval.update(tier=tier, best_source=suggestion.get('source'), alternative=normalized(row.get('suggestion')) != key[1])
                        best[key] = (rank, interval)
    except OSError:
        pass
    result = []
    for e in entries:
        cov = labels.get(normalized(e['label']), {})
        share = cov.get('estimated_share_of_column_rows')
        # Aggregate label coverage cannot identify one document's contribution.
        estimated_share = share / max(1, cov.get('all_entries', 1)) if isinstance(share, (float, int)) else None
        note = e.get('note', '')
        unblock = bool(re.search(r'one (?:visible |complete |horizontal )?(?:course|row)|single[^.]{0,35}course|no vertical (?:recurrence|repeat)|no[^.]{0,25}vertical recurrence|only[^.]{0,30}(?:course|row)|(?:lack|without|insufficient)[^.]{0,35}(?:vertical|repeat)', note, re.I))
        result.append(dict(doc_id=e['doc_id'], doc_name=e['doc_name'], legend_id=e['legend_id'], label=e['label'],
                           state=e['state'], note=note[:220] + ('…' if len(note) > 220 else ''),
                           unblocks_generation=unblock, estimated_share=estimated_share,
                           share_basis='label row share ÷ all same-name board entries; document impact is estimated',
                           interval=best.get((e['doc_id'], normalized(e['label'])), (None, None))[1]))
    return sorted(result, key=lambda r: (not r['unblocks_generation'], -(r['estimated_share'] or 0), r['doc_name'], str(r['legend_id'])))


def assembler_file(request_path):
    base = (DATA / 'assemble').resolve()
    path = (base / unquote(request_path.removeprefix('/assembler/'))).resolve()
    allowed = {'.html', '.png', '.jpg', '.jpeg', '.svg', '.webp', '.gif', '.css'}
    if base not in path.parents or path.suffix.lower() not in allowed or not path.is_file():
        raise ValueError('Invalid assembler report asset')
    return path


def png(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, data, kind):
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'public, max-age=30' if kind == 'image/png' else 'no-store')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ('/', '/index.html'):
                return self.send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if url.path == '/board.json':
                return self.send(200, BOARD.get(), 'application/json; charset=utf-8')
            if url.path == '/progress.json':
                return self.send(200, BOARD.get_progress(), 'application/json; charset=utf-8')
            if url.path.startswith('/assembler/'):
                path = assembler_file(url.path)
                return self.send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            if url.path in ('/tile', '/sheet'):
                path = proposal_file(q['doc'][0], q['f'][0])
                if url.path == '/sheet':
                    return self.send(200, path.read_bytes(), 'image/png')
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert('RGB'))
                h, w = rgb.shape[:2]
                strip = np.tile(rgb, (math.ceil(480 / h) + 1, math.ceil(128 / w) + 1, 1))
                offset = min(h // 2, 120) if q.get('offset') == ['1'] else 0
                return self.send(200, png(Image.fromarray(strip[offset:offset + 240, :64])), 'image/png')
            if url.path == '/crop':
                did = q['doc'][0]
                box = [int(v) for v in q['box'][0].split(',')]
                if not re.fullmatch(r'[0-9a-f]{16}', did) or not boxes([box]) or min(box) < 0 or box[2]-box[0] > 1600 or box[3]-box[1] > 1600:
                    raise ValueError('Invalid crop')
                return self.send(200, BOARD.crop(did, box), 'image/png')
            if url.path == '/favicon.ico':
                return self.send(204, b'', 'image/x-icon')
            self.send(404, {'error': 'Not found'}, 'application/json')
        except (ValueError, KeyError, OSError) as exc:
            self.send(400, {'error': str(exc)}, 'application/json')
        except Exception as exc:
            self.send(503, {'error': str(exc)}, 'application/json')


if __name__ == '__main__':
    BOARD = Board()
    threading.Thread(target=BOARD.poll, daemon=True).start()
    threading.Thread(target=BOARD.track, daemon=True).start()
    print('Fleet board: http://127.0.0.1:8782', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8782), Handler).serve_forever()
