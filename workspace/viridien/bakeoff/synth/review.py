"""Local stdlib review server; append-only decisions and reproducible approvals."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from synth.common import HOME, DATA, read_json, write_json, local_file, relative, pid_slug


class ReviewStore:
    def __init__(self, data=DATA):
        self.data = Path(data).resolve()
        self.data.relative_to(HOME)
        self.data.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._mtimes = None
        self.reload()
        self.events = []
        path = self.data / 'review.jsonl'
        if path.exists():
            lines = path.read_text(encoding='utf-8').splitlines()
            for i, line in enumerate(lines):
                try: self.events.append(json.loads(line))
                except json.JSONDecodeError:
                    if i != len(lines)-1: raise
                    # A torn last write cannot consume the next event.
                    path.write_text('\n'.join(lines[:i])+'\n', encoding='utf-8')
        self.rebuild()

    def reload(self):
        """Re-read catalog, sheet index and picks when mining or reviewers rewrite them (no restart needed)."""
        files = [self.data / 'catalog.json', self.data / 'sheets' / 'index.json', self.data / 'selected.json', self.data / 'lookalikes.json', self.data / 'legend_crops.json', self.data / 'colour_gate.json']
        mtimes = tuple(f.stat().st_mtime if f.exists() else 0 for f in files)
        if mtimes == self._mtimes: return
        self._mtimes = mtimes
        self.catalog = read_json(files[0], {'patterns': []})
        self.patterns = {p['pid']: p for p in self.catalog['patterns']}
        self.index = read_json(files[1], {})
        self.selected = read_json(files[2], {})
        self.lookalikes = read_json(files[3], {})
        self.legend_crops = read_json(files[4], {})
        self.colour_gate = read_json(files[5], {})

    def stacks(self):
        stacks = {}
        for event in self.events:
            stack = stacks.setdefault(event['pid'], [])
            if event.get('kind') == 'undo':
                if stack and stack[-1]['id'] == event.get('undo_of'): stack.pop()
            else: stack.append(event)
        return stacks

    def decisions(self):
        return {pid: stack[-1]['decision'] for pid, stack in self.stacks().items() if stack}

    def candidates(self, pid):
        # Hide stretches that fail the colour filter (synth/colour_gate.py); numbers stay stable.
        hidden = {str(n) for n in self.colour_gate.get(pid, {}).get('hidden', [])}
        return {n: c for n, c in self.index.get(pid, {}).get('candidates', {}).items() if n not in hidden}

    def pattern(self, pid):
        self.reload()
        if pid not in self.patterns: raise ValueError('Unknown pattern')
        decision = self.decisions().get(pid)
        selected = self.selected.get(pid, {'picks': [], 'notes': '', 'issues': []})
        picks = decision['picks'] if decision else selected.get('picks', [])
        valid = self.candidates(pid)
        rivals = [dict(r, swatches=self.patterns.get(r['pid'], {}).get('swatches', [])[:1]) for r in self.lookalikes.get(pid, [])]
        return dict(pattern=self.patterns[pid], selected=selected, decision=decision, rivals=rivals, legend_crops=self.legend_crops.get(pid, []),
                    picks=[n for n in picks if str(n) in valid], candidates=valid)

    def bootstrap(self):
        self.reload()
        states = self.decisions()
        return dict(patterns=[dict(pid=p['pid'], family=p['family'], label=p['label'], n=len(self.candidates(p['pid'])),
                                   status=states.get(p['pid'], {}).get('status', 'unreviewed'),
                                   issues=self.selected.get(p['pid'], {}).get('issues', []))
                              for p in self.catalog['patterns']])

    def append(self, event):
        with (self.data / 'review.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False)+'\n')
            f.flush()
            import os
            os.fsync(f.fileno())
        self.events.append(event)
        self.rebuild()

    def save(self, pid, status, picks, note='', verdicts=None):
        with self.lock:
            if pid not in self.patterns: raise ValueError('Unknown pattern')
            if status not in ('approved', 'bad', 'repick', 'unreviewed'): raise ValueError('Invalid status')
            if not isinstance(picks, list) or any(type(n) is not int or str(n) not in self.candidates(pid) for n in picks):
                raise ValueError('Invalid candidate number')
            picks = sorted(set(picks))
            if not isinstance(note, str) or len(note) > 2000: raise ValueError('Invalid note')
            verdicts = verdicts or {}
            if not isinstance(verdicts, dict) or any(str(k) not in self.candidates(pid) or v not in ('y', 'n') for k, v in verdicts.items()):
                raise ValueError('Invalid verdicts')
            event = dict(id=uuid.uuid4().hex, timestamp=time.time(), kind='decision', pid=pid,
                         decision=dict(status=status, picks=picks, note=note.strip(), verdicts={str(k): v for k, v in verdicts.items()}))
            self.append(event)
            return event

    def undo(self, pid):
        with self.lock:
            stack = self.stacks().get(pid, [])
            if not stack: return None
            event = dict(id=uuid.uuid4().hex, timestamp=time.time(), kind='undo', pid=pid,
                         undo_of=stack[-1]['id'], decision=stack[-2]['decision'] if len(stack)>1 else None)
            self.append(event)
            return event

    def rebuild(self):
        """Review log is the authority; selected.json alone never grants approval."""
        approved = {}
        expected = set()
        for pid, decision in self.decisions().items():
            if decision['status'] != 'approved': continue
            rows = []
            for i, number in enumerate(decision['picks'], 1):
                c = self.candidates(pid).get(str(number))
                if not c: continue
                source = local_file(c['file'])
                output = self.data / 'approved' / pid_slug(pid) / f'{i:02d}.png'
                output.parent.mkdir(parents=True, exist_ok=True)
                expected.add(output.resolve())
                # Always refresh: a new index can refer to different source bytes.
                shutil.copyfile(source, output)
                rows.append(dict(file=relative(output), **{k: c[k] for k in ('log', 'top_y', 'bottom_y', 'col_w', 'h')}))
            approved[pid] = rows
        write_json(self.data / 'approved.json', approved)
        write_json(self.data / 'review_notes.json', {pid: dict(status=d['status'], note=d.get('note', ''), picks=d['picks'])
                                                     for pid, d in self.decisions().items() if d and d.get('note')})
        for path in (self.data / 'approved').glob('*/*.png'):
            if path.resolve() not in expected: path.unlink()
        return approved


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def reply(self, value, code=200):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/api/bootstrap':
                with store.lock: return self.reply(store.bootstrap())
            if url.path == '/api/pattern':
                try:
                    with store.lock: value = store.pattern(parse_qs(url.query).get('pid', [''])[0])
                    return self.reply(value)
                except ValueError as e: return self.reply({'error': str(e)}, 404)
            try:
                path = HOME / 'review.html' if url.path == '/' else local_file(unquote(url.path).lstrip('/'))
                # Only UI files and known data images are served, never caches.
                if path.name != 'review.html' and not (path.is_relative_to(store.data) and path.suffix.lower() == '.png'):
                    return self.reply({'error': 'Not found'}, 404)
                content = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8' if path.suffix == '.html' else 'image/png')
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Cache-Control', 'no-cache' if path.suffix == '.html' else 'public, max-age=3600')
                self.end_headers(); self.wfile.write(content)
            except (OSError, ValueError): self.reply({'error': 'Not found'}, 404)

        def do_POST(self):
            # Same-origin browser writes; no permissive CORS and no cross-site POSTs.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.reply({'error': 'Cross-origin write rejected'}, 403)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384: raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(size))
                if self.path == '/api/save': result = store.save(request['pid'], request['status'], request['picks'], request.get('note', ''), request.get('verdicts'))
                elif self.path == '/api/undo': result = store.undo(request['pid'])
                else: return self.reply({'error': 'Not found'}, 404)
                return self.reply({'event': result})
            except (ValueError, KeyError, TypeError): self.reply({'error': 'Invalid review decision'}, 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    store = ReviewStore()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(store))
    print(f'Review: http://{args.host}:{args.port} ({len(store.patterns)} patterns)', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
