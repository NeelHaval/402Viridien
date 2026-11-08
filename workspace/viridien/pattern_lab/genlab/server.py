"""Generator lab server: rebuilds whenever a label is saved in the labeller.

    python -X utf8 -m genlab.server            (from viridien_work/pattern_lab)
    open http://127.0.0.1:8781

It only reads the labeller's data/reviews; it writes under data/genlab.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from . import paths, proposals
from .build import Builder, labstore
from .reading import ReadOnlyStore

STATIC = Path(__file__).resolve().parent / 'static'
REVIEWS = paths.DATA / 'reviews'


class Lab:
    def __init__(self):
        from .dataset_eval import configure_loop
        configure_loop()
        self.builder = Builder()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.status = dict(building=False, error=None, last_change=None, builds=0)
        self.seen = None

    def fingerprint(self):
        files = sorted(REVIEWS.glob('*.json')) if REVIEWS.exists() else []
        sig = [(f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files]
        q = paths.QUEUE
        if q.exists():
            sig.append((q.name, q.stat().st_mtime_ns))
        for dependency in (paths.DATA/'normalized/current.json',paths.DATA/'normalized/split-policy.json',paths.PREFILL):
            if dependency.exists():
                stamp = dependency.stat()
                sig.append((str(dependency),stamp.st_mtime_ns,stamp.st_size))
        for folder in (paths.CANDIDATES, paths.VERDICTS):
            for f in sorted(folder.glob('*.json')):
                sig.append((str(f), f.stat().st_mtime_ns, f.stat().st_size))
        for f in sorted(list(paths.PROPOSALS.glob('*/*.json'))+list(paths.PROPOSALS.glob('*/*.png'))):
            try:
                stat = f.stat()
                sig.append((str(f), stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                pass
        return hash(tuple(sig))

    def loop(self):
        while True:
            fp = self.fingerprint()
            if fp != self.seen or self.wake.is_set():
                self.wake.clear()
                self.seen = fp
                self.rebuild()
            self.wake.wait(3)

    def rebuild(self):
        with self.lock:
            self.status['building'] = True
            try:
                self.builder.build()
                self.status.update(error=None, builds=self.status['builds'] + 1, last_change=time.time())
            except Exception:  # keep serving the last good state
                self.seen = None  # retry transient reader/writer races next poll
                self.status['error'] = traceback.format_exc()[-1500:]
                print(self.status['error'], flush=True)
            finally:
                self.status['building'] = False


LAB: Lab


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, kind='application/json', cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'public, max-age=86400' if cache else 'no-store')
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # A tab reload or closed browser can cancel an image mid-response.
            return

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ('/', '/index.html'):
                return self.send(200, (STATIC / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if url.path == '/state.json':
                state = json.loads(paths.STATE.read_text(encoding='utf-8')) if paths.STATE.exists() else {}
                state['status'] = LAB.status
                state['fleet'] = proposals.read_json(paths.FLEET / 'progress.json', {})
                return self.send(200, state)
            if url.path.startswith('/out/'):
                from urllib.parse import unquote
                target = (paths.OUT / unquote(url.path[5:])).resolve()
                if paths.OUT.resolve() not in target.parents or not target.is_file():
                    return self.send(404, {'error': 'not found'})
                return self.serve_artifact(target)
            if url.path == '/crop':
                return self.crop(q)
            if url.path == '/dataset.zip':
                return self.dataset()
            if url.path == '/api/datasets':
                from .dataset import ROOT, read
                from .dataset_loop import api_index
                payload = api_index(ROOT)
                payload['quarantine'] = read(ROOT/'quarantine.json',{})
                for version in payload.get('versions',[]):
                    if version['version'] in payload['quarantine']:
                        version.update(download_ready=False,quarantined=True)
                for key in ('evaluation','feedback'):
                    if payload.get(key,{}).get('version') in payload['quarantine']:
                        payload[key] = {}
                return self.send(200, payload)
            if url.path.startswith('/datasets/'):
                return self.versioned_dataset(url.path)
            return self.send(404, {'error': 'not found'})
        except Exception as e:  # noqa: BLE001
            return self.send(500, {'error': str(e)})

    def crop(self, q):
        doc_id = q['doc'][0]
        box = [int(v) for v in q['box'][0].split(',')]
        if box[3] - box[1] > 1600 or box[2] - box[0] > 1600:
            return self.send(400, {'error': 'crop too large'})
        doc = ReadOnlyStore(labstore.Store()).get(doc_id)
        with LAB.builder.images.lock:
            rgb = LAB.builder.crops.get(doc, box)
        out = io.BytesIO()
        Image.fromarray(rgb).save(out, format='PNG')
        return self.send(200, out.getvalue(), 'image/png', cache=True)

    def versioned_dataset(self, request_path):
        from .dataset import ROOT
        from urllib.parse import unquote
        target = (ROOT / unquote(request_path[len('/datasets/'):])).resolve()
        if not target.is_relative_to(ROOT.resolve()) or not target.is_file():
            return self.send(404, {'error': 'not found'})
        return self.serve_artifact(target)

    @staticmethod
    def quarantined_artifact(target):
        """Both URL aliases and every asset share the resolved-path policy."""
        from .dataset import ROOT, read, digest
        from . import paths
        target = Path(target).resolve()
        root = ROOT.resolve()
        if not target.is_relative_to(root):
            return False
        registry = {name.casefold() for name in read(root/'quarantine.json',{})}
        first = target.relative_to(root).parts[0].casefold()
        version = first.split('.',1)[0]
        if first in registry or version in registry:
            return True
        folder = root/version
        eligibility = read(folder/'eligibility.json',{})
        if eligibility:
            current = digest(read(paths.DATA/'normalized/split-policy.json',{}))
            if eligibility.get('policy_sha256')!=current:
                from .dataset_eval import revoke_unsafe_versions
                revoke_unsafe_versions(root)
                return True
        return False

    def serve_artifact(self, target):
        import mimetypes
        if self.quarantined_artifact(target):
            return self.send(409,{'error':'Dataset quarantined: revoked source eligibility; archive is retained locally for comparison'})
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
        if target.suffix == '.zip':
            self.send_header('Content-Disposition', 'attachment; filename="' + target.name + '"')
        self.send_header('Content-Length', str(target.stat().st_size))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        try:
            with target.open('rb') as source:
                while chunk := source.read(256 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def dataset(self):
        out = io.BytesIO()
        with LAB.lock, zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(paths.GEN.rglob('*')):
                if f.is_file():
                    if self.quarantined_artifact(f):
                        return self.send(409,{'error':'Generated archive contains a quarantined artifact'})
                    z.write(f, f.relative_to(paths.GEN).as_posix())
        self.send_response(200)
        self.send_header('Content-Type', 'application/zip')
        self.send_header('Content-Disposition', 'attachment; filename="generated-patterns-%s.zip"'
                         % time.strftime('%Y%m%d-%H%M'))
        self.send_header('Content-Length', str(out.tell()))
        self.end_headers()
        self.wfile.write(out.getvalue())

    def do_POST(self):
        url = urlparse(self.path)
        size = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(size) or b'{}') if size else {}
        if url.path == '/rate':
            if body.get('verdict') not in ('good', 'bad', 'clear') or not isinstance(body.get('file'), str):
                return self.send(400, {'error': 'verdict must be good, bad or clear'})
            rec = dict(file=body['file'], verdict=body['verdict'], note=str(body.get('note', ''))[:500],
                       at=time.time())
            # Keep audit provenance even after this strip is superseded/pruned.
            state = proposals.read_json(paths.STATE, {})
            card = next((c for c in state.get('cards', []) if rec['file'] in c['samples']), {})
            if card.get('source') in ('agent-made', 'unlisted'):
                rec.update(source=card['source'], method=card.get('method', 'unknown'))
            paths.RATINGS.parent.mkdir(parents=True, exist_ok=True)
            with paths.RATINGS.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            LAB.wake.set()
            return self.send(200, {'ok': True})
        if url.path == '/rebuild':
            LAB.wake.set()
            return self.send(200, {'ok': True})
        return self.send(404, {'error': 'not found'})


def main():
    global LAB
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8781)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    LAB = Lab()
    threading.Thread(target=LAB.loop, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print('Generator lab on http://%s:%d' % (args.host, args.port), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
