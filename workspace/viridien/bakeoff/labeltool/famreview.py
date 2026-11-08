"""Quick review of dataset-wide pattern families (labeltool/families_all.py output).

    python -X utf8 -B labeltool/famreview.py --port 8771   -> http://127.0.0.1:8771

Each family shows its 12 numbered sheet chunks (the same numbers the LLM checkers used); expand to see
more members. Tap a chunk to mark it wrong; marks go to analysis/families_all/marks.jsonl.
LLM verdicts are read from analysis/families_all/llm.json when present.
"""
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import io
import json
import numpy as np
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'analysis' / 'families_all'
PAGE = Path(__file__).resolve().parent / 'static' / 'famreview.html'
LOCK = threading.Lock()
LDIR = ROOT / 'analysis' / 'legend_families'
LPAGE = Path(__file__).resolve().parent / 'static' / 'legendreview.html'
CDIR = ROOT / 'analysis' / 'categories'
SWDIR = ROOT.parent / 'analysis' / 'swatches'
MPAGE = Path(__file__).resolve().parent / 'static' / 'mixreview.html'
CPAGE = Path(__file__).resolve().parent / 'static' / 'catreview.html'
LDIRS = {'1': LDIR, '2': ROOT / 'analysis' / 'legend_families2'}


def leg_marks(d=LDIR):
    out = {}
    p = d / 'marks.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                e = json.loads(line); out[f"{e['group']}|{e.get('k', '')}"] = e
    return out


def cat_marks():
    out = {}
    p = CDIR / 'marks.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                e = json.loads(line); out[f"{e['cat']}|{e.get('id', '')}"] = e
    return out


def read_json(p, default):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def marks():
    out = {}
    p = DIR / 'marks.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                e = json.loads(line)
                out[f"{e['family']}|{e['log']}|{e['y']}"] = e
    return out


def png(im):
    b = io.BytesIO(); im.save(b, 'PNG'); return b.getvalue()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, code=200, ctype='application/json'):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'max-age=3600' if ctype == 'image/png' else 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == '/':
                return self.send(PAGE.read_bytes(), ctype='text/html; charset=utf-8')
            if u.path == '/mix':
                return self.send(MPAGE.read_bytes(), ctype='text/html; charset=utf-8')
            if u.path == '/api/mix':
                return self.send(dict(read_json(CDIR / 'mix.json', {}), marks=cat_marks(), llm=read_json(CDIR / 'pairs_llm.json', {})))
            if u.path.startswith('/sw/'):
                f = SWDIR / Path(u.path).name
                if f.suffix == '.png' and f.exists():
                    return self.send(f.read_bytes(), ctype='image/png')
                return self.send({'error': 'not found'}, 404)
            if u.path == '/conf':
                return self.send((Path(__file__).resolve().parent / 'static' / 'confreview.html').read_bytes(), ctype='text/html; charset=utf-8')
            if u.path == '/api/conf':
                return self.send(dict(read_json(CDIR / 'conf.json', {}), marks=cat_marks(), llmrev=read_json(CDIR / 'llmrev4' / 'summary.json', None) or read_json(CDIR / 'llmrev' / 'summary.json', {})))
            if u.path == '/nsw':  # native-resolution legend swatch (legend parser v2)
                from labeltool import legend_classify as LC
                for lb, a in LC._native().get(q['log'], []):
                    if lb == q['label']:
                        return self.send(png(Image.fromarray(np.ascontiguousarray(a[..., :3]).astype(np.uint8))), ctype='image/png')
                return self.send({'error': 'no swatch'}, 404)
            if u.path == '/cat':
                return self.send(CPAGE.read_bytes(), ctype='text/html; charset=utf-8')
            if u.path == '/api/cat':
                return self.send(dict(read_json(CDIR / 'review.json', {'categories': []}), marks=cat_marks()))
            if u.path == '/legend':
                return self.send(LPAGE.read_bytes(), ctype='text/html; charset=utf-8')
            if u.path == '/api/legend':
                d = LDIRS.get(q.get('v', '1'), LDIR)
                return self.send(dict(read_json(d / 'groups.json', {'groups': []}), llm=read_json(d / 'llm.json', {}), marks=leg_marks(d)))
            if u.path.startswith('/lf/') or u.path.startswith('/lf2/'):
                f = ((LDIRS['2'] if u.path.startswith('/lf2/') else LDIR) / Path(u.path).name)
                if f.suffix == '.png' and f.exists():
                    return self.send(f.read_bytes(), ctype='image/png')
                return self.send({'error': 'not found'}, 404)
            if u.path == '/api/families':
                fams = read_json(DIR / 'families.json', {'families': []})
                return self.send(dict(fams, llm=read_json(DIR / 'llm.json', {}), marks=marks()))
            if u.path == '/chunk':
                ctx = jev2.Context.get(q['log'])
                y = int(q['y']); h = min(400, int(q.get('h', 48)))
                with ctx['lock']:
                    rgb = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=y, y1=y + h)
                return self.send(png(Image.fromarray(rgb[:, :, :3])), ctype='image/png')
            if u.path == '/swatch':
                for t in jev2.Context.get(q['log'])['tiles']:
                    if t['label'] == q['label']:
                        return self.send(png(Image.fromarray(t['swatch'][:, :, :3])), ctype='image/png')
                return self.send({'error': 'no swatch'}, 404)
            return self.send({'error': 'not found'}, 404)
        except Exception as e:
            return self.send({'error': type(e).__name__}, 400)

    def do_POST(self):
        if self.headers.get('Origin') and self.headers['Origin'] != 'http://' + self.headers.get('Host', ''):
            return self.send({'error': 'cross-origin'}, 403)
        try:
            n = int(self.headers.get('Content-Length', 0))
            if not 0 < n <= 8192:
                raise ValueError
            r = json.loads(self.rfile.read(n))
            if urlparse(self.path).path == '/api/catmark':
                e = dict(cat=str(r['cat'])[:60], id=str(r.get('id', ''))[:40], verdict=str(r.get('verdict', ''))[:20],
                         label=str(r.get('label', ''))[:80], log=str(r.get('log', ''))[:120], note=str(r.get('note', ''))[:1000], t=time.time())
                with LOCK, (CDIR / 'marks.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(e) + chr(10))
                return self.send({'ok': True})
            if urlparse(self.path).path == '/api/legmark':
                e = dict(group=int(r['group']), k=(int(r['k']) if r.get('k') not in (None, '') else ''),
                         verdict=str(r.get('verdict', ''))[:20], note=str(r.get('note', ''))[:1000], t=time.time())
                with LOCK, (LDIRS.get(str(r.get('v', '1')), LDIR) / 'marks.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(e) + chr(10))
                return self.send({'ok': True})
            if urlparse(self.path).path != '/api/mark':
                return self.send({'error': 'not found'}, 404)
            e = dict(family=int(r['family']), log=str(r['log']), y=int(r['y']), wrong=bool(r['wrong']),
                     note=str(r.get('note', ''))[:500], t=time.time())
            with LOCK, (DIR / 'marks.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(e) + chr(10))
            return self.send({'ok': True})
        except Exception:
            return self.send({'error': 'bad request'}, 400)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8771)
    a = ap.parse_args()
    print(f'Families review: http://127.0.0.1:{a.port}')
    ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
