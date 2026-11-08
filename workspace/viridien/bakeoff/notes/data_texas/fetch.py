"""Fetch public source data. Run with python -I. Never execute source data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[3] / 'external_untrusted' / 'texas'

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.words = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1
        for k, v in attrs:
            if k in ('href', 'src', 'action') and v:
                self.links.append(v)
    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)
    def handle_data(self, data):
        if not self.skip and data.strip():
            self.words.append(data.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('slug')
    ap.add_argument('--limit-mb', type=int, default=30)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    if not args.slug.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use a plain source slug')
    folder = ROOT / args.slug
    folder.mkdir(parents=True, exist_ok=True)
    host = urllib.parse.urlsplit(args.url).hostname
    if urllib.parse.urlsplit(args.url).scheme != 'https':
        raise ValueError('Use HTTPS')
    clock_file = ROOT / 'request_times.json'
    clocks = json.loads(clock_file.read_text()) if clock_file.exists() else {}
    time.sleep(max(0, 1.1 - (time.time() - clocks.get(host, 0))))
    clocks[host] = time.time()
    clock_file.write_text(json.dumps(clocks), encoding='utf-8')
    stamp = time.time()
    receipt = {'url': args.url, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(stamp)),
               'user_agent': 'Mozilla/5.0 (research crawler)'}
    try:
        req = urllib.request.Request(args.url, headers={'User-Agent': receipt['user_agent']})
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read(args.limit_mb * 1024 * 1024 + 1)
            if len(data) > args.limit_mb * 1024 * 1024:
                raise ValueError('Source exceeds download cap')
            receipt.update(status=response.status, final_url=response.url,
                           content_type=response.headers.get('Content-Type'), headers=dict(response.headers))
    except urllib.error.HTTPError as exc:
        data = exc.read(1024 * 1024)
        receipt.update(status=exc.code, error=str(exc), headers=dict(exc.headers))
    except Exception as exc:
        data = b''
        receipt.update(error=str(exc))
    suffix = '.pdf' if data.startswith(b'%PDF') else '.html'
    if data[:4] in (b'II*\x00', b'MM\x00*'):
        suffix = '.tif'
    if data.startswith(b'PK\x03\x04'):
        suffix = '.xlsx' if urllib.parse.urlsplit(args.url).path.endswith('.xlsx') else '.zip'
    if data.startswith(b'\xd0\xcf\x11\xe0'):
        suffix = '.xls'
    name = hashlib.sha256(args.url.encode()).hexdigest()[:12]
    dest = folder / (name + suffix)
    dest.write_bytes(data)
    receipt.update(path=str(dest.relative_to(ROOT)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (folder / (name + '.receipt.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt if not args.quiet else {k: receipt.get(k) for k in ('url', 'status', 'error', 'path', 'bytes', 'sha256')}, indent=2))
    if suffix == '.html':
        p = Page()
        p.feed(data.decode('utf-8', errors='replace'))
        parsed = {'text': '\n'.join(p.words), 'links': list(dict.fromkeys(urllib.parse.urljoin(receipt.get('final_url', args.url), v) for v in p.links))}
        (folder / (name + '.parsed.json')).write_text(json.dumps(parsed, indent=2), encoding='utf-8')
        if not args.quiet:
            print(json.dumps(parsed, indent=2)[:23000])

if __name__ == '__main__':
    main()
