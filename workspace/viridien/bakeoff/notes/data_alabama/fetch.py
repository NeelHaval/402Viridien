"""Read public source files. Run with Python -I from the repository root."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'external_untrusted' / 'alabama'
AUDIT = BASE / 'access.jsonl'
MAX_BYTES = 40 * 1024 * 1024


def fetch(url, relative, limit=MAX_BYTES, method='GET', payload=None):
    """Save at most 40 MiB. Use only public Alabama state source hosts."""
    host = urlparse(url).hostname or ''
    if not (host.endswith('.state.al.us') or host.endswith('.alabama.gov')):
        raise ValueError('Source host is not an Alabama state host')
    target = (BASE / relative).resolve()
    if BASE.resolve() not in target.parents:
        raise ValueError('Path leaves the source folder')
    row = {'url': url, 'path': target.relative_to(ROOT).as_posix(),
           'accessed_utc': datetime.now(timezone.utc).isoformat(),
           'user_agent': 'Mozilla/5.0', 'authentication': 'none',
           'method': method, 'payload': payload}
    start = time.monotonic()
    try:
        with requests.request(method, url, json=payload, headers={'User-Agent': 'Mozilla/5.0'},
                          timeout=(15, 60), stream=True) as response:
            row.update(status=response.status_code, final_url=response.url,
                       content_type=response.headers.get('Content-Type'),
                       etag=response.headers.get('ETag'),
                       last_modified=response.headers.get('Last-Modified'))
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > min(limit, MAX_BYTES):
                    raise ValueError('Source file exceeds the size limit')
                if time.monotonic() - start > 180:
                    raise TimeoutError('Source transfer exceeds three minutes')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            row.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except Exception as exc:
        row['error'] = str(exc)
    with AUDIT.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
    return row


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('relative')
    args = parser.parse_args()
    print(json.dumps(fetch(args.url, args.relative), indent=2))
