"""Live Progress checks and desktop/phone screenshots (Playwright Chromium)."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import urlopen

sys.dont_write_bytecode = True
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8782'


def fetch(path):
    with urlopen(urljoin(BASE, path), timeout=10) as response:
        return response.read()


snapshot = json.loads(fetch('/progress.json'))
assert snapshot['current']['schema'] == 2, snapshot.get('error')
assert len(snapshot['current']['streams']) == 8
first_at = snapshot['current']['at']
started = time.monotonic()
with ThreadPoolExecutor(max_workers=8) as executor:
    readings = list(executor.map(fetch, ['/progress.json'] * 16))
elapsed = time.monotonic()-started
assert all(json.loads(r)['current']['at'] == first_at for r in readings)
assert elapsed < 3, elapsed
print(f'PASS: 16 cached Progress reads in {elapsed:.3f}s; no per-request tracker run')


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.sources.extend(v for k, v in attrs if k == 'src')


parser = Images()
parser.feed(fetch('/assembler/report.html').decode('utf-8'))
assert parser.sources
for source in parser.sources:
    blob = fetch(urljoin('/assembler/report.html', source))
    assert blob, source
print(f'PASS: assembler report and {len(parser.sources)} relative images served read-only')
for route in ('/assembler/%2e%2e/tracker.py', '/assembler/stats.json', '/assembler/%2e%2e/%2e%2e/normalized/current.json'):
    try:
        fetch(route)
        raise AssertionError('Non-report file exposed')
    except HTTPError as error:
        assert error.code == 400
print('PASS: assembler traversal and non-report files rejected')

errors = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    for width, height in ((1280, 1000), (390, 844)):
        page = browser.new_page(viewport=dict(width=width, height=height), device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(BASE + '/#progress')
        page.wait_for_selector('.stream-row')
        page.wait_for_function('data !== null')
        page.wait_for_function("document.getElementById('overall-good').textContent.includes('%')")
        assert page.locator('.stream-row').count() == 8
        assert page.locator('#fleet-view').is_hidden()
        assert page.locator('#trend svg').count() == 1
        assert page.locator('#label-next .label-item').count() == len(snapshot['label_next'])
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), width
        page.evaluate('scrollTo(0,0)')
        page.screenshot(path=str(HERE / f'progress-{width}.png'), full_page=True)
        page.screenshot(path=str(HERE / f'progress-{width}-top.png'))
        page.locator('[data-stream="dataset_loop"]').scroll_into_view_if_needed()
        page.screenshot(path=str(HERE / f'progress-{width}-gates.png'))
        page.locator('#label-next').scroll_into_view_if_needed()
        page.screenshot(path=str(HERE / f'progress-{width}-labels.png'))
        # Trend controls and local Dataset tab must work without library or lab writes.
        page.locator('#trend-controls input[value="assembler"]').uncheck()
        assert not page.locator('#trend-controls input[value="assembler"]').is_checked()
        page.locator('#dataset-tab').click()
        page.locator('#dataset').wait_for(state='visible')
        assert page.locator('#dataset').is_visible()
        assert page.locator('#dataset-body a').count() >= 3
        page.locator('#progress-tab').click()
        page.locator('#progress').wait_for(state='visible')
        assert page.locator('#progress').is_visible()
        # Refresh failures keep existing gates visible.
        page.route('**/progress.json', lambda route: route.fulfill(status=503, body='delayed'))
        page.evaluate('refreshProgress()')
        page.wait_for_function("document.getElementById('progress-error').textContent.includes('delayed')")
        assert page.locator('.stream-row').count() == 8
        page.unroute('**/progress.json')
        print(f'PASS: Progress at {width}px; eight streams, SVG chart, labels, tabs, outage retention, no overflow')
        page.close()
    browser.close()
assert not errors, errors
print('PASS: Progress has no browser JavaScript errors')
(HERE / 'progress-checked.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
