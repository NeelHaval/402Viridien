"""Local read-only checks and browser captures; outputs stay beside this file."""
import io
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from urllib.error import HTTPError

sys.dont_write_bytecode = True
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8782'


def fetch(url):
    with urlopen(url if url.startswith('http') else BASE + url, timeout=30) as r:
        return r.read(), r.headers.get_content_type()


s = json.loads(fetch('/board.json')[0])
assert s['total'] == sum(s['counts'].values()) == 861
assert s['document_count'] == 51
assert s['training_total'] == sum(s['training_counts'].values())
assert len({e['key'] for e in s['entries']}) == 861
print('PASS: 861 unique legend entries, 51 documents, state and training totals')
lab = next(e for e in s['entries'] if e['state'] == 'In the lab' and e['strips'])
proposal = next(e for e in s['entries'] if e['tile'])
for label, url in [('lab strip', lab['strips'][0]), ('proposal strip', proposal['tile']), ('real crop', lab['real'][0])]:
    blob, mime = fetch(url)
    assert mime == 'image/png' and blob.startswith(b'\x89PNG\r\n\x1a\n')
    with Image.open(io.BytesIO(blob)) as im:
        im.verify()
    with Image.open(io.BytesIO(blob)) as im:
        size = im.size
    if label == 'proposal strip':
        assert size == (64, 240)
    print('PASS:', label, mime, size, url)

for route in ['/sheet?doc='+proposal['doc_id']+'&f=../../../../board/board.py', '/tile?doc=../../..&f=board.py']:
    try:
        fetch(route)
        raise AssertionError('Traversal allowed')
    except HTTPError as e:
        assert e.code == 400
print('PASS: proposal image traversal rejected')

errors = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    for name, size in [('desktop', (1440, 1080)), ('phone', (390, 844)), ('narrow', (320, 720))]:
        page = browser.new_page(viewport=dict(width=size[0], height=size[1]))
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('response', lambda r: errors.append('HTTP '+str(r.status)+' '+r.url) if r.status >= 400 else None)
        page.goto(BASE)
        page.wait_for_selector('#done-body .card')
        page.wait_for_timeout(700)
        assert page.locator('#done-body .card').count() + page.locator('#todo-body .todo-row').count() == 861
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), (name, 'overflow')
        page.screenshot(path=str(HERE / f'shot-{name}.png'))
        first = page.locator('#done-body .card').first
        first.scroll_into_view_if_needed()
        first.locator('img').evaluate_all("imgs=>imgs.forEach(i=>i.loading='eager')")
        page.wait_for_function("[...document.querySelector('#done-body .card').querySelectorAll('img')].every(i=>i.complete&&i.naturalWidth>0)")
        first.screenshot(path=str(HERE / f'shot-{name}-card.png'))
        widths = first.locator('img').evaluate_all('imgs=>imgs.map(i=>i.getBoundingClientRect().width)')
        assert all(w == (88 if size[0] > 640 else 64) for w in widths), widths
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.select_option('#state', 'Failed')
        assert page.locator('#done-body .card').count() == 0
        assert page.locator('#todo-body .todo-row').count() == page.evaluate("data.counts['Failed']")
        page.select_option('#state', '')
        page.select_option('#source', 'human')
        assert page.locator('#done-body .card').count() == page.evaluate("data.entries.filter(e=>e.source==='human').length")
        page.select_option('#source', '')
        page.fill('#search', 'Shale')
        assert page.locator('#done-body .card').count() + page.locator('#todo-body .todo-row').count() == page.evaluate("data.entries.filter(e=>(e.label+' '+e.description+' '+e.looks_like).toLowerCase().includes('shale')).length")
        page.select_option('#document', s['documents'][0]['doc_id'])
        page.fill('#search', '')
        page.evaluate('window.scrollTo(0,500)')
        before = page.evaluate('({y:scrollY,doc:document.getElementById("document").value})')
        page.evaluate('refresh()')
        after = page.evaluate('({y:scrollY,doc:document.getElementById("document").value})')
        assert before == after, (name, 'refresh moved scroll/filters', before, after)
        changed = page.evaluate('JSON.parse(JSON.stringify(data))')
        changed['entries'][0]['note'] += ' Updated reviewer note.'
        page.route('**/board.json', lambda route: route.fulfill(json=changed))
        page.evaluate('refresh()')
        after = page.evaluate('({y:scrollY,doc:document.getElementById("document").value})')
        assert before == after, (name, 'data change moved scroll/filters', before, after)
        page.unroute('**/board.json')
        page.select_option('#document', '')
        page.locator('#unlisted img').evaluate_all("imgs=>imgs.forEach(i=>i.loading='eager')")
        page.wait_for_function("[...document.querySelectorAll('#unlisted img')].every(i=>i.complete&&i.naturalWidth>0)")
        page.evaluate("document.getElementById('unlisted').scrollIntoView({block:'start'})")
        page.wait_for_timeout(200)
        page.screenshot(path=str(HERE / f'shot-{name}-unlisted.png'))
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        print('PASS:', name, 'fits width; strips', widths, '; filters and refresh preserve state')
        page.close()
    browser.close()
assert not errors, errors
print('PASS: no browser JS errors or HTTP failures')
(HERE / 'checked-state.json').write_text(json.dumps(s, indent=2), encoding='utf-8')
