"""Headless check: source filters, all four tabs, dataset sheets and phone width."""
import sys
from playwright.sync_api import sync_playwright

out = sys.argv[1] if len(sys.argv) > 1 else 'data/genlab/scratch'
errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    request = p.request.new_context()
    for name, size in (('desk', (1440, 1000)), ('phone', (390, 844))):
        # New fleet rounds keep arriving during the check. Hold one real state
        # response per viewport so polling cannot replace eager images with new
        # lazy elements halfway through the all-images assertion.
        snapshot = request.get('http://127.0.0.1:8781/state.json').json()
        page = b.new_page(viewport=dict(width=size[0], height=size[1]))
        page.route('**/state.json', lambda route: route.fulfill(json=snapshot))
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('console', lambda m: m.type == 'error' and errors.append(m.text))
        page.on('response', lambda r: r.status >= 400 and errors.append(f'HTTP {r.status} {r.url}: {r.text()[:200]}'))
        page.goto('http://127.0.0.1:8781/')
        page.evaluate('localStorage.clear()')
        page.reload()
        page.wait_for_selector('#gen .card')
        page.wait_for_load_state('networkidle')
        print(name, 'page loaded; checking all pattern images', flush=True)
        page.evaluate("document.querySelectorAll('#gen img').forEach(i => i.loading = 'eager')")
        page.wait_for_function("[...document.querySelectorAll('#gen img')].every(i => i.complete)", timeout=120000)
        pattern_broken = page.evaluate("[...document.querySelectorAll('#gen img')].filter(i => i.naturalWidth === 0).length")
        assert pattern_broken == 0, (name, 'broken pattern images', pattern_broken)
        previews = page.evaluate("""[...document.querySelectorAll('#gen .card')].map(c => {
            const strip = c.querySelector('.vote')?.parentElement.querySelector('img');
            const preview = c.querySelector('img[alt^="Learned repeat"]');
            return preview && strip ? {width: preview.naturalWidth, strip: strip.naturalWidth,
                height: preview.naturalHeight, display: preview.getBoundingClientRect().width,
                stripDisplay: strip.getBoundingClientRect().width} : null;
        }).filter(Boolean)""")
        assert all(p['width'] == p['strip'] and 48 <= p['height'] <= 240 and
                   p['display'] == p['stripDisplay'] for p in previews), (name, 'repeat preview scale')
        print(name, 'repeat preview scale PASS', len(previews), 'native and display widths match')
        counts = {}
        for source in ('human', 'machine-verified', 'agent-made', 'borrowed', 'swatch', 'unlisted', 'all'):
            page.select_option('#source', source)
            result = page.evaluate("""source => {
                const cards = [...document.querySelectorAll('#gen .card')];
                return {expected: cards.filter(c => source === 'all' || c.dataset.source === source).length,
                        visible: cards.filter(c => !c.classList.contains('hide')).length};
            }""", source)
            assert result['visible'] == result['expected'], (name, source, result)
            counts[source] = result['visible']
        print(name, 'source filter counts', counts)
        assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 2'), (name, 'patterns overflow')
        widths = page.evaluate("[...document.querySelectorAll('#gen .strip img')].map(i=>Math.round(i.getBoundingClientRect().width))")
        print(name, 'strip widths', sorted(set(w for w in widths if w)))
        page.screenshot(path=f'{out}/{name}-gen.png', full_page=False)
        # Element screenshots otherwise capture the sticky header over the
        # first card's title when a source filter shortens the document.
        page.evaluate("document.querySelector('header').style.position = 'static'")
        for source, limit in (('agent-made', 3), ('unlisted', 2)):
            page.select_option('#source', source)
            cards = page.locator(f'#gen .card[data-source="{source}"]:not(.hide)')
            count = min(limit, cards.count())
            for i in range(count):
                cards.nth(i).scroll_into_view_if_needed()
                cards.nth(i).screenshot(path=f'{out}/{name}-{source}-{i+1}.png')
            print(name, source, 'cards opened and captured', count)
        page.select_option('#source', 'all')
        page.evaluate("document.querySelector('header').style.removeProperty('position')")
        for tab in ('cols', 'acc', 'datasets'):
            page.click(f'nav button[data-tab={tab}]')
            if tab == 'acc':
                assert page.get_by_role('heading', name='Reviewer accuracy', exact=True).is_visible()
            if tab == 'datasets':
                page.wait_for_selector('#datasets h2')
                assert page.get_by_role('heading', name='Versioned datasets', exact=True).is_visible()
                assert page.locator('#datasets a[href$=".zip"]').count() > 0
                assert page.locator('#datasets svg').count() == 1
            page.wait_for_load_state('networkidle')
            for y in range(0, 6000, 600):   # let lazy thumbnails load
                page.evaluate(f'window.scrollTo(0,{y})'); page.wait_for_timeout(150)
            page.wait_for_load_state('networkidle'); page.evaluate('window.scrollTo(0,0)')
            assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 2'), (name, tab, 'overflow')
            page.screenshot(path=f'{out}/{name}-{tab}.png', full_page=name == 'desk')
            print(name, tab, 'PASS fits width and captured', flush=True)
        wide = page.evaluate('document.documentElement.scrollWidth > window.innerWidth + 2')
        print(name, 'horizontal overflow' if wide else 'fits width')
        broken = page.evaluate("[...document.images].filter(i=>i.complete&&i.naturalWidth===0&&i.getBoundingClientRect().width>0).length")
        print(name, 'broken images', broken)
        if broken:
            errors.append(f'{name}: {broken} broken images')
        page.close()
    b.close()
    request.dispose()
print('errors', errors)
if errors:
    sys.exit(1)
