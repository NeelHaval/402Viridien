from pathlib import Path
from playwright.sync_api import sync_playwright

out=Path(__file__).resolve().parent
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch()
    for name,width,height in [('desktop',1440,1000),('phone',390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height})
        page.add_init_script("localStorage.tab='datasets'")
        page.on('pageerror',lambda err:errors.append(str(err)))
        page.goto('http://127.0.0.1:8781/')
        page.click('button[data-tab="datasets"]')
        page.get_by_role('heading',name='Closed-set within-document legend recognition',exact=True).wait_for()
        page.get_by_role('heading',name='Cross-log name generalisation',exact=True).wait_for()
        body=page.locator('#datasets').inner_text()
        assert 'Chance % / macro chance %' in body and 'Chance %' in body
        evaluation=page.request.get('http://127.0.0.1:8781/api/datasets').json()['evaluation']
        assert f"Latest evaluation: {evaluation['version']}" in body
        for variants in evaluation['closed_set_within_document'].values():
            for row in variants.values():
                percent=f"{row['accuracy']:.1f}".rstrip('0').rstrip('.')+'%'
                assert percent in body, percent
        assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth+2')
        page.locator('#datasets h2').first.scroll_into_view_if_needed()
        page.screenshot(path=str(out/f'{name}-dataset-overview.png'))
        page.get_by_role('heading',name='Closed-set within-document legend recognition',exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(out/f'{name}-dataset-scores.png'))
        print('PASS Dataset tab:',name,'closed-set and cross-log scores, chance levels; no horizontal overflow',flush=True)
        page.close()
    browser.close()
assert not errors,errors
print('PASS browser JavaScript errors:',errors)
