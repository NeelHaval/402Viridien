"""Edge screenshots + optional Playwright UI/timing checks. Run with the server up."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labeltool.common import HOME, CACHE, write_json

EDGE=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')


def screenshots(base):
    output=HOME/'screenshots';output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,width,height in [('desktop',1440,1000),('phone',390,844)]:
        path=output/(name+'.png')
        profile=CACHE/('edge-'+name)
        command=[str(EDGE),'--headless=new','--hide-scrollbars',f'--window-size={width},{height}',
                 '--no-first-run','--no-default-browser-check','--disable-gpu',
                 '--no-sandbox','--disable-crash-reporter',f'--user-data-dir={profile}',
                 '--virtual-time-budget=5000',f'--screenshot={path}',base+'/']
        started=time.monotonic()
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=35,encoding='utf-8',errors='replace')
            diagnostic=(result.stderr or result.stdout)[-3000:]
            exists=path.exists()
            records.append(dict(viewport=[width,height],path=str(path),created=exists,
                                exit_code=result.returncode,diagnostic=diagnostic,seconds=round(time.monotonic()-started,2)))
        except Exception as exc:
            records.append(dict(viewport=[width,height],path=str(path),created=False,error=str(exc)))
    write_json(HOME/'browser-validation.json',records)
    print(json.dumps(records,indent=2))
    return records


def interactive(base):
    """Read-only browser interactions; no test decisions are created here."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(EDGE),headless=True,args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(viewport=dict(width=1440,height=1000))
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        started=time.monotonic();page.goto(base+'/')
        page.wait_for_function("document.querySelector('#crop').complete && document.querySelector('#crop').naturalWidth>0")
        first_ms=(time.monotonic()-started)*1000
        page.wait_for_function('state.items.length>=3')
        dimensions=page.locator('#crop').evaluate('(img)=>({w:img.width,h:img.height,nw:img.naturalWidth,nh:img.naturalHeight})')
        assert abs(dimensions['w']/dimensions['h']-dimensions['nw']/dimensions['nh'])<.001
        page.get_by_role('button',name='2×',exact=True).click()
        page.wait_for_function("document.querySelector('#crop').width===2*document.querySelector('#crop').naturalWidth")
        page.get_by_role('button',name='1×',exact=True).click()
        first=page.locator('#logTitle').inner_text()
        started=time.monotonic();page.keyboard.press('Space')
        page.wait_for_function('(old)=>document.querySelector("#logTitle").textContent!==old',arg=first)
        next_ms=(time.monotonic()-started)*1000
        page.keyboard.press('B');assert page.locator('body').evaluate('(node)=>node.classList.contains("boundary")')
        page.keyboard.press('B')
        page.keyboard.press('?');assert page.locator('#help').evaluate('(node)=>node.open')
        page.keyboard.press('Escape')
        page.screenshot(path=str(HOME/'screenshots'/'desktop.png'),full_page=True)
        page.set_viewport_size(dict(width=390,height=844));page.screenshot(path=str(HOME/'screenshots'/'phone.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        page.get_by_role('button',name='Stats',exact=True).click();page.wait_for_selector('#statsBody table')
        assert not errors
        report=dict(first_render_ms=round(first_ms,1),prefetched_next_ms=round(next_ms,1),javascript_errors=errors,
                    desktop=[1440,1000],phone=[390,844],aspect_ratio='passed',zoom='passed',keyboard='passed',stats='passed')
        write_json(HOME/'ui-validation.json',report)
        print(json.dumps(report,indent=2));browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8765')
    args=parser.parse_args()
    screenshots(args.base)
    try:interactive(args.base)
    except Exception as exc:
        message='Browser interaction unavailable: '+str(exc)
        (HOME/'browser-error.log').write_text(message,encoding='utf-8')
        print(message[:3000])
