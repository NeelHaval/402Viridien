"""Actual Chromium-rendered UI checks when Windows blocks the Edge launcher.

This fallback is explicitly identified as Qt WebEngine, never as Edge.
"""
import json
import argparse
import os
from pathlib import Path
import sys
import time
import uuid
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labeltool.common import HOME, CACHE, write_json, read_json

os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--no-sandbox --disable-gpu --single-process --disable-dev-shm-usage'
os.environ['QTWEBENGINE_DISABLE_SANDBOX']='1'
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',default='http://127.0.0.1:8765')
args=parser.parse_args()
app=QApplication([])
profile=QWebEngineProfile()  # off-the-record: no external disk caches
profile.setCachePath(str(CACHE/'qt-cache'))
profile.setPersistentStoragePath(str(CACHE/'qt-storage'))
view=QWebEngineView()
page=QWebEnginePage(profile,view)
view.setPage(page)
view.resize(1440,1000)
view.show()
output=HOME/'screenshots';output.mkdir(parents=True,exist_ok=True)
results=dict(engine='Qt WebEngine Chromium (Edge launcher blocked)',checks={},screenshots=[])
started=time.monotonic()
phase='initial';next_started=None;settled=False
save_started=None;save_item=None;save_done=None


def inspect():
    script='''(()=>({ready:!!state.current && document.querySelector('#crop').complete && document.querySelector('#crop').naturalWidth>0,
       id:state.current?.id, buffered:state.items.length,
       progress:!!state.progress,swatches:[...document.querySelectorAll('#tiles img')].every(img=>img.complete && img.naturalWidth>0),
       prefetched:state.items.slice(0,3).every(item=>state.images.get(item.id)?.complete),nextPaint:window.__labeltoolNextPaintMs,
       log:state.current?.log,done:state.progress?.done,busy:state.busy,top:state.top,bottom:state.bottom,
       handleMatches:parseFloat(document.querySelector('#topHandle').style.top)===(state.top-state.current.context_top)*state.zoom,
       aspect:document.querySelector('#crop').width/document.querySelector('#crop').height,
       nativeAspect:document.querySelector('#crop').naturalWidth/document.querySelector('#crop').naturalHeight,
       xOverflow:document.documentElement.scrollWidth>window.innerWidth,
       boundary:document.body.classList.contains('boundary'),help:document.querySelector('#help').open,
       stats:!!document.querySelector('#statsBody table')}))()'''
    page.runJavaScript('JSON.stringify('+script+')',handle)


def handle(value):
    global phase,next_started,settled,save_started,save_item,save_done
    if isinstance(value,str):
        try:value=json.loads(value)
        except json.JSONDecodeError:value=None
    if not isinstance(value,dict) or not value.get('ready'):
        QTimer.singleShot(30,inspect);return
    if phase=='initial':
        results.setdefault('first_render_ms',round((time.monotonic()-started)*1000,1))
        results['checks']['aspect']=abs(value['aspect']-value['nativeAspect'])<.001
        results['checks']['desktop_no_overflow']=not value['xOverflow']
        if value['buffered']<3 or not(value['progress'] and value['swatches'] and value['prefetched']):QTimer.singleShot(30,inspect);return
        results.setdefault('first_complete_item_ms',round((time.monotonic()-started)*1000,1))
        if not settled:
            settled=True;QTimer.singleShot(180,inspect);return
        results['initial_id']=value['id']
        path=output/'desktop-qt.png';view.grab().save(str(path));results['screenshots'].append(str(path))
        phase='next';next_started=time.monotonic()
        page.runJavaScript("window.__labeltoolNextStarted=performance.now();document.dispatchEvent(new KeyboardEvent('keydown',{key:' ',bubbles:true}));requestAnimationFrame(()=>requestAnimationFrame(()=>window.__labeltoolNextPaintMs=performance.now()-window.__labeltoolNextStarted))");QTimer.singleShot(5,inspect)
    elif phase=='next':
        if value['id']==results['initial_id']:QTimer.singleShot(5,inspect);return
        if not value.get('nextPaint'):QTimer.singleShot(5,inspect);return
        results['prefetched_next_ms']=round(value['nextPaint'],1)
        phase='boundary';page.runJavaScript("document.dispatchEvent(new KeyboardEvent('keydown',{key:'B',bubbles:true}))");QTimer.singleShot(30,inspect)
    elif phase=='boundary':
        results['checks']['boundary_mode']=value['boundary']
        phase='boundary_fields'
        page.runJavaScript("document.querySelector('#topRow').value=state.current.top_y+1;document.querySelector('#topRow').dispatchEvent(new Event('change'))")
        QTimer.singleShot(30,inspect)
    elif phase=='boundary_fields':
        results['checks']['boundary_field_updates_handle']=value['handleMatches']
        phase='zoom';page.runJavaScript("document.querySelector('#topRow').value=state.current.top_y;document.querySelector('#topRow').dispatchEvent(new Event('change'));toggleBoundary();setZoom(2,true)");QTimer.singleShot(30,inspect)
    elif phase=='zoom':
        results['checks']['zoom_aspect']=abs(value['aspect']-value['nativeAspect'])<.001
        phase='phone';page.runJavaScript('setZoom(1,true)');view.resize(390,844);QTimer.singleShot(300,inspect)
    elif phase=='phone':
        if not value['swatches']:QTimer.singleShot(50,inspect);return
        results['checks']['phone_no_overflow']=not value['xOverflow']
        path=output/'phone-qt.png';view.grab().save(str(path));results['screenshots'].append(str(path))
        event_id='labeltool-e2e-ui-'+uuid.uuid4().hex
        registry_path=CACHE/'api-test-events.json'
        registry=read_json(registry_path,dict(event_ids=[],logs=[],original_csvs=[]))
        registry['event_ids'].append(event_id);registry['logs'].append(value['log'])
        write_json(registry_path,registry)
        save_item=value['id'];save_done=value['done'];save_started=time.monotonic();phase='save'
        # Tag only this test's UI save so cleanup can preserve every real label.
        script="""window.__originalFetch=window.fetch;
        window.fetch=(url,options)=>{if(url==='/api/decision'){const body=JSON.parse(options.body);body.request_id=EVENT;options={...options,body:JSON.stringify(body)}}return window.__originalFetch(url,options)};
        document.dispatchEvent(new KeyboardEvent('keydown',{key:'1',bubbles:true}));
        document.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));""".replace('EVENT',json.dumps(event_id))
        page.runJavaScript(script);QTimer.singleShot(30,inspect)
    elif phase=='save':
        if value['id']==save_item or value['busy']:QTimer.singleShot(20,inspect);return
        results['checks']['keyboard_pick_enter_saves_and_advances']=value['done']==save_done+1
        results['save_and_advance_ms']=round((time.monotonic()-save_started)*1000,1)
        phase='undo';page.runJavaScript("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Z',bubbles:true}))");QTimer.singleShot(30,inspect)
    elif phase=='undo':
        if value['id']!=save_item or value['busy']:QTimer.singleShot(30,inspect);return
        results['checks']['keyboard_undo_restores_item']=value['done']==save_done
        phase='stats';page.runJavaScript("window.fetch=window.__originalFetch;document.querySelector('#statsButton').click()");QTimer.singleShot(150,inspect)
    elif phase=='stats':
        if not value['stats']:QTimer.singleShot(100,inspect);return
        results['checks']['stats_table']=True
        write_json(HOME/'qt-ui-validation.json',results)
        print(json.dumps(results,indent=2),flush=True)
        app.quit()


def timeout():
    results['error']='Timed out in phase '+phase
    write_json(HOME/'qt-ui-validation.json',results)
    print(json.dumps(results),flush=True)
    app.quit()


view.load(QUrl(args.base+'/'))
QTimer.singleShot(50,inspect)
QTimer.singleShot(45000,timeout)
app.exec()
raise SystemExit(1 if results.get('error') or not all(results['checks'].values()) else 0)
