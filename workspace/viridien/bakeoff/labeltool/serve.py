"""Local stdlib HTTP server. Run: python -X utf8 -B labeltool/serve.py."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import random
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode, urlparse

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labeltool.common import (ROOT, HOME, DATA, CACHE, HUMAN, GOLD, read_json, normalize,
                              external_sources, suggestions_for)
from labeltool.prefill import best_guess
from labeltool.store import Store
from labeltool.score_human import score_human, compare, source_paths, wilson


def round_robin_order(items):
    """Stable rounds: filtering completed/prefetched items must not restart a round."""
    groups=defaultdict(list)
    for item in items:
        if not item.get('truth_only'):groups[(item['priority'],item['log'])].append(item)
    order={}
    for (priority,name),rows in groups.items():
        for index,item in enumerate(sorted(rows,key=lambda r:r['top_y'])):
            order[item['id']]=(priority,index,name)
    return order


class Application:
    def __init__(self, human_dir=HUMAN):
        path=DATA/'queue.jsonl'
        if not path.exists(): raise RuntimeError('Build the queue first: python -X utf8 -B labeltool/build_queue.py')
        self.items=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        self.by_id={r['id']:r for r in self.items}
        self.review_by_log=defaultdict(list)
        for item in self.items:
            if not item['truth_only']:self.review_by_log[item['log']].append(item)
        names={r['log'] for r in self.items}
        self.logs={name:read_json(DATA/'logs'/(name+'.json'),{}) for name in sorted(names)}
        self.store=Store(human_dir)
        self.lock=threading.RLock()
        self.image_locks=defaultdict(threading.Lock)
        self.random=random.Random()
        self.spot_order={r['id']:self.random.random() for r in self.items}
        self.pool=ThreadPoolExecutor(max_workers=2)
        self.stats_cache=None
        self.stats_at=0
        self.external_signature=None
        self.refresh_sources()
        # Warm only the first screen; page requests never wait for the whole dataset.
        initial=self.select('uncertain','',set(),4)
        for item in initial: self.pool.submit(self.image_file,item['log'],'crop',item['context_top'],item['context_bottom'])

    def refresh_sources(self):
        signature=[]
        folder=ROOT/'labelling'/'out'
        for pattern in ('*/consensus.csv','*/S.csv','*/T.csv','*/vocab.json'):
            for p in folder.glob(pattern):
                stat=p.stat();signature.append((str(p),stat.st_mtime_ns,stat.st_size))
        signature=sorted(signature)
        with self.lock:
            if signature==self.external_signature:return False
            for name in self.logs:
                sources=external_sources(name)
                for item in (r for r in self.items if r['log']==name):
                    item['suggestions']=[s for s in item['suggestions'] if not s['source'].startswith('labelling/')]
                    item['suggestions'].extend(suggestions_for(sources,item['top_y'],item['bottom_y']))
                    item.update(best_guess(item))
            self.external_signature=signature
            self.uncertain_order=round_robin_order(self.items)
            self.disputed_order=round_robin_order([r for r in self.items if r['disputed']])
            self.stats_cache=None
        return True

    def public_item(self,item):
        meta=self.logs[item['log']]
        box=meta['box']
        # Context includes 150 native rows on each side, never stretches pixels.
        top=max(box[1],item['top_y']-150)
        bottom=min(box[3],item['bottom_y']+150)
        query=urlencode(dict(log=item['log'],kind='crop',y0=top,y1=bottom))
        return dict(item,context_top=top,context_bottom=bottom,width=box[2]-box[0],
                    image_url='/api/image?'+query,meta=meta)

    def select(self,mode,name,exclude,limit):
        with self.lock:
            done=self.store.completed()
            omitted=done|exclude
            candidates=[r for r in self.items if not r['truth_only'] and r['id'] not in omitted]
            if mode=='walk':
                candidates=sorted((r for r in candidates if r['log']==name),key=lambda r:r['top_y'])
            elif mode=='spot':
                candidates=sorted((r for r in candidates if r['tier']=='high'),key=lambda r:self.spot_order[r['id']])
            elif mode in ('uncertain','disputed'):
                if mode=='disputed': candidates=[r for r in candidates if r['disputed']]
                order=self.disputed_order if mode=='disputed' else self.uncertain_order
                candidates=heapq.nsmallest(limit,candidates,key=lambda r:order[r['id']])
            else: raise ValueError('Unknown mode')
            return [self.public_item(r) for r in candidates[:limit]]

    def progress(self):
        active={e['item_id']:e for e in self.store.active()}
        counts=Counter(e['log'] for e in active.values())
        logs={};families={}
        for name,meta in self.logs.items():
            rec=dict(log=name,family=meta['family'],total=len(self.review_by_log[name]),done=counts[name],truth_only=name==GOLD)
            logs[name]=rec
            fam=families.setdefault(meta['family'],dict(total=0,done=0,logs=0))
            fam['total']+=rec['total'];fam['done']+=rec['done'];fam['logs']+=1
        spot=[e for e in active.values() if e['mode']=='spot' and e['tier']=='high']
        right=sum(normalize(e['label'])==normalize(e['suggestion_shown']) and not e['boundary_edited'] for e in spot)
        return dict(logs=logs,families=families,done=len(active),total=sum(v['total'] for v in logs.values()),
                    spot=dict(n=len(spot),right=right,rate=right/len(spot) if spot else None,ci95=wilson(right,len(spot))))

    def stats(self):
        with self.lock:
            if self.stats_cache is not None and time.monotonic()-self.stats_at<10:return self.stats_cache
            table=score_human(human_dir=self.store.directory)
            gold=[]
            from labeltool.common import load_rows
            truth=load_rows(ROOT/'gt'/'gold'/(GOLD+'.csv'))
            for source,logs in source_paths().items():
                if GOLD not in logs:continue
                result=compare(load_rows(logs[GOLD]),truth)
                gold.append(dict(source=source,**result,row_accuracy=result['correct_rows']/result['n_rows'] if result['n_rows'] else None))
            self.stats_cache=dict(table=table,gold=gold,progress=self.progress())
            self.stats_at=time.monotonic()
            return self.stats_cache

    def image_file(self,name,kind,top=None,bottom=None):
        if name not in self.logs: raise ValueError('Unknown log')
        assets=CACHE/'assets'/name
        if kind in ('legend','strip'):return assets/(kind+'.png')
        if kind.startswith('swatch_'):
            index=int(kind.removeprefix('swatch_'))
            if index not in {t['id'] for t in self.logs[name]['tiles']}: raise ValueError('Unknown swatch')
            return assets/(kind+'.png')
        if kind!='crop':raise ValueError('Unknown image type')
        box=self.logs[name]['box']
        top=max(box[1],int(top));bottom=min(box[3],int(bottom))
        if not top<bottom or bottom-top>10000: raise ValueError('Crop must contain 1–10000 native rows')
        key=hashlib.sha1(f'{name}:{top}:{bottom}'.encode()).hexdigest()
        path=CACHE/'crops'/(key+'.png')
        with self.image_locks[key]:
            if not path.exists():
                from harness import io
                from PIL import Image
                log=io.open_log(name)
                try:
                    rgb=io.read_rgb(log,'interpreted_lithology_column',y0=top,y1=bottom)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(rgb).save(path)
                finally:log.close()
        return path


class Handler(BaseHTTPRequestHandler):
    server_version='Labeltool/1'
    protocol_version='HTTP/1.1'

    def send(self,data,status=200,content_type='application/json; charset=utf-8',cache=False):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','public, max-age=86400' if cache else 'no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

    def do_GET(self):
        route=urlparse(self.path);params=parse_qs(route.query)
        get=lambda key,default='':params.get(key,[default])[0]
        app=self.server.app
        try:
            if route.path=='/':self.send((HOME/'static'/'index.html').read_bytes(),content_type='text/html; charset=utf-8')
            elif route.path=='/api/bootstrap':self.send(dict(progress=app.progress(),summary=read_json(DATA/'summary.json',{})))
            elif route.path=='/api/items':
                items=app.select(get('mode','uncertain'),get('log'),set(filter(None,get('exclude').split(','))),min(8,max(1,int(get('limit','4')))))
                self.send(dict(items=items))
                for item in items:app.pool.submit(app.image_file,item['log'],'crop',item['context_top'],item['context_bottom'])
            elif route.path=='/api/progress':self.send(app.progress())
            elif route.path=='/api/stats':self.send(app.stats())
            elif route.path=='/api/image':
                path=app.image_file(get('log'),get('kind','crop'),get('y0'),get('y1'))
                self.send(path.read_bytes(),content_type='image/png',cache=True)
            elif route.path=='/api/vocab':
                name=get('log')
                if name not in app.logs:raise ValueError('Unknown log')
                labels={t['label'] for t in app.logs[name]['tiles']}
                folder=ROOT/'labelling'/'out'
                for p in folder.glob('*/vocab.json'):
                    data=read_json(p,{})
                    if data.get('log')==name:labels.update(e['label'] for e in data.get('entries',[]) if e.get('label'))
                self.send(dict(labels=sorted(labels)))
            elif route.path=='/api/jev2':
                from labeltool import jev2
                item=app.by_id[get('item')]
                top=int(get('top') or item['top_y']);bottom=int(get('bottom') or item['bottom_y'])
                if not item['top_y']-2000<=top<bottom<=item['bottom_y']+2000:raise ValueError('Rows outside the item')
                result=jev2.decide(item['log'],top,bottom,jev2.ask(item['log'],top,bottom))
                if result.get('panel'):result['panel_url']='/api/jev2_panel?'+urlencode(dict(key=result['panel']))
                self.send(result)
            elif route.path=='/api/jev2_panel':
                key=get('key')
                if not key.isalnum() or len(key)>64:raise ValueError('Bad key')
                self.send((CACHE/'jev2'/(key+'.png')).read_bytes(),content_type='image/png',cache=True)
            elif route.path=='/favicon.ico':self.send(b'',204,content_type='image/x-icon')
            else:self.send(dict(error='Not found'),404)
        except (ValueError,KeyError,TypeError) as exc:self.send(dict(error=str(exc)),400)
        except Exception as exc:self.log_error('%s',exc);self.send(dict(error=str(exc)),500)

    def do_POST(self):
        app=self.server.app
        try:
            # Require same-origin JSON; reject cross-site form submissions on team VMs.
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Send application/json')
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=32768:raise ValueError('Invalid request size')
            payload=json.loads(self.rfile.read(length))
            with app.lock:
                if self.path=='/api/decision':
                    item=app.by_id[payload['item_id']]
                    event=app.store.save(item,payload)
                    app.stats_cache=None
                    self.send(dict(event=event,progress=app.progress()))
                elif self.path=='/api/undo':
                    event=app.store.undo();app.stats_cache=None
                    self.send(dict(event=event,item=app.public_item(app.by_id[event['item_id']]) if event else None,progress=app.progress()))
                elif self.path=='/api/refresh':self.send(dict(changed=app.refresh_sources(),progress=app.progress()))
                else:self.send(dict(error='Not found'),404)
        except (ValueError,KeyError,TypeError) as exc:self.send(dict(error=str(exc)),400)
        except Exception as exc:self.log_error('%s',exc);self.send(dict(error=str(exc)),500)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--human-dir',type=Path,default=HUMAN,help='Optional isolated journal for automated tests')
    args=parser.parse_args()
    # Keep all writes within the user-authorized directories.
    target=args.human_dir.resolve()
    if not (target.is_relative_to(HOME.resolve()) or target.is_relative_to(HUMAN.resolve())):
        parser.error('--human-dir must be inside labeltool/ or gt/human/')
    app=Application(target)
    try:
        server=ThreadingHTTPServer((args.host,args.port),Handler)
    except OSError as exc:
        app.pool.shutdown(wait=False,cancel_futures=True)
        parser.exit(1,f'Cannot listen on {args.host}:{args.port}: {exc}\nTry --port 18765 if that port is already in use.\n')
    server.daemon_threads=True
    server.app=app
    print(f'Labelling tool: http://{args.host}:{args.port}/ ({len(app.items)} intervals)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();app.pool.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__':main()
