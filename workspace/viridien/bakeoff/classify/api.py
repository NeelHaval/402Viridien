"""Content-addressed API caching, including failures, with auditable call records."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from strategies._jev import Client, image_asset
from classify.data import OUT, dump, label_key, panel

VERSION = 'classify-v2'


def json_object(output):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{',output):
        try:
            obj,_ = decoder.raw_decode(output[match.start():])
            if isinstance(obj,dict) and 'predictions' in obj:
                return obj
        except ValueError:
            pass
    return {}


class APIs:
    def __init__(self):
        self.client = Client(timeout=35,retries=0)
        self.failures = 0
        self.terra_failures = 0
        self.retry_failed = False
        self.cache = OUT/'api_cache'
        self.cache.mkdir(parents=True,exist_ok=True)
        self.logpath = OUT/'calls.jsonl'
        self.events = []
        if self.logpath.exists():
            self.events = [json.loads(s) for s in self.logpath.read_text(encoding='utf-8').splitlines() if s.strip()]
        self.terra_calls = sum(e.get('calls',0) for e in self.events if e['classifier']=='terra' and not e.get('cache_hit'))

    def save_event(self,event):
        self.events.append(event)
        with self.logpath.open('a',encoding='utf-8') as f:
            f.write(json.dumps(event,ensure_ascii=False)+'\n')

    def request(self,case,patches,name):
        vocab = case['vocab']
        im = panel(case,patches,'choice' if name=='jev_choice' else 'gallery')
        criteria = {str(i):v for i,v in enumerate(vocab)}
        if name=='terra':
            prompt = ('Classify each numbered QUERY PATCH by comparing its lithology colour and hatch pattern with the supplied legend. '
                      'Use only the vocabulary below. Consider all visible parts of each full-width patch. '
                      'Return JSON only, no tools, no file operations: {"predictions":[{"patch":1,"ranking":[{"label":"exact vocabulary label","probability":0.7}, ...]}, ...]}. '
                      'Rank ALL vocabulary labels for each patch. Probabilities sum to one. '
                      'Vocabulary: '+json.dumps(vocab))
        else:
            prompt = 'Which legend rock best matches the QUERY PATCH? Compare colour, hatch orientation, density and shapes. Choose only from the supplied rock vocabulary.'
        key = hashlib.sha256(VERSION.encode()+name.encode()+im.tobytes()+prompt.encode()+json.dumps(vocab).encode()).hexdigest()
        path = self.cache/(key+'.json')
        imagepath = self.cache/(key+'.png')
        event = dict(classifier=name,key=key,patch_ids=[p['patch_id'] for p in patches],source=case['info']['source'],
                     family=case['info']['family'],timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        if name=='terra' and case['info']['source']=='SAMPLE2':
            return self.sample2_terra_request(im,prompt,imagepath,event)
        cached=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        use_cache=cached is not None and not (self.retry_failed and cached['status']!='ok')
        if use_cache:
            record = cached
            event.update(cache_hit=True,calls=0,seconds=0,status=record['status'])
        else:
            if cached is not None:
                archive=self.cache/'attempts'/f'{key}_{time.time_ns()}.json'
                dump(archive,cached)
            im.save(imagepath)
            record = dict(status='unavailable',calls=0,seconds=0,response=None)
            started = time.perf_counter()
            if name.startswith('jev'):
                if self.failures >= (1 if case['info']['source']=='SAMPLE2' else 3):
                    record['error']='JEV circuit open; no fallback predictions'
                elif not self.client.key:
                    record['error']='JEV key unavailable'
                else:
                    before = self.client.calls
                    response = self.client.ask({'task':prompt,'image':'benchmark_image'},
                        {'benchmark_image':image_asset(im)},
                        {'rock':{'type':'choice','instructions':prompt,'criteria':criteria}})
                    record['calls']=self.client.calls-before
                    record['response']=response
                    if response is not None:
                        record['status']='ok'
                    else:
                        self.failures+=1
                        record['error']='JEV returned no response (network/proxy/auth/server failure; shared client hides HTTP details)'
            else:
                if self.terra_calls >= 60:
                    record['error']='60-call total Terra cap reached'
                elif self.terra_failures >= 3:
                    record['error']='Circuit open after three Terra execution failures'
                else:
                    env = os.environ.copy()
                    env.update(CODEX_HOME='C:/Users/hssil/.codex-accounts/kira181204',PYTHONDONTWRITEBYTECODE='1')
                    for sub in ['_codex_state','_codex_logs','_tmp']:
                        (OUT/sub).mkdir(exist_ok=True)
                    env.update(CODEX_SQLITE_HOME=str(OUT/'_codex_state'),TEMP=str(OUT/'_tmp'),TMP=str(OUT/'_tmp'))
                    # Windows cmd wrappers can consume/alter JSON prompt quotes. Send it via stdin.
                    script=Path(os.environ['APPDATA'])/'npm/node_modules/@openai/codex/bin/codex.js'
                    prefix=[shutil.which('node'),str(script)] if script.exists() else [shutil.which('codex') or 'codex']
                    cmd = prefix+['exec','--skip-git-repo-check','--ephemeral','--ignore-user-config','-s','read-only','-m','gpt-6.1-terra',
                           '-c','model_reasoning_effort="low"','-c','mcp_servers={}','-c','features.shell_tool=false',
                           '-c','history.persistence="none"','-c','web_search="disabled"',
                           '-c',f'log_dir="{(OUT/"_codex_logs").as_posix()}"',
                           '-c',f'sqlite_home="{(OUT/"_codex_state").as_posix()}"','-i',str(imagepath),'-']
                    record['calls']=1
                    self.terra_calls+=1
                    try:
                        proc = subprocess.run(cmd,input=prompt,capture_output=True,text=True,encoding='utf-8',errors='replace',
                                              timeout=180,env=env,cwd=str(OUT.parent))
                        record.update(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode)
                        record['response']=json_object(proc.stdout)
                        if proc.returncode==0 and record['response']:
                            record['status']='ok'
                        else:
                            record['error']='Terra CLI failed or returned invalid prediction JSON'
                            self.terra_failures+=1
                    except subprocess.TimeoutExpired as exc:
                        record.update(error='Terra CLI timeout',stdout=str(exc.stdout or ''),stderr=str(exc.stderr or ''))
                        self.terra_failures+=1
                    except OSError as exc:
                        record['error']=f'Terra launch failed: {type(exc).__name__}'
                        self.terra_failures+=1
            record['seconds']=time.perf_counter()-started
            dump(path,record)
            event.update(cache_hit=False,calls=record['calls'],seconds=record['seconds'],status=record['status'])
        event['usage']=(record.get('response') or {}).get('usage',{})
        if record.get('error'):
            event['error']=record['error']
        self.save_event(event)
        return record,event

    def sample2_terra_request(self,im,prompt,imagepath,event):
        """Use the reference readers' shared quota and failure cache for SAMPLE2."""
        from classify.sample2 import CallBudget, Reader, CallCap, ReaderFailure
        if not hasattr(self,'sample2_reader'):
            self.sample2_reader=Reader(CallBudget())
        imagepath.parent.mkdir(parents=True,exist_ok=True)
        if not imagepath.exists():im.save(imagepath)
        record=dict(status='unavailable',calls=0,seconds=0,response=None)
        try:
            result,cached=self.sample2_reader.invoke(imagepath,prompt,'gpt-6.1-terra','low',
                'sample2:bench:terra:'+event['key'])
            record.update(status='ok',calls=1,seconds=result['seconds'],response=result['answer'])
            event.update(cache_hit=cached,calls=0 if cached else 1,seconds=0 if cached else result['seconds'],status='ok')
        except (CallCap,ReaderFailure) as exc:
            record['error']=str(exc)
            failed=getattr(exc,'record',None)
            cached=getattr(exc,'cache_hit',False)
            if failed:
                record.update(calls=1,seconds=failed['seconds'])
            event.update(cache_hit=cached,calls=0 if cached or not failed else 1,
                         seconds=0 if cached or not failed else failed['seconds'],status='unavailable',error=str(exc))
        event['usage']={}
        self.save_event(event)
        return record,event

    def jev(self,case,name):
        predictions,meta = [],[]
        for patch in case['patches']:
            record,event = self.request(case,[patch],name)
            answer = (record.get('response') or {}).get('answers',{}).get('rock',{})
            probs = answer.get('probabilities',{})
            p = np.asarray([float(probs.get(str(i),0)) for i in range(len(case['vocab']))])
            valid = p.size and np.isfinite(p).all() and (p>=0).all() and p.sum()>0
            predictions.append(p/p.sum() if valid else None)
            meta.append(dict(seconds=record['seconds'],calls=record['calls'],cache_hit=event['cache_hit'],
                             status='ok' if valid else record.get('error','Missing choice probabilities'),cache_key=event['key']))
        return predictions,meta

    def terra(self,case):
        predictions,meta = [],[]
        vocab = case['vocab']; lookup={label_key(v):i for i,v in enumerate(vocab)}
        for offset in range(0,len(case['patches']),8):
            patches=case['patches'][offset:offset+8]
            record,event=self.request(case,patches,'terra')
            result=(record.get('response') or {}).get('predictions',[])
            by_id={int(p.get('patch',-1)):p for p in result if isinstance(p,dict)} if isinstance(result,list) else {}
            for i,patch in enumerate(patches):
                answer=by_id.get(i+1,{})
                ranked=answer.get('ranking',[])
                p=np.zeros(len(vocab))
                seen=set()
                for rank,entry in enumerate(ranked):
                    if isinstance(entry,str):
                        label,prob=entry,1/(rank+1)
                    else:
                        label,prob=entry.get('label',''),entry.get('probability',1/(rank+1))
                    if label_key(label) in lookup:
                        p[lookup[label_key(label)]]=float(prob)
                        seen.add(lookup[label_key(label)])
                valid=len(seen)==len(vocab) and np.isfinite(p).all() and (p>=0).all() and p.sum()>0
                predictions.append(p/p.sum() if valid else None)
                meta.append(dict(seconds=record['seconds']/len(patches),calls=record['calls']/len(patches),
                    cache_hit=event['cache_hit'],status='ok' if valid else record.get('error','Missing patch prediction'),cache_key=event['key']))
        return predictions,meta
