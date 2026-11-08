"""Run: python -X utf8 classify/bench.py [--true-only | --poll-minutes 100]."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse
import csv
import json
import threading
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import f1_score
from classify.data import HERE, OUT, MANIFEST, datasets, dump, prepare, label_key, label_root
from classify.models import LOCAL, DEVICE, classify, release
from classify.api import APIs

CLASSIFIERS = LOCAL+['jev_choice','jev_gallery','terra','hybrid']


class Resources:
    def __enter__(self):
        self.peak=0
        self.done=threading.Event()
        self.proc=psutil.Process()
        self.started=time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start()
        return self

    def sample(self):
        while True:
            try:
                rss=self.proc.memory_info().rss
                for child in self.proc.children(recursive=True):
                    try: rss+=child.memory_info().rss
                    except psutil.Error: pass
                self.peak=max(self.peak,rss)
            except psutil.Error: pass
            if self.done.wait(.03):break

    def __exit__(self,*args):
        if torch.cuda.is_available():torch.cuda.synchronize()
        self.seconds=time.perf_counter()-self.started
        self.done.set();self.thread.join()
        self.gpu=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_initialized() else 0


def prediction_rows(case,name,predictions,meta,resource):
    rows=[]
    for patch,p,m in zip(case['patches'],predictions,meta):
        vocab=case['vocab']
        truth=patch['lithology']
        base=dict(source=case['info']['source'],family=case['info']['family'],key=case['info']['key'],
            log=case['info']['log'],patch_id=patch['patch_id'],classifier=name,truth=truth,
            original_truth=patch['original_truth'],top_y=patch['top_y'],bottom_y=patch['bottom_y'],
            patch_y0=patch['patch_y0'],patch_y1=patch['patch_y1'],patch_path=patch['path'],n_labels=len(vocab),
            peak_rss_gb=resource.peak/1024**3,peak_gpu_gb=resource.gpu,**m)
        if p is not None:
            order=np.argsort(-np.asarray(p),kind='stable')
            ranked=[vocab[i] for i in order]
            rank=1+next(i for i,v in enumerate(ranked) if label_key(v)==label_key(truth))
            base.update(predicted=ranked[0],rank_of_truth=rank,probabilities=json.dumps(dict(zip(vocab,map(float,p)))),
                ranking=json.dumps(ranked),margin=float(p[order[0]]-p[order[1]]) if len(order)>1 else 1,
                correct=rank==1,top3=rank<=3)
        else:
            base.update(predicted='',rank_of_truth=None,probabilities='{}',ranking='[]',margin=None,correct=False,top3=False)
        rows.append(base)
    return rows


def run_case(case,apis):
    rows=[]
    n=len(case['patches'])
    if not n:
        return rows
    for name in CLASSIFIERS[:-1]:
        active=case
        if name=='terra' and case['info']['source']=='SAMPLE2':
            active=dict(case,patches=[p for p in case['patches'] if p['patch_id'] in apis.sample2_terra_ids])
            if not active['patches']:continue
        n=len(active['patches'])
        path=OUT/'prediction_cache'/(case['info']['key']+'_'+name+'.json')
        signature=case['info']['truth_sha256']+json.dumps(case['vocab'])+','.join(p['patch_id'] for p in active['patches'])
        if case['info']['source']=='SAMPLE2':signature+=':sample2-v1'
        if path.exists():
            cached=json.loads(path.read_text(encoding='utf-8'))
            quota_skipped=(name=='terra' and case['info']['source']=='SAMPLE2' and
                any(r.get('status','').startswith('Shared ') for r in cached['rows']))
            if cached.get('signature')==signature and not quota_skipped and not (apis.retry_failed and any(not r['predicted'] for r in cached['rows'])):
                rows.extend(cached['rows']);print(f"CACHED {case['info']['key']} {name}",flush=True);continue
        error=None;extra={}
        with Resources() as resource:
            try:
                if name in LOCAL:
                    predictions,extra=classify(active,name)
                    meta=None
                elif name=='terra':
                    predictions,meta=apis.terra(active)
                else:
                    predictions,meta=apis.jev(active,name)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                dump(OUT/'errors'/(case['info']['key']+'_'+name+'.json'),dict(error=error,traceback=traceback.format_exc()))
                predictions=[None]*n;meta=None
        if meta is None:
            meta=[dict(seconds=resource.seconds/n,calls=0,cache_hit=False,status=error or 'ok') for _ in range(n)]
        current=prediction_rows(active,name,predictions,meta,resource)
        dump(path,dict(signature=signature,rows=current,extra=extra,total_seconds=resource.seconds))
        rows.extend(current)
        valid=[r for r in current if r['predicted']]
        acc=sum(r['correct'] for r in valid)/len(valid) if valid else None
        print(json.dumps(dict(key=case['info']['key'],classifier=name,n=n,n_valid=len(valid),top1=acc,
                              seconds=round(resource.seconds,3),error=error,extra=extra)),flush=True)
        if name in ('dinov2','siglip','cnn_synth'):release()
    return rows


def hybrid(rows):
    df=pd.DataFrame(rows)
    true=df[(df.source=='TRUE') & df.classifier.isin(LOCAL) & df.predicted.ne('')]
    if true.empty:
        output=list(rows)
        for row in rows:
            if row['classifier']=='fft_hatch':
                output.append(dict(row,classifier='hybrid',sent_to_jev=False,jev_used=False,gate_threshold=0))
        return output,dict(status='No TRUE calibration; fixed FFT local-only policy',local='fft_hatch',threshold=0)
    means=true.groupby('classifier').correct.mean().to_dict()
    best=max(LOCAL,key=lambda x:(means.get(x,-1),-LOCAL.index(x)))
    local=df[df.classifier==best].set_index('patch_id')
    jev=df[df.classifier=='jev_choice'].set_index('patch_id')
    train=local[local.source=='TRUE']
    available=train.margin.dropna().astype(float).to_numpy()
    unique=np.unique(available)
    thresholds=[0.0]+[(a+b)/2 for a,b in zip(unique,unique[1:])]+([float(np.nextafter(unique[-1],np.inf))] if len(unique) else [])
    candidates=[]
    for threshold in thresholds:
        hits=sent=0
        for pid,row in train.iterrows():
            gate=not row.predicted or float(row.margin)<threshold
            sent+=gate
            chosen=jev.loc[pid] if gate and pid in jev.index and jev.loc[pid].predicted else row
            hits+=bool(chosen.correct)
        candidates.append((hits,-sent,-threshold,threshold))
    threshold=max(candidates)[3]
    policy=dict(local=best,jev='jev_choice',threshold=threshold,tuning_source='log06 TRUE only',
                local_true_top1=means[best],status='ok',selection_bias='Local choice and threshold are fitted on these 17 TRUE patches; TRUE hybrid result is resubstitution, not held-out validation.')
    paired_jev=sum(bool(jev.loc[pid].predicted) for pid in train.index if pid in jev.index)
    policy['n_true_jev_available']=paired_jev
    if not paired_jev:
        policy.update(status='local_only_no_jev_calibration',
                      limitation='No JEV predictions exist on TRUE. All thresholds tie on hybrid accuracy; fewer-call tie-break selects zero. Hybrid gain cannot be measured.')
    output=list(rows)
    for pid,row in local.iterrows():
        gate=not row.predicted or pd.isna(row.margin) or float(row.margin)<threshold
        jr=jev.loc[pid] if pid in jev.index else None
        success=gate and jr is not None and bool(jr.predicted)
        chosen=(jr if success else row).to_dict()
        chosen.update(patch_id=pid,classifier='hybrid',sent_to_jev=bool(gate),jev_used=bool(success),
            seconds=float(row.seconds)+(float(jr.seconds) if gate and jr is not None else 0),
            calls=float(jr.calls) if gate and jr is not None else 0,
            status='ok' if chosen['predicted'] else 'Local and JEV unavailable',
            peak_rss_gb=max(float(row.peak_rss_gb),float(jr.peak_rss_gb) if gate and jr is not None else 0),
            peak_gpu_gb=float(row.peak_gpu_gb))
        chosen['gate_threshold']=threshold
        output.append(chosen)
    dump(OUT/'hybrid_policy.json',policy)
    return output,policy


def wilson(hits,n):
    if not n:return None,None
    z=1.959963984540054
    p=hits/n;denominator=1+z*z/n
    centre=(p+z*z/(2*n))/denominator
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0.,float(centre-radius)),min(1.,float(centre+radius))


def family_choices(summary):
    choices=[]
    for family,g in summary[summary.source=='SAMPLE2'].groupby('family',sort=True):
        measured=g[g.top1.notna() & (g.n_scored>0)].sort_values(['top1','classifier'],ascending=[False,True])
        best=measured.iloc[0] if len(measured) else None
        runner=measured.iloc[1] if len(measured)>1 else None
        winner=best.classifier if runner is not None and best.top1_ci_low>runner.top1 else 'tie'
        choices.append(dict(family=family,classifier=winner,best=best.classifier if best is not None else None,
            n=int(best.n_patches) if best is not None else 0,
            logs_covered=int(best.logs_covered) if best is not None else 0,
            top1=float(best.top1) if best is not None else None,
            ci_low=float(best.top1_ci_low) if best is not None else None,
            ci_high=float(best.top1_ci_high) if best is not None else None,
            runner_up=runner.classifier if runner is not None else None,
            runner_up_top1=float(runner.top1) if runner is not None else None))
    return choices


def summarize(rows):
    df=pd.DataFrame(rows)
    output=[]
    for (source,family,name),group in df.groupby(['source','family','classifier'],sort=False):
        n=len(group);valid=group.predicted.ne('');nv=int(valid.sum())
        y=[label_key(v) for v in group.truth]
        yp=[label_key(v) if v else '__unavailable__' for v in group.predicted]
        low,high=wilson(int(group.correct.sum()),n) if nv else (None,None)
        output.append(dict(source=source,family=family,classifier=name,n_patches=n,n_scored=nv,coverage=nv/n,
            logs_covered=int(group.log.nunique()),top1_ci_low=low,top1_ci_high=high,
            top1=float(group.correct.mean()) if nv else None,top3=float(group.top3.mean()) if nv else None,
            macro_f1=float(f1_score(y,yp,labels=sorted(set(y)),average='macro',zero_division=0)) if nv else None,
            sec_per_patch=float(group.seconds.mean()),calls_per_patch=float(group.calls.mean()),
            total_calls=float(group.calls.sum()),peak_rss_gb=float(group.peak_rss_gb.max()),peak_gpu_gb=float(group.peak_gpu_gb.max()),
            share_sent_jev=float(group.sent_to_jev.map(lambda x:x is True or x==True).mean()) if 'sent_to_jev' in group else 0,
            share_jev_used=float(group.jev_used.map(lambda x:x is True or x==True).mean()) if 'jev_used' in group else 0))
    return pd.DataFrame(output)


def outputs(rows,prepared,apis,state):
    from classify.report import build
    rows,policy=hybrid(rows)
    result_df=pd.DataFrame(rows)
    result_df['rank_of_truth']=pd.array(result_df.rank_of_truth,dtype='Int64')
    result_df.to_csv(OUT/'results.csv',index=False)
    summary=summarize(rows)
    # Preserve every requested family in the consensus table even when upstream labelling fails.
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    placeholders=[]
    for key,item in manifest.items():
        if not ((summary.source=='CONSENSUS') & (summary.family==item['family'])).any():
            for name in CLASSIFIERS:
                placeholders.append(dict(source='CONSENSUS',family=item['family'],classifier=name,n_patches=0,n_scored=0,
                    logs_covered=0,top1_ci_low=None,top1_ci_high=None,
                    coverage=0,top1=None,top3=None,macro_f1=None,sec_per_patch=None,calls_per_patch=None,total_calls=0,
                    peak_rss_gb=None,peak_gpu_gb=None,share_sent_jev=None,share_jev_used=None,status='ground_truth_unavailable'))
    summary['status']=np.where(summary.n_scored>0,'measured','classifier_unavailable')
    if placeholders:summary=pd.DataFrame(summary.to_dict('records')+placeholders)
    sample_plan=OUT/'sample2_plan.json'
    if sample_plan.exists():
        missing=[]
        for item in json.loads(sample_plan.read_text(encoding='utf-8'))['families']:
            for name in CLASSIFIERS:
                if not ((summary.source=='SAMPLE2') & (summary.family==item['family']) & (summary.classifier==name)).any():
                    missing.append(dict(source='SAMPLE2',family=item['family'],classifier=name,n_patches=0,n_scored=0,
                        logs_covered=0,coverage=0,top1=None,top1_ci_low=None,top1_ci_high=None,top3=None,macro_f1=None,
                        sec_per_patch=None,calls_per_patch=None,total_calls=0,peak_rss_gb=None,peak_gpu_gb=None,
                        share_sent_jev=None,share_jev_used=None,status='reference_or_subsample_unavailable'))
        if missing:summary=pd.DataFrame(summary.to_dict('records')+missing)
    summary['_source_order']=summary.source.map({'SAMPLE2':0,'TRUE':1,'CONSENSUS':2})
    summary=summary.sort_values('_source_order',kind='stable').drop(columns='_source_order')
    summary.to_csv(OUT/'summary.csv',index=False)
    choices=family_choices(summary)
    dump(OUT/'sample2_choices.json',choices)
    events=apis.events
    actual=[e for e in events if not e.get('cache_hit')]
    calls={name:sum(e['calls'] for e in actual if e['classifier']==name) for name in ['jev_choice','jev_gallery','terra']}
    usage={name:{token:sum(e.get('usage',{}).get(token,0) for e in actual if e['classifier']==name)
                 for token in ['input_tokens','output_tokens']} for name in calls}
    metadata=dict(device=DEVICE,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if DEVICE=='cuda' else None,
        classifiers=CLASSIFIERS,manifest=str(MANIFEST),label_root=str(label_root()),state=state,
        actual_calls=calls,usage=usage,monetary_cost='Unknown: no verified JEV tariff or Terra account billing; calls and reported tokens are recorded.',
        hybrid=policy,windows=[c['info'] for c in prepared],created=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        metrics='Patch-weighted top1/top3; missing predictions count wrong when partial coverage exists; entirely unavailable cells are NA. Macro-F1 uses classes present in truth. Local probabilities are softmax similarity scores, Terra probabilities self-reported; neither is calibrated. Ties use legend order.',
        timing='Includes model loading, swatch embedding, CNN training and inference amortized over window patches; input/legend preparation excluded. API cached reruns retain original request latency/calls; calls.jsonl separately records actual requests and cache hits. Hybrid is simulated from cached standalone outputs and makes no additional requests.',
        memory='Sampled process RSS including children at 30ms; GPU is peak PyTorch reserved allocator memory, includes already-resident context/models and excludes remote compute.')
    metadata['family_choices']=choices
    for field,file in [('sample2_plan','sample2_plan.json'),('sample2_agreement','sample2_agreement.json'),('sample2_status','sample2_status.json')]:
        if (OUT/file).exists():metadata[field]=json.loads((OUT/file).read_text(encoding='utf-8'))
    metadata['metrics']+=' Top-1 intervals are 95% Wilson intervals over eligible patches (unavailable predictions count wrong); they do not account for within-log correlation. SAMPLE2 Terra uses a deterministic 25% per-family subset, rounded down; its n and log coverage refer only to that subset. Family choice requires the best top1 Wilson lower bound to exceed the runner-up point estimate; otherwise tie. This rule is descriptive, not a paired significance test.'
    upstream=label_root()/'summary.json'
    if upstream.exists():
        metadata['consensus_job']=json.loads(upstream.read_text(encoding='utf-8'))
    render=OUT/'edge_check.json'
    if render.exists():metadata['edge_check']=json.loads(render.read_text(encoding='utf-8'))
    dump(OUT/'run.json',metadata)
    build(pd.DataFrame(rows),summary,prepared,metadata)
    measured=summary[summary.n_patches>0]
    print(measured[['source','family','classifier','n_patches','n_scored','top1','top3','sec_per_patch','calls_per_patch']].to_string(index=False),flush=True)
    print(f"CONSENSUS: {int((summary.source=='CONSENSUS').sum()/len(CLASSIFIERS))} families listed; {int(summary[summary.source=='CONSENSUS'].n_patches.max()) if (summary.source=='CONSENSUS').any() else 0} maximum eligible patches/family",flush=True)


def run_sample2(budget=None):
    """Add SAMPLE2 while retaining all old result rows and the frozen TRUE policy."""
    from classify.sample2 import CallBudget, Reader, jsonl, REFERENCE, reference_cases, terra_subset
    if not REFERENCE.exists() or not jsonl(REFERENCE):
        print('No agreed SAMPLE2 references; run sample2.py --label first.',flush=True)
        return
    apis=APIs()
    apis.sample2_reader=Reader(budget or CallBudget())
    apis.sample2_terra_ids=terra_subset(jsonl(REFERENCE))
    prior=OUT/'results.csv'
    rows=[]
    if prior.exists():
        old=pd.read_csv(prior)
        for column in ['predicted','truth','original_truth','status','ranking','probabilities']:
            if column in old:old[column]=old[column].fillna('')
        rows=old[(old.source!='SAMPLE2') & (old.classifier!='hybrid')].to_dict('records')
    prepared=[]
    errors=[]
    for case in reference_cases():
        try:
            rows.extend(run_case(case,apis))
            prepared.append(case)
        except Exception as exc:
            errors.append(dict(log=case['info']['log'],error=type(exc).__name__))
    if rows:outputs(rows,prepared,apis,dict(stage='sample2_complete',errors=errors))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--true-only',action='store_true')
    parser.add_argument('--poll-minutes',type=float,default=100)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--retry-failed',action='store_true',help='Retry unavailable API outputs, retaining prior responses and the 60-call Terra ledger')
    parser.add_argument('--sample2',action='store_true',help='Benchmark agreed SAMPLE2 references, preserving older results')
    args=parser.parse_args()
    if args.sample2:
        run_sample2();return
    OUT.mkdir(parents=True,exist_ok=True)
    apis=APIs();rows=[];prepared=[];done=set();errors=[]
    apis.retry_failed=args.retry_failed
    if args.report_only:
        for item in datasets(True):
            try:
                case=prepare(item);prepared.append(case)
                for name in CLASSIFIERS[:-1]:
                    path=OUT/'prediction_cache'/(item['key']+'_'+name+'.json')
                    if path.exists():rows.extend(json.loads(path.read_text(encoding='utf-8'))['rows'])
            except Exception as exc:errors.append(dict(key=item['key'],error=str(exc)))
        from classify.sample2 import REFERENCE, reference_cases
        if REFERENCE.exists():
            try:
                for case in reference_cases():
                    prepared.append(case)
                    for name in CLASSIFIERS[:-1]:
                        path=OUT/'prediction_cache'/(case['info']['key']+'_'+name+'.json')
                        if path.exists():
                            cached=json.loads(path.read_text(encoding='utf-8'))
                            if cached.get('signature','').startswith(case['info']['truth_sha256']):
                                rows.extend(cached['rows'])
            except Exception as exc:errors.append(dict(key='SAMPLE2',error=type(exc).__name__))
        prior=json.loads((OUT/'run.json').read_text(encoding='utf-8')) if (OUT/'run.json').exists() else {}
        outputs(rows,prepared,apis,dict(prior.get('state',{}),errors=errors))
        return
    def consume(include):
        for item in datasets(include):
            if item['key'] in done:continue
            try:
                case=prepare(item)
                prepared.append(case)
                rows.extend(run_case(case,apis))
                done.add(item['key'])
                outputs(rows,prepared,apis,dict(stage='running',completed_keys=sorted(done),errors=errors))
            except Exception as exc:
                errors.append(dict(key=item['key'],error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
                dump(OUT/'errors'/(item['key']+'_prepare.json'),errors[-1]);print('ERROR',errors[-1]['key'],errors[-1]['error'],flush=True)
    consume(not args.true_only)
    state=dict(stage='complete',completed_keys=sorted(done),errors=errors)
    if not args.true_only:
        start=time.monotonic();deadline=start+args.poll_minutes*60
        summary=label_root()/'summary.json'
        while not summary.exists() and time.monotonic()<deadline:
            state.update(stage='waiting_consensus',elapsed_minutes=(time.monotonic()-start)/60,
                         next_poll_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(time.time()+180)))
            dump(OUT/'poll_status.json',state)
            print('WAIT consensus summary absent; next poll in 180 seconds',flush=True)
            # Keep the driver observable while polling only once every three minutes.
            remaining=min(180,max(0,deadline-time.monotonic()))
            while remaining>0:
                duration=min(30,remaining);time.sleep(duration);remaining-=duration
            consume(True)
        consume(True)
        state.update(stage='complete',consensus_summary_present=summary.exists(),waited_minutes=(time.monotonic()-start)/60,
                     missing_families=[k for k in json.loads(MANIFEST.read_text()) if k not in done])
    if len(done)<1+len(json.loads(MANIFEST.read_text())) or not any(r['classifier']=='jev_choice' and r['predicted'] for r in rows):
        state['outcome']='partial: external APIs and/or consensus ground truth unavailable'
    if rows:outputs(rows,prepared,apis,state)
    else:dump(OUT/'run.json',dict(state=state,error='No patches could be prepared'))


if __name__=='__main__':main()
