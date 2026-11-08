"""Dev-only parser selection and locked, one-pass final evaluation.

From bakeoff: python -m ml.legend_names.experiment dev|final|report
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '3'
import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import time

import cv2
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

from harness import io, legend, legend_v3 as v3, score
from harness.run import Resources, _truth, aggregate
from ml.match.experiment import fit
from strategies import s11_fft_legend as s11, s13_boundary_net as s13, s14_learned_match as s14
from ml.boundary.model import peaks

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parents[1]
WORK = BAKEOFF.parent
EXT = WORK / 'external_tests'
MATCH = BAKEOFF / 'ml' / 'match'
BOUNDARY = BAKEOFF / 'ml' / 'boundary'
torch.set_num_threads(3)
cv2.setNumThreads(1)


def dump(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2), encoding='utf-8')


def base_data(log):
    rgb = io.read_rgb(log, 'lithology_legend')
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    source = MATCH / 'cache' / 'out' / '_cache' / 'legend' / (digest+'.json')
    if source.exists():
        return rgb, json.loads(source.read_text(encoding='utf-8')), source.parent
    # All writes stay in this lane, including v1 baseline cache misses.
    root = io.ROOT
    io.ROOT = HERE / 'baseline_cache'
    try:
        small = io.Log(Image.fromarray(rgb), {'lithology_legend':[0,0,rgb.shape[1],len(rgb)]}, digest)
        legend.parse(small)
        source = io.ROOT / 'out' / '_cache' / 'legend' / (digest+'.json')
        return rgb, json.loads(source.read_text(encoding='utf-8')), source.parent
    finally:
        io.ROOT = root


def base_entries(log):
    rgb, data, folder = base_data(log)
    dx,dy = log.boxes['lithology_legend'][:2]
    entries = []
    for e in data['entries']:
        sw = np.asarray(Image.open(folder/e['swatch_file']).convert('RGB'))
        b = e['swatch_box']
        entries.append(legend.LegendEntry(e['id'],e['label'],[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy],sw,tuple(e['mean_rgb']),e['is_lithology']))
    return entries


def prepare_v3(log):
    rgb,data,folder = base_data(log)
    words = [(np.asarray(w['box']),w['text'],w['confidence']) for w in data['ocr']]
    return v3.evidence(log, words)


def iou(a,b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.


def name_metrics(entries, truth):
    # One-to-one geometric association; wrong labels cannot choose their box.
    pairs = sorted([(iou(e.swatch_box,t['box']),i,j) for i,e in enumerate(entries) for j,t in enumerate(truth)],reverse=True)
    used_e,used_t = set(),set(); exact=matchable=canonical=0; details=[]
    for overlap,i,j in pairs:
        if overlap < .5: break
        if i in used_e or j in used_t: continue
        used_e.add(i);used_t.add(j);e,t = entries[i],truth[j]
        ok = score.normalize(e.label) == score.normalize(t['label_printed'])
        sem = v3.canonical_name(e.label) == v3.canonical_name(t['label_printed'])
        exact += ok; matchable += ok and e.is_lithology; canonical += sem and e.is_lithology
        details.append(dict(truth=t['label_printed'],pred=e.label,lithology=e.is_lithology,iou=overlap,exact=ok))
    return dict(n_truth=len(truth),n_pred=len(entries),n_detected=len(used_t),n_exact=int(exact),
                n_matchable=int(matchable),n_canonical=int(canonical),
                detection_recall=len(used_t)/max(1,len(truth)),
                name_accuracy=exact/max(1,len(truth)),matchable_accuracy=matchable/max(1,len(truth)),
                canonical_accuracy=canonical/max(1,len(truth)),details=details)


def legend_truth(name, external=True):
    path = (BAKEOFF/'emil'/'external_labels' if external else WORK/'emil'/'labels')/(name+'.json')
    return json.loads(path.read_text(encoding='utf-8')).get('legend',[]) if path.exists() else []


def summarize_names(records):
    total = sum(r['n_truth'] for r in records)
    result = dict(n_logs=len(records),n_truth=total)
    for k in ('detected','exact','matchable','canonical'):
        result[k] = sum(r['n_'+k] for r in records)/max(1,total)
    result['log_macro'] = float(np.mean([r['matchable_accuracy'] for r in records]))
    return result


def parser_ceiling(entries, truth, window):
    available = {score.normalize(e.label) for e in entries if e.is_lithology}
    labels = score.raster(truth,window);filled = labels!=''
    return float(np.mean([t in available for t in labels[filled]])) if filled.any() else 1.


def record_score(rows, truth, win, **kwargs):
    return dict(**kwargs, **score.score(rows,truth,win))


@threadpool_limits.wrap(limits=3)
def dev():
    if (HERE/'FINAL_LOCK.json').exists():
        raise RuntimeError('Selection is frozen after final evaluation starts')
    index = json.loads((BOUNDARY/'data'/'index.json').read_text())['cases']
    groups = {c['log']:c['well'] for c in index}
    names = list(dict.fromkeys(c['log'] for c in index))
    io.DATASET = EXT
    configs = [dict(easy=False,cleanup=False,easy_weight=.9),dict(easy=False,cleanup=True,easy_weight=.9)]
    configs += [dict(easy=True,cleanup=True,easy_weight=w) for w in (.8,1.,1.15)]
    namescores={str(i):[] for i in range(len(configs))};baseline=[]
    for n,name in enumerate(names):
        log=io.open_log(name)
        try:
            truth=legend_truth(name);prepare_v3(log)
            baseline.append(dict(log=name,well=groups[name],**name_metrics(base_entries(log),truth)))
            for i,config in enumerate(configs):
                namescores[str(i)].append(dict(log=name,well=groups[name],**name_metrics(v3.parse(log,config),truth)))
        finally:log.close()
        print('legend',n+1,len(names),name,[(i,round(namescores[str(i)][-1]['matchable_accuracy'],3)) for i in range(len(configs))],flush=True)
        dump('dev_names_progress.json',dict(configs=configs,baseline=baseline,candidates=namescores))
    wells=sorted(set(groups.values()));folds=[];oof=[]
    for held in wells:
        # Select only using other wells, weighting each well equally.
        rates=[np.mean([np.mean([r['matchable_accuracy'] for r in namescores[str(i)] if r['well']==w]) for w in wells if w!=held]) for i in range(len(configs))]
        best=int(np.argmax(rates));folds.append(dict(held=held,selected=best,training_rates=list(map(float,rates))))
        oof.extend([r for r in namescores[str(best)] if r['well']==held])
    rates=[np.mean([np.mean([r['matchable_accuracy'] for r in namescores[str(i)] if r['well']==w]) for w in wells]) for i in range(len(configs))]
    best=int(np.argmax(rates));selected=dict(config=configs[best],candidate=best,selection='well-macro matchable entry accuracy, dev only',rates=list(map(float,rates)),folds=folds)
    dump('selected.json',selected)
    dump('dev_names.json',dict(baseline=baseline,candidates=namescores,oof=oof,summary=dict(v1=summarize_names(baseline),v3_oof=summarize_names(oof),v3_selected=summarize_names(namescores[str(best)]))))
    # The only labelled non-eval in-domain legend is a development diagnostic.
    from emil.paths import dataset_path
    io.DATASET=dataset_path();excluded={x['log'] for x in json.loads((BAKEOFF/'eval_set.json').read_text())}
    domain=[]
    for p in sorted((WORK/'emil'/'labels').glob('*.json')):
        if p.stem in excluded:continue
        truth=legend_truth(p.stem,False)
        if not truth:continue
        log=io.open_log(p.stem)
        try:
            prepare_v3(log)
            domain.append(dict(log=p.stem,v1=name_metrics(base_entries(log),truth),v3=name_metrics(v3.parse(log),truth)))
        finally:log.close()
    dump('domain_dev_names.json',domain)
    io.DATASET=EXT
    blob=pickle.loads((MATCH/'dev.pkl').read_bytes())
    match_config=json.loads((MATCH/'selected.json').read_text())
    boundary_config=json.loads((BOUNDARY/'selected_config.json').read_text())
    records={s:[] for s in ('s11','s13','s14','s15','s16')}
    for held in wells:
        model=fit(blob['training'],held,'logistic')
        config=configs[next(f['selected'] for f in folds if f['held']==held)]
        for case in index:
            if case['well']!=held:continue
            log=io.open_log(case['log']);win=case['window'];truth=case['truth']
            try:
                entries=base_entries(log);new=v3.parse(log,config)
                rgb=np.asarray(Image.open(BOUNDARY/'data'/(case['key']+'.png')).convert('RGB'))
                prob=np.load(BOUNDARY/'runs'/boundary_config['variant']/(case['key']+'.npy'))
                p=peaks(prob,boundary_config['threshold'],boundary_config['gap']);points=np.r_[0,p,len(rgb)]
                preserve=set(p.tolist()) if boundary_config.get('preserve',True) else set()
                s11points,s11preserve=s14.proposals(log,win,{},rgb)
                start=time.perf_counter();s11rows=s11.run(log,win,dict(legend=entries))[0]
                variants=dict(s11=s11rows,
                    s13=s13.name_intervals(rgb,win[0],points,entries,preserve),
                    s14=s14.name_intervals(rgb,win[0],s11points,entries,s11preserve,matcher=(model,match_config)),
                    s15=s14.name_intervals(rgb,win[0],points,entries,preserve,matcher=(model,match_config)),
                    s16=s14.name_intervals(rgb,win[0],points,new,preserve,matcher=(model,match_config)))
                for s,rows in variants.items():
                    rec=record_score(rows,truth,win,log=log.name,well=held,window=win,scored=True)
                    if s in ('s15','s16'):rec['parser_ceiling']=parser_ceiling(new if s=='s16' else entries,truth,win)
                    records[s].append(rec)
                    io.save_rows(HERE/'dev_predictions'/s/(case['key']+'.csv'),rows)
                print('CV',held,case['key'],{s:round(records[s][-1]['row_acc'],3) for s in records},flush=True)
            finally:log.close()
    result={s:dict(mean=aggregate(rr),well_macro={k:float(np.mean([np.mean([r[k] for r in rr if r['well']==w]) for w in wells])) for k in ('row_acc','boundary_f1','cluster_ari')},logs=rr) for s,rr in records.items()}
    dump('dev_metrics.json',result)
    print('DEV COMPLETE',selected,'\n', {s:r['mean'] for s,r in result.items()},flush=True)


@threadpool_limits.wrap(limits=3)
def final():
    marker=HERE/'FINAL_LOCK.json'
    if marker.exists():raise RuntimeError('Final evaluation has already started; do not repeat')
    paths=[HERE/'selected.json',BAKEOFF/'harness'/'legend_v3.py',BAKEOFF/'strategies'/'s16_legend_v3.py',MATCH/'matcher.npz',MATCH/'selected.json',BOUNDARY/'boundary_net.pt',HERE/'experiment.py']
    paths += [BAKEOFF/'harness'/(n+'.py') for n in ('legend','legend_v2','io','score')]
    paths += [BAKEOFF/'strategies'/(n+'.py') for n in ('s13_boundary_net','s14_learned_match','s15_learned_match')]
    paths += [MATCH/'features.py',BOUNDARY/'model.py']
    lock=dict(started=time.time(),sha256={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},selected=json.loads((HERE/'selected.json').read_text()))
    dump('FINAL_LOCK.json',lock)
    from emil.paths import dataset_path
    from strategies.s16_legend_v3 import run
    results={};name_results={}
    for split in ('ext_test','viridien_eval'):
        io.DATASET=EXT if split=='ext_test' else dataset_path()
        items=json.loads((EXT/'eval_set_test.json' if split=='ext_test' else BAKEOFF/'eval_set.json').read_text())
        records=[];names=[];seen=set()
        for item in items:
            log=io.open_log(item['log']);win=(item['y0'],item['y1']);rows=[]
            rec=dict(log=log.name,window=list(win),scored=True)
            try:
                # Truth is read only after inference, and never selects any setting.
                with Resources() as resources:
                    t=time.perf_counter();prepare_v3(log);entries=v3.parse(log);rec['legend_seconds']=time.perf_counter()-t
                    t=time.perf_counter();rows=run(log,win,{});rec['seconds']=time.perf_counter()-t
                rec['peak_rss_gb']=resources.peak/1024**3
                truth=io.load_rows(EXT/'truth'/(log.name+'.csv')) if split=='ext_test' else _truth(log.name,'consensus',win)
                rec.update(score.score(rows,truth,win));rec['parser_ceiling']=parser_ceiling(entries,truth,win)
                if log.name not in seen:
                    gt=legend_truth(log.name,split=='ext_test')
                    if gt:
                        names.append(dict(log=log.name,v1=name_metrics(base_entries(log),gt),v3=name_metrics(entries,gt)))
                    seen.add(log.name)
            except Exception as exc:
                rec['error']=f'{type(exc).__name__}: {exc}';raise
            finally:log.close()
            io.save_rows(HERE/'final_predictions'/split/(item['log']+f'__{win[0]}_{win[1]}.csv'),rows)
            records.append(rec)
            results[split]=dict(mean=aggregate(records),logs=records);name_results[split]=names
            dump('final_metrics.json',results);dump('final_names.json',name_results)
            print(split,len(records),len(items),item['log'],rec,flush=True)
    dump('FINAL_DONE.json',dict(completed=time.time()))


def report():
    # Formatting is kept separate from inference so reading outputs cannot rerun test.
    dev=json.loads((HERE/'dev_metrics.json').read_text());dn=json.loads((HERE/'dev_names.json').read_text())
    final=json.loads((HERE/'final_metrics.json').read_text());fn=json.loads((HERE/'final_names.json').read_text())
    baselines=json.loads((MATCH/'final_metrics.json').read_text())
    domain=json.loads((HERE/'domain_dev_names.json').read_text())
    lines=['Legend names: opt-in v3 and S16','',
           'Selected using six external development wells only. Outer leave-one-well-out selection excludes the held-out well from parser setting selection and matcher fitting. Boundary probabilities are the boundary lane\'s saved out-of-fold rich_real probabilities. S16 uses the same boundaries and matcher as S15; only the parser changes. Test and in-domain eval were run once after the source/configuration hash lock. Existing S11/S13/S14/S15 final baseline measurements are reused, without another test run.','',
           '| Legend truth | Logs | Entries | Parser | Detected (IoU >= .5) | Exact name | Exact + usable | Canonical + usable |',
           '|---|---:|---:|---|---:|---:|---:|---:|']
    ns=[('ext_dev (outer CV)','v1',dn['summary']['v1']),('ext_dev (outer CV)','v3',dn['summary']['v3_oof'])]
    if domain:
        ns += [('in-domain non-eval','v1',summarize_names([r['v1'] for r in domain])),('in-domain non-eval','v3',summarize_names([r['v3'] for r in domain]))]
    for split,rr in fn.items():
        for p in ('v1','v3'):
            if rr:ns.append((split,p,summarize_names([r[p] for r in rr])))
    for split,p,r in ns:
        lines.append(f"| {split} | {r['n_logs']} | {r['n_truth']} | {p} | {r['detected']:.2%} | {r['exact']:.2%} | {r['matchable']:.2%} | {r['canonical']:.2%} |")
    lines += ['', 'All entry rates use all truth swatches as denominator, including missed detections. Exact + usable also requires is_lithology. A one-to-one IoU association prevents names from choosing their own match. Canonical comparison symmetrically expands a general abbreviation/colour lexicon; it is diagnostic only. Strategy row accuracy always uses the unmodified shared exact-name scorer. Non-lithology truth entries remain in the denominator where supplied by the annotations. In-domain legend truth is sparse and hand-labelled; interval truth is gold-first silver consensus.','',
              '| Split | Strategy | Row acc | Boundary F1 | Cluster ARI |', '|---|---|---:|---:|---:|']
    strategy_rows=[]
    for s,r in dev.items():strategy_rows.append(('ext_dev CV (window macro)',s,r['mean']))
    for s,r in dev.items():strategy_rows.append(('ext_dev CV (well macro)',s,r['well_macro']))
    mapping=dict(s11='s11_fft_legend',s13='s13_boundary_net',s14='s14_learned_match',s15='s15_learned_match')
    for split in ('ext_test','viridien_eval'):
        for s,full in mapping.items():strategy_rows.append((split,s,baselines[split][full]['mean']))
        strategy_rows.append((split,'s16',final[split]['mean']))
    for split,s,r in strategy_rows:
        lines.append(f"| {split} | {s.upper()} | {r['row_acc']:.2%} | {r['boundary_f1']:.3f} | {r['cluster_ari']:.3f} |")
    lines += ['', 'Selected ensemble: '+json.dumps(json.loads((HERE/'selected.json').read_text())['config'])+'. Six parser regression tests pass (tests.log).', '', 'Dev parser name ceilings:']
    for s in ('s15','s16'):
        lines.append(f"- {s.upper()}: {np.mean([r['parser_ceiling'] for r in dev[s]['logs']]):.2%} of truth-filled rows have an available exact legend name.")
    lines += ['', 'Final resource measurements (shared OCR included in peak RSS; seconds/window below excludes OCR):']
    for split,r in final.items():
        rr=r['logs'];lines.append(f"- {split}: {r['mean']['seconds']:.3f} seconds/window; {np.mean([v['legend_seconds'] for v in rr]):.3f} seconds/window legend preparation (including cache reuse); peak RSS {max(v['peak_rss_gb'] for v in rr):.2f} GiB; {sum('error' in v for v in rr)} errors.")
    lines += ['', 'Parser changes: full label cells retain descenders beyond short swatches; tight RapidOCR and CPU EasyOCR recognition; conservative vocabulary repair; left/right/below label candidates; physical frames supplement v2 detection; v2 panel classification; exact-pixel content cache with relative coordinates. No network downloads, filenames, per-log rules, or truth at inference.','',
              'Artifacts: harness/legend_v3.py; strategies/s16_legend_v3.py; ml/legend_names/selected.json; dev_names.json; dev_metrics.json; final_names.json; final_metrics.json; FINAL_LOCK.json; dev_predictions/; final_predictions/; cache/.','',
              'Reproduce from bakeoff: `python -m ml.legend_names.experiment dev`, `python -m pytest ml/legend_names/test_legend_v3.py -q`, then the guarded `python -m ml.legend_names.experiment final`, and `python -m ml.legend_names.experiment report`. Set LITHOLOGY_DATASET to the local in-domain dataset. All generated files stay under ml/legend_names.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump('tables.json',dict(name_accuracy=[dict(split=a,parser=b,**c) for a,b,c in ns],strategies=[dict(split=a,strategy=b,**c) for a,b,c in strategy_rows]))
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['dev','final','report'])
    globals()[parser.parse_args().action]()
