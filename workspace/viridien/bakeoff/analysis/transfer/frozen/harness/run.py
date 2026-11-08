"""Execute strategies with shared preprocessing, truth, scoring and figures."""
import argparse
from collections import Counter
from dataclasses import replace
import importlib
import json
from pathlib import Path
import threading
import time
import traceback
import numpy as np
from PIL import Image
import psutil
from . import io, legend, depth, score, synth, viz

def _strategy(name):
    if name=='_dummy':return importlib.import_module('harness._dummy')
    matches=sorted((io.ROOT/'strategies').glob(name+'*.py'))
    if len(matches)!=1:raise ValueError(f'Strategy {name!r} must resolve to one file; found {len(matches)}')
    return importlib.import_module('strategies.'+matches[0].stem)

def _truth(name,source,window):
    if source!='consensus': return io.load_rows(io.ROOT/'gt'/source/(name+'.csv'))
    paths=list((io.ROOT/'gt'/'silver').glob('*/'+name+'.csv'))
    gold=io.load_rows(io.ROOT/'gt'/'gold'/(name+'.csv'))
    if gold is not None:return gold
    votes=[score.raster(io.load_rows(p),window) for p in paths]
    if not votes:return None
    labels=[]
    for column in zip(*votes):
        count=Counter(v for v in column if v)
        labels.append(count.most_common(1)[0][0] if count else '')
    # Preserve the original spelling from a contributing CSV.
    spelling={score.normalize(r['lithology']):r['lithology'] for p in paths for r in io.load_rows(p)}
    rows=[];start=0
    for end in range(1,len(labels)+1):
        if end==len(labels) or labels[end]!=labels[start]:
            if labels[start]:rows.append(dict(top_y=window[0]+start,bottom_y=window[0]+end,lithology=spelling[labels[start]]))
            start=end
    return rows

class Resources:
    def __init__(self):
        self.peak=0;self.stop=threading.Event();self.process=psutil.Process()
        self.thread=threading.Thread(target=self.sample,daemon=True)

    def sample(self):
        while True:
            try:
                rss=self.process.memory_info().rss
                for child in self.process.children(recursive=True):
                    try:rss+=child.memory_info().rss
                    except psutil.Error:pass
                self.peak=max(self.peak,rss)
            except psutil.Error:pass
            if self.stop.wait(.02):break

    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()

def _clean(rows,window,calib):
    result=[];last=window[0]
    for row in rows:
        r=dict(row);a=max(window[0],int(r['top_y']));b=min(window[1],int(r['bottom_y']))
        if b<=a:continue
        if a<last:raise ValueError('Strategy returned unsorted or overlapping intervals')
        r.update(top_y=a,bottom_y=b,lithology=str(r.get('lithology','Unknown')),cluster_id=int(r.get('cluster_id',0)))
        for edge,y in [('top',a),('bottom',b)]:
            if r.get(edge+'_depth') in (None,'') or y!=int(row[edge+'_y']):r[edge+'_depth']=round(calib.to_depth(y),1)
            else:r[edge+'_depth']=round(float(r[edge+'_depth']),1)
        result.append(r);last=b
    return result

def _synthetic_log(original,rgb,entries):
    colw=rgb.shape[1];h=rgb.shape[0]
    legend_rgb=io.read_rgb(original,'lithology_legend');lh,lw=legend_rgb.shape[:2]
    page=Image.new('RGB',(colw+20+lw,max(h,lh)),'white')
    page.paste(Image.fromarray(rgb),(0,0));page.paste(Image.fromarray(legend_rgb),(colw+20,0))
    old=original.boxes['lithology_legend'];dx=colw+20-old[0];dy=-old[1]
    shifted=[replace(e,swatch_box=[e.swatch_box[0]+dx,e.swatch_box[1]+dy,e.swatch_box[2]+dx,e.swatch_box[3]+dy]) for e in entries]
    log=io.Log(page,dict(interpreted_lithology_column=[0,0,colw,h],depth_column=[colw,0,colw+1,h],lithology_legend=[colw+20,0,colw+20+lw,lh]),original.name)
    return log,shifted

def aggregate(records):
    fields=['score','row_acc','boundary_f1','cluster_ari','depth_mae_ft','seconds','peak_rss_gb','peak_gpu_gb','api_calls','legend_seconds','depth_seconds']
    means={}
    for field in fields:
        values=[r[field] for r in records if r.get(field) is not None]
        means[field]=float(np.mean(values)) if values else None
    means.update(crashes=sum('error' in r for r in records),n_windows=len(records),n_scored=sum(r.get('scored',False) for r in records),api_calls=sum(r.get('api_calls',0) for r in records))
    return means

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',required=True)
    parser.add_argument('--set',choices=['eval','full','synthetic'],default='eval')
    parser.add_argument('--gt',default='gold')
    parser.add_argument('--items',type=Path,default=io.ROOT/'eval_set.json')
    parser.add_argument('--gt-dir',type=Path,help='Truth CSV directory; overrides --gt')
    parser.add_argument('--out-tag',help='Subdirectory under the strategy output')
    parser.add_argument('--min-layer-px',type=int,default=0,help='Merge thinner touching layers before scoring (default: off)')
    args=parser.parse_args(argv)
    if args.min_layer_px<0:parser.error('--min-layer-px must be nonnegative')
    if args.out_tag and (args.out_tag in ('.','..') or '/' in args.out_tag or '\\' in args.out_tag):
        parser.error('--out-tag must be a single directory name')
    module=_strategy(args.strategy)
    strategy_id=module.__name__.split('.')[-1]
    out=io.ROOT/'out'/strategy_id
    if args.out_tag:out=out/args.out_tag
    if args.set=='synthetic':out=out/'synthetic'
    out.mkdir(parents=True,exist_ok=True)
    if args.set=='full':items=[dict(log=name) for name in io.list_logs()]
    else:items=json.loads(args.items.read_text(encoding='utf-8'))
    counts=Counter(item['log'] for item in items)
    import torch
    device='cuda' if torch.cuda.is_available() else 'cpu'
    records=[]
    def save():
        data=dict(strategy=strategy_id,name=getattr(module,'NAME',strategy_id),set=args.set,gt=args.gt,logs=records,mean=aggregate(records),gpu_measurement='PyTorch peak reserved allocator bytes; RSS includes child processes')
        if args.gt_dir:data['gt_dir']=str(args.gt_dir)
        if args.out_tag:data['out_tag']=args.out_tag
        if args.min_layer_px:data['min_layer_px']=args.min_layer_px
        (out/'metrics.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    for item in items:
        original=None
        try:
            original=io.open_log(item['log'])
            t=time.perf_counter();entries=legend.parse(original);legend_seconds=time.perf_counter()-t
            t=time.perf_counter();calib=depth.calibrate(original);depth_seconds=time.perf_counter()-t
            box=original.boxes['interpreted_lithology_column']
            window=(item.get('y0',box[1]),item.get('y1',box[3]))
            if args.set=='synthetic':
                cases=[]
                for i,(rgb,truth) in enumerate(synth.make(original,n=3)):
                    log,shifted=_synthetic_log(original,rgb,entries)
                    for r in truth:
                        for edge in ('top','bottom'):r[edge+'_depth']=round(calib.to_depth(r[edge+'_y']),1)
                    cases.append((log,(0,len(rgb)),truth,shifted,original.name+f'__synthetic_{i}'))
            else:
                truth=(io.load_rows(args.gt_dir/(original.name+'.csv')) if args.gt_dir else
                       None if args.set=='full' else _truth(original.name,args.gt,window))
                if args.gt_dir and truth is None:raise FileNotFoundError(args.gt_dir/(original.name+'.csv'))
                key=original.name
                if counts[key]>1:key+=f'__y{window[0]}_{window[1]}'
                cases=[(original,window,truth,entries,key)]
            for log,win,truth,case_entries,key in cases:
                record=dict(log=original.name,key=key,window=list(win),scored=truth is not None,legend_seconds=legend_seconds,depth_seconds=depth_seconds,api_calls=0)
                rows=[];stages={};resource=None
                try:
                    if getattr(module,'NEEDS_GPU',False) and device=='cuda':
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    cache=io.ROOT/'out'/'_cache'/strategy_id;cache.mkdir(parents=True,exist_ok=True)
                    with Resources() as resource:
                        t=time.perf_counter()
                        try:
                            value=module.run(log,win,dict(legend=case_entries,calib=calib,device=device,cache_dir=cache))
                            if getattr(module,'NEEDS_GPU',False) and device=='cuda':torch.cuda.synchronize()
                        finally:record['seconds']=time.perf_counter()-t
                    record['peak_rss_gb']=resource.peak/1024**3
                    record['peak_gpu_gb']=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_initialized() else 0.0
                    rows,stages=value if isinstance(value,tuple) else (value,{})
                    record['api_calls']=int(stages.get('_api_calls',0))
                    rows=_clean(rows,win,calib)
                    if truth is not None:record.update(score.score(rows,truth,win,min_layer_px=args.min_layer_px))
                except Exception as exc:
                    record.update(error=f'{type(exc).__name__}: {exc}',score=0.0)
                    record['traceback']=traceback.format_exc()
                    rows=[]
                    if resource is not None:record['peak_rss_gb']=resource.peak/1024**3
                    record['peak_gpu_gb']=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_initialized() else 0.0
                io.save_rows(out/(key+'.csv'),rows)
                try:viz.render(log,rows,out/(key+'.png'),gt_rows=truth,window=win,stages=stages,legend_entries=case_entries,title=getattr(module,'NAME',strategy_id)+' | '+key)
                except Exception as exc:
                    record.update(error=f'Render: {exc}',score=0.0)
                record.update(csv=key+'.csv',png=key+'.png')
                records.append(record);save();print(json.dumps(record),flush=True)
                if log is not original:log.close()
        except Exception as exc:
            record=dict(log=item['log'],key=item['log'],error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),score=0.0,scored=False)
            # External preprocessing failures must count in accuracy means too.
            if args.gt_dir and 'y0' in item and 'y1' in item:
                truth=io.load_rows(args.gt_dir/(item['log']+'.csv'))
                if truth is not None:
                    win=(item['y0'],item['y1'])
                    record.update(score.score([],truth,win,min_layer_px=args.min_layer_px))
                    record.update(score=0.0,scored=True,window=list(win))
            records.append(record)
            save();print(item['log'],exc,flush=True)
        finally:
            if original is not None:original.close()
    print(json.dumps(aggregate(records),indent=2))

if __name__=='__main__':main()
