"""Seeded scan robustness benchmark. Run from the repository root.

python -m bakeoff.external.scanify measure
python -m bakeoff.external.scanify build
python -m bakeoff.external.scanify evaluate --split dev
python -m bakeoff.external.scanify report

Only the two explicit dev/test item lists are consumed; other splits are never
loaded. Skew is a horizontal shear, preserving exact horizontal truth rows.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '3'
import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'bakeoff'))
from harness import io
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(1)
DEFAULT_OUT = REPO/'external_tests_scanned'
LEVELS = ('light', 'medium', 'heavy', '1-bit')



def checked_items(root, split):
    """Read explicit allowed lists. Reject a stale dev list before image access."""
    if split not in ('dev', 'test'):
        raise ValueError('Only explicit dev and test build lists are allowed')
    items = json.loads((root / f'eval_set_{split}.json').read_text())
    forbidden = ('forsythb2-17', 'barlow2-14', 'marcotte1-12', 'red-brook-1')
    if any(any(well in item['log'].lower() for well in forbidden) for item in items):
        raise ValueError('Split contains a sealed or shared verify well')
    return items


def refresh(source, out):
    """Keep rendered pages; refresh lists from the new explicit split contract.

    Does not read image pixels or truth. Remove obsolete generated pages by ID.
    """
    items = {split: checked_items(source, split) for split in ('dev', 'test')}
    names = {item['log'] for values in items.values() for item in values}
    for level in LEVELS:
        root = out / level
        geometries = json.loads((root / 'geometry.json').read_text())
        for name in set(geometries) - names:
            for folder, suffix in (('images', '.tif'), ('bbox_annotation', '.json'), ('truth', '.csv')):
                (root / folder / (name + suffix)).unlink(missing_ok=True)
        geometries = {name: geometries[name] for name in sorted(names)}
        dump(root / 'geometry.json', geometries)
        combined = []
        for split, values in items.items():
            transformed = [dict(item, y0=round(item['y0']*geometries[item['log']]['sy']),
                                y1=round(item['y1']*geometries[item['log']]['sy'])) for item in values]
            dump(root / f'eval_set_{split}.json', transformed)
            combined.extend(transformed)
        dump(root / 'eval_set.json', combined)
        with (root / 'manifest.csv').open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [row for row in reader if row['id'] in names]
        with (root / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    dump(out / 'split_contract.json', dict(dev_windows=len(items['dev']), test_windows=len(items['test']),
        generated_logs=len(names), test_evaluated=False, shared_verify_used=False,
        note='Final benchmark belongs to the orchestrator. Old scores are not used.'))

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def stats(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = (grad < 12) & (gray > 160)
    residual = g-cv2.GaussianBlur(g, (0, 0), 1)
    noise = float(np.median(np.abs(residual[flat]))/.67449) if flat.any() else 0.
    lines = cv2.HoughLinesP(cv2.Canny(gray, 60, 160), 1, np.pi/1800, 50,
                            minLineLength=max(40, gray.shape[1]//4), maxLineGap=8)
    angles = []
    for x0,y0,x1,y1 in ([] if lines is None else lines.reshape(-1,4)):
        angle = math.degrees(math.atan2(int(y1)-int(y0), int(x1)-int(x0)))
        angle = (angle+45) % 90-45
        if abs(angle) < 3: angles.append(angle)
    return dict(laplacian_variance=float(lap.var()), noise_sigma=noise,
                saturation=float(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1].mean()/255),
                edge_blur_proxy=float(np.mean(grad)/(np.mean(np.abs(lap))+1e-6)),
                skew_deg=float(np.median(angles)) if angles else None,
                paper_rgb=np.median(rgb[gray>220],axis=0).tolist() if (gray>220).any() else [255]*3)


def measure(out, dataset):
    """No challenge truth or eval pixels are used; sample three small crops/log."""
    from harness import depth
    if (out/'evaluation_freeze.json').exists():raise RuntimeError('Measurement is frozen after evaluation starts')
    io.DATASET = dataset
    excluded = {v['log'] for v in json.loads((REPO/'bakeoff/eval_set.json').read_text())}
    records=[]
    for name in io.list_logs():
        if name in excluded: continue
        log=io.open_log(name)
        try:
            x0,y0,x1,y1=log.boxes['interpreted_lithology_column']
            samples=[]
            for frac in (.15,.5,.85):
                y=round(y0+max(0,y1-y0-512)*frac)
                samples.append(stats(io.read_rgb(log,[x0,y,min(x1,x0+512),min(y1,y+512)])))
            r=dict(log=name,mode=log.mode,size=list(log.size),
                   bits_per_sample=list(log.image.tag_v2.get(258,())) if hasattr(log.image,'tag_v2') else [],
                   compression=str(log.image.info.get('compression','unknown')),
                   palette_entries=len(log.image.getpalette() or [])//3 if log.mode=='P' else None,
                   samples=samples)
            # One bounded depth strip per log. Fit only image OCR numeric anchors.
            dx0,dy0,dx1,dy1=log.boxes['depth_column']
            top=dy0+max(0,(dy1-dy0-2400)//2)
            try:
                rgb=io.read_rgb(log,[dx0,top,dx1,min(dy1,top+2400)])
                points=depth._read_points(rgb,top,'rapidocr')
                a,b,n,rmse=depth._fit(points)
                unit,_=depth._unit_context(log)
                r.update(px_per_ft=1/(a*(depth.METRES_TO_FEET if unit=='m' else 1)),
                         depth_unit=unit or 'assumed_ft',depth_anchors=n,depth_rmse_px=rmse)
            except Exception as exc:r['depth_error']=str(exc)
            records.append(r)
            dump(out/'challenge_statistics.json',dict(excluded_eval_logs=sorted(excluded),logs=records))
            print('measured',name,r.get('px_per_ft'),flush=True)
        finally:log.close()
    samples=[s for r in records for s in r['samples']]
    summary={}
    for key in ('laplacian_variance','noise_sigma','saturation','edge_blur_proxy','skew_deg'):
        values=[s[key] for s in samples if s[key] is not None]
        summary[key]=dict(zip(('p10','p50','p90'),map(float,np.quantile(values,[.1,.5,.9]))))
    ppfs=[r['px_per_ft'] for r in records if r.get('px_per_ft') and .2<r['px_per_ft']<30]
    summary['px_per_ft']=dict(zip(('p10','p50','p90'),map(float,np.quantile(ppfs,[.1,.5,.9])))) if ppfs else None
    summary['modes']={m:sum(r['mode']==m for r in records) for m in sorted({r['mode'] for r in records})}
    summary['n_logs']=len(records)
    summary['n_depth_fits']=len(ppfs)
    dump(out/'challenge_statistics.json',dict(excluded_eval_logs=sorted(excluded),summary=summary,logs=records))
    # Presets fixed from challenge image statistics, without external scores.
    # Blur proxy is not an identifiable Gaussian PSF: these are explicit stress
    # approximations, not a fitted generative scan model.
    ppf=summary['px_per_ft'] or dict(p10=1.5,p50=3.,p90=7.5)
    skew=max(.05,min(.8,abs(summary['skew_deg']['p90'])))
    noise=max(1.,min(10.,summary['noise_sigma']['p90']))
    presets={}
    for i,level in enumerate(LEVELS):
        presets[level]=dict(px_per_ft=ppf['p90' if i==0 else 'p50' if i==1 else 'p10'],
            blur_sigma=(.25,.55,1.,.65)[i],noise_sigma=noise*(.4,.75,1.5,1.)[i],
            saturation=(.95,.75,.45,0.)[i],fade=(.02,.07,.15,.08)[i],
            colour_shift=[(1.,.995,.98),(1.,.98,.94),(1.,.95,.85),(1.,1.,1.)][i],
            skew_deg=skew*(.3,.6,1.,1.)[i],jpeg_quality=(90,65,40,55)[i],one_bit=i==3)
    dump(out/'presets.json',dict(seed=20261108,levels=presets,
        notes='PPF quantiles from challenge OCR; photometric stress approximations. Horizontal shear preserves y.'))


def geometry(size, scale, skew_deg):
    w,h=size
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    sx,sy=nw/w,nh/h
    shear=math.tan(math.radians(skew_deg))
    offset=max(0.,-shear*nh)
    width=nw+math.ceil(abs(shear)*nh)
    return dict(sx=sx,sy=sy,shear=shear,offset=offset,width=width,height=nh)


def transform_box(box,g):
    x0,y0,x1,y1=box
    ya,yb=round(y0*g['sy']),round(y1*g['sy'])
    xs=[x*g['sx']+g['shear']*y+g['offset'] for x in (x0,x1) for y in (ya,yb)]
    return [max(0,math.floor(min(xs))),ya,min(g['width'],math.ceil(max(xs))),yb]


def transform_rows(rows,g):
    result=[]
    for row in rows:
        r=dict(row)
        r['top_y']=round(row['top_y']*g['sy'])
        r['bottom_y']=round(row['bottom_y']*g['sy'])
        if r['bottom_y']<=r['top_y']:
            raise ValueError('Scale would erase an exact truth interval')
        result.append(r)
    return result


def degrade(rgb,preset,seed):
    rng=np.random.default_rng(seed)
    information_scale=preset.get('information_scale',1.)
    if information_scale<1.:
        h,w=rgb.shape[:2]
        reduced=cv2.resize(rgb,(max(1,round(w*information_scale)),max(1,round(h*information_scale))),interpolation=cv2.INTER_AREA)
        rgb=cv2.resize(reduced,(w,h),interpolation=cv2.INTER_LINEAR)
    a=(cv2.GaussianBlur(rgb,(0,0),preset['blur_sigma']) if preset['blur_sigma']>0 else rgb).astype(np.float32)
    gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)[:,:,None]
    a=gray+(a-gray)*preset['saturation']
    a=(a*(1-preset['fade'])+255*preset['fade'])*np.array(preset['colour_shift'],np.float32)
    a+=rng.normal(0,preset['noise_sigma'],a.shape).astype(np.float32)
    a=np.clip(a,0,255).astype(np.uint8)
    if preset['jpeg_quality'] is not None:
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(a,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,preset['jpeg_quality']])
        if not ok:raise RuntimeError('JPEG encoding failed')
        a=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    if preset['one_bit']:
        # PIL Floyd-Steinberg dithering is deterministic.
        return np.asarray(Image.fromarray(a).convert('1')).astype(np.uint8)*255
    return a


def build(source,out,levels=None):
    if (out/'evaluation_freeze.json').exists():raise RuntimeError('Generated data is frozen after evaluation starts')
    config=json.loads((out/'presets.json').read_text())
    items={s:checked_items(source,s) for s in ('dev','test')}
    names={v['log'] for values in items.values() for v in values}
    with (source/'manifest.csv').open(newline='',encoding='utf-8-sig') as f:
        manifest=[r for r in csv.DictReader(f) if r['id'] in names]
    meta={r['id']:r for r in manifest}
    for level,preset in config['levels'].items():
        if levels is not None and level not in levels:continue
        root=out/level
        geometries={}
        for name in sorted(names):
            annotation=json.loads((source/'bbox_annotation'/f'{name}.json').read_text())
            path=source/'images'/annotation['image_name']
            with Image.open(path) as im:
                requested_scale=preset['px_per_ft']/float(meta[name]['px_per_ft'])
                # Integer interval rows cannot represent two classes in one
                # output row: guarantee every source row survives, independent
                # of labels/split. Simulate lower information resolution inside
                # the retained raster instead of erasing annotation support.
                g=geometry(im.size,max(1.,requested_scale),preset['skew_deg'])
                g['requested_scale']=requested_scale
                g['resolution_floor_applied']=requested_scale<1.
                g['information_scale']=min(1.,requested_scale)
                geometries[name]=g
                root.joinpath('images').mkdir(parents=True,exist_ok=True)
                # External pages are bounded; render in <=1024-row strips to a
                # disk-backed array, avoiding a full-page float/RGB working copy.
                scratch=root/f'.{name}.pixels'
                channels=1 if preset['one_bit'] else 3
                shape=(g['height'],g['width']) if channels==1 else (g['height'],g['width'],3)
                output=np.memmap(scratch,dtype=np.uint8,mode='w+',shape=shape)
                sx,sy=g['sx'],g['sy']
                for y in range(0,g['height'],1024):
                    end=min(g['height'],y+1024)
                    a=max(0,math.floor(y/sy)-8);b=min(im.height,math.ceil(end/sy)+8)
                    crop=np.asarray(im.crop((0,a,im.width,b)).convert('RGB'))
                    matrix=np.array([[sx,g['shear']*sy,g['offset']+g['shear']*a*sy],[0,sy,a*sy-y]],np.float32)
                    rgb=cv2.warpAffine(crop,matrix,(g['width'],end-y),flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
                    seed=int.from_bytes(hashlib.sha256(f'{config["seed"]}:{level}:{name}:{y}'.encode()).digest()[:8],'little')
                    output[y:end]=degrade(rgb,dict(preset,information_scale=g['information_scale']),seed)
                output.flush()
                image=Image.fromarray(output)
                if preset['one_bit']:image=image.convert('1',dither=Image.Dither.NONE)
                image.save(root/'images'/f'{name}.tif',compression='tiff_lzw')
                del image,output
                scratch.unlink()
            annotation.update(image_name=f'{name}.tif',width=g['width'],height=g['height'])
            annotation['bounding_boxes']={k:transform_box(v,g) for k,v in annotation['bounding_boxes'].items()}
            dump(root/'bbox_annotation'/f'{name}.json',annotation)
            io.save_rows(root/'truth'/f'{name}.csv',transform_rows(io.load_rows(source/'truth'/f'{name}.csv'),g))
        combined=[]
        for split,values in items.items():
            transformed=[dict(v,y0=round(v['y0']*geometries[v['log']]['sy']),y1=round(v['y1']*geometries[v['log']]['sy'])) for v in values]
            dump(root/f'eval_set_{split}.json',transformed)
            combined.extend(transformed)
        dump(root/'eval_set.json',combined)
        dump(root/'geometry.json',geometries)
        with (root/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(manifest[0]))
            writer.writeheader()
            for r in manifest:
                writer.writerow(dict(r,px_per_ft=float(r['px_per_ft'])*geometries[r['id']]['sy']))
        print('built',level,len(names),flush=True)



def validate(source,out):
    """Audit coordinate bookkeeping; never used to choose preset parameters."""
    counts={}
    for level in LEVELS:
        root=out/level
        geometries=json.loads((root/'geometry.json').read_text())
        nrows=0
        for name,g in geometries.items():
            original=json.loads((source/'bbox_annotation'/f'{name}.json').read_text())
            annotation=json.loads((root/'bbox_annotation'/f'{name}.json').read_text())
            assert annotation['bounding_boxes']=={k:transform_box(v,g) for k,v in original['bounding_boxes'].items()}
            with Image.open(root/'images'/annotation['image_name']) as image:
                assert image.size==(g['width'],g['height'])
                assert image.mode==('1' if level=='1-bit' else 'RGB')
            truth=io.load_rows(root/'truth'/f'{name}.csv')
            assert truth==transform_rows(io.load_rows(source/'truth'/f'{name}.csv'),g)
            assert all(0<=r['top_y']<r['bottom_y']<=g['height'] for r in truth)
            assert all(a['bottom_y']<=b['top_y'] for a,b in zip(truth,truth[1:]))
            nrows+=len(truth)
        combined=[]
        for split in ('dev','test'):
            original=checked_items(source,split)
            values=json.loads((root/f'eval_set_{split}.json').read_text())
            assert len(original)==len(values)
            for a,b in zip(original,values):
                assert b==dict(a,y0=round(a['y0']*geometries[a['log']]['sy']),y1=round(a['y1']*geometries[a['log']]['sy']))
            combined.extend(values)
        dump(root/'eval_set.json',combined)
        counts[level]=dict(logs=len(geometries),windows=len(combined),truth_intervals=nrows,resolution_floor_pages=sum(g.get('resolution_floor_applied',False) for g in geometries.values()))
    dump(out/'validation.json',dict(passed=True,levels=counts))
    print(json.dumps(counts,indent=2))



def audit(source,out):
    """Report achieved DEV pixel statistics without changing any preset."""
    summaries={}
    names=sorted({r['log'] for r in checked_items(source,'dev')})
    for level in ('clean',)+LEVELS:
        root=source if level=='clean' else out/level
        io.DATASET=root
        samples=[]
        for name in names:
            log=io.open_log(name)
            try:
                x0,y0,x1,y1=log.boxes['interpreted_lithology_column']
                for frac in (.15,.5,.85):
                    y=round(y0+max(0,y1-y0-512)*frac)
                    samples.append(stats(io.read_rgb(log,[x0,y,x1,min(y1,y+512)])))
            finally:log.close()
        summaries[level]={}
        for key in ('laplacian_variance','noise_sigma','saturation','edge_blur_proxy','skew_deg'):
            values=[r[key] for r in samples if r[key] is not None]
            summaries[level][key]=dict(zip(('p10','p50','p90'),map(float,np.quantile(values,[.1,.5,.9])))) if values else None
        print('audited',level,flush=True)
    dump(out/'achieved_dev_statistics.json',summaries)


def strategies():
    return sorted(p for p in (REPO/'bakeoff/strategies').glob('s*.py') if int(p.name[1:3])==11 or int(p.name[1:3])>=13)


def evaluate(source,out,split):
    if split != 'dev':
        raise RuntimeError('Only ext_dev is allowed. The orchestrator owns the final benchmark.')
    checked_items(source, 'dev')
    import torch
    from harness import legend, legend_v3, legend_v2, depth, score
    from harness.run import _clean, Resources, aggregate
    torch.set_num_threads(3)
    modules=[importlib.import_module('strategies.'+p.stem) for p in strategies()]
    # Snapshot code and learned weights before evaluating held-out test.
    files=strategies()+list((REPO/'bakeoff/ml').glob('**/*.pt'))+list((REPO/'bakeoff/ml').glob('**/*.npz'))+list((REPO/'bakeoff/ml').glob('**/selected.json'))
    files += [Path(__file__)]+list((REPO/'bakeoff/ml').glob('**/*.pkl'))+list((REPO/'bakeoff/ml').glob('**/*.py'))+list((REPO/'bakeoff/harness').glob('*.py'))+[REPO/'bakeoff/strategies/s02_features_pelt.py']
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    lock=out/'evaluation_freeze.json'
    freeze=dict(hashes=hashes,presets=json.loads((out/'presets.json').read_text()))
    if lock.exists() and json.loads(lock.read_text())!=freeze:
        raise RuntimeError('Strategies/weights/presets changed since freeze; refusing incomparable results')
    dump(lock,freeze)
    if split=='test':
        for level in ('clean',)+LEVELS:
            if not (out/'results'/'dev'/level/'metrics.json').exists():raise RuntimeError('Complete dev evaluation first')
    with (source/'manifest.csv').open(newline='',encoding='utf-8-sig') as f:meta={r['id']:r for r in csv.DictReader(f)}
    original_root=io.ROOT
    for level in ('clean',)+LEVELS:
        root=source if level=='clean' else out/level
        result_dir=out/'results'/split/level
        result_path=result_dir/'metrics.json'
        # Resume only at completed window-strategy records; never rerun test.
        records=json.loads(result_path.read_text())['records'] if result_path.exists() else []
        done={(r['strategy'],r['log'],tuple(r['window'])) for r in records}
        io.DATASET=root
        # Dedicated cache per condition, shared across the two disjoint splits.
        io.ROOT=out/'preprocessing'/level
        legend_v3.HOME=io.ROOT/'legend_v3'
        legend_v3.HOME.mkdir(parents=True,exist_ok=True)
        selected=REPO/'bakeoff/ml/legend_names/selected.json'
        if selected.exists():(legend_v3.HOME/'selected.json').write_bytes(selected.read_bytes())
        legend_v2.CACHE=io.ROOT/'legend_v2'
        items=checked_items(root,split)
        for item in items:
            name=item['log'];win=(item['y0'],item['y1'])
            pending=[m for m in modules if (m.__name__.split('.')[-1],name,win) not in done]
            if not pending:continue
            log=io.open_log(name)
            try:
                prep_error=None;calib=None;entries=[];prep_stage='legend'
                prep_start=time.perf_counter()
                prep_resource=Resources()
                prep_resource.__enter__()
                try:
                    entries=legend.parse(log)
                    prep_stage='depth'
                    calib=depth.calibrate(log)
                    prep_stage=None
                except Exception as exc:prep_error=f'{type(exc).__name__}: {exc}'
                prep_resource.__exit__(None,None,None)
                prep_seconds=time.perf_counter()-prep_start
                truth=io.load_rows(root/'truth'/f'{name}.csv')
                for module in pending:
                    strategy=module.__name__.split('.')[-1]
                    record=dict(strategy=strategy,log=name,window=win,source=meta[name]['source'],well=meta[name]['well'],scored=True)
                    record.update(preprocessing_seconds=prep_seconds,preprocessing_failure=prep_stage,legend_entries=len(entries),lithology_entries=sum(e.is_lithology for e in entries))
                    rows=[]
                    start=time.perf_counter()
                    with Resources() as resource:
                        try:
                            if prep_error:raise RuntimeError(prep_error)
                            cache=result_dir/'cache'/strategy;cache.mkdir(parents=True,exist_ok=True)
                            value=module.run(log,win,dict(legend=entries,calib=calib,device='cpu',cache_dir=cache))
                            rows=_clean(value[0] if isinstance(value,tuple) else value,win,calib)
                        except Exception as exc:record['error']=f'{type(exc).__name__}: {exc}'
                    record.update(seconds=time.perf_counter()-start,peak_rss_gb=max(resource.peak,prep_resource.peak)/1024**3)
                    record.update(score.score(rows,truth,win))
                    key=f'{name}__y{win[0]}_{win[1]}'
                    io.save_rows(result_dir/strategy/f'{key}.csv',rows)
                    records.append(record)
                    dump(result_path,dict(records=records,mean={m.__name__.split('.')[-1]:aggregate([r for r in records if r['strategy']==m.__name__.split('.')[-1]]) for m in modules}))
                    print(split,level,strategy,name,round(record['row_acc'],3),record.get('error',''),flush=True)
            finally:log.close()
    io.ROOT=original_root



def diagnose(source,out):
    """DEV-only S11 interventions, preserving clean OCR labels (never GT labels).

    Restore clean image-derived depth calibration, then restore clean legend
    detection/OCR while still sampling swatches from the degraded image.
    These are diagnostic interventions, not strategies selected for test.
    """
    import torch
    from dataclasses import replace
    from harness import legend, depth, score
    from harness.run import _clean
    from strategies import s11_fft_legend as strategy
    torch.set_num_threads(3)
    items=checked_items(source,'dev')
    records=[]
    for item in items:
        name=item['log']
        io.DATASET=source;io.ROOT=out/'preprocessing'/'clean'
        original=io.open_log(name)
        try:
            try:clean_entries=legend.parse(original);clean_calib=depth.calibrate(original)
            except Exception as exc:
                records.append(dict(log=name,original_window=[item['y0'],item['y1']],error=str(exc),skipped=True))
                continue
            for level in LEVELS:
                root=out/level
                g=json.loads((root/'geometry.json').read_text())[name]
                io.DATASET=root;io.ROOT=out/'preprocessing'/level
                log=io.open_log(name)
                win=(round(item['y0']*g['sy']),round(item['y1']*g['sy']))
                truth=io.load_rows(root/'truth'/f'{name}.csv')
                calib=depth.Calib(clean_calib.a/g['sy'],clean_calib.b,clean_calib.n_points,clean_calib.rmse_px*g['sy'],'transformed-clean-OCR')
                try:
                    for mode in ('restore_depth','restore_depth_and_legend'):
                        error=None;rows=[]
                        try:
                            if mode=='restore_depth':entries=legend.parse(log)
                            else:
                                entries=[]
                                for e in clean_entries:
                                    box=transform_box(e.swatch_box,g)
                                    swatch=io.read_rgb(log,box)
                                    entries.append(replace(e,swatch_box=box,swatch=swatch,mean_rgb=tuple(swatch.mean((0,1)))))
                            value=strategy.run(log,win,dict(legend=entries,calib=calib,device='cpu',cache_dir=out/'diagnostic_cache'))
                            rows=_clean(value[0] if isinstance(value,tuple) else value,win,calib)
                        except Exception as exc:error=str(exc)
                        record=dict(log=name,original_window=[item['y0'],item['y1']],window=win,level=level,mode=mode,**score.score(rows,truth,win))
                        if error:record['error']=error
                        records.append(record)
                    print('diagnosed',level,name,flush=True)
                finally:log.close()
        finally:original.close()
        dump(out/'dev_diagnostics.json',records)
    dump(out/'dev_diagnostics.json',records)



def ablate(source,out):
    """DEV crop study: first window of each well, fixed clean OCR labels.

    Identifies feature sensitivity to individual degradation components. This
    is conditional on clean OCR and uses crop-local shear, not an end-to-end
    re-evaluation of the generated pages. No settings are selected from it.
    """
    import torch
    from dataclasses import replace
    from harness import legend, depth, score
    from harness.run import _clean
    from strategies import s11_fft_legend as strategy
    torch.set_num_threads(3)
    with (source/'manifest.csv').open(newline='',encoding='utf-8-sig') as f:meta={r['id']:r for r in csv.DictReader(f)}
    items=checked_items(source,'dev')
    selected={}
    for item in items:selected.setdefault(meta[item['log']]['well'],item)
    presets=json.loads((out/'presets.json').read_text())['levels']
    records=[]
    io.DATASET=source;io.ROOT=out/'preprocessing'/'clean'
    for well,item in selected.items():
        name=item['log'];original=io.open_log(name)
        try:
            entries=legend.parse(original)
            rgb=io.read_rgb(original,'interpreted_lithology_column',item['y0'],item['y1'])
            raw_truth=io.load_rows(source/'truth'/f'{name}.csv')
            for level in ('heavy','1-bit'):
                base=dict(presets[level],information_scale=min(1.,presets[level]['px_per_ft']/float(meta[name]['px_per_ft'])))
                modes=['all','no_resolution_loss','no_blur','no_noise','no_colour_fade','no_jpeg','no_skew']
                if level=='1-bit':modes.append('no_dither')
                for mode in modes:
                    p=dict(base)
                    if mode=='no_resolution_loss':p['information_scale']=1.
                    if mode=='no_blur':p['blur_sigma']=0.
                    if mode=='no_noise':p['noise_sigma']=0.
                    if mode=='no_colour_fade':p.update(saturation=1.,fade=0.,colour_shift=[1.,1.,1.])
                    if mode=='no_jpeg':p['jpeg_quality']=None
                    if mode=='no_skew':p['skew_deg']=0.
                    if mode=='no_dither':p['one_bit']=False
                    seed=int.from_bytes(hashlib.sha256(f'ablation:{well}:{level}'.encode()).digest()[:8],'little')
                    def render(pixels):
                        g=geometry((pixels.shape[1],len(pixels)),1.,p['skew_deg'])
                        matrix=np.array([[1.,g['shear'],g['offset']],[0,1.,0]],np.float32)
                        warped=cv2.warpAffine(pixels,matrix,(g['width'],g['height']),borderValue=(255,255,255))
                        a=degrade(warped,p,seed)
                        return np.repeat(a[:,:,None],3,axis=2) if a.ndim==2 else a
                    column=render(rgb)
                    altered=[replace(e,swatch=render(e.swatch)) for e in entries if e.swatch.size]
                    log=io.Log(Image.fromarray(column),dict(interpreted_lithology_column=[0,0,column.shape[1],len(column)]),name)
                    truth=[dict(top_y=r['top_y']-item['y0'],bottom_y=r['bottom_y']-item['y0'],lithology=r['lithology']) for r in raw_truth]
                    try:
                        value=strategy.run(log,(0,len(column)),dict(legend=altered,calib=depth.Calib(1,0,0,0,'diagnostic'),device='cpu',cache_dir=out/'diagnostic_cache'))
                        rows=_clean(value[0] if isinstance(value,tuple) else value,(0,len(column)),depth.Calib(1,0,0,0,'diagnostic'))
                        metrics=score.score(rows,truth,(0,len(column)))
                        records.append(dict(well=well,source=meta[name]['source'],log=name,level=level,mode=mode,**metrics))
                    finally:log.close()
                print('ablated',well,level,flush=True)
        finally:original.close()
        dump(out/'dev_component_ablations.json',records)
    dump(out/'dev_component_ablations.json',records)


def report(out):
    """Write dev results only. Never open test scores or truth."""
    datasets={level:json.loads((out/'results'/'dev'/level/'metrics.json').read_text())['records']
              for level in ('clean',)+LEVELS}
    rows=[]
    for strategy in sorted({r['strategy'] for r in datasets['clean']}):
        for source in ['ALL']+sorted({r['source'] for r in datasets['clean']}):
            for level,records in datasets.items():
                values=[r for r in records if r['strategy']==strategy and (source=='ALL' or r['source']==source)]
                clean=[r for r in datasets['clean'] if r['strategy']==strategy and (source=='ALL' or r['source']==source)]
                row=dict(split='dev',strategy=strategy,source=source,level=level,n=len(values),errors=sum('error' in r for r in values))
                for metric in ('row_acc','boundary_f1','cluster_ari','score','depth_mae_ft'):
                    v=[r[metric] for r in values if r.get(metric) is not None]
                    c=[r[metric] for r in clean if r.get(metric) is not None]
                    row[metric]=float(np.mean(v)) if v else None
                    row[metric+'_drop']=float(np.mean(c)-np.mean(v)) if v and c else None
                rows.append(row)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(out/'summary.json',rows)
    lines=['# Scan test on dev data', '',
        'This run uses 51 dev windows from five wells. Each window has the same weight. No model or setting is selected here.',
        'The test set has no score in this report. The orchestrator will run the final test. Shared verify and sealed wells are excluded.', '',
        '| Strategy | Source | Clean accuracy | Light drop | Medium drop | Heavy drop | 1-bit drop |',
        '|---|---|---:|---:|---:|---:|---:|']
    for strategy in sorted({r['strategy'] for r in rows}):
        for source in ['ALL']+sorted({r['source'] for r in rows if r['source']!='ALL'}):
            group={r['level']:r for r in rows if r['strategy']==strategy and r['source']==source}
            drops=' | '.join(f'{100*group[level]["row_acc_drop"]:.2f}' for level in LEVELS)
            lines.append(f'| {strategy[:3]} | {source} | {100*group["clean"]["row_acc"]:.2f} | {drops} |')
    lines += ['', 'Accuracy and drops are in percentage points. A positive drop is a loss.', '',
        '## Image measurements', '',
        'The survey uses only the 38 challenge logs outside the eval set. It samples three small crops from each log. Depth scale uses OCR anchors from image pixels. It uses no truth labels.', '',
        '| Measure | p10 | Median | p90 |', '|---|---:|---:|---:|']
    survey=json.loads((out/'challenge_statistics.json').read_text())
    for key in ('px_per_ft','laplacian_variance','noise_sigma','saturation','edge_blur_proxy','skew_deg'):
        v=survey['summary'][key]
        lines.append(f'| {key} | {v["p10"]:.3f} | {v["p50"]:.3f} | {v["p90"]:.3f} |')
    lines += ['', f'Image modes: {survey["summary"]["modes"]}. Valid depth fits: {survey["summary"]["n_depth_fits"]}.', '',
        'The edge and noise measures also depend on the printed pattern. The levels are fixed stress tests. They are not a fitted scanner model.',
        'The light and medium levels use about 20 and 10 pixels per foot. Heavy and 1-bit use about 2 pixels per foot of image detail. They keep the original raster size to preserve each exact truth row. The image is reduced and enlarged to this raster. The geometry file records this limit.',
        'Skew uses horizontal shear. This keeps truth boundaries horizontal. All boxes, windows, and rows use the same scale. Physical depth limits do not change.', '',
        '## Failed steps', '',
        '| Level | Windows | Legend errors | Depth errors | Empty legends | Peak RAM (GB) |',
        '|---|---:|---:|---:|---:|---:|']
    failures=[]
    for level,records in datasets.items():
        unique={(r['log'],tuple(r['window'])):r for r in records}
        row=dict(level=level,windows=len(unique),legend_failures=sum(r.get('preprocessing_failure')=='legend' for r in unique.values()),
                 depth_failures=sum(r.get('preprocessing_failure')=='depth' for r in unique.values()),
                 empty_legends=sum(r.get('lithology_entries',0)==0 for r in unique.values()),peak_rss_gb=max(r['peak_rss_gb'] for r in records))
        failures.append(row)
        lines.append(f'| {level} | {row["windows"]} | {row["legend_failures"]} | {row["depth_failures"]} | {row["empty_legends"]} | {row["peak_rss_gb"]:.2f} |')
    dump(out/'failure_summary.json',failures)
    lines += ['', '## Scope', '',
        'The learned models are inherited. Earlier training and selection used six dev wells, including ForsythB2-17. These weights have historical shared-verify exposure. This lane does not train or tune them. These scores measure dev sensitivity, not held-out model quality.',
        'No final test score or in-domain eval truth was read for this work. Old scanify scores are not part of this report.',
        'The boundary match tolerance is in pixels. Its depth width changes when the raster scale changes.',
        'The levels change several image properties at once. A combined score loss alone cannot identify its cause. See dev_diagnostic_summary.json and dev_component_summary.json for separate S11 checks when present.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['measure','build','refresh','validate','audit','evaluate','diagnose','ablate','report'])
    parser.add_argument('--source',type=Path,default=REPO/'external_tests')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--dataset',type=Path,default=Path(os.environ.get('LITHOLOGY_DATASET',REPO.parent/'dataset')).expanduser())
    parser.add_argument('--levels',nargs='+',choices=LEVELS,help='Build only these levels; useful after interrupted generation')
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    args=parser.parse_args()
    if args.action=='measure':measure(args.out,args.dataset)
    elif args.action=='build':build(args.source,args.out,args.levels)
    elif args.action=='refresh':refresh(args.source,args.out)
    elif args.action=='validate':validate(args.source,args.out)
    elif args.action=='audit':audit(args.source,args.out)
    elif args.action=='diagnose':diagnose(args.source,args.out)
    elif args.action=='ablate':ablate(args.source,args.out)
    elif args.action=='evaluate':evaluate(args.source,args.out,args.split)
    else:report(args.out)

if __name__=='__main__':main()
