"""Build full-column candidate intervals: python -X utf8 -B labeltool/build_queue.py."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import sys
import time

sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from labeltool.common import (ROOT, DATA, CACHE, GOLD, family, harness_for, load_rows,
                              external_sources, suggestions_for, tier_for, write_json)



# Same colour estimator as ../scripts/label_mix.py, copied without its side effects.
def lab(rgb):
    a = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    a[:, 0] *= 100 / 255
    a[:, 1:] -= 128
    return a


def coloured_mask(rgb):
    mx, mn = rgb.max(-1).astype(int), rgb.min(-1).astype(int)
    ink, paper, grey = mx < 80, mn > 225, (mx - mn) < 18
    return ~ink & ~paper & ~grey, ink


def dominant_lab(rgb):
    col, _ = coloured_mask(rgb)
    px = rgb[col]
    if len(px) < max(20, .05 * rgb.shape[0] * rgb.shape[1]): return None
    q = (px // 24).astype(int)
    keys = q[:, 0]*10000 + q[:, 1]*100 + q[:, 2]
    vals, count = np.unique(keys, return_counts=True)
    return lab(px[keys == vals[count.argmax()]]).mean(0)


def reduced_column(io, log):
    """Four native rows per feature row, <=96px wide, strip-safe throughout."""
    box = log.boxes['interpreted_lithology_column']
    strips = []
    for top in range(box[1], box[3], 4000):
        bottom = min(top + 4000, box[3])
        rgb = io.read_rgb(log, box, top, bottom)
        strips.append(cv2.resize(rgb, (min(96, rgb.shape[1]), math.ceil(len(rgb)/4)), interpolation=cv2.INTER_AREA))
    return np.concatenate(strips)


def segment(rgb, top, bottom, extra=()):
    """Cheap pooled row change points, mapped to native rows; merge <12px slivers."""
    inner = rgb[:, max(1, rgb.shape[1]//10):max(2, rgb.shape[1]-rgb.shape[1]//10)]
    if not inner.size: inner = rgb
    labs = cv2.cvtColor(inner, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)
    feat = np.column_stack((np.median(labs, axis=1), (gray < 200).mean(1)*80,
                            (np.abs(np.diff(gray.astype(float), axis=1))>40).mean(1)*80))
    smooth = cv2.blur(feat, (1, 3))
    delta = np.zeros(len(feat))
    delta[3:-3] = np.linalg.norm(smooth[6:] - smooth[:-6], axis=1)
    # Suppress neighbouring responses to the same boundary, retain short real beds.
    peaks = (delta > 12) & (delta >= cv2.dilate(delta[:, None], np.ones((7, 1))).ravel())
    candidates = [top + round(int(i)*(bottom-top)/len(rgb)) for i in np.flatnonzero(peaks)]
    points = sorted({top, bottom, *candidates, *[int(y) for y in extra if top<y<bottom]})
    merged = [top]
    for y in points[1:-1]:
        if y-merged[-1] >= 12 and bottom-y >= 12: merged.append(y)
    merged.append(bottom)
    # Keep review crops bounded even in an apparently constant column.
    result = [top]
    for a, b in zip(merged, merged[1:]):
        count = max(1, math.ceil((b-a)/1600))
        result.extend(round(a+(b-a)*i/count) for i in range(1, count+1))
    return list(zip(result, result[1:]))


def native_samples(io, log, intervals):
    """Read each TIFF strip once for the S11 native-scale centre patches."""
    box = log.boxes['interpreted_lithology_column']
    requests = {}
    for index, (a, b) in enumerate(intervals):
        y = max(a, (a+b)//2-min(32, (b-a)//2))
        block = box[1] + ((y-box[1])//4000)*4000
        requests.setdefault(block, []).append((index, y, min(b, y+64)))
    patches = {}
    for block, group in requests.items():
        end = min(box[3], max(v[2] for v in group))
        rgb = io.read_rgb(log, box, block, end)
        x = max(0, (rgb.shape[1]-64)//2)
        for index, a, b in group: patches[index] = rgb[a-block:b-block, x:x+64].copy()
    return patches


def fft_descriptor(patch, mono=False):
    """S11's descriptor, with fixed-bin histograms pooled by bincount."""
    from strategies import s11_fft_legend as s11
    n=s11.N
    if min(patch.shape[:2])<2:patch=np.tile(patch,(2,2,1))
    h,w=patch.shape[:2]
    if h<n or w<n:patch=np.tile(patch,(max(1,(n+h-1)//h),max(1,(n+w-1)//w),1))
    patch=patch[:n,:n]
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    ink=float(np.mean(gray<.82))
    f=np.abs(np.fft.fft2((gray-gray.mean())*s11.window2))
    radial=np.bincount((s11.rad*n).astype(int).ravel(),weights=f.ravel(),minlength=n)[1:26]
    peak=(np.argmax(radial)+1)/n if radial.any() else .1
    rel=s11.rad/max(peak,2/n)
    rb=np.clip((np.log2(np.maximum(rel,.25))+2)*2,0,7.999).astype(int)
    ab=np.minimum((s11.ang*12).astype(int),11)
    valid=(s11.rad>=2/n)&(s11.rad<.48)
    bins=np.bincount((rb[valid]*12+ab[valid]),weights=f[valid],minlength=96).reshape(8,12).astype(np.float32)
    bins=np.sqrt(bins);bins/=np.linalg.norm(bins)+1e-8
    labs=cv2.cvtColor(patch,cv2.COLOR_RGB2LAB)
    q=labs.reshape(-1,3).astype(np.int16)
    indices=(q[:,0]//64)*25+((q[:,1]*5)//256)*5+((q[:,2]*5)//256)
    hist=np.bincount(indices,minlength=100).astype(np.float32);hist/=hist.sum()+1e-8
    colour=np.median(q,axis=0).astype(np.float32)/255
    return bins,ink,hist,colour,peak


def fft_references(refs):
    return (np.asarray([[np.roll(r[0],shift,axis=0) for shift in (-1,0,1)] for r in refs]),
            np.asarray([r[1] for r in refs]),np.asarray([r[2] for r in refs]),np.asarray([r[3] for r in refs]))


def fft_distances(desc, refs, mono):
    spectra,ink,hist,colour=refs
    spec=np.linalg.norm(desc[0][None,None]-spectra,axis=(2,3)).min(axis=1)/1.414
    value=.55*spec+.45*np.minimum(1.,np.abs(desc[1]-ink)*2)
    if not mono:
        hd=.5*np.abs(desc[2]-hist).sum(axis=1)
        cd=np.minimum(1.,np.linalg.norm(desc[3]-colour,axis=1)*2)
        value=.45*value+.35*hd+.20*cd
    return value


def build_log(name):
    start = time.perf_counter()
    cv2.setNumThreads(1)
    io, legend, depth = harness_for(name)
    from strategies import s11_fft_legend as s11
    log = io.open_log(name)
    errors = []
    try:
        entries = [e for e in legend.parse(log) if e.is_lithology and e.swatch.size]
        box = list(map(int, log.boxes['interpreted_lithology_column']))
        rgb = reduced_column(io, log)
        sources = external_sources(name)
        for key, folder in [('S11/saved', ROOT/'out'/'s11_fft_legend'),
                            ('silver/A', ROOT/'gt'/'silver'/'A'), ('silver/B', ROOT/'gt'/'silver'/'B')]:
            rows = load_rows(folder/(name+'.csv'))
            if rows: sources[key] = rows
        if name == GOLD: sources['gold'] = load_rows(ROOT/'gt'/'gold'/(name+'.csv'))
        extra = [r[edge] for rows in sources.values() for r in rows for edge in ('top_y', 'bottom_y')]
        intervals = segment(rgb, box[1], box[3], extra)
        patches = native_samples(io, log, intervals) if entries else {}
        mono = log.mode == '1'
        refs = [fft_descriptor(e.swatch, mono) for e in entries]
        reference_batch=fft_references(refs) if refs else None
        colours = [(e.label, dominant_lab(e.swatch)) for e in entries]
        colours = [(name, value) for name, value in colours if value is not None]
        items = []
        for i, (a, b) in enumerate(intervals):
            lo = max(0, round((a-box[1])*len(rgb)/(box[3]-box[1])))
            hi = max(lo+1, round((b-box[1])*len(rgb)/(box[3]-box[1])))
            patch = rgb[lo:hi]
            inset = max(1, patch.shape[1]//10)
            inner = patch[:, inset:-inset] if patch.shape[1]>2*inset else patch
            suggestions = suggestions_for(sources, a, b)
            if refs:
                desc = fft_descriptor(patches[i], mono)
                ds = fft_distances(desc,reference_batch,mono)
                order = np.argsort(ds)
                gap = float(ds[order[1]]-ds[order[0]]) if len(ds)>1 else 1.
                label = entries[int(order[0])].label if ds[order[0]]<=.65 and gap>=.001 else 'Unknown'
                suggestions.insert(0, dict(source='S11/native FFT', label=label, tier='medium',
                    reason='S11 descriptor and distance on native-scale centre sample',
                    distance=round(float(ds[order[0]]),4), margin=round(gap,4)))
            colour = dominant_lab(inner)
            d1, margin = None, None
            if colour is not None and colours:
                distances = np.array([np.linalg.norm(colour-value) for _, value in colours])
                order = np.argsort(distances)
                d1 = round(float(distances[order[0]]), 3)
                margin = round(float(distances[order[1]]-d1),3) if len(order)>1 else 99.
                suggestions.append(dict(source='colour/Lab', label=colours[int(order[0])][0], d1=d1,
                    margin=margin, tier='high' if d1<=12 and margin>=15 else 'medium' if d1<=25 else 'low',
                    reason=f'Nearest swatch: d1={d1}, margin={margin}'))
            tier, reason = tier_for(suggestions, d1, margin, bool(entries))
            best = next((s['label'] for s in suggestions if s['source']=='colour/Lab' and d1<=25),
                        next((s['label'] for s in suggestions if not s.get('truth')), 'Unknown'))
            identity = hashlib.sha1(f'{name}:{a}:{b}'.encode()).hexdigest()[:16]
            items.append(dict(id=identity, log=name, family=family(name), top_y=a, bottom_y=b,
                 suggestions=suggestions, tier=tier, reason=reason, suggestion=best, d1=d1, margin=margin,
                 priority={'low':0,'medium':1,'high':2}[tier], truth_only=name==GOLD,
                 disputed=any(s.get('disputed') for s in suggestions)))
        try:
            calib = depth.calibrate(log).__dict__
        except Exception as exc:
            # A missing calibration is explicit; no invented depths are written.
            calib = None
            errors.append('Depth calibration: '+str(exc))
        assets = CACHE / 'assets' / name
        assets.mkdir(parents=True, exist_ok=True)
        tiles = []
        for e in entries:
            Image.fromarray(e.swatch).save(assets/f'swatch_{e.id}.png')
            tiles.append(dict(id=e.id, label=e.label, width=e.swatch.shape[1], height=e.swatch.shape[0]))
        legend_rgb = io.read_rgb(log, 'lithology_legend')
        Image.fromarray(legend_rgb).save(assets/'legend.png')
        # Full strip preserves native proportions. The UI uses a scrollable viewport.
        thumb_scale = min(1., 30/rgb.shape[1])
        thumb = cv2.resize(rgb, (max(1,round((box[2]-box[0])*min(1,30/(box[2]-box[0])))),
                         max(1,round((box[3]-box[1])*min(1,30/(box[2]-box[0]))))), interpolation=cv2.INTER_AREA)
        Image.fromarray(thumb).save(assets/'strip.png')
        meta = dict(log=name, family=family(name), box=box, tiles=tiles, calibration=calib,
                    legend_parser=legend.__name__, errors=errors, seconds=round(time.perf_counter()-start,3))
        write_json(DATA/'logs'/(name+'.json'), meta)
        return items, meta
    finally:
        log.close()


def build(names=None, workers=3, output=DATA):
    from harness import io
    started = time.perf_counter()
    names = names or io.list_logs()
    items, metas = [], []
    if workers == 1:
        results = (build_log(name) for name in names)
        for rows, meta in results:
            items.extend(rows); metas.append(meta)
            print(f'{len(metas)}/{len(names)} {meta["log"]} {len(rows)} intervals {meta["seconds"]}s', flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(build_log, name): name for name in names}
            for future in as_completed(futures):
                rows, meta = future.result()
                items.extend(rows); metas.append(meta)
                print(f'{len(metas)}/{len(names)} {meta["log"]} {len(rows)} intervals {meta["seconds"]}s', flush=True)
    items.sort(key=lambda r:(r['log'], r['top_y']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output/'queue.jsonl.tmp'
    with temporary.open('w',encoding='utf-8') as f:
        for item in items: f.write(json.dumps(item, ensure_ascii=False)+'\n')
    temporary.replace(output/'queue.jsonl')
    tiers = {tier:sum(r['tier']==tier and not r['truth_only'] for r in items) for tier in ('low','medium','high')}
    families = {}
    for meta in metas:
        record = families.setdefault(meta['family'],dict(logs=0,low=0,medium=0,high=0,truth=0))
        record['logs'] += 1
    for item in items: families[item['family']]['truth' if item['truth_only'] else item['tier']] += 1
    summary = dict(logs=len(metas), seconds=round(time.perf_counter()-started,3), tiers=tiers,
                   families=families, truth_intervals=sum(r['truth_only'] for r in items),
                   errors={m['log']:m['errors'] for m in metas if m['errors']})
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return items, summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',nargs='+',help='Optional exact log stems; default all 51')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    build(args.logs,args.workers)
