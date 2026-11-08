"""Independent Codex vision labelling; all generated files stay in labelling/.

Run from the bakeoff root: python -X utf8 labelling/llm_label.py
Gold is opened only by validate(), after both labellers' CSVs are saved.
"""
from __future__ import annotations
import argparse
import base64
import concurrent.futures as futures
import csv
import hashlib
import html
import io as bio
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'labelling'
OUT = BASE / 'out'
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from harness import io, depth, legend, score
from classify.sample2 import CallBudget

MODELS = {'S': ('gpt-6.1-sol', 'high'), 'T': ('gpt-6.1-terra', 'medium')}
GOLD_LOG = '0000000009_d-582_0000020435-0001_log06_fe_5in_md'
FIELDS = ['top_depth', 'bottom_depth', 'lithology', 'top_y', 'bottom_y', 'cluster_id']
LOCK = threading.Lock()
LIMIT = 220
BUDGET = CallBudget(OUT / 'calls.json')


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def font(size=18):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)


def parse_json(text):
    decoder = json.JSONDecoder()
    for m in re.finditer(r'[\[{]', text):
        try:
            val, _ = decoder.raw_decode(text[m.start():])
            if isinstance(val, (dict, list)):
                return val
        except ValueError:
            pass
    raise ValueError('No JSON returned')


def redact(text):
    return re.sub(r'(?i)(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+\S+|(?:api[_-]?key|access_token|refresh_token)[\s\"=:]+[^\s\",}]+)', '[REDACTED]', text)


def invoke(png, prompt, who, key):
    model, effort = MODELS[who]
    digest = hashlib.sha256(png.read_bytes() + prompt.encode() + model.encode() + effort.encode()).hexdigest()
    dest = OUT / '_responses' / (digest + '.json')
    if dest.exists():
        cached = read(dest)
        if cached.get('ok'):
            return cached['answer']
        # Failed startup attempts are recorded and counted, but are safe to retry.
        dump(dest.with_name(digest + '.attempt_' + str(cached['call']['number']) + '.json'), cached)
    with LOCK:
        call = BUDGET.reserve(key, model, effort, digest)
        print(f"CALL {call['number']}/{LIMIT} {key} {who}", flush=True)
    env = os.environ.copy()
    env['CODEX_HOME'] = 'C:/Users/hssil/.codex-accounts/kira181204'
    env['PYTHONUTF8'] = '1'
    env['CODEX_SQLITE_HOME'] = str(OUT / '_codex_state')
    runtime_tmp=OUT/'_tmp'
    runtime_tmp.mkdir(parents=True,exist_ok=True)
    env['TEMP']=env['TMP']=str(runtime_tmp)
    # Invoke the installed executable through node to avoid cmd.exe prompt quoting.
    script = Path(os.environ['APPDATA']) / 'npm/node_modules/@openai/codex/bin/codex.js'
    prefix = [shutil.which('node'), str(script)] if script.exists() else [shutil.which('codex') or 'codex']
    if os.environ.get('LLM_LABEL_CODEX'):
        prefix=[os.environ['LLM_LABEL_CODEX']]
    cmd = prefix + ['exec', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config', '--sandbox', 'read-only',
                    '-m', model, '-c', f'model_reasoning_effort="{effort}"',
                    '-c', 'mcp_servers={}', '-c', 'features.shell_tool=false',
                    '-c', 'history.persistence="none"', '-c', 'web_search="disabled"',
                    '-c', f'log_dir="{(OUT / "_codex_logs").as_posix()}"',
                    '-c', f'sqlite_home="{(OUT / "_codex_state").as_posix()}"',
                    '-i', str(png.resolve()), '-']
    t = time.perf_counter()
    result = {'ok': False, 'call': call, 'prompt': prompt, 'image': str(png.relative_to(BASE))}
    try:
        proc = subprocess.run(cmd, input=prompt, cwd=BASE, env=env, capture_output=True,
                              text=True, encoding='utf-8', errors='replace', timeout=600)
        result.update(returncode=proc.returncode, stdout=redact(proc.stdout), stderr=redact(proc.stderr))
        if proc.returncode:
            raise RuntimeError(f'Codex exit {proc.returncode}: {redact(proc.stderr)[-1500:]}')
        result.update(answer=parse_json(proc.stdout), ok=True)
    except Exception as exc:
        result['error'] = redact(str(exc))
    result['seconds'] = time.perf_counter() - t
    dump(dest, result)
    with LOCK:
        BUDGET.finish(call, result['ok'], result['seconds'])
    if not result['ok']:
        raise RuntimeError(result['error'])
    return result['answer']


def fit_image(image, max_side=1550, min_side=512):
    scale = min(1.5, max_side / max(image.size))
    image = image.resize(tuple(max(1, round(s * scale)) for s in image.size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (max(min_side, image.width), max(min_side, image.height)), 'white')
    canvas.paste(image, (0, 0))
    return canvas


def vocabulary(log, folder, key):
    path = folder / 'vocab.json'
    if path.exists():
        return read(path)
    box = log.boxes['lithology_legend']
    rgb = io.read_rgb(log, box, box[1], box[3])
    image = fit_image(Image.fromarray(rgb))
    png = folder / 'legend.png'
    image.save(png)
    prompt = ('Read this scanned mud-log legend. Return ONLY JSON: {"entries": '
              '[{"label":"exact printed rock/lithology name", "box":[x0,y0,x1,y1], '
              '"pattern":"brief distinguishing colour and hatch description"}]}. '
              'Coordinates are pixels of this attached image; each box must enclose BOTH the swatch '
              'and its printed name, tightly, with no neighbouring entry. Include every ROCK or '
              'LITHOLOGY entry, preserving exact spelling and capitalization; join wrapped words '
              'with spaces. Ignore modifiers, accessories, minerals, fossils, oil/gas shows and '
              'drilling symbols. Do not invent absent rocks. Read all columns. An empty swatch '
              'can be a valid rock if it has a printed rock name. Do not inspect files or use tools.')
    answer = invoke(png, prompt, 'S', key + ':vocab')
    entries = answer['entries']
    if not entries or any(not isinstance(e.get('label'), str) for e in entries):
        raise ValueError('Empty or malformed vocabulary')
    vocab = {'log': log.name, 'entries': entries, 'image_size': list(image.size),
             'source_box': box, 'method': 'one vision call; no legend.parse dependency'}
    dump(path, vocab)
    return vocab


def rock_panel(folder, vocab):
    source = Image.open(folder / 'legend.png').convert('RGB')
    entries = vocab['entries']
    cols = 2 if len(entries) > 8 else 1
    row_h = min(160, max(58, 1150 // math.ceil(len(entries) / cols)))
    panel = Image.new('RGB', (cols * 400, 42 + math.ceil(len(entries) / cols) * row_h), 'white')
    d = ImageDraw.Draw(panel)
    d.text((8, 6), 'ROCK ENTRIES (swatch + original name)', font=font(17), fill='black')
    for i, e in enumerate(entries):
        x, y = (i % cols) * 400, 42 + (i // cols) * row_h
        b = [int(round(float(v))) for v in e['box']]
        b = [max(0,b[0]-2), max(0,b[1]-2), min(source.width,b[2]+2), min(source.height,b[3]+2)]
        if b[2] <= b[0] or b[3] <= b[1]:
            raise ValueError(f'Bad legend entry box: {e}')
        crop = source.crop(b)
        scale = min(390/crop.width, (row_h-22)/crop.height)
        crop = crop.resize((max(1,round(crop.width*scale)), max(1,round(crop.height*scale))), Image.Resampling.NEAREST)
        panel.paste(crop, (x+4,y))
        d.text((x+4,y+row_h-21), e['label'], font=font(14), fill='black')
    panel.save(folder / 'rock_entries.png')
    return panel


def tiles(y0, y1, height=580):
    # Distribute the remainder so every tile has about 400-600 rows, overlap exactly 60.
    n = max(1, math.ceil((y1-y0-60)/(height-60)))
    span = y1-y0 + 60*(n-1)
    edges = [round(i*span/n) for i in range(n+1)]
    return [(y0+edges[i]-60*i, y0+edges[i+1]-60*i) for i in range(n)]


def tile_image(rgb, a, b, y0, panel, path, iteration):
    scale = 1.8 if iteration == 1 else 2.0
    col = Image.fromarray(rgb[a-y0:b-y0])
    target_w = min(420, max(190, round(col.width*scale)))
    target_h = round((b-a)*scale)
    col = col.resize((target_w,target_h), Image.Resampling.NEAREST)
    width = 105 + target_w + 24 + panel.width
    height = max(512, target_h+92, panel.height+50)
    image = Image.new('RGB', (width,height), 'white')
    image.paste(col,(100,60))
    image.paste(panel,(124+target_w,45))
    d=ImageDraw.Draw(image)
    d.text((8,8), f'COLUMN: native rows [{a}, {b})', font=font(21), fill='black')
    for y in [a] + list(range(math.ceil(a/50)*50,b,50)) + [b]:
        yy=60+round((y-a)*scale)
        d.text((5,yy-9), str(y), font=font(18), fill='black')
        d.line((82,yy,98,yy),fill='black',width=1)
    assert 512 <= image.width <= 1600 and 512 <= image.height <= 1600, image.size
    image.save(path)


def boundary_signal(rgb):
    # Compare colour, ink occupancy, gradient orientation and spatial texture above/below.
    x = rgb[:, max(1,rgb.shape[1]//18):-max(1,rgb.shape[1]//18)]
    small = cv2.resize(x,(64,len(x)),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    gray = cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    feat=np.concatenate([small.mean(1), (gray<.75).mean(1)[:,None],
                         np.abs(gx).mean(1)[:,None],np.abs(gy).mean(1)[:,None],
                         cv2.resize(gray,(8,len(x)),interpolation=cv2.INTER_AREA)],axis=1)
    c=np.vstack([np.zeros((1,feat.shape[1])),np.cumsum(feat,axis=0)])
    signal=np.zeros(len(x))
    r=12
    for y in range(r,len(x)-r):
        signal[y]=np.mean(np.abs((c[y]-c[y-r])/r-(c[y+r]-c[y])/r))
    return signal


def normalize_intervals(answer, a, b, vocab, signal, y0):
    rows=answer.get('intervals',[]) if isinstance(answer,dict) else answer
    allowed={score.normalize(e['label']):e['label'] for e in vocab['entries']}
    parsed=[]
    for row in rows:
        top=max(a,min(b,int(round(float(row['top_y'])))))
        bottom=max(a,min(b,int(round(float(row['bottom_y'])))))
        if bottom<=top: continue
        label=allowed.get(score.normalize(row['label']),'Unknown')
        parsed.append((top,bottom,label,max(0,min(1,float(row.get('confidence',.5))))))
    if not parsed: raise ValueError('Model returned no usable intervals')
    parsed.sort()
    # A midpoint reconciles small numeric gaps; never manufacture a new semantic boundary.
    edges=[a]+[round((p[1]+q[0])/2) for p,q in zip(parsed,parsed[1:])]+[b]
    for i in range(1,len(edges)-1):
        at=edges[i]-y0
        lo=max(1,at-20,edges[i-1]+1-y0)
        hi=min(len(signal)-1,at+20,edges[i+1]-1-y0)
        if hi>=lo:
            candidate=lo+int(np.argmax(signal[lo:hi+1]))
            local=signal[max(0,at-60):min(len(signal),at+61)]
            baseline=float(np.median(local));mad=float(np.median(np.abs(local-baseline)))
            if signal[candidate]>max(.008,baseline+1.5*mad):
                edges[i]=candidate+y0
    return [dict(top_y=edges[i],bottom_y=edges[i+1],label=p[2],confidence=p[3])
            for i,p in enumerate(parsed) if edges[i+1]>edges[i]]


def encode_rows(labels, confidence, y0, calib, agreed=None):
    rows=[]
    ids={v:i for i,v in enumerate(sorted(set(labels)))}
    start=0
    for end in range(1,len(labels)+1):
        if end==len(labels) or labels[end]!=labels[start] or (agreed is not None and agreed[end]!=agreed[start]):
            row=dict(top_y=y0+start,bottom_y=y0+end,lithology=str(labels[start]),cluster_id=ids[labels[start]],
                     top_depth=round(calib.to_depth(y0+start),1),bottom_depth=round(calib.to_depth(y0+end),1))
            if agreed is not None: row['agreed']=bool(agreed[start])
            rows.append(row);start=end
    return rows


def save_csv(path, rows, consensus=False):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS+(['agreed'] if consensus else []))
        w.writeheader()
        for row in rows:
            r=dict(row)
            if consensus:r['agreed']=str(r['agreed']).lower()
            w.writerow(r)


def get_calib(log):
    # Reuse existing harness calibration read-only; compute missing caches under labelling.
    cached=ROOT/'out/_cache/depth'/f'{log.name}.rapidocr.json'
    if cached.exists() and (data:=read(cached)).get('version')==depth.VERSION:
        return depth.Calib(**data['calibration'])
    return depth.calibrate(log)


def label_window(key, item, iteration=1, workers=4):
    folder=OUT/key
    folder.mkdir(parents=True,exist_ok=True)
    log=io.open_log(item['log'])
    started=time.perf_counter()
    try:
        y0,y1=item['rows']
        rgb=io.read_rgb(log,'interpreted_lithology_column',y0,y1)
        vocab=vocabulary(log,folder,key)
        panel=rock_panel(folder,vocab)
        calib=get_calib(log)
        dump(folder/'calibration.json',calib.__dict__)
        signal=boundary_signal(rgb)
        plans=tiles(y0,y1,580 if iteration==1 else 500)
        run=folder/f'iteration_{iteration}'
        run.mkdir(exist_ok=True)
        jobs=[]
        for i,(a,b) in enumerate(plans):
            png=run/f'tile_{i:02d}_{a}_{b}.png'
            tile_image(rgb,a,b,y0,panel,png,iteration)
            prompt=(f'Independently label the lithology COLUMN at left, native image rows [{a},{b}). '
                    'The ruler labels are NATIVE ROWS, not depth and not rendered image pixels. '
                    'Compare the actual hatch strokes, spacing, shapes and colour with rock swatches at right. '
                    'Return ONLY JSON {"intervals":[{"top_y":integer,"bottom_y":integer,"label":string,"confidence":0.0}]}. '
                    'Intervals must be contiguous, sorted, nonoverlapping and cover the entire tile, bottom exclusive. '
                    'Use only these exact labels: '+json.dumps([e['label'] for e in vocab['entries']])+', or "Unknown". '
                    'Trace the dominant pattern across the full column width; subordinate thin streaks or lateral '
                    'percentage strips do not define a whole-row rock change. Detect every vertical transition, '
                    'including thin beds, but ignore scan noise and grid/ruler lines. An unpatterned white fill '
                    'may be a named rock: compare to white/very sparse swatches before choosing Unknown. '
                    'Do not use general geological expectations when the legend pattern differs. '
                    'Estimate transition rows using the nearest 50-row ticks. Do not inspect files or use tools.')
            if iteration>=2:
                prompt+=(' Carefully distinguish diagonal hatching from dots, brickwork, short dashes and crosses. '
                         'Compare repeated motifs, not overall darkness. Text outside the swatch is its name. '
                         'A faint pattern and a truly blank swatch are different. Look at the entire column width '
                         'and do not mistake periodic hatch strokes for a new bed. First identify each pattern '
                         'visually, then locate its top/bottom with the row ruler.')
            for who in MODELS:jobs.append((who,i,a,b,png,prompt))
        answers={who:[] for who in MODELS}
        def work(job):
            who,i,a,b,png,prompt=job
            answer=invoke(png,prompt,who,f'{key}:i{iteration}:tile{i}')
            return who,i,a,b,normalize_intervals(answer,a,b,vocab,signal,y0)
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(work,jobs):
                who,i,a,b,rows=result
                answers[who].append((i,a,b,rows))
        dump(run/'tiles.json',answers)
        raster={}; conf={}; output={}
        for who in MODELS:
            labels=np.full(y1-y0,'Unknown',dtype=object)
            confidence=np.zeros(y1-y0)
            priority=np.full(y1-y0,-np.inf)
            for i,a,b,rows in sorted(answers[who]):
                ys=np.arange(a,b)
                # Keep the prediction furthest from a tile edge in overlaps; confidence breaks ties.
                centrality=np.minimum(ys-a+1,b-ys)
                for row in rows:
                    aa=row['top_y'];bb=row['bottom_y']
                    quality=centrality[aa-a:bb-a]+row['confidence']*.01
                    target=slice(aa-y0,bb-y0)
                    mask=quality>priority[target]
                    labels[target][mask]=row['label']
                    confidence[target][mask]=row['confidence']
                    priority[target][mask]=quality[mask]
            raster[who]=labels;conf[who]=confidence
            output[who]=encode_rows(labels,confidence,y0,calib)
            save_csv(run/f'{who}.csv',output[who])
            save_csv(folder/f'{who}.csv',output[who])
        agree=(raster['S']==raster['T'])
        # Disputes remain a concrete, reviewable best guess; higher confidence wins (S on a tie).
        combined=np.where(conf['T']>conf['S'],raster['T'],raster['S'])
        output['consensus']=encode_rows(combined,np.maximum(conf['S'],conf['T']),y0,calib,agree)
        for dest in (run/'consensus.csv',folder/'consensus.csv'):
            save_csv(dest,output['consensus'],True)
        Image.fromarray(rgb).save(folder/'column.webp',quality=88,method=6)
        shares={str(label):float(np.mean(combined==label)) for label in sorted(set(combined))}
        ledger=read(OUT/'calls.json')
        record=dict(key=key,log=log.name,family=item['family'],rows=[y0,y1],iteration=iteration,
                    n_intervals=len(output['consensus']),agreed_share=float(agree.mean()),labels=shares,
                    calls_used=sum(c['key'].split(':')[0]==key for c in ledger['calls']),
                    seconds=time.perf_counter()-started,calibration=calib.__dict__)
        dump(folder/'summary.json',record)
        print('DONE '+json.dumps(record),flush=True)
        return record,output,agree
    finally:
        log.close()


def validate(record, outputs, agreed):
    # This is the ONLY gold read, deliberately after label_window returns saved predictions.
    gold=io.load_rows(ROOT/'gt/gold'/f'{GOLD_LOG}.csv')
    window=tuple(record['rows'])
    metrics={name:score.score(rows,gold,window) for name,rows in outputs.items()}
    gt=score.raster(gold,window)
    pred=score.raster(outputs['consensus'],window)
    mask=agreed & (gt!='')
    return dict(iteration=record['iteration'],metrics=metrics,agreed_only_accuracy=float(np.mean(pred[mask]==gt[mask])) if mask.any() else None,
                agreed_share=float(agreed.mean()),agreed_scored_rows=int(mask.sum()),
                calls_used=len(read(OUT/'calls.json')['calls']),
                consensus_policy='higher model confidence on disputed rows; S breaks ties',
                gold_read_after_predictions=True)


def review(records):
    palette=['#a9d5f2','#f1cc8e','#d2b9df','#a4d7af','#e9b2ac','#c9d0a0','#a5d5d5','#d6c3a4']
    parts=['<!doctype html><html><meta charset="utf-8"><title>Lithology label review</title><style>',
           'body{font:15px Arial;margin:24px;background:#fafafa;color:#222}section{margin:40px 0;border-top:2px solid #bbb}',
           '.view{display:flex;gap:12px;align-items:flex-start}.track{position:relative;flex:none;border:1px solid #777}',
           '.band{position:absolute;left:0;right:0;box-sizing:border-box;border-top:1px solid #555;overflow:hidden;font-size:12px;padding:2px}',
           '.dispute{background-image:repeating-linear-gradient(135deg,transparent,transparent 5px,#d3282877 5px,#d3282877 8px)}',
           '.head{font-weight:bold;margin-bottom:8px}.legend{max-width:800px}table{border-collapse:collapse;margin:12px 0}td,th{padding:5px 12px;border:1px solid #ccc;text-align:left}',
           'button{padding:8px}details{margin:12px 0}img{image-rendering:auto}</style><body><h1>Lithology label review</h1>',
           '<p>Native image rows; bottom exclusive. Red hatching marks S/T disagreement. Consensus selects the more confident model on disputed rows (S on ties). Agreement is not proof of correctness.</p>']
    def uri(image):
        buf=bio.BytesIO();image.save(buf,format='WEBP',quality=86,method=6)
        return 'data:image/webp;base64,'+base64.b64encode(buf.getvalue()).decode()
    for rec in records:
        key=rec['key'];folder=OUT/key;y0,y1=rec['rows']
        rows={k:io.load_rows(folder/f'{k}.csv') for k in ['S','T','consensus']}
        labels=sorted({r['lithology'] for rr in rows.values() for r in rr})
        colours={label:palette[i%len(palette)] for i,label in enumerate(labels)}
        height=min(6000,y1-y0);scale=height/(y1-y0)
        image=Image.open(folder/'column.webp').convert('RGB')
        image=image.resize((max(140,min(300,image.width*2)),height))
        parts.append(f'<section id="{key}"><h2>{html.escape(key)} — {html.escape(rec["family"])}</h2><p>{html.escape(rec["log"])}<br>Rows {y0}–{y1}; agreement {rec["agreed_share"]:.1%}; {rec["n_intervals"]} consensus intervals</p><div class="view"><div><div class="head">Column</div><img width="{image.width}" height="{height}" src="{uri(image)}"></div>')
        for name in ['consensus','S','T']:
            parts.append(f'<div><div class="head">{name}</div><div class="track" style="width:220px;height:{height}px">')
            for r in rows[name]:
                dispute=name=='consensus' and r.get('agreed')=='false'
                label=html.escape(r['lithology']);top=(r['top_y']-y0)*scale;h=(r['bottom_y']-r['top_y'])*scale
                text=f'{r["top_y"]}–{r["bottom_y"]}: {label}'
                parts.append(f'<div class="band {"dispute" if dispute else ""}" style="top:{top:.2f}px;height:{h:.2f}px;background-color:{colours[r["lithology"]]}" title="{text}">{text}</div>')
            parts.append('</div></div>')
        parts.append('</div><details><summary>Legend rock entries</summary>')
        parts.append(f'<img class="legend" src="{uri(Image.open(folder/"rock_entries.png").convert("RGB"))}"></details>')
        parts.append('<details open><summary>Consensus intervals</summary><table><tr><th>Native rows</th><th>Depth (ft)</th><th>Label</th><th>Agreed</th></tr>')
        for r in rows['consensus']:
            parts.append(f'<tr><td>{r["top_y"]}–{r["bottom_y"]}</td><td>{r["top_depth"]}–{r["bottom_depth"]}</td><td>{html.escape(r["lithology"])}</td><td>{r["agreed"]}</td></tr>')
        parts.append('</table></details></section>')
    parts.append('</body></html>')
    path=BASE/'review.html';path.write_text('\n'.join(parts),encoding='utf-8')
    assert path.stat().st_size<25_000_000


def blocked_preview(manifest, reason):
    """Review the exact requested windows without inventing unreturned labels."""
    ledger=read(OUT/'calls.json') if (OUT/'calls.json').exists() else {'calls':[]}
    parts=['<!doctype html><html><meta charset="utf-8"><title>Lithology review — inference blocked</title>',
           '<style>body{font:15px Arial;margin:24px}section{border-top:2px solid #aaa;margin:30px 0}.view{display:flex;gap:16px}.track{width:180px;background:repeating-linear-gradient(135deg,#fff,#fff 8px,#eee 8px,#eee 16px);border:1px solid #888;padding:10px;box-sizing:border-box}.error{color:#9b1919;background:#fee;padding:16px}img{max-width:100%}</style><body>',
           '<h1>Lithology review — inference unavailable</h1>',f'<p class="error">{html.escape(reason)}</p>',
           '<p>These are source images for human inspection. No predicted bands, interval labels or agreement values have been fabricated. Legend panels have not been interpreted by an LLM.</p>']
    records=[]
    def uri(im):
        buf=bio.BytesIO();im.save(buf,format='WEBP',quality=88,method=6)
        return 'data:image/webp;base64,'+base64.b64encode(buf.getvalue()).decode()
    for key,item in manifest.items():
        folder=OUT/key;folder.mkdir(parents=True,exist_ok=True)
        log=io.open_log(item['log'])
        try:
            y0,y1=item['rows']
            rgb=io.read_rgb(log,'interpreted_lithology_column',y0,y1)
            col=Image.fromarray(rgb)
            col.save(folder/'column.webp',quality=88,method=6)
            col=col.resize((max(140,min(300,col.width*2)),min(5000,col.height)))
            b=log.boxes['lithology_legend']
            leg=fit_image(Image.fromarray(io.read_rgb(log,b,b[1],b[3])))
            leg.save(folder/'legend.webp',quality=90,method=6)
            record=dict(key=key,log=log.name,family=item['family'],rows=[y0,y1],status='blocked',
                        n_intervals=None,agreed_share=None,labels={},
                        calls_used=sum(c['key'].split(':')[0]==key for c in ledger['calls']),error=reason)
            records.append(record);dump(folder/'summary.json',record)
            parts.append(f'<section><h2>{html.escape(key)} — {html.escape(item["family"])}</h2><p>{html.escape(log.name)}<br>Native rows [{y0}, {y1})</p><div class="view"><div><h3>Column</h3><img width="{col.width}" height="{col.height}" src="{uri(col)}"></div>')
            for name in ['Consensus','S','T']:
                parts.append(f'<div><h3>{name}</h3><div class="track" style="height:{col.height}px">No model output</div></div>')
            parts.append(f'</div><details><summary>Original legend image</summary><img width="800" src="{uri(leg)}"></details></section>')
        finally:log.close()
    parts.append('</body></html>')
    page=BASE/'review.html';page.write_text('\n'.join(parts),encoding='utf-8')
    assert page.stat().st_size<25_000_000
    elapsed=time.time()-read(OUT/'timing.json')['started_epoch']
    dump(OUT/'summary.json',dict(status='blocked',windows=records,calls_used=len(ledger['calls']),
         successful_calls=sum(c.get('ok',False) for c in ledger['calls']),wall_seconds=elapsed,
         review_bytes=page.stat().st_size,error=reason,hard_call_cap=LIMIT))
    dump(OUT/'validation.json',dict(status='blocked',iterations=[],gold_read=False,
         metrics=None,error=reason,baseline_row_acc={'A':.74,'B':.32}))
    print(f'Unlabelled source review: {page.stat().st_size} bytes; {len(records)} windows; {elapsed:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--preview-only',action='store_true',help='Build explicitly unlabelled review when model startup is unavailable')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    # Redirect any harness-owned caches before calling calibration or OCR.
    io.ROOT=BASE/'_harness'
    manifest=read(ROOT.parent/'analysis/mix_manifest.json')
    manifest['d582']['rows']=[4404,11195]
    manifest['d582_log02']=dict(family='D-582 1-bit grid (second log)',log='0000000009_d-582_0000020435-0001_log02_fe_5in_md',rows=[4387,6787])
    timing=OUT/'timing.json'
    if not timing.exists():dump(timing,{'started_epoch':time.time()})
    dump(OUT/'manifest.json',manifest)
    if args.preview_only:
        blocked_preview(manifest,'Codex CLI failed during in-process app-server initialization: Windows Access is denied (os error 5). No model output or gold scores exist.')
        return
    validations=[];records=[]
    for iteration in range(1,4):
        record,outputs,agreed=label_window('d582',manifest['d582'],iteration,args.workers)
        result=validate(record,outputs,agreed)
        validations.append(result)
        dump(OUT/'validation.json',{'iterations':validations,'baseline_row_acc':{'A':.74,'B':.32}})
        print('VALIDATION '+json.dumps(result),flush=True)
        if (result['agreed_only_accuracy'] or 0)>=.85:break
    records.append(record)
    if not args.validation_only:
        for key,item in manifest.items():
            if key=='d582':continue
            rec,_,_=label_window(key,item,record['iteration'],args.workers)
            records.append(rec)
            dump(OUT/'summary.json',{'windows':records,'calls_used':len(read(OUT/'calls.json')['calls'])})
    review(records)
    elapsed=time.time()-read(timing)['started_epoch']
    summary=dict(windows=records,calls_used=len(read(OUT/'calls.json')['calls']),wall_seconds=elapsed,
                 review_bytes=(BASE/'review.html').stat().st_size,hard_call_cap=LIMIT)
    dump(OUT/'summary.json',summary)
    dump(timing,{'started_epoch':read(timing)['started_epoch'],'finished_epoch':time.time(),'wall_seconds':elapsed})
    print('SUMMARY '+json.dumps(summary),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        dump(OUT/'last_error.json',{'status':'blocked','error':redact(str(exc)),
                                  'epoch':time.time(),'gold_scored':False})
        raise
