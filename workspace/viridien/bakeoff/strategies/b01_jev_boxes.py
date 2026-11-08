"""Legend-pattern interval baseline and annotation-independent box locator."""
import base64
import csv
import io as bufferio
import json
import os
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw
from harness import io

NAME = 'B01 JEV boxes and legend patterns'
NEEDS_GPU = False
PAID_API = 'jev'
KEYS = ('depth_column', 'interpreted_lithology_column', 'lithology_legend')
_network_failed = False


def _features(rgb):
    """Coarse colour distribution; invariant to tiling and horizontal shifts."""
    a = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    q = (a.astype(np.int32) // 32).clip(0, 7)
    hist = np.bincount(q[:, 0]*64+q[:, 1]*8+q[:, 2], minlength=512).astype(float)
    hist /= max(hist.sum(), 1)
    ink = (np.asarray(rgb).mean(axis=2) < 160).astype(np.float32)
    texture = []
    for offset in (1, 2, 4, 8):
        for axis in (0, 1):
            if ink.shape[axis] <= offset:
                texture.append(0.)
            elif axis == 0:
                texture.append(float(np.mean(np.abs(ink[offset:] - ink[:-offset]))))
            else:
                texture.append(float(np.mean(np.abs(ink[:, offset:] - ink[:, :-offset]))))
    return np.r_[hist, np.asarray(texture)*.3]


def run(log, window, ctx):
    entries = [e for e in ctx['legend'] if e.is_lithology and e.swatch.size]
    if not entries:
        return [], {}
    rgb = io.read_rgb(log, 'interpreted_lithology_column', *window)
    h, w = rgb.shape[:2]
    # Drop persistent border rulings from the feature calculation only.
    border = (rgb.mean(axis=2) < 100).mean(axis=0) > .8
    usable = ~border if border.sum() < w*.25 and (~border).sum() >= 5 else np.ones(w, dtype=bool)
    refs = np.stack([_features(e.swatch) for e in entries])
    # A full-width dark rule marks many exact boundaries, especially synthetic ones.
    darkness = (rgb.mean(axis=2) < 90).mean(axis=1)
    rules = np.flatnonzero(darkness > .85)
    rules = [int(y) for i, y in enumerate(rules) if i == 0 or y > rules[i-1] + 1]
    cuts = sorted(set([0, h] + rules + list(range(0, h, 40))))
    predictions = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a: continue
        crop = rgb[a:b, usable]
        # Ignore the ruling itself while identifying its following segment.
        if b-a > 2 and darkness[a] > .85: crop = crop[1:]
        feat = _features(crop)
        dist = np.sum(np.abs(refs-feat), axis=1)
        blank = np.mean(crop > 244) > .993
        predictions.append([a, b, -1 if blank else int(np.argmin(dist))])
    # Suppress short fluctuations in non-rule regions without erasing true rules.
    rule_set = set(rules)
    for i in range(1, len(predictions)-1):
        a, b, k = predictions[i]
        if a not in rule_set and b not in rule_set and b-a < 45 and predictions[i-1][2] == predictions[i+1][2]:
            predictions[i][2] = predictions[i-1][2]
    rows = []
    for a, b, k in predictions:
        if k < 0: continue
        e = entries[k]
        if rows and rows[-1]['bottom_y'] == window[0]+a and rows[-1]['cluster_id'] == e.id and a not in rule_set:
            rows[-1]['bottom_y'] = window[0]+b
        else:
            rows.append(dict(top_y=window[0]+a, bottom_y=window[0]+b, lithology=e.label, cluster_id=e.id))
    strip = np.full((h, 36, 3), 255, np.uint8)
    for a, b, k in predictions:
        if k >= 0:
            strip[a:b] = np.asarray(entries[k].mean_rgb, np.uint8)
    rules_img = rgb.copy()
    for y in rules: rules_img[y:min(y+2,h)] = (255,0,0)
    return rows, {'1 source': rgb, '2 detected rules': rules_img, '3 matched swatches': strip, '_api_calls': 0}


def _page(log):
    w, h = log.size
    # The header and the beginning of the body retain legible column headers.
    bottom = min(h, max(1500, min(4000, int(w*2.4))))
    source = io.read_rgb(log, [0, 0, w, bottom])
    scale = min(1024/w, 1024/bottom)
    resized = cv2.resize(source, (max(1, round(w*scale)), max(1, round(bottom*scale))), interpolation=cv2.INTER_AREA)
    ih, iw = resized.shape[:2]
    canvas = np.full((max(512, ih), max(512, iw), 3), 255, np.uint8)
    canvas[:ih,:iw] = resized
    return canvas, scale, bottom


def _jev(image):
    global _network_failed
    if _network_failed: return None
    path = Path('C:/Users/hssil/Downloads/03_Programming_Projects/misc/.env')
    key = os.environ.get('JEV_API_KEY')
    if not key and path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip().startswith('JEV_API_KEY='):
                key = line.split('=',1)[1].strip().strip('"\'')
                break
    if not key: return None
    buf = bufferio.BytesIO(); Image.fromarray(image).save(buf, format='PNG')
    questions = {
        'depth_column': {'type':'box','instructions':'Locate the depth scale column in the body of this mud log. Include the depth values.'},
        'interpreted_lithology_column': {'type':'box','instructions':'Locate the interpreted lithology pattern column in the mud log body.'},
        'lithology_legend': {'type':'box','instructions':'Locate the complete lithology legend containing pattern swatches and labels.'}}
    payload = {'model':'jev-latest','state':{'crop':'imgA'},'assets':{'imgA':{'type':'image','format':'png','data_base64':base64.b64encode(buf.getvalue()).decode()}},'questions':questions}
    request = urllib.request.Request('https://api.typesafe.ai/v1/systemone', data=json.dumps(payload).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','TypeSafe-Danger-Zone-Images':'true'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response).get('answers', {})
    except Exception:
        _network_failed = True
        return None


def _baseline(log, image, scale, bottom):
    w, h = log.size
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    valid = gray[:max(1,round(bottom*scale)), :max(1,round(w*scale))]
    dark = (valid < 125).astype(np.uint8)
    # Long vertical rulings in the body form candidate track edges.
    body = dark[len(dark)//2:]
    strengths = body.mean(axis=0)
    candidates = np.flatnonzero(strengths > .35)
    lines = [int(round(np.mean(g)/scale)) for g in np.split(candidates, np.where(np.diff(candidates)>1)[0]+1) if len(g)]
    lines = sorted(set([0, w] + lines))
    spans = [(a,b) for a,b in zip(lines,lines[1:]) if 12 <= b-a <= w*.25]
    if not spans: spans = [(int(w*.15),int(w*.2)),(int(w*.2),int(w*.28))]
    # Track candidates closer to the left edge are favoured for depth; adjacent
    # pattern tracks are favoured for interpreted lithology.
    depth = min(spans, key=lambda ab: abs((ab[0]+ab[1])/2-w*.15))
    lith = min((s for s in spans if s != depth), key=lambda ab: abs((ab[0]+ab[1])/2-(depth[1]+w*.08)), default=depth)
    content_y = np.flatnonzero(dark.mean(axis=1) > .015)
    top = int(content_y[0]/scale) if len(content_y) else 0
    legend = [int(w*.03), top, int(w*.85), min(bottom, int(bottom*.5))]
    return {'depth_column':[depth[0], int(bottom*.35), depth[1], h], 'interpreted_lithology_column':[lith[0], int(bottom*.35), lith[1], h], 'lithology_legend':legend}, lines


def _locate(log, use_jev=True):
    image, scale, bottom = _page(log)
    boxes, lines = _baseline(log, image, scale, bottom)
    answers = _jev(image) if use_jev else None
    if answers:
        ih, iw = image.shape[:2]; w,h = log.size
        for key in KEYS:
            b = answers.get(key, {}).get('box', {})
            try:
                x0 = int(float(b['left'])*iw/scale); x1 = int(float(b['right'])*iw/scale)
                y0 = int(float(b['top'])*ih/scale); y1 = int(float(b['bottom'])*ih/scale)
                x0,x1 = max(0,min(w-1,x0)),max(1,min(w,x1))
                y0,y1 = max(0,min(h-1,y0)),max(1,min(h,y1))
                if x1<=x0 or y1<=y0: continue
                if key != 'lithology_legend':
                    # Snap only nearby line edges, avoiding unrelated tracks.
                    width=x1-x0
                    for index, x in enumerate((x0,x1)):
                        nearest=min(lines,key=lambda v:abs(v-x))
                        if abs(nearest-x) < max(8,width*.3):
                            if index == 0: x0=nearest
                            else: x1=nearest
                    y1=h
                if x1>x0: boxes[key]=[x0,y0,x1,y1]
            except (KeyError, TypeError, ValueError): pass
    return boxes, bool(answers)


def locate(log):
    return _locate(log)[0]


def _iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def main():
    out=io.ROOT/'out'/'b01'; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in io.list_logs():
        log=io.open_log(name)
        try:
            pred, used=_locate(log); base,_=_locate(log,False)
            rec={'log':name,'jev_used':used}
            for key in KEYS:
                rec[key+'_iou']=_iou(pred[key],log.boxes[key]); rec[key+'_baseline_iou']=_iou(base[key],log.boxes[key])
            records.append(rec)
            # Show full page in bounded sampled strips, without a full RGB page.
            thumb=np.asarray(io.read_rgb(log,[0,0,*log.size],scale=min(700/log.size[0],1400/log.size[1])))
            img=Image.fromarray(thumb.copy()); draw=ImageDraw.Draw(img); sx=img.width/log.size[0]; sy=img.height/log.size[1]
            for key,colour in zip(KEYS,('blue','red','green')):
                for box,width in ((log.boxes[key],2),(pred[key],1)):
                    draw.rectangle((box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy),outline=colour,width=width)
            img.save(out/(name+'.png'))
            print(json.dumps(rec),flush=True)
        except Exception as exc:
            records.append({'log':name,'error':str(exc)})
            print(name, type(exc).__name__, str(exc),flush=True)
        finally: log.close()
    fields=['log','jev_used']+[k+s for k in KEYS for s in ('_iou','_baseline_iou')]+['error']
    with (out/'boxes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    summary={'n_logs':len(records),'jev_used':sum(bool(r.get('jev_used')) for r in records),'errors':sum('error' in r for r in records)}
    for field in fields[2:-1]:
        values=[r[field] for r in records if field in r];summary[field]=float(np.mean(values)) if values else None
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
