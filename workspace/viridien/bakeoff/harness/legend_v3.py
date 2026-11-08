"""Opt-in, content-cached legend parser with complete label-cell recognition.

Runtime code reads image pixels and OCR weights only. Printed spellings remain
the output labels; canonical_name is an optional symmetric comparison utility.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re

import cv2
import numpy as np
from PIL import Image

from . import io, legend as v1, legend_v2 as v2
from .legend import LegendEntry

VERSION = 1
HOME = Path(__file__).resolve().parents[1] / 'ml' / 'legend_names'
DEFAULT = dict(easy=True, cleanup=True, easy_weight=.9)

# General geological vocabulary, not a list of labels from evaluation pages.
ROCKS = ('Anhy Bent Brec Cht Clyst Coal Congl Dol Gyp Igne Lmst Ls Meta '
         'Mrlst Salt Shale Shcol Shgy Sltst Ss Till Sh Tuff Basalt Carbsh '
         'Clay Sand Sandstone Silt Siltstone Limestone Dolomite Dolostone '
         'Mudstone Marl Chalk Chert Gypsum Anhydrite Conglomerate Breccia '
         'Lignite Halite Granite Volcanic Metamorphic Bentonite').split()
MODIFIERS = ('gy gray grey brn brown red rd green gn grn blk black white wh '
             'ltgy dkgy medgy dkbrn rdbrn gygn gybrn yelbrn gyred gyredbrn '
             'ltgrn lt light dk dark med medium blue blu crm cream tan tn yel '
             'rose salmon orng orange col vcol crs coarse sdy sandy limy '
             'arg dol flour metamorphics off muddy').split()


def _key(text):
    return re.sub('[^a-z]', '', str(text).lower())


def _distance(a, b):
    """Small edit distance with inexpensive recurring OCR glyph confusions."""
    prev = list(range(len(b)+1))
    confusions = ({'i', 'l'}, {'v', 'y'}, {'r', 'b'}, {'h', 'b'}, {'a', 'g'}, {'n', 'g'})
    for i, ca in enumerate(a):
        row = [i+1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else (.3 if {ca, cb} in confusions else 1)
            row.append(min(row[-1]+1, prev[j+1]+1, prev[j]+cost))
        prev = row
    return prev[-1]


def clean_name(label):
    """Conservative spelling repair; preserve meaningful separators/modifiers."""
    text = re.sub(r'^[^A-Za-z]+|[^A-Za-z0-9]+$', '', str(label).strip())
    text = re.sub(r'\b[Iil]\s+(?=mst\b|s\b)', 'L', text)
    tokens = re.split(r'(\s+|-+|/)', text)
    for i, token in enumerate(tokens):
        key = _key(token)
        if not key:
            continue
        vocabulary = ROCKS if i == 0 else MODIFIERS + ROCKS
        exact = next((v for v in vocabulary if _key(v) == key), None)
        if exact:
            tokens[i] = exact
            continue
        best = sorted((_distance(key, _key(v)), v) for v in vocabulary)
        limit = .65 if len(key) <= 3 else 1.05
        if best[0][0] <= limit and (len(best) < 2 or best[1][0] > best[0][0]+.15):
            tokens[i] = best[0][1]
    return ''.join(tokens).strip()


def canonical_name(label):
    """Symmetric rock/colour expansion; never silently used by harness scores."""
    # Comparison expands synonyms only. OCR spelling repair belongs to the
    # parser and must not retroactively give a baseline credit for a misread.
    text = str(label).strip().lower()
    aliases = dict(lmst='limestone', ls='limestone', sltst='siltstone',
                   clyst='claystone', ss='sandstone', sh='shale', dol='dolomite',
                   anhy='anhydrite', cht='chert', cgl='conglomerate', congl='conglomerate',
                   brec='breccia', gyp='gypsum', gy='grey', gray='grey',
                   brn='brown', tn='tan', gn='green', grn='green', blk='black',
                   wh='white', blu='blue', lt='light', dk='dark', med='medium')
    text = re.sub(r'\b(dk|lt|med)(?=gy\b)', r'\1 ', text)
    return ''.join(aliases.get(t, t) for t in re.findall('[a-z]+', text))


@lru_cache(maxsize=1)
def _easy():
    import torch
    import easyocr
    torch.set_num_threads(min(3, torch.get_num_threads()))
    # Recognition only avoids loading CRAFT, and never attempts a download.
    return easyocr.Reader(['en'], gpu=False, detector=False, verbose=False,
                          download_enabled=False)


def _framed(rgb):
    ink = (np.min(rgb, axis=2) < 130).astype('uint8')
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    result = []
    for x, y, w, h, area in stats[1:]:
        if not (15 <= w < rgb.shape[1]*.25 and 8 <= h < rgb.shape[0]*.3 and .7 <= w/h <= 14):
            continue
        p = ink[y:y+h, x:x+w]
        sides = [p[:3].mean(1).max(), p[-3:].mean(1).max(),
                 p[:, :3].mean(0).max(), p[:, -3:].mean(0).max()]
        if min(sides) >= .8:
            result.append(tuple(map(int, (x, y, x+w, y+h))))
    return result


def _regions(box, boxes, words, shape):
    """Pair label cells on right/left/below, bounded by neighbouring swatches.

    The cell extends to the next swatch row. Cropping at the swatch bottom
    removes descenders from text whose baseline lies below a narrow sample.
    """
    x, y, x1, y1 = box
    w, h = x1-x, y1-y
    right = min([b[0]-2 for b in boxes if b[0] > x1+2 and abs(b[1]-y) < h*.65] or [shape[1]])
    down = min([b[1]-1 for b in boxes if b[1] > y1+2 and abs(b[0]-x) < w*.6] or [shape[0]])
    bottom = min(down, y1+max(10, int(h*.65)))
    candidates = [('right', (x1+2, max(0, y-2), min(right, x1+w*5), bottom))]
    # Alternative directions require OCR evidence, preventing arbitrary nearby
    # text from becoming additional entries.
    for p, text, confidence in words:
        a, b = p.min(0); c, d = p.max(0)
        if c <= x+2 and x-c < w*3 and y-h*.2 <= (b+d)/2 <= y1+h*.2:
            candidates.append(('left', (max(0, int(a)-3), max(0, y-3), x-1, bottom)))
        if y1 <= b < min(down, y1+h*1.3) and x-w*.2 <= (a+c)/2 <= x1+w*.2:
            candidates.append(('below', (max(0, int(a)-3), y1+1, min(shape[1], int(c)+4), min(down, int(d)+4))))
    return candidates


def _tight(rgb, box):
    x, y, x1, y1 = map(int, box)
    crop = rgb[max(0,y):min(len(rgb),y1), max(0,x):min(rgb.shape[1],x1)]
    if not crop.size:
        return None
    mask = np.min(crop, axis=2) < 210
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    # Remove long rules, and isolated remnants of a neighbouring swatch frame.
    for i, (a,b,w,h,area) in enumerate(stats[1:], 1):
        if w > crop.shape[1]*.65 and h <= 2:
            mask[labels == i] = False
    yy, xx = np.where(mask)
    if not len(xx) or xx.max()-xx.min() < 2 or yy.max()-yy.min() < 4:
        return None
    crop = crop[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()
    return cv2.copyMakeBorder(crop, 4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))


def evidence(log, words=None):
    """Cache raw recognitions independently of the configurable ensemble rule."""
    cv2.setNumThreads(1)
    rgb = io.read_rgb(log, 'lithology_legend')
    digest = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    cache = HOME / 'cache' / f'v{VERSION}_{digest}.json'
    if cache.exists():
        return rgb, json.loads(cache.read_text(encoding='utf-8'))
    words = v1.ocr(rgb) if words is None else words
    boxes, wide = v2._boxes(rgb, words)
    for box in _framed(rgb):
        if not any(v2._overlap(box, b) > .5 for b in boxes):
            boxes.append(box)
    boxes.sort(key=lambda b: (b[1]//5, b[0]))
    layout, sections = v2._layout(rgb, boxes, words)
    edges = v2._panel_edges(rgb, boxes, sections) if layout == 'panels' else []
    items, crops, owners = [], [], []
    for box in boxes:
        item = dict(box=list(map(int, box)), candidates=[])
        label = v1._label(box, boxes, words, rgb.shape)
        if label:
            item['candidates'].append(dict(text=label, confidence=.8, engine='page', direction='right'))
        for direction, region in _regions(box, boxes, words, rgb.shape):
            crop = _tight(rgb, region)
            if crop is not None:
                crops.append(crop); owners.append((len(items), direction))
        if layout == 'panels':
            item['section'] = sections[int(np.searchsorted(edges, (box[0]+box[2])/2))][1]
        else:
            # A spatial section heading suppresses non-rock panels even when
            # a modifier happens to contain a rock word.
            above = [(p,t) for p,t,c in words if v2._section(t) and p[:,1].mean() < box[1]]
            if above and layout != 'flat':
                p,t = min(above, key=lambda pt: abs(pt[0][:,0].mean()-(box[0]+box[2])/2)+.25*(box[1]-pt[0][:,1].mean()))
                item['section'] = v2._section(t)
        items.append(item)
    if crops:
        recognized, _ = v1._ocr().text_rec([cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in crops])
        for (i,direction), (text, conf) in zip(owners, recognized):
            items[i]['candidates'].append(dict(text=text, confidence=float(conf), engine='rapid', direction=direction))
        reader = _easy()
        for (i,direction), crop in zip(owners, crops):
            recognized = reader.recognize(crop, workers=0, batch_size=1)
            if recognized:
                text = ' '.join(v[1] for v in recognized)
                conf = float(np.mean([v[2] for v in recognized]))
                items[i]['candidates'].append(dict(text=text, confidence=conf, engine='easy', direction=direction))
    data = dict(version=VERSION, digest=digest, layout=layout, items=items,
                ocr=[dict(box=p.tolist(), text=t, confidence=c) for p,t,c in words])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return rgb, data


def choose(candidates, config):
    # On column grids the label to the left belongs to the previous swatch.
    # Use alternative directions only when this cell has no readable right text.
    right = [c for c in candidates if c['direction'] == 'right' and
             c['engine'] != 'page' and len(_key(c['text'])) >= 2 and c['confidence'] >= .45]
    if right:
        candidates = [c for c in candidates if c['direction'] == 'right']
    ranked = []
    for c in candidates:
        if c['engine'] == 'easy' and not config['easy']:
            continue
        raw = c['text']
        text = clean_name(raw) if config['cleanup'] else re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$', '', raw).strip()
        if not text:
            continue
        quality = c['confidence']
        if c['engine'] == 'easy': quality *= config['easy_weight']
        if c['engine'] == 'page': quality *= .7
        if _key(re.split(r'[\s/-]', text)[0]) in {_key(v) for v in ROCKS}:
            quality += .25
        if c['direction'] == 'right': quality += .03
        ranked.append((quality, text))
    chosen = max(ranked, default=(0, 'Unknown'))[1]
    # Recognition may collapse two printed hyphens. Page OCR retains their
    # geometry; restore its punctuation only when all letters agree.
    for c in candidates:
        if c['engine'] == 'page':
            page = clean_name(c['text']) if config['cleanup'] else c['text'].strip()
            if _key(page) == _key(chosen) and '-' in page:
                return page
    return chosen


def parse(log, config=None, words=None):
    if config is None:
        path = HOME / 'selected.json'
        config = json.loads(path.read_text(encoding='utf-8'))['config'] if path.exists() else DEFAULT
    rgb, data = evidence(log, words)
    entries = []
    dx,dy = log.boxes['lithology_legend'][:2]
    for i,item in enumerate(data['items']):
        x,y,x1,y1 = item['box']; w,h = x1-x,y1-y
        label = choose(item['candidates'], config)
        swatch = rgb[y+max(2,h//14):y1-max(2,h//14),x+max(2,w//14):x1-max(2,w//14)].copy()
        occupied = np.mean(np.min(swatch, axis=2) < 225) if swatch.size else 0
        rock = v2._rock(label) or _key(re.split(r'[\s/-]', label)[0]) in {_key(v) for v in ROCKS}
        lith = bool(occupied > .008 and label != 'Unknown' and not v1._excluded(label))
        if data['layout'] == 'flat': lith = lith and rock
        if item.get('section') not in (None,'lithology'): lith = False
        entries.append(LegendEntry(i,label,[x+dx,y+dy,x1+dx,y1+dy],swatch,
                                   tuple(map(float,swatch.mean((0,1)))),lith))
    # Adjacent samples sometimes share one label after the final sample. Keep
    # separate visual prototypes and assign that shared name to unnamed ones.
    for e in reversed(entries):
        if e.label != 'Unknown': continue
        x,y,x1,y1 = e.swatch_box;w,h = x1-x,y1-y
        following = [q for q in entries if 0 <= q.swatch_box[0]-x1 <= w*.4 and
                     abs(q.swatch_box[1]-y) <= h*.3 and q.is_lithology]
        if following:
            q = min(following,key=lambda q:q.swatch_box[0]);e.label=q.label;e.is_lithology=True
    return entries
