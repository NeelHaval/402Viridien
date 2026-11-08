"""Legend discovery with independent caching and layout-aware section handling.

Uses the original rectangle primitives and public LegendEntry type. Neither
parse() nor its helpers call legend.parse() or write the original cache.
"""
import json
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import io, legend as v1
from .legend import LegendEntry

VERSION = 2
CACHE = io.ROOT / 'out' / '_cache' / 'legend_v2'


def _section(text):
    """OCR often drops ALL spaces; singular SYMBOL must also be accepted."""
    text = re.sub(r'[^a-z]', '', text.lower()).replace('and', '')
    if re.fullmatch(r'litholog(?:y|ical)(?:symbols?)?', text):
        return 'lithology'
    if re.fullmatch(r'modifiers?', text):
        return 'modifiers'
    if re.fullmatch(r'(?:shows?(?:drilling)?|drilling)(?:symbols?)?', text):
        return 'shows'
    return None


def _rock(label):
    # Positive identification is needed for flattened grids: dry/water symbols
    # and Gas_Good escape v1's word-boundary exclusion rule.
    text = re.sub(r'[^a-z]', '', label.lower())
    return bool(re.search(
        r'shale|clay|sand|silt|silstone|limest|dolomit|marl|chalk|chert|'
        r'halite|gypsum|anhydrit|volcan|granit|sylvit|bitumen|^tar$|'
        r'coal|lignite|salt|breccia|conglomer|quartz|basalt|metamorph|'
        r'tuff|mudstone|dolostone|evaporit|siderit|limonit|phosphat', text)
        and not re.search(r'water|gas|oil|show|casing|core|survey', text))


def _overlap(b, q):
    area = max(0, min(b[2], q[2])-max(b[0], q[0])) * max(0, min(b[3], q[3])-max(b[1], q[1]))
    return area / min((b[2]-b[0])*(b[3]-b[1]), (q[2]-q[0])*(q[3]-q[1]))


def _boxes(rgb, words):
    """Preserve v1's proven detection on the existing layouts."""
    boxes = []
    for kernel in (1, 3, 5, 7):
        for colour in (True, False):
            boxes.extend(v1._rectangles(rgb, kernel, colour))
    boxes = [b for b in boxes if not any(
        p[:, 0].min()-1 <= (b[0]+b[2])/2 <= p[:, 0].max()+1 and
        p[:, 1].min()-1 <= (b[1]+b[3])/2 <= p[:, 1].max()+1 and
        b[3]-b[1] <= np.ptp(p[:, 1])*1.6 and
        np.mean(np.min(rgb[b[1]:b[3], b[0]:b[2]], axis=2) < 225) < .60
        for p, t, c in words)]
    for b in v1._pattern_boxes(rgb, words):
        if not any(_overlap(b, q) > .5 for q in boxes):
            boxes.append(b)
    unique = []
    for b in sorted(boxes, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
        if not any(_overlap(b, q) > .70 for q in unique):
            unique.append(b)
    boxes = unique
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    anchors = [b for b in boxes if (b[2]-b[0])*(b[3]-b[1]) > 350 and b[3]-b[1] > 14
               and np.mean(hsv[b[1]:b[3], b[0]:b[2], 1] > 65) > .45]
    if boxes:
        dims = np.asarray([(b[2]-b[0], b[3]-b[1]) for b in boxes], float)
        support = []
        for width, height in dims:
            near = (abs(np.log(dims[:, 0]/width)) < .30) & (abs(np.log(dims[:, 1]/height)) < .30)
            support.append(np.sum(np.prod(dims[near], axis=1)))
        typical = dims[int(np.argmax(support))]
        if len(anchors) >= 2:
            anchored = np.median([(b[2]-b[0], b[3]-b[1]) for b in anchors], axis=0)
            if typical[1] < anchored[1]*.6:
                typical = anchored
        boxes = [b for b in boxes if .70 < (b[2]-b[0])/typical[0] < 1.7
                 and .70 < (b[3]-b[1])/typical[1] < 1.5]
        boxes = [(b[0], b[1], int(b[0]+typical[0]) if b[2]-b[0] > typical[0]*1.2 else b[2], b[3])
                 for b in boxes]
        unique = []
        for b in sorted(boxes, key=lambda b: -(b[2]-b[0])*(b[3]-b[1])):
            if not any(_overlap(b, q) > .45 for q in unique):
                unique.append(b)
        boxes = unique
    boxes, wide = v1._wide_grid(rgb, boxes)
    if not wide:
        boxes = v1._complete_grid(rgb, v1._missing_cells(rgb, boxes))
    return sorted(boxes, key=lambda b: (b[1]//5, b[0])), wide


def _layout(rgb, boxes, words):
    headings = [(p, _section(t)) for p, t, c in words if _section(t)]
    kinds = {kind for p, kind in headings}
    if not {'lithology', 'modifiers', 'shows'} <= kinds:
        return None, []
    # The banner must be above the grid, not a label alongside a swatch.
    top = min((b[1] for b in boxes), default=0)
    headings = [(p, t) for p, t in headings if p[:, 1].mean() < top]
    if len(headings) < 3:
        return None, []
    headings.sort(key=lambda pt: pt[0][:, 0].mean())
    # Real panel dividers take precedence over any rock vocabulary. A rock
    # word in a Modifier or Show panel is never promoted to lithology.
    gray = np.min(rgb, axis=2) < 200
    body = gray[top:]
    # Coverage alone mistakes vertically aligned filled swatches for a rule.
    vertical = cv2.morphologyEx(body.astype('uint8'), cv2.MORPH_OPEN,
                                np.ones((max(3, int(len(body)*.70)), 1), np.uint8))
    ruled = any(vertical[:, int(a[:, 0].mean()):int(b[:, 0].mean())].any()
                for (a, _), (b, _) in zip(headings, headings[1:])) if body.size else False
    labels = [v1._label(b, boxes, words, rgb.shape) for b in boxes]
    rows = {}
    height = np.median([b[3]-b[1] for b in boxes])
    for b, label in zip(boxes, labels):
        key = next((y for y in rows if abs(y-b[1]) < height*.4), b[1])
        rows.setdefault(key, []).append(_rock(label))
    # Some exporters flatten all panels to one row-major grid but retain the
    # three headings. Require both a full rock row and a full technical row,
    # spanning more columns than the headings, before ignoring that banner.
    flat = (not ruled and any(len(r) > len(headings) and all(r) for r in rows.values())
            and any(len(r) >= len(headings) and not any(r) for r in rows.values()))
    return ('flat' if flat else 'panels'), headings


def _panel_edges(rgb, boxes, sections):
    """Prefer physical rules; otherwise use midpoints between heading centers."""
    top = min(b[1] for b in boxes)
    body = (np.min(rgb[top:], axis=2) < 200).astype('uint8')
    vertical = cv2.morphologyEx(body, cv2.MORPH_OPEN,
                                np.ones((max(3, int(len(body)*.70)), 1), np.uint8))
    edges = []
    for (a, _), (b, _) in zip(sections, sections[1:]):
        left, right = int(a[:, 0].mean()), int(b[:, 0].mean())
        candidates = np.flatnonzero(vertical[:, left:right].any(axis=0))+left
        edges.append(float(np.median(candidates)) if len(candidates) else (left+right)/2)
    return edges


def _refine_words(rgb, words):
    """Batch recognition of tight text boxes avoids resizing an entire grid."""
    crops, indices = [], []
    result = list(words)
    for i, (p, text, conf) in enumerate(words):
        if _section(text):
            continue
        x, y = np.floor(p.min(axis=0)).astype(int)
        x1, y1 = np.ceil(p.max(axis=0)).astype(int)
        crop = rgb[max(0, y-1):min(rgb.shape[0], y1+1), max(0, x-1):min(rgb.shape[1], x1+1)]
        if crop.size:
            crops.append(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            indices.append(i)
    if crops:
        recognized, _ = v1._ocr().text_rec(crops)
        for i, (text, confidence) in zip(indices, recognized):
            if text and confidence >= .65:
                result[i] = (words[i][0], text, float(confidence))
    return result


def parse(log):
    """Return original LegendEntry objects, with full-image swatch coordinates."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (log.name+'.json')
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('version') == VERSION:
            return [LegendEntry(e['id'], e['label'], e['swatch_box'],
                                np.asarray(Image.open(CACHE/e['swatch_file']).convert('RGB')),
                                tuple(e['mean_rgb']), e['is_lithology']) for e in data['entries']]
    rgb = io.read_rgb(log, 'lithology_legend')
    words = v1.ocr(rgb)
    boxes, wide = _boxes(rgb, words)
    layout, sections = _layout(rgb, boxes, words)
    panel_edges = _panel_edges(rgb, boxes, sections) if layout == 'panels' else []
    if layout:
        words = _refine_words(rgb, words)
    # Keep the original interpretation on established layouts. The new
    # normalized section logic is restricted to recognized three-part legends.
    headings = []
    for p, text, conf in words:
        if re.fullmatch(r'(?:modifiers?|shows?(?: (?:and|&) drilling)?(?: symbols)?|drilling(?: symbols)?|fossils?|accessor(?:y|ies)|technical|mineral|lithology(?: symbols)?|lithological symbols)', text.strip().rstrip(':'), re.I):
            cx, cy = p.mean(axis=0)
            adjacent = any(b[2] <= cx and abs(cy-(b[1]+b[3])/2) < (b[3]-b[1])*.6
                           and p[:, 0].min()-b[2] < (b[2]-b[0])*2 for b in boxes)
            if not adjacent:
                headings.append((p, text))
    offset = log.boxes['lithology_legend']
    entries, serial = [], []
    debug = Image.fromarray(rgb)
    draw = ImageDraw.Draw(debug)
    for i, box in enumerate(boxes):
        x, y, x1, y1 = box
        label = v1._label(box, boxes, words, rgb.shape)
        if not label or wide:
            right = min([b[0]-2 for b in boxes if b[0] > x1+2 and abs(b[1]-y) < (y1-y)*.6] or [rgb.shape[1]])
            right = min(right, x1+(x1-x)*5)
            crop = rgb[max(0, y-2):min(rgb.shape[0], y1+3), x1+1:right]
            if crop.size and crop.shape[1] > 5:
                crop = cv2.copyMakeBorder(crop, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                label = ' '.join(w[1] for w in v1.ocr(crop))
        swatch = rgb[y+max(2, (y1-y)//14):y1-max(2, (y1-y)//14),
                     x+max(2, (x1-x)//14):x1-max(2, (x1-x)//14)].copy()
        occupied = np.mean(np.min(swatch, axis=2) < 225) if swatch.size else 0
        lith = bool(occupied > .008 and label and not v1._excluded(label))
        if layout == 'flat':
            lith = bool(occupied > .008 and _rock(label) and not v1._excluded(label))
        elif layout == 'panels':
            section = sections[int(np.searchsorted(panel_edges, (x+x1)/2))][1]
            lith = lith and section == 'lithology'
        else:
            above = [(p, t) for p, t in headings if p[:, 1].mean() < y+3]
            if above:
                nearest = min(above, key=lambda pt: abs(pt[0][:, 0].mean()-(x+x1)/2)+.25*(y-pt[0][:, 1].mean()))
                if re.search(r'modifier|show|drilling|fossil|accessor|technical|mineral', nearest[1], re.I):
                    lith = False
        label = label or ('Empty' if occupied <= .008 else 'Unknown')
        full = [x+offset[0], y+offset[1], x1+offset[0], y1+offset[1]]
        mean = tuple(float(v) for v in swatch.mean(axis=(0, 1)))
        entries.append(LegendEntry(i, label, full, swatch, mean, lith))
        filename = log.name+f'.swatch_{i:02d}.png'
        Image.fromarray(swatch).save(CACHE/filename)
        serial.append(dict(id=i, label=label, swatch_box=full, mean_rgb=mean,
                           is_lithology=lith, swatch_file=filename))
        draw.rectangle(box, outline='lime' if lith else 'red', width=2)
        draw.text((x, max(0, y-12)), f'{i}: {label}', fill='blue')
    debug.save(CACHE/(log.name+'.debug.png'))
    path.write_text(json.dumps(dict(version=VERSION, layout=layout, entries=serial,
                                   ocr=[dict(box=p.tolist(), text=t, confidence=c) for p, t, c in words]),
                               indent=2), encoding='utf-8')
    return entries
