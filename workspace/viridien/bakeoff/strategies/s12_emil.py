"""Emil's BoundaryNet + bed clustering + legend descriptor naming (fair OCR legend)."""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from harness import io
from emil import pipeline as p

NAME = 'S12 Emil CNN + legend (detect OCR, Drive weights)'
NEEDS_GPU = True
PAID_API = None


def map_name(name, entries):
    norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())
    target = norm(name)
    choices = [(SequenceMatcher(None, target, norm(e.label)).ratio(), e.label) for e in entries if e.is_lithology]
    if not choices: return name
    similarity, label = max(choices)
    return label if similarity >= .72 else name


def detect_legend(log, cache_dir):
    # Cache by legend pixels; repeated continuation pages need the same OCR work.
    import hashlib
    legend_box = log.boxes['lithology_legend']
    rgb = io.read_rgb(log, legend_box)
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    cache = Path(cache_dir) / f'legend_{digest}.json'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        found = json.loads(cache.read_text(encoding='utf-8'))
        found = [dict(legend_id=e['legend_id'], lithology=e['lithology'], box=[e['box'][0]+legend_box[0],e['box'][1]+legend_box[1],e['box'][2]+legend_box[0],e['box'][3]+legend_box[1]]) for e in found]
    else:
        # Read the legend from its bounded crop; retain the notebook detector/OCR.
        from PIL import Image
        image = Image.fromarray(rgb)
        local_box = [0,0,image.width,image.height]
        boxes = p.detect_swatches(image, local_box)
        names = p.read_swatch_labels(image, boxes, local_box)
        local = [dict(legend_id=i,lithology=n or f'legend_{i}',box=list(b)) for i,(b,n) in enumerate(zip(boxes,names))]
        cache.write_text(json.dumps(local, indent=2), encoding='utf-8')
        found = [dict(e,box=[e['box'][0]+legend_box[0],e['box'][1]+legend_box[1],e['box'][2]+legend_box[0],e['box'][3]+legend_box[1]]) for e in local]
    return pd.DataFrame(found, columns=['legend_id','lithology','box'])


def predict(log, window, legend):
    torch.set_num_threads(6)
    y0, y1 = map(int, window)
    col = io.read_rgb(log, 'interpreted_lithology_column', y0, y1)
    raw = p.raw_row_features(col)
    page = dict(col=col, y_offset=y0, R=raw, F=p.zscore(raw))
    beds = p.beds_from_page(page)
    segments = p.cluster_beds(beds,raw,y0)
    named, layers = p.lithology_layers(segments,col,y0,legend,log.image)
    return layers, segments, beds, named


def run(log, window, ctx):
    """No truth is accessed: only scan pixels, pretrained weights and shared OCR names."""
    legend = detect_legend(log, ctx['cache_dir'])
    layers, _, _, _ = predict(log,window,legend)
    rows = [dict(top_y=int(r.y_top),bottom_y=int(r.y_bottom),lithology=map_name(r.lithology,ctx['legend']),cluster_id=int(r.legend_id)) for r in layers.itertuples() if r.legend_id >= 0]
    return rows
