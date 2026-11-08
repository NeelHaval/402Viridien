"""Read fleet outputs without changing files owned by the fleet."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import learn, paths


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        # Writers can be between writes; withhold incomplete work until next poll.
        return default


def inventory(doc_id):
    folder = paths.PROPOSALS / doc_id
    reviews = {}
    result = []
    for path in sorted(folder.glob('*.json')):
        match = re.fullmatch(r'(legend|unlisted)-(\d+)\.json', path.name)
        if not match:
            continue
        p = read_json(path, {})
        if not isinstance(p, dict) or p.get('doc_id') != doc_id:
            continue
        kind, number = match.groups()
        lid = int(number) if kind == 'legend' else 'u' + number
        if p.get('legend_id' if kind == 'legend' else 'unlisted_id') != lid:
            continue
        round_ = p.get('round')
        if not isinstance(round_, int) or round_ < 1:
            continue
        review_path = folder / f'{kind}-review-r{round_}.json'
        if review_path not in reviews:
            reviews[review_path] = read_json(review_path, {})
        review = reviews[review_path]
        verdict = review.get(str(lid), {}) if isinstance(review, dict) else {}
        tile = (folder / str(p.get('tile_png', ''))).resolve()
        # A fleet tile must belong to this proposal's document directory.
        if tile.parent != folder.resolve() or tile.suffix.lower() != '.png':
            continue
        try:
            stat = tile.stat()
        except OSError:
            continue
        result.append(dict(p, legend_id=lid, source='agent-made' if kind == 'legend' else 'unlisted',
                           label=p.get('label', '') if kind == 'legend' else 'Unlisted: ' + p.get('description', ''),
                           reviewer=verdict, tile_path=str(tile),
                           tile_stamp=[stat.st_mtime_ns, stat.st_size],
                           approved=p.get('status') == 'made' and isinstance(verdict, dict)
                           and verdict.get('verdict') == 'confident'))
    return result


def examples(p, doc):
    boxes = p.get('column_boxes') or []
    if not boxes:
        swatch = next((e.get('box') for e in doc['legend'] if e['id'] == p['legend_id']), None)
        boxes = [swatch] if swatch else []
    return [dict(id=f"{p['source']}-{p['legend_id']}-{i}", doc_id=p['doc_id'],
                 legend_id=p['legend_id'], label=p['label'], box=list(box),
                 provenance=p['source'], flagged=False)
            for i, box in enumerate(boxes)
            if isinstance(box, (list, tuple)) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]]


def load(p, exs):
    with Image.open(Path(p['tile_path'])) as image:
        tile = np.asarray(image.convert('RGB')).copy()
    h, w = tile.shape[:2]
    model = dict(key=f"{p['doc_id']}:{p['legend_id']}", doc_id=p['doc_id'], legend_id=p['legend_id'],
                 label=p['label'], kind='tile', tile_size=[w, h], period=dict(x=w, y=h),
                 examples=len(p.get('column_boxes') or []), sources=[e['id'] for e in exs],
                 suspects=[], suspect_count=0, voters=0, method=p.get('method', 'unknown'),
                 looks_like=p.get('looks_like'), proposal_round=p['round'], proposal_notes=p.get('notes'),
                 column_match=p.get('column_match'), swatch_match=p.get('swatch_match'),
                 tile_source='agent proposal', evidence_kind='column' if p.get('column_boxes') else 'swatch')
    return model, dict(kind='tile', tile=tile, bg=learn.background(tile), anchored=False, phase=(0, 0))
