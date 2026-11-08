"""JEV v2: one JEV call per interval with a fair, same-scale picture.

Fixes over the prefill JEV path:
- legend swatches and the query are pasted at the SAME native scale (no lossy shrinking),
  so dash length, spacing and dot size can be compared directly;
- the query is the interval interior: boundary lines and column frame are trimmed;
- OCR label typos are corrected (LImestone -> Limestone) for the prompt;
- the owner's own labels from the same log are shown as confirmed examples.

CLI:  python -X utf8 -B labeltool/jev2.py --log <log> --top 4399 --bottom 4455
      python -X utf8 -B labeltool/jev2.py --eval-human     (re-ask every human-labelled item)
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool.common import CACHE, HUMAN, harness_for, normalize  # noqa: E402

VERSION = 'jev2-2'
OUT = CACHE / 'jev2'
VOCAB = ['Limestone', 'Dolomite', 'Anhydrite', 'Halite', 'Gypsum', 'Salt', 'Coal', 'Lignite', 'Clay',
         'Claystone', 'Shale', 'Mudstone', 'Siltstone', 'Silty Sandstone', 'Sand and Sandstone', 'Sandstone',
         'Sand', 'Chert', 'Marl', 'Tuff', 'Conglomerate', 'Cement', 'Igneous', 'Volcanics', 'Chalk',
         'Calcareous Shale', 'Silty Shale', 'Sandy Shale', 'Shaly Sandstone', 'Calcareous Sandstone',
         'Dolomitic Limestone', 'Limy Dolomite', 'Bentonite', 'Granite', 'Basement', 'Unknown']
_lock = threading.Lock()
_font_cache = {}


def font(size):
    if size not in _font_cache:
        for name in ('arialbd.ttf', 'arial.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
            try:
                _font_cache[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def fix_label(raw):
    """Correct OCR slips (I/l/1, rn/m, joined lines) against a lithology vocabulary, else tidy case."""
    text = re.sub(r'\s+', ' ', str(raw or '')).strip()
    if not text:
        return text
    probe = text.replace('1', 'l').replace('|', 'l')
    fixed_i = re.sub(r'(?<=[a-zA-Z])I(?=[a-z])|(?<=^[A-Z])I', 'i', probe)
    candidates = {v.lower(): v for v in VOCAB}
    for attempt in (text, probe, fixed_i, fixed_i.replace('ll', 'l')):
        if attempt.lower() in candidates:
            return candidates[attempt.lower()]
    match = difflib.get_close_matches(fixed_i.lower(), list(candidates), n=1, cutoff=0.84)
    if match:
        return candidates[match[0]]
    return fixed_i if fixed_i.isupper() is False else fixed_i.title()


def trim_interior(rgb):
    """Drop dark boundary rows near the top/bottom and the column frame on the left/right."""
    ink = rgb.mean(axis=2) < 128

    def cut(profile, margin):
        """(start, end): skip solid lines (any thickness) that begin within `margin` cells of an edge."""
        n = len(profile)
        solid = profile > 0.92
        start, end = 0, n
        first = next((i for i in range(min(margin, n)) if solid[i]), None)
        if first is not None:
            start = first
            while start < n and solid[start]:
                start += 1
            start = min(start + 2, n)
        last = next((i for i in range(n - 1, max(n - 1 - margin, -1), -1) if solid[i]), None)
        if last is not None and last >= start:
            end = last
            while end > start and solid[end - 1]:
                end -= 1
            end = max(end - 2, start)
        return (start, end) if end > start else (0, n)

    top, bottom = cut(ink.mean(axis=1), 6)
    left, right = cut(ink[top:bottom].mean(axis=0), 12)
    return rgb[top:bottom, left:right]


def read_interval(io, log, top, bottom, max_rows=230):
    """Interval interior at native scale; long intervals are centred to max_rows."""
    if bottom - top > max_rows:
        mid = (top + bottom) // 2
        top, bottom = mid - max_rows // 2, mid + max_rows // 2
    rgb = io.read_rgb(log, 'interpreted_lithology_column', y0=int(top), y1=int(bottom))
    return trim_interior(rgb)


class Context:
    """Per-log harness objects, legend and corrected labels (cached in-process)."""
    _logs = {}

    @classmethod
    def get(cls, name):
        with _lock:
            if name not in cls._logs:
                io, legend, _ = harness_for(name)
                log = io.open_log(name)
                entries = [e for e in legend.parse(log) if e.is_lithology and e.swatch.size]
                tiles = []
                seen = set()
                for e in entries:
                    fixed = fix_label(e.label)
                    if normalize(fixed) in seen:
                        continue
                    seen.add(normalize(fixed))
                    tiles.append(dict(id=e.id, raw=e.label, label=fixed, swatch=e.swatch, mean=float(e.swatch.mean())))
                cls._logs[name] = dict(io=io, log=log, tiles=tiles, lock=threading.Lock())
            return cls._logs[name]


def human_examples(name, exclude=(), limit=4):
    """Most recent confirmed labels for this log (undo-aware), one per label, skipping excluded rows."""
    path = HUMAN / 'labels.jsonl'
    if not path.exists():
        return []
    events = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    undone = {e.get('target') for e in events if e.get('action') == 'undo'}
    active = [e for e in events if e.get('action') == 'label' and e['event_id'] not in undone
              and e.get('log') == name and normalize(e.get('label', '')) not in ('', 'unknown')]
    picked, labels = [], set()
    for e in reversed(active):
        if any(not (e['bottom_y'] <= a or e['top_y'] >= b) for a, b in exclude):
            continue
        key = normalize(fix_label(e['label']))
        if key in labels:
            continue
        labels.add(key)
        picked.append(e)
        if len(picked) >= limit:
            break
    return picked


def build_panel(ctx, query, examples, tiles=None):
    """Everything pasted at native scale on one canvas (JEV later scales the whole canvas uniformly)."""
    tiles = tiles or ctx['tiles']
    sw = max(t['swatch'].shape[1] for t in tiles)
    sh = max(t['swatch'].shape[0] for t in tiles)
    sw, sh = min(sw, 190), min(sh, 140)
    cols = 4 if len(tiles) > 9 else 3
    cell_w, cell_h = sw + 22, sh + 34
    qw = max(query.shape[1], max([x['rgb'].shape[1] for x in examples], default=0))
    left_w = qw + 30
    ex_h = sum(x['rgb'].shape[0] + 34 for x in examples)
    rows = (len(tiles) + cols - 1) // cols
    width = left_w + cols * cell_w + 20
    height = max(40 + query.shape[0] + 40 + (30 + ex_h if examples else 0), 40 + rows * cell_h) + 10
    canvas = Image.new('RGB', (width, height), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), 'QUERY (classify this)', font=font(16), fill='black')
    y = 34
    canvas.paste(Image.fromarray(query), (10, y))
    d.rectangle([8, y - 2, 10 + query.shape[1] + 1, y + query.shape[0] + 1], outline=(220, 0, 0), width=2)
    y += query.shape[0] + 24
    if examples:
        d.text((10, y), 'CONFIRMED EXAMPLES (same log)', font=font(14), fill=(0, 90, 0))
        y += 24
        for x in examples:
            canvas.paste(Image.fromarray(x['rgb']), (10, y))
            d.text((10, y + x['rgb'].shape[0] + 3), 'is ' + x['label'], font=font(14), fill=(0, 90, 0))
            y += x['rgb'].shape[0] + 34
    x0 = left_w
    d.text((x0, 8), 'LEGEND OPTIONS (same scale as query)', font=font(16), fill='black')
    for i, t in enumerate(tiles):
        cx, cy = x0 + (i % cols) * cell_w, 34 + (i // cols) * cell_h
        s = t['swatch'][:sh, :sw]
        canvas.paste(Image.fromarray(s), (cx, cy))
        d.rectangle([cx - 1, cy - 1, cx + s.shape[1], cy + s.shape[0]], outline=(150, 150, 150))
        d.text((cx, cy + s.shape[0] + 4), f'{i + 1}. {t["label"]}', font=font(14), fill='black')
    return canvas


PROMPT = ('You classify a rock symbol in a scanned mud log. The red-boxed QUERY is a strip of the '
          'lithology column. The LEGEND OPTIONS are the printed symbols at the SAME pixel scale. '
          'Ignore borders and solid separator lines. Compare: fill colour; line orientation '
          '(horizontal, vertical, diagonal); dash length, gap and thickness; whether dash rows are '
          'offset like bricks; presence, size and spacing of dots; ink density. Similar symbols differ '
          'mainly in dash length/spacing and dots, so compare them carefully. CONFIRMED EXAMPLES '
          '(if any) are the same log labelled by a human: if the query looks like one of them, '
          'that label is very likely right. Choose exactly one legend option.')


def ask(name, top, bottom, client=None, use_examples=True, only=None):
    """Return dict(label, raw_label, probability, ranking, panel, calls, status)."""
    from strategies._jev import Client, image_asset
    ctx = Context.get(name)
    with ctx['lock']:
        query = read_interval(ctx['io'], ctx['log'], top, bottom)
        examples = []
        if use_examples:
            for e in human_examples(name, exclude=[(top, bottom)]):
                rgb = read_interval(ctx['io'], ctx['log'], e['top_y'], e['bottom_y'], max_rows=110)
                if rgb.size:
                    examples.append(dict(rgb=rgb, label=fix_label(e['label'])))
    tiles = [t for t in ctx['tiles'] if only is None or t['raw'] in only] or ctx['tiles']
    if len(tiles) == 1:
        t = tiles[0]
        one = [dict(label=t['label'], raw_label=t['raw'], probability=1.0)]
        return dict(status='ok', label=t['label'], raw_label=t['raw'], probability=1.0, ranking=one, panel=None,
                    examples=[], version=VERSION, calls=0, query_ink=round(ink(query), 3),
                    query_rows=int(query.shape[0]), dropped=[])
    panel = build_panel(ctx, query, examples, tiles)
    labels = [t['label'] for t in tiles]
    key = hashlib.sha256(VERSION.encode() + panel.tobytes() + json.dumps(labels).encode()).hexdigest()[:24]
    OUT.mkdir(parents=True, exist_ok=True)
    panel_path = OUT / (key + '.png')
    panel.save(panel_path)
    cached = OUT / (key + '.json')
    if cached.exists():
        return ink_filter(ctx, query, dict(json.loads(cached.read_text(encoding='utf-8')), calls=0, cached=True))
    client = client or Client(timeout=45, retries=1)
    if not client.key:
        return dict(status='no_key', label=None, panel=key, calls=0)
    before = client.calls
    response = client.ask({'task': PROMPT, 'image': 'panel'}, {'panel': image_asset(panel)},
                          {'rock': {'type': 'choice', 'instructions': PROMPT,
                                    'criteria': {str(i + 1): f'{i + 1}. {l}' for i, l in enumerate(labels)}}})
    probs = (response or {}).get('answers', {}).get('rock', {}).get('probabilities', {})
    p = np.asarray([float(probs.get(str(i + 1), 0) or 0) for i in range(len(labels))])
    if not len(p) or not np.isfinite(p).all() or p.sum() <= 0:
        return dict(status='unavailable', label=None, panel=key, calls=client.calls - before)
    p = p / p.sum()
    order = np.argsort(-p)
    ranking = [dict(label=labels[i], raw_label=tiles[i]['raw'], probability=round(float(p[i]), 3))
               for i in order[:4]]
    result = dict(status='ok', label=ranking[0]['label'], raw_label=ranking[0]['raw_label'],
                  probability=ranking[0]['probability'], ranking=ranking, panel=key,
                  examples=[x['label'] for x in examples], version=VERSION)
    cached.write_text(json.dumps(result), encoding='utf-8')
    return ink_filter(ctx, query, dict(result, calls=client.calls - before))


def ink(rgb):
    return float((rgb.mean(axis=2) < 128).mean()) if rgb.size else 0.0


def ink_filter(ctx, query, result, gap=0.45):
    """Drop options whose ink coverage is far from the query's (e.g. solid Coal for a thin strip of dashes)."""
    q = ink(query)
    tiles = {t['raw']: ink(t['swatch'][4:-4, 4:-4]) for t in ctx['tiles']}  # all tiles of the log
    keep = [x for x in result.get('ranking', []) if abs(tiles.get(x['raw_label'], q) - q) <= gap]
    dropped = [x['label'] for x in result.get('ranking', []) if x not in keep]
    if not keep or not dropped:
        return dict(result, query_ink=round(q, 3), query_rows=int(query.shape[0]), dropped=[])
    total = sum(x['probability'] for x in keep) or 1
    keep = [dict(x, probability=round(x['probability'] / total, 3)) for x in keep]
    return dict(result, ranking=keep, label=keep[0]['label'], raw_label=keep[0]['raw_label'],
                probability=keep[0]['probability'], query_ink=round(q, 3), query_rows=int(query.shape[0]), dropped=dropped)


# Symbols that are printed near-identically in some legends (measured on d-582: same dash rows and
# brick offset, dash 13 vs 15 px); JEV cannot separate them from the picture alone.
LOOKALIKES = [{'clay', 'shale'}, {'claystone', 'shale'}, {'clay', 'claystone'}, {'mudstone', 'shale'}]


def decide(name, top, bottom, result, near=800):
    """Turn a JEV ranking into the tool's suggestion: flag look-alike ties and break them with the
    owner's nearest confirmed label in the same log (never with gold)."""
    ranking = result.get('ranking') or []
    if result.get('query_rows', 99) < 8:
        return dict(result, ambiguous=True, tiebreak=None,
                    why=f"Only {result['query_rows']} pattern rows after removing borders: too thin to read; "
                        "use the neighbouring interval's label or merge it.")
    if len(ranking) < 2:
        return dict(result, ambiguous=False, why='')
    a, b = ranking[0], ranking[1]
    pair = {normalize(a['label']), normalize(b['label'])}
    tied = any(pair <= g for g in LOOKALIKES) or (a['probability'] - b['probability'] < 0.12)
    if not tied:
        return dict(result, ambiguous=False, why=f"JEV {a['probability']:.0%} vs next {b['probability']:.0%}")
    options = {normalize(a['label']): a, normalize(b['label']): b}
    best, dist = None, None
    for e in human_examples(name, exclude=[(top, bottom)], limit=999):
        key = normalize(fix_label(e['label']))
        if key not in options:
            continue
        d = max(0, e['top_y'] - bottom, top - e['bottom_y'])
        if dist is None or d < dist:
            best, dist = options[key], d
    if best is not None and dist <= near:
        return dict(result, label=best['label'], raw_label=best['raw_label'], probability=best['probability'],
                    ambiguous=True, tiebreak='human', why=f"{a['label']} and {b['label']} look the same in this legend; "
                    f"your nearest label {dist} rows away is {best['label']}")
    return dict(result, ambiguous=True, tiebreak=None,
                why=f"{a['label']} or {b['label']}: the legend symbols are near-identical; JEV is guessing "
                    f"({a['probability']:.0%} vs {b['probability']:.0%}). Decide from context.")


def eval_human():
    path = HUMAN / 'labels.jsonl'
    events = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    undone = {e.get('target') for e in events if e.get('action') == 'undo'}
    active = [e for e in events if e.get('action') == 'label' and e['event_id'] not in undone]
    right = old_right = 0
    for e in active:
        r = ask(e['log'], e['top_y'], e['bottom_y'])
        ok = normalize(fix_label(r.get('label') or '')) == normalize(fix_label(e['label']))
        old = normalize(fix_label(e.get('suggestion_shown') or '')) == normalize(fix_label(e['label']))
        right += ok
        old_right += old
        print(f"{e['log'][-28:]} {e['top_y']}-{e['bottom_y']} human={e['label']!r} old={e.get('suggestion_shown')!r} "
              f"jev2={r.get('label')!r} p={r.get('probability')} top={[x['label'] for x in r.get('ranking', [])]} "
              f"{'OK' if ok else 'WRONG'} panel={r.get('panel')}")
    n = len(active)
    print(f'n={n} old_suggestion_correct={old_right}/{n} jev2_correct={right}/{n}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log')
    ap.add_argument('--top', type=int)
    ap.add_argument('--bottom', type=int)
    ap.add_argument('--no-examples', action='store_true')
    ap.add_argument('--eval-human', action='store_true')
    a = ap.parse_args(argv)
    if a.eval_human:
        return eval_human()
    r = ask(a.log, a.top, a.bottom, use_examples=not a.no_examples)
    print(json.dumps(r, indent=1))
    print('panel image:', OUT / (r['panel'] + '.png'))


if __name__ == '__main__':
    main()
