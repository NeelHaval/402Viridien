"""Seeded, source-disjoint synthetic mud-log columns with exact interval truth.

Only ``assets`` opens in-domain images; the eval denylist is checked before IO.
Windows are reproducible from the asset manifest and seed, independently of OS.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import cv2
import numpy as np
from PIL import Image
from harness import io, legend

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parents[1]
FAMILIES = ['Gulf 3-panel', 'Geoservices grid', 'D-582 1-bit grid',
            'MC765 colour grid', 'MC761 mineral/lithology',
            "Texas 'Rock Types' box", 'One-off layout']


def family_map():
    with (BAKEOFF.parent/'analysis'/'label_shares_per_log.csv').open(encoding='utf-8-sig') as f:
        return {r['log']: r['family'] for r in csv.DictReader(f)}


def excluded():
    return {r['log'] for r in json.loads((BAKEOFF/'eval_set.json').read_text())}


def assets():
    """Extract native legend tiles, rules, text and scan statistics, never truth."""
    directory = HERE/'assets'
    directory.mkdir(exist_ok=True)
    original = io.ROOT
    io.ROOT = HERE/'cache'
    records = []
    try:
        families = family_map()
        available = set(io.list_logs()) - excluded()
        for family in FAMILIES:
            names = sorted(n for n in available if families.get(n) == family)
            for j, name in enumerate(names):
                assert name not in excluded()
                key = hashlib.sha256(name.encode()).hexdigest()[:16]
                # Existing parser output is visual preprocessing, not labels.
                src = BAKEOFF/'out'/'_cache'/'legend'
                dst = io.ROOT/'out'/'_cache'/'legend'
                dst.mkdir(parents=True, exist_ok=True)
                for p in src.glob(name+'.*'):
                    if p.is_file() and not (dst/p.name).exists():
                        shutil.copy2(p, dst/p.name)
                log = io.open_log(name)
                try:
                    entries = [e for e in legend.parse(log) if e.is_lithology and e.swatch.size
                               and min(e.swatch.shape[:2]) >= 3]
                    if len(entries) < 2:
                        continue
                    tiles, labels, boxes = [], [], []
                    for e in entries:
                        k = len(tiles)
                        tile = f'{key}_{k:02d}.png'
                        Image.fromarray(e.swatch).save(directory/tile)
                        tiles.append(tile); labels.append(e.label); boxes.append(e.swatch_box)
                    x0, y0, x1, y1 = log.boxes['interpreted_lithology_column']
                    samples = [io.read_rgb(log, 'interpreted_lithology_column',
                               int(y), min(y1, int(y)+512))
                               for y in np.linspace(y0, max(y0, y1-512), 5)]
                    sample = np.concatenate(samples)
                    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
                    rule = (gray < 100).mean(0)
                    # Keep stable narrow vertical rules, discard dense filled panels.
                    positions = np.flatnonzero(rule > .72)
                    positions = [int(v) for k, v in enumerate(positions)
                                 if k == 0 or v-positions[k-1] > 2]
                    lr = io.read_rgb(log, 'lithology_legend')
                    # Text overlays come from native OCR polygons in this source.
                    cache = dst/(name+'.json')
                    ocr = json.loads(cache.read_text()).get('ocr', []) if cache.exists() else []
                    text_files = []
                    lx, ly = log.boxes['lithology_legend'][:2]
                    for word in ocr[:12]:
                        p = np.array(word['box'])
                        a, b = np.floor(p.min(0)).astype(int)
                        c, d = np.ceil(p.max(0)).astype(int)
                        # Parser OCR polygons are legend-local coordinates.
                        a, c = max(0, a), min(lr.shape[1], c)
                        b, d = max(0, b), min(len(lr), d)
                        if c > a+3 and d > b+3:
                            fn = f'{key}_text{len(text_files):02d}.png'
                            Image.fromarray(lr[b:d, a:c]).save(directory/fn)
                            text_files.append(fn)
                    record = dict(log=name, family=family, split='held' if j == len(names)-1 else 'train',
                                  tiles=tiles, labels=labels, boxes=boxes, width=x1-x0,
                                  rules=[v/(x1-x0) for v in positions], text=text_files,
                                  mode=log.mode, legend_box=log.boxes['lithology_legend'],
                                  background=float(np.percentile(gray, 90)),
                                  ink_fraction=float((gray < 128).mean()),
                                  legend_sha256=hashlib.sha256(lr.tobytes()).hexdigest())
                    records.append(record)
                    print('asset', family, name, len(tiles), record['split'], flush=True)
                finally:
                    log.close()
    finally:
        io.ROOT = original
    assert not ({r['log'] for r in records} & excluded())
    # Assign after filtering: the last nominal source may have no usable tiles.
    for family in FAMILIES:
        eligible = [r for r in records if r['family'] == family]
        for k, record in enumerate(eligible):
            record['split'] = 'held' if k == len(eligible)-1 else 'train'
    (HERE/'assets.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    used = {r['log'] for r in records}
    audit = dict(usable=sorted(used), omitted=[dict(log=n,family=families.get(n),
                 reason='fewer than two usable shared-parser swatches') for n in sorted(available-used)])
    (HERE/'source_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return records


def tile_native(tile, h, w, rng):
    scale = float(rng.choice([.85, 1., 1., 1., 1.15]))
    tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    tile = np.roll(tile, (int(rng.integers(len(tile))), int(rng.integers(tile.shape[1]))), (0, 1))
    return np.tile(tile, (h//len(tile)+2, w//tile.shape[1]+2, 1))[:h, :w].copy()


def degrade(rgb, rng, binary=False):
    """Horizontal shear preserves exact row truth; no vertical warp is applied."""
    rgb = rgb.astype(np.float32)
    fade = rng.uniform(0, .18)
    rgb = rgb*(1-fade)+255*fade
    rgb += rng.normal(0, rng.uniform(0, 3.5), rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rng.random() < .35:
        rgb = cv2.GaussianBlur(rgb, (3, 3), float(rng.uniform(.35, .85)))
    if rng.random() < .35:
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 96))])
        if ok:
            rgb = cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
    if rng.random() < .25:
        shear = rng.uniform(-.004, .004)
        rgb = cv2.warpAffine(rgb, np.float32([[1, shear, -shear*len(rgb)/2], [0, 1, 0]]),
                             (rgb.shape[1], len(rgb)), borderValue=(255, 255, 255))
    if binary or rng.random() < .06:
        # Ordered dithering gives deterministic, bounded-memory 1-bit scans.
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
        threshold = np.tile(bayer, (len(g)//4+1, g.shape[1]//4+1))[:len(g), :g.shape[1]]
        g = np.where(g > 105+threshold*4, 255, 0).astype(np.uint8)
        rgb = np.repeat(g[:, :, None], 3, axis=2)
    return rgb


def make(source, seed, height=1536):
    rng = np.random.default_rng(seed)
    width = int(np.clip(source['width'], 48, 320))
    tiles = [np.array(Image.open(HERE/'assets'/p).convert('RGB')) for p in source['tiles']]
    family = source['family']
    rgb = np.full((height, width, 3), 255, np.uint8)
    rows = []; y = 0; previous = -1
    gulf = family == 'Gulf 3-panel'
    # Gulf panels include a primary interpreted track and percentage mixtures.
    fractions = np.array([0, .25, .75, 1.]) if gulf else np.array([0., 1.])
    edges = np.rint(fractions*width).astype(int)
    while y < height:
        bottom = min(height, y+int(rng.integers(48, 350)))
        choices = [i for i in range(len(tiles)) if i != previous]
        idx = int(rng.choice(choices))
        blank = rng.random() < .08
        if not blank:
            for panel, (a, b) in enumerate(zip(edges, edges[1:])):
                # Central primary track always defines the exact lithology.
                rgb[y:bottom, a:b] = tile_native(tiles[idx], bottom-y, b-a, rng)
                if gulf and panel != 1:
                    other = int(rng.integers(len(tiles)))
                    mix = int(rng.uniform(.15, .7)*(b-a))
                    rgb[y:bottom, a:a+mix] = tile_native(tiles[other], bottom-y, mix, rng)
            rows.append(dict(top_y=y, bottom_y=bottom, lithology=source['labels'][idx], cluster_id=idx))
        if y and rng.random() < .35:
            rgb[y:y+int(rng.integers(1, 3))] = int(rng.integers(20, 130))
        previous, y = idx, bottom
    # Source-measured stable lines plus family-specific percentage grid ticks.
    for position in source['rules']:
        x = min(width-1, int(position*width)); rgb[:, x:x+1] = 50
    if gulf:
        for x in edges[1:-1]:
            rgb[:, x:x+1] = 60
        for x in np.linspace(0, edges[1], 6, dtype=int):
            for y in range(int(rng.integers(20, 90)), height, int(rng.integers(80, 180))):
                rgb[y:y+1, max(0,x-2):min(width,x+3)] = 90
    elif 'grid' in family:
        period = int(rng.integers(80, 300))
        colour = (150, 175, 190) if family == 'Geoservices grid' else (180, 155, 155)
        for y in range(int(rng.integers(period)), height, period):
            rgb[y:y+1] = colour if family != 'D-582 1-bit grid' else 0
        # Independent percentage rules must not become lithology boundaries.
        for x in np.linspace(0, width-1, int(rng.integers(4, 12)), dtype=int):
            rgb[:, x:x+1] = colour if family != 'D-582 1-bit grid' else 0
    elif family == 'MC761 mineral/lithology':
        # Mineral bars sit beside the central lithology track. The central
        # track still has the class in rows; side panels are nuisance texture.
        side = max(2, width//5)
        for row in rows:
            a, b = row['top_y'], row['bottom_y']
            other = int(rng.integers(len(tiles)))
            rgb[a:b, :side] = tile_native(tiles[other], b-a, side, rng)
            rgb[a:b, -side:] = tile_native(tiles[other], b-a, side, rng)
        rgb[:, side:side+1] = 70
        rgb[:, -side:-side+1] = 70
    if source['text'] and rng.random() < .65:
        for _ in range(int(rng.integers(1, 5))):
            text = np.array(Image.open(HERE/'assets'/str(rng.choice(source['text']))).convert('RGB'))
            if text.shape[1] > width//2:
                text = cv2.resize(text, (max(4,width//2), max(4,round(len(text)*width/2/text.shape[1]))))
            h, w = text.shape[:2]
            if h < height and w < width:
                y = int(rng.integers(height-h)); x = int(rng.integers(width-w))
                rgb[y:y+h,x:x+w] = np.minimum(rgb[y:y+h,x:x+w], text)
    rgb = degrade(rgb, rng, family == 'D-582 1-bit grid')
    return rgb, rows


def legend_page(source, seed):
    """Source-layout legend, shuffled entries; labels/boxes remain exact."""
    rng = np.random.default_rng(seed)
    boxes = source['boxes']; lx, ly, rx, by = source['legend_box']
    page = np.full((by-ly, rx-lx, 3), 255, np.uint8)
    order = rng.permutation(len(boxes)); truth = []
    for b, idx in zip(boxes, order):
        a,y,c,d = [b[0]-lx,b[1]-ly,b[2]-lx,b[3]-ly]
        a,y,c,d = max(0,a),max(0,y),min(rx-lx,c),min(by-ly,d)
        if c<=a or d<=y: continue
        tile = np.array(Image.open(HERE/'assets'/source['tiles'][idx]).convert('RGB'))
        page[y:d,a:c] = tile_native(tile,d-y,c-a,rng)
        cv2.rectangle(page,(a,y),(c-1,d-1),(40,40,40),1)
        cv2.putText(page,source['labels'][idx],(min(c+3,page.shape[1]-1),min(d-2,len(page)-1)),
                    cv2.FONT_HERSHEY_SIMPLEX,.32,(20,20,20),1,cv2.LINE_AA)
        truth.append(dict(label=source['labels'][idx],box=[a,y,c,d],tile=source['tiles'][idx]))
    return page, truth
