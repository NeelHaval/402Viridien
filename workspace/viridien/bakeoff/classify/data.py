"""Read-only input preparation; all derived files live under classify/."""
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from harness import io, legend, score

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'classify'
OUT = HERE / 'out'
MANIFEST = ROOT.parent / 'analysis' / 'mix_manifest.json'
GOLD = '0000000009_d-582_0000020435-0001_log06_fe_5in_md'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def font(size=16):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    except OSError:
        return ImageFont.load_default()


def label_key(label):
    return score.normalize(label)


def label_root():
    # The requested location is preferred; also accept the parallel job's sibling folder.
    candidates = [ROOT / 'labelling' / 'out', ROOT.parent / 'labelling' / 'out']
    return next((p for p in candidates if p.exists()), candidates[0])


def vocab_names(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(value, dict):
        value = value.get('vocabulary', value.get('labels', value.get('rocks', value.get('entries', value.get('vocab', value)))))
    if isinstance(value, dict):
        value = list(value.values()) if all(isinstance(x, str) for x in value.values()) else list(value)
    return [str(v.get('label', v.get('name', ''))) if isinstance(v, dict) else str(v) for v in value]


def stratified(rows, cap=40):
    groups = defaultdict(list)
    for row in rows:
        groups[label_key(row['lithology'])].append(row)
    # Round-robin label quotas, then evenly spaced intervals within each label.
    groups=dict(sorted(groups.items()));quotas={k:0 for k in groups};total=0
    while total<min(cap,len(rows)):
        for key in groups:
            if quotas[key]<len(groups[key]):quotas[key]+=1;total+=1
            if total==min(cap,len(rows)):break
    chosen=[]
    for key,count in quotas.items():
        if count:
            chosen.extend(groups[key][i] for i in np.linspace(0,len(groups[key])-1,count,dtype=int))
    return sorted(chosen, key=lambda r: int(r['top_y']))


def get_entries(log):
    """Reuse existing parser cache, redirecting any parser writes to classify/."""
    target = OUT / '_cache' / 'legend'
    target.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'out' / '_cache' / 'legend'
    path = original / (log.name + '.json')
    if path.exists() and not (target / path.name).exists():
        shutil.copy2(path, target / path.name)
        for item in json.loads(path.read_text(encoding='utf-8'))['entries']:
            sw = original / item['swatch_file']
            if sw.exists():
                shutil.copy2(sw, target / sw.name)
    old = io.ROOT
    try:
        io.ROOT = HERE
        return [e for e in legend.parse(log) if e.is_lithology and e.swatch.size]
    finally:
        io.ROOT = old


def datasets(include_consensus=True):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    items = [dict(key='true_d582', source='TRUE', family=manifest['d582']['family'], log=GOLD,
                  truth_path=ROOT / 'gt' / 'gold' / (GOLD + '.csv'), rows=None)]
    if include_consensus:
        for key, item in manifest.items():
            path = label_root() / key / 'consensus.csv'
            if path.exists():
                items.append(dict(item, key=key, source='CONSENSUS', truth_path=path))
    return items


def prepare(item):
    log = io.open_log(item['log'])
    try:
        entries = get_entries(log)
        vocab = list(dict.fromkeys(e.label for e in entries))
        fallback = not vocab
        if fallback:
            path = label_root() / item['key'] / 'vocab.json'
            if not path.exists():
                raise ValueError('Parser found no rocks and vocab.json is absent')
            vocab = list(dict.fromkeys(vocab_names(path)))
        mapping = {label_key(v): v for v in vocab}
        box = log.boxes['interpreted_lithology_column']
        window = item.get('rows') or [box[1], box[3]]
        accepted, excluded = [], []
        for raw in read_csv(item['truth_path']):
            if item['source'] == 'CONSENSUS' and str(raw.get('agreed', '')).casefold() != 'true':
                excluded.append(dict(raw, reason='not agreed=true'))
                continue
            a = max(int(float(raw['top_y'])), int(window[0]), box[1])
            b = min(int(float(raw['bottom_y'])), int(window[1]), box[3])
            truth = raw.get('lithology', raw.get('label', raw.get('consensus', '')))
            if b-a < 12 or label_key(truth) not in mapping:
                excluded.append(dict(raw, reason='under 12 rows' if b-a < 12 else 'truth outside legend vocabulary'))
                continue
            accepted.append(dict(top_y=a, bottom_y=b, lithology=mapping[label_key(truth)], original_truth=truth))
        chosen = stratified(accepted)
        folder = OUT / 'inputs' / item['key']
        folder.mkdir(parents=True, exist_ok=True)
        lb = log.boxes['lithology_legend']
        legend_rgb = io.read_rgb(log, lb, lb[1], lb[3])
        Image.fromarray(legend_rgb).save(folder / 'legend.png')
        patches = []
        for i, row in enumerate(chosen):
            height = min(160, row['bottom_y']-row['top_y'])
            a = (row['top_y']+row['bottom_y']-height)//2
            rgb = io.read_rgb(log, box, a, a+height)
            pid = f"{item['key']}_{a}_{a+height}"
            path = folder / (pid + '.png')
            Image.fromarray(rgb).save(path)
            patches.append(dict(row, patch_id=pid, patch_y0=a, patch_y1=a+height, path=str(path), rgb=rgb))
        info = {k: str(v) if isinstance(v, Path) else v for k, v in item.items()}
        info.update(vocabulary=vocab, parser_failed=fallback, n_accepted=len(accepted), n_selected=len(chosen),
                    excluded=excluded, mode=log.mode, legend_box=lb, column_box=box,
                    truth_sha256=hashlib.sha256(Path(item['truth_path']).read_bytes()).hexdigest())
        dump(folder / 'metadata.json', info)
        return dict(info=info, entries=entries, vocab=vocab, patches=patches, legend=Image.fromarray(legend_rgb), folder=folder)
    finally:
        log.close()


def gallery(case):
    entries, vocab = case['entries'], case['vocab']
    if not entries:
        return case['legend'].copy()
    columns, cellw, cellh = min(4, len(entries)), 250, 100
    canvas = Image.new('RGB', (columns*cellw, ((len(entries)+columns-1)//columns)*cellh), 'white')
    d = ImageDraw.Draw(canvas)
    for i, entry in enumerate(entries):
        x, y = i % columns*cellw, i//columns*cellh
        im = Image.fromarray(entry.swatch)
        im.thumbnail((cellw-12, 60), Image.Resampling.NEAREST)
        canvas.paste(im, (x+5, y+2))
        # Wrap complete names instead of truncating fine-grained rock labels.
        text = f'{vocab.index(entry.label)}: {entry.label}'
        lines = [text[j:j+31] for j in range(0, len(text), 31)]
        d.multiline_text((x+5, y+62), '\n'.join(lines[:2]), fill='black', font=font(12), spacing=1)
    return canvas


def rock_entries(case):
    if not case['entries']:
        return case['legend'].copy()
    # Preserve original printed cells, including labels below or right of swatches.
    box = case['info']['legend_box']
    rows = []
    for e in case['entries']:
        x0, y0, x1, y1 = e.swatch_box
        right=min([other.swatch_box[0]-4 for other in case['entries']
                   if other.swatch_box[0]>x1+4 and abs(other.swatch_box[1]-y0)<(y1-y0)*.6] or [box[2]])
        bottom=min([other.swatch_box[1]-4 for other in case['entries']
                    if other.swatch_box[1]>y1+4 and abs(other.swatch_box[0]-x0)<(x1-x0)*.6]
                   or [min(box[3],y1+max(45,int((y1-y0)*1.05)))])
        crop=case['legend'].crop((max(0,x0-box[0]-3),max(0,y0-box[1]-3),right-box[0],bottom-box[1]))
        crop.thumbnail((240,155),Image.Resampling.NEAREST)
        cell=Image.new('RGB',(250,195),'white');cell.paste(crop,(3,3))
        d=ImageDraw.Draw(cell)
        text=f"{case['vocab'].index(e.label)}: {e.label}"
        d.multiline_text((3,158),'\n'.join(text[j:j+31] for j in range(0,len(text),31)),font=font(12),fill='black',spacing=1)
        rows.append(cell)
    columns=min(4,len(rows))
    canvas=Image.new('RGB',(columns*250,((len(rows)+columns-1)//columns)*195),'white')
    for i,row in enumerate(rows):canvas.paste(row,((i%columns)*250,(i//columns)*195))
    return canvas


def panel(case, patches, kind='gallery'):
    ref = rock_entries(case) if kind == 'choice' else gallery(case)
    # Fit JEV's 1024-square canvas without oversized blank borders.
    n = len(patches)
    if n == 1:
        ref.thumbnail((720, 700), Image.Resampling.NEAREST)
        canvas = Image.new('RGB', (1024, max(330, ref.height+42)), 'white')
        canvas.paste(ref, (8, 30))
        im = Image.fromarray(patches[0]['rgb']); im.thumbnail((270, 240), Image.Resampling.NEAREST)
        canvas.paste(im, (745, 65))
        d = ImageDraw.Draw(canvas)
        d.text((12, 8), 'LEGEND', font=font(18), fill='black')
        d.text((745, 30), 'QUERY PATCH', font=font(18), fill='black')
        return canvas
    ref.thumbnail((950, 650), Image.Resampling.NEAREST)
    width, gridh = 1000, ((n+3)//4)*205
    canvas = Image.new('RGB', (width, gridh+ref.height+55), 'white')
    d = ImageDraw.Draw(canvas)
    for i, patch in enumerate(patches):
        x, y = (i%4)*250, (i//4)*205
        im = Image.fromarray(patch['rgb']); im.thumbnail((240, 170), Image.Resampling.NEAREST)
        canvas.paste(im, (x+5, y+28))
        d.text((x+5,y+3), f'PATCH {i+1}', font=font(18), fill='black')
    d.text((10,gridh+3), 'LEGEND', font=font(18), fill='black')
    canvas.paste(ref, (10,gridh+32))
    return canvas
