"""Make blind, document-specific vision grids and score matrix answers.

This tool writes experiment files only. It never accepts a live annotation.
"""
import argparse
import collections
import hashlib
import json
import math
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from model import HOME, Images, Store, read
from review_queue import ReviewQueue

OUT = HOME / 'data/terra-grid-20261108'
CONTRACT = 'terra-grid-v1: source legend only; row-major; match/mixed/uncertain/bad_crop; high/medium/low; no approvals'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)


def cached_crop(doc, box, images):
    key = hashlib.sha256((doc['id']+':'+','.join(map(str, box))).encode()).hexdigest()
    path = HOME/'data/review-crops'/(key+'.png')
    if path.exists():
        with Image.open(path) as im:
            return im.convert('RGB')
    return images.crop(doc, box).convert('RGB')


def paste_fit(canvas, im, rect, enlarge=4):
    x, y, w, h = rect
    scale = min(w/im.width, h/im.height, enlarge)
    size = (max(1, round(im.width*scale)), max(1, round(im.height*scale)))
    im = im.resize(size, Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS)
    canvas.paste(im, (x+(w-im.width)//2, y+(h-im.height)//2))


def legend_panels(doc, dest, images):
    entries = [e for e in doc['legend'] if not e.get('deprecated')]
    paths = []
    for page in range(math.ceil(len(entries)/24)):
        subset = entries[page*24:(page+1)*24]
        im = Image.new('RGB', (1024, 52+math.ceil(len(subset)/2)*72), 'white')
        d = ImageDraw.Draw(im)
        d.text((12, 9), 'Source legend: choose only these IDs. Hatch and colour both matter.', font=font(19), fill='black')
        for i, entry in enumerate(subset):
            x, y = (i % 2)*512, 48+(i//2)*72
            d.rectangle((x+4, y, x+506, y+68), outline='#c6cbd2')
            d.text((x+10, y+18), 'L'+str(entry['id']), font=font(20), fill='black')
            paste_fit(im, cached_crop(doc, entry['box'], images), (x+66, y+7, 150, 54))
            label = '\n'.join(textwrap.wrap(entry['label_printed'], width=25)[:3])
            d.multiline_text((x+224, y+5), label, font=font(16), fill='black', spacing=2)
        path = dest/f'legend-{page+1}.png';im.save(path);paths.append(str(path.resolve()))
    return paths


def grid_image(items, asset_root, path):
    cols = 2 if len(items) <= 4 else 4 if len(items) <= 16 else 8 if len(items) <= 64 else 16
    rows = math.ceil(len(items)/cols)
    cell_w, cell_h = 1024//cols, min(238, 952//rows)
    im = Image.new('RGB', (1024, 46+rows*cell_h), 'white');d = ImageDraw.Draw(im)
    d.text((12, 9), 'Crops in row-major order. Return one matrix cell for each numbered item.', font=font(19), fill='black')
    for i, item in enumerate(items):
        x, y = (i % cols)*cell_w, 46+(i//cols)*cell_h
        d.rectangle((x+3, y+2, x+cell_w-4, y+cell_h-3), outline='#adb7c2')
        d.text((x+10, y+7), item['item_id'], font=font(18), fill='black')
        with Image.open(asset_root/item['crop']) as crop:
            paste_fit(im, crop.convert('RGB'), (x+10, y+32, cell_w-20, cell_h-43))
    im.save(path)
    return [rows, cols]


def choose(rows, count, rng):
    rows = list(rows);rng.shuffle(rows)
    # Spread the sample across visual sets before selecting repeats.
    seen = set();ordered = []
    for row in rows:
        if row['pattern_set'] not in seen:
            ordered.append(row);seen.add(row['pattern_set'])
    ordered += [r for r in rows if r not in ordered]
    return ordered[:count]


def prepare(batch_size=8, remaining_only=False):
    audit = HOME/'data/completion-audit-20261108'
    snap = read(audit/'snapshot.json');manifest = read(audit/'manifest.json')
    source = Path(snap['normalization_root']);queue = ReviewQueue(Store());images = Images()
    public = OUT/'blind';public.mkdir(parents=True, exist_ok=True)
    assets = public/'crops';assets.mkdir(exist_ok=True)
    rows = [r for r in manifest['records'] if r['annotation_id'].startswith('queue-') and len(r['targets']) == 1]
    by_doc = collections.defaultdict(list)
    for row in rows:by_doc[row['document_id']].append(row)
    rng = random.Random(61108)
    rich = sorted(by_doc, key=lambda k: (-len(by_doc[k]), k))[:3]
    selected = [r for ident in rich for r in choose(by_doc[ident], 16, rng)]
    other = [r for r in rows if r['document_id'] not in rich]
    stress = [r for r in other if 'd-582' in r['document_name'].lower()]
    selected += choose(stress, 6, rng)
    other = [r for r in other if 'd-582' not in r['document_name'].lower()]
    # At most one per remaining document for a broader layout check.
    rng.shuffle(other);seen = set()
    for row in other:
        if row['document_id'] not in seen:
            selected.append(row);seen.add(row['document_id'])
            if len(seen) == 6:break
    if remaining_only:selected=[]
    truth = {};items = [];grouped = collections.defaultdict(list)
    for n, row in enumerate(selected):
        ident = f'E{n+1:03}';path = assets/(ident+'.png')
        with Image.open(source/row['native_png']) as im:im.convert('RGB').save(path)
        item = dict(item_id=ident, document_id=row['document_id'], crop='crops/'+path.name,
                    source_box=row['box_xyxy'], source_version=row['annotation_version'], cohort='evaluation')
        items.append(item);grouped[row['document_id']].append(item)
        truth[ident] = dict(legend_ids=[t['legend_id'] for t in row['targets']], annotation_id=row['annotation_id'],
                            pattern_set=row['pattern_set'], sample_id=row['sample_id'], document_id=row['document_id'])
    pack = read(HOME/'static/verification-predictions.txt')
    prepared = read(HOME/'data/verification-round-2/prepared.json')
    open_items = []
    with queue.db() as db:
        leases = {r['pattern']:json.loads(r['tasks']) for r in db.execute('select pattern,tasks from leases where session=?', ('017fbd61d7c0cd8f5885a23e',))}
    for group in prepared['sets']:
        for tid in leases.get(group['key'], []):
            task = queue.task_payload(tid)
            if task['segment'].get('approved') or task['segment'].get('status') == 'rejected':continue
            proposal = pack['tasks'].get(tid, {})
            box = proposal.get('box') or task['segment']['box']
            doc = queue.document(task['document_id']);context = task['context_box']
            full = cached_crop(doc, context, images)
            crop = full.crop((box[0]-context[0], box[1]-context[1], box[2]-context[0], box[3]-context[1]))
            ident = f'N{len(open_items)+1:03}';path = assets/(ident+'.png');crop.save(path)
            item = dict(item_id=ident, document_id=doc['id'], crop='crops/'+path.name, source_box=box,
                        source_version=task['version'], cohort='remaining', task_id=tid, pattern_key=group['key'])
            items.append(item);open_items.append(item);grouped[doc['id']].append(item)
    batches = []
    for ident, members in grouped.items():
        doc = queue.document(ident);dest = public/ident;dest.mkdir(exist_ok=True)
        legends = legend_panels(doc, dest, images)
        evaluation = [i for i in members if i['cohort']=='evaluation']
        remaining = [i for i in members if i['cohort']=='remaining']
        groups = [('evaluation', evaluation[i:i+batch_size]) for i in range(0, len(evaluation), batch_size)]
        groups += [('remaining', remaining[i:i+batch_size]) for i in range(0, len(remaining), batch_size)]
        if ident in rich:
            groups += [('repeat4', evaluation[:4]), ('repeat8', evaluation[:8])]
        for j, (purpose, batch) in enumerate(groups):
            if not batch:continue
            key = f'{ident}-{purpose}-{j}';path = dest/(purpose+f'-{j}.png')
            shape = grid_image(batch, public, path)
            batches.append(dict(batch_id=key, purpose=purpose, document_id=ident, grid=str(path.resolve()),
                                legends=legends, shape=shape, item_ids=[i['item_id'] for i in batch],
                                allowed_ids=[e['id'] for e in doc['legend'] if not e.get('deprecated')]))
    for batch in batches:batch['input_hash']=request_hash(batch)
    save(public/'packets.json', dict(schema=1, model='gpt-6.1-terra', batches=batches, items=items))
    pending=[b for b in batches if not (OUT/'cache'/(b['input_hash']+'.json')).exists()]
    save(OUT/'pending-batches.json',dict(contract=CONTRACT,batches=pending))
    for name,purposes in [('base',{'evaluation'}),('small',{'repeat4'}),('medium',{'repeat8'}),('remaining',{'remaining'})]:
        save(public/('run-'+name+'.json'),dict(contract=CONTRACT,batches=[b for b in pending if b['purpose'] in purposes]))
    instructions=('Use each grid with its own source legend. Return row-major matrices with the declared shape. '
                  'Return {model:"gpt-6.1-terra",batches:[{batch_id,input_hash,matrix}]}. '
                  'Echo input_hash exactly. Each real cell has item_id, legend_ids (integer source IDs), '
                  'status (match/mixed/uncertain/bad_crop), confidence (high/medium/low), reason. '
                  'A match has exactly one ID. Use null only for padding. Do not inspect private/ or live labels. '
                  'Do not approve annotations. Save result-base/small/medium/remaining.json as appropriate. '
                  'These are agent-session batches; do not invent API call counts or token savings.')
    (public/'instructions.txt').write_text(instructions,encoding='utf-8')
    if not remaining_only:save(OUT/'private/answer-key.json', truth)
    save(OUT/'private/v4-baseline.json', {i['item_id']:pack['tasks'].get(i.get('task_id')) for i in open_items})
    print(json.dumps(dict(unique_evaluation=len(selected), remaining=len(open_items), batches=len(batches),
                          repeat_predictions=sum(len(b['item_ids']) for b in batches if b['purpose'].startswith('repeat')),
                          documents=len(grouped), blind_manifest=str((public/'packets.json').resolve()))))


def validate_response(batch, result):
    if result.get('batch_id') != batch['batch_id']:raise ValueError('Wrong batch ID')
    matrix = result.get('matrix')
    if not isinstance(matrix, list) or len(matrix) != batch['shape'][0]:raise ValueError('Wrong matrix row count')
    flat = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != batch['shape'][1]:raise ValueError('Wrong matrix column count')
        flat.extend(row)
    expected = batch['item_ids']
    for i, cell in enumerate(flat):
        if i >= len(expected):
            if cell is not None:raise ValueError('Padding cell must be null')
            continue
        if not isinstance(cell, dict) or cell.get('item_id') != expected[i]:raise ValueError('Wrong row-major item mapping')
        ids = cell.get('legend_ids')
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(type(v) is not int or v not in batch['allowed_ids'] for v in ids):raise ValueError('Invalid legend IDs')
        if cell.get('status') not in ('match', 'mixed', 'uncertain', 'bad_crop'):raise ValueError('Invalid status')
        if cell['status']=='match' and len(ids)!=1:raise ValueError('A clean match needs exactly one ID')
        if cell.get('confidence') not in ('high', 'medium', 'low'):raise ValueError('Invalid confidence')
    return flat[:len(expected)]


def request_hash(batch):
    """Cache geometry and label vocabulary together with every input pixel."""
    payload = {k:batch[k] for k in ('shape','item_ids','allowed_ids')}
    payload.update(model='gpt-6.1-terra', contract=CONTRACT, adapter='native-agent-session')
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8'))
    for name in [batch['grid']]+batch['legends']:
        digest.update(hashlib.sha256(Path(name).read_bytes()).digest())
    return digest.hexdigest()


def collect():
    packets=read(OUT/'blind/packets.json');expected={b['batch_id']:b for b in packets['batches']}
    answers={};invalid=[]
    for name in ('base','small','medium','remaining'):
        path=OUT/('result-'+name+'.json')
        if not path.exists():continue
        for result in read(path)['batches']:
            key=result.get('batch_id')
            try:
                if key not in expected:raise ValueError('Unknown batch')
                if key in answers:raise ValueError('Duplicate batch')
                validate_response(expected[key],result)
                digest=request_hash(expected[key])
                if result.get('input_hash')!=digest:raise ValueError('Response belongs to different or unsealed inputs')
                result=dict(result,model='gpt-6.1-terra')
                answers[key]=result
                save(OUT/'cache'/(result['input_hash']+'.json'),result)
            except ValueError as exc:invalid.append(dict(batch_id=key,error=str(exc)))
    for key,batch in expected.items():
        if key in answers:continue
        cached=OUT/'cache'/(request_hash(batch)+'.json')
        if cached.exists():
            result=read(cached)
            if result.get('input_hash')==request_hash(batch):
                validate_response(batch,result);answers[key]=dict(result,batch_id=key)
    retry=[dict(b,input_hash=request_hash(b)) for k,b in expected.items() if k not in answers]
    save(OUT/'retry-batches.json',dict(batches=retry,invalid=invalid))
    save(OUT/'terra-results.json',dict(model='gpt-6.1-terra',contract=CONTRACT,batches=list(answers.values())))
    print(json.dumps(dict(valid_batches=len(answers),retry_batches=len(retry),invalid=invalid)))


def score():
    packets = read(OUT/'blind/packets.json');truth = read(OUT/'private/answer-key.json')
    responses = read(OUT/'terra-results.json')['batches']
    if len({r['batch_id'] for r in responses})!=len(responses):raise ValueError('Duplicate result batches')
    answers = {r['batch_id']:r for r in responses}
    base = {};repeat = [];remaining = [];errors = [];sizes = collections.defaultdict(list)
    for batch in packets['batches']:
        if batch['batch_id'] not in answers:
            errors.append(dict(batch_id=batch['batch_id'], error='missing batch'));continue
        try:cells = validate_response(batch, answers[batch['batch_id']])
        except ValueError as exc:
            errors.append(dict(batch_id=batch['batch_id'], error=str(exc)));continue
        for cell in cells:
            c = dict(cell, batch_id=batch['batch_id'])
            if batch['purpose']=='remaining':remaining.append(c);continue
            c['agreement'] = sorted(c['legend_ids']) == sorted(truth[c['item_id']]['legend_ids'])
            sizes[len(batch['item_ids'])].append(c)
            if batch['purpose'].startswith('repeat'):repeat.append(c)
            else:base[c['item_id']]=c
    def stats(cells):
        high = [c for c in cells if c['status']=='match' and c['confidence']=='high']
        return dict(n=len(cells), exact_agreement=sum(c['agreement'] for c in cells),
                    high_matches=len(high), high_agreement=sum(c['agreement'] for c in high),
                    flagged=sum(c['status']!='match' for c in cells))
    changes = [dict(item_id=c['item_id'], base=base[c['item_id']]['legend_ids'], repeat=c['legend_ids'], batch_id=c['batch_id'])
               for c in repeat if c['item_id'] in base and sorted(c['legend_ids'])!=sorted(base[c['item_id']]['legend_ids'])]
    repeat_by_size=collections.defaultdict(dict)
    for size,cells in sizes.items():
        for cell in cells:repeat_by_size[size][cell['item_id']]=cell
    common=set.intersection(*(set(repeat_by_size[n]) for n in (4,8,16))) if all(n in repeat_by_size for n in (4,8,16)) else set()
    paired={str(n):stats([repeat_by_size[n][i] for i in sorted(common)]) for n in (4,8,16) if n in repeat_by_size}
    report = dict(model='gpt-6.1-terra', note='Agreement with one saved human review, not established accuracy. Purposive small sample; not an independent final test.',
                  evaluation=stats(list(base.values())), grid_sizes={str(k):stats(v) for k,v in sizes.items()},
                  paired_grid_sizes=paired,
                  mapping_errors=errors, repeat_label_changes=changes, remaining_proposals=len(remaining),
                  disagreements=[dict(c, human_ids=truth[c['item_id']]['legend_ids'], source=truth[c['item_id']]) for c in base.values() if not c['agreement']],
                  remaining=remaining)
    save(OUT/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('disagreements','remaining')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action', choices=['prepare','collect','score'])
    parser.add_argument('--batch-size',type=int,choices=[4,8,16,32,64,128],default=8)
    parser.add_argument('--out-dir',type=Path,help='Use a new directory to keep a separate round.')
    parser.add_argument('--remaining-only',action='store_true');args=parser.parse_args()
    if args.out_dir:OUT=args.out_dir.resolve()
    prepare(args.batch_size,args.remaining_only) if args.action=='prepare' else {'collect':collect,'score':score}[args.action]()
