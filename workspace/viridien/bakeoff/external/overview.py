"""Compact overview of external_tests: one small card per test log, grouped by source, with s11 scores."""
import argparse
import csv, json, sys
from collections import defaultdict
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
parser=argparse.ArgumentParser()
parser.add_argument('dataset',type=Path)
parser.add_argument('metrics',type=Path,nargs='?')
parser.add_argument('--only',help='Comma-separated page IDs')
parser.add_argument('--png-only',action='store_true')
args=parser.parse_args()
T=args.dataset.resolve(); M=args.metrics.resolve() if args.metrics else None
out = T / 'overview'; out.mkdir(exist_ok=True)
man = {r['id']: r for r in csv.DictReader(open(T / 'manifest.csv', encoding='utf-8'))}
per = defaultdict(list)
if M:
    for r in json.loads(M.read_text(encoding='utf-8'))['logs']: per[r['log']].append(r)
names = {'kgs': 'Kansas (KGS)', 'idaho': 'Idaho OGCC', 'utah': 'Utah DOGM', 'nl': 'Newfoundland'}
groups = defaultdict(list)
for i, r in man.items():
    if args.only and i not in args.only.split(','): continue
    card = Image.open(T / 'gallery' / f'{i}.png').convert('RGB')
    ann=json.loads((T/'bbox_annotation'/f'{i}.json').read_text(encoding='utf-8'))
    with (T/'truth'/f'{i}.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    box=ann['bounding_boxes']['interpreted_lithology_column']
    top=max(box[1],int(rows[0]['top_y'])-20);bottom=min(box[3],top+1500)
    column_height=max(1,round((bottom-top)*100/(box[2]-box[0])))
    column_y=card.height-column_height-140
    if column_y <= 34: raise ValueError(f'{i}: gallery layout does not match the card builder')
    # Preserve the complete legend and reserve a separate panel for column + truth.
    legend=card.crop((0,0,card.width,column_y-36))
    legend.thumbnail((136,100),Image.Resampling.LANCZOS)
    column=card.crop((0,column_y-32,card.width,min(card.height,column_y-32+round(230*card.width/136))))
    column=column.resize((136,min(230,round(column.height*136/column.width))),Image.Resampling.LANCZOS)
    preview=Image.new('RGB',(136,330),'white')
    preview.paste(legend,((136-legend.width)//2,0));preview.paste(column,(0,100))
    target=out/f'{i}.png'
    backup=target.with_name(target.name+'.bak_review')
    if target.exists() and not backup.exists():backup.write_bytes(target.read_bytes())
    preview.save(target)
    ws = per.get(i, [])
    acc = sum(w.get('row_acc', 0) for w in ws) / len(ws) if ws else None
    bf = sum(w.get('boundary_f1', 0) for w in ws) / len(ws) if ws else None
    err = sum('error' in w for w in ws)
    groups[r['source']].append((i, r, acc, bf, err))
if args.png_only:
    print('Overview PNGs written:',len(args.only.split(',')) if args.only else len(man));raise SystemExit(0)
h = ['<!doctype html><html><head><style>body{margin:0;font-family:var(--font-sans);color:var(--foreground)}h3{font-size:15px;margin:20px 0 6px}p{font-size:13px;margin:4px 0 10px;line-height:1.45}'
     '.g{display:flex;flex-wrap:wrap;gap:8px}.c{width:136px;background:var(--card);border-radius:var(--radius);overflow:hidden}.c img{display:block;width:136px}'
     '.c div{padding:6px 8px;font-size:11px;line-height:1.4;color:var(--muted-foreground)}.c b{color:var(--foreground);font-size:12px}.bad{color:var(--destructive)}</style></head><body>']
tot = sum(len(v) for v in groups.values())
h.append(f'<p>{tot} outside test logs in the challenge layout (image + 3 bboxes + truth CSV). Each card: the log\'s own legend, the start of its lithology column, and the truth coloured by rock type. Numbers are s11 on that log (row accuracy / boundary F1).</p>')
for src in ('kgs', 'idaho', 'utah', 'nl'):
    items = groups.get(src, [])
    if not items: continue
    h.append(f'<h3>{names[src]}: {len(items)} logs</h3><div class="g">')
    for i, r, acc, bf, err in items:
        sc = 'not run' if acc is None else f'{acc:.2f} / {bf:.2f}'
        e = f' <span class="bad">{err} crash</span>' if err else ''
        h.append(f'<div class="c"><img src="{(out / (i + ".png")).as_posix()}"><div><b>{r["well"]} p{r["page"]}</b><br>{r["truth_intervals"]} beds, {r["rock_type_count"]} rock types<br>s11: {sc}{e}</div></div>')
    h.append('</div>')
target=T/'overview.html'
if target.exists() and not target.with_name(target.name+'.bak_review').exists():target.with_name(target.name+'.bak_review').write_bytes(target.read_bytes())
(T / 'overview.html').write_text(''.join(h) + '</body></html>', encoding='utf-8'); print(T / 'overview.html', tot)
