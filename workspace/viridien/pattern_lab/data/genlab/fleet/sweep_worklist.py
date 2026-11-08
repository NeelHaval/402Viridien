"""Sweep work list: every training-well legend entry the board marks Failed / Unsure / Rejected / Not started,
chunked into <= CHUNK entries per unit, with the owner's newest labelled boxes (data/normalized/current.json) attached.
    python -X utf8 data/genlab/fleet/sweep_worklist.py   (from pattern_lab)"""
import json, collections, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHUNK = 8
TODO = {'Failed', 'Unsure', 'Rejected', 'Not started'}

board = json.load(urllib.request.urlopen('http://127.0.0.1:8782/board.json', timeout=60))
todo = collections.defaultdict(list)
for e in board['entries']:
    if e['state'] in TODO:
        todo[e['doc_id']].append(e)
norm = json.load(open(ROOT / 'data/normalized/current.json', encoding='utf-8'))
human = collections.defaultdict(list)
for r in norm['records']:
    if r.get('split') == 'train' and len(r.get('targets') or []) == 1:
        human[(r['document_id'], r['targets'][0]['legend_id'])].append(r['box_xyxy'])
index = []
for doc, ents in sorted(todo.items()):
    base = json.load(open(HERE / f'work/{doc}.json', encoding='utf-8')) if (HERE / f'work/{doc}.json').exists() else None
    if base is None:
        import sys; sys.path.insert(0, str(ROOT)); import model
        d = model.Store().get(doc)
        cand = json.load(open(ROOT / f'data/genlab/candidates/{doc}.json', encoding='utf-8'))
        base = dict(doc_id=doc, name=cand['name'], column_box=d['boxes']['interpreted_lithology_column'], entries=[],
                    legend=[dict(legend_id=x['legend_id'], label=x['label'], swatch_box=x['swatch_box'], tier=x['tier']) for x in cand['entries']])
    cand = {x['legend_id']: x for x in json.load(open(ROOT / f'data/genlab/candidates/{doc}.json', encoding='utf-8'))['entries']}
    old = {x['legend_id']: x for x in base.get('entries', [])}
    rows = []
    for e in ents:
        lid = e['legend_id']
        if not isinstance(lid, int):
            continue
        x = dict(old.get(lid) or dict(legend_id=lid, label=e['label'], tier=cand[lid]['tier'], reason=cand[lid]['reason'],
                                       swatch_box=cand[lid]['swatch_box'], guessed_intervals=[]))
        x['board_state'] = e['state']
        x['last_note'] = e.get('note') or e.get('reviewer_note')
        x['human_boxes'] = human.get((doc, lid), [])[:20]
        rows.append(x)
    rows.sort(key=lambda x: -len(x['human_boxes']))
    for c in range(0, len(rows), CHUNK):
        wid = f'{doc}__s{c // CHUNK}'
        w = dict(base, entries=rows[c:c + CHUNK], chunk=c // CHUNK)
        json.dump(w, open(HERE / f'work/{wid}.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        index.append(dict(wid=wid, doc_id=doc, chunk=c // CHUNK, pending=len(w['entries']),
                          human=sum(bool(x['human_boxes']) for x in w['entries'])))
json.dump(index, open(HERE / 'work/_sweep.json', 'w'), indent=1)
print(len(index), 'units', sum(i['pending'] for i in index), 'entries', sum(i['human'] for i in index), 'with new human boxes')
