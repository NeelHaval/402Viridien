"""Build per-document work files for the pattern fleet.

python -X utf8 data/genlab/fleet/worklist.py   (from pattern_lab)
Writes data/genlab/fleet/work/<doc_id>.json for every training document.
"""
import json, re, glob, os, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C = ROOT / 'data/genlab/candidates'
V = ROOT / 'data/genlab/verdicts'
W = Path(__file__).resolve().parent / 'work'
PREFILL = ROOT.parent / 'bakeoff/labeltool/data/prefill.jsonl'
norm = lambda s: re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def main():
    W.mkdir(parents=True, exist_ok=True)
    state = json.load(open(ROOT / 'data/genlab/state.json', encoding='utf-8')) if (ROOT / 'data/genlab/state.json').exists() else {}
    generated = {(c['doc_id'], c['legend_id']) for c in state.get('cards', [])}
    reports = {}
    for f in glob.glob(str(C / '*.json')):
        d = json.load(open(f, encoding='utf-8'))
        if isinstance(d, dict) and 'entries' in d:
            reports[d['doc_id']] = d
    byname = {d['name']: d['doc_id'] for d in reports.values()}
    pre = collections.defaultdict(list)
    for line in open(PREFILL, encoding='utf-8'):
        r = json.loads(line)
        doc = byname.get(r['log'])
        if doc:
            pre[doc].append(r)
    out = []
    for doc, d in sorted(reports.items()):
        if not d.get('training_allowed'):
            continue
        verdicts = json.load(open(V / f'{doc}.json', encoding='utf-8')) if (V / f'{doc}.json').exists() else {}
        rows_by = collections.Counter()
        for r in pre[doc]:
            rows_by[norm(r.get('suggestion'))] += r['bottom_y'] - r['top_y']
        total = sum(rows_by.values()) or 1
        entries = []
        for e in d['entries']:
            v = verdicts.get(str(e['legend_id']))
            done = (doc, e['legend_id']) in generated and (e['tier'] == 'human' or (v and v['verdict'] == 'confident'))
            if done or e['tier'] == 'not-lithology':
                continue
            ivs = sorted((r for r in pre[doc] if norm(r.get('suggestion')) == norm(e['label'])),
                         key=lambda r: -(r['bottom_y'] - r['top_y']))[:30]
            entries.append(dict(
                legend_id=e['legend_id'], label=e['label'], tier=e['tier'], reason=e['reason'],
                swatch_box=e['swatch_box'], cycles=e.get('cycles'), twins=e.get('twins'),
                seam_x=e.get('seam_x'), seam_y=e.get('seam_y'),
                evidence_boxes=[x['box'] for x in (e.get('evidence') or [])][:12],
                reviewer=v, sheet_png=e.get('sheet_png'), row_y=e.get('row_y'), row_height=e.get('row_height'),
                est_row_share=round(rows_by[norm(e['label'])] / total, 4),
                guessed_intervals=[[r['top_y'], r['bottom_y']] for r in ivs]))
        col = None
        try:
            import sys
            sys.path.insert(0, str(ROOT))
            import model
            col = model.Store().get(doc)['boxes']['interpreted_lithology_column']
        except Exception as ex:  # noqa: BLE001
            col = f'read model.Store().get(doc_id)["boxes"]["interpreted_lithology_column"] ({ex})'
        w = dict(doc_id=doc, name=d['name'], column_box=col, legend_total=len(d['entries']),
                 done_entries=len(d['entries']) - len(entries), entries=entries,
                 legend=[dict(legend_id=e['legend_id'], label=e['label'], swatch_box=e['swatch_box'], tier=e['tier']) for e in d['entries']])
        json.dump(w, open(W / f'{doc}.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        out.append((doc, len(entries), sum(x['est_row_share'] for x in entries)))
    json.dump([dict(doc_id=a, pending=b, share=c) for a, b, c in out], open(W / '_index.json', 'w'), indent=1)
    print(len(out), 'documents', sum(b for _, b, _ in out), 'pending entries')


if __name__ == '__main__':
    main()
