"""Split external_tests by WELL into dev (~67%), test (~22%) and verify (~11%).

Whole wells go to one split so no page/window of a held-out well leaks into dev.
Each held-out split gets one KGS, one Idaho and one NL well; Utah has a single
well (one long log) so it stays in dev.

    python -I bakeoff/external/split_external.py external_tests
writes splits.json, eval_set_dev.json, eval_set_test.json, eval_set_verify.json.
eval_set_verify.json is moved into the vault by lock_verify.py.
"""
import csv, json, sys
from collections import Counter
from pathlib import Path

SPLIT = {
    'verify': ['Barlow2-14', 'Marcotte1-12', 'red-brook-1'],
    'test': ['EmmaWard1', 'flat-bay-3', 'irvin1-19'],
}


def main(root):
    root = Path(root)
    well_of = {r['id']: r['well'] for r in csv.DictReader(open(root / 'manifest.csv', encoding='utf-8'))}
    src_of = {r['id']: r['source'] for r in csv.DictReader(open(root / 'manifest.csv', encoding='utf-8'))}
    split_of_well = {w: s for s, ws in SPLIT.items() for w in ws}
    unknown = [w for w in split_of_well if w not in well_of.values()]
    if unknown:
        sys.exit(f'wells not in manifest: {unknown}')
    windows = json.load(open(root / 'eval_set.json', encoding='utf-8'))
    out = {'dev': [], 'test': [], 'verify': []}
    for w in windows:
        out[split_of_well.get(well_of[w['log']], 'dev')].append(w)
    logs = {s: sorted({w['log'] for w in ws}) for s, ws in out.items()}
    rows = {s: sum(w['y1'] - w['y0'] for w in ws) for s, ws in out.items()}
    total_w, total_r = len(windows), sum(rows.values())
    summary = {s: {'windows': len(out[s]), 'window_share': round(len(out[s]) / total_w, 3),
                   'row_share': round(rows[s] / total_r, 3), 'logs': len(logs[s]),
                   'wells': sorted({well_of[l] for l in logs[s]}),
                   'by_source': dict(Counter(src_of[w['log']] for w in out[s]))} for s in out}
    json.dump({'rule': 'whole wells per split; verify is locked (lock_verify.py) and only opened for the final check',
               'summary': summary, 'logs': logs}, open(root / 'splits.json', 'w', encoding='utf-8'), indent=1)
    for s, ws in out.items():
        json.dump(ws, open(root / f'eval_set_{s}.json', 'w', encoding='utf-8'), indent=1)
    for s, v in summary.items():
        print(s, v['windows'], 'windows', f"{v['window_share']:.0%} of windows", f"{v['row_share']:.0%} of rows", v['by_source'], v['wells'])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'external_tests')
