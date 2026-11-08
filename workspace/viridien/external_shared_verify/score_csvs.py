"""Score per-window prediction CSVs against this folder's exact truth.

    cd viridien/bakeoff
    python ../external_shared/score_csvs.py ../external_shared PRED_DIR

Prediction file per window: PRED_DIR/<id>__y<y0>_<y1>.csv, else PRED_DIR/<id>.csv
(columns top_y,bottom_y,lithology[,cluster_id,top_depth,bottom_depth]; full-image rows,
bottom exclusive). A missing prediction scores as an empty prediction.
Uses the bake-off scorer (harness.score), so run it from viridien/bakeoff.
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from harness import io, score  # noqa: E402

METRICS = ('row_acc', 'boundary_f1', 'cluster_ari', 'depth_mae_ft')


def main(root, pred_dir):
    root, pred_dir = Path(root), Path(pred_dir)
    items = json.loads((root / 'eval_set.json').read_text(encoding='utf-8'))
    source = {r['id']: r['source'] for r in csv.DictReader(open(root / 'manifest.csv', encoding='utf-8'))}
    by_source = defaultdict(list)
    print(f"{'window':52} " + ' '.join(f'{m:>12}' for m in METRICS))
    for it in items:
        win = (it['y0'], it['y1'])
        truth = io.load_rows(root / 'truth' / f"{it['log']}.csv")
        pred_path = pred_dir / f"{it['log']}__y{win[0]}_{win[1]}.csv"
        if not pred_path.exists():
            pred_path = pred_dir / f"{it['log']}.csv"
        pred = (io.load_rows(pred_path) or []) if pred_path.exists() else []
        res = score.score(pred, truth, win)
        by_source[source.get(it['log'], '?')].append(res)
        name = f"{it['log']} {win[0]}-{win[1]}" + ('' if pred_path.exists() else ' (missing)')
        print(f'{name:52} ' + ' '.join(_fmt(res.get(m)) for m in METRICS))
    print()
    allres = [r for rs in by_source.values() for r in rs]
    for name, rs in sorted(by_source.items()) + [('ALL', allres)]:
        print(f'{name + " (" + str(len(rs)) + ")":52} ' + ' '.join(_fmt(_mean(rs, m)) for m in METRICS))


def _mean(rs, m):
    vals = [r[m] for r in rs if r.get(m) is not None]
    return statistics.mean(vals) if vals else None


def _fmt(v):
    return f'{v:12.3f}' if isinstance(v, (int, float)) else f'{"-":>12}'


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
