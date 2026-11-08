"""Suggestion pipeline for the label tool: cheap filters first, JEV only on what is left.

Stage 1 colour: in a coloured legend, keep only entries whose fill colour matches the interval's fill.
Stage 2 family: keep entries in the visual families JEV gives real weight to (labeltool/twostage.py).
Stage 3 JEV v2: a same-scale panel with just the survivors (no call when one survives), then the
look-alike tie-break from jev2.decide (owner's nearest label in the same log).

    python -X utf8 -B labeltool/suggest.py --eval      # score on gold log06, owner labels, review-page confirmations
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2  # noqa: E402
from labeltool.common import normalize  # noqa: E402

VERSION = 'suggest-1'


def fill_colours(rgb, k=3):
    """Most common non-ink colours (quantised to 32 levels) with their shares."""
    a = np.asarray(rgb)[..., :3].reshape(-1, 3).astype(int)
    if not len(a):
        return []
    a = a[a.sum(1) > 150]  # drop black ink and dark borders
    if not len(a):
        return []
    q = a // 32
    values, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)[:k]
    return [(values[i] * 32 + 16, counts[i] / counts.sum()) for i in order]


def coloured(colours):
    return any(int(np.ptp(c)) >= 40 for c, _ in colours)


def colour_match(query_cols, tile_cols, tol=48):
    """The tile's main fill must be one of the query's top fills (or vice versa)."""
    if not query_cols or not tile_cols:
        return True
    q, t = query_cols[0][0], tile_cols[0][0]
    return any(int(np.abs(t - c).max()) <= tol for c, s in query_cols if s >= 0.1) or \
        any(int(np.abs(q - c).max()) <= tol for c, s in tile_cols if s >= 0.1)


_family_index = None


def family_index():
    global _family_index
    if _family_index is None:
        from labeltool import twostage
        try:
            _family_index = twostage.family_index()
        except FileNotFoundError:
            _family_index = {}
    return _family_index


def suggest(name, top, bottom, client=None, use_family=True):
    ctx = jev2.Context.get(name)
    with ctx['lock']:
        query = jev2.read_interval(ctx['io'], ctx['log'], top, bottom)
    tiles = ctx['tiles']
    qcols = fill_colours(query)
    tile_cols = {t['raw']: fill_colours(t['swatch']) for t in tiles}
    legend_coloured = sum(coloured(c) for c in tile_cols.values()) >= 2
    stages = dict(all=[t['label'] for t in tiles], coloured_legend=legend_coloured)
    keep = [t for t in tiles if not legend_coloured or colour_match(qcols, tile_cols[t['raw']])] or tiles
    stages['after_colour'] = [t['label'] for t in keep]
    if use_family and len(keep) > 1:
        from labeltool import twostage
        index = family_index()
        if index:
            from strategies._jev import Client
            client = client or Client(timeout=45, retries=1)
            fam = twostage.query_family(query, client) if client.key else None
            if fam:
                ranked = sorted(fam['probs'].items(), key=lambda kv: -kv[1])
                fams = [k for k, v in ranked if v >= ranked[0][1] - 0.25 and v > 0.05]
                narrowed = [t for t in keep if index.get((name, t['id']), {}).get('pattern') in fams]
                stages['family'] = fams
                # Families are a soft filter: never let them remove every colour match.
                if narrowed:
                    keep = narrowed
    stages['after_family'] = [t['label'] for t in keep]
    result = jev2.decide(name, top, bottom, jev2.ask(name, top, bottom, client=client, only={t['raw'] for t in keep}))
    return dict(result, stages=stages, pipeline=VERSION)


def cases():
    from labeltool import twostage
    out = twostage.cases()
    root = Path(__file__).resolve().parents[1]
    review = root / 'synth' / 'data' / 'review.jsonl'
    if review.exists():
        catalog = json.loads((root / 'synth' / 'data' / 'catalog.json').read_text(encoding='utf-8'))
        labels = {p['pid']: p['label'] for p in catalog['patterns']}
        index = json.loads((root / 'synth' / 'data' / 'sheets' / 'index.json').read_text(encoding='utf-8'))
        stacks = {}
        for line in review.read_text(encoding='utf-8').splitlines():
            e = json.loads(line)
            s = stacks.setdefault(e['pid'], [])
            if e.get('kind') == 'undo':
                if s: s.pop()
            else:
                s.append(e)
        for pid, s in stacks.items():
            if not s:
                continue
            verdicts = s[-1]['decision'].get('verdicts') or {}
            for n, v in verdicts.items():
                c = index.get(pid, {}).get('candidates', {}).get(n)
                if v == 'y' and c:
                    out.append(dict(log=c['log'], top=c['top_y'], bottom=c['bottom_y'], truth=labels[pid], src='review'))
    return out


def evaluate(limit=None, workers=6):
    from concurrent.futures import ThreadPoolExecutor
    from strategies._jev import Client
    from labeltool.twostage import same
    rows = cases()[:limit] if limit else cases()

    def run(c):
        client = Client(timeout=45, retries=1)
        try:
            base = jev2.decide(c['log'], c['top'], c['bottom'], jev2.ask(c['log'], c['top'], c['bottom'], client=client))
            new = suggest(c['log'], c['top'], c['bottom'], client=client)
        except Exception as e:  # keep the run going; report the failure
            return dict(c, error=type(e).__name__)
        return dict(c, base=base.get('label'), new=new.get('label'), ambiguous=new.get('ambiguous'),
                    stages=new['stages'], base_ok=same(base.get('label'), c['truth']), new_ok=same(new.get('label'), c['truth']),
                    truth_in_after_colour=any(same(x, c['truth']) for x in new['stages']['after_colour']),
                    truth_in_after_family=any(same(x, c['truth']) for x in new['stages']['after_family']))
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(run, rows))
    out = Path(__file__).resolve().parents[1] / 'analysis' / 'suggest_eval.json'
    out.write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
    by = {}
    for r in res:
        b = by.setdefault(r['src'], dict(n=0, base=0, new=0, colour_kept_truth=0, family_kept_truth=0, errors=0))
        b['n'] += 1
        if 'error' in r:
            b['errors'] += 1; continue
        b['base'] += r['base_ok']; b['new'] += r['new_ok']
        b['colour_kept_truth'] += r['truth_in_after_colour']; b['family_kept_truth'] += r['truth_in_after_family']
    print(json.dumps(by, indent=1))
    for r in res:
        if 'error' in r or not r['new_ok']:
            print(r['src'], r['log'][:30], r['top'], 'truth', r['truth'], '| base', r.get('base'), '| new', r.get('new'),
                  '| after colour', len(r.get('stages', {}).get('after_colour', [])), '| after family', r.get('stages', {}).get('after_family'), r.get('error', ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    if a.eval:
        evaluate(a.limit)
