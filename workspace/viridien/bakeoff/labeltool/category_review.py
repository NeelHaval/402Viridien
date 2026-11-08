"""Named-category review of the detector output (labeltool/data/prefill.jsonl: S11 FFT + JEV + colour).

Every predicted legend label is mapped to a canonical rock category (analysis/categories/catmap.json, from an
LLM pass over all legend names). Per category: share of the column rows it covers, confidence, the
legend labels inside it and how hard each is (confidence, low tier, S11-vs-JEV disagreement), the other
categories it shares legend pictures with (look-alikes), and 20 example intervals spread across logs.

    python -X utf8 -B labeltool/category_review.py  -> analysis/categories/review.json  (page: famreview.py /cat)
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'categories'
TIER_CONF = {'high': 0.85, 'medium': 0.55, 'low': 0.25}
N_EX = 20


def confidence(r):
    for s in r['suggestions']:
        if s.get('label') == r.get('suggestion') and s.get('source') == r.get('best_source'):
            p = s.get('probability', s.get('confidence'))
            if isinstance(p, (int, float)):
                return float(p)
    return TIER_CONF.get(r.get('tier'), 0.25)


def agree(r):
    """S11 and JEV both voted and named the same label: None if one of them did not vote."""
    s11 = next((s['label'] for s in r['suggestions'] if s['source'].startswith('S11/native')), None)
    jev = next((s['label'] for s in r['suggestions'] if s['source'].startswith('JEV')), None)
    return None if s11 is None or jev is None else s11 == jev


def main():
    cm = json.loads((OUT / 'catmap.json').read_text(encoding='utf-8'))['map']
    cat = lambda lab: (cm.get(lab) or {}).get('cat', 'Unmapped') if lab else 'No prediction'  # noqa: E731
    recs = [json.loads(l) for l in (ROOT / 'labeltool' / 'data' / 'prefill.jsonl').open(encoding='utf-8')]
    total = sum(r['bottom_y'] - r['top_y'] for r in recs)
    by = defaultdict(list)
    for r in recs:
        r['conf'] = confidence(r); r['agree'] = agree(r); r['h'] = r['bottom_y'] - r['top_y']
        by[cat(r.get('suggestion'))].append(r)
    # look-alike categories: one legend picture (repeats merged) carrying names of two categories, or two
    # pictures the LLM checker put in the same "same picture" row of a v2 group
    LD = ROOT / 'analysis' / 'legend_families2'
    G = json.loads((LD / 'groups.json').read_text(encoding='utf-8'))['groups']
    llm = {v['id']: v for v in json.loads((LD / 'llm.json').read_text(encoding='utf-8')).get('groups', [])} if (LD / 'llm.json').exists() else {}
    pair = Counter()
    skip = {'Not lithology', 'Unknown', 'Unmapped'}
    for g in G:
        byk = {it['k']: it for it in g['items']}
        sets = [[it] for it in g['items']] + [[byk[k] for k in row if k in byk] for row in (llm.get(g['id'], {}).get('split') or [])]
        for s in sets:
            cs = sorted({cat(n) for it in s for n in it['names']} - skip)
            for i, a in enumerate(cs):
                for b in cs[i + 1:]:
                    pair[(a, b)] += 1
    rng = random.Random(0)
    cats = []
    for name, rs in by.items():
        rows = sum(r['h'] for r in rs)
        confs = [r['conf'] for r in rs]
        ag = [r['agree'] for r in rs if r['agree'] is not None]
        labels = []
        for lab, lr in Counter(r.get('suggestion') for r in rs).most_common():
            sub = [r for r in rs if r.get('suggestion') == lab]
            sag = [r['agree'] for r in sub if r['agree'] is not None]
            labels.append(dict(label=lab, rows=sum(r['h'] for r in sub), share_in_cat=sum(r['h'] for r in sub) / max(1, rows),
                               intervals=len(sub), logs=len({r['log'] for r in sub}),
                               conf_median=round(statistics.median(r['conf'] for r in sub), 3),
                               low_share=round(sum(r['tier'] == 'low' for r in sub) / len(sub), 3),
                               disagree=round(1 - sum(sag) / len(sag), 3) if sag else None))
        # 20 examples: intervals tall enough to see, round-robin over logs, then low confidence first
        cand = [r for r in rs if r['h'] >= 20] or rs
        rng.shuffle(cand)
        per_log = defaultdict(list)
        for r in cand:
            per_log[r['log']].append(r)
        ex, queues = [], list(per_log.values())
        while len(ex) < N_EX and any(queues):
            for q in queues:
                if q and len(ex) < N_EX:
                    ex.append(q.pop())
        ex.sort(key=lambda r: r['conf'])
        looks = sorted(((b if a == name else a), n) for (a, b), n in pair.items() if name in (a, b))
        cats.append(dict(
            name=name, rows=rows, share=rows / total, intervals=len(rs), logs=len({r['log'] for r in rs}),
            conf_median=round(statistics.median(confs), 3), low_share=round(sum(r['tier'] == 'low' for r in rs) / len(rs), 3),
            disagree=round(1 - sum(ag) / len(ag), 3) if ag else None,
            lookalikes=[dict(cat=c, groups=n) for c, n in sorted(looks, key=lambda x: -x[1])],
            labels=labels,
            examples=[dict(id=r['id'], log=r['log'], top_y=r['top_y'], bottom_y=r['bottom_y'], label=r.get('suggestion'),
                           conf=round(r['conf'], 3), tier=r['tier'], agree=r['agree'], source=r.get('best_source'),
                           reason=r.get('reason', '')[:120]) for r in ex]))
    cats.sort(key=lambda c: -c['rows'])
    logs = {r['log'] for r in recs}
    (OUT / 'review.json').write_text(json.dumps(dict(categories=cats, total_rows=total, intervals=len(recs), logs=len(logs),
                                                     detector='prefill: S11 native FFT + JEV cluster + colour'), indent=1), encoding='utf-8')
    for c in cats[:40]:
        print(f"{c['name'][:22]:22s} {c['share'] * 100:5.1f}%  conf {c['conf_median']:.2f}  low {c['low_share'] * 100:3.0f}%  "
              f"disagree {'' if c['disagree'] is None else round(c['disagree'] * 100)}%  labels {len(c['labels'])}")


if __name__ == '__main__':
    main()
