"""Score the current classifier against the blind LLM picks of notes/llm_review.py (no tuning on them).
A pick counts as agreement when the LLM's legend entry is the classifier's label or inside its twin group."""
import json, sys, pickle
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import legend_classify as C  # noqa: E402
D = Path(__file__).resolve().parents[1] / 'analysis' / 'categories'
R = [r for r in json.loads((D / 'llmrev' / 'results.json').read_text(encoding='utf-8'))['rows'] if not r.get('dup') and r['llm']]
CACHE = D / 'llmrev' / 'windows.pkl'
W = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
for r in R:
    if (r['log'], r['y']) not in W:
        W[(r['log'], r['y'])] = C.read_window(r['log'], r['y'], r['h'])
CACHE.write_bytes(pickle.dumps(W))
fam = {}
import csv
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]  # noqa: E731
for x in csv.DictReader(open(D.parents[2] / 'analysis' / 'label_shares_per_log.csv', encoding='utf-8')):
    fam[key(x['log'])] = x['family']


def score(tag):
    c, f = Counter(), defaultdict(Counter)
    for r in R:
        o = C.classify(W[(r['log'], r['y'])], r['log'])
        llm = r['llm']; lab = o.get('label')
        names = set(lab.split(' / ')) if lab else set()
        if o['state'] != 'ok':
            k = 'rej_llm_mixed' if llm in ('mixed', 'none') else 'rej_llm_label'
        else:
            k = 'agree' if llm in names else ('named_llm_mixed' if llm in ('mixed', 'none') else 'disagree')
        c[k] += 1; c[k + ('_sure' if r['sure'] == 'high' else '')] += r['sure'] == 'high'
        f[fam.get(key(r['log']), '?')][k] += 1
    n = c['agree'] + c['disagree']; ns = c['agree_sure'] + c['disagree_sure']
    print(f"{tag:28s} named&LLM-labelled agree {c['agree']}/{n} = {c['agree'] / max(1, n):.0%} | LLM very sure {c['agree_sure']}/{ns} = {c['agree_sure'] / max(1, ns):.0%} "
          f"| rejected: LLM mixed {c['rej_llm_mixed']} LLM label {c['rej_llm_label']} | named but LLM mixed {c['named_llm_mixed']}")
    return f


if __name__ == '__main__':
    print(len(R), 'blind-labelled windows')
    variants = [('v3 (scales off, no twins)', dict(scales=(), twin_dE=-1)), ('+ twins', dict(scales=(), twin_dE=4.0)),
                ('+ twins + scale', dict(scales=(0.5, 0.75, 1.0, 1.5, 2.0), twin_dE=4.0)),
                ('+ twins + scale + frame .6', dict(scales=(0.5, 0.75, 1.0, 1.5, 2.0), twin_dE=4.0, frame=0.6))]
    for tag, kw in variants:
        C.P.update(kw); C.legend.cache_clear(); C.twins.cache_clear(); C.log_scale.cache_clear()
        f = score(tag)
    for k, v in sorted(f.items()):
        n = v['agree'] + v['disagree']; print(f'   {k:28s} {v["agree"]}/{n}')
