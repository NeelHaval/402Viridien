import json, sys
from collections import Counter
sys.path.insert(0, 'bakeoff/notes/council/flywheel')
import importlib.util
spec = importlib.util.spec_from_file_location('h', 'bakeoff/notes/council/flywheel/human_vs_v4.py')
src = open('bakeoff/notes/council/flywheel/human_vs_v4.py', encoding='utf-8').read().split("H = human_sets()")[0]
g = {'__file__': 'bakeoff/notes/council/flywheel/human_vs_v4.py'}; exec(src, g)
H = g['human_sets'](); rows_of = g['rows_of']; match = g['match']; key = g['key']; B = g['B']
beds = {k: json.loads((B / 'beds' / f'{k}.json').read_text(encoding='utf-8')) for k in H}
def v4_at(k, y):
    for b in beds[k]['beds']:
        if b['top'] <= y < b['bottom']:
            return b['label'] if b['state'] == 'ok' else None
R = [r for r in json.loads((B / 'llmrev' / 'results.json').read_text(encoding='utf-8'))['rows'] if not r.get('dup') and r['llm']]
for k in H:
    S = [r for r in R if key(r['log']) == k]
    lo = min(t for t, _, _ in H[k]); hi = max(b for _, b, _ in H[k])
    print('==', k[:50], 'blind windows', len(S), 'inside human range', sum(lo <= r['y'] and r['y'] + r['h'] <= hi for r in S))
    for r in S:
        hum = Counter(x for x in rows_of(H[k], r['y'], r['y'] + r['h']) if x is not None)
        hl = hum.most_common(1)[0] if hum else None
        v4 = v4_at(k, r['y'] + r['h'] // 2)
        print(f"   y={r['y']:6d} band={r['band'][:9]:9s} v3={str(r['label'])[:18]:18s} v4={str(v4)[:18]:18s} llm={str(r['llm'])[:18]:18s} sure={r['sure']:6s} human={hl}")
