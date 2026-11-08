"""Re-run the classifier on windows v3 rejected as mixed (after the dark-edge fix), in place."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import legend_classify as C, jev2  # noqa: E402
from harness import io  # noqa: E402
D = Path(__file__).resolve().parents[1] / 'analysis' / 'categories' / 'classified'
names = {n.replace('#', '_').replace(' ', '_')[:90]: n for n in io.list_logs()}
tot = Counter()
for f in sorted(D.glob('*.json')):
    log = names[f.stem]; out = json.loads(f.read_text(encoding='utf-8'))
    ys = [o for o in out if o['state'] == 'mixed']
    if not ys:
        continue
    ctx = jev2.Context.get(log); y0 = out[0]['y']
    with ctx['lock']:
        col = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=y0, y1=out[-1]['y'] + out[-1]['h'])[..., :3]
    for o in ys:
        r = C.classify(col[o['y'] - y0:o['y'] - y0 + o['h']], log)
        o.update(state=r['state'], label=r.get('label'), second=r.get('second'), margin=r.get('margin'))
        tot[r['state']] += 1
    f.write_text(json.dumps(out), encoding='utf-8')
    print(f.stem[:40], Counter(o['state'] for o in ys), flush=True)
print('re-checked mixed ->', tot)
