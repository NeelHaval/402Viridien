"""Visual page of what JEV saw and answered for every labelled test stretch (cache only, no new calls
unless something is uncached).

    python -X utf8 -B labeltool/jev_view.py   ->  analysis/jev_view.html (+ analysis/jev_view/*.png)
"""
import html
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2, suggest  # noqa: E402
from labeltool.common import CACHE  # noqa: E402
from labeltool.twostage import same  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'jev_view'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(suggest.cases()):
        r = jev2.decide(c['log'], c['top'], c['bottom'], jev2.ask(c['log'], c['top'], c['bottom']))
        ctx = jev2.Context.get(c['log'])
        with ctx['lock']:
            q = jev2.read_interval(ctx['io'], ctx['log'], c['top'], c['bottom'])
        qi = Image.fromarray(q)
        k = max(1, min(3, 150 // max(1, qi.width)))
        qi = qi.resize((qi.width * k, qi.height * k), Image.Resampling.NEAREST)
        qp = OUT / f'q{i:03d}.png'
        qi.save(qp)
        panel = CACHE / 'jev2' / f"{r.get('panel')}.png" if r.get('panel') else None
        rows.append(dict(c, ok=same(r.get('label'), c['truth']), label=r.get('label'), why=r.get('why', ''),
                         ranking=r.get('ranking') or [], ambiguous=r.get('ambiguous'), query=str(qp).replace('\\', '/'),
                         panel=str(panel).replace('\\', '/') if panel and panel.exists() else None))
    rows.sort(key=lambda r: (r['ok'], r['src'], r['truth']))
    n, ok = len(rows), sum(r['ok'] for r in rows)
    by = {}
    for r in rows:
        b = by.setdefault(r['src'], [0, 0]); b[0] += 1; b[1] += r['ok']
    e = html.escape
    parts = [f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>What JEV saw</title><style>
body{{font:14px/1.4 var(--font-sans,system-ui);margin:12px;color:var(--foreground,#17212b);background:var(--background,#fff)}}
h1{{font-size:20px;margin:0 0 4px}}.sum{{margin:0 0 12px;color:var(--muted-foreground,#55616b)}}
.case{{display:grid;grid-template-columns:auto minmax(0,1fr);gap:12px;border:1px solid var(--border,#c5ccd3);border-left:6px solid #215d42;border-radius:6px;padding:10px;margin:0 0 10px}}
.case.bad{{border-left-color:#9d3030}}.q img{{display:block;image-rendering:pixelated;max-height:260px}}
.panel img{{display:block;max-width:100%;height:auto;image-rendering:pixelated;border:1px solid var(--border,#c5ccd3)}}
.bar{{display:flex;align-items:center;gap:6px;font-size:13px}}.bar i{{display:block;height:10px;background:#2362a0;border-radius:2px}}
.t{{font-weight:600}}.why{{color:var(--muted-foreground,#55616b);font-size:13px}}.meta{{font-size:12px;color:var(--muted-foreground,#55616b);overflow-wrap:anywhere}}
@media(max-width:640px){{.case{{grid-template-columns:minmax(0,1fr)}}}}
</style><h1>What JEV saw and answered</h1>
<p class=sum>{ok} of {n} right. """ + ' · '.join(f"{k}: {v[1]}/{v[0]}" for k, v in by.items()) +
             ". Wrong ones first. Left: the column stretch. Right: the panel JEV was shown (the stretch beside the log's legend entries at the same scale) and its ranked answer.</p>"]
    for r in rows:
        bars = ''.join(f"<div class=bar><span style='width:150px'>{e(x['label'])}</span><i style='width:{int(200*x['probability'])}px'></i>{x['probability']:.0%}</div>" for x in r['ranking'])
        verdict = 'right' if r['ok'] else 'WRONG'
        parts.append(f"""<div class="case {'bad' if not r['ok'] else ''}"><div class=q><img src="{e(r['query'])}" alt="column stretch"></div>
<div><div class=t>{verdict}: truth {e(r['truth'])} · JEV said {e(str(r['label']))}{' · look-alike' if r['ambiguous'] else ''}</div>
<div class=why>{e(r['why'])}</div>{bars}<p class=meta>{e(r['src'])} · {e(r['log'][:60])} · rows {r['top']}-{r['bottom']}</p>
{f'<details><summary>What JEV saw</summary><div class=panel><img src="{e(r["panel"])}" alt="JEV panel"></div></details>' if r['panel'] else ''}</div></div>""")
    path = ROOT / 'analysis' / 'jev_view.html'
    path.write_text('\n'.join(parts), encoding='utf-8')
    print(path, ok, n)


if __name__ == '__main__':
    main()
