import json, base64, sys
d = json.load(open('data.json'))
open('col.png','wb').write(base64.b64decode(d['col_png'])); open('zoom.png','wb').write(base64.b64decode(d['zoom_png']))
H = 620; D0, D1 = d['D0'], d['D1']; k = H / (D1 - D0)
y = lambda dep: (dep - D0) * k
col = {'Shale': 'var(--chart-1)', 'Siltstone': 'var(--chart-3)', 'Sandstone': 'var(--chart-4)'}
X = dict(axis=44, scan=54, truth=150, pred=250, err=350); W = 80
s = []
for dep in range(int(D0), int(D1) + 1, 50):
    s.append(f'<text x="{X["axis"]}" y="{y(dep)+4:.1f}" class="t" text-anchor="end">{dep}</text><line x1="{X["axis"]+3}" x2="{X["scan"]}" y1="{y(dep):.1f}" y2="{y(dep):.1f}" class="g"/>')
s.append(f'<image href="{sys.argv[1]}/col.png" x="{X["scan"]}" y="0" width="60" height="{H}" preserveAspectRatio="none"/>')
def band(ivs, x):
    for r in ivs:
        s.append(f'<rect x="{x}" y="{y(r["top_depth"]):.1f}" width="{W}" height="{y(r["bottom_depth"])-y(r["top_depth"]):.1f}" fill="{col[r["lithology"]]}" stroke="var(--background)" stroke-width="1"/>')
band(d['truth'], X['truth']); band(d['pred'], X['pred'])
def at(ivs, dep):
    for r in ivs:
        if r['top_depth'] <= dep < r['bottom_depth']: return r['lithology']
dep = D0; st = None
while dep <= D1:
    bad = dep < D1 and at(d['truth'], dep) != at(d['pred'], dep)
    if bad and st is None: st = dep
    if not bad and st is not None:
        s.append(f'<rect x="{X["err"]}" y="{y(st):.1f}" width="14" height="{max(y(dep)-y(st),1.5):.1f}" fill="var(--destructive)"/>'); st = None
    dep = round(dep + 0.05, 2)
notes = [(25420.5, 'boundary 1 ft too low'), (25489, 'boundary 2 ft too high'), (25545, 'missed 10 ft siltstone bed'), (25635, 'sandstone called siltstone')]
for dep, t in notes:
    s.append(f'<line x1="{X["err"]+18}" x2="{X["err"]+30}" y1="{y(dep):.1f}" y2="{y(dep):.1f}" class="g"/><text x="{X["err"]+34}" y="{y(dep)+4:.1f}" class="t">{t}</text>')
for x, t in [(X['scan'], 'scan'), (X['truth'], 'their answer'), (X['pred'], 'our output'), (X['err'], 'wrong')]:
    s.append(f'<text x="{x}" y="-10" class="h">{t}</text>')
svg = f'<svg viewBox="0 -24 560 {H+30}" width="100%" style="max-width:560px">{"".join(s)}</svg>'
f = {r['tol']: r for r in d['f1s']}
rows = [
 ('Label accuracy by depth', 'Share of the column (each 0.05 ft row) where our rock name matches theirs', f"{d['acc']*100:.1f}%", '23 of 300 ft wrong'),
 ('Boundary F1, within ±0.5 ft (10 px)', 'A boundary counts if it lands within the tolerance of one of theirs', f"{f[0.5]['f1']:.2f}", f"{f[0.5]['tp']} of {f[0.5]['n_true']} found"),
 ('Boundary F1, within ±1 ft (20 px)', '', f"{f[1.0]['f1']:.2f}", f"{f[1.0]['tp']} of {f[1.0]['n_true']} found"),
 ('Boundary F1, within ±2 ft (40 px)', '', f"{f[2.0]['f1']:.2f}", f"{f[2.0]['tp']} of {f[2.0]['n_true']} found"),
 ('Cluster agreement (ARI)', 'Do we group the same beds together? Ignores what we call the groups', f"{d['ari']:.2f}", '1.00 = same grouping'),
]
tr = ''.join(f'<tr><td><b>{a}</b><div class="m">{b}</div></td><td class="v">{c}</td><td class="m">{e}</td></tr>' for a, b, c, e in rows)
res = [(0.05, 8), (0.10, 19), (0.13, 7), (0.25, 2), (0.33, 2), (0.42, 1), (0.50, 10), (0.67, 2)]
bars = ''.join(f'<div class="br"><span class="bl">{v:.2f} ft</span><span class="bb" style="width:{n*14}px{";background:var(--chart-2)" if v==0.05 else ""}"></span><span class="m">{n} logs</span></div>' for v, n in res)
html = f'''<!doctype html><html><head><style>
body{{margin:0;font-family:var(--font-sans);color:var(--foreground)}}
h3{{font-size:15px;margin:22px 0 6px}} p{{margin:4px 0 10px;font-size:13px;line-height:1.45}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}} .tile{{background:var(--card);border-radius:var(--radius);padding:12px 14px}}
.tile b{{font-size:20px;display:block}} .tile span{{font-size:12px;color:var(--muted-foreground)}}
.t{{font-size:11px;fill:var(--muted-foreground)}} .h{{font-size:12px;font-weight:600;fill:var(--foreground)}} .g{{stroke:var(--border)}}
.key{{display:flex;gap:14px;font-size:12px;margin:6px 0}} .key i{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td{{padding:8px 6px;border-top:1px solid var(--border);vertical-align:top}}
.v{{font-size:17px;font-weight:600;white-space:nowrap;font-variant-numeric:tabular-nums}} .m{{font-size:12px;color:var(--muted-foreground)}}
.zoom{{width:100%;border-radius:var(--radius);display:block}}
.br{{display:flex;align-items:center;gap:8px;margin:3px 0}} .bl{{width:56px;font-size:12px;text-align:right;font-variant-numeric:tabular-nums}} .bb{{height:12px;background:var(--chart-1);border-radius:2px}}
</style></head><body>
<h3>How fine is the example answer?</h3>
<div class="tiles">
<div class="tile"><b>20 px</b><span>per foot of depth (400 dpi scan, 5 in per 100 ft)</span></div>
<div class="tile"><b>0.05 ft</b><span>per pixel row, about 1.5 cm</span></div>
<div class="tile"><b>10 ft</b><span>every boundary in example.csv is a whole multiple of 10 ft</span></div>
<div class="tile"><b>17 beds</b><span>over 300 ft (25,400 to 25,700 ft); thinnest 10 ft = 200 px</span></div>
</div>
<p>So their answer is drawn to the nearest 10 ft, but each boundary is placed to the pixel: the red lines below are the CSV's <code>top_y</code> values on the real scan, and they sit on the drawn rock change within a few pixels.</p>
<img class="zoom" src="{sys.argv[1]}/zoom.png">
<p class="m">Example log at full resolution, 25,490 to 25,525 ft. Red = boundaries from example.csv. Right-hand column is the lithology column; left is the depth column.</p>
<h3>How they can score us: our CSV against theirs</h3>
<p>For the held-out logs they keep a CSV like example.csv. Ours is lined up with theirs by depth (or pixel row) and compared. Below, a made-up output with four mistakes is scored against the real example.</p>
<div class="key"><span><i style="background:var(--chart-1)"></i>Shale</span><span><i style="background:var(--chart-3)"></i>Siltstone</span><span><i style="background:var(--chart-4)"></i>Sandstone</span><span><i style="background:var(--destructive)"></i>we disagree</span></div>
{svg}
<table>{tr}</table>
<h3>The same 1 px error means different things on different logs</h3>
<p>Feet per pixel across the 51 logs in the dataset. The example is one of the finest (highlighted). A ±1 ft tolerance is 20 px of slack on the example but only 1.5 px on the coarsest logs, so the same tolerance is much harder there.</p>
{bars}
</body></html>'''
open('page.html', 'w', encoding='utf8').write(html); print(len(html))
