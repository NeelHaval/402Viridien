"""Make a compact visual from the final audited score records."""
from pathlib import Path
import hashlib
import html
import json

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'scores.json').read_text(encoding='utf-8'))
records_digest = hashlib.sha256(json.dumps(data['records'], sort_keys=True,
    separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
for name in ('audit_saved_predictions.json', 'result_audit.json'):
    audit = json.loads((HERE / name).read_text(encoding='utf-8'))
    if audit['errors'] or audit['records_sha256'] != records_digest:
        raise ValueError('The final audit does not match the score records')
if not data['complete'] or len(data['records']) != 272:
    raise ValueError('The final matrix is incomplete')
names = {'s19_combine': 's19', 's20_v4': 'v4', 's21_v4dino': 'v4 + DINO',
         's23_v4dino_identity': 'DINO, separate captions'}
colors = ['var(--chart-1,#1577b8)', 'var(--chart-2,#168675)',
          'var(--chart-3,#bc631f)', 'var(--chart-4,#91529c)']
def cohort(suite, title, detail):
    rows = [r for r in data['accepted_summaries'] if r['suite'] == suite and r['family'] == 'all']
    parts = [f'<section><h2>{title}</h2><p class="muted">{detail}</p><div class="chart">']
    for row, color in zip(rows, colors):
        value = row['metrics']['score']
        parts.append(f'<div class="barrow"><div class="barlabel"><span>{names[row["strategy"]]}</span><b>{value:.3f}</b></div><div class="track"><div class="bar" style="width:{100*value:.6f}%;background:{color}"></div></div></div>')
    parts.append('</div><table><thead><tr><th>Pipeline</th><th>Row accuracy</th><th>Boundary F1</th></tr></thead><tbody>')
    for row in rows:
        m = row['metrics']
        parts.append(f'<tr><td>{names[row["strategy"]]}</td><td>{m["row_acc"]:.3f}</td><td>{m["boundary_f1"]:.3f}</td></tr>')
    parts.append('</tbody></table></section>')
    return ''.join(parts)
style = '''<style>:root{color-scheme:light dark}*{box-sizing:border-box}html,body{margin:0;font-family:var(--font-sans,Arial,sans-serif);color:var(--foreground,#19272d)}main{max-width:760px;margin:auto}h1{font-size:24px;line-height:1.2;margin:12px 0}h2{font-size:19px;margin:0 0 8px}p{font-size:14px;line-height:1.5}.muted{color:var(--muted-foreground,#64737b)}section{padding:16px;margin:16px 0;border:1px solid var(--border,#cbd6dc);border-radius:var(--radius,8px);background:var(--card,#f5f8fa)}.chart{height:218px}.barrow{height:52px}.barlabel{display:flex;justify-content:space-between;gap:8px;font-size:14px;margin:6px 0}.track{height:13px;background:var(--secondary,#dfe7ec)}.bar{height:13px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 4px;border-bottom:1px solid var(--border,#cbd6dc);text-align:right}th:first-child,td:first-child{text-align:left}th{font-weight:600}img{display:block;width:100%;height:auto}a{color:var(--primary,#1476b4)}figcaption{font-size:13px;line-height:1.5;margin-top:8px}figure{margin:12px 0}.foot{font-size:11px;overflow-wrap:anywhere}</style>'''
parts = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">', style, '</head><body><main><h1>One grade, two useful views</h1><p>All 272 case results are present. The charts use the approved Clay/Shale family rule. Each score is row accuracy × boundary F1, averaged per log. This is a local measure, not a confirmed Viridien grade.</p>']
parts.append(cohort('overall_unique_log', 'All development data · 60 logs',
    's19 is the practical default for this input set. Hadrian is separate. Overlapping sources are counted once.'))
parts.append('<p>The frozen v4 pipelines lack native legend inputs on all 51 outside DEV windows, across 46 logs. Their zero label scores are a coverage failure. This comparison does not isolate the DINO matcher.</p>')
parts.append(cohort('viridien_eval', 'Viridien set · 13 logs',
    'Separate-caption DINO has the highest mean here. Its paired interval against baseline DINO includes zero; a broad gain is not established.'))
parts.append('<section><h2>Read the label error beside the source</h2><p>This is one saved baseline DINO prediction on D-582 log06. The raw names and contacts are kept.</p><figure><a href="images/log06_baseline_accepted_4700_5150.png"><img src="images/log06_baseline_accepted_4700_5150.png" alt="Real D-582 column with truth and DINO prediction intervals and an equivalence error strip"></a><figcaption>The detail and the full-window overview use the same saved prediction. Open the image for full-size labels.</figcaption></figure><table><thead><tr><th>Truth</th><th>Prediction</th><th>Label credit</th></tr></thead><tbody><tr><td>Shale</td><td>Clay / Shale</td><td>Full · green</td></tr><tr><td>Siltstone</td><td>Sand and Sandstone</td><td>Zero · red</td></tr></tbody></table><p>The full report has all 12 worst-log images and the strict score series.</p></section>')
parts.append('<p>Correct labels can still have a low product when contacts are missing or extra. For example, 0.90 row accuracy × 0.60 boundary F1 = 0.54. The Clay/Shale rule changes label credit; it keeps the boundary test.</p><p><a href="index.html">Open the full scoreboard</a> · <a href="scores.json">All numbers</a> · <a href="CHALLENGE_ALIGNMENT.md">Challenge criteria and limits</a></p>')
parts.append('<p class="foot muted">Audited record fingerprint: '+html.escape(records_digest)+'</p></main></body></html>')
(HERE / 'final_overview.html').write_text(''.join(parts), encoding='utf-8')
print('FINAL_OVERVIEW: 272 audited records; two cohorts; actual saved comparison image')
