"""Show the held, receipt-backed JEV comparison. No inference or truth reads."""
from pathlib import Path
import hashlib
import html
import json

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'scores.json').read_text(encoding='utf-8'))
audit=json.loads((HERE/'parent_checkpoint_audit.json').read_text(encoding='utf-8'))
assert not audit['errors'] and audit['scores_sha256']==hashlib.sha256((HERE/'scores.json').read_bytes()).hexdigest()
assert data['paired_cases']==29 and not data['complete'] and not data['selection_promoted']
names={'s19_combine':'s19','s20_v4':'v4','s21_v4dino':'v4 + DINO','s23_v4dino_identity':'DINO, separate captions','s30_jev_sourcecards':'Optimized JEV'}
def rows(suite):
    return [r for r in data['summaries'] if r['suite']==suite and r['family']=='all' and r['view']=='primary' and r['series']=='qualified_development' and r['policy']=='owner_equivalent']
qa=rows('qa_gold_only');dev=rows('ext_dev')
pair=next(r['metrics']['score']['ci'] for r in data['paired'] if r['suite']=='qa_gold_only' and r['family']=='all' and r['view']=='primary' and r['series']=='qualified_development' and r['policy']=='owner_equivalent' and r['a']=='s21_v4dino' and r['b']=='s30_jev_sourcecards')
style='''<style>*{box-sizing:border-box}html,body{margin:0;font-family:var(--font-sans,Arial,sans-serif);color:var(--foreground,#21323b)}main{max-width:760px;margin:auto}h1{font-size:24px;line-height:1.2}h2{font-size:19px;margin:0 0 10px}p{font-size:14px;line-height:1.5}section{padding:16px;border:1px solid var(--border,#c7d2d9);border-radius:var(--radius,8px);background:var(--card,#f6f9fb);margin:16px 0}.chart{height:274px}.row{height:53px}.label{display:flex;justify-content:space-between;gap:8px;font-size:14px;margin:8px 0}.track{height:13px;background:var(--secondary,#dfe7eb)}.bar{height:13px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:9px 3px;text-align:right;border-bottom:1px solid var(--border,#c7d2d9)}td:first-child,th:first-child{text-align:left}.muted{color:var(--muted-foreground,#697a83)}a{color:var(--primary,#176ea5)}.foot{font-size:11px;overflow-wrap:anywhere}</style>'''
parts=['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',style,'</head><body><main><h1>Optimized JEV · held checkpoint</h1><p><b>29 of 68 cases</b> have receipt-backed five-method comparisons. There are 30 saved cases; one stays unpromoted. New API calls are held. This is a partial development comparison with the corrected availability rules.</p><section><h2>QA and gold · 3 logs</h2><p>Approved Clay/Shale product. The same three logs are available for every method.</p><div class="chart">']
colors=['var(--chart-1,#1577b8)','var(--chart-2,#168675)','var(--chart-3,#bc631f)','var(--chart-4,#91529c)','var(--chart-6,#468d9f)']
for r,color in zip(qa,colors):
    value=r['metrics']['score']['mean']
    assert r['metrics']['score']['denominator']['available_logs']==3
    parts.append('<div class="row"><div class="label"><span>'+html.escape(names[r['strategy']])+'</span><b>'+f'{value:.3f}'+'</b></div><div class="track"><div class="bar" style="width:'+str(value*100)+'%;background:'+color+'"></div></div></div>')
parts+=['</div><table><thead><tr><th>Method</th><th>Row accuracy</th><th>Boundary F1</th></tr></thead><tbody>']
for r in qa:
    parts.append('<tr><td>'+html.escape(names[r['strategy']])+'</td><td>'+f'{r["metrics"]["row_acc"]["mean"]:.3f}'+'</td><td>'+f'{r["metrics"]["boundary_f1"]["mean"]:.3f}'+'</td></tr>')
parts+=['</tbody></table><p>The paired DINO minus JEV gap is '+f'{pair["mean"]:.3f}'+', with a 95% log-bootstrap interval ['+f'{pair["low"]:.3f}, {pair["high"]:.3f}'+']. The interval includes zero. These three logs cannot establish a broad winner.</p><p class="muted">Shenzi contacts may use machine boundary anchors. The s19 Shenzi timeout stays in the denominator as zero. These are means of per-log products, not the product of the two means in the table.</p></section><section><h2>Current DEV subset · 11 logs</h2><p>There are 12 matched windows. s19 has product '+f'{next(r for r in dev if r["strategy"]=="s19_combine")["metrics"]["score"]["mean"]:.3f}'+'. JEV has '+f'{next(r for r in dev if r["strategy"]=="s30_jev_sourcecards")["metrics"]["score"]["mean"]:.3f}'+'. The frozen v4 pipelines have zero label credit because their native legend inputs are missing.</p><p>JEV uses a shared machine-legend fallback. The legend input paths differ, so this is a pipeline comparison. It does not isolate the visual matcher.</p></section><p>No model, label or truth was changed for this rescore. The full run remains incomplete under the owner hold. Both strict and approved-equivalence series are in the <a href="report.html">checkpoint report</a>.</p><p class="foot muted">Checkpoint data SHA-256: '+audit['scores_sha256']+'</p></main></body></html>']
(HERE/'overview.html').write_text(''.join(parts),encoding='utf-8')
(HERE/'overview_data.json').write_text(json.dumps(dict(scores_sha256=audit['scores_sha256'],qa_gold=qa,DEV=dev,paired_gap=pair),indent=2)+'\n',encoding='utf-8')
print('JEV_OVERVIEW:29 receipt-backed cases; 3 QA/gold logs; common masks; no launch or champion')
