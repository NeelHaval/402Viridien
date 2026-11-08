"""Show audited metric availability. No truth or image reads."""
from pathlib import Path
import hashlib
import html
import json

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/'derived_scores.json').read_text(encoding='utf-8'))
audit = json.loads((HERE/'parent_audit.json').read_text(encoding='utf-8'))
assert not audit['errors'] and audit['derived_sha256'] == hashlib.sha256((HERE/'derived_scores.json').read_bytes()).hexdigest()
old = json.loads((HERE.parent/'scores.json').read_text(encoding='utf-8'))
names = {'s19_combine':'s19','s20_v4':'v4','s21_v4dino':'v4 + DINO','s23_v4dino_identity':'DINO, separate captions'}
def summary(suite, strategy, series='qualified_development'):
    matches = [r for r in data['summaries'] if r['suite']==suite and r['strategy']==strategy and r['family']=='all' and r['view']=='primary' and r['series']==series and r['policy']=='owner_equivalent']
    assert len(matches)==1
    return matches[0]
def number(v):return 'Unavailable' if v is None else f'{v:.3f}'
values=[]
for strategy in data['candidates']:
    baseline=next(r for r in old['accepted_summaries'] if r['suite']=='viridien_eval' and r['family']=='all' and r['strategy']==strategy)
    values.append(dict(strategy=strategy,old_eval=baseline['metrics']['score'],
        reference_eval=summary('viridien_eval',strategy,'reference_diagnostic'),
        qualified_eval=summary('viridien_eval',strategy),
        qualified_overall=summary('overall_unique_log',strategy)))
(HERE/'overview_data.json').write_text(json.dumps(dict(derived_sha256=audit['derived_sha256'],selection_promoted=False,values=values),indent=2)+'\n',encoding='utf-8')
style='''<style>*{box-sizing:border-box}html,body{margin:0;color:var(--foreground,#20313b);font-family:var(--font-sans,Arial,sans-serif)}main{max-width:760px;margin:auto}h1{font-size:24px;line-height:1.2}h2{font-size:19px;margin:0 0 10px}p,li{font-size:14px;line-height:1.5}.muted{color:var(--muted-foreground,#687881)}section{padding:16px;border:1px solid var(--border,#c7d3db);border-radius:var(--radius,8px);background:var(--card,#f6f9fb);margin:16px 0}.notice{border-left:5px solid var(--warning,#b66b1f)}.cells{display:flex;gap:4px;height:26px;margin:16px 0}.cell{flex:1;border:1px solid var(--border,#a8b8c1)}.gold{background:var(--chart-2,#168675)}.silver{background:var(--secondary,#d4dfe5)}.invalid{background:var(--warning,#c98d36)}.blank{background:var(--background,#fff)}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:10px 3px;text-align:right;border-bottom:1px solid var(--border,#c7d3db)}th:first-child,td:first-child{text-align:left}th{font-weight:600}a{color:var(--primary,#166ea5)}.barrow{height:54px}.barlabel{font-size:14px;display:flex;justify-content:space-between;gap:10px;margin:8px 0}.track{height:14px;background:var(--secondary,#dfe7eb)}.bar{height:14px}.chart{height:225px}.foot{font-size:11px;overflow-wrap:anywhere}</style>'''
style += '<style>.silver{background:var(--chart-5,#93a8b4);opacity:.45}</style>'
parts=['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',style,'</head><body><main><h1>Availability V2</h1><section class="notice"><h2>The earlier choice is withdrawn</h2><p>The old score gave full row credit to an empty truth window. Some silver truth also has invalid legend names. These results correct availability. They do not promote a champion or change a prediction.</p></section><section><h2>What the 13-log set can measure</h2><div class="cells" aria-label="One gold log, nine unreviewed reference logs, two invalid vocabulary logs, one blank diagnostic"><span class="cell gold"></span>']
parts += ['<span class="cell silver"></span>']*9
parts += ['<span class="cell invalid"></span>']*2
parts += ['<span class="cell blank"></span></div><p><b>1</b> gold log can supply qualified label scores.<br><b>9</b> unreviewed source-A logs can supply descriptive label scores.<br><b>2</b> logs have invalid truth vocabulary.<br><b>1</b> log has no labelled truth rows.</p><p class="muted">All 12 silver rasters use source A when A and B tie. They are not AI consensus. Their contact measures remain source-A diagnostics. The 12 silver truth files have no depth fields.</p></section><section><h2>Same predictions, different denominators</h2><p>Approved Clay/Shale product. Each column below has its own eligible log set. A larger value is not a model improvement.</p><table><thead><tr><th>Pipeline</th><th>Old V1<br>13 logs</th><th>Reference V2<br>10 logs</th><th>Qualified V2<br>1 log</th></tr></thead><tbody>']
for row in values:
    parts.append('<tr><td>'+html.escape(names[row['strategy']])+'</td><td>'+number(row['old_eval'])+'</td><td>'+number(row['reference_eval']['metrics']['score']['mean'])+'</td><td>'+number(row['qualified_eval']['metrics']['score']['mean'])+'</td></tr>')
parts += ['</tbody></table><p>Neptune is unavailable in V2. It receives neither zero nor a perfect score. Failed predictions on valid truth still count as zero.</p></section><section><h2>Eligible development reference · 49 logs</h2><p>Overlap is counted once. Hadrian is separate. The four pipelines use the same available cases here.</p><div class="chart">']
colors=['var(--chart-1,#1577b8)','var(--chart-2,#168675)','var(--chart-3,#bc631f)','var(--chart-4,#91529c)']
for row,color in zip(values,colors):
    metric=row['qualified_overall']['metrics']['score'];value=metric['mean']
    assert metric['denominator']['available_logs']==49
    parts.append('<div class="barrow"><div class="barlabel"><span>'+html.escape(names[row['strategy']])+'</span><b>'+number(value)+'</b></div><div class="track"><div class="bar" style="width:'+str(value*100)+'%;background:'+color+'"></div></div></div>')
parts += ['</div><p class="muted">Supplied DEV truth is a development reference. The frozen v4 pipelines lack native legends on all 46 DEV logs. These bars do not isolate matcher quality. Known well exposure prevents a blind-test claim.</p></section><p>The full report keeps strict and approved-equivalence scores, all metric denominators, exclusion reasons and paired intervals. A missing depth error stays unavailable.</p><p><a href="index.html">Corrected report</a> · <a href="derived_scores.json">Every derived number</a> · <a href="../index_baseline_v1.html">Preserved V1 report</a></p><p class="foot muted">Derived data SHA-256: '+audit['derived_sha256']+'</p></main></body></html>']
(HERE/'overview.html').write_text(''.join(parts),encoding='utf-8')
print('AVAILABILITY_OVERVIEW: final audited bytes; 13-log coverage; corrected means; no champion')
