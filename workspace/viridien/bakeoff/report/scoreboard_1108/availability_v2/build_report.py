"""Render the audited availability view. No inference, truth reads, or image reads."""
from __future__ import annotations
import argparse,html,json
from collections import Counter
from pathlib import Path
from report_contract import validate_report,validate_parent_audit,sha

HERE=Path(__file__).resolve().parent
NAMES={'s19_combine':'S19 · combine','s20_v4':'V4','s21_v4dino':'V4 + DINO','s23_v4dino_identity':'DINO · distinct captions'}
SUITES={'overall_unique_log':'Overall · deduplicated','viridien_eval':'Viridien evaluation','human_qa':'Human QA','human_gold':'D-582 gold','qa_gold_only':'QA and gold only','ext_dev':'Outside DEV','hadrian_unqa':'Hadrian · not QA'}
METRICS={'score':'Row × boundary','row_acc':'Row accuracy','boundary_f1':'Boundary F1','cluster_ari':'ARI','depth_mae_ft':'Depth MAE · ft'}
SERIES={'qualified_development':'Eligible development reference','reference_diagnostic':'Reference diagnostic'}
e=lambda x:html.escape(str(x),quote=True)
def number(x):return 'Unavailable' if x is None else f'{x:.4f}'

def cell(metric):
    d=metric['denominator']
    reasons=', '.join(f'{k}: {v}' for k,v in sorted(d.get('reason_windows',{}).items()))
    return f'<td><strong>{number(metric["mean"])}</strong><small>{d["available_logs"]}/{d["total_logs"]} logs · {d["available_windows"]}/{d["total_windows"]} windows available</small><small>{d["eligible_windows"]} eligible · {d["excluded_windows"]} excluded</small>'+ (f'<small class="reason">{e(reasons)}</small>' if reasons else '')+'</td>'

def table(rows):
    out=['<div class="table-scroll" tabindex="0" role="region" aria-label="Metric table; scroll horizontally"><table><thead><tr><th scope="col">Strategy</th>']
    out += [f'<th scope="col">{e(label)}</th>' for label in METRICS.values()]
    out.append('</tr></thead><tbody>')
    for row in rows:
        out.append(f'<tr><th scope="row">{e(NAMES[row["strategy"]])}</th>')
        out += [cell(row['metrics'][m]) for m in METRICS]
        out.append('</tr>')
    return ''.join(out)+ '</tbody></table></div>'

def summary_section(data,series,policy,families=False):
    rows=[s for s in data['summaries'] if s['series']==series and s['policy']==policy and s['view']=='primary' and (s['family']!='all' if families else s['family']=='all')]
    groups={}
    for row in rows:groups.setdefault((row['suite'],row['family']),[]).append(row)
    out=[]
    for suite in SUITES:
        for (s,family),items in sorted(groups.items()):
            if s!=suite:continue
            title=SUITES[suite]+(' · '+family if families else '')
            out.append(f'<h3>{e(title)}</h3>'+table(sorted(items,key=lambda r:data['candidates'].index(r['strategy']))))
    return ''.join(out)

def paired(data,series,policy):
    rows=[r for r in data['paired'] if r['family']=='all' and r['view']=='primary' and r['series']==series and r['policy']==policy]
    out=[]
    for suite in SUITES:
        current=[r for r in rows if r['suite']==suite]
        if not current:continue
        out.append(f'<h3>{e(SUITES[suite])}</h3><div class="table-scroll" tabindex="0"><table><thead><tr><th>Difference · A minus B</th>')
        out += [f'<th>{e(m)}</th>' for m in METRICS.values()];out.append('</tr></thead><tbody>')
        for r in current:
            out.append(f'<tr><th scope="row">{e(NAMES[r["a"]])}<br>minus {e(NAMES[r["b"]])}</th>')
            for metric in METRICS:
                m=r['metrics'][metric];ci=m['ci']
                out.append(f'<td><strong>{number(ci.get("mean"))}</strong><small>95% log CI {number(ci.get("low"))} to {number(ci.get("high"))}</small><small>{ci["n_logs"]} common logs · {m["paired_windows"]} matched windows</small></td>')
            out.append('</tr>')
        out.append('</tbody></table></div>')
    return ''.join(out)

def coverage_chart(data):
    rows=[s for s in data['summaries'] if s['suite']=='overall_unique_log' and s['family']=='all' and s['strategy']==data['candidates'][0] and s['series']=='qualified_development' and s['policy']=='owner_equivalent' and s['view']=='primary']
    if not rows:return ''
    out=['<svg viewBox="0 0 420 420" role="img" aria-labelledby="coverage-title coverage-desc"><title id="coverage-title">Available development-reference logs by metric</title><desc id="coverage-desc">Teal is available. Pale orange is excluded. Each metric can have a different denominator. Exact counts follow each bar.</desc>']
    for i,(metric,label) in enumerate(METRICS.items()):
        d=rows[0]['metrics'][metric]['denominator'];y=22+i*80;total=d['total_logs'];width=280*d['available_logs']/max(total,1)
        out.append(f'<text x="12" y="{y}">{e(label)}</text><rect x="12" y="{y+12}" width="280" height="26" fill="#fde6d3"/><rect x="12" y="{y+12}" width="{width:.2f}" height="26" fill="#087f82"/><text x="304" y="{y+30}">{d["available_logs"]} / {total} logs</text>')
    return ''.join(out)+'</svg><p class="caption">Coverage shown for S19 in the deduplicated eligible development view. Depth availability also depends on matched prediction depths. The tables give all strategy denominators.</p>'

CSS='''*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#182b3a;font:16px/1.55 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:28px 20px 60px}h1{font-size:clamp(2rem,6vw,3.4rem);line-height:1.12;max-width:900px;margin:14px 0 20px}h2{font-size:1.65rem;margin:36px 0 12px}h3{font-size:1.15rem;margin:26px 0 10px}p{max-width:900px}a{color:#075ca8;text-underline-offset:3px}header{border-top:8px solid #b84e0b;padding-top:18px}.tag{font-weight:750;letter-spacing:.06em;text-transform:uppercase;font-size:.8rem;color:#92400c}.notice{border-left:5px solid #c15613;background:#fff1e5;padding:16px 20px;margin:22px 0}.notice strong{font-size:1.2rem}.links{display:flex;flex-wrap:wrap;gap:12px 22px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.card{background:white;border:1px solid #d2dde6;padding:18px;border-top:4px solid #087f82}.card strong{display:block}.table-scroll{overflow-x:auto;border:1px solid #cfdae3;background:white;margin:12px 0 24px}table{border-collapse:collapse;width:100%;min-width:950px;text-align:left}th,td{padding:12px;vertical-align:top;border-bottom:1px solid #dce4eb;width:16.6%}thead{background:#e9f0f6;color:#234b70}tbody th{color:#125d82;font-weight:700}td strong{font-variant-numeric:tabular-nums;font-size:1.13rem}small{display:block;font-size:.75rem;line-height:1.45;margin-top:5px;color:#496171}.reason{color:#8a430b}details{border:1px solid #cbd9e4;background:#fff;padding:14px;margin:18px 0}summary{font-size:1.1rem;font-weight:700;cursor:pointer}svg{width:100%;max-width:750px;height:auto;background:white;border:1px solid #d2dde6}svg text{font:14px system-ui,sans-serif;fill:#19384b}.caption{font-size:.86rem;color:#496171}.reason-table{min-width:640px}code{font-size:.8rem;overflow-wrap:anywhere}footer{margin-top:40px;border-top:2px solid #bbcdd9;padding-top:18px;font-size:.85rem}@media(max-width:480px){main{padding:20px 14px 40px}h1{font-size:2.1rem}.notice{padding:14px}.cards{grid-template-columns:1fr}th,td{padding:10px}.links{gap:10px 16px}}'''

def render(data,audit):
    truth=Counter(r['truth_status'] for r in data['records'] if r['strategy']==data['candidates'][0])
    status=''.join(f'<tr><th scope="row">{e(k)}</th><td>{v} case windows</td></tr>' for k,v in sorted(truth.items()))
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Availability v2 · lithology scoreboard correction</title><style>{CSS}</style><main><header><span class="tag">Versioned correction · local proxy</span><h1>Show what the references can support.</h1><p>The prior strategy default is withdrawn. Denominator changes correct the metric; they do not show a model gain. This view separates available reference diagnostics from eligible development references. It names no replacement champion.</p></header><div class="notice"><strong>No selection is promoted.</strong><p>Audit success confirms this derived report and its recorded inputs. It does not certify independent human truth, unseen wells, official challenge grading, or the new 1–2 minute latency goal.</p></div><nav class="links" aria-label="Evidence"><a href="derived_scores.json">All derived numbers</a><a href="audit.json">Bound audit</a><a href="parent_audit.json">Parent audit</a><a href="../../../strategies/SCORING_CONTRACT_V2.md">Scoring contract v2</a><a href="../index_baseline_v1.html">Preserved v1 · withdrawn claim</a><a href="../baseline_v1_preservation_receipt.json">Preservation receipt</a></nav><h2>Two views. Explicit limits.</h2><div class="cards"><div class="card"><strong>Eligible development reference</strong><p>All unreviewed silver lithology, ARI, and product values are excluded. Hadrian without QA is excluded. QA, gold, and supplied DEV remain development references with exposure limits.</p></div><div class="card"><strong>Reference diagnostic</strong><p>Valid unreviewed silver remains a source-A diagnostic. The two vocabulary-invalid cases and empty Neptune cannot supply lithology accuracy or product. This view is not independent truth.</p></div><div class="card"><strong>Contacts and depth</strong><p>Silver boundary F1 is a source-A contact diagnostic in both views. Shenzi contacts may follow machine boundaries. Missing depth truth or unmatched prediction depth is unavailable, not zero error.</p></div></div><h2>Availability comes before a mean</h2>{coverage_chart(data)}<p>Each cell gives available logs and windows, eligible windows, and exclusions. Available plus excluded equals the total. Eligible counts can exceed available counts when prediction depth is missing. Metric denominators can differ within one row.</p><h2>Eligible development reference · owner equivalence</h2><p>Clay, Shale, Clay/Shale, and Clay-Shale receive symmetric label credit. This is the owner-configured local policy. Raw labels and the shared scorer are unchanged.</p>{summary_section(data,'qualified_development','owner_equivalent')}<details><summary>Strict labels · eligible development reference</summary>{summary_section(data,'qualified_development','strict')}</details><details><summary>Reference diagnostic · owner equivalence</summary><p>These values include valid source-A silver labels. They must not be called independent or human-reviewed accuracy.</p>{summary_section(data,'reference_diagnostic','owner_equivalent')}</details><details><summary>Reference diagnostic · strict labels</summary>{summary_section(data,'reference_diagnostic','strict')}</details><h2>Paired differences · eligible development reference</h2><p>A minus B, on exact common available cases, then common log means. The confidence interval uses the recorded 10,000-draw log bootstrap. A one-log interval cannot establish transfer. Repeated pages and five provisional DEV source groups limit independence; these are not held-out well claims.</p>{paired(data,'qualified_development','owner_equivalent')}<details><summary>Paired differences · strict labels</summary>{paired(data,'qualified_development','strict')}</details><details><summary>Paired differences · reference diagnostic</summary>{paired(data,'reference_diagnostic','owner_equivalent')}</details><details><summary>Family breakdown · eligible development reference</summary>{summary_section(data,'qualified_development','owner_equivalent',True)}</details><details><summary>Family breakdown · reference diagnostic</summary>{summary_section(data,'reference_diagnostic','owner_equivalent',True)}</details><h2>Reference status and exclusions</h2><div class="table-scroll"><table class="reason-table"><thead><tr><th>Status</th><th>Case windows · each strategy</th></tr></thead><tbody>{status}</tbody></table></div><p>Silver labels are unreviewed-two-source-A-tie-break. The saved raster followed source A when sources differed. Empty Neptune supplies no filled truth rows. AAllegheny and ARio have legend-vocabulary conflicts. A valid contact metric does not make those lithology labels valid.</p><h2>What this report cannot decide</h2><p>The product is row accuracy × boundary F1 for each available window. Window products are averaged within each log, then logs are averaged. It is not the product of the displayed component means. Row accuracy ignores blank truth rows. ARI, depth, runtime, and memory are separate; they do not enter this local product.</p><p>Overall uses the recorded deduplication and QA → gold → evaluation → DEV precedence. DEV is supplied development truth, not certified human truth. Evaluation exposure and shared source groups block an unseen-well generalization claim. All DEV native v4 legends were absent in this frozen run, so its coverage failures cannot isolate matcher quality.</p><p>Archived runtime mixes local source-page calls and remote predecoded crops. It does not establish an efficiency ranking. The formal S19 Shenzi whole-attempt timeout remains a supervised prediction failure where the metric is eligible. Its inference runtime is unknown.</p><p>The Astra review and recheck covered the earlier code and development method. These new availability masks address later owner findings. No new Astra endorsement, official rule certification, or new champion is claimed.</p><h2>Images under the owner hold</h2><p>Status: <code>not-generated_due_to_owner_hold</code>. No new protected truth overlays were generated or inspected for v2. Historical v1 images remain in the preserved archive. Vocabulary-invalid truth images there are historical diagnostics, not approved target labels. This is an owner hold, not a render failure.</p><footer><p>Bound data SHA-256: <code>{e(audit['derived_sha256'])}</code>. Complete record coverage is distinct from valid metric coverage and submission readiness. This report reads cached results and review metadata only.</p><p><a href="metadata.json">Derivation metadata</a> · <a href="README.md">Availability method</a> · <a href="../CHALLENGE_ALIGNMENT.md">Challenge alignment · historical development evidence</a> · <a href="../readiness_preview.json">Submission-readiness diagnostic · not a score</a></p></footer></main></html>'''
    return content

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=HERE/'derived_scores.json');p.add_argument('--audit',type=Path,default=HERE/'audit.json');p.add_argument('--output',type=Path,default=HERE/'index.html');args=p.parse_args()
    data,audit=validate_report(args.data,args.audit,HERE.parent)
    validate_parent_audit(HERE/'parent_audit.json',args.data,len(data['records']))
    args.output.write_text(render(data,audit),encoding='utf-8')
    receipt=dict(version='availability_v2',data_sha256=sha(args.data),audit_sha256=sha(args.audit),parent_audit_sha256=sha(HERE/'parent_audit.json'),html_sha256=sha(args.output),selection_promoted=False,generalization_verified=False,image_status='not-generated_due_to_owner_hold',generated_truth_images=0,image_render_failures=0,records=len(data['records']))
    (HERE/'report_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Availability report built: {len(data["records"])} records; no selection promoted; no new truth images.')
if __name__=='__main__':main()
