"""Publish an explicit partial scoreboard while the complete audit is pending."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import html
import json

R=Path(__file__).resolve().parent
NAMES={'s19_combine':'s19','s20_v4':'v4','s21_v4dino':'v4 + DINO baseline','s23_v4dino_identity':'v4 + DINO, distinct labels'}
FIELDS=('score','row_acc','boundary_f1','cluster_ari','seconds')
IDENTITY=('log','suite','window','image_sha256','annotation_sha256','truth_sha256','truth_sources')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish-index',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((R/'manifest_1108.json').read_text(encoding='utf-8'))
    data=json.loads((R/'raw_results.json').read_text(encoding='utf-8'))
    candidates=manifest['candidates'];cases={c['case_id']:c for c in manifest['cases']}
    grouped=defaultdict(dict)
    for row in data['records']:
        case=cases.get(row['case_id'])
        if case is None or row['strategy'] not in candidates:
            raise ValueError('Unexpected case or candidate')
        if row['strategy'] in grouped[row['case_id']]:
            raise ValueError('Duplicate checkpoint key')
        if any(row.get(k)!=case.get(k) for k in IDENTITY):
            raise ValueError('Record does not match manifest')
        grouped[row['case_id']][row['strategy']]=row
    matched={k:v for k,v in grouped.items() if set(v)==set(candidates)}
    selected=[r for case in matched.values() for r in case.values()]
    summary=[]
    for suite in sorted({r['suite'] for r in selected}):
        for strategy in candidates:
            rr=[r for r in selected if r['suite']==suite and r['strategy']==strategy]
            logs=defaultdict(list)
            for r in rr:logs[r['log']].append(r)
            metrics={}
            for f in FIELDS:
                means=[]
                for group in logs.values():
                    values=[r[f] for r in group if r.get(f) is not None]
                    if values:means.append(sum(values)/len(values))
                metrics[f]=sum(means)/len(means) if means else None
            summary.append(dict(suite=suite,strategy=strategy,n_logs=len(logs),n_cases=len(rr),
                                failures=sum(bool(r.get('error')) for r in rr),**metrics))
    generated=datetime.now(timezone.utc).isoformat()
    report=dict(status='INCOMPLETE — no champion',generated_utc=generated,
                matched_cases=len(matched),expected_cases=len(cases),saved_records=len(data['records']),
                expected_records=len(cases)*len(candidates),summaries=summary,records=selected,
                note='Only exact cases with all four candidates are shown. Final CSV replay, complete suite statistics, paired intervals and images are pending. No selection is permitted from this page.')
    (R/'progress_scores.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    css='body{margin:0;color:#183142;background:white;font:16px/1.5 Arial,sans-serif}main{max-width:1100px;margin:auto;padding:24px 16px}h1{font-size:32px;line-height:1.15}.notice{background:#fff0d9;padding:16px;border:1px solid #bd853e}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:10px;text-align:right;border-bottom:1px solid #cad6dd;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e7f0f5}a{color:#075c98}h2{margin-top:30px}.note{max-width:80ch;color:#425d6c}.bars{max-width:760px;margin:16px 0}.barrow{display:grid;grid-template-columns:155px 1fr 48px;gap:10px;margin:9px 0;font-size:13px;align-items:center}.track{height:20px;background:#e7edf2}.bar{height:20px}.value{text-align:right;font-variant-numeric:tabular-nums}'
    chunks=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lithology scoreboard — run in progress</title><style>'+css+'</style><body><main><h1>Lithology scoreboard: run in progress</h1>',
            '<p class="notice"><strong>INCOMPLETE — no champion.</strong> '+str(len(matched))+' of '+str(len(cases))+' cases have all four predictions. '+str(len(data['records']))+' of '+str(len(cases)*len(candidates))+' records are saved.</p>',
            '<p class="note">Only matched cases are shown. Each log has equal weight after its window scores are averaged. Score is the mean of per-window accuracy × boundary F1. These figures have not yet passed the final independent CSV replay. Partial suites must not be compared as completed suites.</p>',
            '<p>Updated '+html.escape(generated)+'. <a href="progress_scores.json">Partial data</a> · <a href="CLAY_SHALE_FIX.md">Fix and PR notes</a> · <a href="CLAY_SHALE_FIX_MEASURED.md">First measured fix result</a></p>']
    for suite in sorted({r['suite'] for r in summary}):
        count=sum(c['suite']==suite for c in cases.values())
        complete=sum(cases[k]['suite']==suite for k in matched)
        chunks+=['<h2>'+html.escape(suite)+' — '+str(complete)+'/'+str(count)+' cases</h2><div class="scroll"><table><tr><th>Pipeline</th><th>Logs</th><th>Score</th><th>Row acc.</th><th>Boundary F1</th><th>ARI</th><th>Run s</th><th>Failed</th></tr>']
        for row in [r for r in summary if r['suite']==suite]:
            chunks+=['<tr><td>'+html.escape(NAMES[row['strategy']])+'</td><td>'+str(row['n_logs'])+'</td>']
            chunks+=['<td>'+('unknown' if row[f] is None else f"{row[f]:.3f}")+'</td>' for f in FIELDS]
            chunks+=['<td>'+str(row['failures'])+'</td></tr>']
        chunks+=['</table></div>']
        chunks+=['<div class="bars">']
        for row,color in zip([r for r in summary if r['suite']==suite],['#2862a0','#148074','#bc5e25','#8258a5']):
            chunks+=['<div class="barrow"><span>'+html.escape(NAMES[row['strategy']])+'</span><div class="track"><div class="bar" style="width:'+str(100*row['score'])+'%;background:'+color+'"></div></div><span class="value">'+f"{row['score']:.3f}"+'</span></div>']
        chunks+=['</div>']
    chunks+=['<p class="note">Viridien is development regression evidence. Human QA, supplied gold, unreviewed Hadrian and outside DEV remain separate. Test and verify are closed. Runtime includes cache and process effects on a shared machine; it is not a controlled speed comparison.</p></main></body></html>']
    doc=''.join(chunks)
    (R/'progress.html').write_text(doc,encoding='utf-8')
    if args.publish_index and not (R/'scores.json').exists():
        (R/'index.html').write_text(doc,encoding='utf-8')
    print(f'Partial scoreboard: {len(matched)}/{len(cases)} matched cases; {len(data["records"])}/{len(cases)*len(candidates)} saved records; no champion')


if __name__=='__main__':main()
