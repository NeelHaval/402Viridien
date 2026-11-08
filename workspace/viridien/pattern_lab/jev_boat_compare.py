"""Replay paired saved CSVs with the existing shared scorer. Run on boat only."""
from collections import defaultdict
import html
import json
from pathlib import Path
import sys
import worker

ROOT=Path(__file__).resolve().parent
METHODS=['s19_combine','s20_v4','s21_v4dino','s23_v4dino_identity','s30_jev_sourcecards']
NAMES=['s19','v4','v4 + DINOv2','DINOv2, separate captions','Jev source cards']


def main():
    io,adapter,api,env=worker.init()
    from harness.score import score
    from strategies.grading_policy import score_equivalent
    records=worker.read(ROOT/'comparison-inputs/records.json')['records'];checked=[]
    for record in records:
        c=record['case'];path=ROOT/record['prediction'];truth=ROOT/record['truth']
        if worker.sha(path)!=record['prediction_sha256'] or worker.sha(truth)!=c['truth_sha256']:raise ValueError('Comparison input changed')
        pred=io.load_rows(path);gt=io.load_rows(truth)
        if record.get('error'):
            strict=dict(score=0.,row_acc=0.,boundary_f1=0.,cluster_ari=0.,depth_mae_ft=None);equivalent=dict(strict)
        else:
            strict=score(pred,gt,tuple(c['window']),min_layer_px=0);equivalent=score_equivalent(pred,gt,tuple(c['window']),min_layer_px=0)
            for key in ['score','row_acc','boundary_f1','cluster_ari','depth_mae_ft']:
                a,b=strict[key],record['recorded_strict'].get(key)
                if a is None and b is None:continue
                if a is None or b is None or abs(a-b)>1e-9:raise ValueError('Saved strict score did not replay: '+c['case_id']+' '+record['strategy']+' '+key)
        checked.append(dict(case_id=c['case_id'],suite=c['suite'],family=c['family'],log=c['log'],window=c['window'],strategy=record['strategy'],strict_harness=strict,clay_shale_equivalent_v1=equivalent,prediction_sha256=record['prediction_sha256'],truth_sha256=c['truth_sha256'],image_sha256=c['image_sha256'],annotation_sha256=c['annotation_sha256'],error=record.get('error')))
    by={(r['case_id'],r['strategy']):r for r in checked}
    identities=defaultdict(set)
    for r in checked:identities[r['case_id']].add((r['log'],tuple(r['window']),r['truth_sha256'],r['image_sha256'],r['annotation_sha256']))
    if any(len(v)!=1 for v in identities.values()):raise ValueError('Paired identity mismatch')
    complete=[i for i in identities if all((i,m) in by for m in METHODS)]
    suites=sorted({by[(i,METHODS[0])]['suite'] for i in complete});tables=[]
    for suite in suites:
        ids=[i for i in complete if by[(i,METHODS[0])]['suite']==suite];methods={}
        for method in METHODS:
            rr=[by[(i,method)] for i in ids];policies={}
            for policy in ['strict_harness','clay_shale_equivalent_v1']:
                groups=defaultdict(list)
                for r in rr:groups[r['log']].append(r[policy])
                values={k:sum(sum(v[k] for v in group)/len(group) for group in groups.values())/len(groups) for k in ['score','row_acc','boundary_f1']}
                policies[policy]=values
            methods[method]=dict(policies,failures=sum(bool(r.get('error')) for r in rr))
        tables.append(dict(suite=suite,case_ids=ids,n_logs=len({by[(i,METHODS[0])]['log'] for i in ids}),methods=methods))
    result=dict(status='checkpoint',expected_cases=68,jev_cases=sum(r['strategy']==METHODS[-1] for r in checked),common_cases=len(complete),tables=tables,records=checked,metric_sha256=worker.sha(ROOT/'source/harness/score.py'),grading_sha256=worker.sha(ROOT/'source/strategies/grading_policy.py'),source_fingerprint=env['source_fingerprint'],input_sha256=worker.sha(ROOT/'comparison-inputs/records.json'),note='Exact paired windows and source hashes. Equal physical-log means. Shared scorer replay; failed cases remain zero. Development only. Strict and owner-approved Clay/Shale equivalence are separate.')
    output=ROOT/'comparison';output.mkdir(exist_ok=True);worker.write(output/'comparison.json',result)
    rows=''
    for table in tables:
        for method,name in zip(METHODS,NAMES):
            v=table['methods'][method];a=v['strict_harness'];b=v['clay_shale_equivalent_v1']
            rows+='<tr><td>'+html.escape(table['suite'])+'</td><td>'+str(len(table['case_ids']))+'</td><td>'+name+'</td><td>'+f"{a['score']:.4f}"+'</td><td>'+f"{b['score']:.4f}"+'</td><td>'+f"{b['row_acc']:.1%}"+'</td><td>'+f"{b['boundary_f1']:.1%}"+'</td></tr>'
    payload=json.dumps(tables)
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:15px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;color:#172331;background:white}p{line-height:1.5}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccd3dd}.row{margin:15px 0}.line{display:flex;justify-content:space-between;gap:12px}.track{background:#e8edf2;height:17px;margin-top:5px}.bar{background:#35749c;height:17px}.jev{background:#bd6b28}.parts{color:#526170;font-size:12px;margin-top:4px}select{font:inherit;padding:8px;width:100%;margin:8px 0}#bars{height:390px}code{overflow-wrap:anywhere}</style></head><body><h1>Jev end-to-end checkpoint</h1><p>Jev cases: OWN of 68. Complete five-way pairs: COMMON of 68. The same windows and source hashes are used for each comparison.</p><label>Suite<select id="suite"></select></label><label>Grading<select id="policy"><option value="clay_shale_equivalent_v1">Clay/Shale equivalent</option><option value="strict_harness">Strict shared harness</option></select></label><div id="bars"></div><p>Score is row accuracy × boundary F1 for each case. The chart shows equal-log means of those scores. Raw captions and bed contacts stay unchanged. Clay, Shale, Clay/Shale and Clay-Shale receive equal credit only in the approved view. This is development evidence. It does not show unseen-test performance.</p><table><thead><tr><th>Suite</th><th>Cases</th><th>Method</th><th>Strict score</th><th>Equivalent score</th><th>Equivalent rows</th><th>Boundaries</th></tr></thead><tbody>ROWS</tbody></table><p>Metric fingerprint: <code>METRIC</code><br>Grading fingerprint: <code>GRADING</code></p><script>const tables=DATA;const methods=METHODS;const names=NAMES;const suite=document.getElementById('suite');const policy=document.getElementById('policy');tables.forEach((t,i)=>{let o=document.createElement('option');o.value=i;o.textContent=t.suite+' · '+t.case_ids.length+' cases';suite.append(o)});function draw(){let t=tables[+suite.value];if(!t)return;document.getElementById('bars').innerHTML=methods.map((m,i)=>{let v=t.methods[m][policy.value];return `<div class="row"><div class="line"><span>${names[i]}</span><b>${v.score.toFixed(4)}</b></div><div class="track"><div class="bar ${i===4?'jev':''}" style="width:${100*v.score}%"></div></div><div class="parts">Rows ${(v.row_acc*100).toFixed(1)}% · Boundaries ${(v.boundary_f1*100).toFixed(1)}%</div></div>`}).join('')}suite.onchange=policy.onchange=draw;draw()</script></body></html>'''
    for key,value in [('OWN',str(result['jev_cases'])),('COMMON',str(len(complete))),('ROWS',rows),('METRIC',result['metric_sha256']),('GRADING',result['grading_sha256']),('DATA',payload),('METHODS',json.dumps(METHODS)),('NAMES',json.dumps(NAMES))]:page=page.replace(key,value)
    (output/'report.html').write_text(page,encoding='utf-8')
    worker.write(output/'receipt.json',dict(status='ok',input_sha256=result['input_sha256'],source_fingerprint=env['source_fingerprint'],files={n:worker.sha(output/n) for n in ['comparison.json','report.html']}))
    print('PAIRED_REPLAY_OK',result['jev_cases'],'Jev cases',len(complete),'five-way pairs',flush=True)


if __name__=='__main__':main()
