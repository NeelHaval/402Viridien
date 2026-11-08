"""Replay, pair and compare the Jev run with the separate live scoreboard.

Never accepts unmatched windows or averages scores over different case sets.
"""
from collections import Counter,defaultdict
import html
import json
from pathlib import Path
import numpy as np

from jev_e2e import OUT,REFERENCE,BAKE,load,read,save,sha,validate_case
from harness import io,score

BASELINES=['s19_combine','s20_v4','s21_v4dino','s23_v4dino_identity']
NAMES={'s19_combine':'s19','s20_v4':'v4','s21_v4dino':'v4 + DINO','s23_v4dino_identity':'DINO, separate labels','s30_jev_sourcecards':'Jev source cards'}
FIELDS=['score','row_acc','boundary_f1','cluster_ari','seconds','preprocess_seconds','end_to_end_seconds']


def replay(record,case):
    for key in ['log','suite','window','truth_sha256','image_sha256','annotation_sha256']:
        if record.get(key)!=case[key]:raise ValueError('Baseline case identity mismatch '+key)
    if record.get('error'):
        if any(record.get(k)!=0 for k in ['score','row_acc','boundary_f1','cluster_ari']):raise ValueError('Failed case not scored at zero')
    else:
        pred=io.load_rows(record['prediction_csv'])
        if pred is None:raise ValueError('Missing saved baseline prediction')
        s=score.score(pred,io.load_rows(case['truth_csv']),tuple(case['window']),min_layer_px=0)
        for k,v in s.items():
            if v is None and record.get(k) is None:continue
            if v is None or record.get(k) is None or abs(v-record[k])>1e-9:raise ValueError('Baseline replay mismatch '+k)
    return dict(record,prediction_sha256=sha(record['prediction_csv']))


def means(rows):
    bylog=defaultdict(list)
    for r in rows:bylog[r['log']].append(r)
    result={}
    for f in FIELDS:
        values=[]
        for rr in bylog.values():
            v=[r[f] for r in rr if isinstance(r.get(f),(int,float))]
            if v:values.append(float(np.mean(v)))
        result[f]=float(np.mean(values)) if values else None
    return dict(metrics=result,n_cases=len(rows),n_logs=len(bylog),failures=sum(bool(r.get('error')) for r in rows))


def ci(values,units=None):
    if not values:return dict(mean=None,low=None,high=None,n=0)
    values=np.asarray(values,float)
    if units is None:units=list(range(len(values)))
    groups=[values[np.asarray(units)==u] for u in sorted(set(units))]
    rng=np.random.default_rng(1108);ix=rng.integers(0,len(groups),(10000,len(groups)))
    sums=np.array([a.sum() for a in groups]);counts=np.array([len(a) for a in groups]);draw=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(mean=float(values.mean()),low=float(np.percentile(draw,2.5)),high=float(np.percentile(draw,97.5)),n=len(values),n_units=len(groups))


def build():
    m=load();cases={c['case_id']:c for c in m['cases']};own=read(OUT/'results.json')['records']
    baseline_source=REFERENCE/'raw_results.json';raw=read(baseline_source)
    baseline=[];seen=set()
    for r in raw['records']:
        if r['strategy'] not in BASELINES or r['case_id'] not in cases:continue
        key=(r['case_id'],r['strategy'])
        if key in seen:raise ValueError('Duplicate baseline record')
        seen.add(key);baseline.append(replay(r,cases[r['case_id']]))
    save(OUT/'baseline-snapshot.json',dict(source=str(baseline_source),source_sha256=sha(baseline_source),records=baseline))
    allrows=own+baseline;by={(r['case_id'],r['strategy']):r for r in allrows}
    # A separate table for Jev on all scored cases. Comparisons use only full pairs.
    solo={s:means([r for r in own if r['suite']==s]) for s in sorted({r['suite'] for r in own})}
    summaries=[];paired=[];pair_ids={}
    for b in BASELINES:
        ids=[ident for ident in cases if (ident,b) in by and (ident,'s30_jev_sourcecards') in by]
        pair_ids[b]=ids
        for suite in sorted({cases[i]['suite'] for i in ids}):
            subset=[i for i in ids if cases[i]['suite']==suite]
            for family in ['all']+sorted({cases[i]['family'] for i in subset}):
                use=[i for i in subset if family=='all' or cases[i]['family']==family]
                a=[by[(i,'s30_jev_sourcecards')] for i in use];bb=[by[(i,b)] for i in use]
                summaries.append(dict(baseline=b,suite=suite,family=family,case_ids=use,jev=means(a),reference=means(bb)))
                diffs=defaultdict(list)
                for i in use:diffs[cases[i]['log']].append(by[(i,'s30_jev_sourcecards')]['score']-by[(i,b)]['score'])
                vals=[float(np.mean(v)) for v in diffs.values()]
                paired.append(dict(baseline=b,suite=suite,family=family,delta_jev_minus_reference=ci(vals),per_log={k:float(np.mean(v)) for k,v in diffs.items()}))
    common=[i for i in cases if all((i,s) in by for s in BASELINES+['s30_jev_sourcecards'])]
    common_tables=[]
    for suite in sorted({cases[i]['suite'] for i in common}):
        ids=[i for i in common if cases[i]['suite']==suite]
        common_tables.append(dict(suite=suite,case_ids=ids,expected=sum(c['suite']==suite for c in cases.values()),strategies={s:means([by[(i,s)] for i in ids]) for s in BASELINES+['s30_jev_sourcecards']}))
    result=dict(complete_jev=len(own)==len(cases),complete_comparison=len(common)==len(cases),expected_cases=len(cases),jev_cases=len(own),baseline_counts=dict(Counter(r['strategy'] for r in baseline)),common_cases=len(common),solo=solo,summaries=summaries,paired=paired,common_tables=common_tables,note='Strict unchanged shared scorer. Mean of per-case products, equal physical-log weights. Failed cases stay at zero. Paired tables never use different case sets. Development evidence only; external DEV only. Runtime includes cold state and concurrent use by other work. Original DINO and separate-caption DINO both retained.')
    save(OUT/'comparison.json',result)
    lines=['# End-to-end Jev source-card evaluation','',
           f"Jev cases: {len(own)}/{len(cases)}. Complete five-way pairs: {len(common)}/{len(cases)}.",
           '',
           'The source scoreboard remains separate. Baseline CSVs were replayed through the shared scorer before pairing. An incomplete table is a checkpoint, not a final suite ranking.',
           '',
           '## Same cases for all five strategies','',
           '| Suite | Matched cases / planned | Strategy | Score | Row accuracy | Boundary F1 | Inference seconds |',
           '|---|---:|---|---:|---:|---:|---:|']
    for table in common_tables:
        for name,s in table['strategies'].items():
            v=s['metrics'];secs=f"{v['seconds']:.2f}" if v['seconds'] is not None else 'not recorded'
            lines.append(f"| {table['suite']} | {len(table['case_ids'])}/{table['expected']} | {NAMES[name]} | {v['score']:.4f} | {v['row_acc']:.1%} | {v['boundary_f1']:.1%} | {secs} |")
    lines+=['','## Method','',
            '- Source image to machine legend, shared depth calibration, frozen v4 row detector, its 48-row crops, Jev matching, v4 interval grouping, shared output cleaning, unchanged shared scorer.',
            '- The inference strategy sees no truth intervals, human boxes, saved review labels or test answer keys. Score inputs match the scoreboard case manifest.',
            '- The source legend uses the same frozen native cache as v4 when present. For uncached outside logs, Jev uses the shared machine legend. This is an explicit difference from historical v4, which has no unseen-log cache repair.',
            '- Individual printed labels remain separate. The output validator rejects invented twin labels. A genuine single compound caption remains unchanged. No new scorer aliases or truth-aware label mapping are used.',
            '- The API uses four queries per request and four concurrent requests. Local ranks keep top four hatch and top two colour options. Exact RGB hashes reduce duplicate queries; approximate clusters are not propagated.',
            '- This is the proposal path: it returns a best label even when manual review is required. Bad-crop and failed API results return Unknown. Only almost completely white paper is omitted as blank. Dark solid rocks stay candidates.',
            '- Low-confidence guesses are not accepted human labels. The automatic score is measured before any human edit. Review flags and per-chunk traces remain saved.',
            '- Score is row accuracy times boundary F1 per case. Suite means average those products; they do not multiply suite mean accuracy by suite mean F1.',
            '- QA human truth, D-582 gold, Hadrian unQA, Viridien development regression and external DEV stay separate. The sealed test and verify splits are not read.',
            '- Interval grouping can change bed contacts when labels change. This test measures that effect; it does not supply true bed boxes to Jev.',
            '- Model and shared-machine timing are descriptive. Per-case API cache state and preprocessing costs are saved. Baseline and Jev timings are not a controlled hardware speed trial.',
            '',
            '## Files','',
            '- manifest.json / lock.json: fixed cases, source hashes and declared policy.',
            '- predictions/: interval CSVs.',
            '- diagnostics/: raw chunk choices, source caption IDs, API timings, cache identities and review flags.',
            '- baseline-snapshot.json: the baseline records used in this comparison.',
            '- comparison.json: matched suite and family tables, plus paired log bootstrap intervals.',
            '- audit.json: independent replay of Jev CSVs after the run.',
            '',
            'No champion is established from an incomplete run. A single-log confidence interval has no useful precision. These are development cases, not an independent final test.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    chart_tables=[dict(suite=t['suite'],n=len(t['case_ids']),expected=t['expected'],methods={k:{f:v for f,v in s['metrics'].items() if f in ['score','row_acc','boundary_f1']} for k,s in t['strategies'].items()}) for t in common_tables]
    page='''<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:var(--font-sans);color:var(--foreground);margin:0;font-size:14px}select{font:inherit;width:100%;padding:9px;background:var(--card);color:var(--foreground);border:1px solid var(--border);border-radius:var(--radius)}h3{font-size:17px;margin:15px 0 8px}p{color:var(--muted-foreground);line-height:1.5}.row{margin:14px 0}.label{display:flex;justify-content:space-between;gap:12px;margin-bottom:5px}.bar{height:18px;background:var(--muted);border-radius:4px;overflow:hidden}.fill{height:100%;background:var(--chart-1)}.jev{background:var(--chart-2)}#bars{height:282px}.parts{font-size:12px;color:var(--muted-foreground);margin-top:4px}.note{padding:16px;background:var(--card);border-radius:var(--radius)}label{display:block;margin:8px 0}.numbers{display:flex;gap:16px;flex-wrap:wrap;font-weight:600}button{font:inherit}</style></head><body><div class="numbers"><span>Jev: OWN / TOTAL cases</span><span>Five-way pairs: COMMON / TOTAL</span></div><h3>End-to-end combined score</h3><label for="suite">Compare the same cases</label><select id="suite"></select><p id="coverage"></p><div id="bars"></div><p class="note">Score = labelled-row accuracy × boundary F1, computed per case. The bars show equal-log means of those scores. No human corrections are included.</p><p>Separate Clay and Shale captions stay separate. Weak guesses still need review. All results are development evidence. Shared-machine timings do not establish a speed ranking.</p><script>const tables=DATA;const names=NAMES;const select=document.getElementById('suite');tables.forEach((t,n)=>{let o=document.createElement('option');o.value=n;o.textContent=t.suite;select.append(o)});function draw(){let t=tables[Number(select.value)];if(!t)return;document.getElementById('coverage').textContent=t.n+' of '+t.expected+' planned cases have all five results.';document.getElementById('bars').innerHTML=Object.entries(t.methods).map(([k,m])=>`<div class="row"><div class="label"><span>${names[k]}</span><strong>${m.score.toFixed(3)}</strong></div><div class="bar"><div class="fill ${k==='s30_jev_sourcecards'?'jev':''}" style="width:${m.score*100}%"></div></div><div class="parts">Rows ${(m.row_acc*100).toFixed(1)}% · Boundaries ${(m.boundary_f1*100).toFixed(1)}%</div></div>`).join('')}select.addEventListener('change',draw);draw();</script></body></html>'''
    page=page.replace('OWN',str(len(own))).replace('TOTAL',str(len(cases))).replace('COMMON',str(len(common))).replace('DATA',json.dumps(chart_tables)).replace('NAMES',json.dumps(NAMES))
    (OUT/'comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(jev_cases=len(own),common_cases=len(common),baseline_counts=result['baseline_counts'],complete=result['complete_comparison'],common_suites={t['suite']:len(t['case_ids']) for t in common_tables})),flush=True)


if __name__=='__main__':build()
