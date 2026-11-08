"""Rescore saved CSVs under the owner's symmetric grading policy. No inference."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=HERE.parents[1]
sys.path.insert(0,str(B));sys.path.insert(0,str(HERE))
from harness import io,score
from strategies import grading_policy
from report_diagnostics import runtime_summary,runtime_table,native_legend_summary,native_table
METRICS=('score','row_acc','boundary_f1','cluster_ari','depth_mae_ft')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def policy_metadata():
    source=Path(grading_policy.__file__).resolve();scorer=B/'harness'/'score.py'
    return dict(id=grading_policy.POLICY_ID,source_path=str(source),source_sha256=sha(source),equivalent_labels=list(grading_policy.EQUIVALENT_LABELS),canonical_label='Clay/Shale',shared_scorer_path=str(scorer),shared_scorer_sha256=sha(scorer),owner_configured=True,official_challenge_rule_verified=False,postprocessor_path=str(Path(__file__).resolve()),postprocessor_sha256=sha(__file__),grade_status='Local proxy; no official judging formula or equivalence was verified.')


def enrich(data):
    """Return a copy with accepted metrics; retain every original strict scalar."""
    out=copy.deepcopy(data);truth_cache={};pred_cache={}
    def load(path,cache):
        if path not in cache:
            rows=io.load_rows(path)
            if rows is None:raise ValueError('Missing saved interval CSV: '+path)
            cache[path]=rows
        return cache[path]
    for record in out['records']:
        truth=load(record['truth_csv'],truth_cache);window=tuple(record['window'])
        for scope in ('primary','pixel'):
            target=record if scope=='primary' else record.get('pixel_only')
            if target is None:continue
            pred=load(target['prediction_csv'],pred_cache)
            accepted=grading_policy.score_equivalent(pred,truth,window,min_layer_px=0)
            if scope=='primary' and record.get('error'):
                accepted=dict(score=0.0,row_acc=0.0,boundary_f1=0.0,cluster_ari=0.0,depth_mae_ft=None)
            for metric in ('boundary_f1','depth_mae_ft'):
                strict=target.get(metric);current=accepted[metric]
                if strict is None or current is None:
                    if strict!=current:raise ValueError(scope+' '+metric+' changed under label-only grading')
                elif abs(strict-current)>1e-10:raise ValueError(scope+' '+metric+' changed under label-only grading')
            target['accepted_metrics']={metric:accepted[metric] for metric in METRICS}
    out['grading_policy']=policy_metadata()
    return out



def merge_preview_shard(data,shard,manifest):
    """Merge validated remote S20 records into a copy for preview only."""
    from report_inputs import IDENTITY
    from report_integrity import manifest_digest
    frozen=manifest_digest(manifest)
    if shard.get('manifest_digest')!=frozen:raise ValueError('Remote shard manifest digest mismatch')
    cases={c['case_id']:c for c in manifest['cases']}
    out=copy.deepcopy(data);seen=set()
    for row in out['records']:
        key=(row['case_id'],row['strategy'])
        if key in seen:raise ValueError('Duplicate local preview record key')
        seen.add(key)
    remote=[]
    for row in shard.get('records',[]):
        if row.get('strategy')!='s20_v4':raise ValueError('Remote preview accepts S20 only')
        case=cases.get(row.get('case_id'))
        if case is None:raise ValueError('Remote shard has an undeclared case')
        if row.get('manifest_digest')!=frozen:raise ValueError('Remote row manifest digest mismatch')
        for field in IDENTITY:
            if field not in case or field not in row or row[field]!=case[field]:raise ValueError('Remote row case identity mismatch: '+field)
        key=(row['case_id'],row['strategy'])
        if key in seen:raise ValueError('Duplicate remote/local preview record key')
        seen.add(key);remote.append(copy.deepcopy(row))
    out['records'].extend(remote)
    out['preview_merge']=dict(remote_records=len(remote),local_records=len(data['records']),preview_only=True,source='optional remote shard; raw local checkpoint is unchanged',runtime_note='Remote CPU inference and local timing have different scopes. Preserve per-record runtime/provenance; do not use their pooled mean as a speed tie-break.')
    return out

def matched_quartets(records,manifest):
    from report_inputs import IDENTITY
    candidates=set(manifest['candidates']);by=defaultdict(list);matched=[]
    for r in records:by[r['case_id']].append(r)
    for case in manifest['cases']:
        rr=by.get(case['case_id'],[])
        if len(rr)==len(candidates) and {r['strategy'] for r in rr}==candidates:
            if any(any(r.get(k)!=case[k] for k in IDENTITY if k in case) for r in rr):raise ValueError('Mismatched preview case identity')
            matched.extend(rr)
    return matched


def add_summaries(data,manifest):
    # The runner's pure summary function is used on saved data only.
    spec=importlib.util.spec_from_file_location('scoreboard_summary_only',HERE/'score_all.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    strict=matched_quartets(data['records'],manifest);accepted=copy.deepcopy(strict)
    for r in accepted:r.update(r['accepted_metrics'])
    data['summaries'],data['paired']=runner.summarize(strict)
    data['accepted_summaries'],data['accepted_paired']=runner.summarize(accepted)
    from report_inputs import is_initial_manual_stop
    data['provisional_cases']=[dict(case_id=r['case_id'],strategy=r['strategy'],log=r['log'],reason='Initial operator interruption; formal same-cap attempt pending',wall_upper_bound_seconds=r.get('checkpoint_to_stop_wall_interval',{}).get('elapsed_upper_bound_seconds')) for r in strict if is_initial_manual_stop(r)]
    data['runtime_regimes']=runtime_summary(strict)
    data['native_legend_status']=native_legend_summary(strict)
    data['progress']=dict(total_records=len(data['records']),expected_records=len(manifest['cases'])*len(manifest['candidates']),matched_records=len(strict),matched_cases=len(strict)//len(manifest['candidates']),expected_cases=len(manifest['cases']),candidate_count=len(manifest['candidates']),summaries_use_only_matched_cases=True)
    data['progress']['suite_counts']=[dict(suite=s,matched_cases=len({r['case_id'] for r in strict if r['suite']==s}),expected_cases=sum(c['suite']==s for c in manifest['cases'])) for s in sorted({c['suite'] for c in manifest['cases']})]
    return data


def partial_page(data):
    import html
    esc=lambda x:html.escape(str(x))
    names={'s19_combine':'s19','s20_v4':'v4','s21_v4dino':'DINO baseline','s23_v4dino_identity':'DINO identity'}
    display=lambda x:names.get(x,x)
    p=data['progress'];rows=[r for r in data['accepted_summaries'] if r['family']=='all']
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Incomplete lithology proxy scores</title><style>body{font:16px/1.5 Arial,sans-serif;color:#193849;margin:20px auto;padding:0 14px;max-width:1100px}h1{line-height:1.1}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #cbd8e0;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left;white-space:normal;min-width:115px;max-width:155px;overflow-wrap:anywhere}.scroll{overflow:auto}.notice{background:#fff0d6;padding:16px}a{color:#075f9c}summary{cursor:pointer}</style><h1>INCOMPLETE · local proxy scores</h1><p class="notice">No champion is declared. '+str(p['total_records'])+' of '+str(p['expected_records'])+' saved records are available. Tables use only '+str(p['matched_cases'])+' of '+str(p['expected_cases'])+' cases with all '+str(p['candidate_count'])+' candidates present.</p><p>The primary local proxy treats Clay, Shale, Clay/Shale and Clay-Shale as equivalent in both directions. It calls the unchanged harness scorer on copied labels. This policy and the score formula are not verified official challenge rules and do not predict judging.</p><p>Score = row accuracy × boundary F1 within each window, then equal-log means. Components stay separate. Wide tables scroll horizontally. Strict grading is retained as a reference. Inference was not rerun.</p><div class="scroll"><table><tr><th>Suite / strategy</th><th>Logs</th><th>Score</th><th>Row acc.</th><th>Boundary F1</th><th>ARI</th><th>Mixed run s</th></tr>']
    if data.get('provisional_cases'):
        bounds=[x['wall_upper_bound_seconds'] for x in data['provisional_cases'] if x.get('wall_upper_bound_seconds') is not None]
        known=(f'{max(bounds):.2f}-second checkpoint-to-stop wall upper bound' if bounds else 'checkpoint-to-stop wall upper bound with no measured inference duration')
        notice='<p class="notice">Human QA is provisional: S19 Shenzi was stopped by the operator. Its zero is not a demonstrated 600-second failure; only a '+known+' is known. One formal attempt with the common cap is pending. No human-QA verdict is made. The complete 13-log Viridien cohort is unaffected. Shenzi contacts may use machine anchors.</p>'
        parts[0]=parts[0].replace('<p>The primary local proxy',notice+'<p>The primary local proxy')

    for r in rows:
        m=r['metrics'];parts+=['<tr><td>'+esc(r['suite'])+'<br>'+esc(display(r['strategy']))+'</td><td>'+str(r['n_logs'])+'</td>'+''.join('<td>'+('unknown' if m.get(k) is None else f'{m[k]:.4f}')+'</td>' for k in ('score','row_acc','boundary_f1','cluster_ari','seconds'))+'</tr>']
    parts+=['</table></div><h2>Matched cohort coverage</h2><ul>']
    for cohort in p['suite_counts']:parts+=['<li>'+esc(cohort['suite'])+': '+str(cohort['matched_cases'])+' of '+str(cohort['expected_cases'])+' cases have all candidates; '+('cohort complete' if cohort['matched_cases']==cohort['expected_cases'] else 'incomplete')+'.</li>']
    parts+=['</ul><h2>Viridien paired score differences</h2><p>Accepted local proxy, A minus B. Each interval uses 10,000 paired bootstrap draws (seed 1108). The primary interval resamples whole logs; the sensitivity interval resamples provisional source groups. A positive value favours A. This cohort result does not complete the other suites or declare a champion.</p><div class="scroll"><table><tr><th>A minus B</th><th>Logs</th><th>Mean Δ</th><th>Log 95% CI</th><th>Groups</th><th>Group 95% CI</th></tr>']
    for pair in data['accepted_paired']:
        if pair['suite']!='viridien_eval':continue
        log=pair['metrics']['score']['log_ci'];well=pair['metrics']['score']['well_ci']
        fmt=lambda value:'unknown' if value is None else f'{value:.4f}'
        parts+=['<tr><td>'+esc(display(pair['a'])+' minus '+display(pair['b']))+'</td><td>'+str(log.get('n',0))+'</td><td>'+fmt(log.get('mean'))+'</td><td>['+fmt(log.get('low'))+', '+fmt(log.get('high'))+']</td><td>'+str(well.get('n_units',0))+'</td><td>['+fmt(well.get('low'))+', '+fmt(well.get('high'))+']</td></tr>']
    parts+=['</table></div><details><summary>Strict reference · matched cases only</summary><div class="scroll"><table><tr><th>Suite / strategy</th><th>Score</th><th>Row acc.</th><th>Boundary F1</th></tr>']
    for r in data['summaries']:
        if r['family']=='all':parts+=['<tr><td>'+esc(r['suite'])+'<br>'+esc(display(r['strategy']))+'</td>'+''.join('<td>'+f'{r["metrics"][k]:.4f}'+'</td>' for k in ('score','row_acc','boundary_f1'))+'</tr>']
    parts+=['</table></div></details><h2>Separate timing regimes</h2><p>The mixed run-time aggregate is descriptive and cannot establish efficiency. Remote CPU crop inference and local source-page calls have different scopes. Missing time means unknown; raw seconds are unchanged. No speed tie-break is used.</p>',runtime_table(data['runtime_regimes'],display),'<h2>Native legend availability</h2><p>Counts use recorded primary or pixel-only evidence and read both local and portable diagnostic fields. Missing native entries are a pipeline limit; unknown means evidence is absent. S19 uses another legend path.</p>',native_table(data['native_legend_status'],display),'<p><a href="accepted_preview.json">All preview scores and paired intervals</a> · <a href="CLAY_SHALE_FIX.md">Caption identity experiment</a> · <a href="CHALLENGE_ALIGNMENT.md">Challenge alignment</a></p><p>The local product does not directly include ARI, depth error, runtime or memory; these are separate measures. Row accuracy ignores truth-blank rows. A window grade is not proof of full-scan readiness. <a href="readiness_preview.json">Readiness diagnostic</a> · <a href="remote/remote_results_shard.json">Remote result shard</a>.</p><p>Some S20 records can use remote CPU inference on decoded lossless windows. Their timing scope differs from local source-page IO and CUDA calls; do not compare pooled run times as a speed tie-break.</p><p>This is an observed development comparison. Historical OCR/depth cache bytes were not captured at every read. External test and verify remain closed. Final selection requires complete coverage and both bound audits.</p></html>']
    (HERE/'index.html').write_text(''.join(parts),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');ap.add_argument('--no-page',action='store_true');ap.add_argument('--remote-shard',type=Path);args=ap.parse_args()
    if args.final and args.remote_shard:ap.error('--remote-shard is preview-only; the runner must merge final records')
    source=HERE/('scores.json' if args.final else 'raw_results.json');data=read(source);manifest=read(HERE/'manifest_1108.json')
    if args.remote_shard:
        data=merge_preview_shard(data,read(args.remote_shard),manifest)
        data['preview_merge']['shard_path']=str(args.remote_shard.resolve());data['preview_merge']['shard_sha256']=sha(args.remote_shard)
    if args.final:
        from report_inputs import validate_records
        validate_records(data,manifest)
        backup=HERE/'scores_strict.json'
        if backup.exists():
            if read(backup)['records']!=data['records'] and 'grading_policy' not in data:raise ValueError('Existing strict backup differs from final results')
        else:write(backup,data)
    enriched=add_summaries(enrich(data),manifest)
    target=HERE/('scores.json' if args.final else 'accepted_preview.json');write(target,enriched)
    if not args.final and not args.no_page:partial_page(enriched)
    print('Saved',target.name,enriched['progress'])
    for r in enriched['records']:
        if r['suite']=='viridien_eval' and 'log06' in r['log']:print(r['strategy'],'strict',r['score'],r['row_acc'],r['boundary_f1'],'accepted',r['accepted_metrics'])
if __name__=='__main__':main()
