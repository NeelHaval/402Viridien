"""Describe the frozen final outputs; never rerun inference or select settings."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from harness import io,score
from harness.run import _truth

HERE=Path(__file__).resolve().parent
BAKEOFF=HERE.parents[1]
REPO=BAKEOFF.parent


def main():
    result=json.loads((HERE/'final_metrics.json').read_text())
    details={};confusions=[]
    for split,res in result.items():
        windows=[]
        for rec in res['logs']:
            name=rec['log'];win=tuple(rec['window'])
            pred=io.load_rows(HERE/'final'/split/(name+f'__{win[0]}_{win[1]}.csv'))
            gt=io.load_rows(REPO/'external_tests'/'truth'/(name+'.csv')) if split=='ext_test' else _truth(name,'consensus',win)
            # Counts and confusions only: the final scorer is not rerun.
            p=score.raster(pred,win);g=score.raster(gt,win);filled=g!=''
            counts={}
            for actual,guess in zip(g[filled],p[filled]):
                if actual!=guess:counts[(actual,guess)]=counts.get((actual,guess),0)+1
            for (actual,guess),n in sorted(counts.items(),key=lambda v:-v[1])[:3]:
                confusions.append(dict(split=split,log=name,window=list(win),truth=actual,prediction=guess or '(blank)',rows=n))
            windows.append(dict(log=name,window=list(win),score=rec['score'],row_acc=rec['row_acc'],
                boundary_f1=rec['boundary_f1'],cluster_ari=rec['cluster_ari'],
                filled_truth_rows=int(filled.sum()),predicted_internal_boundaries=len(score.boundaries(pred,win)),
                truth_internal_boundaries=len(score.boundaries(gt,win)),depth_mae_ft=rec['depth_mae_ft']))
        depth_rows=[r for r in res['logs'] if r.get('depth_mae_ft') is not None]
        worst=max(depth_rows,key=lambda r:r['depth_mae_ft'])
        path=HERE/'cache'/'out'/'_cache'/'depth'/(worst['log']+'.rapidocr.json')
        cache=json.loads(path.read_text())
        details[split]=dict(windows=windows,zero_score_windows=sum(r['score']==0 for r in res['logs']),
            runtime_failures=sum('error' in r for r in res['logs']),depth_contributing_windows=len(depth_rows),
            largest_depth_error=dict(log=worst['log'],depth_mae_ft=worst['depth_mae_ft'],calibration=cache['calibration']),
            max_seconds=max(r['seconds'] for r in res['logs']))
    lock=json.loads((HERE/'FINAL_LOCK.json').read_text())
    assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==digest for p,digest in lock['sha256'].items())
    validation=json.loads((HERE/'submission_validation.json').read_text())
    assert len(validation['checks'])==13 and all(c['valid'] for c in validation['checks'])
    assert len(list((HERE/'submission_preview').glob('*.csv')))==13
    with (HERE/'final_confusions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,['split','log','window','truth','prediction','rows']);writer.writeheader();writer.writerows(confusions)
    (HERE/'final_diagnostics.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    family=list(csv.DictReader((HERE/'per_family.csv').open()))
    s18family=[r for r in family if r['strategy']=='s18_best']
    text=[
        'S18 won dev selection and the external test composite; S15 retains the higher in-domain composite.',
        '',
        'External test: S18 score 0.1116 versus S14 0.0883 (the strongest archived external-test composite). In-domain: S18 score 0.3468 versus S15 0.3674. S18 has the highest window-mean boundary F1 and ARI on both final sets. No pipeline setting was changed after either final result.',
        '',
        'The extra boundaries on MC765 matter: S18 predicts '+str(next(r['predicted_internal_boundaries'] for r in details['viridien_eval']['windows'] if r['log'].startswith('MC765_')))+' internal boundaries, while the current consensus truth has '+str(next(r['truth_internal_boundaries'] for r in details['viridien_eval']['windows'] if r['log'].startswith('MC765_')))+'. Its row accuracy is 0.8213, boundary F1 is zero, and composite score is zero. This window alone contributes -1/13 to S18 versus S15’s composite difference; improvements on other families offset most of that loss. The fixed drawn-rule policy was selected on dev and was retained.',
        '',
        'D-582 remains a naming failure: mean row accuracy is 0.0094 despite boundary F1 0.6249. This is descriptive evidence of name errors, not proof that the matcher or parser alone caused them. final_confusions.csv lists the largest per-window label confusions.',
        '',
        'External depth MAE is 21.8205 ft over '+str(details['ext_test']['depth_contributing_windows'])+' windows with matched depth boundaries. One shared calibration, ext_kgs_EmmaWard1_p002, has mean matched-boundary depth error 384.4000 ft. Its cached scale is 3.66429 ft/pixel, while the supplied truth’s interval depth increments are approximately 0.13333 ft/pixel. This is a calibration failure inherited from the shared depth stage, not a runtime failure. It is retained in the final result. The matched-boundary-only metric changes its contributing boundaries with each strategy, so baseline depth MAEs are not a constant calibration comparison.',
        '',
        'There were zero runtime failures. Six external windows and one in-domain window have zero composite score. Peak RSS across preprocessing and inference was 1.2202 GiB. Maximum strategy time was '+f"{details['ext_test']['max_seconds']:.3f}"+' s on external test and '+f"{details['viridien_eval']['max_seconds']:.3f}"+' s in-domain; thread count was capped at three.',
        '',
        'Confidence has limited resolution. Grouped dev selection chose logistic C=0.3 with 0.25 shrinkage toward 0.5; Brier loss is 0.2473 versus 0.2500 for fixed 0.5. These are selection estimates. Confidence is provided per interval in sidecars and does not alter the six-column submission schema.',
        '',
        'All 13 submission CSVs pass the SPEC.md fallback checks: exact column order, integer full-image pixel rows, sorted non-overlapping bottom-exclusive intervals, finite monotone depths rounded to 0.1. They cover only the 13 fixed evaluation windows. The frozen code/model hashes still match FINAL_LOCK.json.',
        '',
        'All strategies by source: per_source.csv. All strategies by in-domain family: per_family.csv, mapped using analysis/label_shares_per_log.csv. Complete comparisons: REPORT.md and comparison.json.',
    ]
    (HERE/'FINAL_ANALYSIS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    report=HERE/'REPORT.md'
    marker='\nPost-evaluation failure analysis: [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md). This reads saved predictions and truth; it does not rerun inference or change the frozen pipeline.\n'
    if marker not in report.read_text():report.write_text(report.read_text()+marker,encoding='utf-8')
    print(json.dumps({split:{k:v for k,v in d.items() if k!='windows'} for split,d in details.items()},indent=2))


if __name__=='__main__':main()
