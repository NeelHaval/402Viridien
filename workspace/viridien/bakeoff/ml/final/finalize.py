"""Audit and freeze before the single final run; write its tables afterward."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import json
import shutil
import numpy as np
import pandas as pd
import torch
from harness import bench

HERE=Path(__file__).resolve().parent
BAKEOFF=HERE.parents[1]
REPO=BAKEOFF.parent
STRATEGIES='s11,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22'
SUITES='viridien_eval,ext_dev,ext_test,ext_shared_verify'
STATUS={s:('historical_contamination' if s in ('s13','s14','s15','s16','s17','s18','s20') else 'clean_refit' if s in ('s19','s21','s22') else 'fixed_baseline') for s in STRATEGIES.split(',')}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    bench._write_json(path,value)

def now():
    return datetime.now(timezone.utc).isoformat()

def audit():
    if (HERE/'FREEZE.json').exists():
        raise RuntimeError('Audit must precede freeze')
    items=read(REPO/'external_tests/eval_set_dev.json')
    expected={(r['log'],r['y0'],r['y1']) for r in items}
    assert len(items)==51
    eval_ids={r['log'] for r in read(BAKEOFF/'eval_set.json')}
    index=read(BAKEOFF/'ml/combine/index.json')
    wells=set(r['well'] for r in index['dev'])
    assert len(wells)==5 and 'ForsythB2-17' not in wells
    assert not eval_ids & {r[0] for r in expected}
    records=[]
    for lane in ('combine','synth_v2','boundary_v2'):
        root=BAKEOFF/'ml'/lane
        path=root/'boundary_net.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert set(payload['training_wells'])==wells
        assert payload.get('held_well') is None
        if 'dev_sha256' in payload:
            assert payload['dev_sha256']==bench.file_hash(REPO/'external_tests/eval_set_dev.json')
        meta={k:v for k,v in payload.items() if k!='state_dict'}
        records.append(dict(lane=lane,path=path.relative_to(REPO).as_posix(),sha256=bench.file_hash(path),metadata=meta))
    real=torch.load(BAKEOFF/'ml/combine/boundary_real.pt',map_location='cpu',weights_only=True)
    assert set(real['training_wells'])==wells and real['synthetic_fraction']==0.
    for deployed, source in [('boundary_net.pt','train_0.35/all.pt'),
                             ('boundary_real.pt','train_0/all.pt'),
                             ('matcher.npz','train_0.35/all.npz'),
                             ('matcher_real.npz','train_0/all.npz')]:
        assert bench.file_hash(BAKEOFF/'ml/combine'/deployed)==bench.file_hash(BAKEOFF/'ml/combine'/source)
    training=read(BAKEOFF/'ml/combine/training.json')
    assert training['inherited_weights_loaded'] is False
    for fit in training['fits']:
        assert set(fit['training_wells'])==wells-{fit['held_well']}
    for lane,path,key in [('combine','index.json','dev'),('synth_v2','dev_index.json',None),('boundary_v2','data/index.json','cases')]:
        data=read(BAKEOFF/'ml'/lane/path)
        cases=data[key] if key else data
        assert {(r['log'],*r['window']) for r in cases}==expected
        assert {r['well'] for r in cases}==wells
    for lane in ('synth_gulf','synth_v2'):
        records_syn=read(BAKEOFF/'ml'/lane/'synthetic_index.json')
        train={r['source'] for r in records_syn if r['split']=='train'}
        held={r['source'] for r in records_syn if r['split']=='held'}
        assert not train & held and not (train|held)&eval_ids
    old=read(BAKEOFF/'ml/boundary/training.json')
    assert 'ForsythB2-17' in old['training_wells']
    assert bench.file_hash(BAKEOFF/'ml/boundary/boundary_net.pt')==old['weights_sha256']
    gulf=torch.load(BAKEOFF/'ml/synth_gulf/boundary_net.pt',map_location='cpu',weights_only=True)
    assert 'ForsythB2-17' in gulf['training_wells']
    configs={n:bench.load_suite(n) for n in SUITES.split(',')}
    assert {n:len(c['resolved_items']) for n,c in configs.items()}==dict(viridien_eval=13,ext_dev=51,ext_test=20,ext_shared_verify=8)
    # Resolve only the declared benchmark IDs. Do not enumerate external data.
    missing=[]
    for name,config in configs.items():
        for item in config['resolved_items']:
            annotation=config['dataset']/'bbox_annotation'/(item['log']+'.json')
            if not annotation.is_file():
                missing.append(dict(suite=name,log=item['log'],kind='annotation'))
                continue
            try:
                bench._image_path(config['dataset'],item['log'],read(annotation))
            except FileNotFoundError:
                missing.append(dict(suite=name,log=item['log'],kind='image'))
    inventory={}
    for folder in ('external_tests_round3','external_unlabelled'):
        root=REPO/folder
        if (root/'manifest.csv').is_file():
            with (root/'manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            inventory[folder]=dict(manifest_records=len(rows),summary=read(root/'build_summary.json'))
            if (root/'round4_conversion.json').is_file():
                r=read(root/'round4_conversion.json')
                inventory[folder]['round4']={k:v for k,v in r.items() if k!='documents'}
    ready={}
    for lane,filename in [('combine','READY.json'),('synth_v2','READY.json'),('legend_v4','PROVENANCE.json')]:
        obj=read(BAKEOFF/'ml'/lane/filename)
        hashes=obj.get('hashes',obj.get('sha256',{}))
        mismatches=[]
        checked=0
        for relative,digest in hashes.items():
            if relative.startswith('bakeoff/'):
                path=REPO/relative
            elif relative.startswith(('ml/','strategies/','harness/')):
                path=BAKEOFF/relative
            else:
                path=BAKEOFF/'ml'/lane/relative
            checked+=1
            if not path.is_file() or bench.file_hash(path)!=digest:
                mismatches.append(relative)
        ready[lane]=dict(checked=checked,changed_after_review=mismatches)
    result=dict(audited_utc=now(),dev_sha256=bench.file_hash(REPO/'external_tests/eval_set_dev.json'),training_wells=sorted(wells),clean_checkpoints=records,
        clean_final_real_checkpoint_sha256=bench.file_hash(BAKEOFF/'ml/combine/boundary_real.pt'),
        statuses=STATUS,readiness_hash_checks=ready,missing_files=missing,
        suites={n:dict(windows=len(c['resolved_items']),wells=len({r['well'] for r in c['resolved_items']})) for n,c in configs.items()},
        new_data=inventory,limits=['Old S13-S18 and deployed S20 remain contaminated. They are historical comparisons only.',
        'Old Emil local weights used six eval images; default Drive provenance is incomplete. S12 is not requested.',
        'Earlier final scores already exist. Clean refits do not remove inherited design exposure.',
        'No sealed image or truth was opened. Only declared split metadata was read.',
        'No real non-eval in-domain interval truth is present. Source-held synthetic checks provide the permitted in-domain evidence.'])
    write(HERE/'audit.json',result)
    print('AUDIT:',len(records),'clean deployed boundary checkpoints; dev windows',len(items),'missing files',len(missing))
    assert not missing

def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'FINAL_STARTED.json').exists():
        raise RuntimeError('Final choice already frozen')
    aud=read(HERE/'audit.json')
    check=read(HERE/'selection_recheck.json')
    assert check['strategy']=='s19' and check['objective']>.5
    test_tail=(HERE/'final_tests.log').read_text().splitlines()[-1]
    assert 'passed' in test_tail and 'failed' not in test_tail
    assert all(bench._strategy_spec(s) for s in STRATEGIES.split(','))
    hashes={p.relative_to(REPO).as_posix():bench.file_hash(p) for p in bench.dependency_files()}
    result=dict(frozen_utc=now(),chosen_strategy='s19',strategy_ids=STRATEGIES.split(','),suites=SUITES.split(','),statuses=STATUS,
        dependency_hash=bench._dependencies(),source_fingerprints=hashes,dev_sha256=aud['dev_sha256'],
        selection=dict(dev=check['dev']['well_macro'],synthetic=check['synthetic']['well_macro'],objective=check['objective'],rule='0.75 dev well mean + 0.25 non-eval source-held synthetic family mean'),
        candidate_evidence=dict(s11=dict(dev_score=.1041,source='ml/boundary_v2/REPORT.md'),
            s19=dict(objective=check['objective'],source='ml/final/selection_recheck.json'),
            s21=dict(objective=.3101402246290079,source='ml/synth_v2/fit.json'),
            s22=dict(dev_score=.1839,source='ml/boundary_v2/REPORT.md',limit='Synthetic check measures proposals only. No comparable in-domain full-pipeline score.'),
            s20=dict(eligible=False,reason='Deployed S18 checkpoint contamination. Clean parser CV is a component comparison only.')),
        final_tests_summary=test_tail,command=f'python -m harness.bench --strategy {STRATEGIES} --suite {SUITES} --workers 4',
        workers=4,threads_per_worker=1,cache_version=bench.CACHE_VERSION,
        limits=check['limits'],no_tuning_after_final=True)
    write(HERE/'FREEZE.json',result)
    print('FROZEN:',result['chosen_strategy'],result['dependency_hash'],test_tail)

def report(run_dir):
    frozen=read(HERE/'FREEZE.json')
    assert frozen['dependency_hash']==bench._dependencies(),'Code or model changed after freeze'
    run_dir=Path(run_dir)
    result=read(run_dir/'results.json')
    df=pd.DataFrame(result['results'])
    assert set(df.dependency_hash)=={frozen['dependency_hash']}
    assert len(df)==11*(13+51+20+8) and not df.duplicated(['strategy','suite','log','y0','y1']).any()
    assert result['execution']['workers']==4 and result['execution']['threads_per_worker']==1
    table=[]
    for (strategy,suite),group in df.groupby(['strategy','suite'],sort=True):
        sid=strategy.split('_')[0]
        record=dict(strategy=sid,strategy_name=strategy,suite=suite,status=STATUS[sid],chosen=sid==frozen['chosen_strategy'],windows=len(group),scored=int(group.scored.sum()),wells=group.well.nunique(),failures=int(group.failure.sum()),render_failures=int(group.get('render_error',pd.Series(dtype=object)).notna().sum()),cached=int(group.cached.sum()))
        for metric in bench.METRICS:
            values=pd.to_numeric(group[metric],errors='coerce')
            record[metric]=float(values.mean()) if values.notna().any() else None
            record[metric+'_n']=int(values.notna().sum())
        record['peak_rss_gb_max']=float(group.peak_rss_gb.max()) if group.peak_rss_gb.notna().any() else None
        table.append(record)
    failure=df[df.failure].to_dict('records')
    write(HERE/'final_table.json',dict(chosen_strategy=frozen['chosen_strategy'],selection_frozen_utc=frozen['frozen_utc'],run_dir=run_dir.relative_to(BAKEOFF).as_posix(),dependency_hash=frozen['dependency_hash'],execution=result['execution'],table=table,failures=failure,summaries=result['summaries'],historical_contamination=STATUS))
    pd.DataFrame(table).to_csv(HERE/'final_table.csv',index=False)
    shutil.copyfile(run_dir/'results.json',HERE/'final_windows.json')
    shutil.copyfile(run_dir/'results.csv',HERE/'final_windows.csv')
    write(HERE/'failures.json',failure)
    aud=read(HERE/'audit.json')
    lines=['# Wave 3 final results','',f"Report time: {now()}. The choice was fixed at {frozen['frozen_utc']}.",'',
        '**Chosen strategy: S19.** The final scores did not change this choice. No model, setting, threshold, or code was changed after the freeze.','',
        '## Selection evidence','',
        'Selection used the current 51 dev windows from five wells and synthetic columns from non-eval in-domain source logs. Each dev fold excluded the whole query well from both model fits. Training and held synthetic source logs were separate. No real non-eval interval labels are present in this mirror. The synthetic check is a limited substitute, not real-log interval CV.','',
        f"The corrected, fixed S19 decoder gives dev score {frozen['selection']['dev']['score']:.6f} by well and synthetic score {frozen['selection']['synthetic']['score']:.6f} by family. The selection value is {frozen['selection']['objective']:.6f}. It gives dev weight 0.75 and synthetic weight 0.25. S21 reports 0.310140 for the same weighting, with a different synthetic set. These are selection estimates. They are not nested CV or an independent outer-well test.",'',
        'S11 has dev score 0.1041 by well. S22 has dev score 0.1839 by well; its synthetic check measures boundary proposals only. S20 has clean parser CV, but its deployed wrapper loads old S18 weights. It cannot be selected as a clean full system. Different synthetic checks limit a direct rank across lanes. S19 has the strongest permitted full-pipeline selection evidence among the clean deployed candidates.','',
        'The S19 check used the lane settings without a new grid. The bug fixes did not change its CV scores. See `ml/final/selection_recheck.json` and `ml/final/FREEZE.json` for exact values and settings.','',
        'A separate smoke check used two permitted dev windows and S19-S22. All eight predictions passed. Four workers used one compute thread each. Peak total process-tree RSS was 2.301 GiB. This checked execution and memory only; it did not change the choice or any setting. See `ml/final/dev_smoke.log`.','',
        '## Final table','',
        'Each mean gives every window equal weight. This differs from the well means used for selection. Score is row accuracy times boundary F1 for each window. Quality failures count as zero. Depth MAE is in feet and uses matched boundaries with finite depth values only. A dash means no value. The machine table gives the count for each metric. ARI can be negative. The final dev results use full-dev model fits; they are training-set checks, not CV.','',
        'S13, S14, S15, S16, S17, S18 and S20 have historical Forsyth contamination. Their rows are comparisons only. S19, S21 and S22 use clean new weights. S11 has no learned checkpoint. Do not treat shared-verify scores for contaminated models as held-out evidence.','',
        '| Suite | Strategy | Windows | Failures | Row accuracy | Boundary F1 | ARI | Depth MAE ft | Score | Seconds | Mean RSS GiB | Max RSS GiB | API calls |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.6f}'
    for r in table:
        lines.append('| '+ ' | '.join([r['suite'],r['strategy'],str(r['windows']),str(r['failures'])]+[fmt(r[k]) for k in ('row_acc','boundary_f1','cluster_ari','depth_mae_ft','score','seconds','peak_rss_gb','peak_rss_gb_max','api_calls')])+' |')
    lines+=['','The JSON file also contains well-bootstrap 95% confidence limits for all reported means. It records all per-window rows, failures, source hashes, truth hashes and prediction cache keys. A single shared-verify well gives no estimate of between-well variation.','',
        '## Execution and failures','',
        f"The one final invocation wrote `{run_dir.relative_to(BAKEOFF).as_posix()}`. It used four workers and one compute thread per worker. There are {len(df)} result rows, {int(df.failure.sum())} prediction or preprocessing failures, and {sum(r['render_failures'] for r in table)} render failures. Cached predictions: {int(df.cached.sum())}. Peak total process-tree RSS: {result['execution'].get('peak_total_rss_gb', 0):.3f} GiB. All 11 requested strategies were present. No strategy ID was omitted.",'',
        'The source fingerprint includes harness, strategy and ML code, selected settings and deployed model files. Prediction, strategy and preprocessing caches include this fingerprint. Image and annotation hashes identify each source. Depth caches also identify exact depth-track and unit-context pixels. Old prediction keys cannot match this run. Raw parser OCR caches use exact image pixels.','',
        f"Final dependency SHA-256: `{frozen['dependency_hash']}`.",'']
    if failure:
        lines+=['| Strategy | Suite | Log | Error |','|---|---|---|---|']
        for r in failure:
            lines.append('| '+' | '.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ('strategy','suite','log','error'))+' |')
    else:
        lines+=['No prediction or preprocessing failed.']
    resources=HERE/'resources.json'
    if resources.is_file():
        lines+=['',f"Resource monitor: `{json.dumps(read(resources),sort_keys=True)}`."]
    lines+=['','## Review findings and fixes','',
        'Both review files were read before any final evaluation. The detailed resolution is in `notes/wave3_reviews/final_resolution.md`.','',
        '- The checkpoint audit confirms old Forsyth use in S13-S18 and deployed S20. These assets were kept as labelled historical comparisons. New S19, S21 and S22 checkpoint metadata name only the five approved wells. The audit checked current dev identities and synthetic source separation. S19 fold hashes match its fit records. S19 starts its CNNs from random weights and its matchers from new fits. It does not load the old confidence weights.',
        '- Old boundary and matcher caches have unsafe positional identities and remain outside this lane ownership. They were not used for the choice or for new training. The historical S18 prepare and select commands now refuse to resume finalized experiments or reuse unversioned interval caches. A full clean legacy rebuild was not safe within the confirmed-fix scope.',
        '- Benchmark keys now include ML code, settings and weights. Strategy and preprocessing caches use the same digest. Paired comparisons reject source or truth hash differences.',
        '- Depth unit detection now reads “Depth in feet” before unrelated metre text. Equal-support depth fits prefer lower residual before longer span. Pixel content identifies the depth cache. No fixed depth offset or per-log rule was added.',
        '- The runner and scorer reject non-finite and conflicting boundary depths. Legacy quality failures now count as zero in all main quality means. Figure errors do not change valid prediction scores. Truth-read errors cannot trigger a second unguarded read. Both runners sort labeler paths for a fixed consensus tie rule.',
        '- Joint decoding splits at known blank gaps before the duration check. Rule restoration checks each consecutive cut and applies the strong-probability policy separately. Restored rows use the probability at their actual top. The S19 allowed CV check was repeated after these fixes.',
        '- The old S18 “nested” diagnostic is not independent outer-well validation. It was not used for this choice. Earlier final runs exist for inherited designs; the present refits cannot remove that history. The local Emil retrain used six reserved eval images for pseudo labels. Default Drive model provenance is incomplete. S12 was not requested and was not run.','',
        '## Tests','',
        'Commands ran from bakeoff with the project Python environment and CPU thread limits. The logs retain the full output.','',
        '```text',
        'python -m pytest harness/test_bench.py harness/test_depth.py harness/test_run.py harness/test_score.py harness/test_wave3_fixes.py ml/integrate/test_integrate.py -q -o cache_dir=ml/final/.pytest_cache',
        (HERE/'review_tests.log').read_text().splitlines()[-1],
        '',
        'python -m pytest ml/combine/test_combine.py ml/legend_v4/test_legend_v4.py ml/synth_v2/test_synth_v2.py ml/boundary_v2/test_boundary_v2.py -q -o cache_dir=ml/final/.pytest_cache',
        (HERE/'lane_tests.log').read_text().splitlines()[-1],
        '',
        "python -m pytest harness/test_bench.py harness/test_depth.py harness/test_run.py harness/test_score.py harness/test_wave3_fixes.py ml/integrate/test_integrate.py ml/combine/test_combine.py ml/legend_v4/test_legend_v4.py ml/synth_v2/test_synth_v2.py ml/boundary_v2/test_boundary_v2.py -q -k 'not test_equal_support_depth_fit_uses_residual_before_span' -o cache_dir=ml/final/.pytest_cache",
        frozen['final_tests_summary'],'```','',
        'The initial lane failure is an obsolete assertion in another lane: it expects the shared depth fitter to keep the old wrong slope greater than 3 ft per pixel. The fixed fitter now gives about 0.13323. That test file was not changed. The final test command omits that one obsolete assertion. The owned replacement `test_depth_equal_support_prefers_residual_before_span` checks the correct result.','',
        '## Data found and request list','',
        'The benchmark has 13 in-domain eval logs, 51 external dev windows from five wells, 20 external test windows from three wells, and eight Forsyth shared-verify windows from one well. The three sealed wells were not opened, searched for, or rebuilt. The split declaration was read only as metadata. No sealed suite was run.','',
        'Round three has 32 page logs and 32 exact-truth windows with 250 intervals from one Mirror Lake well. Keep these pages in one well group. Round three also supplied 643 unlabelled pages. Round four added eight Oregon scan pages and no exact-truth pages. These sets were not used for model choice or the final table. Current manifest record counts are in `ml/final/audit.json`; images were not scored.', '',
        'BSEE pilot request list: `notes/bsee_list/pilot_request.csv`. It lists 60 files from 20 API well groups, about 1.49 GB. The full list is `notes/bsee_list/request_documents.csv`, with 959 files from 126 groups, about 16.04 GB. The owner instructions are in `notes/bsee_list/REPORT.md`. No request was sent.','',
        '## Files','',
        '- `ml/final/final_table.json` and `final_table.csv`: complete suite means, metric counts and failures.',
        '- `ml/final/final_windows.json` and `final_windows.csv`: every benchmark window and source identity.',
        '- `ml/final/FREEZE.json`: the choice, selection evidence and code/model fingerprints before final scoring.',
        '- `ml/final/audit.json`: checkpoint and data audit.',
        f"- `{run_dir.relative_to(BAKEOFF).as_posix()}/report.html`: the flat benchmark report with source, predictions, truth and legends.",'',
        'No services were changed. No child agent was started. No commit or push was made.']
    (BAKEOFF/'notes/wave3_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(HERE/'FINAL_DONE.json',dict(completed_utc=now(),run_dir=run_dir.relative_to(BAKEOFF).as_posix(),chosen_strategy=frozen['chosen_strategy'],rows=len(df),failures=int(df.failure.sum()),dependency_hash=frozen['dependency_hash']))
    print('FINAL:',len(table),'table rows;',len(df),'windows;',int(df.failure.sum()),'failures; choice',frozen['chosen_strategy'])

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['audit','freeze','report'])
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.command=='audit':audit()
    elif args.command=='freeze':freeze()
    else:report(args.run_dir)
