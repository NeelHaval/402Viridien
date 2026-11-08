"""Validate available cohorts, provenance, split means and score serialization."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from emil import pipeline as p
from emil.split_results import membership, selected
from harness import io, score
from harness.run import aggregate


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sha=digest(p.DEFAULT_MODEL)
    windows=read(p.WORK/'external_tests/eval_set.json')
    expected={(w['log'],(w['y0'],w['y1'])) for w in windows}
    assert len(expected)==79
    lookup=membership()
    assert all(lookup[name] in ['dev','test'] for name,win in expected)
    n_records=0; counts={};compared=0;recomputed_s11=0;rejections={}
    for domain,n,modes in [('viridien',2,['detect','labels','s11']),('external',79,['detect','labels','s11'])]:
        root=HERE/'results_live'/domain;rr=read(root/'records.json')
        assert len(rr)==n*len(modes) and len({(r['key'],r['mode']) for r in rr})==len(rr)
        assert not any('error' in r for r in rr)
        for mode in modes:
            mm=[r for r in rr if r['mode']==mode];assert len(mm)==n
            if domain=='external':assert {(r['log'],tuple(r['window'])) for r in mm}==expected
            assert {f.stem for f in (root/mode/'predictions').glob('*.json')}=={r['key'] for r in mm}
            for r in mm:
                if mode!='s11':assert r['checkpoint_sha256']==sha
                rec=read(root/mode/'predictions'/(r['key']+'.json'))
                assert not p.record_errors(rec)
                assert rec['stem']==r['key'] and rec['written_by']['local_evaluation']['legend_source']==mode
                rows=io.load_rows(root/mode/'csv'/(r['key']+'.csv'))
                truth=io.load_rows(root/'truth_csv'/(r['key']+'.csv'))
                actual=score.score(rows,truth,tuple(r['window']))
                assert all(actual[k] is None and r[k] is None or actual[k] is not None and abs(actual[k]-r[k])<1e-12 for k in ['row_acc','boundary_f1','cluster_ari'])
                if mode=='s11':
                    prior=io.load_rows(HERE/'results'/domain/'s11/csv'/(r['key']+'.csv'))
                    keys=['top_y','bottom_y','lithology','cluster_id']
                    if r.get('prediction_source','').startswith('s11.run'):
                        assert not prior
                        recomputed_s11+=1
                    else:
                        assert [[x[k] for k in keys] for x in rows]==[[x[k] for k in keys] for x in prior]
                        compared+=1
                n_records+=1
            splits=['dev','test','dev_test'] if domain=='external' else [None]
            for split in splits:
                result=read(root/split/mode/'metrics.json') if split else read(root/mode/'metrics.json')
                group=selected(mm,split) if split else mm
                assert result['harness']['n_windows']==len(group)
                assert result['harness']['crashes']==0
                for kind in ['segmentation','classification','legend']:
                    evaluator=result['emil'][kind]
                    failed=evaluator.get('failed',{})
                    assert len(evaluator['per_file'])+len(failed)==len(group)
                    assert set(evaluator['per_file'])|set(failed)=={r['key']+'.json' for r in group}
                    for filename,error in failed.items():
                        assert 'prediction has no valid' in error
                        assert not read(root/mode/'predictions'/filename)['layers']
                    rejections[f'{domain}/{split or "all"}/{mode}/{kind}']=len(failed)
        counts[domain]=dict(windows=n,modes=modes)
    for strategy in ['s11_fft_legend','s12_emil','oracle_emil']:
        data=read(io.ROOT/'out'/strategy/'external/metrics.json')
        assert len(data['logs'])==79
        assert {(r['log'],tuple(r['window'])) for r in data['logs']}==expected
        for split,n in [('dev',59),('test',20),('dev_test',79)]:
            mm=aggregate(selected(data['logs'],split)); assert mm==data['splits'][split]
            assert mm['n_scored']==n
        assert data['splits']['dev_test']['crashes']==5
    resources=read(HERE/'vm_resources.json')
    assert len(resources)==162 and all(r['matches_saved_predictions'] for r in resources)
    assert all(r['device']=='cpu' and r['peak_gpu_gb']==0 and not r['traced'] for r in resources)
    assert read(HERE/'smoke_live.json')['status']=='passed'
    secondary=read(HERE/'results_retrained_live/viridien/records.json');secondary_sha=digest(HERE/'boundary_net.pt')
    assert len(secondary)==4 and all('error' not in r and r['checkpoint_sha256']==secondary_sha for r in secondary)
    port=read(HERE/'port_manifest.json')
    for kind,sha_e in port['evaluators'].items():assert digest(HERE/'evaluators'/(kind+'.py'))==sha_e
    validation=read(HERE/'validation.json');assert validation['main_checkpoint_sha256']==sha
    incomplete=read(HERE/'results_live/incomplete/records.json')
    assert len(incomplete)==2 and all(r['incomplete'] for r in incomplete)
    assert next(r for r in incomplete if r['mode']=='labels').get('error') is None
    tests=(HERE/'tests_live.log').read_text();harness_tests=(HERE/'tests_harness_vm.log').read_text()
    assert 'failed' not in harness_tests and 'passed' in harness_tests
    result=dict(data_acceptance='passed',complete_cohorts=counts,external_splits=dict(dev=59,test=20,dev_test=79),
        validated_prediction_records=n_records,s11_csvs_unchanged=compared,s11_partial_csvs_recomputed=recomputed_s11,
        team_evaluator_empty_prediction_rejections=rejections,main_checkpoint_sha256=sha,
        secondary_checkpoint_sha256=secondary_sha,cpu_replay_exact_matches=162,all_evaluators_unchanged=True,
        external_shared_preprocessing_failures=5,incomplete_excluded=True,
        experiment_api_smoke='passed',tests_harness=harness_tests.strip().splitlines()[-1],
        tests_full=tests.strip().splitlines()[-1],full_test_suite_green=False,
        environment_limits=['External tests lack original dev PDF 386825.pdf; no replacement reconstructed.',
            'Existing metric-PDF robustness test hits unchanged converter 160-million-pixel render guard.',
            'Uncached gc743 detect unavailable: EasyOCR model files absent, download DNS unavailable.'],
        code_sha256={name:digest(p.WORK/name) for name in ['bakeoff/emil/pipeline.py','bakeoff/strategies/s12_emil.py','bakeoff/strategies/s11_fft_legend.py']})
    (HERE/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
