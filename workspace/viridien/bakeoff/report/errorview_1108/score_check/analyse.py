"""Read saved predictions; verify the complete cohort and compare the real example."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
BAKE = WORK / 'bakeoff'
PROJECT = WORK.parent / 'viridien_challenge/Viridien_HTB_challenge/hack27/project'
sys.path.insert(0, str(BAKE))
from harness import io, score
from strategies import challenge_export, grading_policy
sys.path.insert(0, str(WORK.parent.parent/'viridian402-errorview/src'))
from viridian402.errorview_data import build_case


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


preview_path = BAKE/'report/scoreboard_1108/accepted_preview.json'
preview = read(preview_path)
cohort = [r for r in preview['records'] if r['suite'] == 'viridien_eval']
assert len(cohort) == 52
strategies = sorted({r['strategy'] for r in cohort})
assert len(strategies) == 4
summaries = []
verified = []
for strategy in strategies:
    rows = [r for r in cohort if r['strategy'] == strategy]
    assert len(rows) == 13 and len({r['log'] for r in rows}) == 13
    computed = []
    for r in rows:
        pred, truth = io.load_rows(r['prediction_csv']), io.load_rows(r['truth_csv'])
        actual = grading_policy.score_equivalent(pred, truth, r['window'])
        if r.get('error'):
            actual = dict(score=0., row_acc=0., boundary_f1=0., cluster_ari=0., depth_mae_ft=None)
        for key in ('row_acc','boundary_f1','score','cluster_ari','depth_mae_ft'):
            a,b = actual[key], r['accepted_metrics'][key]
            assert a == b if a is None or b is None else abs(a-b)<1e-12
        computed.append(actual)
        verified.append({'case_id':r['case_id'],'strategy':strategy,'prediction_sha256':sha(Path(r['prediction_csv'])),
                         'truth_sha256':sha(Path(r['truth_csv'])),'metrics':actual})
    metrics = {k:sum(m[k] for m in computed)/len(computed) for k in ('row_acc','boundary_f1','score','cluster_ari')}
    saved = next(s for s in preview['accepted_summaries'] if s['suite']=='viridien_eval' and s['strategy']==strategy and s['family']=='all')
    assert all(abs(metrics[k]-saved['metrics'][k])<1e-12 for k in metrics)
    summaries.append({'strategy':strategy,'logs':13,'metrics':metrics,'failures':sum(bool(r.get('error')) for r in rows)})

expected_path = PROJECT/'example/example.csv'
expected = io.load_rows(expected_path)
window = [min(r['top_y'] for r in expected),max(r['bottom_y'] for r in expected)]
exported = []
for strategy in ('s20_v4','s21_v4dino','s23_v4dino_identity'):
    record = next(r for r in cohort if r['strategy']==strategy and 'log06' in r['log'])
    target = HERE/strategy
    manifest = challenge_export.export(record['prediction_csv'], target, image_height=12050)
    generated = target/manifest['notebook']['file']
    pred = io.load_rows(generated)
    with expected_path.open(encoding='utf-8-sig',newline='') as f: expected_header=next(csv.reader(f))
    with generated.open(encoding='utf-8-sig',newline='') as f: generated_header=next(csv.reader(f))
    assert generated_header == expected_header
    assert pred == io.load_rows(record['prediction_csv'])
    strict = score.score(pred,expected,window)
    accepted = grading_policy.score_equivalent(pred,expected,window)
    full = grading_policy.score_equivalent(pred,expected,record['window'])
    view = build_case(expected,pred,window,scorer=score)
    for run in view['runs']:
        a = score.normalize(grading_policy.canonical_grading_label(run['truth']))
        b = score.normalize(grading_policy.canonical_grading_label(run['prediction']))
        if a == b and a:
            run['kind'] = 'correct'
    examples = []
    for truth in expected[:4]:
        overlaps = [{**p,'overlap_rows':min(p['bottom_y'],truth['bottom_y'])-max(p['top_y'],truth['top_y'])}
                    for p in pred if p['top_y']<truth['bottom_y'] and p['bottom_y']>truth['top_y']]
        examples.append({'expected':truth,'predicted_overlaps':overlaps})
    exported.append({'strategy':strategy,'csv':str(generated),'slide_csv':str(target/manifest['slide']['file']),
                     'predicted_beds':len(pred),'expected_beds':len(expected),'window':window,
                     'metrics':accepted,'strict_metrics':strict,'column_window':record['window'],'column_metrics':full,
                     'overlap_examples':examples,'runs':view['runs']})
shutil.copyfile(expected_path,HERE/'expected.csv')
output={'policy':grading_policy.POLICY_ID,'source_preview':str(preview_path),'source_sha256':sha(preview_path),
        'cohort':'13 Viridien development logs, all four strategies; fixed complete preview cohort',
        'summaries':summaries,'expected_csv':str(expected_path),'exports':exported,'verified_cases':verified,
        'limitations':['This is a local score, not an official challenge grade.',
                       'The complete all-suite evaluation is still pending. No held-out accuracy claim.',
                       'Row accuracy ignores unlabelled truth rows. Clay/Shale equivalence is owner configured.',
                       'CSV exports replay saved inference outputs; inference was not rerun.']}
(HERE/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
print('Cohort check: 52 saved predictions match accepted scores')
print('CSV check: 3 exports match example headers and round-trip source predictions')
for row in summaries: print(row['strategy'],{k:round(v*100,2) if k!='cluster_ari' else round(v,4) for k,v in row['metrics'].items()})
for row in exported: print('EXAMPLE',row['strategy'],row['expected_beds'],row['predicted_beds'],row['metrics'])
print('Checks passed: 52 score matches; 3 CSV header checks and source round-trips')
