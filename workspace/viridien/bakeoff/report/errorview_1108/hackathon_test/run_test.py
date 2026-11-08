"""Hypothetical demo rubric on a fixed 13-log cohort; no inference or source edits."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
BAKE = WORK/'bakeoff'
sys.path.insert(0,str(BAKE))
from harness import io
from strategies import grading_policy, challenge_export

# Declared before inspecting the new results. This is not an official rubric.
POLICY = dict(label_weight=.7, boundary_weight=.3, min_layer_px=25,
              class_rule=grading_policy.POLICY_ID, csv_gate='nonempty, valid challenge fields and finite depths',
              official=False, aggregation='equal-log mean; same 13 logs for all candidates',
              rationale='Additive credit for labels and contacts; optional removal of very thin layers.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(pred,truth,window,min_layer_px,failed=False):
    if failed:
        return {'row_acc':0.,'boundary_f1':0.,'weighted':0.}
    m=grading_policy.score_equivalent(pred,truth,window,min_layer_px=min_layer_px)
    return {'row_acc':m['row_acc'],'boundary_f1':m['boundary_f1'],
            'weighted':POLICY['label_weight']*m['row_acc']+POLICY['boundary_weight']*m['boundary_f1']}


def self_check():
    truth=[dict(top_y=0,bottom_y=100,lithology='Shale',cluster_id=0)]
    same=[dict(top_y=0,bottom_y=100,lithology='Clay',cluster_id=9)]
    wrong=[dict(top_y=0,bottom_y=100,lithology='Sandstone',cluster_id=0)]
    assert assess(same,truth,[0,100],0)['weighted']==1
    # Correct lack of internal contacts gets 30%; wrong rock gets no label credit.
    assert assess(wrong,truth,[0,100],0)['weighted']==.3
    assert assess(same,truth,[0,100],0,failed=True)['weighted']==0
    assert truth[0]['lithology']=='Shale' and same[0]['lithology']=='Clay'


def main():
    self_check()
    fixed=json.loads((HERE.parent/'score_check/analysis.json').read_text(encoding='utf-8'))
    preview=json.loads(Path(fixed['source_preview']).read_text(encoding='utf-8'))
    observed={(r['case_id'],r['strategy']):r for r in preview['records']}
    cases=[]
    for evidence in fixed['verified_cases']:
        r=observed[evidence['case_id'],evidence['strategy']]
        assert r['suite']=='viridien_eval'
        assert sha(r['prediction_csv'])==evidence['prediction_sha256']
        assert sha(r['truth_csv'])==evidence['truth_sha256']
        pred,truth=io.load_rows(r['prediction_csv']),io.load_rows(r['truth_csv'])
        try:
            checked=challenge_export.read_rows(r['prediction_csv'])
            valid=bool(checked) and not r.get('error')
        except (ValueError,OSError):
            valid=False
        cases.append(dict(log=r['log'],strategy=r['strategy'],case_id=r['case_id'],
                          csv_valid=valid,failed=bool(r.get('error')),
                          standard=evidence['metrics'],
                          weighted=assess(pred,truth,r['window'],0,bool(r.get('error'))),
                          demo=assess(pred,truth,r['window'],25,bool(r.get('error')))))
    summary=[]
    for strategy in sorted({c['strategy'] for c in cases}):
        rr=[c for c in cases if c['strategy']==strategy]
        assert len(rr)==13 and len({c['log'] for c in rr})==13
        mean=lambda path:sum(c[path[0]][path[1]] for c in rr)/len(rr)
        summary.append(dict(strategy=strategy,logs=13,csv_valid=sum(c['csv_valid'] for c in rr),
                            failures=sum(c['failed'] for c in rr),
                            standard_product=mean(('standard','score')),
                            additive_no_merge=mean(('weighted','weighted')),
                            demo_score=mean(('demo','weighted')),
                            demo_labels=mean(('demo','row_acc')),demo_boundaries=mean(('demo','boundary_f1'))))
    output={'policy':POLICY,'summary':summary,'cases':cases,
            'limits':['Hypothetical quality rubric, not predicted judging marks.',
                      '25 pixels has a different physical thickness on different logs.',
                      'Thin-bed merging can hide real beds and can also reduce a score.',
                      'CSV validity is a separate gate, not free quality points.',
                      'Depth accuracy, clusters, full-document automation and runtime require separate review.',
                      'Development examples; no unseen-test accuracy claim.']}
    (HERE/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for row in summary:
        print(row['strategy'],'standard',round(row['standard_product']*100,1),
              'additive',round(row['additive_no_merge']*100,1),'demo',round(row['demo_score']*100,1),
              'CSV',str(row['csv_valid'])+'/13')
    print('Checks passed: 4 rubric assertions; 52 source hashes and same-cohort records verified')


if __name__=='__main__':
    main()
