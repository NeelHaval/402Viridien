"""Check the new identity guard against saved real prediction CSVs."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parents[1]))
from harness import io
from strategies import scoreboard_v4
from strategies.legend_identity import validate_prediction_labels

records=json.loads((R/'raw_results.json').read_text(encoding='utf-8'))['records']
checked=[]
for rec in records:
    if rec['strategy']!='s23_v4dino_identity' or rec.get('error'):
        continue
    classifier=scoreboard_v4.make_classifier(SimpleNamespace(name=rec['log']),{})
    captions=[label for label,_ in classifier.legend(rec['log'])]
    validate_prediction_labels(io.load_rows(Path(rec['prediction_csv'])),captions)
    checked.append(rec['case_id'])
baseline=next(r for r in records if r['strategy']=='s21_v4dino' and r['suite']=='viridien_eval' and 'log06' in r['log'])
classifier=scoreboard_v4.make_classifier(SimpleNamespace(name=baseline['log']),{})
validate_prediction_labels(io.load_rows(Path(baseline['prediction_csv'])),[label for label,_ in classifier.legend(baseline['log'])])
result=dict(corrected_records_checked=len(checked),case_ids=checked,baseline_family_accepted=True,
            grading_policy='clay_shale_equivalent_v1',
            note='Owner-approved Clay/Shale family is valid. Raw labels and contacts are preserved.')
(R/'identity_output_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(f'Identity guard: {len(checked)} distinct-label files accepted; baseline log06 Clay/Shale family accepted')
