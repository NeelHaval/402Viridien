"""Exercise the live notebook experiment API, validated records and real evaluators."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from emil.bootstrap import installed_arrow
installed_arrow()
from emil import pipeline as p


def main():
    p.torch.set_num_threads(6)
    stem='60811405120000_1inch tvd mudlog'
    p.configure(labels=p.WORK/'emil/labels',model=p.DEFAULT_MODEL)
    result=p.run_experiment('smoke_detect',legend_source='detect',stems=[stem])
    assert len(result)==1 and result.iloc[0]['stem']==stem
    p.check_run([stem])
    rec=json.loads(p.record_path(stem).read_text())
    assert not p.record_errors(rec)
    assert rec['written_by']['experiment']['legend_source']=='detect'
    assert rec['legend'] and all('composition' in l for l in rec['layers'])
    expected=json.loads((HERE/'results_live/viridien/detect/classification.json').read_text())['per_file'][stem+'.json']['pixel_accuracy']
    actual=float(result.iloc[0]['pixel_accuracy'])
    assert abs(actual-expected)<.01,(actual,expected)
    data=dict(status='passed',api='run_experiment',legend_source='detect',model_sha256=hashlib.sha256(p.MODEL_PATH.read_bytes()).hexdigest(),
              n_legend=len(rec['legend']),n_layers=len(rec['layers']),pixel_accuracy=actual,
              reference_pixel_accuracy=expected,measurements=result.to_dict('records'))
    (HERE/'smoke_live.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data,indent=2),flush=True)


if __name__=='__main__':main()
