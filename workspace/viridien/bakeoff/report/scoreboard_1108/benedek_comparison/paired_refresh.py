"""Fetch a checkpoint and record completed, failed and unprocessed model cases."""
import argparse,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[2]
p=argparse.ArgumentParser();p.add_argument('--job',choices=['first3','newtest'],required=True);a=p.parse_args()
name='benedek-'+a.job+'-1010';remote='/home/user/viridian402-jobs/v4-cpu-20261108T063000Z-f785e5/'+name
out=B/'out'/('benedek_compare_'+a.job)
options=['-i','C:/Users/hssil/.ssh/ascii_box_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
def get(file,target):subprocess.run(['scp','-P','19036',*options,'user@213.32.26.26:'+remote+'/'+file,str(target)],check=True,timeout=60)
get('predictions.json',out/'predictions_remote.json');get('*.csv',out)
cmd='cd '+remote+' && cat queue.state && (cat worker.exit 2>/dev/null || true)'
state=subprocess.check_output(['ssh','-p','19036',*options,'user@213.32.26.26',cmd],text=True,timeout=20).splitlines()
x=json.loads((out/'predictions_remote.json').read_text(encoding='utf-8'));m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
key=lambda r:(r['case_id'],r.get('strategy','s24_benedek'))
done={key(r) for r in x['records']};missing=[dict(case_id=r['case_id'],strategy=r.get('strategy','s24_benedek')) for r in m['cases'] if key(r) not in done]
receipt=dict(remote=remote,observed_state=state[0],worker_exit=int(state[1]) if len(state)>1 else None,completed_model_cases=len(done),expected_model_cases=len(m['cases']),unprocessed=missing,timed_out=state[1:]==['124'],accuracy_available=a.job=='first3',truth_supplied_to_worker=False,compute_timeout_seconds=300,queue_timeout_seconds=60,outer_timeout_seconds=380)
(out/'queue_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
for r in x['records']:r['prediction_csv']=str(out/Path(r['prediction_csv']).name)
(out/'predictions_local.json').write_text(json.dumps(x,indent=2),encoding='utf-8')
print('CHECKPOINT_SAVED state='+state[0]+' complete='+str(len(done))+'/'+str(len(m['cases']))+' errors='+str(sum(bool(r.get('error')) for r in x['records'])))
