"""One uniform-cap recheck of the initial operator-interrupted full Shenzi case."""
import importlib.util,json,os,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('score_all_formal',H/'score_all.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec2=importlib.util.spec_from_file_location('bounded_formal',H/'bounded_local.py');b=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(b)
KEY=('ae5c647135c45e6b','s19_combine')
def main():
    receipt=json.loads((H/'bounded_controller_receipt.json').read_text(encoding='utf-8-sig'))
    try:
        p=psutil.Process(receipt['pid']);born=datetime.fromisoformat(receipt['creation_utc'].replace('Z','+00:00')).timestamp()
        if abs(p.create_time()-born)<1 and p.is_running():raise RuntimeError('Main local controller is active; formal recheck blocked')
    except psutil.NoSuchProcess:pass
    completed_supervisor=m.read(H/'supervisor_complete.json')
    if completed_supervisor.get('complete') is not True:raise RuntimeError('Main local supervisor is not complete')
    worker_receipt=m.read(m.OUT/'supervisor'/'controller_receipt.json')
    for event in worker_receipt['events']:
        try:
            child=psutil.Process(event['pid']);born=datetime.fromisoformat(event['worker_started_utc']).timestamp()
            if abs(child.create_time()-born)<2 and child.is_running():raise RuntimeError('Owned warm worker is still active; formal recheck blocked')
        except psutil.NoSuchProcess:pass
    manifest=m.read(H/'manifest_1108.json');path=H/'raw_results.json';raw=m.read(path);m.validate_records(raw['records'],manifest)
    planned={(c['case_id'],s) for c in manifest['cases'] for s in ('s19_combine','s21_v4dino','s23_v4dino_identity')}
    actual=[(r['case_id'],r['strategy']) for r in raw['records']]
    if len(actual)!=len(set(actual)) or not planned.issubset(actual) or len(planned)!=204:raise RuntimeError('Exact204 local candidate keys must be durable before formal recheck')
    if len(actual)!=214 or len([key for key in actual if key[1]=='s20_v4'])!=10:raise RuntimeError('Retained ten S20 records must remain intact')
    old=next(r for r in raw['records'] if (r['case_id'],r['strategy'])==KEY)
    if not old.get('operational_timeout') or 'checkpoint_to_stop_wall_interval' not in old:raise RuntimeError('Primary is not the initial operator-interrupted case; do not rerun')
    history=H/'attempt_history';history.mkdir(parents=True,exist_ok=True);m.write(history/'shenzi_initial_manual_result.json',old)
    old_csv=Path(old['prediction_csv']);shutil.copyfile(old_csv,history/'shenzi_initial_manual_prediction.csv')
    m.write(history/'checkpoint_before_formal_shenzi.json',raw)
    raw['records']=[r for r in raw['records'] if (r['case_id'],r['strategy'])!=KEY];m.write(path,raw)
    progress=H/'formal_shenzi_active.json';case=next(c for c in manifest['cases'] if c['case_id']==KEY[0]);digest=m.manifest_digest(manifest)
    def completed():return {(r['case_id'],r['strategy']) for r in m.read(path)['records']}
    def timeout(attempt,stop):
        current=m.read(path)
        if KEY in completed():return
        prediction=m.OUT/KEY[1]/(KEY[0]+'.csv');m.io.save_rows(prediction,[])
        row={**case,'strategy':KEY[1],'scored':True,'manifest_digest':digest,'prediction_csv':str(prediction),'score':0.,'row_acc':0.,'boundary_f1':0.,'cluster_ari':0.,'depth_mae_ft':None,
            'error':'OperationalTimeout: formal full-window Shenzi S19 attempt reached the uniform600s whole-attempt cap','failure_phase':attempt['phase'],'operational_timeout':True,'worker_terminated':True,'seconds':None,'preprocess_seconds':None,'end_to_end_seconds':None,
            'attempt_id':attempt['attempt_id'],'attempt_started_utc':attempt['attempt_started_utc'],'timeout_receipt':stop,'recorded_utc':b.utc(),'initial_manual_attempt':str(history/'shenzi_initial_manual_result.json'),
            'runtime_scope':'Formal uniform600s whole-attempt cap. Inference did not return; seconds remain unknown. Full original window retained.'}
        if attempt.get('initial_shared_depth_calibration'):row['initial_shared_depth_calibration']=attempt['initial_shared_depth_calibration']
        current['records'].append(row);m.write(path,current)
    command=lambda:[sys.executable,'-X','utf8',str(H/'score_all.py'),'--resume','--case-id',KEY[0],'--strategies',KEY[1],'--progress-file',str(progress)]
    os.environ['PYTHONPATH']=str(m.B);os.environ['HF_HUB_OFFLINE']='1'
    result=b.supervise({KEY},completed,command,progress,timeout,m.OUT/'shenzi_formal_supervisor',cap=600)
    current=m.read(path)
    for row in current['records']:
        if (row['case_id'],row['strategy'])==KEY:row['initial_manual_attempt']=str(history/'shenzi_initial_manual_result.json');row['formal_uniform_cap_recheck']=True
    m.write(path,current);m.write(H/'formal_shenzi_complete.json',result)
    print('FORMAL SHENZI COMPLETE: one full-window primary result, original manual attempt preserved')
if __name__=='__main__':main()
