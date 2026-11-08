"""Finish one inference interrupted by a telemetry file access error."""
import importlib.util,json,os,shutil,sys
from datetime import datetime
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('score_retry',H/'score_all.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec2=importlib.util.spec_from_file_location('bounded_retry',H/'bounded_local.py');b=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(b)

def main():
    if m.read(H/'formal_shenzi_complete.json').get('complete') is not True:raise RuntimeError('Formal Shenzi must complete first')
    receipt=json.loads((H/'formal_controller_receipt.json').read_text(encoding='utf-8-sig'))
    try:
        p=psutil.Process(receipt['pid']);born=datetime.fromisoformat(receipt['creation_utc']).timestamp()
        if abs(p.create_time()-born)<1 and p.is_running():raise RuntimeError('Formal controller is still active')
    except psutil.NoSuchProcess:pass
    for event in m.read(m.OUT/'shenzi_formal_supervisor'/'controller_receipt.json')['events']:
        try:
            p=psutil.Process(event['pid']);born=datetime.fromisoformat(event['worker_started_utc']).timestamp()
            if abs(p.create_time()-born)<2 and p.is_running():raise RuntimeError('Formal child is still active')
        except psutil.NoSuchProcess:pass
    manifest=m.read(H/'manifest_1108.json');path=H/'raw_results.json';raw=m.read(path);m.validate_records(raw['records'],manifest)
    failed=[r for r in raw['records'] if r['strategy']=='s19_combine' and r['log']=='ext_nl_storm_p081' and 'active_local_attempt.json.tmp' in r.get('error','')]
    if len(failed)!=1:raise RuntimeError('Exactly one preserved telemetry failure required')
    old=failed[0];key=(old['case_id'],old['strategy']);history=H/'attempt_history';history.mkdir(exist_ok=True)
    m.write(history/'storm_p081_telemetry_result.json',old);m.write(history/'checkpoint_before_telemetry_retry.json',raw)
    shutil.copyfile(old['prediction_csv'],history/'storm_p081_telemetry_prediction.csv')
    raw['records']=[r for r in raw['records'] if (r['case_id'],r['strategy'])!=key];m.write(path,raw)
    progress=H/'telemetry_retry_active.json';case=next(c for c in manifest['cases'] if c['case_id']==key[0]);digest=m.manifest_digest(manifest)
    def completed():return {(r['case_id'],r['strategy']) for r in m.read(path)['records']}
    def timeout(attempt,stop):
        current=m.read(path)
        if key in completed():return
        prediction=m.OUT/key[1]/(key[0]+'.csv');m.io.save_rows(prediction,[])
        row={**case,'strategy':key[1],'scored':True,'manifest_digest':digest,'prediction_csv':str(prediction),'score':0.,'row_acc':0.,'boundary_f1':0.,'cluster_ari':0.,'depth_mae_ft':None,'error':'OperationalTimeout: telemetry-failure continuation reached600s whole-attempt cap','failure_phase':attempt['phase'],'operational_timeout':True,'worker_terminated':True,'seconds':None,'preprocess_seconds':None,'end_to_end_seconds':None,'timeout_receipt':stop,'recorded_utc':b.utc()}
        if attempt.get('initial_shared_depth_calibration'):row['initial_shared_depth_calibration']=attempt['initial_shared_depth_calibration']
        current['records'].append(row);m.write(path,current)
    command=lambda:[sys.executable,'-X','utf8',str(H/'score_all.py'),'--resume','--case-id',key[0],'--strategies',key[1],'--progress-file',str(progress)]
    os.environ['PYTHONPATH']=str(m.B);os.environ['HF_HUB_OFFLINE']='1'
    result=b.supervise({key},completed,command,progress,timeout,m.OUT/'telemetry_retry_supervisor',cap=600)
    current=m.read(path)
    for row in current['records']:
        if (row['case_id'],row['strategy'])==key:row['initial_telemetry_failure']=str(history/'storm_p081_telemetry_result.json');row['bounded_operational_continuation']=True
    m.write(path,current);m.write(H/'telemetry_retry_complete.json',result)
    print('TELEMETRY CONTINUATION COMPLETE: preserved initial attempt; exact full-window inference retained')
if __name__=='__main__':main()
