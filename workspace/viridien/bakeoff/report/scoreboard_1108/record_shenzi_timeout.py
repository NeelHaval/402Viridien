from pathlib import Path
from datetime import datetime,timezone
import csv,json
H=Path(__file__).resolve().parent;B=H.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
raw=read(H/'raw_results.json');manifest=read(H/'manifest_1108.json')
assert len(raw['records'])==52,'Checkpoint advanced unexpectedly'
c=next(c for c in manifest['cases'] if c['case_id']=='ae5c647135c45e6b')
assert not any(r['case_id']==c['case_id'] and r['strategy']=='s19_combine' for r in raw['records'])
now=datetime.now(timezone.utc);last=raw['records'][-1]['recorded_utc'];prev=datetime.fromisoformat(last.replace('Z','+00:00'))
cache=read(B/'out'/'_cache'/'depth'/(c['log']+'.rapidocr.json'))
p=B/'out'/'scoreboard_1108'/'s19_combine'/(c['case_id']+'.csv');p.parent.mkdir(parents=True,exist_ok=True)
with p.open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerow(['top_y','bottom_y','lithology','cluster_id','top_depth','bottom_depth'])
r={**c,'strategy':'s19_combine','scored':True,'manifest_digest':raw['manifest_digest'],'prediction_csv':str(p),'score':0.,'row_acc':0.,'boundary_f1':0.,'cluster_ari':0.,'depth_mae_ft':None,
 'error':'OperationalTimeout: owner stopped full-window S19 Shenzi PELT after the saved checkpoint stalled; no intervals were returned','failure_phase':'strategy_run','operational_timeout':True,'worker_terminated':True,'seconds':None,'legend_seconds':None,'depth_seconds':None,'preprocess_seconds':None,'end_to_end_seconds':None,'over_budget_60s':None,
 'timeout_scope':'Initial manual stop had no START receipt. Exact inference seconds and exact stop timestamp are unknown; no duration is invented. The continued run uses a 600 s whole-attempt cap.',
 'checkpoint_to_stop_wall_interval':dict(checkpoint_utc=last,stop_utc=None,stop_confirmed_utc=now.isoformat(),elapsed_upper_bound_seconds=(now-prev).total_seconds()),
 'initial_shared_depth_calibration':cache['calibration'],'initial_context_evidence':'Read from the successful shared depth cache after stop; VM agent separately validated exact depth-pixel signature. No OCR rerun.',
 'stack_evidence':str(B/'out'/'scoreboard_worker_stack_0703.txt'),'process_id':42828,'process_creation_utc':'2026-11-08T06:47:04.1399740Z','recorded_utc':now.isoformat(),'api_calls':0}
raw['records'].append(r);(H/'raw_results.json').write_text(json.dumps(raw,indent=2,allow_nan=False),encoding='utf-8')
(H/'shenzi_timeout_receipt.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print('TIMEOUT RECORDED: checkpoint52 preserved, full Shenzi S19 failed at zero; exact inference seconds UNKNOWN; records53')
