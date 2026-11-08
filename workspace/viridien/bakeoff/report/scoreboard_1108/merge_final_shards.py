"""Merge only complete, identity-checked local and remote strict predictions."""
import hashlib,importlib.util,json,time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import psutil
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('score_all_merge',HERE/'score_all.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from harness import io,score

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def record_error(record,message):raise RuntimeError(record['case_id']+': '+message)
def main():
    for receipt_path in (HERE/'local_remaining_receipt.json',HERE/'bounded_controller_receipt.json',HERE/'formal_controller_receipt.json',HERE/'telemetry_retry_controller_receipt.json'):
        if not receipt_path.exists():continue
        receipt=read(receipt_path)
        try:
            proc=psutil.Process(receipt['pid']);born=datetime.fromisoformat(receipt['creation_utc'].replace('Z','+00:00')).timestamp()
            if abs(proc.create_time()-born)<1 and proc.is_running():raise RuntimeError('Owned local writer/controller is still active; merge is blocked')
        except psutil.NoSuchProcess:pass
    for name in ('supervisor','shenzi_formal_supervisor','telemetry_retry_supervisor'):
        receipt_path=m.OUT/name/'controller_receipt.json'
        if not receipt_path.exists():raise RuntimeError('Missing completed owned worker receipt '+name)
        for event in read(receipt_path)['events']:
            try:
                proc=psutil.Process(event['pid']);born=datetime.fromisoformat(event['worker_started_utc']).timestamp()
                if abs(proc.create_time()-born)<2 and proc.is_running():raise RuntimeError('Owned child is still active; merge is blocked')
            except psutil.NoSuchProcess:pass
    for name in ('supervisor_complete.json','formal_shenzi_complete.json','telemetry_retry_complete.json'):
        if read(HERE/name).get('complete') is not True:raise RuntimeError('Incomplete owned controller '+name)
    manifest=m.read(HERE/'manifest_1108.json');local=m.read(HERE/'raw_results.json');remote=m.read(HERE/'remote'/'remote_results_shard.json')
    digest=m.manifest_digest(manifest)
    m.validate_records(local['records'],manifest)
    if remote.get('manifest_digest')!=digest:raise RuntimeError('Remote shard manifest digest mismatch')
    cases={c['case_id']:c for c in manifest['cases']};records=list(local['records']);seen={(r['case_id'],r['strategy']) for r in records};proof=[]
    pinned={p.replace(chr(92),'/'):h for p,h in manifest['source_hashes'].items()}
    required=['strategies/s20_v4.py','strategies/scoreboard_v4.py','harness/io.py','harness/score.py']+[f'strategies/scoreboard_v4_snapshot/{n}' for n in ('beds.py','legend_classify.py','native_swatches.pkl','log_scale.json')]
    for original in remote['records']:
        row=dict(original);key=(row['case_id'],row['strategy'])
        if row['strategy']!='s20_v4':record_error(row,'unexpected remote strategy')
        if key in seen:record_error(row,'duplicate local/remote ownership')
        if row['case_id'] not in cases:record_error(row,'undeclared case')
        case=cases[row['case_id']];m.validate_records([row],manifest)
        if not row.get('remote_job_receipt') or not Path(row['remote_job_receipt']).is_file():record_error(row,'missing job receipt')
        if not row.get('remote_original_result') or not Path(row['remote_original_result']).is_file():record_error(row,'missing original result artifact')
        source=row.get('remote_source_sha256',{})
        for path in required:
            if source.get(path)!=pinned.get(path):record_error(row,'frozen source mismatch '+path)
        peers=[r for r in local['records'] if r['case_id']==row['case_id'] and r['strategy']!='s20_v4']
        initial=[r.get('initial_shared_depth_calibration') for r in peers if r.get('initial_shared_depth_calibration')]
        if initial:
            actual=row.get('initial_shared_depth_calibration') or {}
            for field in ('a','b','n_points','rmse_px','method'):
                if actual.get(field)!=initial[0].get(field):record_error(row,'initial depth context mismatch '+field)
        elif any(r.get('depth_failure') for r in peers):
            if not row.get('error') or any(row.get(k)!=0 for k in ('score','row_acc','boundary_f1','cluster_ari')):record_error(row,'common depth failure was not retained')
        else:record_error(row,'no validated local initial depth context')
        log=None
        try:
            log=io.Log(Image.open(case['image']),m.read(case['annotation'])['bounding_boxes'],case['log'])
            rgb=io.read_rgb(log,'interpreted_lithology_column',case['y0'],case['y1'])
            crop_digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            if crop_digest!=row.get('crop_rgb_sha256'):record_error(row,'native source crop bytes differ')
        finally:
            if log is not None:log.close()
        prediction=io.load_rows(row['prediction_csv'])
        if prediction is None:record_error(row,'prediction CSV absent')
        if not row.get('error'):
            replay=score.score(prediction,io.load_rows(case['truth_csv']),tuple(case['window']),min_layer_px=0)
            for metric in ('score','row_acc','boundary_f1','cluster_ari','depth_mae_ft'):
                a,b=replay.get(metric),row.get(metric)
                if (a is None)!=(b is None) or a is not None and abs(a-b)>1e-9:record_error(row,'strict metric replay differs '+metric)
        row.setdefault('strategy_diagnostics',row.get('diagnostics',{}));row.setdefault('api_calls',0)
        row['remote_original_record_sha256']=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
        records.append(row);seen.add(key);proof.append(dict(case_id=row['case_id'],crop_rgb_sha256=crop_digest,receipt=row['remote_job_receipt'],initial_context_matched=True))
    expected={(c['case_id'],st) for c in manifest['cases'] for st in manifest['candidates']}
    if seen!=expected:raise RuntimeError(f'Incomplete merge: {len(seen)}/{len(expected)} keys; missing {len(expected-seen)}, extra {len(seen-expected)}')
    m.validate_records(records,manifest);m.check_sources(manifest['source_hashes'])
    summaries,paired=m.summarize(records)
    data=dict(version=3,manifest_digest=digest,manifest=str(HERE/'manifest_1108.json'),method=manifest['method'],records=records,summaries=summaries,paired=paired,complete=True,grading='Original strict shared-scorer policy; owner equivalence is a later explicit postprocess',merge_proof=dict(local_records=len(local['records']),remote_records=len(remote['records']),checks=proof))
    stamp=time.strftime('%H%M%S',time.gmtime());m.write(HERE/f'local_checkpoint_before_merge_{stamp}.json',local)
    m.write(HERE/'strict_scores.json',data);m.write(HERE/'scores.json',data)
    m.write(HERE/'raw_results.json',dict(manifest=str(HERE/'manifest_1108.json'),manifest_digest=digest,records=records,merged_remote=True))
    m.write(HERE/'merge_receipt.json',dict(records=len(records),expected=len(expected),manifest_digest=digest,remote_native_crop_checks=len(proof),checks=proof))
    print(f'MERGE PASS: {len(records)}/{len(expected)} exact records; {len(proof)} remote native crops, initial contexts and strict metric replays checked')
if __name__=='__main__':main()
