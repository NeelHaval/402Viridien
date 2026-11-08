"""Finite checkpoint-aware owner of one warm worker at a time."""
from __future__ import annotations
import argparse,json,math,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def supervise(planned,completed,command,progress_path,on_timeout,log_dir,cap=600,poll=1,max_no_progress_exits=2):
    """A timeout kills only this Popen child; durable rows drive every restart."""
    planned=set(planned);log_dir=Path(log_dir);log_dir.mkdir(parents=True,exist_ok=True)
    no_progress_exits=0;events=[]
    for generation in range(len(planned)+max_no_progress_exits+1):
        before=set(completed())
        if planned<=before:return dict(complete=True,events=events,starts=generation)
        progress_path=Path(progress_path)
        if progress_path.exists():progress_path.unlink()
        with (log_dir/f'worker_{generation:03d}.log').open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command(),stdout=stream,stderr=subprocess.STDOUT)
            started=time.perf_counter();event=dict(generation=generation,pid=child.pid,worker_started_utc=utc());events.append(event)
            atomic(log_dir/'controller_receipt.json',dict(planned_keys=len(planned),events=events))
            timed_out=False;last_phase=None;attempt=None
            # A finite queue of N attempts permits at most (N+1)*cap wall time
            # in one warm child. Every active attempt still has its own cap.
            max_ticks=math.ceil(cap*(len(planned-before)+1)/poll)+2
            for _ in range(max_ticks):
                if child.poll() is not None:break
                if progress_path.exists():
                    candidate=read(progress_path)
                    if candidate.get('pid')==child.pid:
                        attempt=candidate
                        key=(attempt.get('case_id'),attempt.get('strategy'))
                        if key not in planned:child.terminate();child.wait(timeout=10);raise RuntimeError('Child announced an unplanned key')
                        phase_key=(attempt.get('attempt_id'),attempt.get('phase'))
                        if phase_key!=last_phase:
                            last_phase=phase_key
                            with (log_dir/'phase_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(attempt)+'\n')
                now=time.perf_counter()
                active=attempt and attempt.get('phase')!='completed'
                elapsed=now-attempt['attempt_started_perf'] if active else now-(attempt.get('phase_observed_perf',started) if attempt else started)
                # A completed checkpoint can spend time in final source/input
                # validation. That is bounded worker finalization, not inference.
                limit_reached=elapsed>=cap
                if limit_reached:
                    stop=dict(stop_requested_utc=utc(),stop_requested_perf=now,scope=attempt.get('phase') if active else 'startup_or_finalization',hard_cap_seconds=cap)
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
                    stop.update(stop_confirmed_utc=utc(),stop_confirmed_perf=time.perf_counter(),worker_terminated=True,pid=child.pid)
                    event['stop']=stop
                    if active:
                        stop['whole_attempt_elapsed_seconds']=stop['stop_confirmed_perf']-attempt['attempt_started_perf']
                        if attempt.get('strategy_started_perf') is not None:stop['run_wall_observed_upper_bound_seconds']=stop['stop_confirmed_perf']-attempt['strategy_started_perf']
                        on_timeout(attempt,stop);timed_out=True
                    else:event['noncase_timeout']=True
                    break
                time.sleep(poll)
            else:
                child.terminate();child.wait(timeout=10);raise RuntimeError('Finite worker wall budget exhausted')
            event.update(exit_code=child.poll(),worker_ended_utc=utc(),durable_keys=len(set(completed())&planned),timeout_recorded=timed_out)
            atomic(log_dir/'controller_receipt.json',dict(planned_keys=len(planned),events=events))
            after=set(completed())
            if after==before:no_progress_exits+=1
            else:no_progress_exits=0
            if no_progress_exits>=max_no_progress_exits:raise RuntimeError('Two owned-child exits without durable progress; stopped')
    raise RuntimeError('Finite restart budget exhausted')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cap',type=float,default=600);args=parser.parse_args()
    import importlib.util
    H=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('score_all_supervised',H/'score_all.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    manifest=m.read(H/'manifest_1108.json');path=H/'raw_results.json';progress=H/'active_local_attempt.json';strategies=['s19_combine','s21_v4dino','s23_v4dino_identity']
    planned={(c['case_id'],st) for c in manifest['cases'] for st in strategies};digest=m.manifest_digest(manifest)
    m.validate_records(m.read(path)['records'],manifest);m.check_sources(manifest['source_hashes'])
    def completed():return {(r['case_id'],r['strategy']) for r in m.read(path)['records']}
    def timeout(attempt,stop):
        raw=m.read(path);key=(attempt['case_id'],attempt['strategy'])
        if key in completed():return
        case=next(c for c in manifest['cases'] if c['case_id']==key[0]);m.validate_inputs(case)
        pred=m.OUT/key[1]/(key[0]+'.csv');m.io.save_rows(pred,[])
        record={**case,'strategy':key[1],'scored':True,'manifest_digest':digest,'prediction_csv':str(pred),'error':f'OperationalTimeout: whole case/strategy attempt exceeded {args.cap:g} s in {attempt["phase"]}',
            'failure_phase':attempt['phase'],'operational_timeout':True,'worker_terminated':True,'seconds':None,'legend_seconds':attempt.get('legend_seconds'),'depth_seconds':attempt.get('depth_seconds'),'preprocess_seconds':None,'end_to_end_seconds':None,'api_calls':0,
            'score':0.,'row_acc':0.,'boundary_f1':0.,'cluster_ari':0.,'depth_mae_ft':None,'attempt_id':attempt['attempt_id'],'attempt_started_utc':attempt['attempt_started_utc'],'timeout_receipt':stop,'recorded_utc':utc(),
            'runtime_scope':'600s whole attempt includes shared preprocessing and strategy. Failed run seconds are unknown, not600. Models stay warm between successful attempts; restart cold costs are labelled.'}
        if attempt.get('initial_shared_depth_calibration'):record['initial_shared_depth_calibration']=attempt['initial_shared_depth_calibration']
        if attempt['phase']=='shared_depth':record['depth_failure']=True
        raw['records'].append(record);m.write(path,raw);m.write(H/'timeouts'/f'{key[0]}_{key[1]}.json',record)
        print(f'TIMEOUT {key} phase={attempt["phase"]} run_seconds=UNKNOWN',flush=True)
    command=lambda:[sys.executable,'-X','utf8',str(H/'score_all.py'),'--resume','--strategies',','.join(strategies),'--progress-file',str(progress)]
    os.environ['PYTHONPATH']=str(m.B);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    result=supervise(planned,completed,command,progress,timeout,m.OUT/'supervisor',cap=args.cap)
    m.write(H/'supervisor_complete.json',result)
    print(f'LOCAL BOUNDED COMPLETE: {len(planned)} local candidate keys; {len(m.read(path)["records"])} retained total records')
if __name__=='__main__':main()
