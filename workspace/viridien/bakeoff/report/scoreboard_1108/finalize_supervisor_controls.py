from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'score_all.py';s=p.read_text(encoding='utf-8').replace("attempt.update(phase=phase,observed_utc=", "attempt.update(phase=phase,phase_observed_perf=time.perf_counter(),observed_utc=")
s=s.replace("'first_strategy_call_in_process':strategy not in first_by_strategy}", "'first_strategy_call_in_process':strategy not in first_by_strategy,'attempt_id':attempt['attempt_id'],'whole_attempt_started_utc':attempt['attempt_started_utc'],'worker_pid':os.getpid()}")
p.write_text(s,encoding='utf-8')
p=H/'bounded_local.py';s=p.read_text(encoding='utf-8').replace("elapsed=now-attempt['attempt_started_perf'] if active else now-started", "elapsed=now-attempt['attempt_started_perf'] if active else now-(attempt.get('phase_observed_perf',started) if attempt else started)")
p.write_text(s,encoding='utf-8')
p=H/'migrate_manifest.py';s=p.read_text(encoding='utf-8').replace("reason='Owner authorized unfinished CPU offload; local runner paused after41 records using source-guard comment only. Frozen model/classifier/scorer algorithms unchanged.'", "reason='Parent-approved600s checkpoint-aware warm-child supervisor after manual Shenzi S19 timeout. No inference algorithm, window or shared scorer change; preserved53 durable records and original histories.'")
p.write_text(s,encoding='utf-8')
