from pathlib import Path
import hashlib,json,time
HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'manifest_1108.json').read_text(encoding='utf-8'));raw=json.loads((HERE/'raw_results.json').read_text(encoding='utf-8'))
done={(r['case_id'],r['strategy']) for r in raw['records']}
work={strategy:[c for c in m['cases'] if (c['case_id'],strategy) not in done] for strategy in ('s19_combine','s20_v4')}
result=dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),owned_local_process=dict(pid=11120,created_utc='2026-11-08T06:18:10.0530580Z',state='exited_after_checkpoint'),completed_records=len(raw['records']),completed_by_strategy={s:sum(r['strategy']==s for r in raw['records']) for s in m['candidates']},
    candidates=m['candidates'],source_manifest_sha256=hashlib.sha256((HERE/'manifest_1108.json').read_bytes()).hexdigest(),work=work,
    staging_note='Separate remote output shard. Never mutate local raw_results.json while its runner is active. Initial calibration must be exactly validated/staged locally; native crop bytes must match original loader output. All grades are postprocessed after inference.')
p=HERE/'unfinished_cpu_cases.json';p.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('CHECKPOINT',len(raw['records']),'records',result['completed_by_strategy'])
print('UNFINISHED',{s:len(v) for s,v in work.items()})
