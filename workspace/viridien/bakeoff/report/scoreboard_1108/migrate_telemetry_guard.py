"""Pin a stopped orchestration-only atomic publication correction."""
import importlib.util,shutil,time
from pathlib import Path
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('score_telemetry_migration',H/'score_all.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
    manifest=m.read(H/'manifest_1108.json');raw=m.read(H/'raw_results.json');m.validate_records(raw['records'],manifest)
    allowed={'report/scoreboard_1108/score_all.py','report/scoreboard_1108/bounded_local.py'}
    changed=[k for k,v in manifest['source_hashes'].items() if m.sha(m.B/k)!=v]
    if set(k.replace(chr(92),'/') for k in changed)!=allowed:raise RuntimeError('Unexpected source changes '+str(changed))
    stamp=time.strftime('%H%M%S',time.gmtime());shutil.copyfile(H/'manifest_1108.json',H/f'manifest_before_telemetry_{stamp}.json');shutil.copyfile(H/'raw_results.json',H/f'checkpoint_before_telemetry_{stamp}.json')
    manifest['source_hashes']=m.sources()
    manifest.setdefault('execution_revisions',[]).append(dict(at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reason='Bounded retry of atomic JSON replacement after Windows delete-sharing conflict. Orchestration only; all inference algorithms, windows, scorer and model bytes unchanged. Preserved original failed telemetry attempt and formal Shenzi history.',changed_sources=changed,old_manifest_digest=m.manifest_digest(m.read(H/f'manifest_before_telemetry_{stamp}.json'))))
    digest=m.manifest_digest(manifest)
    for r in raw['records']:r.setdefault('previous_manifest_digests',[]).append(r['manifest_digest']);r['manifest_digest']=digest
    raw['manifest_digest']=digest;m.validate_records(raw['records'],manifest);m.write(H/'manifest_1108.json',manifest);m.write(H/'raw_results.json',raw)
    print('TELEMETRY MIGRATION PASS:',len(raw['records']),'records;',len(changed),'orchestration source changes; digest',digest)
if __name__=='__main__':main()
