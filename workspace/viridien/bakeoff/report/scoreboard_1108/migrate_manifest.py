"""Migrate a stopped checkpoint after input and executed-code checks."""
import importlib.util,shutil,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'score_all.py';spec=importlib.util.spec_from_file_location('score_all',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
manifest=m.read(HERE/'manifest_1108.json');raw=m.read(HERE/'raw_results.json')
m.validate_records(raw['records'],manifest,allow_legacy=True)
changed=[key for key,value in manifest['source_hashes'].items() if key.replace(chr(92),'/')!='report/scoreboard_1108/score_all.py' and m.sha(m.B/key)!=value]
if changed:raise RuntimeError('Executed code drift '+str(changed))
stamp=time.strftime('%H%M%S',time.gmtime())
shutil.copyfile(HERE/'manifest_1108.json',HERE/f'manifest_pre_m2_{stamp}.json')
shutil.copyfile(HERE/'raw_results.json',HERE/f'checkpoint_pre_m2_{stamp}.json')
manifest['version']=3;manifest['candidates']=m.CANDIDATES;manifest['variant_protocol_addendum']=str(HERE/'variant_protocol_addendum.json');manifest['source_hashes']=m.sources();manifest['freeze']=m.freeze_metadata()
manifest['freeze']['migration_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
manifest['freeze']['legacy_records']=len(raw['records'])
manifest['freeze']['migration_note']='Validated exact record identities, all declared current input bytes and all unchanged executed sources before adding metadata. Input validation is present-state evidence for legacy records; original read-time hashes were not asserted. Source change adds guards and provenance only.'
manifest.setdefault('execution_revisions',[]).append(dict(at_utc=manifest['freeze']['migration_utc'],reason='Parent-approved600s checkpoint-aware warm-child supervisor after manual Shenzi S19 timeout. No inference algorithm, window or shared scorer change; preserved53 durable records and original histories.',remaining_cpu_contract=str(HERE/'unfinished_cpu_cases.json'),remote_output='Separate shard; merge only after local writer stops. Source-coordinate lossless native crops and exact initial calibration require receipts. VM and local runtime are distinct regimes.'))
digest=m.manifest_digest(manifest)
for record in raw['records']:
 record.setdefault('previous_manifest_digests',[]).append(record.get('manifest_digest'));record['manifest_digest']=digest;record.setdefault('freeze_migration','Revalidated input and identity before candidate-set migration; original read-time input guards are retained when present');record.setdefault('process_start_utc',None);record.setdefault('resumed_process',None);record.setdefault('first_strategy_call_in_process',None)
raw['manifest_digest']=digest
m.validate_records(raw['records'],manifest)
m.write(HERE/'manifest_1108.json',manifest);m.write(HERE/'raw_results.json',raw)
print('MIGRATION PASS:',len(raw['records']),'records; all inputs and unchanged inference sources matched')
