from pathlib import Path
p=Path(__file__).with_name('migrate_manifest.py');s=p.read_text(encoding='utf-8')
s=s.replace("digest=m.manifest_digest(manifest)", "manifest.setdefault('execution_revisions',[]).append(dict(at_utc=manifest['freeze']['migration_utc'],reason='Owner authorized unfinished CPU offload; local runner paused after41 records using source-guard comment only. Frozen model/classifier/scorer algorithms unchanged.',remaining_cpu_contract=str(HERE/'unfinished_cpu_cases.json'),remote_output='Separate shard; merge only after local writer stops. Source-coordinate lossless native crops and exact initial calibration require receipts. VM and local runtime are distinct regimes.'))\ndigest=m.manifest_digest(manifest)")
p.write_text(s,encoding='utf-8')
