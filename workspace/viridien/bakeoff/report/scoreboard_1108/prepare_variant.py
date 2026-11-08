from pathlib import Path
import json,time
HERE=Path(__file__).resolve().parent
p=HERE/'score_all.py';s=p.read_text(encoding='utf-8').replace('s23_v4dino_distinct','s23_v4dino_identity')
s=s.replace("REPORT/'reviewed_dev_wells.json',", "REPORT/'reviewed_dev_wells.json',REPORT/'variant_protocol_addendum.json',")
p.write_text(s,encoding='utf-8')
addendum=dict(status='User-authorized development follow-up after partial baseline scores',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 trigger='Owner asked to fix false Clay/Shale merging after the log06 development error was diagnosed. Thirteen baseline records existed when the scoring process stopped at a checkpoint.',
 candidate='s23_v4dino_identity',policy='Preserve all separate printed legend labels; disable visual-twin caption merging in the new candidate. A genuine single printed Clay/Shale caption remains Clay/Shale. This is not a Clay-to-Shale mapping.',
 frozen_baselines=['s19_combine','s20_v4','s21_v4dino'],metric='Unchanged harness.score.score and its existing normalization/aliases. No truth-aware label remapping or membership credit.',
 design='Same 68 input cases, native image policy, initial depth/legend context, frozen DINO model/config, and runtime measurement. Exact four-candidate matrix is272 records.',
 interpretation='Corrected candidate is a development follow-up. Reused log06 and other development truth is not independent validation. Baseline checkpoints are retained. No test or verify data is opened.',
 review='Original Astra review covered the baseline method and code. The new candidate is outside that original code scope unless the owner authorizes another review.')
(HERE/'variant_protocol_addendum.json').write_text(json.dumps(addendum,indent=2),encoding='utf-8')
p=HERE/'migrate_manifest.py';s=p.read_text(encoding='utf-8').replace("manifest['version']=2;manifest['source_hashes']", "manifest['version']=3;manifest['candidates']=m.CANDIDATES;manifest['variant_protocol_addendum']=str(HERE/'variant_protocol_addendum.json');manifest['source_hashes']")
s=s.replace("record['manifest_digest']=digest;record['freeze_migration']='legacy input and identity validated at migration; original read-time hash guard absent';", "record.setdefault('previous_manifest_digests',[]).append(record.get('manifest_digest'));record['manifest_digest']=digest;record.setdefault('freeze_migration','Revalidated input and identity before candidate-set migration; original read-time input guards are retained when present');")
p.write_text(s,encoding='utf-8')
