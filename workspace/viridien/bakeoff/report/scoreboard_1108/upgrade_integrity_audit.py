from pathlib import Path
p=Path(__file__).with_name('audit_run_integrity.py');s=p.read_text(encoding='utf-8')
s=s.replace('import hashlib,json,sys','import hashlib,json,sys,importlib.util')
s=s.replace("ids=[(r['case_id'],r['strategy']) for r in records]", "spec=importlib.util.spec_from_file_location('score_all_audit',HERE/'score_all.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)\n    try:\n        runner.validate_records(records,manifest)\n        if data.get('manifest_digest')!=runner.manifest_digest(manifest):errors.append('Top-level manifest digest mismatch')\n    except Exception as exc:errors.append('Input/resume identity audit: '+str(exc))\n    ids=[(r['case_id'],r['strategy']) for r in records]")
s=s.replace("if not Path(row['prediction_csv']).exists():", "model=row.get('strategy_diagnostics',{}).get('model',{})\n        if model:\n            for key in ('weights_sha256','config_sha256'):\n                if model.get(key)!=manifest['freeze']['dino'][key]:errors.append('DINO artifact pin mismatch '+key)\n        if not Path(row['prediction_csv']).exists():")
s=s.replace("candidate_count=len(candidates),complete", "candidate_count=len(candidates),manifest_digest=data.get('manifest_digest'),complete")
p.write_text(s,encoding='utf-8')
