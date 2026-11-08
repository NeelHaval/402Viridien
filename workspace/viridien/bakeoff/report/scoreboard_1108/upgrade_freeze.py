from pathlib import Path
p=Path(__file__).with_name('score_all.py');s=p.read_text(encoding='utf-8')
s=s.replace('import argparse, csv, hashlib, importlib, json, os, re, sys, time, traceback, shutil','import argparse, csv, hashlib, importlib, json, os, re, sys, time, traceback, shutil, platform\nfrom importlib.metadata import version as package_version')
s=s.replace("paths=[Path(__file__),REPORT/'evaluation_protocol.json']", "paths=[Path(__file__),REPORT/'evaluation_protocol.json',B/'strategies'/'s02_features_pelt.py']")
pos=s.index('\ndef image_path')
s=s[:pos]+'''
def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()

def validate_inputs(case):
    for pathkey,hashkey in [('image','image_sha256'),('annotation','annotation_sha256'),('truth_csv','truth_sha256')]:
        if sha(case[pathkey])!=case[hashkey]:raise RuntimeError('Input drift: '+case[pathkey])
    for source in case['truth_sources']:
        if sha(source['path'])!=source['sha256']:raise RuntimeError('Original truth drift: '+source['path'])

def validate_records(records,manifest):
    cases={c['case_id']:c for c in manifest['cases']};seen=set()
    for record in records:
        key=(record['case_id'],record['strategy'])
        if key in seen:raise RuntimeError('Duplicate resume record '+str(key))
        seen.add(key)
        if record['strategy'] not in ('s19_combine','s20_v4','s21_v4dino'):raise RuntimeError('Unknown resume strategy')
        case=cases.get(record['case_id'])
        if case is None:raise RuntimeError('Unknown resume case')
        for field in ('log','suite','window','image','annotation','truth_csv','image_sha256','annotation_sha256','truth_sha256','truth_sources'):
            if record.get(field)!=case.get(field):raise RuntimeError('Resume identity mismatch '+field)
        if record.get('manifest_digest') not in (None,manifest_digest(manifest)):raise RuntimeError('Resume manifest digest mismatch')
    for case in manifest['cases']:validate_inputs(case)

''' + s[pos:]
s=s.replace("torch.set_num_threads(2)", "torch.set_num_threads(2)\n    process_start=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());first_by_strategy=set()")
s=s.replace("done={(r['case_id'],r['strategy']) for r in records}", "validate_records(records,manifest)\n    digest=manifest_digest(manifest)\n    done={(r['case_id'],r['strategy']) for r in records}")
s=s.replace("check_sources(manifest['source_hashes'])\n            rec=", "check_sources(manifest['source_hashes']);validate_inputs(case)\n            rec=")
s=s.replace("rec={**case,'strategy':strategy,'scored':True}", "rec={**case,'strategy':strategy,'scored':True,'manifest_digest':digest,'process_start_utc':process_start,'resumed_process':args.resume,'first_strategy_call_in_process':strategy not in first_by_strategy}")
s=s.replace("rec['preprocess_seconds']=", "rec['initial_shared_depth_calibration']=dict(a=calib.a,b=calib.b,method=calib.method,n_points=calib.n_points,rmse_px=calib.rmse_px)\n                rec['depth_note']='Initial shared depth; s19 may refine its own output depths'\n                rec['preprocess_seconds']=")
s=s.replace("rec['strategy_diagnostics']=getattr(module.run,'last_diagnostics',{})", "rec['strategy_diagnostics']=getattr(module.run,'last_diagnostics',{})\n                if strategy=='s21_v4dino' and rec['strategy_diagnostics'].get('weights_sha256'):\n                    for key in ('weights_sha256','config_sha256'):\n                        if rec['strategy_diagnostics'][key]!=manifest['freeze']['dino'][key]:raise RuntimeError('Pinned DINO artifact drift '+key)")
s=s.replace("io.save_rows(pred,rows);records.append(rec)", "rec['recorded_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())\n            first_by_strategy.add(strategy)\n            io.save_rows(pred,rows);records.append(rec)")
s=s.replace("dict(manifest=str(REPORT/'manifest_1108.json'),records=records)", "dict(manifest=str(REPORT/'manifest_1108.json'),manifest_digest=digest,records=records)")
s=s.replace("summary,paired=summarize(records)", "for case in manifest['cases']:validate_inputs(case)\n    check_sources(manifest['source_hashes'])\n    validate_records(records,manifest)\n    summary,paired=summarize(records)")
s=s.replace("dict(version=1,method=manifest['method']", "dict(version=1,manifest_digest=digest,method=manifest['method']")
p.write_text(s,encoding='utf-8')
