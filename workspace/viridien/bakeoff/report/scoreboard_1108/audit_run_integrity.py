"""Bind full coverage, source/input identity and model evidence to saved records."""
import hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    data=read(HERE/'scores.json');manifest=read(HERE/'manifest_1108.json');records=data['records'];errors=[]
    spec=importlib.util.spec_from_file_location('score_all_integrity',HERE/'score_all.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    try:
        runner.validate_records(records,manifest)
        if data.get('manifest_digest')!=runner.manifest_digest(manifest):errors.append('Canonical manifest digest mismatch')
    except Exception as exc:errors.append('Input/resume identity audit: '+str(exc))
    ids=[(r['case_id'],r['strategy']) for r in records]
    expected={(c['case_id'],s) for c in manifest['cases'] for s in manifest['candidates']}
    exact=set(ids)==expected and len(ids)==len(set(ids))
    if not exact:errors.append(f'Case-key mismatch: missing={len(expected-set(ids))}, extra={len(set(ids)-expected)}, duplicates={len(ids)-len(set(ids))}')
    counts={'primary':0,'pixel':0};pins=manifest['freeze']['dino']
    def validate_model(diag,kind,record):
        chunks=diag.get('dino_chunk_vectors',0);model=diag.get('model')
        if chunks>0 and not model:errors.append(f'Missing {kind} DINO model evidence: {record["case_id"]}/{record["strategy"]}')
        if model:
            counts[kind]+=1
            for key in ('weights_sha256','config_sha256'):
                if model.get(key)!=pins[key]:errors.append(f'{kind} DINO pin mismatch: {record["case_id"]}/{key}')
    for row in records:
        if abs(row['score']-row['row_acc']*row['boundary_f1'])>1e-10:errors.append('Product mismatch '+row['case_id'])
        if not Path(row['prediction_csv']).exists():errors.append('Missing primary prediction '+row['case_id'])
        if row.get('error') and any(row[k]!=0 for k in ('score','row_acc','boundary_f1','cluster_ari')):errors.append('Nonzero failure quality '+row['case_id'])
        validate_model(row.get('strategy_diagnostics',{}),'primary',row)
        if row.get('pixel_only'):
            diagnostic=row['pixel_only'];validate_model(diagnostic.get('strategy_diagnostics',{}),'pixel',row)
            if not Path(diagnostic['prediction_csv']).exists():errors.append('Missing pixel prediction '+row['case_id'])
    for path,digest in manifest['source_hashes'].items():
        p=B/path
        if not p.is_file() or sha(p)!=digest:errors.append('Executed source drift '+path)
    copied=[]
    for name in ('selected.json','boundary_net.pt','matcher.npz','boundary_real.pt','matcher_real.npz'):
        src=B/'ml'/'combine'/name;dst=B/'out'/'scoreboard_1108'/'s19_runtime'/name
        if src.exists():
            same=dst.exists() and sha(src)==sha(dst);copied.append(dict(name=name,sha256=sha(src),copy_matches=same))
            if not same:errors.append('S19 copied artifact mismatch '+name)
    inputs=set()
    for case in manifest['cases']:
        inputs.update(case[k] for k in ('image','annotation','truth_csv'))
        inputs.update(v['path'] for v in case['truth_sources'])
    if data.get('complete') is not True:errors.append('Runner complete flag is false')
    result=dict(records=len(records),expected=len(expected),complete=exact and not errors,errors=errors,exact_key_proof=exact,
        manifest_sha256=sha(HERE/'manifest_1108.json'),records_sha256=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest(),
        manifest_digest=data.get('manifest_digest'),primary_model_evidence_checks=counts['primary'],pixel_model_evidence_checks=counts['pixel'],input_hash_checks=len(inputs),source_hash_checks=len(manifest['source_hashes']),s19_artifacts=copied)
    (HERE/'result_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"{'PASS' if not errors else 'FAIL'}: {len(records)}/{len(expected)} records, exact keys {exact}, {len(errors)} integrity errors")
    for error in errors:print(error)
    return bool(errors)
if __name__=='__main__':sys.exit(main())
