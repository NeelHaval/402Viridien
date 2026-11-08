"""Audit completed score artifacts without opening any sealed data."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=HERE.parents[1]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    data=read(HERE/'scores.json');manifest=read(HERE/'manifest_1108.json');records=data['records'];errors=[]
    ids=[(r['case_id'],r['strategy']) for r in records]
    if len(ids)!=len(set(ids)):errors.append('Duplicate candidate/case record')
    candidates=['s19_combine','s20_v4','s21_v4dino']
    expected={(c['case_id'],s) for c in manifest['cases'] for s in candidates}
    if set(ids)!=expected:errors.append(f'Result key mismatch: missing={len(expected-set(ids))}, extra={len(set(ids)-expected)}')
    for row in records:
        if abs(row['score']-row['row_acc']*row['boundary_f1'])>1e-10:errors.append('Product mismatch '+row['case_id'])
        if not Path(row['prediction_csv']).exists():errors.append('Missing prediction '+row['case_id'])
        if row.get('error') and any(row[k]!=0 for k in ('score','row_acc','boundary_f1','cluster_ari')):errors.append('Failure score nonzero '+row['case_id'])
    for path,digest in manifest['source_hashes'].items():
        p=B/path
        if not p.is_file() or sha(p)!=digest:errors.append('Executed source drift '+path)
    copied=[]
    for name in ('selected.json','boundary_net.pt','matcher.npz','boundary_real.pt','matcher_real.npz'):
        src=B/'ml'/'combine'/name;dst=B/'out'/'scoreboard_1108'/'s19_runtime'/name
        if src.exists():
            same=dst.exists() and sha(src)==sha(dst);copied.append(dict(name=name,sha256=sha(src),copy_matches=same))
            if not same:errors.append('S19 copied artifact mismatch '+name)
    audit=dict(records=len(records),cases=len(manifest['cases']),candidate_count=len(candidates),complete=data.get('complete'),errors=errors,s19_artifacts=copied)
    (HERE/'result_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(f"{'PASS' if not errors else 'FAIL'}: {len(records)} records, {len(manifest['cases'])} cases, {len(errors)} audit errors")
    for e in errors:print(e)
    return bool(errors)
if __name__=='__main__':sys.exit(main())
