"""Decision gates and explicit source groups for the saved 1108 results."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path

STRATEGIES=('s19_combine','s20_v4','s21_v4dino')
APPROVED_VARIANT='s23_v4dino_identity'
IDENTITY=('suite','log','y0','y1','window','family','well','dataset','image','annotation','column_box','truth_csv','truth_sha256','image_sha256','annotation_sha256','truth_sources')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_records(raw,manifest,expected_case_count=68):
    """Fail closed. A count alone cannot establish complete paired coverage."""
    cases=manifest['cases']; records=raw.get('records',[])
    if len(cases)!=expected_case_count:raise ValueError('Manifest case count differs from the declared experiment')
    candidates=manifest.get('candidates')
    if not isinstance(candidates,list) or len(candidates) not in (3,4) or len(candidates)!=len(set(candidates)):
        raise ValueError('Manifest must declare three frozen candidates and at most one approved additional candidate')
    if not set(STRATEGIES).issubset(candidates) or not set(candidates).issubset(set(STRATEGIES)|{APPROVED_VARIANT}) or not all(isinstance(s,str) and s for s in candidates):
        raise ValueError('Manifest candidates omit a frozen baseline or have an invalid identifier')
    expected={c['case_id']:c for c in cases}
    if len(expected)!=len(cases):raise ValueError('Duplicate manifest case ID')
    wanted={(c,s) for c in expected for s in candidates};actual=set()
    for r in records:
        if is_initial_manual_stop(r):raise ValueError('Provisional operator stop awaits formal capped attempt')
        key=(r.get('case_id'),r.get('strategy'))
        if key in actual:raise ValueError('Duplicate result key: '+str(key))
        if key not in wanted:raise ValueError('Unmatched result key: '+str(key))
        actual.add(key);case=expected[key[0]]
        for f in IDENTITY:
            if f not in case or f not in r or r[f]!=case[f]:raise ValueError('Stale or mismatched case identity: '+str(key)+' / '+f)
        for m in ('score','row_acc','boundary_f1','cluster_ari'):
            if not isinstance(r.get(m),(int,float)):raise ValueError('Missing primary metric: '+str(key)+' / '+m)
    if actual!=wanted:raise ValueError(f'Incomplete case coverage: {len(actual)} of {len(wanted)} exact keys')
    if raw.get('complete') is not True:raise ValueError('Runner did not declare the saved run complete')
    return {'status':'valid','cases':len(cases),'strategies':len(candidates),'candidate_names':candidates,'records':len(records),'exact_case_keys':True,'identity_fields':list(IDENTITY)}

def case_identity(r):return tuple(json.dumps(r.get(f),sort_keys=True) for f in ('case_id',)+IDENTITY)

def paired_case_rows(rows,a,b,suite):
    selected=[r for r in rows if r['suite']==suite and r['strategy'] in (a,b)]
    by={a:{},b:{}}
    for r in selected:
        s=r['strategy'];k=r.get('case_id')
        if not k:raise ValueError('A paired result requires a case ID')
        if k in by[s]:raise ValueError('Duplicate paired case key')
        by[s][k]=r
    if set(by[a])!=set(by[b]):raise ValueError('Unmatched paired case keys')
    out=[]
    for key in sorted(by[a]):
        ra,rb=by[a][key],by[b][key]
        if case_identity(ra)!=case_identity(rb):raise ValueError('Mismatched paired case identity')
        out.append((ra,rb))
    return out

def build_dev_map(manifest):
    groups={};counts=Counter()
    for c in manifest['cases']:
        if c['suite']!='ext_dev':continue
        name=c['log']
        if name=='ext_kgs_386825':group='KGS 386825'
        elif name.startswith('ext_utah_4304150011_p'):group='Utah 4304150011'
        elif name.startswith('ext_idaho_barlow3-14_p'):group='Idaho Barlow 3-14'
        elif name.startswith('ext_nl_flat-bay-2_p'):group='Flat Bay 2'
        elif name.startswith('ext_nl_storm_p'):group='Storm'
        else:raise ValueError('Unreviewed DEV source name: '+name)
        groups[name]=group;counts[group]+=1
    expected={'KGS 386825':2,'Utah 4304150011':21,'Idaho Barlow 3-14':6,'Flat Bay 2':9,'Storm':13}
    if dict(counts)!=expected:raise ValueError('DEV source group counts differ from reviewed cases')
    return {'source':'Only the allowed DEV names in manifest_1108.json. Astra identified five provisional source groups; the parent requested this explicit mapping. No geological independence is certified.','group_case_counts':dict(counts),'log_to_group':groups}

def records_digest(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()

def validate_audit(audit,records,manifest_path):
    if audit.get('errors')!=[]:raise ValueError('Independent saved-prediction audit did not pass')
    if audit.get('records')!=len(records) or audit.get('expected')!=len(records):raise ValueError('Independent audit coverage differs from results')
    if audit.get('manifest_sha256')!=digest(manifest_path):raise ValueError('Independent audit uses a stale manifest')
    if audit.get('records_sha256')!=records_digest(records):raise ValueError('Independent audit uses stale result records')
    return {'status':'valid','records_sha256':audit['records_sha256'],'manifest_sha256':audit['manifest_sha256']}


def is_initial_manual_stop(record):
    return record.get('operational_timeout') is True and str(record.get('timeout_scope','')).startswith('Initial manual stop')


def validate_prediction_bytes(audit,records):
    expected={r['prediction_csv'] for r in records}
    expected.update(r['pixel_only']['prediction_csv'] for r in records if r.get('pixel_only'))
    pins=audit.get('prediction_csv_sha256')
    if not isinstance(pins,dict) or set(pins)!=expected:raise ValueError('Numerical audit prediction CSV path set differs from results')
    for path in sorted(expected):
        if digest(path)!=pins[path]:raise ValueError('Prediction CSV bytes changed after numerical audit: '+path)
    return dict(status='valid',prediction_csv_count=len(expected))
