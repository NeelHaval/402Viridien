"""Validate availability reporting. This module does not score predictions."""
from __future__ import annotations
import hashlib,math,json
from pathlib import Path

LITHOLOGY_METRICS={'row_acc','cluster_ari','score'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_metric(metric,name,filled_rows=None,series=None):
    if not isinstance(metric.get('available'),bool) or not isinstance(metric.get('eligible'),bool):
        raise ValueError('Metric availability and eligibility must be explicit booleans')
    value=metric.get('value');available=metric['available']
    if available:
        if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('An available metric requires a finite value')
        if filled_rows==0:raise ValueError('Zero-filled truth cannot provide a supervised metric')
        if not metric['eligible']:raise ValueError('An available metric must be eligible')
    else:
        if value is not None:raise ValueError('Unavailable metric must be null, not zero or perfect')
        if not metric.get('reasons'):raise ValueError('Unavailable metric requires an exclusion reason')
    return metric


def validate_denominators(metric):
    for unit in ('windows','logs'):
        keys=[part+'_'+unit for part in ('total','available','excluded','eligible')]
        if not all(isinstance(metric.get(k),int) and metric[k]>=0 for k in keys):raise ValueError('Missing or invalid '+unit+' denominators')
        total,available,excluded,eligible=[metric[k] for k in keys]
        if available+excluded!=total or not available<=eligible<=total:raise ValueError('Inconsistent '+unit+' denominator')
    value=metric.get('mean')
    if metric['available_windows']==0 or metric['available_logs']==0:
        if value is not None:raise ValueError('No available observations cannot produce a mean')
    elif not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Available observations require a finite mean')
    return metric


def verify_preservation(receipt,baseline):
    baseline=Path(baseline)
    paths={'preserved_html_sha256':baseline/'index_baseline_v1.html','scores_json_sha256':baseline/'scores.json','raw_results_json_sha256':baseline/'raw_results.json','shared_scorer_sha256':baseline.parents[1]/'harness'/'score.py'}
    if receipt.get('original_html_sha256')!=receipt.get('preserved_html_sha256'):raise ValueError('Baseline HTML copy receipt differs')
    for key,path in paths.items():
        if sha(path)!=receipt.get(key):raise ValueError('Preserved baseline changed: '+key)
    return True


def validate_report(data_path,audit_path,baseline):
    """Check the bound audit and report inputs without opening truth or images."""
    baseline=Path(baseline)
    data=json.loads(Path(data_path).read_text(encoding='utf-8'))
    audit=json.loads(Path(audit_path).read_text(encoding='utf-8'))
    receipt=json.loads((baseline/'baseline_v1_preservation_receipt.json').read_text(encoding='utf-8'))
    verify_preservation(receipt,baseline)
    if data.get('version')!='availability_v2' or data.get('complete') is not True:raise ValueError('Derived data is incomplete or has the wrong version')
    if data.get('selection_promoted') is not False or data.get('generalization_verified') is not False:raise ValueError('This report cannot promote a selection or certify generalization')
    if audit.get('complete') is not True or audit.get('errors')!=[] or audit.get('exact_keys') is not True:raise ValueError('Availability audit did not pass')
    if audit.get('derived_sha256')!=sha(data_path):raise ValueError('Availability audit does not bind these derived bytes')
    if audit.get('source_pins')!=data.get('pins'):raise ValueError('Availability audit source pins differ')
    # These are code and existing review metadata. Do not open CSV or image inputs.
    for path,digest in data['pins'].items():
        if sha(path)!=digest:raise ValueError('A pinned code or review input changed: '+path)
    original=json.loads((baseline/'scores.json').read_text(encoding='utf-8'))
    key=lambda r:(r['case_id'],r['strategy'])
    wanted=[key(r) for r in original['records']];actual=[key(r) for r in data['records']]
    if len(set(actual))!=len(actual) or set(actual)!=set(wanted):raise ValueError('Derived case matrix is duplicate, incomplete, or stale')
    if audit.get('records')!=len(actual) or audit.get('expected')!=len(wanted):raise ValueError('Audit record counts differ')
    for record in data['records']:
        for view,series_set in record['views'].items():
            for series,policies in series_set.items():
                if series not in ('qualified_development','reference_diagnostic'):raise ValueError('Unknown report series')
                for policy,metrics in policies.items():
                    if policy not in ('strict','owner_equivalent'):raise ValueError('Unknown grading policy')
                    for name,metric in metrics.items():validate_metric(metric,name,record['structural_supervision']['filled_rows'],series)
    for summary in data['summaries']:
        for metric in summary['metrics'].values():validate_denominators(dict(metric['denominator'],mean=metric['mean']))
    return data,audit


def validate_parent_audit(path,data_path,count):
    audit=json.loads(Path(path).read_text(encoding='utf-8'))
    if audit.get('errors')!=[] or audit.get('derived_sha256')!=sha(data_path) or audit.get('records')!=count:raise ValueError('Parent audit is stale or failed')
    if audit.get('original_scores_unchanged') is not True or audit.get('selection_promoted') is not False or audit.get('inference_started') is not False:raise ValueError('Parent audit scope differs')
    return audit
