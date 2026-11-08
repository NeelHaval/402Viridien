"""Final report provenance gate. This module does not run inference."""
from __future__ import annotations
import hashlib
import json
from report_inputs import validate_audit


def manifest_digest(manifest):
    # This is the runner's canonical digest. File-byte SHA is a separate audit binding.
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def require_run_provenance(raw,manifest):
    declared=manifest_digest(manifest)
    if raw.get('manifest_digest')!=declared:raise ValueError('Top-level manifest_digest is stale')
    pins=manifest.get('freeze',{}).get('dino',{})
    if not all(pins.get(k) for k in ('weights_sha256','config_sha256')):raise ValueError('Manifest is missing DINO artifact pins')
    evidence={'primary_embedding_records':0,'pixel_embedding_records':0,'primary_model_records':0,'pixel_model_records':0}
    for record in raw['records']:
        if record.get('manifest_digest')!=declared:raise ValueError('Record manifest_digest is stale: '+record['case_id'])
        for scope,diag in [('primary',record.get('strategy_diagnostics',{})),('pixel',record.get('pixel_only',{}).get('strategy_diagnostics',{}))]:
            chunks=diag.get('dino_chunk_vectors',0) or 0
            model=diag.get('model')
            if chunks>0:
                evidence[scope+'_embedding_records']+=1
                if not model:raise ValueError(scope+' embedding chunks lack model evidence: '+record['case_id'])
            if model:
                evidence[scope+'_model_records']+=1
                for key in ('weights_sha256','config_sha256'):
                    if model.get(key)!=pins[key]:raise ValueError(scope+' model pin mismatch: '+key+' / '+record['case_id'])
    return evidence


def require_integrity_audit(audit,raw,manifest,manifest_path):
    binding=validate_audit(audit,raw['records'],manifest_path)
    if audit.get('complete') is not True:raise ValueError('Run integrity audit is incomplete')
    if audit.get('exact_key_proof') is not True:raise ValueError('Run integrity audit has no exact-key proof')
    if audit.get('manifest_digest')!=manifest_digest(manifest):raise ValueError('Run integrity audit manifest_digest is stale')
    for proof in ('primary_model_evidence_checks','pixel_model_evidence_checks','input_hash_checks','source_hash_checks'):
        if proof not in audit:raise ValueError('Run integrity audit is missing proof: '+proof)
    provenance=require_run_provenance(raw,manifest)
    return dict(binding,exact_keys_valid=True,provenance=provenance)
