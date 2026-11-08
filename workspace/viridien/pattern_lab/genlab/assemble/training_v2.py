"""Training-only records and label-free exclusions; never read the mixed index.

The builder consumes per-document record shards, selected by policy BEFORE open.
An upstream annotation exporter must supply these shards and a label-free pixel
exclusion index. Do not reconstruct duplicate-of-test decisions from test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'data/genlab/fasttrack/training-v2.json'
EXCLUSIONS = ROOT/'data/genlab/fasttrack/exclusions-v2.json'
# Single swap point for the future root-owned versioned registry.
PROTECTED_REGISTRY = ROOT.parents[1]/'management/dataset-review/split-registry-v1.json'
PROTECTED_REGISTRY_SHA256 = '3fb4451a180cfbcec3ebf83d89a8e51f937460bb81b3bf62013e21dd7c0e6eaa'


def protected_registry():
    raw = PROTECTED_REGISTRY.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTECTED_REGISTRY_SHA256:
        raise ValueError('Protected registry changed; versioned review required')
    value = json.loads(raw)
    if value.get('version') == 'protected-wells-v1':
        groups = value['protectedGroups']
        value = dict(protected_groups=groups, aliases=value['aliases'],
                     protected_document_ids=[d for d, g in value['aliases'].items() if g in groups])
    if not value.get('protected_document_ids') or not value.get('protected_groups'):
        raise ValueError('Protected registry missing document/group exclusions')
    return value


def require_unprotected(did, policy):
    registry = protected_registry()
    group = policy['document_groups'].get(did)
    if (did in registry['protected_document_ids'] or group in registry['protected_groups'] or
            registry.get('aliases', {}).get(group) in registry['protected_groups']):
        raise ValueError('Protected evaluation ancestry forbidden for training')


def check_lineage(lineage, policy=None):
    """Training gate: protected ancestry and split checks are independent."""
    policy = policy or json.loads((ROOT/'data/normalized/split-policy.json').read_text(encoding='utf8'))
    roles = {'base', 'texture', 'sequence', 'calibration'}
    if not isinstance(lineage, dict) or set(lineage) != roles or any(
            not isinstance(lineage[k], list) or not lineage[k] for k in roles):
        raise ValueError('Complete typed ancestry required')
    if len(lineage['base']) != 1:
        raise ValueError('Exactly one base required')
    ids = sorted(set(sum(lineage.values(), [])))
    # Check the entire protected union before doing the separate split check.
    for did in ids:
        require_unprotected(did, policy)
    return {did: require_training_id(did, policy) for did in ids}


def lineage_eligibility(lineage, policy=None):
    """Metadata only; a diagnostic decision requires a separate bound review."""
    policy = policy or json.loads((ROOT/'data/normalized/split-policy.json').read_text(encoding='utf8'))
    registry = protected_registry()
    ids = sorted(set(sum(lineage.values(), [])))
    groups = sorted({policy['document_groups'].get(d) for d in ids} - {None})
    protected = bool(set(ids) & set(registry['protected_document_ids']) or
                     set(groups) & set(registry['protected_groups']))
    try:
        check_lineage(lineage, policy)
        eligible = True
    except ValueError:
        eligible = False
    base = lineage.get('base', [None])[0]
    group = policy['document_groups'].get(base)
    return dict(trainingEligible=eligible, diagnosticEvalEligible=False,
                protectedDescendant=protected, groupId=group, baseWell=group,
                ancestor_documents=ids, ancestor_wells=groups,
                protected_registry_sha256=hashlib.sha256(PROTECTED_REGISTRY.read_bytes()).hexdigest())


def split_for_id(did, policy):
    group = policy['document_groups'].get(did)
    if group is None:
        raise ValueError('Unknown document ancestry')
    if int(hashlib.sha256((policy['seed']+':'+group).encode()).hexdigest()[:8], 16) % 5 == 0:
        return group, 'test'
    n = int(hashlib.sha256(('genlab-validation-v1:'+group).encode()).hexdigest()[:8], 16)
    return group, 'validation' if n % 5 == 0 else 'train'


def require_training_id(did, policy=None):
    policy = policy or json.loads((ROOT/'data/normalized/split-policy.json').read_text(encoding='utf8'))
    require_unprotected(did, policy)
    group, split = split_for_id(did, policy)
    if split != 'train':
        raise ValueError('Only training ancestry may be opened')
    return group


def training_id(did, policy=None):
    try:
        require_training_id(did, policy)
        return True
    except ValueError:
        return False


def load_export(path=None, policy=None):
    path = Path(path or os.environ.get('GENLAB_TRAINING_EXPORT', DEFAULT))
    policy = policy or json.loads((ROOT/'data/normalized/split-policy.json').read_text(encoding='utf8'))
    value = json.loads(path.read_text(encoding='utf8'))
    if value.get('schema') != 'generator-training-v2':
        raise ValueError('Expected audited training-only export; run training_v2 builder')
    signature = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    if value.get('policy_sha256') != signature:
        raise ValueError('Training export split policy changed')
    if value.get('protected_registry_sha256') != hashlib.sha256(PROTECTED_REGISTRY.read_bytes()).hexdigest():
        raise ValueError('Training export protected registry changed; rebuild export')
    for row in value['records']:
        require_training_id(row['document_id'], policy)
        if row.get('split') != 'train':
            raise ValueError('Non-training record in training export')
    return value


def read_json(path):
    path = Path(path)
    if path.resolve() == (ROOT/'data/normalized'/('current'+'.json')).resolve():
        raise ValueError('Mixed normalized index forbidden')
    if path.name == 'training-v2.json':
        return load_export(None if path.resolve()==DEFAULT.resolve() else path)
    return json.loads(path.read_text(encoding='utf8'))


class TrainingPolicy:
    """Read-only replacement for the labeller's mixed-index SplitPolicy."""
    def __init__(self,store):
        self.config=json.loads((ROOT/'data/normalized/split-policy.json').read_text(encoding='utf8'))
        self.rows={r['sample_id']:r for r in load_export()['records']}
    def load_index(self):pass
    def split(self,doc):return split_for_id(doc['id'],self.config)[1]
    def group(self,doc):return split_for_id(doc['id'],self.config)[0]
    def allowed(self,doc,segment):
        from normalized import sample_id
        from review_queue import version
        require_training_id(doc['id'],self.config)
        row=self.rows.get(sample_id(doc,segment))
        return bool(row and row['annotation_version']==version(segment) and row.get('usable_for_generation'))


def load_exclusions(path=None):
    path = Path(path or os.environ.get('GENLAB_EXCLUSION_INDEX', EXCLUSIONS))
    value = json.loads(path.read_text(encoding='utf8'))
    if value.get('schema') != 'generator-exclusions-v2':
        raise ValueError('Expected label-free exclusion index')
    if set(value) - {'schema', 'policy_sha256', 'denied_pixel_sha256', 'denied_boxes', 'reserved_spans'}:
        raise ValueError('Exclusion index contains non-contract fields')
    signature = hashlib.sha256(json.dumps(json.loads((ROOT/'data/normalized/split-policy.json').read_text(encoding='utf8')), sort_keys=True).encode()).hexdigest()
    if value.get('policy_sha256') != signature:
        raise ValueError('Exclusion split policy changed')
    return value


def pixels_allowed(rgb):
    digest = hashlib.sha256(f'{rgb.shape[1]}x{rgb.shape[0]}:'.encode()+rgb.tobytes()).hexdigest()
    return digest not in set(load_exclusions()['denied_pixel_sha256'])


def lineage_allowed(proposal, docs, policy=None):
    """Enforce split, current training target, revision and pixel quarantine."""
    try:
        current = {r['sample_id']:r for r in load_export()['records']}
        denied = set(load_exclusions()['denied_pixel_sha256'])
        seen, terminal = set(), []
        def visit(node):
            if not isinstance(node, dict) or id(node) in seen:
                raise ValueError('Unknown or cyclic ancestry')
            seen.add(id(node))
            for did in node.get('ancestor_docs', []): require_training_id(did, policy)
            did = node.get('doc_id') or node.get('document_id') or node.get('real_doc_id')
            if did:
                require_training_id(did, policy)
                if did not in docs: raise ValueError('Unresolved ancestry')
            for ref in node.get('normalized_sources', []):
                row = current.get(ref['sample_id'])
                if not row or not row.get('usable_for_generation') or row.get('quality_issues'):
                    raise ValueError('Revoked source')
                for key in ('annotation_version','legend_version','pixel_sha256','box_xyxy'):
                    if ref.get(key) != row.get(key): raise ValueError('Changed source')
            children = []
            for key in ('lineage','donor','recovery','provenance','render_model'):
                value = node.get(key)
                children += value if isinstance(value,list) else [value] if isinstance(value,dict) else []
            if children:
                for child in children: visit(child)
            elif did:
                if not node.get('pixel_sha256') or node['pixel_sha256'] in denied:
                    raise ValueError('Unbound or quarantined terminal source')
                if node.get('document_revision', node.get('source_revision')) != docs[did]['revision']:
                    raise ValueError('Changed document')
                terminal.append(did)
            seen.remove(id(node))
        visit(proposal)
        return bool(terminal)
    except (ValueError,KeyError,FileNotFoundError):
        return False


def build(records_dir, exclusion_index, output, policy):
    """Open only <train document ID>.json. Never enumerate or parse held shards."""
    records = [];manifest=[];stretches=[]
    for did in sorted(policy['document_groups']):
        if split_for_id(did, policy)[1] != 'train': continue
        try:
            require_unprotected(did, policy)
        except ValueError:
            continue
        file = Path(records_dir)/(did+'.json')
        if not file.exists(): continue
        shard = json.loads(file.read_text(encoding='utf8'))
        for row in shard['records']:
            if row['document_id'] != did or row.get('split') != 'train':
                raise ValueError('Invalid training shard; no output written')
            records.append(row)
        for key,dest in (('manifest',manifest),('stretches',stretches)):
            for row in shard.get(key,[]):
                if row['document_id']!=did or row.get('split')!='train':
                    raise ValueError('Non-training annotation in training shard')
                dest.append(row)
    if not records:
        raise ValueError('No training record shards supplied')
    # Label-free information comes from the annotation/export owner, never labels.
    exclusions = json.loads(Path(exclusion_index).read_text(encoding='utf8'))
    allowed = {'denied_pixel_sha256', 'denied_boxes', 'reserved_spans'}
    if set(exclusions)-allowed or not allowed.issubset(exclusions):
        raise ValueError('Expected label-free pixel hashes, boxes and reserved spans only')
    denied = set(exclusions['denied_pixel_sha256'])
    for row in records:
        if row['pixel_sha256'] in denied:
            row['usable_for_generation'] = False
            row['quality_issues'] = sorted(set(row.get('quality_issues', [])+['pixel_exclusion']))
    signature = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    value = dict(schema='generator-training-v2', policy_sha256=signature, records=records)
    value['protected_registry_sha256'] = hashlib.sha256(PROTECTED_REGISTRY.read_bytes()).hexdigest()
    value['snapshot'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    if not any(output.resolve().is_relative_to(p.resolve()) for p in
               (ROOT/'data/genlab/fasttrack',ROOT/'data/genlab/scratch/generators')):
        raise ValueError('Training export output must stay in fasttrack or generator scratch')
    output.mkdir(parents=True, exist_ok=True)
    for name, obj in [('training-v2.json', value), ('exclusions-v2.json',
                     dict(exclusions, schema='generator-exclusions-v2', policy_sha256=signature))]:
        with (output/name).open('w', encoding='utf8') as stream:
            for chunk in json.JSONEncoder(sort_keys=True).iterencode(obj): stream.write(chunk)
    if manifest or stretches:
        (output/'training-inputs').mkdir(exist_ok=True)
        with (output/'training-inputs/manifest.jsonl').open('w',encoding='utf8') as stream:
            for row in manifest:stream.write(json.dumps(row)+'\n')
        with (output/'training-inputs/stretches.json').open('w',encoding='utf8') as stream:
            for chunk in json.JSONEncoder().iterencode(stretches):stream.write(chunk)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--records-dir', required=True)
    p.add_argument('--exclusion-index', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    policy = json.loads((ROOT/'data/normalized/split-policy.json').read_text(encoding='utf8'))
    build(args.records_dir, args.exclusion_index, args.output, policy)


if __name__ == '__main__': main()
