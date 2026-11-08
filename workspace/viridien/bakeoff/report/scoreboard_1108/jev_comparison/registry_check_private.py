"""Metadata-only audit. This tool cannot authorize an API launch."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
REGISTRY = ROOT / 'management/dataset-review/split-registry-v1.json'
BINDING = ROOT / 'management/dataset-review/packs/jev68-binding-v1.json'
REGISTRY_SHA = '3fb4451a180cfbcec3ebf83d89a8e51f937460bb81b3bf62013e21dd7c0e6eaa'
BINDING_SHA = '88cf4afd526b47ba200f6498d2e54a1c24642dc79ed1c98e47b28187ea962409'
COUNTS = {'exclude': 15, 'unknown': 2, 'allow': 51}


def load_pinned(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('File hash mismatch: ' + str(path))
    return json.loads(data)


def validate(registry, binding, metadata):
    if binding['authoritySha256'] != REGISTRY_SHA or binding['newApiLaunchAllowed'] is not False:
        raise ValueError('Authority or global hold mismatch')
    entries = binding['entries']
    records = metadata['records']
    ids = [item['caseId'] for item in entries]
    record_ids = [item['caseId'] for item in records]
    if len(ids) != 68 or len(set(ids)) != 68 or len(record_ids) != 68 or len(set(record_ids)) != 68 or set(ids) != set(record_ids):
        raise ValueError('Expected the same 68 distinct case identities')
    counts = dict(Counter(item['decision'] for item in entries))
    if counts != COUNTS or binding['counts'] != COUNTS:
        raise ValueError('Decision counts differ from pinned policy')
    by_id = {item['caseId']: item for item in records}
    protected = set(registry['protectedGroups'])
    for entry in entries:
        record = by_id[entry['caseId']]
        for field in ('imagePath', 'imageSha256', 'annotationSha256', 'window'):
            if entry[field] != record[field]:
                raise ValueError('Identity mismatch for ' + entry['caseId'] + ': ' + field)
        if entry['apiLaunchAllowed'] is not False:
            raise ValueError('Per-case launch hold missing')
        if entry['decision'] == 'allow':
            ancestors = entry['ancestorGroupIds']
            if (record['developmentDesignation'] != 'DEV' or record['suite'] != 'ext_dev'
                    or not ancestors or entry['groupId'] not in ancestors
                    or protected.intersection(ancestors)
                    or entry['allowedPurposes'] != ['jev-dev-prelabel']):
                raise ValueError('Allowed entry lacks safe DEV or lineage binding')
        elif entry['allowedPurposes']:
            raise ValueError('Excluded or unknown entry has an allowed purpose')
    return {'counts': counts, 'cases': len(ids), 'api_launch_authorized': False}


def main():
    registry = load_pinned(REGISTRY, REGISTRY_SHA)
    binding = load_pinned(BINDING, BINDING_SHA)
    metadata = load_pinned(Path(binding['sourceMetadataPath']), binding['sourceMetadataSha256'])
    if metadata['sourceManifestSha256'] != binding['sourceManifestSha256']:
        raise ValueError('Manifest authority mismatch')
    load_pinned(Path(metadata['sourceManifestPath']), binding['sourceManifestSha256'])
    print(json.dumps(validate(registry, binding, metadata)))


if __name__ == '__main__':
    main()
