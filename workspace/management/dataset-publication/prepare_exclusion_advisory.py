"""Prepare a local exclusion overlay. Publication requires the root receipt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / 'advisory-v1-draft'
REGISTRY = PROJECT / 'management/dataset-review/split-registry-v1.json'
EXPECTED = '3fb4451a180cfbcec3ebf83d89a8e51f937460bb81b3bf62013e21dd7c0e6eaa'
BASE = Path('S:/CodexArtifacts/hacktheburgh/dataset-review-20261108-v1')
API = 'repos/NeelHaval/viridian402/releases/438114990'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


registry_bytes = REGISTRY.read_bytes()
assert digest(registry_bytes) == EXPECTED, 'Canonical registry hash mismatch'
registry = json.loads(registry_bytes.decode('utf-8'))
assert len(registry['protectedLogs']) == len(set(registry['protectedLogs'])) == 13
assert len(registry['protectedGroups']) == len(set(registry['protectedGroups'])) == 11
assert registry['metadataOnly'] is True and registry['frozenParentChanged'] is False
assert all(key in registry for key in ['version', 'aliases', 'evalMetadataSha256'])

publication = json.loads((HERE / 'publication-receipt.json').read_text(encoding='utf-8'))
interim = json.loads((HERE / 'interim-advisory-receipt.json').read_text(encoding='utf-8'))
manifest_bytes = (BASE / 'DATASET.json').read_bytes()
manifest_pin = next(a for a in publication['assets'] if a['name'] == 'DATASET.json')
assert 'sha256:' + digest(manifest_bytes) == manifest_pin['digest']
manifest = json.loads(manifest_bytes.decode('utf-8'))
frozen_path = 'management/dataset-review/freezes/labeller-first-20261108/inputs/labeller-inventory.json'
with zipfile.ZipFile(BASE / 'human-training-01.zip') as archive:
    frozen_bytes = archive.read(frozen_path)
assert digest(frozen_bytes) == manifest['files'][frozen_path]['sha256']
frozen = json.loads(frozen_bytes.decode('utf-8'))
assert frozen['summary']['heldoutExcluded'] is True

remote_result = subprocess.run(['gh', 'api', API], capture_output=True, encoding='utf-8')
if remote_result.returncode:
    raise RuntimeError(remote_result.stderr.strip())
remote = json.loads(remote_result.stdout)
fingerprints = {a['name']: dict(id=a['id'], size=a['size'], digest=a['digest'], state=a['state'])
                for a in remote['assets']}
assert fingerprints == interim['after'], 'Frozen release assets changed'

OUT.mkdir(exist_ok=True)
(OUT / 'split-registry-v1.json').write_bytes(registry_bytes)
overlay = dict(schema='effective-exclusion-overlay-v1', status='draft-awaiting-root-receipt',
               createdUtc=datetime.now(timezone.utc).isoformat(),
               rootReceiptReceived=False, rootConsumerTests='pending', canonicalIdentityFixes='pending',
               publicationPermitted=False, finalEligibilityEstablished=False,
               baseRelease=dict(url=publication['url'], tag=publication['tag'],
                                manifestSha256=digest(manifest_bytes), assets=fingerprints),
               registry=dict(file='split-registry-v1.json', sha256=EXPECTED,
                             version=registry['version'], protectedLogs=13, protectedWellGroups=11,
                             evalMetadataSha256=registry['evalMetadataSha256']),
               globalGate=dict(train=False, tune=False, labelGeneration=False, reviewConfirmation=False,
                               reason='Await root receipt and validated canonical identity consumers.'),
               protectedWellPolicy=dict(deny=['train', 'tune', 'label-generation', 'review-confirmation'],
                                        includeDescendants=True, matchWith='root-validated canonical identity consumer',
                                        allowFromFrozenReadyFlag=False),
               historicalClaimCorrection=dict(sourceAsset='human-training-01.zip',
                   sourceMember=frozen_path, sourceMemberSha256=digest(frozen_bytes),
                   field='summary.heldoutExcluded', recordedValue=True,
                   recordedValuePreserved=True, effectiveHeldoutExcluded=False,
                   reason='The frozen TRAIN subset contains protected evaluation well/page overlap. '
                          'The old flag does not establish effective well exclusion.',
                   cleanTrainingEligibility='not established', blindVerificationEligibility=False),
               exposurePolicy='Prior labels remain exposed. An exclusion overlay cannot restore blindness.',
               djsPolicy=dict(previousGalleryThumbnails='truth overlays; deny model/blind-review use',
                              sourceOnlyRepair='reported upstream; frozen snapshot not rewritten'),
               finalExcludedRecordIds=None, finalEligibleRecordIds=None,
               requiredRootReceipt=dict(path=None, sha256=None,
                   mustConfirm=['canonical registry SHA-256', 'canonical identity fixes',
                                'training and review-confirmation consumer tests',
                                'effective exclusions and affected frozen-record mapping']))
write(OUT / 'effective-exclusions-v1.draft.json', overlay)
advisory = f'''# Exclusion advisory v1 — draft

Publication is pending the root receipt. Root consumer tests and canonical
identity fixes are still in progress. Final training eligibility is not established.

The registry version is `{registry['version']}`. It lists 13 protected logs and
11 protected well groups. Its SHA-256 is `{EXPECTED}`.

Preserve the original release assets and parent freeze. Apply a separate,
versioned exclusion overlay after the root receipt confirms the consumers.
The registry denies training, tuning, label generation and review confirmation
for protected wells and their descendants. Use the root canonical identity
consumer to resolve aliases. Do not infer eligibility from a frozen `ready` flag.

The frozen human inventory records `summary.heldoutExcluded: true`.
That historical value remains unchanged. Its effective exclusion claim is false:
the frozen subset contains protected evaluation well/page overlap. The bundle
is not approved as clean eligible training data or blind verification. Prior
label exposure remains part of its history. Removing records cannot restore
blindness for previously exposed labels.

ASTRA-01 reported 12 current ready TRAIN pages across seven protected well
groups, including seven exact evaluation pages. These remain reported counts.
The final mapping to frozen release records is pending the root receipt.

Previous DJS gallery thumbnails contained truth overlays. Do not use them as
prediction inputs or blind-review evidence. The upstream source-only inventory
repair is not claimed for this frozen snapshot.

This draft is a local preparation artifact. It has not been attached to GitHub.
No source data, frozen labels, model files or original release assets were changed.
'''
(OUT / 'ADVISORY-v1.draft.md').write_text(advisory, encoding='utf-8')
pins = {p.name: digest(p.read_bytes()) for p in sorted(OUT.iterdir())
        if p.is_file() and p.name != 'PREPARATION-RECEIPT.json'}
assert (OUT / 'split-registry-v1.json').read_bytes() == registry_bytes
assert REGISTRY.read_bytes() == registry_bytes
assert (BASE / 'DATASET.json').read_bytes() == manifest_bytes
assert overlay['publicationPermitted'] is False and overlay['finalEligibilityEstablished'] is False
assert overlay['historicalClaimCorrection']['effectiveHeldoutExcluded'] is False
write(OUT / 'PREPARATION-RECEIPT.json', dict(status='draft-awaiting-root-receipt',
      registrySha256=EXPECTED, registryLogs=13, registryWellGroups=11,
      originalAssetsUnchanged=len(fingerprints), frozenInventorySha256=digest(frozen_bytes),
      rootReceiptReceived=False, attachmentPublished=False, finalEligibilityEstablished=False,
      files=pins))
print('ADVISORY_DRAFT_CHECK_OK registry_sha256=match logs=13 wells=11 '
      'original_assets_unchanged=13 root_receipt=pending attachment=not_published final_eligibility=not_established')
