"""Add an interim exclusion notice without changing frozen release assets."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
API = 'repos/NeelHaval/viridian402/releases/438114990'
MARKER = '<!-- dataset-exclusion-advisory-v1: interim -->'
NOTICE = '''<!-- dataset-exclusion-advisory-v1: interim -->
**URGENT: training eligibility and evaluation exposure advisory — registry pending.**

Do not treat `human-training-01.zip`, the `human-crops` selection, or the frozen
`ready` flags as approval for clean eligible training or blind verification.
ASTRA-01 reports that 12 current ready TRAIN pages share seven protected
evaluation well groups, including seven exact evaluation pages. These counts
are reported findings; the final exclusion registry and receipt are pending.
The frozen manifest's readiness flags do not apply the new well exclusions.

The previous DJS gallery thumbnails were truth overlays. Do not give these
thumbnails to a prediction model or use them for blind review. The outside
source-only inventory has been repaired upstream. That repair is not claimed
for this frozen snapshot.

All existing release assets remain unchanged. A separate versioned exclusion
registry and advisory will be attached after the final registry receipt is
available. Apply that registry before any training eligibility decision.

---

'''


def gh(*args):
    result = subprocess.run(['gh', *args], capture_output=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def fingerprints(release):
    return {a['name']: dict(id=a['id'], size=a['size'], digest=a['digest'], state=a['state'])
            for a in release['assets']}


before = gh('api', API)
original = json.loads((HERE / 'publication-receipt.json').read_text(encoding='utf-8'))
expected = {a['name']: a['digest'] for a in original['assets']}
assert {a['name']: a['digest'] for a in before['assets']} == expected
body = before['body']
if MARKER not in body:
    request = HERE / 'interim-notice-request.json'
    request.write_text(json.dumps(dict(body=NOTICE + body)), encoding='utf-8')
    gh('api', '--method', 'PATCH', API, '--input', str(request))
after = gh('api', API)
assert MARKER in after['body']
assert fingerprints(before) == fingerprints(after)
assert before['tag_name'] == after['tag_name'] == 'dataset-review-20261108-v1'
assert not after['draft']
pending = dict(schema='dataset-exclusion-advisory-v1', status='awaiting-final-registry',
               createdUtc=datetime.now(timezone.utc).isoformat(),
               releaseUrl=after['html_url'], reportedFindings=dict(trainPages=12,
               protectedWellGroups=7, exactEvaluationPages=7, source='owner advisory: outside ASTRA-01',
               independentlyVerified=False), registryPath=None, registrySha256=None, receiptPath=None,
               promotionGuard='Do not promote the frozen human bundle as clean eligible training or blind verification.',
               djsGuard='Previous gallery thumbnails were truth overlays; source-only upstream repair is not backported.',
               publicationRule='Attach a new versioned registry/advisory only after final registry receipt. Do not overwrite source assets.',
               frozenAssets=fingerprints(after))
(HERE / 'dataset-exclusion-advisory-v1.pending.json').write_text(
    json.dumps(pending, indent=2) + '\n', encoding='utf-8')
receipt = dict(status='interim-notice-published', releaseUrl=after['html_url'],
               checkedUtc=datetime.now(timezone.utc).isoformat(), assetsUnchanged=len(after['assets']),
               finalAttachmentPublished=False, finalRegistryReceived=False,
               before=fingerprints(before), after=fingerprints(after))
(HERE / 'interim-advisory-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(f'INTERIM_ADVISORY_OK assets_unchanged={len(after["assets"])} registry=pending attachment=not_published')
