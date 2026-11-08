"""Attach checked v2 metadata and caveats without replacing dataset assets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / 'metadata-v2'
REPO = 'NeelHaval/viridian402'
TAG = 'dataset-review-20261108-v1'
API = f'repos/{REPO}/releases/438114990'
RECEIPT = 'management/dataset-review/packs/root-containment-v2.json'
EXPECTED = '852491616c54b03774c5de51ecb2a5d235edd8bba840cb6a2189e4318979e0a6'


def sha(data): return hashlib.sha256(data).hexdigest()
def encoded(value): return (json.dumps(value, indent=2) + '\n').encode('utf-8')
def gh(*args):
    result = subprocess.run(['gh', *args], capture_output=True, encoding='utf-8')
    if result.returncode: raise RuntimeError(result.stderr.strip())
    return result.stdout


receipt_bytes = (PROJECT / RECEIPT).read_bytes()
assert sha(receipt_bytes) == EXPECTED
receipt = json.loads(receipt_bytes)
snapshot = {}
for path, expected in receipt['files'].items():
    data = (PROJECT / path).read_bytes()
    assert sha(data) == expected, 'V2 binding changed: ' + path
    snapshot[path] = data
summary = receipt['summary']
assert summary['extDevAutoLabelAllowed'] is False and summary['newApiCallsAllowed'] is False
assert summary['originalV1Preserved'] and summary['parentFreezeUnchanged']
assert (summary['outsideRecords'], summary['outsideDistinctPhysicalPages'], summary['revisionPairs']) == (396,257,130)
before = json.loads(gh('api', API))
frozen = {a['name']: (a['id'],a['digest'],a['size']) for a in before['assets']}
OUT.mkdir(exist_ok=True)
policy = dict(schema='local-dataset-policy-v2', rootReceiptSha256=EXPECTED,
    autoLabelAllowed=False, modelApiCallsAllowed=False, implicitInferenceAllowed=False,
    protectedRegistrySha256=receipt['files']['management/dataset-review/split-registry-v1.json'],
    catalogRecords=1399, catalogReady=481, outsideVersionRecords=396,
    outsidePhysicalPages=257, revisionPairs=130, parentBundlesUnchanged=True,
    availableDownloads='Frozen v1 payloads plus this v2 metadata. No v2 data replacement is included.',
    djsDepthAvailability=dict(available=19,unavailable=5,wrongFit=1),
    frozen130DepthUnits='OPEN', synthV2Accepted=False, synthGate='OPEN',
    independentAccuracyClaimAllowed=False, priorExposurePreserved=True)
archive = OUT / 'dataset-metadata-v2.zip'
members = {RECEIPT:receipt_bytes, 'POLICY.json':encoded(policy)}
for path,data in snapshot.items():
    if ('outside-manifests/' in path or path.endswith(('CAVEATS.md','split-registry-v1.json',
        'root-containment-v1.json','jev68-binding-v1.json','outside-inventory.json','outside-revision-check-v2.json'))):
        members[path] = data
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path,data in members.items(): bundle.writestr('dataset-metadata-v2/'+path,data)
with zipfile.ZipFile(archive) as bundle:
    for path,data in members.items(): assert bundle.read('dataset-metadata-v2/'+path) == data
advisory = OUT / 'dataset-metadata-advisory-v2.md'
advisory.write_text(f'''# Checked metadata advisory v2

Root receipt SHA-256: `{EXPECTED}`. All {len(snapshot)} bindings matched at publication.
The original payloads, parent freeze, registry, JEV binding and v1 receipts remain unchanged.
This attachment contains metadata and caveats only. It does not replace images,
annotations, labels, scorers or the existing payload ZIP files.

The current catalog has 1,399 records and 481 ready records. The outside inventory
has 396 version records on 257 physical pages, including 130 original/revision
pairs. Do not count the two 130-page versions as 260 independent pages.

DJS v2 changes depth metadata; image content, lithology labels and y coordinates
are unchanged. Five depth calibrations remain unavailable and one has a wrong
fit. Prior truth-overlay exposure remains recorded. The dedicated DJS v2 metadata
does not turn the old frozen DJS payload into a checked v2 payload.

The original 130-page depth units remain OPEN. Synthetic v2 is unaccepted and its
gate is OPEN. DEV autoLabelAllowed is false. The unchanged JEV binding grants no
new calls. This update authorizes no inference, prelabelling or model API calls.
No independent accuracy claim is made. Keep protected truth isolated and retain
the v1 well exclusions and all prior exposure history.
''', encoding='utf-8')
assets = OUT / 'dataset-metadata-assets-v2.json'
assets.write_bytes(encoded(dict(rootReceiptSha256=EXPECTED, artifacts=[dict(name=p.name,
    bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in [archive,advisory]],
    members={path:sha(data) for path,data in members.items()})))
for path in [archive,advisory,assets]:
    existing = next((a for a in before['assets'] if a['name']==path.name),None)
    if existing: assert existing['digest']=='sha256:'+sha(path.read_bytes()), 'Versioned asset differs'
    else: gh('release','upload',TAG,str(path),'--repo',REPO)
after = json.loads(gh('api', API))
remote = {a['name']:a for a in after['assets']}
assert all((remote[n]['id'],remote[n]['digest'],remote[n]['size'])==value for n,value in frozen.items())
for path in [archive,advisory,assets]:
    assert remote[path.name]['digest']=='sha256:'+sha(path.read_bytes())
    assert remote[path.name]['size']==path.stat().st_size
body = ('<!-- dataset-metadata-advisory-v2: checked -->\n'
    '**Checked metadata v2:** [advisory](https://github.com/'+REPO+'/releases/download/'+TAG+'/'+advisory.name+'). '
    'Metadata only; original payloads are unchanged. DJS: five depth limits and one wrong fit remain. '
    'Original/revision 130-page sets are paired versions. DEV auto-labelling is disabled. '
    'Depth units and synth v2 gates remain open. No new inference or JEV authority, or independent accuracy claim.\n\n'
    + before['body'])
request = OUT/'request.json'; request.write_bytes(encoded(dict(body=body)))
gh('api','--method','PATCH',API,'--input',str(request))
final = json.loads(gh('api',API)); final_assets={a['name']:a for a in final['assets']}
assert all((final_assets[n]['id'],final_assets[n]['digest'],final_assets[n]['size'])==v for n,v in frozen.items())
(OUT/'PUBLICATION-RECEIPT.json').write_bytes(encoded(dict(status='published',rootReceiptSha256=EXPECTED,
    previousAssetsUnchanged=len(frozen),newAssets=[remote[p.name] for p in [archive,advisory,assets]],
    modelInferenceRun=False,modelApiCalls=False,independentAccuracyClaim=False)))
print(f'METADATA_V2_PUBLICATION_OK bindings={len(snapshot)} previous_assets_unchanged={len(frozen)} new_assets=3 inference=false')
