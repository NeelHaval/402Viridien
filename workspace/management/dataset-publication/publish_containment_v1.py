"""Publish a separate containment advisory; never replace frozen assets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import types
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / 'containment-v1'
BASE = Path('S:/CodexArtifacts/hacktheburgh/dataset-review-20261108-v1')
REPO = 'NeelHaval/viridian402'
TAG = 'dataset-review-20261108-v1'
API = f'repos/{REPO}/releases/438114990'
ROOT_RECEIPT = 'management/dataset-review/packs/root-containment-v1.json'
ROOT_SHA = 'b5bcffb817cf137264cfd63494e092a76992b629c43e303bfba93148d8af029e'
REGISTRY_PATH = 'management/dataset-review/split-registry-v1.json'
REGISTRY_SHA = '3fb4451a180cfbcec3ebf83d89a8e51f937460bb81b3bf62013e21dd7c0e6eaa'
POLICY_PATH = 'management/dataset-review/split_policy.py'
FROZEN_PATH = 'management/dataset-review/freezes/labeller-first-20261108/inputs/labeller-inventory.json'
ROUTE_FILES = {f'management/dataset-review/inputs/outside-manifests/{name}.json' for name in
               ['ext_dev', 'ext_test', 'ext_shared_verify', 'mirror_lake', 'djs_idaho_test', 'frozen130_unlabelled']}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def gh(*args):
    result = subprocess.run(['gh', *args], capture_output=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def release():
    return json.loads(gh('api', API))


def fingerprints(info):
    return {a['name']: dict(id=a['id'], size=a['size'], digest=a['digest'], state=a['state'])
            for a in info['assets']}


def main():
    receipt_bytes = (PROJECT / ROOT_RECEIPT).read_bytes()
    assert sha(receipt_bytes) == ROOT_SHA, 'Root receipt hash mismatch'
    root = json.loads(receipt_bytes)
    checked, audit = {}, []
    for rel, expected in root['files'].items():
        path = (PROJECT / rel).resolve()
        path.relative_to(PROJECT)
        data = path.read_bytes()
        actual = sha(data)
        matched = actual == expected
        audit.append(dict(path=rel, receiptSha256=expected, currentSha256=actual,
                          matchesReceipt=matched, appliedByThisAttachment=False))
        if matched:
            checked[rel] = data
        elif rel not in ROUTE_FILES:
            raise ValueError('Required binding changed: ' + rel)
    drift = [r for r in audit if not r['matchesReceipt']]
    assert sha(checked[REGISTRY_PATH]) == REGISTRY_SHA
    reg = json.loads(checked[REGISTRY_PATH])
    assert sha((PROJECT / 'viridien_work/bakeoff/eval_set.json').read_bytes()) == reg['evalMetadataSha256']
    assert root['summary']['status'] == 'PASS' and root['parentFreezeChanged'] is False
    assert (len(reg['protectedLogs']), len(reg['protectedGroups'])) == (13, 11)
    print(f'ROOT_SCOPE_CHECK_OK matched={len(checked)} drifted_routes={len(drift)} '
          'route_updates_excluded=true', flush=True)

    source_receipt = json.loads((HERE / 'publication-receipt.json').read_text(encoding='utf-8'))
    interim = json.loads((HERE / 'interim-advisory-receipt.json').read_text(encoding='utf-8'))
    before = release()
    original_assets = interim['after']
    assert all(fingerprints(before).get(name) == value for name, value in original_assets.items())
    manifest_bytes = (BASE / 'DATASET.json').read_bytes()
    pin = next(a for a in source_receipt['assets'] if a['name'] == 'DATASET.json')
    assert 'sha256:' + sha(manifest_bytes) == pin['digest']
    manifest = json.loads(manifest_bytes)
    assert sha(checked[FROZEN_PATH]) == manifest['files'][FROZEN_PATH]['sha256']
    human = json.loads(checked[FROZEN_PATH])
    assert human['summary']['heldoutExcluded'] is True

    OUT.mkdir(exist_ok=True)
    authority = OUT / 'split-registry-v1.json'
    authority.write_bytes(checked[REGISTRY_PATH])
    policy = types.ModuleType('checked_root_split_policy')
    policy.__file__ = str(PROJECT / POLICY_PATH)
    exec(compile(checked[POLICY_PATH], policy.__file__, 'exec'), policy.__dict__)
    policy.REGISTRY = authority
    human_by_id = {r['id']: r for r in human['records']}
    decisions, eligible, excluded = [], [], []
    for record in manifest['records']:
        decision = policy.permission(human_by_id.get(record['id'], record), reg)
        is_human = record['selection'] == 'human-crops'
        if is_human:
            (eligible if decision['allowTrain'] else excluded).append(record['id'])
        decisions.append(dict(recordId=record['id'], selection=record['selection'],
            originalReady=record['ready'], groups=decision['groups'],
            protectedGroups=decision['protectedGroups'], protectedWell=decision['protectedWell'],
            denyTrain=not decision['allowTrain'], denyTune=not decision['allowTune'],
            denyLabelGeneration=not decision['allowLabelGeneration'], denyReviewConfirmation=not decision['allowConfirm'],
            protectedWellScreenPassed=decision['allowTrain'],
            cleanTrainingCertified=False, independentAccuracyClaimAllowed=False,
            blindVerificationCertified=False, reason=decision['reason']))
    human_decisions = [r for r in decisions if r['selection'] == 'human-crops']
    excluded_wells = {g for r in human_decisions if r['recordId'] in excluded for g in r['protectedGroups']}
    protected_logs = {x.casefold() for x in reg['protectedLogs']}
    exact = sum(Path(human_by_id[x]['image'].replace('\\', '/')).stem.casefold() in protected_logs for x in excluded)
    assert (len(eligible), len(excluded), len(excluded_wells), exact) == (15, 12, 7, 7)
    assert root['summary']['eligibleHumanBoxes'] == 422 and root['summary']['excludedHumanBoxes'] == 36
    assert set(eligible) | set(excluded) == set(human_by_id)
    assert set(eligible).isdisjoint(excluded)
    print('HUMAN_OVERLAY_CHECK_OK eligible_pages=15 excluded_pages=12 protected_wells=7 exact_pages=7', flush=True)

    binding_audit = dict(schema='containment-binding-audit-v1', rootReceiptSha256=ROOT_SHA,
        matchedBindings=len(checked), changedOutsideRouteBindings=len(drift), bindings=audit,
        scope='Exclusion overlay and advisory only. No suite/scorer/route update is applied.',
        futureUpdateGate='Changed route manifests, scorer v2 and DJS v2 require a separately checked receipt.')
    overlay = dict(schema='dataset-containment-overlay-v1', status='checked-exclusion-overlay',
        createdUtc=datetime.now(timezone.utc).isoformat(), appliesTo=dict(release=source_receipt['url'],
        frozenManifestSha256=sha(manifest_bytes), originalAssets=original_assets),
        rootReceipt=dict(sha256=ROOT_SHA, evidenceMember=ROOT_RECEIPT, wholeLiveSnapshotVerified=not drift),
        authority=dict(version=reg['version'], sha256=REGISTRY_SHA, evidenceMember=REGISTRY_PATH,
                       protectedLogs=13, protectedWells=11, policySha256=sha(checked[POLICY_PATH])),
        scope='Protected-well exclusions and evidence disclosure. Other training constraints still apply.',
        historicalCorrection=dict(field='summary.heldoutExcluded', sourceMember=FROZEN_PATH,
            sourceMemberSha256=sha(checked[FROZEN_PATH]), recordedValue=True, recordedValuePreserved=True,
            effectiveHeldoutExcluded=False, parentFreezeChanged=False,
            reason='The unfiltered parent contains protected evaluation wells and previously exposed labels.'),
        humanSelection=dict(parentPages=27, parentBoxes=458, eligiblePages=15, eligibleBoxes=422,
            excludedPages=12, excludedBoxes=36, excludedProtectedWells=7, excludedExactEvalPages=7,
            includedRecordIds=eligible, excludedRecordIds=excluded,
            boxCountSource='Checked root receipt; no protected truth was opened to recount boxes.'),
        currentCatalogSnapshot=dict(ready=351, protected=145, units='mixed records and versions; not independent samples'),
        djs=dict(sourceOnlyPreviewCount=25, priorTruthOverlayExposureRecorded=True,
            frozenGalleryRepairApplied=False, blindVerificationCertified=False),
        silverLabels=dict(status='unreviewed two-source silver labels', tieBreak='source A', aiConsensus=False),
        metricGuards=dict(missingTruthDepth='depth accuracy unavailable',
            neptuneEmptyTruth='row accuracy unavailable; exclude from accuracy means',
            independentAccuracyClaimAllowed=False),
        routeUpdatesApplied=False, driftedOutsideRoutes=drift,
        pending='Scorer v2 and DJS v2 require a future separately checked snapshot.',
        modelChanges=False, modelInferenceRun=False, modelApiCalls=False,
        jevNewCallsAllowed=False, originalReadyFlagsRewritten=False, records=decisions)
    overlay_name = 'dataset-containment-overlay-v1.json'
    advisory_name = 'dataset-containment-advisory-v1.md'
    evidence_name = 'dataset-containment-evidence-v1.zip'
    assets_name = 'dataset-containment-assets-v1.json'
    (OUT / overlay_name).write_bytes(encode(overlay))
    advisory = f'''# Dataset containment advisory v1

This advisory applies to `{TAG}`. It adds a separate exclusion overlay. All 13
original release assets and the 27-page, 458-box parent freeze remain unchanged.

The frozen inventory's `heldoutExcluded: true` claim is retained as history.
Its effective exclusion claim is false. The unfiltered human bundle includes
12 pages and 36 approved boxes from seven protected evaluation wells. Seven
are exact evaluation pages. The overlay selects 15 pages and 422 boxes after
the protected-well exclusion check. This is a human diagnostic subset, not a
certified clean training set or blind verification set. Prior exposure remains.

Apply the overlay's included record IDs before using the human selection.
Deny training, tuning, label generation and review confirmation for protected
wells and descendants. Unknown well identity also denies those uses. A frozen
`ready` flag does not grant permission. Passing the well screen does not remove
other source, fold, label or reuse restrictions. Existing unguarded adapters do
not enforce this sidecar automatically; their callers must apply the overlay.

Authority: `{reg['version']}`, 13 logs and 11 well groups.
Registry SHA-256: `{REGISTRY_SHA}`.
Root correction receipt SHA-256: `{ROOT_SHA}`.
The human record split was reproduced with the hash-pinned root identity policy.
The 422/36 box counts are bound to the checked root receipt. No protected truth
was opened for a new count. No model changes, inference or model API calls ran.

The checked current catalog reports 351 ready records and 145 protected records.
These mixed records and versions are not independent samples. These are upstream
catalog counts, not replacement counts for the frozen release manifest.

All 25 corrected DJS previews are source-only in the checked upstream catalog.
Previous DJS gallery thumbnails were truth overlays. That exposure remains
recorded. This sidecar does not replace or repair the frozen gallery assets.
Do not use old truth overlays as prediction inputs or blind-review evidence.

Current silver labels are unreviewed two-source labels with source A used on
ties. They are not AI consensus or human-certified truth. Missing truth depth
columns make depth accuracy unavailable. Neptune's empty truth makes row
accuracy unavailable; exclude it from accuracy means. No independent accuracy
claim follows from this containment correction. Retain the source caveats.

Binding audit: {len(checked)} of 31 current files match the receipt. Six outside
route manifests have changed since that snapshot. Their expected and current
hashes are recorded in the evidence. Those route changes are excluded from this
attachment and are not approved here. Scorer v2 and DJS v2 remain pending a
future separately checked receipt. This publication grants no new JEV API calls.

The evidence ZIP contains the exact root receipt, registry, reference identity
policy, split-check result, JEV binding, caveats and binding audit. It does not
install code or replace dataset files. Keep it separate from model inputs.
'''
    (OUT / advisory_name).write_text(advisory, encoding='utf-8')
    evidence_members = {ROOT_RECEIPT: receipt_bytes, 'binding-audit-v1.json': encode(binding_audit)}
    for rel in [REGISTRY_PATH, POLICY_PATH, FROZEN_PATH,
                'management/dataset-review/packs/split-enforcement-check.json',
                'management/dataset-review/packs/jev68-binding-v1.json',
                'management/dataset-review/CAVEATS.md']:
        evidence_members[rel] = checked[rel]
    with zipfile.ZipFile(OUT / evidence_name, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in evidence_members.items():
            archive.writestr('dataset-containment-v1/' + name, data)
    with zipfile.ZipFile(OUT / evidence_name) as archive:
        for name, data in evidence_members.items():
            assert archive.read('dataset-containment-v1/' + name) == data
    new_names = [advisory_name, overlay_name, evidence_name, assets_name]
    artifact_manifest = dict(schema='dataset-containment-assets-v1', rootReceiptSha256=ROOT_SHA,
        registrySha256=REGISTRY_SHA, artifacts=[dict(name=name, bytes=(OUT/name).stat().st_size,
        sha256=sha((OUT/name).read_bytes())) for name in new_names[:-1]],
        evidenceMembers={name: sha(data) for name, data in evidence_members.items()})
    (OUT / assets_name).write_bytes(encode(artifact_manifest))
    expected = {name: dict(size=(OUT/name).stat().st_size, digest='sha256:'+sha((OUT/name).read_bytes())) for name in new_names}
    for name in new_names:
        existing = fingerprints(release()).get(name)
        if existing:
            assert existing['digest'] == expected[name]['digest'], 'Versioned attachment already differs: ' + name
        else:
            gh('release', 'upload', TAG, str(OUT/name), '--repo', REPO)
        print('ATTACHMENT_UPLOADED ' + name, flush=True)
    uploaded = release()
    current = fingerprints(uploaded)
    assert all(current.get(name) == value for name, value in original_assets.items())
    for name, info in expected.items():
        assert current[name]['digest'] == info['digest'] and current[name]['size'] == info['size']
        assert current[name]['state'] == 'uploaded'
    old_body = before['body'].replace('\r\n', '\n')
    (OUT / 'release-description-before.txt').write_text(old_body, encoding='utf-8')
    if old_body.startswith('<!-- dataset-exclusion-advisory-v1: interim -->'):
        old_body = old_body.split('\n---\n\n', 1)[1]
    download = f'https://github.com/{REPO}/releases/download/{TAG}'
    notice = f'''<!-- dataset-containment-advisory-v1: checked -->
**Checked containment advisory v1 — apply the separate exclusion overlay.**

The unfiltered human bundle is not approved as clean training or blind verification.
The parent has 27 pages and 458 boxes; the derived human diagnostic selects
15 pages and 422 boxes, and excludes 12 pages and 36 boxes from seven protected wells.
The old `heldoutExcluded: true` claim is retained as history and corrected by this sidecar.

Read the [advisory]({download}/{advisory_name}), apply the
[overlay]({download}/{overlay_name}), and retain the
[evidence]({download}/{evidence_name}) and [checksums]({download}/{assets_name}).
All 13 original assets remain unchanged. DJS prior truth-overlay exposure remains
recorded. Silver labels are unreviewed; missing depth and empty Neptune truth
make their affected metrics unavailable. This is not an independent accuracy claim.
Six changed outside route manifests are excluded from this attachment.
Scorer v2 and DJS v2 remain pending a future checked update.

---

'''
    request = OUT / 'release-description-request.json'
    request.write_bytes(encode(dict(body=notice + old_body)))
    gh('api', '--method', 'PATCH', API, '--input', str(request))
    after = release()
    assert '<!-- dataset-containment-advisory-v1: checked -->' in after['body']
    assert all(fingerprints(after).get(name) == value for name, value in original_assets.items())
    write_receipt = dict(status='advisory-and-overlay-published', releaseUrl=after['html_url'],
        rootReceiptSha256=ROOT_SHA, registrySha256=REGISTRY_SHA, originalAssetsUnchanged=13,
        newAssets={name:fingerprints(after)[name] for name in new_names},
        humanEligiblePages=15, humanEligibleBoxes=422, humanExcludedPages=12, humanExcludedBoxes=36,
        matchedRootBindings=len(checked), changedRouteBindingsExcluded=len(drift),
        independentAccuracyClaim=False, modelInferenceRun=False, modelApiCalls=False)
    (OUT / 'PUBLICATION-RECEIPT.json').write_bytes(encode(write_receipt))
    print('CONTAINMENT_PUBLICATION_OK new_assets=4 original_assets_unchanged=13 '
          'human_pages=15 human_boxes=422 excluded_pages=12 excluded_boxes=36 '
          f'changed_routes_excluded={len(drift)} independent_accuracy_claim=false', flush=True)


if __name__ == '__main__':
    main()
