# Dataset limits after the v1.1 owner report

The frozen v1.1 release remains **69 ready pages for diagnostics only**. Five empty Gulf pages and eight MC765 pages remain blocked or withdrawn. These notes do not change inputs, labels, scores or inventories. A future synth-v2 release needs a separate review and explicit acceptance.

The owner reports these additional limits:

- Hadrian has 10,072 scored rows painted over blank edge margins. The reported release row-accuracy effect is at most 0.39–0.47 percentage points. This is a release-level bound, not a measured model gain.
- Nine scored rows on three pages have no supporting synthetic pixels. The affected pages are `synth-v1-geoscale-page-004`, `synth-v1-geoscale-page-023` and `synth-v1-geoseq-page-002`.
- Eleven of 18 Gulf pages use 104,489 pixels from the opposing training fold's reserved stretch. This is training-fold exposure. No test-well pixel leakage was reported.
- One-off pages preserve the reserved source interval `[6624,7648)`. **Do not train on these pages before evaluation on that reserved real span.** The same restriction applies to exposed Gulf fold spans. An ignore CSV does not hide image pixels from a model.
- A constructed attack showed that `validate_v1` can miss truth labels that were changed. No such CSV mislabel was observed in the release. Validator success alone does not certify semantic truth.
- Three reviewer sessions accessed the mixed `normalized/current.json` index, which contains test labels. No measurement use was reported. We cannot certify a no-test-access audit. This update did not open that index or any test labels.

The existing fast gate uses two Hadrian geoseq windows, two Gulf windows, two One-off windows and two outside DEV windows. A metadata-only check found that neither Hadrian window intersects the reported blank margins, and none of its selected pages is one of the three unsupported-row pages. Neither One-off input window intersects `[6624,7648)`; its supplied legend also does not intersect that lithology region. These facts do not remove the full-page training restriction.

The selected Gulf source pages have reported opposing-fold donor pixels. Existing metadata gives page-level counts but no donor locations within each assembled gate input. The exact fast-gate exposure and score effect are **unknown**. Keep the gate's assisted diagnostic status. Do not describe the measured eight-window scores as independent held-out accuracy.

Evidence metadata: `review-v1/quality/audit.json`, `review-v1/quality/mask-focus.json` and `review-v1/adversary/crossfold.json`, under `viridien_work/pattern_lab/data/genlab/release/`. The durable check is `packs/fast-gate-caveat-check.json`. The check uses saved metadata and frozen window bounds only. It does not decode images, run inference or read test truth.

## Protected well containment

The outside review found that the original labeller inventory's `heldoutExcluded=true` claim was false. The frozen 27-page parent has **12 training-labelled pages and 36 approved boxes from seven protected evaluation wells**. Seven are exact evaluation pages; five are variants of those wells. Existing labels and prior model access remain exposed evidence. They must not be described as newly unseen truth.

The single `split-registry-v1.json` authority derives 11 protected well groups from the metadata of 13 evaluation logs. It blocks training, tuning, label generation and review confirmation on protected wells, page variants and identifiable synthetic descendants. Unknown training identity is denied. The frozen parent is unchanged. The derived human diagnostic now uses **15 pages and 422 boxes**. The legacy four-case human development selection retains one Shenzi QA case and excludes three protected cases. The catalog hides protected truth and suggestions and marks 145 records as protected. There are 351 ready records across mixed units and versions; these are not independent samples.

Protected groups: `api:6080740286`, `api:6081140512`, `api:6081140730`, `api:6081140758`, `api:6081741140`, `api:6081840065`, `document:crittercreek2011807h`, `document:rionueces3744unit1212`, `document:shell08852ke001`, `well:d-582` and `well:mc765_p011`. Do not use these evaluation results to tune a model. Scoring them needs an isolated-truth route and a training-exposure declaration. The current assisted DEV fast gate contains no protected well; its frozen input hashes are unchanged.

The DJS review thumbnails previously displayed scorer truth overlays. The owner records this exposure before approximately 08:25 UK on 8 November 2026. The corrected catalog uses 25 source-only previews. It does not link the DJS truth gallery or scorer HTML report. This correction cannot undo prior exposure. No new inference or API calls were made for this correction.

Current silver labels are **unreviewed two-source labels with source A used on ties**, not AI consensus. The catalog carries saved A/B disagreement counts without opening protected truth. For example, the D-582 log06 sources disagree on 5,242 of 6,020 jointly labelled rows. Source-A Allegheny and Rio labels absent from their source legends block lithology scoring pending human review. The Shell source-B spelling alias is a separate unresolved taxonomy issue. Twelve saved evaluation windows lack populated truth depth columns, so depth accuracy is unavailable. Neptune has zero filled truth rows in its saved evaluation window; row accuracy is unavailable and must be excluded from means.

All outside pack routes now copy the suite, environment and ordered case list from `inputs/outside-manifests/`. The DJS route must retain its dedicated suite or its exact required overrides. The frozen unlabelled route is `unlab_test`, not `unlabelled`. The check is `packs/split-enforcement-check.json`.

The immutable JEV binding has 68 exact case identities: 51 outside DEV cases eligible for the stated prelabelling purpose, 15 protected exclusions and two unresolved current-development cases. All new API calls remain denied pending owner handoff and launcher enforcement. Eligibility alone is not permission to dispatch. Unknown ancestry, hash mismatch, a non-DEV case or a protected donor must deny a call.

## Outside route revision v2

The current authoritative outside inventory has 396 versioned records representing 257 distinct physical pages. The additional 130 records are a geometry revision of the same frozen130 images, not new independent pages. The original release remains pinned. Revised x0 boxes affect four pages. The metres-to-feet treatment on p004/p009 versus assumed feet on p005/p008 remains OPEN. Do not pool depth values, scores or denominators across revisions.

DJS now uses the dedicated `djs_idaho_test` suite and `external_tests_round6_v2`, with `BENCH_DJS_*` only. The original v1 remains unchanged. The owner checked 314/314 identical labels, cluster IDs and row bounds, 311 changed stored depth rows and 25/25 identical images. Its calibration has 20 available fits, including p018's wrong fit of exactly 1,000 ft; 19 agree within 0.5 ft. Five fits are unavailable: p007, p009, p022, p025 and p028. Data readiness and calibration quality are separate. No new model inference or depth verification was done in this root refresh.

All seven owner manifests now carry per-purpose truth-access metadata. DEV keeps its actual non-held-out flags; held-out source previews remain guarded. The latest ext_dev manifest explicitly sets `autoLabelAllowed=false`. This restriction overrides any broad eligibility reading of the unchanged JEV identity binding. All new API calls remain denied. Do not equate model evaluation, model-readable truth, tuning and prelabelling permissions. The current policy-manifest hash and the immutable binding hash are recorded together in READY.json and the new root revision receipt.
