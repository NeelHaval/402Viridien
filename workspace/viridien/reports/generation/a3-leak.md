**1. Verdict**

Useful for exploration; not yet fit for clean training and independent evaluation.
Test document IDs are excluded, but quarantined test-identical pixels enter training augmentation.
The biggest problem is inconsistent enforcement of split, quality and source-version rules across consumers.

**2. Findings, ranked by impact**

Audit: 8 November 2026, normalized snapshot `8b8884c431cb8ce8aac5`, 1,028 records. The 51 documents comprise 25 training, seven validation and 19 test documents under the dataset split. Checks used saved metadata and bounded PNGs; peak working set was 124.6 MiB. Reproduction scripts are in `data/genlab/critique/a3-leak/`; run with `PYTHONDONTWRITEBYTECODE=1 python -X utf8`. Source lines describe the code observed during this review; other agents are editing it concurrently.

**Finding 1: The dataset builder bypasses an existing test-duplicate quarantine.**

Severity: **blocker**.

Evidence: `probes.py` found nine quality-rejected records admitted by the live `DatasetBuilder.human_rows()`: five training and four validation. All saved versions v1–v5 contain these records: eight `duplicate_of_test`, one `legend_fill_mismatch`. V5 fitted five augmentation measurements from the rejected training records; three augmented strips actually reference duplicate-of-test measurements. The bypass is at `genlab/dataset.py:307`, which checks review status/flags but never joins the normalized eligibility record; fitting starts at `genlab/dataset.py:359`. The intended quarantine is explicit at `normalized.py:108`.

I rendered and inspected `data/genlab/critique/a3-leak/native-quality.png`. Training shale from `28112eef4c3c645a`, box `[661,186228,731,186343]`, exactly equals test shale from `eb4e39e664edd11d`, box `[661,25828,731,25943]`: 70×115 pixels, 20 colours, zero white pixels. This is a patterned duplicate, not a blank-fill coincidence. The second row shows the rejected yellow dotted crop versus its assigned green striped generated pattern, at native scale.

Fix: make `human_rows()` consume current, accepted, quality-clean normalized records, verify annotation versions, then apply the dataset's train/validation subdivision. Preserve sample ID, annotation version, quality and pixel hash in real/measurement provenance. Apply the same audit to controlled baselines rather than silently inheriting their rejected rows. Verify zero quarantined rows or their measurement descendants in rebuilt manifests. Report shared-symbol performance separately from unseen-motif generalization; identical symbols alone do not establish copied pages.

Effort: **M**. Expected gain: removes a demonstrated test-overlap path and five invalid augmentation inputs; cleans four validation truth records.

**Finding 2: Assembler validation wells are already generation inputs.**

Severity: **blocker for independent validation**.

Evidence: all seven dataset-validation documents occur in assembler statistics, contributing 6,999 human rows and 355,627 machine-labelled rows. They also supply 131/354 noise measurements (37.0%), seven rail libraries and 584 boundary observations. `genlab/assemble/stats.py:38` and `genlab/assemble/noise.py:41` exclude only the original test split, whereas `genlab/dataset.py:51` additionally reserves validation wells. Family/global pooling and appearance fallback therefore carry validation information into generation. The benchmark acknowledges this limitation at `genlab/assemble/bench.py:404`, but its validation selection only excludes document-statistic wells (`:249`, `:261`).

I inspected `data/genlab/critique/a3-leak/native-validation.png`: the validation limestone crop `b9ba393ab59601f2:19`, its generated repeat and saved residual mask share the same native scale. Measurement `6fd22088fd27f6074518` records the validation crop as its source.

Fix: use one shared train/validation/test policy before learning tiles, estimating statistics, collecting rails/gaps/boundaries or selecting substitutes. Fit family/global fallback exclusively on training wells. Verify the transitive ancestry of every generated sample against validation/test groups. Keep any deliberately legend-conditioned evaluation separately identified.

Effort: **L**. Expected gain: makes validation measure transfer to withheld wells rather than reuse of their appearance and distributions.

**Finding 3: Test results enter the automatic regression feedback loop.**

Severity: **major**.

Evidence: a read-only call to `regression_values()` extracted 93 benchmark values, including **41 `/test/` values**. Its recursive traversal includes every nested result (`genlab/assemble/bench.py:878`); `report()` uses them for regression reporting (`:898`). Thus test fidelity/detector changes can guide iterative model selection even though test pixels are barred from fitting. Both `bench/runs/tiny/fidelity.json` and `bench/runs/starter/fidelity.json` already evaluate ten test wells. Five of those ten well groups are listed as previously seen in `data/normalized/split-policy.json`; the current test annotations comprise 52 retrospective and 60 prospective records. Additionally, 19 test candidate reports expose 272 entries for inspection; they are correctly excluded from generation, but remain accessible during development.

Fix: exclude test results from routine regression dashboards, ranking and remediation feedback. Run final test reporting against a frozen model/dataset identifier, and retain an exposure ledger. Report retrospective and prospective cohorts separately, preserving the existing policy caveat. Verify that changing only test metrics leaves every development-selection decision unchanged.

Effort: **M**. Expected gain: prevents test-driven tuning and stops previously seen wells being pooled into an apparent untouched estimate.

**Finding 4: Look-alike suite construction imports test-derived pair selection.**

Severity: **major**.

Evidence: `/api/mix` returns 285 pairs; 64 originate exclusively from test logs. A read-only replay of the suite's matching loop produced 537 training-universe pair entries, including 121 from those test-only pairs. Removing them removes **90 distinct document/pair combinations**. For example, test-derived `Silstone`/`Fine sandstone` is selected for training document `17a317978b2c9139`. `build_suite()` consumes every endpoint pair without filtering its originating logs (`genlab/assemble/bench.py:1097`). The endpoint derives pairs from swatch features and ambiguous row shares, then orders them by aggregate rows at stake (`../bakeoff/labeltool/mix_pairs.py:169`, `:188`). This is data-derived selection, not merely taxonomy metadata.

Fix: compute pair membership, scores and ordering from training documents only, before aggregation; store the originating wells with each selected pair. Verify that deleting all held-out endpoint records leaves training-suite choices and ordering unchanged. Retain the statistics module's narrower metadata-only import at `genlab/assemble/stats.py:471`.

Effort: **M**. Expected gain: removes a measured indirect route from test appearances/row shares into synthetic benchmark design.

**Finding 5: Several input changes never invalidate live results.**

Severity: **major**.

Evidence: `watch_probe.py` changes file metadata in memory, without touching live files. The lab fingerprint ignores split-policy-only, prefill-only and proposal-PNG-only changes; the dataset watcher ignores prefill-only and proposal-PNG-only changes. The statistics watcher ignores candidate-report and cached family/category metadata changes. All three detect `current.json` changes, which is a useful positive control. Relevant code: `genlab/server.py:39`, `genlab/dataset_loop.py:45`, `genlab/assemble/stats.py:885`. Furthermore, the long-lived builder loads prefill only in its constructor (`genlab/build.py:75`), and candidate signatures omit prefill (`genlab/candidates.py:22`). An unrelated rebuild therefore does not reliably repair that stale dependency.

Fix: declare and hash each consumer's actual dependencies, including split policy, proposal pixels, prefill, candidate exclusions and canonical family metadata. Reload cached prefill when its digest changes; incorporate that digest into candidate signatures. Ignore volatile metadata such as fetch timestamps. Immediately invalidate split/quality revocations rather than waiting for a batch-size threshold. Verify each dependency independently using the existing probe pattern.

Effort: **M**. Expected gain: corrections and remakes reliably reach outputs, and a newly held-out well cannot remain published merely because no watched file changed.

**Finding 6: Noise libraries and suite caches do not bind all source versions.**

Severity: **major**.

Evidence: saved noise measurements identify class, document and box, but lack normalized snapshot, annotation version and model signature. Their IDs hash only class key plus box (`genlab/assemble/noise.py:388`); the library header stores a build time rather than dependency hashes (`:418`). Composer checks well eligibility when loading these libraries, but not whether the source annotation/model is still current (`genlab/assemble/compose.py:341`). Suite versions hash the plan and statistics, then can return cached output before loading current cards, noise or `/api/mix` (`build_suite`, approximately `genlab/assemble/bench.py:1062`). All six saved suite manifests have statistics/truth hashes but no noise/model dependency digest.

Failure scenario: an owner rejects a source box or a maker replaces its tile; the old residual can still augment another pattern in that family. Refitting may then overwrite the same residual filename, so an old measurement ID no longer identifies fixed bytes. Updating noise alone also leaves the suite version unchanged.

Fix: make noise artifacts immutable and content-addressed; record label snapshot, annotation versions, tile hashes, policy hash and fitter version. Reject stale sources or refit automatically. Include model/noise/pair/code digests in suite cache keys and preserve dependencies with each run. Verify that changing a source annotation or noise field invalidates the dependent artifact while an archived run remains reproducible.

Effort: **M**. Expected gain: owner-label growth/revocation cannot silently leave old fitted appearance active; saved seeds acquire reproducible meaning.

**Finding 7: The card's “held-out” score has already influenced generator selection.**

Severity: **major**.

Evidence: eight current cards report 139 held-out crop scores with `reused_training=false`; all eight also have selection scores over the entire example set. For `1bac8dda58a39eaf:1`, 24 claimed held-out scores follow generator selection over all 39 crops. `genlab/selection.py:22` scores every example, and `:35` uses those scores to select the generator. `genlab/score.py:184` subsequently treats any crop absent from tile voters/suspects as independent. Moreover, background-majority/suspect selection already examines up to 400 examples (`genlab/learn.py:198`). Exclusion from final pixel voting is insufficient.

Fix: reserve non-overlapping evaluation intervals before cleaning, majority estimation, period fitting and generator selection. Use separate development crops for selection; score the frozen result once on reserved intervals. Until then, label these values post-selection reconstruction scores. Verify that changing reserved pixels cannot alter the chosen tile, source or fitted parameters.

Effort: **M**. Expected gain: removes optimism from all 139 claimed holdout scores without losing useful reconstruction diagnostics.

**Finding 8: Publication gates compare different owner-label populations.**

Severity: **major**.

Evidence: validation truth grows from 79 boxes in v1 to 92 in v2 and 94 in v3. V2 publication records both “discriminator rose more than 5 points” (+8.866) and “Human validation labels changed”; v3 records a macro drop of 2.627 points alongside the same population-change warning. Both are blocked as regressions. `genlab/dataset_loop.py:147` computes deltas and regression flags before checking truth comparability at `:154`; the warning does not disable the decision at `:167`. These numbers cannot identify a model regression independently of newly arriving labels.

Fix: when truth fingerprints differ, re-evaluate both model versions on one frozen cohort, or compare the common unchanged annotations and separately report new-label performance. Make publication decisions only from that paired comparison. Verify with unchanged predictions plus newly added difficult labels: the system must report a population change, not a model regression.

Effort: **M**. Expected gain: prevents label growth from wrongly promoting or rejecting datasets. The deliberately frozen v1-based v4/v5 comparisons are useful and should remain explicit.

**3. Quick wins — each under one hour**

- Enforce normalized quality/version eligibility in `DatasetBuilder.human_rows()`; audit old versions before reuse.
- Remove `/test/` values from automatic regression inputs and routine development feedback.
- Watch split-policy and proposal PNG changes; reload prefill when its fingerprint changes.
- Rename current card “held-out” scores to “post-selection reconstruction” pending a real reservation scheme.
- Disable regression-based publication decisions when truth fingerprints differ until a paired evaluation exists.

**4. What is already good**

The 389-card/1,556-strip live manifest contains zero test-document or test-donor IDs; donor revocation and proposal-round checks deserve preservation.
Assembler statistics exclude all 19 test documents, enforce normalized quality, and match the current label snapshot; machine guesses remain separately identified.
V4/v5 contain zero validation augmentation measurements, unlike v1–v3; their legend-only validation reconstruction is a material improvement.

**5. Doubts**

This was a concurrent, read-only audit, not a frozen release certification. I did not run expensive generation, refit models, rerun recognizers or quantify score inflation after fixes. The eight duplicate crops prove pixel overlap; they do not prove that whole pages were copied. I did not exhaustively search phase-shifted/rescaled duplicate motifs or reconstruct every external detector/agent's historical inputs. Prefill contains no gold-selected training rows in this snapshot, but its upstream model lineage is not fully certified here. Compose and bench exist and have saved outputs; their ongoing implementation changes are not treated as missing functionality. Current statistics were fresh, and controlled-baseline datasets were intentionally frozen; the staleness findings distinguish those facts from demonstrated invalidation gaps.
