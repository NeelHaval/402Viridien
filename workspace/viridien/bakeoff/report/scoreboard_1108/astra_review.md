# Astra review: 1108 development comparison

Reviewer: one GPT-6 Astra reviewer, xhigh. Date: 8 November 2026.

**Verdict: do not name a champion from the present artifacts.** The core scoring method is suitable for a development comparison. The current report has selection blockers: incomplete results, incorrect DEV well groups, and incomplete controls for frozen inputs and paired cases. These are local runner and report issues. They do not require a change to the shared scorer, s19, or the classifiers.

After the checks below are complete, the parent can name a **current practical choice among these three frozen pipelines on these development inputs**. This does not support a held-out, universal, or new-well champion claim. An unscored hybrid must remain **unverified**. A combination of components from winning suite results is not a measured fourth candidate.

This is one review of specified files, not a Git diff review. The owner supplied the file scope and prohibited shared edits. I used the code-review skill's separate standards and specification checks. I did not ask for a Git base, start a child, edit shared files, or run candidate GPU inference. I did not open external test, verify, external_vault, old outside-test summaries, .env, or secrets. External input checks used only the DEV items and their referenced assets. No thread link was supplied.

## Standards

**S1 — P2, possible duplicated code: the report has a second statistical implementation.**

`score_all.py:95-148` and `build_report.py:34-82` each implement log means, pairing, and bootstrap calculations. They do not have the same case-matching rule. The runner pairs case IDs before averaging. The report averages first and then pairs only log names. They also use different well groups. In a synthetic check, the report accepted two disjoint windows as a paired result and returned a delta of 1.0 with CI [1.0, 1.0]. The runner found zero matched logs. This is a code-smell judgment, with a demonstrated consequence; the statistical defect is M3 below. Local fix: let the report render one validated set of aggregate and paired records. Alternatively, use one small shared report helper with explicit case keys. Do not change the shared harness.

No other hard breach of the supplied coding rules was found in the requested files. Dense one-line code makes audit work harder, but it is not, by itself, a reason to reject this experiment. The review kept all new files under this report directory.

## Spec and method

**M1 — P1 blocker: DEV well groups are incomplete and differ between runner and report.**

Evidence: `score_all.py:57` removes `_logNN` suffixes only. Manifest lines 682-690 and 715-723 give Utah pages 135 and 136 different `well` values. All 51 DEV cases form 46 log IDs and 46 stored well IDs. Removing the page suffix gives five provisional source groups: KGS, Utah, Idaho Barlow, Flat Bay, and Storm. The item counts are 2, 21, 6, 9, and 13 respectively. This grouping is based on the allowed DEV names; it is not a check of independent geological wells.

The pinned report builder has a partial repair: `build_report.py:29-37` groups Utah and NL pages. It leaves the two Idaho Barlow pages separate. The report therefore uses six provisional DEV well groups, while the runner's saved paired summaries use 46. The actual report test at `test_report.py:28-35` checks Utah and NL only. It passes but does not cover Idaho or agreement with the runner.

The bootstrap code itself retains all log deltas from a sampled well (`score_all.py:102-110, 144-146`; `build_report.py:73-80`). The grouping inputs remain incomplete. Thus two well-sensitivity estimates in the same score file can disagree, and Idaho dependence is omitted in both. This violates `evaluation_protocol.json:20`, which requires well clustering where continuation pages repeat. `harness/BENCH.md:35-43` also calls fallback identifiers proxy units.

Local fix: provide a reviewed DEV page-to-well map in the report inputs. The five name groups are a conservative starting point. Record the mapping source and any uncertainty. Recompute both CI types from saved case deltas. Keep equal-log means as declared, and show a separate equal-well sensitivity if the choice depends on page count. Do not silently change the primary weighting. Do not claim precise transfer estimates from five source groups.

**M2 — P1 blocker for a frozen-run claim: source coverage and input identity checks are incomplete.**

Evidence: the original run stopped after one result. `out/scoreboard_full.err` records `RuntimeError: Source drift after freeze: harness\\bench.py, harness\\unlabelled_metrics.py`. These files were in the original broad source list. During this review another worker changed `score_all.py:28-35` to an explicit harness list and replaced the manifest. I read that revision. Excluding unused benchmark modules is reasonable; their drift alone does not show that the one prediction changed. The original failure and hashes remain in the review evidence.

The revised source list still omits an executed dependency: `strategies/s14_learned_match.py:10,45-48` calls `strategies/s02_features_pelt.py:36-62`. It also omits the report builder. The snapshot imports live labeltool modules, though the inspected inference path does not call their human-label helpers. DINO records weight and config hashes on first model load, but those hashes are not predeclared in the run manifest (`s21_v4dino.py:45-66,193-198`).

`score_all.py:59` records image, annotation, copied truth, and source-truth hashes. `check_sources()` checks code/artifact paths only. Execution opens the live annotation/image at line 204 and loads truth at lines 221 and 236 without comparison to those hashes. Resume accepts old records by case ID and strategy at lines 190-198; it does not validate each record against the current manifest. `audit_results.py:79-82` checks source hashes, but does not check input hashes. A changed truth file could be re-scored consistently while its stored provenance hash stays stale.

Independent evidence reduces, but does not remove, this risk: **all 264 distinct path/hash checks matched** in the reviewed inputs. This is a present-state check, not proof of every byte used during an earlier execution.

Local fix: add s02 and all executed local dependencies to a precise dependency list. Pin the report builder separately. Record model hashes and package versions. Validate declared asset and truth hashes before using resumed records and again before the final audit. Use a manifest digest in every checkpoint and require exact record/manifest agreement. Preserve the old manifest and record the reason for a revised run; do not rewrite the history as one unchanged freeze. Snapshot or hash the preprocessing cache inputs that can affect execution. These changes belong in the report runner and audit, not in s19 or the shared scorer.

**M3 — P1 blocker: partial or mismatched case sets can produce a report that looks paired.**

Evidence: `build_report.py:191-206` accepts any nonempty record list. It does not require `complete`, the three named strategies, unique keys, or the expected 204 results. `build_report.py:61-71` pairs log means without checking windows, truth hashes, image hashes, or well agreement. `score_all.py:248` sets `complete` from record count only. A duplicate or stale record can therefore satisfy the count without satisfying the required case set.

Concrete test: A scores 1.0 on rows 0-100 of a page. B scores 0.0 on rows 100-200 of the same page. No window is shared. The report returns a paired delta of 1.0 and a degenerate CI [1.0, 1.0]. The runner finds no matched case. See `astra_review_checks.json`, key `disjoint_window_pair_accepted`.

The parent audit already checks duplicate keys and exact expected coverage (`audit_results.py:29-37`). Use that result as a mandatory decision gate. Extend it to require matching window, source, truth, and well metadata. Match cases before log averaging, as the runner does. Display an explicit incomplete status and suppress selection text until those checks succeed. The frozen harness comparison provides a useful model: `harness/bench.py:608-637` rejects duplicate keys and inconsistent fingerprints.

**M4 — P2: the recorded depth calibration can differ from the calibration used by s19.**

`evaluation_protocol.json:8` says depth calibration is shared. The runner supplies the same initial calibration, but `ml/combine/runtime.py:142-169,193-199` can replace a weak calibration and write its own depths. `harness/run.py:68-70` preserves those depths. `score_all.py:224` then records the original shared calibration as `depth_calibration`.

This does not change the pixel score, because score is row accuracy times boundary F1. It does affect the meaning of depth MAE and the run record. Keep s19 unchanged. State that all methods receive a common initial calibration and that s19 can refine it. Rename the record field to `shared_initial_depth_calibration`; mark s19 output depths as potentially refined. Do not claim identical output calibration. Depth error is conditional on matched boundaries and available GT depths, not a full-track error estimate.

**M5 — P2: runtime is valid as observed execution time, but not as a controlled cold/warm comparison.**

Serial calls and CUDA synchronization are present (`score_all.py:195-216`). First model load is inside `run()`. Common preprocessing is reported separately. The 60-second flag is present at line 243. These meet the main timing design.

Limits remain. DINO's model is process-cached; v4's native cache is also process-cached. A depth-failure diagnostic can load and warm a model before a later official inference. Resume starts a new process without a cold/warm marker. Failure records can have no `seconds`, and `over_budget_60s` then becomes false. Shared preprocessing time is null on those failures, although `task_wall_seconds` records the attempted work. GPU reserved memory and process RSS can include state retained from earlier candidates. See `score_all.py:228-243`, `s21_v4dino.py:45`, and `scoreboard_v4.py:25`.

Local fix: keep timing descriptive, state process/case order and cache state, and mark missing timing as unknown. Show cold-load events, diagnostic time, attempted task time, and the number of timed cases. Do not treat a failed case with missing inference time as a fast success. For a speed-based tie-break, use a separate serial timing check with explicit cold and warm runs after scoring is complete. No such timing run was made by this reviewer.

## Checks of the candidate method

- **Strict labels:** all official scores call the unchanged shared scorer with `min_layer_px=0`. Normalization removes whitespace and case and uses the single existing Sand-and-Sandstone alias (`harness/score.py:7-12`). The review probe gave zero row accuracy for `Clay / Shale` against `Shale`, and one for `Sand and Sandstone` against `Sandstone`. No extra membership or synonym credit was found. Vocabulary coverage is diagnostic only; `score_all.py:150-166` does not feed labels back into prediction.
- **Blanks and contacts:** the scorer ignores GT blank rows for accuracy and ARI, ignores predicted contacts wholly inside GT blanks, and retains contacts facing a filled GT row (`harness/score.py:90-111`). It uses strict interior contacts, the specified tolerance, and one-to-one matching. A score therefore does not measure false filling of every blank region. The adapter omits blank beds, preserves unresolved nonblank beds as Unknown, and uses exclusive interval ends (`scoreboard_v4.py:80-96`).
- **Inherited v4 limits:** frozen beds absorb thin pieces and merge adjacent equal states/labels without preserving a drawn same-label rule (`scoreboard_v4_snapshot/beds.py:173-188`). Single-chunk blank runs can also be absorbed by the existing run grouping at lines 137-163. The adapter maps cluster IDs by label, including one Unknown ID, rather than by distinct unknown textures. These limit strict compliance with `SPEC.md:28-32`. They are retained candidate behavior, not a new scorer defect. Score these failures as they are; do not repair them during this comparison or call the adapter fully SPEC-compliant.
- **Common inputs:** all candidates receive the same scored window and initial context. s19 parses its own v3 legend and may refine depth. v4 and DINO use the frozen native legend. Thus this is a comparison of complete pipelines, not an isolated classifier comparison. `scoreboard_v4.py:53-77` limits column reads to the supplied window. Depth and legend preprocessing can read their shared tracks outside that column window, as the harness does.
- **Full v4 adaptation:** the frozen `beds.py`, `legend_classify.py`, native swatch pickle, and scale JSON were byte-identical to their stated current originals. The adapter replaces IO and the classifier namespace. `P['scales']` is empty and the adapter refuses a nonempty setting. Frame statistics use the scored window, not the whole original column; the snapshot README states this. Use the name “frozen v4 adapted to the common window,” not a claim of identical whole-page output.
- **DEV cache failure:** the frozen native cache has 51 logs and contains **none of the 51 DEV case names**. Both v4 candidates therefore lack native legend entries on DEV. DINO skips embedding when there are no swatches (`s21_v4dino.py:164-188`). These results can show the practical inability of the shipped cache path to handle DEV. They cannot show that DINO's visual representation is inferior on those pages. Do not repair the cache and reuse the same candidate name or scores.
- **DINO design:** native pixels tile or crop to 224 by 224; oversized input is cropped at the top left. The embedding joins CLS and mean patch tokens, then normalizes the vector. Swatches use nearest-neighbour scales 0.5, 1, and 2. Similarity takes the best swatch/scale within each twin group. Fusion minimizes `z(hand distance) - z(DINO similarity)` (`s21_v4dino.py:27-34,69-104,123-161`). The hand rejection gate remains separate. Confidence uses `1-exp(-gap)` and affects row-weighted bed votes at `beds.py:164-171`. Thus the change affects confidence and final bed labels/boundaries as well as the nearest label. The two segmentation passes use the same pre-label chunks; the count check guards their reuse. I found no sign reversal or label-order error in this path. The smoke test does not by itself prove numerical fusion correctness for all edge cases.
- **Depth failure:** official failed results receive zero score, accuracy, F1, and ARI, with null depth MAE (`score_all.py:225-237`). Pixel-only diagnostics use `calib=None`, remove any depths, and stay in a separate field and CSV. The pinned report creates a separate diagnostic table at `build_report.py:207-214,241-242`; it retains primary results where no diagnostic exists and labels the substitution. Primary suite and overall tables remain unchanged. This is the correct separation for the declared full pipeline. Do not replace official failures with diagnostic successes for selection, or compare only the surviving windows to select a winner.
- **Aggregation:** the runner averages each window's product within a log, then averages logs. Repeated DEV windows are not independent log votes. Both bootstrap implementations use 10,000 deterministic draws, seed 1108. Their pairing inputs and well groups need M1/M3 fixes. The two implementations use opposite A/B ordering; each declares its direction, so signs must be compared by names, not table position.
- **Overlap and QA:** log06 appears in eval and gold with different extents: [4404,11195] and [4409,10415]. The 1inch QA page also overlaps eval with a different extent. The overall rule chooses QA, then gold, then eval, then DEV per source log; it does not merely remove exact duplicate windows. This is stated in the report and is a defensible descriptive rule. Hadrian is excluded from overall and retains first-composition-only truth. Shenzi machine contact anchors, unreviewed Hadrian composition, and silver consensus remain truth-quality limits. I checked hashes and conversion code; I did not certify human truth quality.

`harness/BENCH.md:16-19` normally forbids selecting methods from the 13 eval truths. The owner's present request and declared protocol explicitly permit a development practical choice. This is a narrower, explicit exception to that benchmark regime, not held-out validation. Keep eval results as regression evidence. Do not tune labels, thresholds, or routing on these results and then call the same comparison independent.

## Test and result evidence

Reviewer command, from bakeoff:

```text
C:/Users/hssil/anaconda3/python.exe -X utf8 -B report/scoreboard_1108/astra_review_checks.py
```

The script runs the actual three runner tests, four report tests, six adapter tests, and seven shared scorer tests. It redirects the legacy CSV test's temporary writes into the review directory. It also checks manifest inputs and runs the concrete disjoint-window and PNG probes. The final log is `astra_review_checks.log`; the first run is preserved as `astra_review_checks_initial.log`. The script and evidence are new review files only.

The first test run ended with `20 passed, 2 warnings in 5.31s` and `ASTRA CHECK EXIT 0`. A second run was required because another worker revised the runner's source list and manifest during review. The final run ended with **`20 passed, 2 warnings in 6.94s`** and **`ASTRA CHECK EXIT 0`**. Its 264 input hash checks matched, and the revised manifest had no source-hash mismatches. A final hash check found no later changes to the pinned reviewed files. This resolves the initial unused-file drift at the review cutoff; it does not resolve the missing dependency or input/resume guards in M2. No candidate GPU test was repeated while scoring was active.

The inspected existing GPU smoke log reports PASS, 5 chunks, 30 swatch vectors, and 30.45 seconds cold, on the RTX 4080 Laptop GPU. It records config SHA `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` and model SHA `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`. This is supplied smoke evidence, not a reviewer rerun or a suite accuracy result.

Full harness: **not run by this reviewer**. The existing final log reports `1 failed, 89 passed in 37.12s`; another current log reports `1 failed, 89 passed in 28.60s`. The initial supplied count was 88 passed. The failing test is `harness/test_bench.py::test_notebook_callable_with_spawn_and_cached_second_run`. It makes a PNG at line 211 and reaches `harness/io.py:46`; the installed tifffile raises `TiffFileError`, which the catch at line 52 does not handle. My in-memory-generated PNG probe reproduced this error. All 68 manifest images are TIFF (67 .tif and one .tiff), so this does not by itself invalidate this TIFF-only comparison. Do not report the full harness as passing. A later loader task can restrict the TIFF backend or catch its format error; no shared fix was made here.

One result snapshot was taken, with no wait or score polling. `scores.json`, `report_data.json`, and `audit_results.json` were absent. `raw_results.json` held one s19 eval record. The inspected run log confirms a stop at the source-drift guard. That record cannot support a ranking. A later restart may finish after this review; the parent must audit its saved CSVs and coverage independently. The result snapshot is saved in `astra_results_observed.json`.

## Required decision gate

1. Keep a versioned manifest and exact candidate/dependency hashes. Resolve M2 before certifying a frozen run.
2. Obtain exactly 68 cases times three candidates, with unique matching case keys and failure zeros. Require the independent CSV audit to succeed. Do not infer completion from a row count alone.
3. Correct DEV well groups and pair identical cases before log aggregation. Rebuild the report from the audited records.
4. Show per-suite scores, component metrics, failure counts, observed timing coverage, and both uncertainty views. State native-cache failures and truth overlap. A pool dominated by many pages from a few wells is not broad evidence.
5. Name a practical development choice only if those results support it. If suites disagree, give the trade-off or retain more than one candidate. Keep every proposed unscored hybrid unverified.

## Frozen review identity

Paths below are relative to bakeoff. `astra_review_hashes.json` contains full SHA-256 values for all reviewed files. `astra_review_sources.json` preserves the four reviewed runner/report/protocol/manifest texts. `astra_review_hashes_start.json` preserves the earlier hashes. No claim in this review covers later uninspected edits.

```text
report/scoreboard_1108/score_all.py
fe1c7d571df7946b5006706d272a92c946b76a684c3f880ff53b45c9895b8ee5
report/scoreboard_1108/build_report.py
4e1e4d096a52f484f0f6d206a1028f59225d2ece681a3bc95e2dd113a20936b2
report/scoreboard_1108/evaluation_protocol.json
3772fe30c89615d5af5aded2e1f968ff79eb384ffe482a2dc1565fd953827463
report/scoreboard_1108/manifest_1108.json
bc96b4101a09f228f7a4a1484c48db78979f6f5fe6854c86217002f2b408ab40
strategies/s20_v4.py
4023a28cdd673a457fec0332c39a6932f36e3aab06d7e4e6ba64bcb3157286ff
strategies/scoreboard_v4.py
058a9e73960d3f3ec87105925fd20313dbe0f0c8388aea46af8624ce3823400a
strategies/s21_v4dino.py
2d2395cf672c519d219e7a6d863b68a8c0dbb7b85bbb31f9dd3f5c7c8eb81a22
strategies/scoreboard_v4_snapshot/beds.py
c5540a6b7404c6651c8ae8cb2ed356fd6bb34df3ac095ae26f1ceac76d779029
strategies/scoreboard_v4_snapshot/legend_classify.py
44451fea6bf696e002901fd216e073ed80752321a957bbec4b3a3bcce1657419
strategies/scoreboard_v4_snapshot/native_swatches.pkl
01427db9311d4b48bcd7452bb1d8e91a69936f0758e0889dfc232657ee3696b5
strategies/scoreboard_v4_snapshot/log_scale.json
75985088446496b926bd7528e86b9c272b1c243482546ce04f5704512180ea24
```

Standards: one P2 judgment finding. Spec/method: five findings, with P1 blockers in well grouping, frozen-run controls, and paired/completion validation. No champion is approved by this review.
