# 1. Verdict

Useful diagnostics, but not yet fit to certify that training or testing on synthetic logs predicts performance on real logs.
The new within-document result is credible for its narrow task: I reproduced 57/79 clean and 58/79 augmented matcher successes; it is not evidence of downstream detector utility.
The biggest problem is the missing independent transfer experiment: validation swatches supply the recognition models, while assembler validation wells still contribute to generation statistics.

# 2. Findings, ranked by impact

Evidence snapshot: 2026-11-08, approximately 03:19 UTC; other agents were actively updating files. I ran `python -B -X utf8 data/genlab/critique/a1-eval/audit.py` and `python -B -X utf8 data/genlab/critique/a1-eval/discriminator_audit.py`. Both completed successfully. The main audit reported 204 MiB resident memory. It used small saved PNGs, never a TIFF page. Its v4 manifest SHA256 starts `8c639ee389a3592709`; evaluation SHA256 starts `bd9fbe0adb0bb0d6a`.

I rendered and inspected [same-scale-v4.png](a1-eval/same-scale-v4.png), with every panel at exactly 2× native pixels, and inspected `data/genlab/assemble/bench/runs/starter/same-scale.png` at native scale. The former shows plausible green shale and orange siltstone, but also recurring doubled strokes in claystone and pale horizontal seams in colored shale. These four cases illustrate why appearance inspection and task scores must accompany each other; they are not a prevalence estimate.

## F1. The primary experiments do not establish synthetic training utility or page detection

**Severity: blocker.**

**Evidence.** Reproduced v4 within-document scores, using 552 synthetic support strips per arm and 79 real boxes:

| Model | Clean correct / top-1 | Augmented correct / top-1 | Uniform candidate chance |
|---|---:|---:|---:|
| Matcher | 57/79 / 72.152% | 58/79 / 73.418% | 5.543% |
| Logistic | 44/79 / 55.696% | 52/79 / 65.823% | 5.543% |

`genlab/dataset.py:708`, `:755`, and `:763` construct support examples from each validation document's legend and fit a classifier separately for that document. This correctly measures legend-conditioned matching. It does not train that classifier on the fleet's 262 training-document classes. A fleet tile could deteriorate without changing this clean within-document score. The inspected sheet contains these fresh swatch candidates, not a representative sample of agent-made tiles.

The older cross-log matcher remains 7/79, with 80 training names and four unsupported-name boxes. Its uniform chance is **1.187%, not approximately 7/79**. It forces a known-name prediction; it does not evaluate unknown rejection. Conversely, an oracle that selects each document's majority truth class scores 41/79 = 51.9%. That is an imbalance diagnostic, not a deployable baseline, because it uses held-out labels.

The page branch reads the ground-truth column box from `bbox_annotation.json` before inference (`genlab/assemble/bench.py:801`), then feeds only that crop to the detector. A detector that always fails to locate the column would receive the same conditional lithology score. At inspection, starter fidelity and realism results existed, but `bench/runs/starter/detector.json` did not; utility execution was unfinished, not a demonstrated failure.

**Fix / exact protocol.** Retain the current experiment under “legend-conditioned crop recognition.” Add a separate transfer experiment in the dataset evaluator and bench: on identical training-well folds, train **real-only, synthetic-only, and real+synthetic** versions of the intended trainable detector/classifier. Use five fixed training seeds, the same architecture, optimizer steps, and real-example exposure where applicable; document synthetic exposure. Tune on inner development wells only. For every outer held well, allow its legend as inference support in all arms, with zero column-derived fitting. Evaluate the same real boxes and independently annotated continuous windows in every arm. Report per-well macro accuracy, all observed document-class recalls, and the paired real+synthetic minus real-only difference.

For page evaluation, pass image bands and permitted legend information to the locator, not the true column box. Report column IoU/AP at IoU 0.50:0.05:0.95, missed-column rate, and downstream row/segment scores; retain an explicitly named oracle-box diagnostic. To validate synthetic tests, run at least three frozen detector configurations on both real and synthetic evaluation sets, compare per-class errors and detector rankings, and measure paired performance gaps. Similar aggregate accuracy for one detector is insufficient.

**Effort: L. Expected gain:** answers whether fakes improve the intended model and whether synthetic tests predict real failures.

## F2. Assembler validation is not independent of generation; TEST enters repeated feedback

**Severity: blocker.**

**Evidence.** All **5/5** documents in the saved starter validation reference also occur in `stats.json`'s global `document_ids`: `3515004b6f643c87`, `7dc28eda27f5c429`, `952caeadf1d8467c`, `aeb414a4367bafd4`, and `b9ba393ab59601f2`. `genlab/assemble/stats.py:489` admits all fixed training wells; `:674`–`:680` pools them into family/global statistics. `bench.py:261` excludes only wells named in document-level statistics. Its caveat at `:404` honestly acknowledges this, but the comparison cannot establish independent generation fidelity. Holding a well out of discriminator fitting also does not remove its pixels from a previously learned generator (`bench.py:629`).

The direct split audit was reassuring: **zero test-document rows** in v4 manifest/real/evaluation candidates and **zero test documents** in the current global statistics pool. However, calling `regression_values` on the saved fidelity result returned **38 TEST-derived values among 72 history values**. The recursion at `bench.py:869` and regression display at `:905` include TEST scores in repeated development feedback. This is a concrete selection-leakage route even though the file write guard prevents direct writes into STATS/COMPOSE. I did not observe anyone using that feedback to change a model.

**Fix / exact protocol.** Freeze the existing test well groups. On the remaining wells, use five outer group folds, balancing families where feasible; tune within each outer training partition. Before fitting anything, pass an explicit allowed-well set through statistics, tile learning, borrowing, noise/residual pools, and page-context selection. Rebuild every family/global pool for that fold. Save each artifact's complete ancestor-well list, including donors, and assert its disjointness from the held fold. Separately tag permitted held-well legend pixels as inference support. Verify with a sentinel change to held-out column data: training artifacts and predictions on fixed inputs must remain unchanged.

Exclude TEST branches from regression history, promotion, and development summaries; expose final test evaluation only after freezing code, artifacts, thresholds, and seeds. Preserve the split policy's distinction between previously seen and untouched groups: earlier exposure cannot be undone by relabeling a split.

**Effort: L. Expected gain:** removes optimistic validation and preserves what remains of the final test set.

## F3. The clean/augmented discriminator comparison uses identical held-out images

**Severity: major.**

**Evidence.** I exactly reproduced **111/158 = 70.253%** clean and **88/158 = 55.696%** augmented discriminator accuracy. All **79/79 held real IDs and 79/79 held fake pixel hashes are identical across these arms**. Across the full validation manifest, **552/552 clean/augmented image pairs are identical**; I independently decoded and verified all 1,104 PNG pixel hashes. The builder skips validation measurements (`genlab/dataset.py:370`); rendering then uses an empty measurement pool (`:410`–`:419`), which produces identity augmentation (`genlab/realism.py:58`). The separately augmented recognition candidates are a different inventory.

Thus the discriminator drop measures changed discriminator training, not improved held-out fake images. All 79 held fakes are swatches; human, agent-made, and borrowed generation are not independently tested by that number. Pairing also substitutes another document in **18/79** validation pairs because `genlab/dataset_eval.py:115`–`:120` falls back from document identity to a shared name. A separate width/height-only logistic probe scored **95/158 = 60.127%**, establishing a sampling-geometry difference, although dimensions are not direct features of the original discriminator. Its near-chance result cannot certify realism; even the naive independent-image Wilson interval for 88/158 spans **47.9–63.2%**.

**Fix / exact protocol.** Generate genuinely distinct held clean/augmented pairs from identical base content and train-only nuisance distributions, and assert the expected transformations and hashes. Match real/fake pairs by document, class, native width, and window height; do not silently substitute another log. Sample real windows and fake windows through the same cropping/padding procedure. For source-specific claims, hold out training wells from generator fitting and generate that source's held examples; mark unsupported sources unavailable.

Freeze one linear and one nonlinear discriminator specification using development wells. Report AUROC and balanced accuracy, per-well results, 95% well-bootstrap intervals, real-vs-real negative controls, and an obvious corruption positive control. Keep this as a diagnostic alongside transfer utility; failure to distinguish is not proof of equivalence. Use the same paired inventory when comparing generator versions.

**Effort: M. Expected gain:** prevents an unchanged image set from being reported as an augmentation realism improvement.

## F4. The sample cannot resolve the promotion thresholds, and promotion ignores the relevant task

**Severity: major.**

**Evidence.** There are **79 boxes, 7 wells, and 25 observed document classes**; **12 classes have fewer than three boxes**, totaling 17 boxes. For 57/79, the naive box-level Wilson interval is **61.4–80.8%**. My exploratory percentile bootstrap, resampling seven whole wells with replacement 20,000 times, seed 20261108, gives **58.8–87.0%**. With only seven clusters, that bootstrap is itself fragile.

The logistic clean-to-augmented gain is 8/79 = **10.127 points**, consisting of 14 improvements and six regressions. The paired well-bootstrap interval is **−3.9 to +26.4 points**; exact enumeration of the 128 well-level sign flips gives two-sided **p=0.3125**. The matcher's gain is one box in one well. These data do not establish a reliable small improvement. Even an independent-box planning approximation at 72% accuracy needs about **310 observations for ±5 points** and **1,937 for ±2 points**, before correlation or repeated model selection.

`genlab/dataset_loop.py:79` monitors cross-log matcher macro and discriminator accuracy, not closed-set recognition or transfer. I replaced every closed-set accuracy/macro with zero in memory; its metrics remained exactly `{'macro': 10.863, 'discriminator': 55.696}`. `dataset_eval.py:243` labels evaluation “passed” merely when two scores exist. Promotion at `dataset_loop.py:149` uses fixed −2/+5 point cutoffs; it compares the immediately previous version (`:144`), even if that version was rejected. For example, macro 80 → 77 is rejected, but 77 → 75.5 passes that drop threshold despite being 4.5 points below the earlier 80.

**Fix / exact protocol.** Separate “evaluation completed” from “utility qualified.” Freeze truth IDs, candidate vocabulary, preprocessing, algorithm, and artifact hashes; changed cohorts require recomputing both versions on their intersection and reporting coverage changes. Compare with the promoted baseline. Publish all observed class recalls with low-n flags, a well-macro primary metric, box-weighted accuracy, uniform-candidate chance, and a training-prior baseline. Keep the oracle majority diagnostic clearly labeled.

Use paired well resampling for version differences and show every well's result. Predeclare a practical tolerance, such as two points of well-macro degradation, but qualify a version only when the appropriate paired interval satisfies it; with present data, return “inconclusive.” Acquire more independently sampled wells and continuous depth windows before treating small score changes as release evidence. Repeated seeds measure training variability, not additional real sample size.

**Effort: M. Expected gain:** stops noise, irrelevant metrics, or a sequence of rejected baselines from controlling promotion.

## F5. The paired legend/column ablation answers a narrower question and retains a known weak representation

**Severity: major.**

**Evidence.** The v4 ablation has **180 strips per arm, 45 matched training document classes, and 68 real validation boxes**. I verified zero train/validation well overlap in those arms. However, `genlab/dataset_eval.py:223`–`:234` evaluates cross-log names after restricting truth to names supported by the selected training subset. It does not measure whether learning this log's column improves its own legend-conditioned matching, nor represent all 262 training classes or agent-made sources.

The ablation still uses squashed 12×20 RGB (`dataset_eval.py:61`), while the main recognition logistic uses native features (`genlab/dataset.py:786`, `:819`). With exactly the same saved ablation images and 68 truths, I reproduced **19/68 versus 9/68** for legend/column using legacy RGB, but obtained **41/68 versus 29/68** using native features. The direction survives this probe; the absolute quality and effect size depend strongly on the representation. In addition, `dataset.py:483`–`:485` advances one RNG through both arms; `genlab/render.py:22` consumes it for phase, so equal sample counts do not mean matched phases.

**Fix / exact protocol.** Name the current result “cross-well training-generator ablation on supported names.” Freeze its selected classes and all 68 truth IDs across versions; report excluded-class coverage. Evaluate both arms with the same intended downstream model/representation. Pair strip dimensions, fractional phase draws, nuisance draws, training seeds, and optimization budget; change only the tile source.

For the separate within-document benefit question, use development/training wells only: partition their column annotations into nonoverlapping depth blocks before fitting any model, fit column tiles on source blocks, and score both arms on disjoint blocks with a buffer of at least the maximum source crop height. Do not learn column models from the outer held validation/test wells. Then test the resulting generation strategy through F1's outer-well transfer protocol.

**Effort: M. Expected gain:** attributes differences to the source of the pattern, with explicit limits on the population covered.

## F6. Boundary F1 counts annotation and processing edges as geological boundaries

**Severity: major.**

**Evidence.** Direct calls to `genlab/assemble/bench.py:158` produced these counterexamples:

| Truth and prediction | Row accuracy | Boundary F1 |
|---|---:|---:|
| Only rows [20,80) labeled A; predict A across [0,100), `labelled_only=True` | 100% | 0 |
| Truth A across [0,100); predict A as adjacent [0,50), [50,100) chunks | 100% | 0 |
| Truth A/B with boundary 50; predict C/D with boundary 50 | 0% | 1 |

`boundaries` at `bench.py:153` takes every interval endpoint. The labeled-row mask at `:171` does not constrain boundary scoring at `:179`. Real truth is clipped at 2048-row processing blocks (`:831`), while predictions are concatenated without merging equivalent adjacent intervals (`:811`). A partial annotation edge or block cut can therefore change F1 with no change in lithology prediction. The third case is legitimate localization-only F1, but must not be mistaken for correct labeled segmentation.

**Fix / verification.** Canonicalize adjacent equal-label prediction runs after joining blocks. Score semantic transition boundaries separately from explicitly annotated visible bed boundaries; do not demand recovery of invisible same-class synthetic cuts. On sparse real truth, score only boundaries whose identity and both neighboring regions are independently verified; mask unknown regions and annotation edges. Add class-aware segment IoU/F1 alongside localization F1. Publish tolerances k=1,3,5 native rows, with scale metadata, rather than only k=5. The first two counterexamples must stop generating false boundary errors; the third must retain localization success but fail labeled segmentation.

**Effort: M. Expected gain:** makes boundary results reflect observable geology rather than annotation coverage and chunking.

## F7. Fidelity scores compare sparse selected crops with complete synthetic geology

**Severity: major.**

**Evidence.** During the audit, starter validation contained **56 real intervals / 4,967 labeled rows in five wells**, compared with **4,715 synthetic layers / 446,172 image rows**. There were **zero adjacent human transitions** in both saved validation and test references. Validation composition TV was **0.84885**, marked fail; **39/40** family/class thickness comparisons were warnings. These are saved-artifact measurements, not independently rerun full-raster evaluations.

`genlab/assemble/bench.py:353` correctly refuses to invent transitions. However, `:357` pools composition conditioned on selected labeled real rows against complete synthetic lithology; `:373` treats crop heights as thickness observations even while noting censoring. `:386` downgrades sparse layer/gap rates, but composition still receives a numerical failure judgment. Different class/family mixtures and the owner's crop selection can drive this discrepancy without establishing a bad geological generator. The inspected starter sheet also pairs different motifs within a well, such as red L-marks versus red dotted motifs; that helps inspect general rendering but does not isolate conditional pattern realism.

**Fix / exact protocol.** Establish a continuous-window reference: select wells and fixed depth windows without looking at predicted classes, annotate every layer and true blank gap in those windows, and mark edge-censored beds. Sample synthetic windows with matching family, scan scale, window length, and declared class-support coverage. Evaluate complete transition chains and uncensored thicknesses, with whole-well uncertainty. Report scan-normalized or calibrated thickness as well as native rows where scan scales differ.

Until that reference exists, label composition and thickness outputs “selected-annotation diagnostics”; do not qualify whole-log fidelity from them. Report conditional per-class appearance separately from class prevalence. Use the same observation mask on synthetic data for a clearly named sparse-observation diagnostic, but acknowledge that a mask alone cannot reproduce an unknown class-dependent annotation-selection process.

**Effort: L. Expected gain:** separates annotation bias and population mismatch from actual errors in transitions, thickness, and composition.

## F8. The seeded suites are useful smoke tests, not a representative evaluation sample

**Severity: minor.**

**Evidence.** Saved manifests contain three images/three wells for typical, thin-beds, rare-classes, and heavy-noise. Typical covers only **four document classes**. Look-alikes contains **three images but two wells**, and all three exercise the same named pair, **Halite/Gypsum**. Page-e2e has **one image, one well, 999 rows, and six layers**. `genlab/assemble/bench.py:968` defaults to three samples; `:1062` orders documents by available generated classes. These are coverage-limited examples, even when suite generation status says pass.

**Fix / exact protocol.** Keep these artifacts as fast regression fixtures. Create a separately versioned evaluation suite with a coverage manifest for family, well, class, pair, thickness bin, and noise level. Use at least ten fixed synthesis seeds per eligible well; cover every supported look-alike pair and report wells per pair. Separate natural-mixture evaluation from deliberately reweighted stress suites. Report well/seed dispersion and coverage, never interpret thousands of rows from one base well as thousands of independent trials. More synthetic seeds do not fix the real-reference shortage in F4/F7.

**Effort: M. Expected gain:** catches failures hidden by a few well-supported classes and prevents smoke-test success from being read as population validity.

# 3. Quick wins

Each is scoped to under an hour; larger protocol work remains above.

- Rename “passed” to “evaluation completed” and the page metric to “oracle-column recognition”; display the task beside every score.
- Assert and display clean/augmented held-image hash equality; flag the present 79 identical fake pairs before presenting a realism improvement.
- Add paired whole-well bootstrap intervals from the already saved closed-set predictions; show 79 boxes, seven wells, and 12 low-count classes together.
- Remove TEST branches from `regression_values` and routine promotion summaries; retain final test artifacts separately.
- Merge adjacent equal-label detector chunks and add the two boundary counterexamples above; label sparse-real boundary scores unavailable pending verified boundary masks.

# 4. What is already good

The within-document candidate restriction, reported chance levels, native-feature recognition, and explicit human-only truth are substantial improvements; I reproduced their scores.
V4 validation candidates record swatch-only ancestry, the checked artifacts contain no test-document generation rows, and BENCH has output-path guards and honest contamination/censoring caveats.
Keep native-scale contact sheets, immutable synthetic input snapshots, missing-prediction penalties, one-to-one boundary matching, and separate stress suites.

# 5. Doubts

I did not run full detector training, the unfinished starter utility job, or an end-to-end page locator. I did not independently reconstruct every saved crop from TIFF or audit every donor/proposal ancestor; zero direct test rows is not proof of complete historical provenance. The fixed split explicitly warns that some groups were seen by earlier matchers.

Source files and starter results changed during review, including a useful correction to compare labeled rows in both utility domains. The quoted source lines refer to the inspected versions; v4 pixel evidence and the supplied scratch scripts make the principal measurements reproducible. Fidelity counts are a snapshot of the saved report, not a claim about later rebuilds. Bootstrap intervals with seven wells are exploratory, and the four visually inspected pattern cases do not establish overall realism.
