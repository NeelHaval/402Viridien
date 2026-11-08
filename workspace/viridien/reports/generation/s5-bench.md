**1. Verdict**

Fit for debugging scorer arithmetic; not yet fit for selecting detectors expected to generalize to real logs.
The biggest problem is suite composition: a constant label per image can explain 98.21% of look-alike rows.
Page localization, boundary censoring, artifact integrity and utility acceptance also need fixes before benchmark scores support release decisions.

**2. Findings, ranked by impact**

**1) Challenge suites allow high row accuracy with almost no discrimination**

Severity: **blocker** for model selection.

Evidence: I scored a truth-selected per-image majority constant predictor, an imbalance diagnostic rather than a deployable detector:

| Suite | Images | Exact classes | Constant-predictor row accuracy |
|---|---:|---:|---:|
| typical | 3 | 4 | 92.07% |
| thin-beds | 3 | 11 | 65.10% |
| rare-classes | 3 | 7 | 89.91% |
| look-alikes | 3 | 6 | 98.21% |
| heavy-noise | 3 | 6 | 63.48% |
| page-e2e | 1 | 4 | 67.37% |

Boundary F1 is zero for all these predictions, so the other metrics do expose the failure if examined. Nevertheless, the look-alike suite's three samples all test **Halite/Gypsum**, with majority row shares 99.34%, 96.97% and 98.32%. Typical has two entirely single-class images. Available generated training inventory was 389 classes across 32 documents. See `data/genlab/critique/s5-bench/extra-results.json`, `data/genlab/assemble/suites/typical/suite.json:124`, and `data/genlab/assemble/suites/look-alikes/suite.json:5400`.

Cause: document selection ranks inventory size, then takes the first documents; pair expansion visits many documents for one pair before moving to the next. Equal composition probabilities do not guarantee balanced **rendered rows** when thickness differs. The acceptance check only requires both names to occur: `genlab/assemble/bench.py:1088`, `:1121`, `:1136`.

Fix: in `build_suite`, stratify across well/family/class and deduplicate normalized pair identities before assigning documents. Require minimum bed counts and realized row shares for both members, for example 25–75% in the balanced look-alike challenge. Preserve natural prevalence in typical, but add class/family coverage targets and publish majority, macro-recall and boundary baselines. Verify that different pairs appear and the balanced challenge's majority baseline stays below its declared ceiling.

Effort: **M**. Expected gain: removes the demonstrated 98.21% row-accuracy shortcut and makes rare-class failures visible.

**2) All-well look-alike analysis can influence generation through test evidence**

Severity: **major**.

Evidence: the fixed policy hashes assign 19 documents to test. Of 285 `/api/mix` pairs, 94 include a test source and 64 have exclusively test source logs. Reproducing the builder's name-based expansion produced 583 eligible document/pair combinations, including 229 with test-source evidence. The builder ignores `logs`, `files` and family provenance when matching pair names into training documents: `genlab/assemble/bench.py:1088`.

The upstream analysis computes pair features and ambiguity row stakes across all logs, aggregates stakes and sorts pairs by them: `../bakeoff/labeltool/mix_pairs.py:149`, `:173`, `:185`, `:192`. Thus increasing suite size or changing inventory can let test-derived resemblance or prevalence select generated challenges. **The current first three selected pairs have training-only source logs; this is a demonstrated exposure path, not a claim of current test-pixel copying.** Counts are saved in `s5-bench/extra-results.json` relative to this report.

Fix: generate a training-only pair catalog before aggregation/ranking, with source document IDs and split-policy hash. Keep all-well analysis read-only for evaluation. Do not merely remove mixed pairs after aggregation; their ordering already incorporates test row stakes. Verify by changing only test swatches/shares and asserting unchanged generation plans and profiles.

Effort: **M**. Expected gain: closes a test-to-generation selection path while retaining independent test evaluation.

**3) page-e2e evaluates classification inside the answer box**

Severity: **major**.

Evidence: `detector` reads `bbox_annotation.json`, extracts the true lithology-column box, crops it, and passes that crop to the same column classifier: `genlab/assemble/bench.py:801`, `:809`. `native_predict` constructs a log whose column box is the complete supplied crop (`:704`). Suite scoring accepts column-relative intervals and has no predicted-box input or localization metric (`:1169`). A system unable to locate any column can therefore achieve perfect page classification when supplied this oracle crop. The published page suite currently has only one image, 999 rows and four classes (`data/genlab/assemble/suites/page-e2e/suite.json:59`).

Fix: label the current result “page rendering, oracle column crop.” Add a genuine page detector interface taking page bands without truth boxes, returning page-coordinate boxes and class intervals. Score box IoU, missed/extra columns and joint localization/classification. Verify that a wrong or absent column box fails even with otherwise correct interval labels; broaden page geometry coverage.

Effort: **L** for full evaluation; **S** for truthful labeling. Expected gain: measures the column-detection failure mode required by the end-to-end goal.

**4) Suite versions do not identify all inputs, and corrupted images pass cache validation**

Severity: **major**.

Evidence: my isolated mock-COMPOSE reproduction built version `8f49928b54b7`, replaced its column image with all-white pixels and removed its only generated card. A second build returned the same version with `status=pass`, retained the corrupted image, and made no additional COMPOSE call. See `s5-bench/extra.py:16` and `s5-bench/extra-results.json`.

Versioning hashes the plan and statistics before reading live cards, pair analysis or noise measurements. Cache validation hashes truth JSON but only checks image existence: `genlab/assemble/bench.py:1061`, `:1064`, `:1072`. Generator/noise code and tile/model bytes also lack version coverage.

Fix: include generator revision, tile/model hashes, inventory/verdict snapshot, split policy, noise measurements and pair catalog in an immutable input manifest. Hash every scored image, mask, page band, bbox and truth file; validate these before returning or scoring a cached suite. Reject mutation of an existing version. Verify with the supplied pixel-tamper/withdrawn-card reproduction and a changed tile under the same class ID.

Effort: **M**. Expected gain: scores become comparable for a version, and image corruption cannot silently survive.

**5) Real boundary F1 scores annotation edges and processing seams**

Severity: **major**.

Evidence: a 100-row constant-class prediction against a correct human interior box `[40,60)` receives row accuracy 1.0 on labeled rows but boundary F1 **0.0**: the scorer invents true boundaries at both annotation edges. Separately, splitting a perfect constant-class prediction into `[0,50)` and `[50,100)` changes boundary F1 from 1.0 to **0.0**, despite identical row labels. Both reproductions are in `s5-bench/audit.py:54` and `s5-bench/results.json`.

`labelled_only` masks row metrics but not boundaries (`genlab/assemble/bench.py:171`, `:179`). Real evaluation clips truth at 2048-row processing edges and concatenates predictions without seam stitching (`:830`, `:831`). Preserving same-class geological bed boundaries is intentional; processing splits and arbitrary human crop edges are different evidence.

Fix: carry boundary observability/censoring separately from labeled rows. Score real boundaries only where human annotation establishes an actual bed edge; leave sparse crop endpoints unknown. Rejoin fragments of the same truth interval by sample ID and stitch predictions specifically across processing seams. Keep explicit same-class bed boundaries in synthetic truth. Verify invariance to block size and interior annotation cropping, while retaining penalties for genuine extra geological boundaries.

Effort: **M**. Expected gain: removes the two demonstrated false-zero boundary scores and makes synthetic/real boundary comparisons meaningful.

**6) Equal failure on real and synthetic data is accepted as useful transfer**

Severity: **major**.

Evidence: I executed `detector` with scratch images, a mock crop reader and a predictor wrong on every row. Synthetic accuracy **0%**, real accuracy **0%**, gap **0**, overall status **pass**. The acceptance logic uses only the pooled `human_all` absolute gap (`genlab/assemble/bench.py:848`). Per-class gaps are reported but do not affect this status. Reproduction: `s5-bench/extra.py:128`; output: `s5-bench/extra-results.json`.

A small discriminator gap is also insufficient evidence of class fidelity: the existing starter discriminator passes at **56.875%**, using 480 held-out windows per domain. In my freshly rendered native-scale comparison, green shale, sandstone and halite resemble their references, while the shale-gray generated patch has a gray background/banding absent from the mostly white real patch. Image inspected: `data/genlab/critique/s5-bench/native-comparison.png`, four matched classes at native scale; this is visual evidence, not a quantified population error rate.

Fix: separate evaluation completion, domain-gap agreement and demonstrated detector utility. Require useful real performance above declared baselines before issuing a utility pass. Compare per-class/family recall and boundary performance, report unsupported strata, and estimate uncertainty by well. Validate with the zero-skill fixture and a fixture whose opposite class errors cancel in the pooled gap. Use validation for tuning; retain test for evaluation only.

Effort: **M**. Expected gain: the demonstrated 0%/0% detector can no longer certify representativeness.

**7) Dataset summaries silently truncate mismatched prediction arrays**

Severity: **minor**; current internal predictors normally return matching lengths.

Evidence: `closed_summary` given four truths, one correct prediction and four candidate counts reports top-1 **25%**, macro **100%**, `macro_n=4`, but per-class support **1**. `summarize` accepts extra predictions; a missing prediction can disappear from per-class counts or cause `IndexError` if its name is in the training vocabulary. Both use unchecked `zip`: `genlab/dataset.py:690`, `genlab/dataset_eval.py:27`. Reproductions and outputs: `s5-bench/audit.py:62`, `s5-bench/extra.py:108`, and their JSON results.

Fix: assert equal truth/prediction/candidate-count lengths at scorer entry. If incomplete predictions are a supported format, explicitly insert misses by stable sample ID before summarizing. Verify missing, extra and reordered predictions; check that per-class supports sum to total truth support.

Effort: **S**. Expected gain: prevents contradictory macro/support numbers and catches failed prediction batches immediately.

**3. Quick wins — each under an hour**

- Display realized row shares and a constant-class baseline next to each suite's build status.
- Deduplicate look-alike names before cycling through source documents.
- Add length assertions to both dataset summary functions and their adversarial cases.
- Relabel the current page metric as classification with an oracle column crop.
- Make utility status warn when both domains fail declared baseline performance, even if their gap is zero.

**4. What is already good**

Core boundary matching passed **2,883** exhaustive small cases; row accuracy, per-class P/R and malformed-interval rejection passed direct checks. **12 targeted existing tests passed; 7 were deselected** to avoid writes outside the allowed scratch directory.
No fixed-policy test document appeared in the inspected statistics class inventory, suite base documents or noise measurements. Heavy-noise sigmas were **11.8608, 23.7216, 8.8956**, all above the recorded **7.413** training p95.
V4 distinguishes tasks: closed-set clean matcher **57/79 = 72.152%**, versus cross-log **7/79 = 8.861%**; retain that distinction and swatch-only held-well candidate provenance.

**5. Doubts and audit limits**

Commands run: `python -B -X utf8 data/genlab/critique/s5-bench/audit.py` and `python -B -X utf8 data/genlab/critique/s5-bench/extra.py`. Existing-test output is saved in `s5-bench/pytest-output.txt`; adversarial inputs and measured results are retained beside the scripts. The extra probe's measured Windows peak working set was about **241 MiB**. No TIFF page was decoded; fresh comparisons used saved native real RGB and generated RGB.

I inspected the fresh four-class comparison plus `data/genlab/assemble/bench/runs/starter/same-scale.png` and `data/genlab/assemble/bench/suite-sheet.png`. These cannot establish fidelity for all classes or overlays. Sparse human boxes cannot supply complete real boundary truth, transition statistics or an unbiased thickness distribution.

The starter detector JSON was absent at inspection, so its actual utility evaluation remains **unfinished**, not failed. I did not rerun costly generation, full TIFF detection or full dataset evaluation. The reported majority predictor deliberately uses truth to diagnose imbalance; its accuracy is not a measured blind detector score. Pair-provenance counts describe a reachable generation path, while the currently selected three pairs have training-only source logs.

Other agents changed code during review. Source copies and SHA-256 hashes are retained in `s5-bench/*.reviewed.txt` and `s5-bench/reviewed-sources.json`; line references identify that inspected source state. Benchmark generation and evaluation outputs were read-only throughout; all review writes stayed in this report or its designated scratch directory.
