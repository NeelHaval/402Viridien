**1. Verdict**

Not yet fit as realistic sequence priors or as the basis of a realism benchmark; useful as an evidence inventory.
The biggest problem is treating the annotation process as geology: missing labels become drawable gaps, and cropped examples become complete beds.
Training/test isolation passed this audit, but sampled layer structure fails even a comparison against a fully labelled training span.

**2. Findings, ranked by impact**

Audit basis: `stats.json` built 2026-11-08 03:09:10 UTC, SHA-256 `2765f660f35a3ee279137bd058bd22521e1d5837b8a6a87ac9fe9e447fc065b6`; normalized snapshot `8b8884c431cb8ce8aac5`. The reconstruction matched that snapshot. Implementation references describe the files inspected during this review; other agents are still changing COMPOSE.

**1) Missing annotations become synthetic gaps — blocker.**

Evidence: `stats.py` puts every unknown run into `gaps` (`genlab/assemble/stats.py:120`, `:441`, `:457`); `Sampler.gap()` then samples these lengths (`genlab/assemble/compose.py:298`). Texas has **98.60% unknown rows**; MC765 has **99.90%**. These are annotation coverage figures, not measured blank-rock frequencies. In Callaway, page box **[250,6096,310,6864]** has zero labelled rows but **73.03% nonwhite pixels** and visibly continuous patterned rock. In the native-scale comparison, seeds 0 and 3 generate **67.19% and 70.18% gaps** over the same 768-row span. Forty full-column simulations average **45.55% gaps**. I inspected [the real/fake comparison](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/pattern_lab/data/genlab/critique/a4-stats/texas-unlabelled-native.png). The harness displays unknown draws as white; current COMPOSE can instead paste a measured gap-source image, but still assigns the sampled rows the gap class (`compose.py:504`).

Fix: separate `annotation_unknown` from visually verified blank/occluded intervals in `extract_layers()` and the schema. Unknown must break estimation chains without becoming a geological emission. Only verified gap observations may drive `Sampler.gap()`; a pasted textured region also needs valid lithology truth. Verify that masking additional annotations in an unchanged scan does not increase generated blank frequency.

Effort: **M**. Expected gain: remove a demonstrated source of long artificial gaps and incorrect background ground truth.

**2) Censoring is counted but ignored by the thickness fit — blocker.**

Evidence: **97/97 Texas observations and 4/4 MC765 observations are censored; neither family has an observed adjacent transition.** Nevertheless, Texas receives `high` composition, transition and thickness confidence. Its shale fit uses **31/31 censored fragments**, reports median **68 rows**, and receives `medium` class confidence. `thickness_fit()` uses every length as an exact observation and only adds `censored_n` afterward (`genlab/assemble/stats.py:166`); block confidence depends on layer count and human-row proportion (`:447`). The inspected `data/genlab/assemble/stats/thickness-texas-rock-types-box.png` presents smooth fits to these fragments. The Callaway comparison above shows why crop-sized beds are inappropriate: real pattern continues through the entire 768-row window while fakes repeatedly change class.

Fix: distinguish boundary-complete beds, boundary-censored intervals and deliberately selected texture examples. Fit exact beds with a density likelihood and appropriate censored observations with a survival/interval likelihood; selected interior crops need an explicit observation model or exclusion. With zero complete beds, mark thickness unidentified and require a suitable external prior or boundary annotation. Compute confidence separately for each field using complete-bed and adjacent-transition support. Verify by cropping known full beds: the inferred distribution must not collapse toward crop height.

Effort: **M**. Expected gain: prevent confidently generating the annotation tool's crop lengths as geological bed thicknesses.

**3) A weight of 0.355 does not correct machine-only sequence bias — major.**

Evidence: remeasuring Hadrian's **31,167 jointly covered rows** gives **673 human versus 380 machine layers**, medians **23 versus 49.5 rows**, and boundary recall **34.73%**. Selected smoothing threshold is zero. Replacing machine weight 1 with **0.3545577** changes normalized transition probabilities by at most **1.2e-13** and fitted log-means by **4.0e-13**: a uniform weight cancels within a machine-only distribution. This follows directly from `probability()`, `thickness_fit()` and calibration (`genlab/assemble/stats.py:132`, `:166`, `:284`). Gulf evidence is **99.44% machine rows**. Geoservices' effective layer pool is **89.17% Hadrian**; MC761's three documents represent **one well group**, so neither is broad independent calibration. I inspected `data/genlab/assemble/stats/composition-along-depth.png`; its broad machine-assigned runs must not be interpreted as independently validated stratigraphy.

Fix: export human-only and machine-only sufficient statistics separately. Calibrate boundary detection and duration errors on contiguous training annotations, stratified by scan scale/family; obtain additional wells before transferring that calibration. Use machine evidence through an observation-error model or shrink it toward an independently supported prior. Report distinct-well support and avoid counting another page of the same well as independent backoff evidence. Verify recovery of layer count, short-bed share and boundary recall on withheld training-well spans. Do not blindly divide every duration by 2.15.

Effort: **L**. Expected gain: address the measured **43.54% layer deficit** rather than merely reducing its nominal evidence weight.

**4) Unvalidated lognormal tails produce too many layers — major.**

Evidence: I simulated the longest continuously human-labelled Hadrian span, relative rows **30466:36112**, using its actual depth fractions. It contains **52 layers**, median **32.5 rows**. Across seeds 0–99, the existing sampler produces **98.15 layers on average**, with a **74.43–125.00** central 95% simulation interval; median thickness averages **30.24 rows**. Thus a plausible median hides an **89% excess layer count**. In fifth-4, grey-shale observed mean thickness is **106.60 rows**; the fitted lognormal mean is **74.68**, **29.94% lower**. This zone has **134 uncensored layers**, so this failure is separate from finding 2. See `genlab/assemble/stats.py:175` and `genlab/assemble/compose.py:289`. I inspected [Hadrian's native-scale real/fake comparison](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/pattern_lab/data/genlab/critique/a4-stats/hadrian-native.png).

A controlled 50-seed diagnostic gives **100.94 layers** with current fits versus **76.40** when sampling weighted observed lengths from the selected document/zone tables. This improves the count but does not establish a complete fix; the span may also need finer conditioning and duration dependence.

Fix: validate whole generated sequences, not just fitted medians. Compare empirical duration sampling, lognormal mixtures and existing fits using contiguous withheld training spans. Preserve enough duration support to represent long beds; five quantiles alone do not preserve tails. Check layer count per depth, occupancy, mean duration, tail quantiles and run-length dependence. Gate “realistic” suites on these comparisons.

Effort: **M**. Expected gain: the measured empirical-duration ablation cuts the excess count from **48.94 to 24.40 layers**, approximately halving this particular error.

**5) Family fallback transfers pixel durations across a fivefold scale difference — major.**

Evidence: MC765 documents `4a056c8ff25cf749` and `59c21186a4a3c9e9` both cover **3686–20707 ft MD**, at 1-inch and 5-inch scales. Their column heights are **34,650 and 173,248 rows**, a **4.99994×** ratio. Ten thousand thickness draws per document both produce median **96 rows**, entirely from the same family fallback. These imply approximately **47.16 versus 9.43 ft** for the same pooled duration prior. Fraction fits exist (`genlab/assemble/stats.py:174`), but COMPOSE reads only pixel `mu`, `sigma` and quantiles (`genlab/assemble/compose.py:271`, `:289`). The inspected composition-depth plot includes both scans as separate pooled documents.

Fix: attach calibrated pixels-per-depth-unit and MD/TVD coordinate type to each document. Pool compatible physical durations, then convert to target pixels. Column fractions are a fallback only when depth extents are comparable; they do not generally remove differences in logged interval length. Until calibrated, restrict duration borrowing to a matching scale and coordinate system. Verify with these two same-interval scans: equal physical durations must render approximately five times taller in the 5-inch scan. Resolve censoring before treating either crop as a measured bed.

Effort: **M**. Expected gain: remove a directly reproduced **fivefold physical-duration inconsistency**.

**6) Order-two backoff uses the wrong support and ignores exported weights — major.**

Evidence: zone transition probabilities are exported without zone transition counts (`genlab/assemble/stats.py:433`, `:452`). COMPOSE validates a zone row using **whole-document** context counts (`genlab/assemble/compose.py:245`). Hadrian fifth-5's **sandstone → siltstone → grey shale** context has **one** observed next event, but passes on whole-document support **38.01**; all **100 diagnostic draws** choose grey shale through `order2_by_zone`. Five Hadrian zone contexts with support below five pass this test, three deterministically. The sample-size backoff weights produced at `genlab/assemble/stats.py:604` have no consumer in COMPOSE: setting every top-level weight to global-only leaves a fixed-seed sequence unchanged.

Order two also lacks demonstrated predictive benefit: fitting raw human-only tables on Hadrian's first 60% and checking 310 contiguous triples in the last 40% gives **46 zero-probability events for order two versus 33 for order one**. This is a same-well diagnostic, not a population estimate. The native Hadrian comparison illustrates the resulting sequences, but does not itself establish which Markov order is best.

Fix: export counts, source quality and distinct-well support for each zone/context. Smooth order two toward order one using that context's support, then back off across document/family/global using field-specific evidence. Either consume the exported weights or replace the misleading contract with the actual selection rule. Select order using contiguous-block and training-well validation. Verify that a one-event zone context cannot become a deterministic high-confidence rule merely because its global context is common.

Effort: **M**. Expected gain: remove demonstrated one-example certainty and make sparsity/backoff controls affect generation.

**7) Adaptive sections are unreachable, and pooled boundaries are not a partition — minor.**

Evidence: `build_block()` orders fifths before sections (`genlab/assemble/stats.py:416`); `Sampler.zone()` returns the first matching entry (`genlab/assemble/compose.py:168`). All **1,001 tested Hadrian positions** select a fifth and **zero** select any of its four adaptive sections. The inspected `composition-along-depth.png` displays adaptive boundaries that generation never uses. Independently taking median pooled bounds (`stats.py:422`) gives Texas section-1 ending at **0.20** and section-2 starting at **0.1875**, an overlap of **1.25%** of a column. The report warns about document-specific bounds, but the consumer has no explicit scheme selection.

Fix: expose separate `fifths` and `sections` conditioning schemes; select one explicitly and use document-specific section bounds. Do not treat cross-document section rank as a shared geological zone. Validate a complete, nonoverlapping partition. Compare schemes empirically: merely preferring sections worsened the same 50-seed Hadrian count from **100.94 to 110.64**, so activating them is not an automatic realism improvement.

Effort: **S** for explicit selection/partition checks; **M** for validation. Expected gain: make reported position conditioning reviewable and usable without silently discarding it.

Reproduction: `data/genlab/critique/a4-stats/audit.py` contains read-only modes `overview`, `weighting`, `simulations`, `structure`, `fitcheck`, and `visuals`. Run `PYTHONDONTWRITEBYTECODE=1 python -X utf8 data/genlab/critique/a4-stats/audit.py MODE` in Git Bash. Full-column simulations used 40 seeds per document; the labelled-span comparison used 100. Numerical simulations made all classes with document thickness evidence available to isolate statistics from missing-pattern coverage; image comparisons used actual drawable patterns. Images are native 1:1 pixels, with appearance noise omitted. TIFF reads were windowed; the visual run peaked at **316 MiB**. No implementation or shared output was edited.

**3. Quick wins — each under an hour**

- Stop emitting gaps from annotation-unknown intervals; retain their masks only for estimation coverage.
- Mark zero-complete-bed thickness and zero-transition blocks unsupported; display `censored_n/n` in every fit plot.
- Export zone transition counts and use them for the existing minimum-support check.
- Add explicit zone-scheme selection and a nonoverlapping-partition assertion; keep sections opt-in pending validation.
- Add a fixed-seed sequence diagnostic reporting layer count, mean duration, tail quantiles and row occupancy beside the existing fit summaries.

**4. What is already good**

The split audit found **32 training documents, 19 excluded documents, and zero test-document hits** in classes or pooled membership; metadata imports exclude endpoint row shares (`stats.py:471`, `:489`, `:545`).
Human precedence, conflict-to-unknown handling and chain breaks are valuable. Existing tests passed: **9 passed in 3.11 s**, using `python -X utf8 -m pytest genlab/tests/test_assemble_stats.py -q -p no:cacheprovider --assert=plain` with bytecode disabled.
Preserve well-group bootstrap resampling, explicit low-support warnings, fractional duration fields and the report's candid calibration limitations.

**5. Doubts**

Hadrian currently has human labels on **31,168/47,921 rows**, not a complete column. Outside contiguous labelled spans, observed runs are not exact full-log ground truth. The comparisons above are training-data diagnostics, not held-out-well performance claims; no test-well labels or pixels were used.

I did not establish physical depth calibration beyond the explicit same-interval MC765 pair, recover missing beds in sparsely labelled families, or verify the upstream training provenance of the prefill producer. No direct test-well leakage was found in the inspected statistics path. Bootstrap population uncertainty remains unidentified for one-well families.

COMPOSE/NOISE/BENCH are changing concurrently. The report identifies reproduced behavior in the inspected sampler and statistics, not unfinished page-composition or benchmark features. Visual evidence comes from two training documents and the existing statistics plots; it is not a review of all 51 scans. Implementation fixes and independent-well validation remain outstanding by design: this is a review only.
