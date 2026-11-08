# 1. Verdict

The compositor is a useful structural prototype, but its current outputs are not fit as representative synthetic logs.
The biggest problem is that composition does not preserve the source log's lithology row shares.
The review below distinguishes missing pattern support from defects in sampling, depth mapping, phase continuity and page stitching.

# 2. Findings, ranked by impact

## 1. Representative runs silently lose the dominant lithology

- **Severity:** major.
- **Evidence:** For MC761 document `97adb9d2c69d860b`, the current document statistics assign **58.36%** of known weighted rows to Shale (`:14`), but `Composer.available()` excludes it because no same-name generated pattern is available. Starter column `column-mc761-mineral-lithology-97adb9d2c69d860b-00` consequently contains **0% Shale**, 51.56% Sandstone and 32.19% Siltstone. In the native-scale comparison at page rows 6373–7573, synthetic A/B contain **73.42% / 82.50% Sandstone**, while the real stretch is visibly dominated by grey shale. I inspected `data/genlab/critique/s4-compose/stretch-97adb9d2c69d860b.png`. Selection and exclusion occur at `genlab/assemble/compose.py:364`; probability renormalization at `genlab/assemble/compose.py:265`. Recording `skipped_classes` at `genlab/assemble/compose.py:518` does not prevent publishing an unrepresentative sample.
- **Fix:** In `available()` and run selection, calculate supported composition mass for each relevant zone before renormalizing. Gate representative runs on a documented coverage threshold, for example 95%, and mark lower-coverage runs explicitly as partial-class experiments. Prioritize the missing MC761 Shale pattern. **Verify:** report supported row mass and compare generated row shares against the supported source distribution over multiple seeds; this document must fail the representative-run gate until coverage improves.
- **Effort:** S for the coverage gate; pattern completion is separate ongoing work.
- **Expected gain:** Removes a measured 58-point class-coverage failure from supposedly representative training data. Missing makers' outputs are unfinished work; silently treating their absence as realistic composition is the compositor defect.

## 2. Composition fallback samples row probabilities as bed probabilities

- **Severity:** major.
- **Evidence:** `genlab/assemble/compose.py:251` passes the row-share composition table directly to `choose()`, then independently draws bed thickness at `genlab/assemble/compose.py:322`. A controlled run with equal target row shares, deterministic 10-row and 100-row beds, seed 173 and 10,000 layers produced **5,046 / 4,954 beds**, but **9.244% / 90.756% of rows**, instead of 50% / 50%. Reproduction: `python -X utf8 data/genlab/critique/s4-compose/audit.py probes`. This path is used throughout the Texas and MC765 starter outputs. The long uninterrupted synthetic colour stretches are visible in `data/genlab/critique/s4-compose/stretch-97adb9d2c69d860b.png`; the controlled experiment establishes the defect independently of that document's missing patterns.
- **Fix:** Convert desired row shares to entry probabilities using `p_enter(class) proportional to row_share(class) / E[thickness(class, zone)]`, using the actual document/family thickness fallback chosen by the sampler. Alternatively, control an explicit row-occupancy budget. **Verify:** repeat the controlled case across seeds; aggregate row shares should approach 50% / 50%, including unequal thicknesses and zone changes. Keep the existing transition test separately.
- **Effort:** M.
- **Expected gain:** Eliminates the measured **40.76 percentage-point** occupancy error when transitions back off to composition.

## 3. Annotation coverage gaps become enormous physical texture runs

- **Severity:** major.
- **Evidence:** MC765 document `4a056c8ff25cf749` has **34,564 unknown rows out of 34,650** in the statistics, reflecting sparse annotation coverage. Its gap texture is the first **2,048 rows**, page y=1623–3671, recorded as `unlabelled-coverage-gap`. Starter column `column-mc765-colour-grid-4a056c8ff25cf749-01` inserts a **13,081-row** gap from output y=6919 to 20000 and cycles that source texture. At real page rows 32599–33799, synthetic B is entirely this mostly white gap, while the real column is predominantly green patterned fill. I inspected `data/genlab/critique/s4-compose/stretch-4a056c8ff25cf749.png`. Source selection is `genlab/assemble/noise.py:525`, especially `:561`; periodic copying is `genlab/assemble/compose.py:494`. Gap lengths are sampled at `genlab/assemble/compose.py:298`.
- **Fix:** Separate missing-label coverage from observed blank/unknown physical appearance. Do not use coverage-hole lengths as blank-bed thickness evidence. For unknown training windows, retain their correct depth context and ignore their semantic labels; learn genuine blank-region lengths from image evidence. Avoid cycling a short source patch through a much longer unknown interval. **Verify:** report annotation coverage and visual blank fraction separately; check the affected native-scale stretch and measure long-lag repetition in generated gaps.
- **Effort:** M.
- **Expected gain:** Prevents sparse labels from creating very long runs of repeated top-of-column paper texture and dominating the generated log.

## 4. Oversized columns stretch depth zones while claiming native page coordinates

- **Severity:** major.
- **Evidence:** The MC761 source column is page y=2197–8198: **6,001 rows**. Its starter output is **20,000 rows** with `real_depth_span=[2197,22197]`. At page y=6373, the true relative position is **0.696**, or fifth-4; the output's corresponding layer is tagged **fifth-2**, because its zone calculation divides by 20,000. The same native-scale comparison image cited above displays this mismatched context. `genlab/assemble/compose.py:485` caps `span_frac` at the remaining real fraction, while `genlab/assemble/compose.py:320` spreads it over the requested output height; `genlab/assemble/compose.py:520` still records a native coordinate span extending beyond the source.
- **Fix:** For comparisons and page-context columns, cap height to `y1-origin`, or reject the oversized request. If arbitrary-length synthetic logs are required, give them an explicit synthetic coordinate system and map source zones separately; do not claim they cover nonexistent source rows. **Verify:** a 20,000-row request against this document must be rejected or capped to 6,001 rows; at source y=6373 the sampler must select fifth-4.
- **Effort:** S.
- **Expected gain:** Restores correct depth conditioning and makes same-depth visual comparisons meaningful.

## 5. Consecutive equal classes restart their pattern at artificial boundaries

- **Severity:** major.
- **Evidence:** MC765 starter column `...-00` contains **192 adjacent equal-class joins**. Replaying the sampler with its recorded seed and matching its stored layer coordinates exactly shows **all 192 joins choose different tile phases**. Its tile period is **12 x 28 pixels**, and `anchored=False`. MC761 starter A also has **118 equal-class joins among 216 joins**. Look at the repeated motif discontinuities in `data/genlab/critique/s4-compose/stretch-4a056c8ff25cf749.png`, and the yellow runs in `stretch-97adb9d2c69d860b.png`. Layers are appended separately at `genlab/assemble/compose.py:328` and rendered separately at `:506`; random per-call phase is selected at `genlab/render.py:22`. Boundary appearance is also applied to every stored layer at `genlab/assemble/compose.py:448`, including equal-class joins.
- **Fix:** Maintain phase across a contiguous run of the same class and pattern source. Coalesce adjacent equal-class intervals when they represent one semantic bed; retain any sampling-event metadata separately. Apply physical boundary effects only at a justified boundary. **Verify:** splitting one same-class run into two sampling events must produce the same pixels as one continuous render; no artificial line should appear at the join.
- **Effort:** S–M.
- **Expected gain:** Removes hundreds of artificial seams and prevents classifiers from learning phase resets as lithology boundaries.

## 6. Page replacement loses crossing lines outside a sparse residual library

- **Severity:** major.
- **Evidence:** Page restoration only considers usable residual measurements at `genlab/assemble/compose.py:537` and copies boundary-connected components at `:551`. Those measurements cover just **1,119 / 38,251 rows (2.93%)** of the Texas page, and **2,976 / 47,921 rows (6.21%)** of Hadrian. On the Texas page, a conservative native-pixel test found 10 real rows with a dark horizontal line spanning over 85% of the column interior and continuing into a neighbouring track; **9 of 10** no longer span the synthetic column. Examples include page y=40055, 40355 and 41255. I inspected the side-by-side crop `data/genlab/critique/s4-compose/crossing-40055.png`: the neighbouring line remains, but its continuation through the replaced column disappears. Broader context: `data/genlab/critique/s4-compose/page-d368760651ded938.png`. The numerical test is in `audit.py crossings`.
- **Fix:** During banded page replacement, detect crossing components from that band's own original pixels and neighbouring tracks, rather than limiting restoration to previously measured swatch boxes. Restore validated lines/ticks/curves after filling, preserving their original coordinates; record their masks or provenance in truth. Export the final embedded column consistently after restoration. **Verify:** preserve all 10 measured crossing rows, inspect y=40055, and retain the existing assertion that other tracks are unchanged.
- **Effort:** M.
- **Expected gain:** Removes a measured **90% crossing-line loss** in this example and reduces a strong page-stitching cue.

# 3. Quick wins

1. Reject or cap source-context columns taller than the remaining source span.
2. Add supported row mass and omitted row mass to each run's manifest and representative-run eligibility check.
3. Carry a fixed phase through adjacent equal-class render calls.
4. Add the deterministic 10/100-row, equal-occupancy fallback check alongside the Markov test.
5. Add the Texas y=40055 crop and crossing-row count to page acceptance checks.

# 4. What is already good

The existing compositor suite passes: **13 tests in 15.89 seconds**, including mask tiling and reproducibility. Banded page output preserved other tracks exactly in the two inspected bands. Current split checks found **zero held-out source documents across 20 starter outputs and 354 noise measurements**; preserve the training-only gates and windowed TIFF reader.

# 5. Doubts

- Source files and outputs were changing during review. I saved fresh native-scale comparison images under the assigned scratch directory; measurements and source hashes are in `data/genlab/critique/s4-compose/measurements.txt`. The initial board sheets were not treated as stable evidence.
- Reviewed three same-document stretches and two page contexts, not all 51 logs. Crossing-line counts use the explicitly stated dark-line heuristic; they do not measure all text, curves or faint ticks.
- Missing patterns are still being built. The coverage finding concerns current run eligibility, not a claim that makers should already have completed them. Source row shares outside fully labelled logs remain weighted estimates.
- No full TIFF page was loaded. Peak RSS was not independently instrumented. I did not rerun downstream detector/classifier benchmarks or claim a measured recognition improvement.
- Reproduction: `python -X utf8 data/genlab/critique/s4-compose/audit.py` with `summary`, `stretches`, `pages`, `crossings`, `probes` or `sources`. Tests: `python -X utf8 -m pytest genlab/tests/test_assemble_compose.py -q --basetemp=data/genlab/critique/s4-compose/pytest-tmp -o cache_dir=data/genlab/critique/s4-compose/pytest-cache`, with `PYTHONDONTWRITEBYTECODE=1`.
