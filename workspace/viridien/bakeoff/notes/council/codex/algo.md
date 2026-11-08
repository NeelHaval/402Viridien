**The main risks are already observable: valid rocks are rejected as blank, distinct legend entries are merged, and bed merging discards evidence while retaining high agreement.** I would fix these before adding another model.

I read all three requested files and the earlier experiment reports. Checks were read-only; no files were written or network calls made. Counts below use the final available snapshot of 51 bed files.

## A. Bugs and silent failure modes

1. **Border removal deletes solid lithologies.**
   [legend_classify.py:52–63](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:52), classification at lines 262–271; similar logic in [beds.py:39–48](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py:39).

   Every edge column with sufficient black coverage is considered a frame, even when the entire image is a solid black rock. A `48×40` black patch shrinks to `48×4`, then fails the retained-width threshold and becomes `blank`.

   **Real reproduction:** all nine D-582 **Coal** swatches classified as blank against their own legends. Across the cached legends, 48 of 813 retained swatches failed a self-match check; examples also included Bitumen, Tar, and Shale. These are swatch diagnostics, not an estimate of column accuracy.

   Detect narrow, persistent frame components using the surrounding column. Do not repeatedly infer the frame independently inside each bed chunk.

2. **A whole intensity range is silently classified as blank.**
   [legend_classify.py:269–271](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:269).

   Uniform RGB `(120,120,120)` is neither ink (`max < 90`) nor fill (`max >= 140`), so `classify(..., check=False)` returns `blank`. Uniform grayscale values from **90 through 139** have this problem.

   This is an implementation gap between two definitions, not merely a poorly chosen threshold. Background, ink, and fill should form a complete model of the pixels.

3. **“Twins” merges genuinely different rocks with distinguishing marks.**
   [legend_classify.py:191–213](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:191).

   The Allegheny legend explicitly distinguishes **Mudstone, Wackstone, and Packstone** using centre letters over a shared brick pattern. The code merges all three. Measured bidirectional NCC minima were:

   | Pair | NCC |
   |---|---:|
   | Mudstone–Packstone | 0.913 |
   | Mudstone–Wackstone | 0.803 |
   | Packstone–Wackstone | 0.791 |

   All exceed `0.75`. The distinct letters are visible in the [cached legend image](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/out/_cache/legend_v2/608114075800%20-%20Allegheny%20Attic%20GC297%20%233%20OCS-G%2008876%20003%20ST00%20BP00%20%20Mud%20Log%203346ft-13925ft%20MD.debug.png).

   Union-find also makes approximate similarity transitive. Three plain gray swatches at 180, 188, and 196 become `A / B / C`: adjacent ΔE values are approximately 2.93 and 2.91, but the endpoints differ by 5.84, exceeding `twin_dE=4`.

   Preserve separate label IDs. Reserve equivalence groups for near-identical registered images, require agreement across the entire group, and retain ambiguity separately from the submitted label.

4. **NCC can ignore most of a swatch.**
   [legend_classify.py:121–134](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:121).

   When the swatch exceeds the window, only its **top-left** `h×w` portion is used. The rest contributes nothing to template matching.

   **Reproduction:** two `16×16` swatches differing in **85.94% of pixels** both returned NCC `1.0` against an `8×8` query because their selected `6×6` corners matched. This directly undermines both classification and twin detection.

   Compare several representative subtemplates, including distinguishing components, and require adequate matched coverage.

5. **Singleton merging destroys real beds and blank gaps.**
   [beds.py:137–167](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py:137).

   A one-chunk run joins the preceding group, or the following group at the beginning, without comparing either neighbour’s evidence.

   Reproductions using controlled chunk predictions:

   - Three 48-row chunks `A, B, B` become one 144-row **B** bed.
   - `A, A, blank` become one 144-row **A** bed.
   - The final saved outputs contain **93 named beds incorporating 6,469 rows whose chunks were classified as blank**.

   The comment that blank stretches remain blank is therefore false for singleton blank runs. Preserve blank/unknown states explicitly and use accumulated class costs to decide whether a short run is noise.

6. **A rejected group can emerge as `state="ok", label=None`.**
   [beds.py:154–158](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py:154).

   The rejection decision uses row counts, but its replacement state uses chunk counts. For an `ok` chunk of 48 rows followed by a blank chunk of 49 rows, named coverage is below half; the tied chunk-state vote chooses the first state, `ok`, while assigning no label.

   **Three saved beds have this invalid state.** One is:

   ```
   60811405800500_608114058005_g16786_001_03_00_tvd5_mud
   [169336, 169433): state=ok, label=null
   ```

   Its exported windows even have confidence `0.66`. Replaying [conf_data_v4.py:89](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:89) on that saved output raises:

   ```
   TypeError: '<' not supported between instances of 'str' and 'NoneType'
   ```

   Enforce `state == "ok" ⇒ label is a valid legend entry`, and choose rejected states using consistent row-weighted evidence.

7. **Thin merging always chooses upward and drops the absorbed evidence.**
   [beds.py:168–183](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py:168).

   Despite the comment, neither colour similarity nor neighbour height is consulted.

   **Reproduction:**

   ```
   Input:  A [0,48), B [48,56), B [56,104)
   Output: A [0,56),             B [56,104)
   ```

   The first output retains `agree=1.0`, although only its original 48 rows have supporting chunks. Absorbed chunks are discarded, so later confidence recomputation cannot recover the disagreement.

   **Saved-output audit:** 1,140 beds have a combined **78,393 rows absent from their chunk evidence**. Reattach all evidence, then recompute label, agreement, margin, and alternative label.

   The following same-label merge also discards every intervening boundary, including potentially drawn separators that the [spec requires preserving](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/SPEC.md:28).

8. **Boundary handling shifts contacts near the crop edges.**
   [beds.py:65–96](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py:65).

   `step[:h] = step[-h:] = 0`, followed by smoothing, suppresses actual edge contacts and can relocate the remaining peak.

   On synthetic 100-row two-colour columns:

   | True contact | Detected contact |
   |---:|---:|
   | 6 | 13 |
   | 12 | 13 |
   | 25 | 25 |
   | 88 | 86 |
   | 94 | 86 |

   The cumulative-sum convention itself is consistent with exclusive interval bottoms. The defect is the edge suppression policy. Use available support near edges, with uncertainty reflecting the shorter comparison band.

9. **An unseen log can have no legend because the cache is treated as exhaustive.**
   [legend_classify.py:154–188](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:154).

   Once `native_swatches.pkl` exists, a new log absent from it gets `_native().get(log, [])`; no parse is attempted. Nonblank windows consequently become unknown. When building the cache, parser exceptions are converted into empty entries and persisted.

   Other invalidation problems:

   - `legend(log, scale)` omits preprocessing settings and image identity from its key.
   - `twins(log)` omits twin thresholds.
   - `log_scale(log)` omits the scale candidates and scoring parameters; its disk cache also uses only the log name.
   - `beds.main()` skips existing outputs without checking code or parameter changes.

   I reproduced stale `log_scale()` output after changing `P["scales"]` in memory. The evaluator clears some memory caches, but that does not invalidate disk entries. See [legend_classify.py:220](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:220), [beds.py:199](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py:199).

   The filename key also collides for `a b` and `a#b`, or names sharing their first 90 characters. I found no current dataset collision.

10. **`NOT_ROCK` rejects geological phrases and misses technical labels.**
    [legend_classify.py:217](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:217).

    Direct regex checks:

    | Label | Current result |
    |---|---|
    | `Oil shale` | Removed |
    | `Gas-bearing sandstone` | Removed |
    | `Water-bearing sandstone` | Removed |
    | `Mud stone` | Removed |
    | `Mudstone` | Retained |
    | `No Sample` | Retained |
    | `Lost Circulation` | Retained |

    **Actual retained technical entry:** `CEMENTCONTAM.` in the FREEMAN legend cache.

    The synthetic phrases establish failure cases; I did not find those particular false removals among the current cached labels. Use legend section ownership and whole-entry interpretation, preserving rock nouns and their modifiers. A blacklist cannot repair incorrect upstream section assignment reliably.

11. **The comparison and confidence outputs overstate what was evaluated.**
    [conf_data_v4.py:44–58](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:44).

    There are several separate problems:

    - **Asymmetric scoring:** v3 requires exact agreement; v4 receives credit for any member of its slash-separated group. Recomputed on the available 583 windows: v3 exact agreement **382/583 = 65.5%**; v4 group agreement **408/583 = 70.0%**; v4 exact-string agreement **330/583 = 56.6%**. Neither interpretation alone settles genuinely ambiguous pictures, but the reported gain is not an equal comparison.
    - **Difficult cases disappear:** 117 `mixed`/`none` judgments are excluded; v4 gives a named state to 115 of them.
    - **Whole windows are represented by their midpoint:** 53 evaluated named-label windows cross their selected bed’s bounds.
    - **Calibration uses its fitting data:** lines 110–116 display training-set agreement. The current fitting selection contains 454 windows but only 339 distinct beds.
    - **Agreement is counted twice:** it enters the logistic model, then multiplies its output at line 72. That product has no demonstrated probability interpretation.
    - **Family IDs are training-cohort information:** a new family receives an unseen all-zero encoding, without a validated fallback.

    The output grid repeats the midpoint problem: **11,126/83,324 windows—13.35%—cross their assigned bed bounds**, and 1,076 trailing rows are omitted by the fixed-grid loop. See [conf_data_v4.py:77–84](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:77).

12. **These CSVs do not meet the submission contract.**
    [conf_data_v4.py:73–76](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:73) emits pixel positions, label, confidence, and state. It does not emit calibrated depths or cluster IDs, and includes blank rows. Slash-created labels are also not necessarily legend entries.

    This is acceptable as a review export, but not as the required interval submission described in [SPEC.md:25–32](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/SPEC.md:25). An explicit submission adapter is still needed for this path.

## B. Threshold-overfit risks and self-calibrating replacements

Here, **LC** means `bakeoff/labeltool/legend_classify.py`; **BED** means `bakeoff/labeltool/beds.py`.

| Constants and source | What they represent | Per-log replacement |
|---|---|---|
| `90`, `235`, `fill_dark=140`; LC:68–75, 93–101, 269–271; BED:53–59 | Scanner-dependent ink, paper, and fill intensity | Estimate paper colour and dark-ink distributions jointly from legend and column; normalize shading. Use a complete pixel mixture model, including dark fills. |
| `frame=.5`, `.6`, paper `.97`, retained width `.6`; LC:52–63, 264; BED:42 | Persistent rails, outside paper, acceptable crop width | Find narrow vertical components over long spans, with width relative to measured stroke thickness. Estimate the track once; preserve broad dark or white lithologies. |
| `blank=.85`, ink `.012`, fill `.05`; LC:95, 271 | Occupancy required to distinguish rock from empty paper | Learn each swatch’s occupancy distribution under cropping and phase changes. A sparse rock must be allowed to look sparse. Compare against an explicit paper model. |
| `row_dE=14`, BED `dE=14`; LC:112, BED:84 | Colour-change magnitude | Estimate within-fill Lab variation and between-legend separation. Use a noise-normalized change statistic; preserve differences that separate two actual legend classes. |
| `.2`, `.3`, `.35`, `.6`, `.05`; LC:101–108; BED:58, 67, 83–88 | Fraction of rows/pixels supporting colour, and the colour-versus-monochrome decision | Use continuous support weights and uncertainty. Avoid switching the whole boundary algorithm abruptly when coloured coverage crosses 5%. |
| `ink_halves=.06`, BED ink `.12`, paper `.35`, monochrome multipliers; LC:116; BED:87–88 | Changes in pattern or paper occupancy | Estimate ordinary variation over one or more measured motif periods. Set change thresholds against that null distribution rather than a fixed density jump. |
| `half=12/24`, smoothing `3`, `min_gap=8`, `thin=10`, `chunk=48`, minimum `6`; BED:26, 68, 90, 106, 141 | Comparison support, resolvable thickness, and texture sample length | Estimate stroke width and repeat pitch from legend/column autocorrelation. Use them for visual support. Use calibrated **depth per pixel**, rather than DPI alone, for geological thickness priors. Preserve a thin-bed component. |
| Two-pixel legend trim, minimum dimensions `3/4/12`; LC:128–132, 181–186 | Border thickness and minimum usable motif area | Detect the frame and measure retained motif coverage. The parser already trims swatches at `harness/legend_v2.py:208–209`; avoid another unconditional trim. |
| `TEX` offsets, colour `/10`, `alpha=1`, `orient=2`, `gamma=1`; LC:29–31, 144–148 | Relative importance of colour, density, dimensions, orientation, and correlation | Estimate within-class versus between-class variability using augmented swatches, then fit a small regularized pair scorer. Report uncertainty where the swatch supplies too few repeats. |
| `scales=()`; LC:31, 224 | Assumption that legend and column motifs have equal drawing scale | Estimate their pitch ratio and test a small local scale bank. DPI measures acquisition scale; it does not guarantee equal drawing scale within a page. Existing scale-cache values are ignored under the default empty setting. |
| `twin_dE=4`, ink `.03`, `twin_ncc=.75`; LC:207–208 | Supposed visual indistinguishability | Require near-identity after registration, including distinguishing components, plus pairwise agreement throughout the group. Otherwise retain separate classes. |
| `unknown=99`, unused `margin_weak=.15`; LC:31, 289–294 | Absolute match adequacy and ambiguity | Calibrate rejection from held-out positive augmentations and hard negatives; include absolute distance and crop quality. The default effectively forces a winner whenever a legend survives. |

Two caveats matter:

- `row_dE`, `row_n`, and `ink_halves` do **not** protect v4 bed chunks: `label_bed()` passes `check=False` at BED:108.
- The “scale-free” descriptors are not fully scale-free. Pixel-valued stabilizers, `orient`’s `+1`, finite-window FFT periods, and fixed chunk heights all change with resolution or sample extent. See LC:44–49, 80–83, 147.

## C. Ranked improvement plan

**What breaks first on an unseen log, in likely order:**

1. **Cache coverage:** an uncached log gets no legend at all.
2. **Legend parsing and label availability:** new layouts, compound names, and technical sections are mishandled before matching starts.
3. **Photometry and crop geometry:** dark fills, sparse patterns, rails, and unfamiliar track widths trigger destructive preprocessing.
4. **Legend-to-column scale and motif differences:** especially monochrome patterns and small distinguishing symbols.
5. **Boundary and merge assumptions:** same-colour transitions, thin interbeds, and blank gaps are suppressed.
6. **Confidence:** family-based calibration can conceal the preceding failures rather than diagnose them.

These rankings are judgments based on the failures above, not measured probabilities on unseen logs.

### Priorities

Costs are engineering estimates; expected gains are qualitative until tested on untouched real logs.

| Rank | Work | Expected gain | Cost | Main risk |
|---:|---|---|---:|---|
| 1 | Repair cropping, invalid states, merge evidence, cache misses/invalidation, and submission export | High confidence of eliminating demonstrated failures | 3–5 h | Changing cropping changes all downstream features; regenerate outputs |
| 2 | Freeze an independent human-reviewed sample by well/template; use identical label rules for comparisons | High gain in trustworthy model selection | 2–4 h plus annotation | Small sample still misses rare styles |
| 3 | Integrate the better existing legend parser, then calibrate photometry and motif scale per log | Potentially largest improvement on unfamiliar legends | 3–6 h | Parser regressions; insufficient swatch support |
| 4 | Replace singleton votes with a decoder using full legend costs, blank/unknown states, and soft duration priors | High potential for thin beds, gaps, and consistent boundaries | 4–8 h | Excessive smoothing or wrong emissions can dominate |
| 5 | Adapt the existing compact boundary model using realistic synthetic contacts and a small real set | Moderate potential; cheap inference | 3–6 h | Learning grid lines or synthetic seams |
| 6 | Time-box a frozen-DINO patch-descriptor matcher; add a small learned projection only if useful | Uncertain; possibly useful on difficult motifs | 2–4 h experiment, 4–8 h extension | Natural-image features may still miss small texture differences |
| 7 | OCR geological descriptions as an auxiliary likelihood | Potentially high on selected ambiguous intervals | 3–6 h for one layout | Text spans and sampled lithology may not align exactly with interpreted beds |

**Why these are different from previous attempts:**

- **Legend parsing is an existing opportunity.** The current classifier builds from `legend_v2`. The later parser report records non-eval in-domain usable-name accuracy improving from **58.27% to 97.64%**, though that is development evidence, not an independent test. Reuse and validate those changes before inventing another parser. [ml/legend_v4/REPORT.md:7–31](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/legend_v4/REPORT.md:7)

- **Viterbi/DP has already been tried.** S04 clustered first, then assigned cluster names; it confused monochrome textures. S18 already used a joint duration-constrained decoder and improved boundaries, but its historical in-domain composite score was below S15’s. Its selected extra boundary weight and penalty were both zero. A new decoder should therefore start from the **current per-legend emission costs**, preserve blank states, and use calibrated motif/depth scales. Test soft duration priors against no prior; do not impose sedimentary transitions as hard rules. [notes/s04_kmeans_viterbi.md:21](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/s04_kmeans_viterbi.md:21), [ml/integrate/REPORT.md:104](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/integrate/REPORT.md:104)

- **DINO nearest-neighbour matching is not new here.** S05 used repeated/resized tiles, CLS embeddings, clustering, and a fixed colour weight; its report identifies shale-to-silty-sandstone confusion. The useful experiment is **local patch tokens**, several motif-preserving crops, masking of rails/paper, and within-legend hard-negative training for a small projection. Keep colour and fine ink evidence alongside the embedding. Use the cached small backbone frozen, batch inference, and apply it initially only to difficult windows. [s05_dinov2.py:27–45](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/strategies/s05_dinov2.py:27), [notes/s05_dinov2.md:5–7](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/s05_dinov2.md:5)

- **More synthetic volume alone is unsupported.** S07 trained per log on repeated swatches and suffered a real-column domain gap. S17 improved its historical in-domain result but reduced external-test composite score. S21 already generated 10,000 windows; its held synthetic D-582 boundary F1 remained **0.3846**. Train a reusable metric on **real approved column appearances paired with their legends**, preserving distinct motifs and varying measured scan properties. Hold out wells and duplicate artwork, not merely filenames. [notes/s07_synth_cnn.md:3–7](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/s07_synth_cnn.md:3), [ml/synth_gulf/REPORT.md:54](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_gulf/REPORT.md:54), [ml/synth_v2/REPORT.md:85–95](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/REPORT.md:85)

- **Boundary learning is already cheap enough.** S13/S22 use small dilated 1-D CNNs. S22’s added scan augmentations reduced development F1, so indiscriminately increasing distortion is a failed direction. Improve contact realism, distinguish grids from contacts, and include actual thin beds. [ml/boundary_v2/REPORT.md:19–29](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/boundary_v2/REPORT.md:19)

- **Description OCR has concrete support on at least one log.** I inspected `60811405120000_1inch tvd mudlog.tif`, rows 955–2155. The right-hand **GEOLOGICAL** column contains readable `SHALE:` and `SANDSTONE:` descriptions. Use OCR as a soft, vertically aligned constraint; first verify whether each paragraph describes the corresponding interpreted interval. This track has no supplied bounding box in that log’s annotation.

For the owner’s hill-climbing loop, reserve the LLM judgments for development. Select on human-reviewed whole-log or whole-template holdouts, and report row accuracy, boundary F1, blank false positives, abstention coverage, and runtime separately. Repeated optimization against the same confident LLM subset measures adaptation to that reader and sample.

### Efficiency findings

I profiled three 6,000-row crops on this machine, with cached legends and OCR bypassed. Times include profiling overhead and are not full cold-start benchmarks.

| Crop | Read | Segmentation | Major costs |
|---|---:|---:|---|
| D-582 log06, width 238 | 0.27 s | 1.62 s | Classification 1.44 s; NCC 0.62 s |
| Gulf `60811405120000`, width 47 | 0.24 s | 2.47 s | Classification 1.55 s; row profiles 0.84 s |
| Rio Nueces 212, width 68 | 0.40 s | 4.50 s | Classification 3.58 s; NCC 2.04 s |

The best immediate optimizations are:

- **Cache mask statistics and template statistics.** NCC repeatedly recomputes window means, template means, and template standard deviations for each label. Texas made 6,532 NCC calls. [LC:121–134](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:121)
- **Use inexpensive colour/texture costs to shortlist candidates**, then NCC or DINO on close competitors. Check shortlist recall before deployment.
- **Batch row statistics and avoid repeated masks/casts.** Per-row Python medians materially dominate colour-column preprocessing. [BED:51–62](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py:51)
- **Stream long columns with overlap.** The loader reads strips, but `read_column()` still materializes the full column, followed by more full-column arrays.
- **Remove avoidable quadratic work:** candidate suppression checks every accepted boundary; `bed_at()` scans beds for every grid window. Use indexed suppression and `searchsorted`/a moving interval pointer. [BED:94–96](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py:94), [conf_data_v4.py:27](/C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:27)
- **Measure cold preprocessing separately.** A missing native cache parses every log, while `Context.get()` also invokes legend parsing. The warm timings above exclude this potentially substantial cost.

## D. What I could not verify

- No independent unseen-log accuracy or improvement estimate: the proposed gains remain hypotheses.
- No fresh v4 blind review was available; `analysis/categories/llmrev4/` was absent.
- Artifacts changed during the review: the bed directory grew from 47 to 51 files. The final `conf.json` still reported 15,323 named beds versus 15,189 in the available bed files. Historical headline scores are therefore not fully reproducible from a consistent frozen snapshot.
- I did not refit confidence: importing scikit-learn triggered a SciPy temporary-file attempt, which the read-only guard blocked before writing.
- I did not run fresh OCR, GPU inference/training, or a cold full-dataset benchmark.
- I visually verified the Allegheny distinguishing symbols and one geological-description track, not every merged pair or every log layout.
