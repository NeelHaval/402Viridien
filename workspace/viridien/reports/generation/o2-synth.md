# o2-synth: end-to-end synthetic data creation (cross-cutting design critique)

## 1. Verdict
- **Hadrian (Geoservices) only:** fit for purpose there. Synthetic layer rate is 199–235 per 10k rows; real human labels give 216 per 10k. The sheet looks right.
- **Every other family:** not yet fit. Neither a detector trained on these columns nor a test score on them would transfer. Outside Hadrian, the model learns from two biased sources:
  - fixed-size owner example boxes (96/103/106/115/124 rows), which say nothing about bed thickness;
  - machine runs, which on Hadrian are 2.15x too thick and find only 0.56x as many layers. They are used uncorrected.
- **Single biggest problem:** the statistics measure how the data was labelled, not the geology. Sparse example boxes become thickness, composition and "gap" statistics. Units are scan rows of logs drawn at 1, 2 and 5 inches per 100 ft, and positions are fractions of whatever depth range was scanned.

## 2. Findings (ranked)

### F1. Outside Hadrian, owner boxes are fixed-size examples, but the stats treat them as beds — blocker
**Evidence**
- In `data/normalized/current.json`, the most common box heights per document are box-tool sizes, not beds:
  - 663e1eb0: 96 rows, 12 of 15 boxes;
  - 7f061d08: 96 rows, 12 of 16;
  - 1bac8dda: 103 rows, 7 of 12;
  - 952caeadf: 106 rows, 6 of 12;
  - 28112eef: 115 rows;
  - d368: 124 rows.
- Only Hadrian has real bed geometry: 682 of 1,028 boxes, 133 distinct heights.
- In stats.json, every Texas thickness entry is censored (stats-report "Censored" column: shale 31/31, limestone 11/11). The Texas p95 is ≤124 for every class, which is the box size.
- MC765 has one box per document, so the composition is a single class (`doc:4a056c8ff25cf749` composition = {clay-s: 1.0}).
- The synthetic MC765 columns are 194–216 layers, all clay-s and all exactly 96 rows (`out/starter/column-mc765-*/truth.json`: max 96).
- Texas `doc:5fbeaf2a0650e013` has a known-row share of 0.023, yet its composition is labelled "medium" confidence. That composition is the owner's choice of which legend entries to box.

**Fix**
- Add a `box_kind` to records: `example` (a clean patch) or `bed` (full boundaries). It could also be inferred: height equal to the document's mode box height, plus no abutting neighbour, means example.
- Example boxes feed only "class is present in this log" and the pattern appearance. They never feed thickness, transitions or composition.
- Grade composition confidence on known-row share, not record count.

**Verify:** re-run stats. The MC765 and Texas thickness blocks should become empty or low confidence. Then compose must back off, or refuse those documents.

**Effort:** M. **Gain:** removes the fake 96-row checkerboard columns and the legend-uniform compositions.

### F2. Machine runs make up 99.4% of Gulf evidence, are biased, and are used uncorrected — blocker
**Evidence**
- Gulf family rows: 8,879 human against 1,569,207 machine (stats-report).
- The Hadrian calibration (stats.json `calibration.raw`):
  - layer_count_ratio 0.565;
  - median_thickness_ratio 2.15;
  - boundary recall 0.347.
- The selected smoothing threshold is 0, so 0 rows were changed. No inverse correction is applied (stats-report "Only bracketed short flips are corrected").
- Machine thickness is quantised to prefill window sizes. For example, Gulf clay quantiles are 16/32/32/32/32, and marl is 12/12/24/36.
- Window sizes vary by log (1bac8dda modes are 20 and 168 rows), so each log has a different minimum bed.
- Mean rows per layer across Gulf training documents runs from 65 (3515004b) to 2,967 (663e1eb0).
- Synthetic Gulf columns contain 7.5–9.5 layers per 10k rows (`starter/column-gulf-3-panel-1bac8dda*`). The real Hadrian-family rate is 216.
- Look at `out/starter/sheet-1bac8dda58a39eaf.png`: the synthetic column is one red class for 1,200 rows.

**Fix**
- Model the machine output as a noisy emission of the hidden true lithology. Use a hidden semi-Markov model (HSMM): states are legend classes, emissions follow the confusion matrix measured on fully labelled stretches, and durations come from a per-family duration model.
- Fit it with EM on each machine-labelled log, holding the emission matrix fixed. This replaces "machine weight 0.35" (which only down-weights a biased estimate; it does not de-bias it).
- Until then, at least:
  - rescale machine-derived thickness by the measured Hadrian ratio and mark it `calibrated-from-one-log`;
  - floor nothing at the window size;
  - treat runs no longer than one window as lower-bound censored.

**Verify:** on Hadrian, fit the HSMM to the machine guesses alone. It should recover the human layer rate (216/10k) within about 20% and the median thickness of 23 rows. Today the machine guesses give 380 layers against 673, and a 49.5-row median.

**Effort:** L (HSMM) or S (rescale). **Gain:** realistic bed rhythm for 12 of the 32 training logs.

### F3. Units: rows mix 1, 2 and 5 inch per 100 ft scans, and zones are fractions of the scanned depth range — major
**Evidence**
- Rows per foot, worked out from the log names (depth range divided by column rows):
  - Gulf 57b1857d "1in", 8921–15721 ft over 16,558 rows: 2.4;
  - Gulf 952caeadf / aea797 / aeb414 / 17a317 "5in": 10.0–10.1;
  - Hadrian "2in": 4.0 (it also has a depth tick every 50 ft at about 200 px; see `o2-synth/page-band40.png`);
  - MC765 1-inch MD: 2.0; 5-inch MD: 10.2, over the same 3686–20707 ft.
- Family thickness pools therefore mix 5x scales.
- `thickness_frac` (fraction of column height) does not fix this. Column heights span 999–307,092 rows (Gulf) because depth ranges differ: MC765 BP01 covers 2,920 ft, ST00 covers 17,021 ft.
- "fifth-1" of one log is 3,686–7,090 ft; of another it is 18,230–18,814 ft. Pooled zone statistics compare unrelated stratigraphy.
- Compose falls back to family thickness often: in MC761, 83–93 of about 220 draws; in Texas, 141–206 of 170–246 draws (`truth.json stats_level_used`).

**Fix**
- Add `rows_per_ft` and `depth_top_ft` per document. Take them from the name first, then from OCR of the depth track: `bounding_boxes.depth_column` exists in `bbox_annotation.json`.
- Store thickness in feet. Define zones in true depth (absolute ft bands), or drop positional zones entirely (see F6).
- Render at the target document's `rows_per_ft`, and add scale augmentation: render the same sequence at 1, 2 and 5 inch.

**Verify:** for each family, the KS distance between thickness of 1-inch and 5-inch documents should shrink after converting to feet.

**Effort:** M. **Gain:** valid pooling, plus synthetic scale variation the real 51 logs need.

### F4. Classes without a pattern are deleted, and their neighbours fuse into invisible "boundaries" — major
**Evidence**
- MC761 97adb9d2 has a drawable share of 0.413, and its shale has no pattern (`skipped_transition_states` lists `family:MC761 mineral/lithology/shale`).
- `marginalize_missing` (`compose.py:182-206`) projects A→shale→A into A→A.
- Truth then holds 116–118 same-class adjacent layer pairs per column. In MC765 it is 192–215 per column, and on the Texas page 270.
- Using the Hadrian-calibrated boundary detector on the synthetic columns: only 41.7 of 98.3 truth boundaries per 10k are class changes in MC761, and 0 of 107.5 in MC765.
- Visually (`o2-synth/view-column-mc761-*.png`, real left, synthetic right): the real column is mostly grey shale with thin orange and yellow streaks. The synthetic column is all orange sandstone and thick yellow blocks. Mean RGB is real (133,131,101) against synthetic (213,157,33).

**Fix**
- Merge equal adjacent layers before writing truth.
- Refuse a document, or pick another, when its drawable known-row share is below about 0.8. Today it proceeds at 0.094 (b1411b67) and 0.151 (9da74bcd).
- Never marginalise out a class with a composition share above about 10%. Use the `family-substitute` pattern or skip the document.

**Verify:** count same-class adjacent pairs; it should be 0. Synthetic and real mean RGB in the same span should be within about 15 per channel.

**Effort:** S. **Gain:** the boundary F1 in every suite stops penalising boundaries that cannot be seen.

### F5. "Gap" rows show real lithology pixels but carry mask 0, covering up to 65% of a column — major
**Evidence**
- `compose.py:504-513` fills `__gap__` with a 2,048-row real crop (`noise.py:544-590`), tiled with a random phase, and leaves `mask` at 0. `class_index` maps 0 to `__gap__`.
- The Texas gap source (`noise/gap-5fbeaf2a0650e013.png`, viewed as `o2-synth/gap-texas.png`) is real blue limestone with brown beds.
- Gap row share per synthetic column: Texas 0.27–0.51, One-off 0.27–0.52, MC765 up to 0.65.
- The gap rate comes from unlabelled stretches between example boxes: Texas 49,751 of 50,928 rows are unknown, with gap heights p50 of 1,130 rows. That measures labelling sparsity, not anything printed on the log.
- A segmenter that treats 0 as background will learn real limestone as background. The bench scorer counts rows predicted inside gaps as errors.

**Fix**
- Write the mask with a distinct ignore index (255), and document that gaps are "real, unlabelled".
- Base the gap rate only on fully labelled stretches; truly blank intervals are rare.
- Do not tile one 2,048-row crop; it makes visible repetition.

**Verify:** a segmenter's loss excludes 255. The share of gap rows in synthetic columns should fall to the blank-row rate of real columns (≈0 in the stretches measured; `real_blank_rows` = 0.0 in `o2-synth/visbound2.json`).

**Effort:** S. **Gain:** removes contradictory supervision from about half the non-Hadrian rows.

### F6. The Markov model is over-parameterised: order 2 by zone loses to order 1 even on the best log; sections never run — major
**Evidence**
- On Hadrian human chains (30 contiguous chains, 673 layers), split by depth (`o2-synth/hadrian_model.py`), held-out bits per transition:
  - order 0: 2.154;
  - order 1: 2.061;
  - order 2: 2.179 (worse than order 0).
- Compose nevertheless takes `order2_by_zone` for 395 of 399 Hadrian draws, which multiplies parameters by 5.
- `Sampler.zone` (`compose.py:168-173`) returns the first matching zone. The fifths cover [0,1] and are listed first, so `section-*` zones are never used: every starter `truth.json` has only `fifth-*` zones.
- Within a chain, thickness has a lag-1 log correlation of −0.14 (n=643). Thickness barely depends on the previous class (sandstone median 16 after either neighbour).
- A lognormal fits 2 of 3 major classes (KS p = 0.65 and 0.20). It fails for the thin-bed class (p = 5e-5; 53% of its beds are ≤16 rows), which is bimodal.

**Fix**
- Default to order 1 with zone as a mixture over latent "sections". Each section is a regime with its own composition and transitions, with a duration in feet. That is a two-level HSMM, which also lets thin interbedded packages (alternating 15-row beds) appear as a regime.
- Choose the order by held-out log-likelihood per family, not by count thresholds.
- Use the empirical or a two-component thickness model for classes where KS fails.

**Effort:** M. **Gain:** fewer spurious statistics and a visible "packages" rhythm, which today's i.i.d. thicknesses lack.

### F7. Texas and MC765 columns are i.i.d. draws from a composition — major
**Evidence**
- `stats_level_used` for Texas holds only `composition:fifth-*` keys (for example 117 + 123 draws). For MC765 it holds 72 + 75 + 46.
- The cause: their transitions are "insufficient adjacent evidence", so `compose.py:251-258` falls back to the composition.
- In `out/starter/sheet-5fbeaf2a0650e013.png`, the real column alternates white/dash shale and blue limestone in 12–60-row beds. Synthetic A shows thick yellow sandstone and brown blocks that do not appear in that real span.

**Fix:** with no transition evidence, back off to the family or global order-1 chain over the classes the document has. If that is also absent, refuse the document. Never draw i.i.d.

**Effort:** S.

### F8. The synthetic data cannot represent half the test set — major (for the transfer question)
**Evidence**
- Held-out documents by family:
  - D-582 1-bit grid: 9 of 19 test documents, with 0 training documents (stats-report "D-582 can have zero training statistics");
  - Gulf: 2;
  - Geoservices: 2;
  - unknown family: 6.
- D-582 test columns are 246–297 px wide. The widest training column is 95 px.
- Compose renders only at the base document's native width and colour. It has no 1-bit binarisation and no width or scale variation.

**Fix**
- Add an appearance-transfer mode: render a training document's sequence in the style of a family that has no training data (1-bit threshold, 250-px width, that family's rails). This uses only page-level style measured from training-side material; take nothing from test labels.
- Report the bench per family. Mark D-582 as "no synthetic coverage" rather than averaging it in.

**Effort:** M. **Gain:** honest coverage, and a chance at the largest test family.

### F9. Page mode is sound for column detection but builds a page that contradicts itself — minor/major depending on the detector
**Evidence**
- `o2-synth/page-band40.png` (Hadrian page, band at 20,480):
  - the real cuttings-percent track (left), the depth track, the mud box and the curves are kept;
  - the synthetic interpreted column (x 525–582) has independent sand streaks that do not line up with the real cuttings sand at about 14,800 ft.
- Only 163 overlay pixels crossing the column were re-applied over 48k rows.
- The bbox annotation is kept, and the legend is real and consistent with the drawn classes.

**Fix**
- Keep page mode, but add a `context_consistent` variant: condition the synthetic sequence on the real cuttings track. Sample the interpreted class from the dominant cuttings class per depth, so the page stays geologically coherent.
- Flag in `truth.json` that the description and cuttings tracks are not consistent with the column. Any model using page context (or a VLM) must not train on the inconsistent variant.

**Effort:** M.

### F10. The bench does not yet prove anything about transfer — major
**Evidence**
- Validation set (`bench/runs/starter/fidelity.json`): 5 wells and 56 real layers, all example boxes, so the comparison side is censored too. It has 0 real transitions (`order1_js_bits n_real 0`). Composition total variation (TV) fails in every zone (0.68–0.99).
- The TEST side has the same shape: 112 layers and 0 transitions.
- Realism scores 0.569 balanced accuracy, from a logistic regression on colour, texture and gradient histograms of 64×128 windows. A weak discriminator at chance does not show the windows are indistinguishable. The dataset work already reached 63% with augmentation. Windows that small also cannot see the column-level failures in F1–F7 (wrong class mix, 96-row checkerboard, missing shale).
- Detector utility is not in `report.html` yet (unfinished). The suites are scored only with perfect predictions (`suites/*/scores/*-perfect-k5.json`), which checks the scorer, not realism.
- The `typical` suite composes doc b4a8617d, which has 0 statistics layers.

**Fix**
- The real proof is a detector-transfer test:
  1. train the same detector (a) on real training wells and (b) on real training wells plus synthetic data;
  2. evaluate both on fully labelled real stretches of held-out wells, per family.
- To make that possible, first get 2–3 contiguous, fully labelled stretches (about 3k rows each) per family from the owner. Do it on validation wells, not only Hadrian.
- Upgrade the discriminator to a small CNN or k-nearest-neighbour model on raw pixels at whole-column height (for example 64×1,024 windows) so it can see rhythm.
- Add one column-level statistic that needs no labels: the visible-boundary rate from a detector calibrated on Hadrian (F1 0.94 there; `o2-synth/visbound2.py`). It is valid only within the families it was calibrated on.

**Effort:** M–L.

### F11. Ground truth does not match the downstream task — minor
**Evidence**
- Truth is a per-row mask and row intervals in the base scan. The real task is depth intervals (ft) mapped to a legend entry, from scans at several scales.
- There is no depth mapping, no legend description and no mixed or percentage rows (the cuttings track).
- Column mode carries no legend, so legend-to-column matching cannot be tested there.

**Fix**
- Add `depth_top_ft` and `depth_bottom_ft` per layer (once `rows_per_ft` exists, see F3), plus `legend_id`, `legend_label_printed` and the legend swatch box.
- Score suites in feet as well as rows.
- In column mode, emit a small synthetic legend strip, drawn from the same models, for legend-matching tasks.

**Effort:** S–M.

### F12. Keeping the stats honest as labels grow — minor
**Evidence**
- `--watch` rebuilds stats every 2 minutes. Suites freeze `generation-stats.json` with its sha256, which is good.
- Bench validation wells are chosen by a hash and then excluded per run (`bench/integration.md`), but family pools already include them ("holdout_caveat").

**Fix**
- Fix 2–3 validation wells per family once and exclude them from all generation levels, family and global included.
- Report every metric against the label snapshot id.
- Require a minimum of fully labelled (`bed`-kind) rows per family before its statistics are marked anything above "low".

**Effort:** S.

## 3. Quick wins (each under an hour)
1. Merge equal adjacent layers in truth, and refuse documents with drawable share below 0.8 (`compose.py:328`, `Sampler.__init__`).
2. Write mask 255 for `__gap__` and document it. Stop tiling a single gap crop (`compose.py:504-513`).
3. Fix `Sampler.zone` so that sections are used deliberately, or remove them (`compose.py:168-173`).
4. Parse `rows_per_ft` from document names (1in ≈ 2.0–2.4, 2in = 4.0, 5in ≈ 10 rows/ft as measured) and store it in stats.json. Stop pooling family thickness across scales until it is in feet.
5. Grade composition and thickness confidence on known-row share and the uncensored layer count. Texas and MC765 then become "low", and compose skips them.

## 4. What is already good
- Generation statistics come from training documents only. Censoring is recorded. Backoff is explicit. Compose refuses unrelated patterns and records `pattern_source`. Suites freeze their statistics with a hash.
- Hadrian end to end is convincing: the layer rate, the sheet, and page mode with real rails and other tracks streamed in bands.
- The stats report is candid about its limits, including the Hadrian machine distortion table.

## 5. Doubts
- The visual boundary detector (`o2-synth/visbound2.py`) is calibrated on Hadrian only. It over-fires on other families' textures, so I used it only for the class-change counts and the Hadrian comparison.
- I did not verify that the Gulf salt sections are genuinely thick. Some Gulf intervals in the real sheet are long single-class runs, so part of F2 may be geology rather than bias. The Hadrian ratios are the only calibration available.
- I did not check `noise.py` blur, noise and overlay fits in depth. Bench detector utility had not run.
- The rows-per-ft figures come from log names (depth range ÷ column rows), not from OCR of the depth ticks. They are checked against the Hadrian 50-ft ticks only.

Scratch work: `data/genlab/critique/o2-synth/` contains `visbound.py`, `visbound2.py`, `hadrian_model.py`, `visbound2.json`, the `view-*.png` and `pair-*.png` image pairs, `page-band40.png` and `gap-texas.png`.
