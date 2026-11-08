The main blockers are damaged legend vocabularies, false “twin” merges, stale CSVs, and insufficient independent truth to establish synthetic-to-real transfer.

**(A) Per-log problem table**

I ran `C.legend(log)` and `C.twins(log)` for **all 51 logs**, with `PYTHONPATH=bakeoff`. They returned **813 swatches and 200 distinct label strings**. Counts ranged from **2 to 47 entries**.

Evidence below comes from `bakeoff/analysis/categories/native_swatches.pkl`, `beds/<log>.json`, `beds_csv/<log>.csv`, and selected native TIFF crops. Filename prefixes identify logs; grouped rows enumerate repeated defects.

| Log | Problem | Evidence | Fix |
|---|---|---|---|
| D-582 **log01–log09** | OCR errors; destructive trimming; residual column frames | Each has 10 entries, including `LImestone`, `Anhydrlte`, `SIllty Sandstone`. On log06, Coal shrinks from **107×148 to 103×4 px**. A 192-row column crop retains **8 nearly solid frame columns after trimming**. | Correct names against the printed legend; distinguish solid-fill rocks from frames; remove rails using column geometry. |
| `60808400140000` — Hadrian | Missing/misnamed rock; non-rock retained | 18 entries: printed **Clay is missing**, Silt becomes `I!S`, and the **MDT tool symbol** survives filtering. | Reparse the complete legend; classify entry roles explicitly. |
| `60811404100000` — GC743 DC141 | False twin merges | `Clay-shale / Marl`, NCC **0.836**; `Argillaceous limesto / Bituminous limestone`, **0.824**. The pictures visibly differ. | Remove these equivalences; compare complete motifs. |
| `60811405800000`, `60811405800100`, `60811405800500`, `60818400600100` | Limestone variants falsely merged | Each has 15 entries; `Limestone / Dolomitic limestone` merges at NCC **0.911**, despite additional dolomitic marks. | Preserve both classes and their distinguishing marks. |
| `60811405800300` | Giant-bed review flag | Only **27 CSV intervals** across 118,668 rows; one Halite interval spans **64,136 rows**, **54.1%** of the box. | Verify against the scan and depth scale before treating this as a segmentation failure. |
| `60811406050000`, `60811406050100` — Ardennes | Degenerate swatches; extreme segmentation scale | Clay-shale becomes **39×4 px**. The latter log has **3,442 intervals**, including a **57,350-row Halite bed**. | Repair trimming; inspect both thin contacts and large salt intervals at native scale. |
| `60811407300100`, `60811407570000` — GC743 DC222/DC802 | OCR inconsistencies; solid swatch trimming | 30 entries each; examples include `Coarsesandstone`, `Oolic limestone`, `Mediumsand`. Coal becomes **41×4 px**. | Preserve printed names plus verified aliases; retain solid-fill geometry. |
| Allegheny `608114075800…` | Four false twin groups; stale blank handling | Merges include Mudstone/Packstone/Wackstone and Claystone/Silty Clay. CSV labels **y=2737–11438**, **8,701 rows**, as Salt; current JSON marks it blank. | Unmerge distinct rocks and regenerate CSVs from the current JSON snapshot. |
| `60812400480000` — OCS-G 26419 | Fluid labels in CSV; damaged Shale reference | CSV contains **31,091 rows of `Salted water`**, **15.8%** of the column. Shale becomes **33×4 px**. Current JSON contains no invalid labels. | Refresh exports after filtering; repair solid-dark swatches. |
| `60812400810000` — OCS-G 16987 | Truncated name; stale fluid labels | Printed Silt becomes `S`. CSV contains **8,384 rows** labelled Dry/Oil/Gas, **50.6%** of the column. | Repair Silt; regenerate the CSV. |
| `60812401020000` — JU103 | Degenerate references; giant-bed flag | Slaty shale and Bitumen become **39×4 px**. Halite spans **101,680 rows**, **49.1%** of the column. | Repair references; verify the salt interval rather than imposing an arbitrary thickness cap. |
| `60817410140000`, `60817410140100` — MC822 | False merge and stale fluid labels | BP01 merges Sandy/Silty clay-shale at NCC **0.763**. CSVs retain **102 and 28 fluid-labelled rows**, respectively. | Separate the patterns; regenerate exports. |
| `60817411420000` — MC540 | OCR/trimming defects; stale labels | `Silstone`; Shale becomes **33×4 px**; CSV retains **40 Oil/Gas rows**. | Correct aliases and trimming; refresh CSV. |
| `60817411700000` — MC776-5 | Major stale fluid classification | CSV contains **42,952 fluid-labelled rows**, including a 42,900-row Oil_Moderate interval. | Replace stale exports with current filtered results. |
| `60817412840000` — MC776-6 | False merge; destroyed Marl swatch | Argillaceous/Bituminous limestone merge at NCC **0.854**; Marl becomes **38×4 px**. | Restore the full references and distinct identities. |
| Callaway 3-33 | Severe legend parsing failure | Only **5 entries**, containing concatenations such as `CEMENT CHALK` and `SALT-PEPPERSAND GRANITE SANDSTONE GYPSUM`. JSON has **782 beds**; **55.7%** of column rows receive `CEMENT CHALK`. | Re-segment individual swatches and their adjacent captions before further classification. |
| Critter Creek **201 and 202** | Tiny references; unreliable existing truth | 23 plausible entries each, but typical references are only **9 rows high**. JSON has **175/343 beds**. For 201, silver A/B give **192 versus 20 intervals** and only **11.5% row-label agreement**. | Verify full swatch extents and independently adjudicate 201’s truth. Gross current legend-crop corruption was not established. |
| All four **MC765** logs | Distinct clay patterns falsely merged | Each has 23 entries. Two groups merge Clay-S/M/SACS and Clay-CCS/SCS; NCC ranges **0.752–0.939**. Distinguishing ticks and dashes are visible. | Preserve the variants; review claims of perfect family accuracy after removing permissive equivalences. |
| Mudlog-Trail Mtn1Palisades | Grid-only paper classified as rock | **199 beds, 0% JSON blank**. CSV calls **1424–2159 IGNEOUS**. The upper crop visibly contains grey grid lines; fresh classification also assigns rock. | Add grid-on-paper negatives and detect column occupancy separately from rock texture. |
| FREEMAN 3 24 ST | Coloured patterns falsely merged | 47 entries. Blue `MARL-CALC` and pink `MARL-DOLO` produce **dE=0, NCC=1** and merge; the merged label covers about **79%** of the column. | Measure coloured ink and palette structure instead of relying on median fill and black-ink masks. |
| Rio San Gabriel | Truncated names and text crops | 14 entries include `:D HBHHHEHHHRH`, `E`, `:LSTONE`, `ESTONE`. Some extracted pictures contain caption text; two are only **5 rows high**. | Re-detect swatches and bind each to its complete caption. |

There were **43 references with a dimension below 8 px across 35 logs**. Some, especially small Coal symbols, are legitimately narrow; the destructive examples above are separate from that. **No processed reference exceeded 98% white paper.** A white-share test therefore misses several real parsing failures.

`C.twins()` creates **30 groups across 22 logs**: nine expected D-582 Clay/Shale groups and **21 additional suspect groups across 13 logs**. No differently named raw swatches were byte-identical. The approximate matching criterion is accepting shared motif fragments as equivalence. Relevant implementation: [legend classification and twin detection](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:192).

The remaining 12 logs had no specific defect established by these checks: GB506 (**12 entries**), 1inch TVD (**10**), Shenzi (**15**), three MC761 logs (**11 each**), Neptune (**6**), Rio Lavaca and both Rio Nueces logs (**23 each**), Shell 08852 (**2**), and Shell AC0336 (**7**). Shell 08852’s two-rock legend is legitimate.

The **final stable snapshot** contained 51 JSONs and 51 CSVs, but **23 JSON/CSV pairs disagreed**. CSVs contained **15,991 intervals**, including **82,597 rows in 27 intervals with labels outside the current legend sets**—**2.06% of total column length**. Current JSONs contained **zero** such invalid labels. This is an export consistency problem, not evidence that the current filter still accepts those fluid labels.

**(B) Cross-log label/family issues**

The review contains **770 presentations of 700 unique crops**. The recorded v4 comparison uses **583 nonduplicate windows with named LLM answers**.

| Family | Logs | Judged windows | Coverage limitation |
|---|---:|---:|---|
| Gulf 3-panel | 18 | 221 | Broadest coverage, but 66 answers are Halite; several classes have only 1–4 examples. |
| D-582 | 9 | 135 | Same named well; 102 answers are Clay/Shale, versus only 3 Sandstone examples. |
| Texas | 7 | 66 | Only 1 answer is SANDSTONE; compound OCR labels are also accepted as answers. |
| Geoservices | 5 | 57 | Five legend vocabularies; only 6–17 judged windows per log. |
| One-off | 5 | 32 | Five different layouts, only 3–12 windows each. Too small for layout-specific tuning. |
| MC765 | 4 | 57 | 52 Clay-S, 3 Sand-FSS, 2 Clay-M; related well variants. |
| MC761 | 3 | 15 | **All 15 answers are Shale.** This does not validate its other ten legend classes. |

Sources: `analysis/label_shares_per_log.csv`, `bakeoff/analysis/categories/llmrev/results.json`, and `compare_v3_v4.json`.

Names need three separate treatments:

- **Verified spelling/format aliases:** `Silstone`/Siltstone, `Anhydrlte`/Anhydrite, `Coarsesandstone`/Coarse sandstone, and spacing/hyphen variants of Clay-shale.
- **Log-specific code dictionaries:** Clay-S, Sand-FSS and Others-H need explicit meanings from their own legends.
- **Distinct geology or appearances:** Claystone, Shale, Marl, sandy clay-shale and silty clay-shale must retain their distinctions. OCS-G 16987 legitimately prints **Medium sand twice with different pictures**; their mutual NCC is only **0.053**. Keep multiple visual prototypes under that printed name.

The share CSV is a **heuristic analysis output, not prevalence truth**. For example, its unweighted MC761 average is **64.6% blank**, whereas current bed CSV blank shares are **0.6–3.2%**. Using these shares directly to allocate synthetic data imports detector behaviour into the training distribution.

Perfect family scores are particularly misleading here: v4 accepts an LLM answer when it is **any member of a merged label**, and confidence is fitted on the same review data used for comparison. See [comparison logic](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:40).

**(C) Ground-truth audit**

Coverage below is the union of annotated **native pixel rows**, divided by **4,000,628 total column rows** across the 51 logs. It is not a percentage of independent wells or physical depth.

| Source | Nonempty logs / files | Intervals | Covered rows | Dataset coverage |
|---|---:|---:|---:|---:|
| `bakeoff/gt/gold` | 1 / 1 | 17 | 6,006 | 0.1501% |
| `bakeoff/gt/human` | 1 / 1 CSV | 3 | 92 | 0.0023% |
| `bakeoff/gt/silver/A` | 12 / 13 | 783 | 71,184 | 1.7793% |
| `bakeoff/gt/silver/B` | 12 / 13 | 732 | 71,161 | 1.7787% |
| `neel_labels/.../gt/human` | 0 / 3 | 0 | 0 | 0% |
| Neel 7 November label JSONs | 2 / 18 | 184 | 23,004 | 0.5750% |
| Neel archived Shenzi JSON, 8 November | 1 / 1 | 223 | 42,232 | 1.0556% |
| Neel `fixed` 1inch JSON | 1 / 1 | 356 | 13,336 | 0.3333% |
| `neel_dataset/human_labels` | 2 / 2 | 646 | 62,005 | 1.5499% |

These sources overlap and include snapshots; **do not sum the table**. Gold, the main human CSV and Neel’s packaged truth together cover **68,103 rows, 1.70%**, across four logs. The main human CSV’s three intervals are adjacent Clay labels: effectively one 92-row stretch, with model suggestions shown during labelling.

Neel’s packaged labels are substantially more usable:

- 1inch: **357 intervals**, covering **96.47%** of its column.
- Shenzi: **289 intervals**, covering **70.19%**.
- JSON labels and exported truth CSVs agree on **all 62,005 covered rows**.
- All **16 checksums** in `HUMAN_TAG.json` matched.
- Ten evaluation windows contain **59,971 scored rows out of 60,000**; the remainder are annotation gaps.

Its provenance remains **human labels plus AI review/adjudication**, with known equal-width mixed-rock ambiguities. It is not a fresh blind test after use in development. See [packaged dataset documentation](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/neel_dataset/human_labels/README.md).

Overlap agreement exposes several distinct problems:

| Comparison | Row agreement | Interpretation |
|---|---:|---|
| Silver A versus B | **40,015/71,138 = 56.25%** | Only **64.13%** even where both provide known labels. |
| Gold versus A | **74.21%** | A omits the Siltstone class on this log. |
| Gold versus B | **32.33% overall**, **99.23% where B names a rock** | B largely abstains on horizontal-hatch intervals. |
| Critter 201, A versus B | **11.53%** | Major disagreement in both segmentation and naming. |
| Allegheny, A versus B | **6.15%** | Dominant labels differ: Clay/Shale versus Claystone. |
| Rio Nueces 212, A versus B | **0%** | A uses foreign names Clay-S/Limestone-L; B uses SHALE/LIMESTONE. |
| Packaged 1inch versus A / B | **99.05% / 99.27%** | Strong agreement over 5,995 overlapping labelled rows. |

Normalisation here folds case/spacing and aliases Sand and Sandstone to Sandstone; it does not collapse distinct rocks.

Archived Shenzi demonstrates why versioned aliases matter: agreement with the package is **35.19% before** the documented `Halite = Shale = Clay` → Halite correction, and **99.04% after** it. Most apparent disagreement is a name repair, with smaller actual edits remaining.

Depth units appear consistent in the packaged data:

- Gold: approximately **20 px/ft**.
- Neel 1inch: **2.0033 px/ft**, two anchors.
- Shenzi: **3.0004 px/ft**, 231 anchors; fitted maximum anchor residual **0.993 ft**.
- Stored metre fields agree with feet × 0.3048 within approximately **0.061 m**, consistent with rounding.

Silver CSVs have no depth fields. Archived Shenzi has **no calibration anchors**. Those sources cannot independently validate depth accuracy. Neptune’s empty silver CSV is documented as intentionally blank, but supplies no explicit negative intervals for testing hallucinated rocks.

**(D) Synthetic data plan and transfer check**

There are already two materially different generators.

The approved-crop pipeline currently has **50 crops, six nonempty patterns, nine source logs, and two represented families**—only **6/476 catalogue patterns, 1.26%**. One approved pattern has a single source log. Separately, **453/472 procedural texture profiles, 96.0%, are tagged `indexed_crops_unverified`**. Those measurements do not establish correct rock identity.

The `synth_v2` corpus contains 10,000 windows, split **8,000/2,000**, with **20/7 source logs and no filename overlap**. However:

- **1,137 windows contain fluid/show labels as rock truth**: **295,179 rows**, **3.13% of filled rows**.
- D-582 and MC765 have **exactly identical legend-image hashes across training and held sources**.
- Only **2.54% of synthetic intervals are under 30 px**, versus **76.75%** on Neel’s 1inch page and **60.55%** on Shenzi.

Those are concrete label, independence and thickness gaps.

| Effect | Already implemented | Missing or insufficiently validated |
|---|---|---|
| Scan blur | `synth_v2`: Gaussian blur, probability 0.35 | Main nuisance stack lacks optical blur; neither distribution is calibrated to scanner families. |
| JPEG / 1-bit | Both have JPEG; v2 has Bayer dithering; main has fixed thresholding | More threshold/dropout/photocopy variation; coherent page-wide degradation. LZW is lossless. |
| Skew | Horizontal shear in both | True rotation, local warping and transformed boundary truth. |
| Line width | Procedural measured strokes; drawn rules | Joint distributions of stroke width, scale, scanner resolution and ink spread. |
| Legend/column mismatch | Real crops preserve some mismatch; v2 scales tiles to 0.85/1/1.15 | Measured relative scale and colour variation, including roughly twofold D-582 differences. |
| Printed/drawn patterns | Quilting and procedural motifs | Complex symbols and handwriting. D-582’s procedural `"sand"` rule also maps Silty Sandstone to dots by name. |
| Bed thickness/contact ink | Lognormal models, thin component, rules and interbeds; v2 uses 48–349 px bulk beds | Real thin-bed prevalence, long salt beds, mixed-width beds, uncertain and irregular contacts. |
| Annotations and blank paper | Real connected-component overlays, text, grids | Real placement correlations and complete annotations. Main generation resets blank gaps to **pure white after degradation**, excluding Trail-like grid-only negatives. |

Relevant code: [main generation](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/generate.py:212), [nuisances](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/nuisances.py:129), [procedural motifs](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/procgen.py:20), [v2 generation](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/generator.py:156).

A concrete generation plan:

1. **Repair identities first.** Use verified log-local entry IDs, explicit aliases, entry roles and adjudicated equivalences. Remove fluids from existing synthetic truth.
2. **Split by well and scan lineage before mining or fitting.** Record identical legend art separately. Keep singleton patterns training-only and disclose unavailable test coverage.
3. **Measure training-source distributions.** Fit thickness in feet and native pixels, including thin beds and long salt intervals; measure relative legend/column scale, ink width, contact style, blank grids and overlay placement.
4. **Generate paired legends and columns.** Combine approved real crops with verified procedural motifs. Apply shared acquisition effects to the page, plus controlled legend/column differences. Preserve underlying rock masks beneath annotations and retain degraded grid-only blank regions.
5. **Validate the full pipeline.** Evaluate fresh legend parsing separately from an oracle-legend condition; reused parser caches currently bypass a major failure source.

There is already a warning sign in `synth_v2/selection.json`: the selected model achieves **73.20% synthetic versus 42.18% real-dev row accuracy**, a **31.02 percentage-point gap**. Boundary F1 differs by **31.68 points**. These are selection results from different domains, not an unbiased transfer experiment. The report explicitly acknowledges this: [synth_v2 report](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/REPORT.md).

For a transfer test, freeze the generator and model choices, then obtain independently labelled random real windows—including blank grids, overlays and thin beds—from untouched well/scan groups. Compare equal-budget legend tiling, quilting and procedural/mixed training. Report:

- Synthetic-minus-real row-accuracy and boundary-F1 gaps.
- Real improvement over the unaugmented baseline.
- Per-family and rare-class results, blank false positives and abstention.
- Confidence intervals resampled by well, not synthetic bed.

A concrete predeclared success gate could require a **real accuracy gain of at least 10 points with a positive confidence bound**, plus a **synthetic-to-real gap below 10 points** on adequately represented families. Families represented by one well remain descriptive evidence.

**(E) What I could not verify**

- Fresh OCR/parser performance: `C.legend()` read the existing native cache; I did not rebuild it.
- Every pixel of every column: visual checks targeted suspect legends and crops; a broader TIFF sampling process did not complete.
- Whether every giant bed is erroneous. Neel’s reviewed Shenzi includes approximately **9,000 ft of salt**, so thickness alone cannot establish failure.
- Independent correctness of every human/LLM label or untouched real-world transfer. No training or new benchmark run was performed.
- Outputs changed during the audit—from 45 to 51 bed JSONs. The final counts above came from a snapshot whose files remained unchanged while read.

All work was read-only. No files were written, `.env` files opened, or network/API calls made.