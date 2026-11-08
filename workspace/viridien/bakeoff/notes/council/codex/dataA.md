## A. Per-log problem table

**The largest risks are incorrect legend identities, false “twin” merges, and inconsistent output versions.** I ran `C.legend(log)` and `C.twins(log)` for all **51 logs** with `PYTHONPATH=bakeoff`, using the existing cache and blocking writes. I also sampled 16 column windows per log and visually inspected selected source legends and column crops.

The audit found **813 legend entries**, ranging from **2 to 47** per log; **30 twin groups across 22 logs**; **43 processed swatches with a dimension below 8 px**; and **28 swatches with completely empty black-ink masks**. Empty masks do **not** necessarily mean empty images: several contain colored marks that the feature extractor misses.

Evidence below comes from [native swatches](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/native_swatches.pkl), [bed JSONs](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/beds), [CSV exports](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/beds_csv), and the source TIFFs/boxes. Log names are abbreviated by distinctive identifiers; shared defects are grouped explicitly.

| Log | Problem | Evidence | Fix |
|---|---|---|---|
| **D-582 log01–log09** | OCR errors and destructive swatch trimming; frame contamination | Each has 10 entries, including `Anhydrlte` and `SIllty Sandstone`. Coal becomes **103×4 px**, entirely black. Seven logs retain persistent black columns several pixels inside the supplied box in sampled windows. | Preserve printed text separately from canonical names. Detect physical frame rails rather than repeatedly stripping dark columns; protect solid-fill classes. |
| **Callaway 3-33** | Legend parser combines multiple rocks into single classes | Only **5 entries**. Examples: `CEMENT CHALK`, `SALT-PEPPERSAND GRANITE SANDSTONE GYPSUM`. Images themselves contain multiple stacked swatches. JSON assigns **28,384 px, 55.73%**, to `CEMENT CHALK`. | Re-segment individual swatch/text pairs. Existing classifications and synthetic examples using these combined classes need regeneration. |
| **Rio San Gabriel 2829** | Missing entries, truncated names, text mistaken for swatches | **14 parsed entries versus 23 visible material swatches**. Names include `:D HBHHHEHHHRH`, `:LSTONE`, `E`, `ESTONE`. Several returned pictures are visibly label text. CSV assigns **6,523 px** to `CLAYS`. | Repair cell detection and text association; reject text-only exemplars before feature extraction. |
| **Critter Creek 201 and 202** | Very shallow exemplars and lost texture information | All **23 native crops per log are only 9 px high**. In 201, both `SILTSTONE` and `SANDSTONE` have **zero black-mask ink**, despite visible dots. | Preserve native color/contrast information and acquire additional verified column exemplars. The inspected source legend really is shallow: this is not evidence that every crop is misplaced. |
| **Hadrian 2 — 60808400140000** | Technical symbol and OCR fragment retained as rock classes | **18 entries**, including `MDT` and `I!S`. The MDT picture is visibly an operational symbol. | Enforce source-panel/type classification and a reviewed vocabulary; the current negative-word filter misses MDT. |
| **OCS-G16987 — 60812400810000** | Duplicate name, incomplete name, fluid labels in exports | **12 entries but 11 distinct names**: two `Medium sand` entries and `S`. CSV contains **8,200 px Dry**, **117 Gas_Good**, **67 Oil_Poor**. Blank share: **53.26% JSON versus 3.95% CSV**. | Repair name association and rebuild exports from a versioned, filtered legend. |
| **DC141 — 60811404100000** | Two false twin groups | `Clay-shale / Marl`, NCC **0.836**; `Argillaceous limesto / Bituminous limestone`, **0.824**. Visible tick/cross distinctions survive in the original images. | Preserve these classes. Similar background and repeating structure are insufficient evidence of identity. |
| **60811405800000, 60811405800100, 60811405800500, 60818400600100** | Limestone and dolomitic limestone merged | All four merge `Dolomitic limestone / Limestone`. In the inspected example, NCC is **0.911**, but dolomitic marks are visibly present in only one picture. | Compare discriminating marks, not predominantly shared brickwork. |
| **MC822 bp01 — 60817410140100** | Sandy/silty shale merged | `Sandy clay-shale / Silty clay-shale`, NCC **0.763**; visibly different dot/dash structure. CSV additionally contains **28 px Oil_Good**. | Undo the merge and remove operational classes from the export vocabulary. |
| **MC822 original — 60817410140000** | Fluid label in CSV | **102 px Oil_Moderate**, absent from the current filtered rock set. | Regenerate and validate export labels against the same legend version used for inference. |
| **Allegheny — 608114075800** | Four false twin groups; inconsistent blank handling | Merges include `Mudstone / Packstone / Wackstone` and `Boundstone / Grainstone`, although their printed letters differ. JSON blank share **41.11%**, CSV **0%**; **8,852 rows** differ. | Preserve letter-bearing symbols; regenerate synchronized outputs. |
| **MC776-6 — 60817412840000** | False limestone merge and major artifact disagreement | `Argill limestone / Bituminous limestone`, NCC **0.854**, despite dash/cross differences. **49,401 px** blank in JSON are Limestone in CSV. | Repair equivalence rules and reconcile artifact provenance before using these labels as training data. |
| **All four MC765 logs** | Distinct clay patterns collapsed | Each has 23 entries but merges `Clay-M / Clay-S / Clay-SACS` and `Clay-CCS / Clay-SCS`. Native pictures show different ticks, dots and dash arrangements. | Maintain separate legend IDs; reserve ambiguity groups for demonstrably indistinguishable symbols. |
| **FREEMAN 3 24 ST** | Colored symbols become a false perfect match | Blue `MARL-CALC` and magenta `MARL-DOLO` both produce zero ink masks: **NCC=1, color-feature distance=0**. Their merged label covers **11,029 px, 79.16%**. | Extract colored ink relative to background. Never interpret two empty feature masks as proof of identical pictures. |
| **OCS-G26419 — 60812400480000** | Non-rock output and blank disagreement | **31,091 px Salted water** in CSV; blank share **16.13% JSON versus 0.29% CSV**. | Re-export after filtering; explicitly score false rock predictions on blank/show regions. |
| **MC540 — 60817411420000** | Operational entries remain in CSV | **26 px Gas_Good**, **14 px Oil_Good**. Sampled box includes persistent rails at offsets 3–4 px. | Clean export vocabulary and separate rail masking from lithology evidence. |
| **MC776-5 — 60817411700000** | Large fluid-labelled stretches | CSV contains **42,926 px Oil_Moderate + 26 px Oil_Good**. Blank share **58.87% JSON versus 13.21% CSV**. | Quarantine these predictions from synthetic mining until source regions are adjudicated. |
| **G08037 — 60818400600100** | Huge blank-label discrepancy | Blank share **69.77% JSON versus 26.00% CSV**; **28,902 rows** differ. An inspected crop contains dense green horizontal hatching, not blank paper. | Check feature rejection of dense hatches and regenerate both artifacts together. |
| **Ardennes 60811406050100; related 60811406050000** | Segmentation outliers requiring review | **3,442 / 849 intervals**; Ardennes median interval **20 px**, maximum **57,350 px**. | Inspect stratified thin-bed and giant-bed samples against depth grids. Counts alone do not establish that these beds are wrong. |
| **XOM JU103; MDN ST01BP01 — 60811405800300** | Extremely long predicted beds | Maximum lengths **101,680 px** and **64,136 px**, respectively; the latter log has only **27 intervals**. | Verify both endpoints and interior samples. Thick salt is plausible; do not impose an arbitrary maximum thickness. |
| **Trail Mtn1Palisades** | Grid interference remains a test case, but the suspected failure was not confirmed | **199 intervals**, **78.28% IGNEOUS**, no blank intervals. Source crops at y=5000 and 30000 visibly contain Tuff and Igneous patterns, matching predictions. No examined 48-row window had >90% white interior. | Add grid-only negative examples and verify boundaries. The dominant Igneous share is not, by itself, evidence of grid hallucination. |
| **Shell AC0336** | White legend entry omitted | **7 returned entries**, while the source also shows a white `Others-V` cell. | Establish its intended semantics and represent it explicitly if it is a valid target; white classes cannot simply be treated as empty paper. |
| **Shell 08852** | Low entry count is legitimate | Only **2 rock entries**, both visible in the source legend; the remaining entries are operational symbols. | No count-based repair warranted. |

Three dataset-wide findings matter:

- **The outputs are inconsistent.** At the final comparison, both directories contained 51 files, with **15,992 JSON intervals versus 15,991 CSV intervals**. Labels differed on **214,222 rows across 23 logs—5.35% of total column length**. CSVs contained **82,597 rows of non-rock labels** excluded by the current `C.legend` set. Export code directly copies bed labels, so this disagreement indicates different output states/versions, not an intended conversion. See [export code](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:65).
- **Twin merging materially changes the dataset.** Current JSONs assign **389,393 rows** to combined twin labels; only **2,331** are the known D-582 Clay/Shale group. The other **387,062 rows** inherit the questionable merges above. No two differently named raw arrays were byte-identical; that strict check does not establish visual distinctness by itself.
- **Boxes need interior masks, not merely tighter rectangles.** In 16×128-row samples per log, **9/51** contained persistent black columns at least three pixels inside the box. In **42/51**, the current trimming rule removed >20% of width in at least one sample. That latter figure includes genuine blank regions and is a warning about trimming behavior, not a count of incorrect boxes. [Trimming and ink extraction](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:52).

## B. Cross-log label/family issues

The family mapping covers all 51 logs. From [LLM review results](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/llmrev/results.json), there are **770 judgments but only 700 unique windows**; **583** nonduplicate windows have a named-rock answer used in the v4 comparison.

| Family | Logs | Unique judged windows | Named-rock comparison windows | Tuning limitation |
|---|---:|---:|---:|---|
| D-582 | 9 | 147 | 135 | Same well/template lineage; nine files are not nine independent wells. |
| Geoservices | 5 | 67 | 57 | Small sample for a large, fine-grained vocabulary. |
| Gulf | 18 | 267 | 221 | Largest group, but related scans and sparse class coverage remain. |
| MC761 | 3 | 20 | 15 | Far too small for family-specific tuning or a convincing 100% claim. |
| MC765 | 4 | 65 | 57 | Same well and byte-identical cached legend sets; false merges simplify the task. |
| One-off | 5 | 44 | 32 | Heterogeneous layouts; pooling them does not create a coherent training family. |
| Texas | 7 | 90 | 66 | Includes severely broken Callaway/Rio San Gabriel legends. |

All families have judged windows, but **coverage is not independence**. Cached legend sets are byte-identical across all nine D-582 logs, all four MC765 logs, and Rio Lavaca plus both Rio Nueces logs.

Names need three distinct fields: **printed label, stable local legend ID, and reviewed canonical concept**.

- Formatting variants include `Clay/Shale`, `Clay-shale`, `Clay Shale`, `ClayShale`; `SHALEGRAY` versus `SHALE GRAY`; and `Coarsesandstone` versus `Coarse sandstone`.
- OCR errors include `Anhydrlte`, `SIllty Sandstone` and the Rio fragments. Some apparent errors—`Silstone`, `Oolic limestone`, truncated labels—are reported as source spellings in [the parser audit](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/legfix.md). Preserve their provenance.
- Do not globally equate Clay, Shale, Marl, or all sandstone subtypes. A visual equivalence valid in one legend is not a geological synonym.
- The scorer currently rewards any member of a combined v4 label, while v3 uses exact equality. Thus false merges can increase measured agreement without improving identification. [Comparison logic](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:49).
- [label_shares_per_log.csv](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/analysis/label_shares_per_log.csv) is useful for family membership, but its shares are not current truth: Callaway’s `CEMENT CHALK` share is **4.0% there versus 55.73% in current bed JSON**. Do not fit synthetic prevalence directly from these inconsistent prediction summaries.

## C. Ground-truth audit

Coverage below is the union of annotated **full-image pixel rows**, clipped to each lithology box. The denominator is **4,000,628 column rows across 51 scans**, not physical footage or independent wells.

| Source | Nonempty logs | Intervals | Covered rows | Entire dataset coverage |
|---|---:|---:|---:|---:|
| `bakeoff/gt/gold` | 1 | 17 | 6,006 | **0.1501%** |
| `bakeoff/gt/human` | 1 | 3 | 92 | **0.0023%** |
| `bakeoff/gt/silver/A` | 12 of 13 files | 783 | 71,184 | **1.7793%** |
| `bakeoff/gt/silver/B` | 12 of 13 files | 732 | 71,161 | **1.7787%** |
| `neel_dataset/human_labels/truth` | 2 | 646 | 62,005 | **1.5499%** |
| `neel_labels/2026-11-07/.../human` | 0 of 3 files | 0 | 0 | **0%** |

**Gold + human + curated Neel comprise 666 intervals on four logs, covering only 1.7023% of total column length.** These sources have no out-of-box, reversed, or overlapping CSV intervals in the checks performed.

Within their individual columns:

- Gold log06: **88.44%** coverage.
- Human log02: **1.35%**.
- Neel `60811405120000`: **96.47%**, 357 beds.
- Neel Shenzi: **70.19%**, 289 beds.

The 13th silver file, Neptune, is empty in both sets. Their notes say its evaluation window is blank; that is useful negative evidence, but contributes no annotated lithology rows.

**Agreement is uneven and sometimes undermined by vocabulary errors.** These are overlap-length-weighted agreements after case/whitespace normalization and the existing Sand-and-Sandstone alias:

| Comparison | Shared rows | Agreement | Interpretation |
|---|---:|---:|---|
| Silver A vs B, overall | 71,138 | **50.28%** | **57.33%** when both labels are known. |
| A vs B, D-582 log06 | 6,020 | **12.92%** | Strong interpretation/abstention disagreement. |
| A vs B, Critter 201 | 6,000 | **11.53%** | **192 intervals versus 20**; serious boundary and identity disagreement. |
| A vs B, Allegheny | 6,000 | **6.15%** | A uses `Clay/Shale`, absent from this legend. |
| A vs B, Rio Nueces 212 | 5,997 | **0%** | A imports MC765-style `Clay-S`/`Limestone-L` names. |
| A vs B, Shell 08852 | 5,125 | **17.07%** | Becomes **99.94%** after punctuation normalization: mostly a naming defect. |
| Gold vs A | 6,006 | **74.21%** | “High confidence” silver is not interchangeable with gold. |
| Gold vs B | 6,006 | **32.33%** | On B’s 1,957 non-Unknown overlapping rows, agreement is **99.23%**; abstention explains most of the low overall score. |
| Curated Neel vs A / B, 1inch log | 5,995 | **99.05% / 99.27%** | Strong local agreement, not validation of the other logs. |

Silver A contains **11,608 rows with labels absent from the respective current legends**: 5,608 Allegheny rows and all 6,000 Rio Nueces rows. Silver B contains **8,769 Unknown rows**. Keep Unknown separate from both “blank” and “wrong rock.”

The three root human events are active labels, but [labels.jsonl](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/gt/human/labels.jsonl) records that model suggestions were shown. The older Neel archive has three label events followed by three undos, explaining its empty CSVs.

**Neel snapshots must not be counted as independent annotations.**

- `neel_labels/2026-11-08/labels`: **223 Shenzi intervals**, 42,232 covered rows, **all without depth values**.
- `neel_dataset/snap2`: **356 + 288 intervals**. Shenzi has one overlapping interval and no depths.
- Curated data: **357 + 289 intervals**, with repaired names, removed show symbols, inserted thin beds and depths.
- Raw Shenzi snapshot agreement with curated labels is only **43.77%**, mostly because **26,960 rows** change from the erroneous name `Halite = Shale = Clay` to `Halite`. This is chiefly a schema/name repair, not wholesale reinterpretation.
- The extra DC141 snapshot contains just **two Unknown intervals covering 342 rows**.

All **16 files listed in the curated Neel hash manifest matched their recorded hashes**. Its [README](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/neel_dataset/human_labels/README.md) documents human annotation plus AI review, remaining gaps, and five approximately balanced side-by-side beds whose dominant label is ambiguous.

**Units are internally consistent where populated, but not interchangeable.**

- Fitted ft/px slopes: gold **0.0499477**, root human **0.05**, Neel 1inch **0.4991673**, Shenzi **0.3332909**.
- Curated Neel feet/metres differ from exact conversion by at most **0.061 m**, consistent with rounding.
- Silver files contain **no depth values**.
- CSV exports in `beds_csv` contain pixel coordinates, not the harness’s depth-bearing interval schema.
- MD and TVD variants and different drawing scales require separate calibration. Internal consistency does not independently validate the printed depth anchors.

## D. Synthetic data plan and transfer check

**The generator already implements several requested effects. The problem is their fidelity, distribution and provenance—not universal absence.**

| Effect | Current implementation | Remaining gap |
|---|---|---|
| Blur/JPEG/1-bit | `synth_v2` has blur, JPEG and Bayer dithering. `synth/nuisances.py` has JPEG, noise and fixed-threshold binarization. | Main nuisance pipeline lacks explicit scan blur; thresholds and degradation ranges are largely hand-set. No calibrated mixture of scan/recompression histories. |
| Skew | Both use horizontal shear. | No true page rotation or vertical/local deformation with transformed truth. |
| Stroke width/printed versus drawn | Procedural rendering varies stroke geometry, roughness and measured spacing. | Limited motif grammar; some topology is selected from rock-name substrings, which can erase class distinctions. Hand-drawn variation is not independently calibrated. |
| Legend/column mismatch | Real-crop generation can retain donor mismatch; v2 scales tiles by 0.85–1.15. | Same source art remains heavily reused; page acquisition changes do not consistently affect legend and column together. |
| Bed thickness | Main generator uses clipped lognormal distributions plus 6–30-row thin beds. V2 samples 48–349-row beds, except truncated tails. | Real thin beds and extreme tails are badly underrepresented. |
| Boundaries/grids | Both add rules; procedural mode measures contact/grid statistics. | Need independent grid/contact geometry, colored/broken rules and reliable thin-bed truth under boundary ink. |
| Annotations | Both overlay text/symbols. | V2 uses legend OCR text; realistic depth ticks, handwriting and geological annotation placement are only approximated. |
| Blank regions | Both generate gaps. | Main nuisance pipeline repaints gaps pure white **after** degradation, removing realistic noise/rules/annotations from them. |

Relevant implementations: [main generation](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/generate.py:53), [procedural motifs](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/procgen.py:20), [nuisances](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/nuisances.py:26), [v2 generation](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/generator.py:121).

The quantitative problems are substantial:

- [approved.json](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/data/approved.json) has only **6 nonempty pattern groups, 50 crops, 9 source logs, and 2 families**. The 472 entries in texture statistics are not 472 human-approved patterns.
- V2 has **10,000 windows: 8,000 training, 2,000 held**, sourced from **20 training logs and 7 held logs**. Every family has only **one held source log**.
- D-582 and MC765 have **identical legend-image hashes across training and held sources**. Filename separation does not test new artwork.
- **1,137 windows**—990 training and 147 held—contain non-rock truth labels excluded by current `C.legend`. They occupy **295,179 of 9,420,844 filled synthetic rows: 3.13%**.
- In curated real truth, **305/357 beds (85.4%)** on the 1inch log and **197/289 (68.2%)** on Shenzi are shorter than 48 px. V2’s ordinary thickness sampler cannot represent them. Shenzi also has a **14,647-px** bed, far beyond v2’s 1,024-row training windows.

A concrete next generation cycle should:

1. **Freeze a corrected source manifest.** Review local legend IDs, swatch geometry, class type and ambiguity groups. Exclude Callaway/Rio garbage and fluid labels until repaired. Version/hash legends, boxes, annotations and generated outputs together.
2. **Split by well and scan/artwork lineage before mining or fitting statistics.** Keep every MD/TVD, rescaled and chromatograph sibling together. Permit a test log’s legend at inference, as the task requires, but never its column crops in training.
3. **Generate paired legend/column appearances.** Render the same underlying symbol with independently measured scale, color and stroke differences, then apply shared page degradation plus local damage. Preserve dot/tick/letter distinctions. Mix reviewed real exemplars with procedural variants.
4. **Fit sequences and thicknesses to reviewed real data.** Include 1–6 px beds, ordinary beds, long homogeneous stretches, same-label contacts and mixed-width intervals. Generate long columns before taking training windows so giant beds and boundary-free windows occur naturally.
5. **Render acquisition and nuisance layers together.** Include frames, independent depth grids, ticks, colored ink, text, handwriting, dropout, blur, resampling, JPEG and several binarization processes. Transform segmentation masks and coordinates with geometric changes. Retain nuisance ink inside blank regions.
6. **Audit generated distributions before training.** Compare real/synthetic thickness CDFs, native widths, ink/color distributions, motif spacing, grid periods, annotation coverage and rejection rates by family. Keep deliberately balanced training data separate from prevalence-weighted evaluation data.

**Transfer check.** The existing results already demonstrate a substantial gap:

| Saved selection result | Synthetic | Real development | Gap |
|---|---:|---:|---:|
| Row accuracy | 73.20% | 42.18% | **31.02 points** |
| Boundary F1 | 81.58% | 49.90% | **31.68 points** |
| Composite score | 62.16% | 20.63% | **41.53 points** |

These are computed from [selection.json](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/selection.json). They are selection results on different source distributions, and the selected boundary model mixes real and synthetic training; they are **not a clean synthetic-only transfer experiment**.

For a convincing test:

- Freeze a synthetic-only model and evaluate it on untouched, independently human-labelled real wells, alongside the current legend matcher. Separately compare real-only training against real-plus-synthetic training using identical real training data.
- Aim for **at least three independent test wells per family**, with complete contiguous labelled windows plus blank/annotation negatives. The present corpus cannot provide that independence for D-582 or MC765; additional wells are needed.
- Report per-family and per-class accuracy, boundary performance stratified by bed thickness, blank false-positive rate, abstention, and depth error. Bootstrap **wells**, not crops.
- Predeclare acceptance criteria—for example, positive real-data improvement with uncertainty excluding zero, no material family regression, and a synthetic–real accuracy gap below 10 points on comparable distributions. These are proposed engineering gates, not guarantees of transfer.
- Run a separate test with fresh legend parsing. Copied parser and depth caches currently make the main synthetic benchmark conditional on those inputs; it cannot validate their extraction. [Documented limitation](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/README.md:454).

## E. What I could not verify

- I did not rerun OCR or rebuild parser caches: those paths write files. The legend audit therefore tests the **cached inputs actually consumed by `C.legend`**.
- All logs received numerical checks and sampled column inspection, but I did not visually adjudicate every swatch, bed or annotation. Giant beds and unusual bed counts remain flagged candidates, not proven errors.
- I verified annotation geometry, overlaps, units and recorded hashes; I did not independently establish geological correctness or reread every depth tick.
- The workspace changed during the audit: bed JSON count increased from **47 to 51**, and some results changed. Final aggregate comparisons use the later read, but this was not an atomic, immutable snapshot.
- No completed fresh `llmrev4` review was available when inspected. Existing judged windows have already influenced tuning.
- I did not retrain or execute a new transfer benchmark. The gap reported above comes from saved selection results.
- No files were written, no `.env` files opened, and no network/API calls made.