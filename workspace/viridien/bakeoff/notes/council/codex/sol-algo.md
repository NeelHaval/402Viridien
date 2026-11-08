The largest problems are **incorrect twin merges, greedy bed erasure, and caches that prevent reliable processing of unseen logs**. The reported v3→v4 agreement gain also uses different correctness rules for the two versions.

All paths below are relative to `bakeoff/`. I reviewed [legend_classify.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py), [beds.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py), and [conf_data_v4.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py), and ran read-only probes.

## A. Bugs and silent failure modes

### 1. “Twins” merge visibly different symbols

**`labeltool/legend_classify.py:191–213`**

I inspected the native swatches:

- Allegheny’s **Mudstone, Wackstone, and Packstone contain different M/W/P symbols**, but become `Mudstone / Packstone / Wackstone`.
- Its **Boundstone and Grainstone** also merge.
- In `60811405800000_608114058000_g16786_001_00_00_md5_mud_show_2`, **Dolomitic limestone has diagonal marks absent from Limestone**, but they merge.

Classifying the Dolomitic limestone swatch itself returned the combined label with **margin 0.813**. Collapsing competitors can therefore manufacture a large margin.

The union operation is also transitive although the similarity test is not. A controlled example with otherwise identical features and Lab values 10, 13, and 16 merged all three labels: adjacent distances pass `<4`, while the endpoints fail.

**Fix:** retain individual legend IDs. Require agreement over the complete symbol before declaring visual indistinguishability; represent unresolved ambiguity separately from an exact lithology prediction.

### 2. NCC can ignore the distinguishing part of a swatch

**`labeltool/legend_classify.py:128–134`**

The template is the **top-left crop** of the swatch, capped at window height/width minus two. Anything outside that crop contributes nothing.

A controlled 12×12 example returned **NCC 1.0** although the swatch differed across its bottom and right edges. Maximising over placements can likewise match common brickwork while avoiding the identifying letter or accessory marks.

**Fix:** use multiple spatially distributed templates or full-coverage alignment. A best matching fragment is insufficient evidence for twin equivalence.

### 3. Singleton runs erase genuine beds and blank intervals

**`labeltool/beds.py:129–172`**

Controlled probes of `segment()` produced:

| Input chunks, each 48 rows | Output |
|---|---|
| A, B, C; equal margins | One 144-row A bed |
| A, A, blank, A, A | One 240-row A bed, including the blank |
| blank, A | One 96-row A bed |
| A, A with margin .01; B with margin 1 | One 144-row B bed |

The final case demonstrates that the weighted vote can select a label supported by **one-third of the rows**. The baseline weight `0.5 + margin` also gives substantial influence to essentially uninformative classifications.

In saved outputs, I counted **90 named beds containing 6,336 rows of explicitly blank chunks**, and **24 named beds whose selected label covered less than half their retained evidence**.

**Fix:** preserve blank runs; retain full label costs and decode jointly. Do not remove a contact merely because either side occupies one chunk.

### 4. Thin-bed merging always chooses the previous bed and discards evidence

**`labeltool/beds.py:173–188`**

The comment promises the closer-colour or taller neighbour. The implementation always extends the previous bed and drops the absorbed bed’s chunks.

Controlled input:

```text
A: 48 rows | B: 8 rows | C: 48 rows
```

Output:

```text
A: 56 rows, chunks covering only its original 48 rows, agree=1.0
C: 48 rows
```

Across the saved 51 outputs, **1,140 beds span rows absent from their retained chunks**, totalling **78,393 rows**. These artifacts lack provenance sufficient to attribute every case to the current revision, but the current code reproduces the underlying defect.

**Fix:** preserve evidence and recompute confidence after every merge. Choose neighbours using actual appearance costs and boundary evidence; allow genuine thin beds.

### 5. Cropping can remove real lithology and then declare it blank

**`labeltool/legend_classify.py:52–63, 262–271`; `labeltool/beds.py:39–48`**

A 48×100 white patch with repeated central black dashes was cropped to width **10**, then rejected as blank because less than 60% of the original width remained.

A solid black 48×40 patch was reduced to four columns and also classified blank through this path. Conversely, the whole-column and per-window croppers use different ink-persistence thresholds, **0.6 versus 0.5**.

**Fix:** establish the track geometry once from long vertical rules. Distinguish external paper from legitimate white texture backgrounds. Avoid redefining the usable width independently for each bed.

### 6. Dark fills fall between the ink and fill definitions

**`labeltool/legend_classify.py:268–271`; `labeltool/beds.py:53–62`**

With `check=False`, uniform greys **90, 120, and 139** are classified blank: they are neither ink `<90` nor fill `>=140`.

These patches contain an obvious printed fill. This is especially dangerous after fading or grayscale conversion.

**Fix:** estimate paper, ink, and coloured/grey fill distributions jointly per log, with an explicit uncertain region.

### 7. Boundary detection cannot see some texture changes

**`labeltool/beds.py:51–97`**

Its profiles retain row colour, coloured share, paper share, and ink share—not symbol geometry.

I constructed a contact at row 50 between vertical and diagonal patterns with identical ink coverage on every row. The detector returned **no boundaries**, with maximum step approximately **9×10⁻¹⁵**. Classification may recover long runs later, but singleton suppression erases short ones.

The edge treatment also displaces contacts: colour changes at rows **5, 50, 95** in a 100-row test were detected at **13, 50, 86**.

**Fix:** add multiscale texture-change evidence and use truncated comparison bands near endpoints.

### 8. Unseen logs are absent from the native-swatch cache

**`labeltool/legend_classify.py:154–170, 174–188`**

When `native_swatches.pkl` exists, `_native()` returns it wholesale. `legend()` then uses `.get(log, [])` without parsing a missing log.

The current pickle contains **51 logs**. An unseen identifier returned an empty legend and `unknown` for a valid filled window.

Additional cache defects:

- `legend()` depends on mutable cropping parameters; `twins()` depends on mutable twin thresholds, but their cache keys omit those parameters.
- Changing `twin_dE` to `-1` after warming the cache still returned `Clay / Shale` in my probe.
- `log_scale.json` keys only by log, ignoring scoring parameters, candidate scales, image changes, and crop changes.
- `beds.main()` silently skips existing outputs after algorithm changes.
- The 90-character filename key can collide.

**References:** `legend_classify.py:173,191,220–250`; `beds.py:29–30,204–206`.

**Fix:** parse missing logs individually and fingerprint input pixels/boxes, parser version, and relevant configuration.

### 9. `NOT_ROCK` is a context-blind substring veto

**`labeltool/legend_classify.py:179–180,217`**

Verified examples:

| Incorrect rejection risk | Non-rock terms it misses |
|---|---|
| `Oil shale` | `Shows` |
| `Gas-bearing sandstone` | `Bitumen` |
| `Mud stone` | `Drilling rate` |
| `Shale, stained` | `Casing` |
| `Fluorescent limestone` | `Cavings` |

These are controlled strings, not claims that those labels occur in this dataset. The currently filtered cached labels looked like fluid/show entries; I did not establish a present false rejection.

**Fix:** classify the complete legend cell using its section heading and rock-name content. A fluid/modifier word should not veto an otherwise valid rock label.

### 10. Rejection is effectively disabled

**`labeltool/legend_classify.py:31,289–294`**

`unknown=99` permits extreme mismatches, while `margin_weak=0.15` is unused.

Seeded random RGB noise returned:

```text
state=ok
label=Sand and Sandstone
d1=7.120
d2=7.127
margin=0.001
```

`segment()` then votes on that answer because it uses `check=False`.

**Fix:** calibrate absolute mismatch and ambiguity separately. Include explicit unknown, blank, and annotation states in decoding.

### 11. Same-label merging removes drawn bed boundaries

**`labeltool/beds.py:180–188`**

Adjacent equal answers merge regardless of the detected boundary strength. This conflicts with preserving an actual drawn contact between otherwise identical lithologies (`SPEC.md:28–29`).

**Fix:** distinguish label continuity from contact existence. Retain independently supported contacts.

### 12. The v3/v4 comparison is asymmetric and omits difficult cases

**`notes/conf_data_v4.py:44–57`**

v3 requires exact equality; v4 gets credit when the reference appears anywhere inside its merged label.

Recomputed from the saved artifacts:

| Agreement rule | Correct / 583 |
|---|---:|
| v3 exact | **382 / 583 — 65.5%** |
| v4 group membership | **408 / 583 — 70.0%** |
| v4 exact | **330 / 583 — 56.6%** |

The **78 group-membership credits** exceed the apparent 26-window improvement. Some ambiguity groups may be justified, but the demonstrably incorrect groups make this comparison unsuitable evidence of improved exact labelling.

Furthermore, **117 mixed/none crops are excluded**; 115 receive a named midpoint bed. This does not prove those intervals are wrong, but the comparison does not evaluate their segmentation correctness.

### 13. Confidence measures selected-reader agreement, not general correctness

**`notes/conf_data_v4.py:34–36,56–58,72,110–116`**

- The fitting set contains **454 windows but only 339 distinct beds**.
- Only “very sure” LLM examples enter training.
- Family identity supplies a shortcut unavailable for genuinely new styles.
- `agree` enters the logistic model and is multiplied into its output again.
- Calibration is displayed on the fitting set.
- Margin and agreement omit evidence discarded during thin merging.

**Fix:** define the confidence target explicitly, fit on representative human-adjudicated examples, split by well/artwork family, and evaluate calibration out of sample. Multiplication by agreement needs separate validation.

### 14. Review windows and exports can misrepresent intervals

**`notes/conf_data_v4.py:73–84`**

The midpoint bed labels an entire 48-row crop. I counted **11,116 of 83,324 windows crossing their assigned bed**, plus **1,076 omitted tail rows**.

The CSV is also a dashboard export, not the specified submission format: it lacks depth and cluster fields and permits combined labels (`SPEC.md:26–31`). I did not verify a downstream adapter.

## B. Threshold-overfit risks and replacements

TIFF metadata across the 51 logs spans **150–400 DPI**, with one missing value. Consequently, 48 rows represent **8.13 mm at 150 DPI versus 3.05 mm at 400 DPI**. DPI alone also cannot account for different geological depth scales.

| Constants and reference | Physical quantity / assumption | Per-log replacement |
|---|---|---|
| Ink `90`, paper `235`, `fill_dark=140` — `legend_classify.py:68–70,93–97`; `beds.py:53–55` | Scanner tone, paper brightness, ink darkness | Robust paper/background estimation; Lab lightness/chroma mixtures or local thresholding, anchored by legend and column samples |
| `frame=.5/.6`, paper `.97`, retained width `.6` — `legend_classify.py:57,264`; `beds.py:42` | Vertical rule persistence and assumed track occupancy | Detect persistent narrow rules and track geometry globally; use run continuity and physical rule width |
| `blank=.85`, ink `.012`, fill `.05`, NCC ink `.005` — `legend_classify.py:95,124–127,271` | Minimum printed coverage | Learn legitimate sparse coverage from legend augmentations; model blank paper explicitly |
| Fill shares `.1/.2/.3/.6`, monochrome gate `.05` — `legend_classify.py:72–75,101–108`; `beds.py:58,67,83` | How much of a row/band must contain colour | Estimate occupancy distributions; use soft colour evidence rather than global binary gates |
| `row_dE=14`, `row_n=3`; boundary `dE=14` — `legend_classify.py:112`; `beds.py:84` | Colour change magnitude and vertical support | Compare within-symbol colour noise against between-entry separation; express support using print scale |
| `half=12`, doubled for monochrome; smoothing `3` — `beds.py:26,67–90` | Comparison bandwidth relative to motif repetition | Estimate repeat lengths through autocorrelation; combine several bandwidths instead of selecting one |
| `min_gap=8`, `thin=10`, `chunk=48`, skip `<6`, border trim `2` — `beds.py:26,102–106`; `legend_classify.py:181` | Minimum visible feature/bed size | Use DPI and stroke size for image operations; depth calibration and a **soft** duration prior for geological beds |
| `TEX` denominators, colour `/10`, texture `×4`, `orient=2`, `gamma=1` — `legend_classify.py:29–31,144–148,276` | Relative feature noise and discrimination | Normalise lengths by stroke/repeat scale; estimate feature variability from augmented legend patches; reuse learned pair scoring |
| `scales=()` — `legend_classify.py:31,224–225` | Legend and column have identical drawing scale | Estimate relative motif scale from column and legend statistics; retain several plausible scales |
| Twin `dE<4`, ink difference `.03`, NCC `.75` — `legend_classify.py:207–208` | Claimed visual equivalence | Full-symbol comparison with spatial coverage; preserve label identities when uncertain |
| Vote `.5+margin`, valid-row threshold `.5` — `beds.py:156–171` | Trust in a chunk versus bed continuity | Integrate calibrated label costs, explicit blank/unknown costs, and boundary likelihoods in a sequence decoder |

The “scale-free” descriptor is not fully scale invariant: additive pixel constants matter. Doubling one real swatch changed its orientation feature from **0.622 to 0.680**, even though the artwork was unchanged (`legend_classify.py:83,147`).

## C. Ranked improvement plan

### What breaks first on an unseen style

1. **Certain operational failure for a missing cache entry:** no legend is loaded.
2. **Legend discovery and cell ownership:** missing/misassigned names cannot be repaired by matching. Earlier external-dev diagnostics improved true-interval accuracy from **53.47% with parsed legends to 87.34% with oracle legends** (`ml/match/REPORT.md:23–30`).
3. **Paper/ink/cropping assumptions:** faint scans, sparse patterns, dark fills, and different frames.
4. **Legend–column scale or rendering mismatch:** native-pixel NCC and stroke statistics lose correspondence.
5. **Texture-only contacts, thin beds, mixtures, and annotations:** row summaries and greedy merging discard them.
6. **Confidence transfer:** the family-conditioned calibration has no demonstrated validity for the new style.

### Work priorities

Expected gains below are judgments, not measured accuracy forecasts.

| Rank | Action | Expected gain | Cost | Risk / distinction from earlier work |
|---|---|---|---:|---|
| 1 | Fix twin equivalence, evidence-preserving merges, blank handling, cache invalidation, and symmetric evaluation | High; removes reproduced failures | 3–5 h | Low implementation risk; accuracy still needs independent checking |
| 2 | Load legends per log and evaluate the existing v4 legend parser | High when entries are missing/misowned | 1–3 h | Its non-eval development check reached 97.64% exact-and-usable, but involved only seven pages; not proof of unseen-style performance (`ml/legend_v4/REPORT.md:7–22`) |
| 3 | Reuse joint DP with full legend costs, blank/unknown states, soft durations, and boundary evidence | High for singleton erasure and fragmentation | 2–4 h | Already implemented in `ml/integrate/core.py:54–105`; adapt it rather than rebuilding an HMM |
| 4 | Test existing boundary-model code and clean weights alongside adaptive texture changes | Medium–high for contacts | 2–4 h | S22 improved dev boundary F1 over S11, but has no final result; D-582 remains weak (`ml/boundary_v2/REPORT.md:19–29,43–61`) |
| 5 | Improve pair matching through synthetic metric training and measured scale variation | Medium; strongest rationale for monochrome confusions | 3–6 h | Extend existing S14/S17 pair scoring, not another per-log classifier trained on tiled swatches |
| 6 | Frozen DINOv2-small matching using spatial patch tokens, multiple scales/phases, and a small learned pair head | Uncertain; useful targeted experiment | 4–8 h | S5 already failed on similar monochrome patterns; use spatial symbol coverage instead of one CLS vector |
| 7 | OCR the adjacent description column as auxiliary evidence | Potentially high for ambiguous symbols | 3–6 h | Requires column discovery, depth alignment, abbreviations, and handling descriptions covering several beds |

**Why these are different from earlier attempts:**

- **S4 already used Viterbi.** Its cluster-to-legend naming failed despite useful boundaries. Proposed decoding uses direct legend-conditioned emissions and explicit blank/unknown states, avoiding the intermediate k-means naming bottleneck (`notes/s04_kmeans_viterbi.md:3–28`).
- **S7 already trained a synthetic CNN per log.** It suffered legend–column appearance mismatch. Proposed metric training shares a pair scorer across logs and validates on held artwork, rather than learning a closed set from repeated tiny swatches (`notes/s07_synth_cnn.md:3–7`).
- **S5 already used DINOv2.** It tiled 96-pixel crops, resized them, and used CLS embeddings. The proposed change retains spatial patch evidence and explicitly tests small distinguishing marks (`strategies/s05_dinov2.py:27–44`; `notes/s05_dinov2.md:5–7`).
- **S17 was not simply a loser.** It improved archived in-domain composite score to **0.432**, versus S15’s **0.367**, but reduced external composite score to **0.059 versus 0.073**. More synthetic data is not automatically better (`ml/synth_gulf/REPORT.md:5–12,54`).
- **Joint decoding also has tradeoffs.** S18 improved external composite performance but trailed S15 in-domain; preserving excessive drawn contacts hurt MC765 (`ml/integrate/FINAL_ANALYSIS.md:1–7`).

I visually confirmed description text beside the lithology tracks in **log06, rows 3904–5404**, and **`60811407300100…`, rows 1043–2543**, in the dataset TIFFs. OCR is therefore practical to investigate. Treat it as supporting evidence: descriptive text may summarise sampled intervals rather than exactly match each interpreted bed.

For the owner’s tuning loop, freeze a human-adjudicated, well/artwork-disjoint real holdout first. Synthetic source separation is insufficient when different logs reuse identical artwork; S17 explicitly documents that issue. Include thin beds, unknowns, mixtures, and independent grid lines, and vary legend and column appearance both jointly and separately (`ml/synth_gulf/REPORT.md:60–64`; `synth/CRITIQUE.md:148–166`).

### Efficiency priorities

My small measurements were:

- **100 fully warmed classifications:** **0.563 s under profiling**; NCC consumed **0.290 s**, features **0.153 s**.
- **6,000-row D-582 crop:** decode **0.468 s**, row profiles **0.158 s**, boundaries **0.003 s**.
- **6,000-row Ardennes crop:** decode **4.455 s**, profiles **0.069 s**, boundaries **0.004 s**.

The Ardennes TIFF is **1680×313800**, stored as one LZW strip. Cropping falls back to Pillow and loads the page (`harness/io.py:34–53`). Metadata showed **15 TIFFs with strips over 4,000 rows**.

Prioritise:

1. **Bound memory and decoding:** single-strip files require sequential decoding or preprocessing into a suitable cache. `jev2.Context` retains opened logs indefinitely, which compounds fallback allocations (`labeltool/jev2.py:106–126`).
2. **Reduce repeated NCC work:** cache template statistics, compute window statistics once, and reserve expensive matching for plausible candidates (`legend_classify.py:121–134,283–285`).
3. **Replace quadratic boundary suppression:** `beds.py:91–96` compares candidates against every accepted boundary. Saved Rio Lavaca output reports **8,045 boundaries**, implying over **32 million comparisons among accepted peaks alone**.
4. **Batch GPU work only where useful:** route ambiguous texture cases to DINOv2; benchmark total time before adopting it across every window.

## D. What I could not verify

- No independently adjudicated accuracy gain from these proposed changes; I did not rerun the full benchmark.
- The fresh `llmrev4` directory contained a manifest but **no results file** when checked.
- Saved beds lack sufficient version/configuration provenance to establish that every artifact came from the current code.
- I could not rerun the scikit-learn confidence fit: importing its SciPy dependency required temporary-file access denied by the read-only environment. Confidence sample counts were reconstructed without running the writing script.
- I did not establish description-column availability across all 51 logs, benchmark DINOv2 GPU throughput, or verify a submission adapter and final annotated-image export.