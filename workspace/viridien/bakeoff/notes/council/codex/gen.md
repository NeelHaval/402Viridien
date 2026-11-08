**D-582 is the most likely unseen-family failure. More broadly, the current results demonstrate adaptation to these 51 logs, not reliable transfer to a new legend family.** The main risks are legend extraction, scale-sensitive matching, and evaluation reuse.

**(A) Per-family overfit table**

Code references below use **C** for [legend_classify.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py), **B** for [beds.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py), and **L** for [legend_v2.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/harness/legend_v2.py). Numbers following these abbreviations are source lines.

There are no explicit family-indexed `P` or `B` dictionaries. Instead, global choices encode experience with the observed families. C:113 explicitly says its mixed-window threshold was tuned on approximately 700 LLM-judged windows; C:265 cites the v3 review; C:275 calls the blend best on the LLM evaluation. **V4 skips the mixed-window test when classifying bed chunks**, so those rejection thresholds should not all be credited with—or blamed for—v4 behaviour (B:108).

Family counts come from [label_shares_per_log.csv](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/analysis/label_shares_per_log.csv). Agreement figures below are **LLM agreement on reused windows**, calculated from [compare_v3_v4.json](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/compare_v3_v4.json), not human accuracy or holdout results.

| Family | Observed accommodation or dependency | Prediction if genuinely unseen |
|---|---|---|
| **D-582, 9 logs** | C:175–176 explicitly identifies D-582’s legend/column scale mismatch. Twin grouping accommodates indistinguishable pictures such as Clay/Shale (C:192–213, 284). B:67–88 doubles comparison width and raises paper/ink thresholds for monochrome columns, citing approximately 25-row pattern repeats. Connecting that branch specifically to D-582 is an inference from its being the sole all-1-bit family. | **Worst.** Colour cannot resolve its patterns; native-scale matching and fixed pattern-spacing assumptions must carry the task. V4 agrees on only **25/135, 18.5%**, despite these accommodations. Identical-looking legend entries also impose an information limit. |
| **Geoservices, 5 logs** | Global colour/texture/orientation/NCC weights decide among similar entries (C:29–31, 144–148, 275–276). The parser’s flattened-grid logic repairs one observed Geoservices log; this is an upstream dependency rather than a family branch in C/B. | **High risk.** Both legend interpretation and closely related patterns matter. V4 agreement is **41/57, 71.9%**. The family’s aggregate colour-collision diagnostic is 85%. |
| **Gulf, 18 logs** | Ten original parser failures prompted layout and heading repairs. The repair was revised after inspecting shortened technical rows in the Ardennes legends. C’s non-rock filtering remains another safeguard (C:217). B’s row summaries and fixed chunk voting assume that aggregating across column width preserves the desired interpretation. | **High upstream risk; moderate matching risk once parsing succeeds.** A new arrangement of lithology/modifier/show sections could remove the correct answer before matching. V4 agreement **190/221, 86.0%** benefits from extensive representation and repairs. |
| **MC761, 3 logs** | Correct exclusion of the Mineral section is essential; the audit documents excluding 13 mineral entries. The global colour/texture rules and bed voting then handle the retained 11 lithologies. I found no MC761-specific C/B branch. | **Uncertain, probably moderate.** **15/15 agreement** is too little evidence for broad transfer. A changed section arrangement could contaminate the candidate set; a new monochrome printing could expose weaknesses hidden by its present appearance. |
| **MC765, 4 logs** | Global texture/NCC weights distinguish many same-colour subtypes; colour alone is insufficient. Fixed chunk voting and merging encode expectations about bed persistence (B:129–183). No explicit MC765 C/B branch. | **Moderate-to-high under new artwork.** **57/57 agreement** tests an unusually repetitive template population. It does not establish performance on a new colour grid with different mark spacing or rare subtypes. |
| **Texas “Rock Types”, 7 logs** | The inherited parser has a dedicated wide-grid detector and compact-coal exception. C/B then use fixed border trims, minimum crop sizes and approximately 48-row chunks. These are consequential when swatches are shallow or sparse. | **High parsing risk.** A familiar-looking box can still fuse neighbouring entries. V4 agreement **60/66, 90.9%** does not show that every printed label was available. |
| **One-off, 5 logs** | No dedicated C/B branch found. All the global geometry, filtering, twin and persistence assumptions apply. This bucket includes logs with **2–47 parsed rock entries**, so it is not one coherent visual family. | **High variance and potentially catastrophic.** V4 agreement **26/32, 81.3%** averages unlike layouts and cannot predict another “one-off”. Holding out this bucket is useful stress testing, but not a clean single-style experiment. |

Sources for parser history and family diagnostics: [legfix.md:7–19](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/legfix.md:7), [legend_quality.md:8–16](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/legend_quality.md:8), [difficulty_by_family.csv](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/analysis/difficulty_by_family.csv), [legend_difficulty.csv](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/analysis/legend_difficulty.csv).

**File-level independence is particularly weak here.** I hashed the ordered native lithology-swatch pixels and dimensions in all 51 [v2 parser caches](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/out/_cache/legend_v2). All **nine D-582 logs share one exact swatch set**, and all **four MC765 logs share another**. Texas has five distinct sets across seven logs; Gulf has sixteen across eighteen. These are artwork identities, not a complete well-lineage audit.

**(B) Ranked failure modes on unseen legends**

This is a qualitative likelihood ranking supported by observed cases. The diagnostic percentages are heuristic measurements, not measured probabilities of failure.

1. **Layout and section interpretation: the correct candidate disappears first.**
   The original parser returned zero lithologies on **11/51 logs**: I counted ten Gulf and one Geoservices failure in `legfix_baseline.json`. The v2 repair recognizes particular heading combinations and distinguishes flattened grids from panels (L:98–129). Labels are principally sought to the right or below a swatch. A new list arrangement, heading vocabulary, or boxed layout can therefore break extraction before any learned matcher runs. Callaway’s cached debug image visibly merges rows; the repair report lists just five compound labels. “Parser did not crash” is not evidence of complete extraction. [Repair evidence](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/legfix.md:63), [label association](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/harness/legend.py:162).

2. **Legend-to-column scale mismatch.**
   C:4 assumes native pixels mean the same scale, while C:175 acknowledges otherwise. The deployed defaults have `scales=()`, so `log_scale()` immediately returns 1.0 (C:31, 224–225). In the saved unsupervised scale sweep, **23/51 logs preferred a non-unit scale**, including all nine D-582 logs. Those preferences are not ground-truth scale estimates, but they contradict a comfortable same-scale assumption. NCC also takes a top-left swatch fragment when the query is smaller (C:128–134). [Saved scale sweep](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/log_scale.json).

3. **Many lookalike patterns and genuine visual ambiguity.**
   Family-level colour-collision diagnostics are **100% MC765, 85% Geoservices and 73% Gulf**. MC765’s current success shows that collisions are sometimes recoverable; it does not demonstrate robustness to new artwork. Twin merging uses fixed colour, ink-density and NCC thresholds, with transitive grouping (C:202–213). A new crowded legend can either keep indistinguishable classes separate or merge distinguishable ones. **A learned model cannot recover a distinction absent from the pixels.** [Family diagnostics](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/analysis/difficulty_by_family.csv).

4. **Colour-to-1-bit shift.**
   All nine D-582 logs are mode `1`; the other 42 are RGB or palette images. Thus, current monochrome evidence comes from one artwork family. The black-and-white boundary branch assumes a particular repeat scale, while sparse symbols compete with blank detection. D-582’s **18.5% v4 agreement** is direct evidence that existing accommodations are insufficient, although it does not isolate binarization from pattern scale and ambiguity. [Per-log modes](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/analysis/legend_difficulty.csv), B:67–88.

5. **Small or heterogeneous swatch sizes.**
   Per-log median swatch-box heights span **12–120 pixels**. My cache inspection found Texas swatch interiors as small as **5 pixels in a dimension**, versus D-582 interiors around 103–107 pixels tall. Detection filters cells relative to one dominant size (L:72–85); C trims another two pixels when possible. A new legend mixing large fills with tiny symbol keys can lose valid entries or retain too little pattern evidence. [Dimension data](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/analysis/legend_difficulty.csv).

6. **Symbol-only patterns, especially sparse unboxed ones.**
   The parser contains special handling for unbordered Rock Types cells and compact coal; the visual audit explicitly records recovering these cases. Sparse marks can be rejected as empty, stripped as paper, or confused with technical symbols. The current matcher’s very permissive `unknown=99` then encourages assigning a surviving candidate. This is an extrapolation from the detection and rejection rules, not a measured symbol-only failure rate. [Parser exceptions](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/harness/legend.py:84), C:31, 52–63, 269–293.

7. **Text inside the swatch.**
   This has the weakest measured prevalence evidence. Pattern discovery removes OCR regions from its detection mask, but returned swatch pixels are rectangular crops; the classifier has no semantic text mask. Embedded letters can become distinctive “texture”, and right/below label association can miss the name. The compact-coal exception demonstrates adjacent geometry problems, but I did **not** verify how many of the 51 legends contain genuine text-in-swatch examples. [Detection and extraction](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/harness/legend.py:71), L:198–209.

A related transfer problem is **bed resolution**: `half=12`, `thin=10`, `chunk=48` and two-chunk persistence are pixel constants, not scale-normalized quantities. Changing scan resolution changes which geological beds survive smoothing and merging (B:26, 129–183).

**(C) Ranked learned-method plan**

**First establish a parser ceiling.** On the archived external-dev experiment, the learned matcher obtained **53.47%** with true intervals and parsed legends versus **87.34%** with oracle legends. That is evidence to audit candidate availability before investing in a larger encoder—not an estimate of the current v4 parser’s error. [Matching report:23–30](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/match/REPORT.md:23).

The ranking below is for hackathon return per engineering hour. Gain ranges are **planning hypotheses against current v4**, not measured improvements; regressions remain possible. Hours exclude annotation.

| Rank | Concrete proposal and what changes | What was already tried | Expected gain; hours; risk |
|---|---|---|---|
| **1. Small learned boundary detector** | Reuse the six-layer row-feature CNN implementation, train fresh within each split, and test it with the current matcher held fixed. Add real examples of thin contacts and pattern strokes that are not boundaries. Test multiple row scales explicitly. | S13 improved archived in-domain boundary F1 **0.546→0.569**. S22 selected synthetic augmentation at **0.544 dev F1**, versus **0.256** for S11, but has no final benchmark. Extra scan augmentation made S22 worse; simply adding more noise is unsupported. Its held synthetic D-582 F1 remained **0.397**. | Target **0–8 percentage points boundary F1**, **0–3 points row accuracy**. **2–4 h** integration/evaluation. **Medium risk:** learning grids or synthetic seams; limited naming benefit. CPU is already sufficient. |
| **2. Duration-aware DP / semi-Markov decoding** | Replace hard chunk flicker removal with per-label evidence and a soft duration prior. Estimate durations only from training logs, normalize by reliable scale, and retain a thin-bed component. Permit blank/unknown states. Start with class-independent duration and switching costs. | S4 already used Viterbi. S18 already implemented joint DP: in-domain F1 rose **0.549→0.668** versus S15, but composite score fell **0.367→0.347**. Selected extra boundary weight and penalty were both **zero**. S19’s fresh combination is still awaiting final evaluation. | Target **0–5 points F1**, **0–3 points row accuracy**. **2–4 h**. **Medium risk:** smoother, confidently wrong labels and erased thin beds. Strong geological transition priors are unjustified with this truth coverage. |
| **3. Legend-conditioned metric learning** | First establish the existing 27-feature logistic matcher as the cheap learned baseline. Then try a small shared CNN or projection head trained episodically: query column patch versus the current log’s candidate swatches. Use independent query/support transformations, same-colour hard negatives, and a small amount of real training-log correspondence. Treat twins as ambiguous positives. | S7 already trained a per-log CNN from tiled swatches; its archived silver/B mean row accuracy is **38.95%**. S14/S15 already learned pair distances. S17’s style synthesis improved in-domain row accuracy **53.42→63.75%**, but reduced external composite **0.073→0.059**. S21 already generated **10,000** synthetic windows and still lacks final results. | Target **0–5 points row accuracy**, potentially more on particular confusions. **6–10 h**. **High risk:** learning the renderer, repeated artwork or synthetic shortcuts. More synthetic volume alone is not the proposed difference. |
| **4. Frozen DINOv2 matching ablation** | Use local DINOv2-small, cache embeddings, compare multiscale **patch-token** descriptors rather than only the CLS token, preserve aspect ratio, and combine them with colour/stroke evidence. Hold segmentation fixed for the first test. | S5 already used frozen DINOv2-small, 96-pixel tiling, 224 resizing, CLS embeddings and colour distance. Its note reports **94.6% synthetic row accuracy**; the archived real silver/B run averages **47.16% row accuracy and 25.97% boundary F1**. This is not an untried foundation-model solution. | Target **0–3 points row accuracy**; zero gain is quite plausible. **2–3 h** for a decisive ablation. **High accuracy risk, low compute risk:** coarse tokens may miss the tiny marks that distinguish these classes. |

Evidence: [S13](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/boundary/MODEL_CARD.md:5), [S22](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/boundary_v2/REPORT.md:19), [S18](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/integrate/REPORT.md:19), [S19](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/combine/MODEL_CARD.md:3), [S17](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_gulf/REPORT.md:5), [S21](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/REPORT.md:31), [S5 implementation](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/strategies/s05_dinov2.py:27), [S5 notes](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/s05_dinov2.md:3). I computed S5/S7 real means from their respective `bakeoff/out/<strategy>/metrics.json` files.

**The GPU is not the limiting resource.** Both small and base DINOv2 weight files exist locally. The archived S5 run reports approximately **0.297 GiB peak GPU allocation** and **4.13 seconds/window mean**. Small metric heads and the existing boundary models are realistic within 12 GB; annotation quality and domain mismatch deserve most of the time budget. These archived timings do not benchmark the proposed token matcher. [S5 metrics](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/out/s05_dinov2/metrics.json).

For synthetic training, preserve separate evaluations for **swatch-rendered columns** and **real-column donor crops**. They test different gaps. The generator’s own critique identifies clean-example selection, artificial seams, repeated source pixels and warm parser caches as limitations. Outside data should supply additional independent boundary/scan examples, while scarce new human work should cover the currently missing families. [Synthetic critique](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/CRITIQUE.md:27).

**(D) The holdout experiment**

1. **Declare the historical exposure before splitting.**
   The blind-review file contains **700 nonduplicate crops covering all 51 logs**, which I counted directly. The current code has already used family-specific observations. Therefore, leave-one-family-out retraining on these logs is a useful **retrospective transfer stress test**, not a pristine prospective test. A genuinely unseen-style claim needs new logs/styles kept sealed after the next design freeze. Fresh windows from the same logs do not provide that claim. [Review data](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/llmrev/results.json).

2. **Create groups before extracting any training assets.**
   Group by well, scan lineage and overlapping depth coverage. Keep exact/near-duplicate artwork together for an additional artwork-holdout test. Apply the assignment to real crops, synthetic donors, legends used for training, overlays, fitted thickness statistics, pseudo-labels and calibration examples. File names alone are insufficient, as the D-582/MC765 hashes demonstrate.

   Run two outer evaluations:
   - **Log/well holdout:** unseen acquisition from a represented family, where independent groups exist.
   - **Seven family holdouts:** exclude an entire family from fitting and hyperparameter selection.

   Report within-family independent-well transfer as **unavailable** where the dataset lacks suitable groups; do not substitute another scan of the same well.

3. **Minimum useful human annotation pilot: 14 logs, 28 contiguous windows.**
   Aim for two independent logs per family, with two randomly located approximately 1,000-row windows per log: **28,000 annotated rows**, plus complete legend annotation for those logs. This is an exploratory minimum, not enough for narrow family-level confidence intervals. Where independent sources do not exist, obtain new sources or explicitly retain the limitation.

   Each annotation needs:
   - Swatch box, exact printed name and rock/modifier/technical status.
   - Bed tops/bottoms, label or admissible label set, blanks and uncertain/mixed regions.
   - Whether a drawn separator is a real contact or a grid line.

   Keep random windows even when they contain long uniform beds. Add a separate thin-bed/lookalike challenge sample if necessary; do not replace the random sample with interesting examples.

   Existing verified truth is concentrated: **17 gold intervals and 3 human intervals in D-582**, plus **646 Neel-labelled beds across two Gulf pages**. Neel’s labels include AI-adjudicated edits and documented ambiguous mixtures; they should retain that provenance. [Neel dataset](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/neel_dataset/human_labels/README.md:3).

4. **Separate inference conditioning from adaptation.**
   A held-out log’s legend is legitimate inference input. Report two tracks:
   - **Frozen model:** test swatches serve as references; no weight updates.
   - **Declared legend-only adaptation:** a fixed procedure generates training examples from that test legend and adapts a head.

   Neither track may use held-out column labels, human-selected test interiors, or test scores to choose augmentation, training time or rejection thresholds. Adaptation time counts toward runtime.

5. **Use nested selection and four diagnostic conditions.**
   Within each outer fold, select settings using only inner held-out wells/families. Train fresh weights; old checkpoints may already contain an excluded well. The repository explicitly documents that problem for earlier S13/S14/S17/S18 assets. [Model-history audit](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/combine/MODEL_CARD.md:30).

   Evaluate each selected method with:

   | Legend input | Boundary input | What it measures |
   |---|---|---|
   | Automatic | Predicted | End-to-end performance |
   | Human-verified | Predicted | Performance with parsing corrected |
   | Automatic | True intervals | Naming with segmentation corrected |
   | Human-verified | True intervals | Matcher and visual-ambiguity limits |

   Freeze label normalization and twin policy across methods. Report strict single-label accuracy separately from admissible-set accuracy.

6. **Score at the independent-source level.**
   Report row accuracy, per-label recall, boundary F1, thin-bed recall, blank false positives, abstention coverage and composite score. Show per-log and equal-family means, including the worst family. Bootstrap independent wells/groups—not pixels or synthetic children. Report actual independent-source counts and insufficient-data cases.

   Measure cold legend parsing plus inference, adaptation time, memory and failures. Oracle boundaries and warm parser caches belong only in diagnostic timings.

7. **Keep the hill-climbing loop inside development data.**
   The present v4 confidence fit uses **461 “very sure” LLM examples**, and its displayed calibration bins reuse the fitting set. Family one-hot features do not establish calibration for an unseen family. Fit calibration on out-of-fold human decisions, then evaluate it separately on held-out logs/families. [Confidence code:34–58, 110–117](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:34).

   Also, the reported v3→v4 comparison uses exact label equality for v3 but accepts membership in a twin group for v4. Re-score both with the same policy before attributing the **382→414 agreements out of 583** entirely to better recognition. [Comparison code:49–50](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:49).

**(E) What I could not verify**

- I did **not** run training, inference or a true family holdout. Worst-family predictions and expected gains are reasoned hypotheses.
- I could not reconstruct which individual observation selected every `P`/`B` value. Explicit comments establish some tuning history, not all of it.
- I inspected three cached legend debug images and all 51 cache records; I did not visually audit every original page or establish every well relationship.
- I did not establish corpus prevalence for text-in-swatch or independently verify the heuristic colour-collision/ambiguity percentages.
- `bakeoff/analysis/categories/llmrev4/` was absent. The promised fresh v4 review was therefore unavailable.
- Archived synthetic, silver/B, consensus and LLM-agreement scores have different denominators and truth quality. They cannot be placed on one accuracy leaderboard.
- Local DINOv2 weights were present, but I did not load them or verify current CUDA operation. No files were written, no network was used, and no `.env` files were opened.
