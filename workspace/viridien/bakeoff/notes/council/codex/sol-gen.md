**The current results show adaptation to these 51 logs, not demonstrated generalisation to unseen families. D-582 is my strongest prediction for the worst unseen-family failure.**

I read the ML reports before forming the recommendations below. One correction to the brief: the saved comparison reports **408/583 = 70.0%** v4 agreement, including Gulf **185/221 = 83.7%** and Texas **59/66 = 89.4%**. These are agreements with LLM picks, not independent human accuracy. [`compare_v3_v4.json`](bakeoff/analysis/categories/compare_v3_v4.json)

**(A) Per-family overfit table**

Two findings apply across the table:

- I counted **700 answered, nonduplicate review crops covering all 51 logs**. Every family contributed to the review evidence available during development. The classifier explicitly describes its mixed-window threshold as “tuned on the 700 LLM-judged windows” and its blended distance as best on that evaluation. [`llmrev/results.json`](bakeoff/analysis/categories/llmrev/results.json); `legend_classify.py:113,275`.
- `P` and `B` are mostly shared constants. I can establish family exposure and identify assumptions favourable to each family; I cannot reconstruct which family caused every parameter choice. [`legend_classify.py:29–31`](bakeoff/labeltool/legend_classify.py); [`beds.py:26`](bakeoff/labeltool/beds.py).

Counts below come from `analysis/label_shares_per_log.csv`. Agreement comes from the saved comparison above.

| Family | Logs; current v4 agreement | Choices developed with this family visible | Predicted failure when unseen |
|---|---:|---|---|
| **D-582 1-bit grid** | 9; **25/135, 18.5%** | Explicit comment that column marks can be about twice the legend scale; twin grouping handles visually equivalent entries. The monochrome boundary branch assumes approximately 25-row repetition, doubles comparison width, and relaxes paper/ink thresholds. Scale search currently defaults **off**. `legend_classify.py:175–176,192–213,224–225`; `beds.py:67–88`. | **Worst expected failure:** colour contributes no discrimination; dash orientation, spacing, phase and scale carry the decision. Without the monochrome accommodations, repeated marks also resemble boundaries. Clay/Shale ambiguity remains even with perfect segmentation. |
| **Geoservices grid** | 5; **41/57, 71.9%** | Shared colour/texture/NCC weights were evaluated on all five logs. Upstream v2 parsing includes a flattened-grid interpretation developed after seeing these layouts. Two logs expose 30 lithology candidates. `legend_classify.py:144–148,275–285`; `notes/legfix.md:38,47–48`. | Fine-grained matching among similar sand, sandstone, clay and carbonate patterns; incorrect section assignment can eliminate the correct candidate before matching. |
| **Gulf 3-panel** | 18; **185/221, 83.7%** | Largest family represented in review. Shared mixed/blank tests and whole-window descriptors see these patterns. Parser repair inspected formerly failing Gulf layouts, including Ardennes’ short technical row. `legend_classify.py:66–117`; `notes/legfix.md:7–19`. | New panel proportions or side-by-side mixtures change the pooled feature statistics. A new heading/export arrangement can again misclassify rocks as modifiers or technical symbols. |
| **MC761 mineral/lithology** | 3; **15/15, 100%** | Shared thresholds saw all three logs; upstream parsing already distinguishes mineral and lithology sections. The audit excluded 13 mineral entries plus two non-rock cells. `notes/legend_quality.md:16`; `legend_classify.py:165–168`. | Section ownership and sparse grey/white patterns. Fifteen agreeing crops provide little evidence about rare classes, thin beds or a differently arranged mineral panel. |
| **MC765 colour grid** | 4; **57/57, 100%** | Shared pattern weights, native-resolution swatches and twin handling were evaluated on these logs. The two-chunk rule suppresses short pattern changes. `legend_classify.py:151–157,275–285`; `beds.py:129–145`. | Same-colour subclasses and legitimate short changes. The perfect reviewed score is especially weak evidence of new artwork: all four logs share identical ordered native swatch arrays in the cache. |
| **Texas “Rock Types” box** | 7; **59/66, 89.4%** | Upstream detector explicitly handles unbordered Rock Types cells. Shared border stripping, swatch trimming and NCC then operate on unusually shallow samples. `harness/legend.py:84–109`; `legend_classify.py:181–187`. | A different shallow grid can fuse samples and labels. Tiny or partial motifs then provide insufficient evidence for texture matching. Callaway already has several rock names fused into each returned entry. `notes/legfix.md:63`. |
| **One-off layout** | 5; **26/32, 81.3%** | All five layouts contributed review crops, but no explicit One-off-specific branch appears in these two classifier files. The category spans **2–47 parsed entries**. `notes/legfix.md:49,70–71,76–77`. | Most unpredictable layout risk. This is a residual category, not one coherent visual family; holding it out mixes several different transfer problems. |

**Why D-582 is the clearest prediction:** it remains poor despite explicit accommodations. Independently, S22’s held-source synthetic boundary F1 is **0.397** for D-582 versus **0.820–0.933** for the other six families. That is synthetic evidence, but it points in the same direction. [`ml/boundary_v2/REPORT.md:43–51`](bakeoff/ml/boundary_v2/REPORT.md)

There is also substantial dependence between “logs.” Hashing shapes and pixel bytes of the ordered native swatch arrays gave **one swatch-set hash across nine D-582 logs**, and **one across four MC765 logs**. A different filename therefore does not necessarily test new legend artwork. Computed from `bakeoff/analysis/categories/native_swatches.pkl`.

**(B) Ranked failure modes on unseen legends**

This ranks engineering risk inferred from the available logs; it is not an estimated probability distribution over future styles.

| Rank | Failure mode | Evidence and expected mechanism |
|---:|---|---|
| **1** | **Layout and label ownership: grid, list, boxed, sectioned** | The original parser returned zero lithologies on **11/51 logs**. Repairs depended on recognising flattened grids, headings and technical rows. A later manual check found **33 Texas entries with 0% detection under v1**, versus 100% under v4. Matching cannot recover omitted candidates. `notes/legfix.md:7–19`; `ml/legend_v4/REPORT.md:70–74`. |
| **2** | **Legend-to-column pattern scale mismatch** | Native-pixel correspondence is an explicit classifier assumption, immediately qualified by the D-582 ×2 example. NCC uses native pixels; scale search is disabled by default. Enabling the existing search actually reduced reviewed agreement from **383/551 to 372/551**. Simply switching it on is already a failed experiment. `legend_classify.py:4,121–134,175–176,224–225`; `notes/eval_blind.log:6–9`. |
| **3** | **Many lookalike patterns** | I counted positive colour-collision diagnostics on **39/51 logs**, including every Geoservices and MC765 log. Candidate counts reach 30 and 47. These are heuristic diagnostics, not error rates. More candidates make small margins and mistaken twin merges more consequential. `analysis/label_mix.json`; `legend_classify.py:206–213,286–294`. |
| **4** | **Colour assumptions meeting 1-bit or sparse monochrome scans** | All nine D-582 logs are 1-bit; their analysed swatches are entirely hatch-dependent. Sparse white patterns can trigger blank rejection, while repeated marks can trigger mixed/boundary detection. The special monochrome branch reduces this problem without solving recognition. `analysis/legend_difficulty.csv`; `legend_classify.py:90–117`; `beds.py:67–88`. |
| **5** | **Different swatch sizes and incomplete motifs** | Cached native swatches range from **5–40 px high in Texas**, versus **103–107 px in D-582**. Fixed two-pixel trimming, a minimum retained size and top-left template truncation have very different effects across that range. Computed from `native_swatches.pkl`; `legend_classify.py:128–134,181–187`. |
| **6** | **Text inside the swatch** | The Allegheny sheet visibly embeds letters in carbonate patterns. These can be class-defining marks, whereas surrounding annotation text is interference. Current ink/run/NCC features make no semantic distinction. Removing all text would erase useful evidence. `analysis/img/mix_oneoff.png`; `legend_classify.py:66–86`. |
| **7** | **Isolated symbol-only entries** | The inspected sheets include isolated glyphs, and the parser audit records mineral, fossil and drilling-symbol exclusions. A future legend that legitimately uses isolated glyphs for lithology would challenge both filtering and repeat-based descriptors. The repository does not establish their dataset-wide frequency, so this rank has the weakest prevalence evidence. `notes/legend2_overview2.png`; `notes/legend_quality.md:12,15–16`. |

The upstream bottleneck is quantitatively large. In S14’s development diagnostic, supplying oracle legend boxes/names raised true-interval matching from **53.47% to 87.34%**. That comparison changes both candidate availability and swatch quality; it is not a pure OCR attribution. [`ml/match/REPORT.md:23–30`](bakeoff/ml/match/REPORT.md)

**(C) Ranked learned-method plan**

**First gate: correct candidate coverage.** Measure legend-entry detection, label ownership and usable-name recall before investing in embeddings. Later parser improvements are separate from the current classifier’s v2 native-swatch cache. `legend_classify.py:151–168`; `ml/legend_v4/REPORT.md:3,24–31`.

The gains below are **planning hypotheses**, not measured improvements over current v4. Units are absolute percentage points on a new human evaluation, with parser quality held fixed.

| Rank | Concrete next experiment | Already tried; what changes this time | Expected gain; effort; risk |
|---:|---|---|---|
| **1** | **Legend-conditioned metric learning:** train a small shared encoder on query–swatch pairs, with hard negatives from the same legend. Generate columns from each training log’s own swatches; hold out source groups. | S7 already trained a per-log CNN on tiled swatches with jitter and Viterbi; its saved partial run has **50.0% row accuracy over nine synthetic windows**, and its note describes a severe real domain gap. S14/S17 instead learned logistic pair scores. The new experiment learns a **shared visual embedding**, rather than S7’s per-log closed-set classifier or S14’s fixed descriptors. It must explicitly test scale mismatch beyond S17’s 0.85/1/1.15 augmentation. `notes/s07_synth_cnn.md:3–7`; `out/s07_synth_cnn/synthetic/metrics.json`; `ml/match/REPORT.md:5–7`; `ml/synth_gulf/REPORT.md:62,68`. | **2–8 row points** is a reasonable target, with a substantial chance of zero. **4–8 hours** using existing generators. Main risk: learning tiling/seam signatures and repeatedly seeing the same artwork. |
| **2** | **Reuse and refit the small boundary CNN**, comparing real-only against synthetic augmentation, while fixing the matcher. | This is already supported by S13: external boundary F1 **0.216→0.324**, in-domain **0.546→0.569**. S22 already tried stronger scan augmentation and found it worse than its selected synthetic variant. The useful change is representative real thin-bed/contact truth and honest family exclusion—not another arbitrary noise sweep. `ml/boundary/MODEL_CARD.md:12–21`; `ml/boundary_v2/REPORT.md:19–29`. | **0–4 row points; 3–10 boundary-F1 points** target. **2–4 hours**, excluding annotation. Low compute risk; moderate risk of learning grid lines or generator seams. |
| **3** | **Semi-Markov/DP decoding over direct legend-match emissions**, with a soft duration prior and explicit blank/unknown states. | S4 already used Viterbi: **91.5% synthetic row accuracy**, but **0 on gold log06**. S18 already implements duration-constrained joint DP. It improved external composite **0.073→0.112** versus S15, but reduced in-domain composite **0.367→0.347**. Its selected extra boundary penalty and boundary-log-odds weight were both zero. The new question is whether a **soft, scale-normalised duration prior** beats fixed pixel cutoffs. `notes/s04_kmeans_viterbi.md:14–26`; `ml/integrate/core.py:54–91`; `ml/integrate/REPORT.md:3,13–27,104`. | **0–3 row points; 0–8 F1 points** target. **2–4 hours**. Moderate risk: smoothing away real thin beds and making confidently wrong labels more persistent. |
| **4** | **Frozen DINOv2 as an auxiliary matcher**, using local patch tokens, multiple scales/phases and native texture/colour evidence. | S5 already used frozen DINOv2 CLS embeddings: **94.6% synthetic row accuracy**, but its real rendering confused shale with silty sandstone. Repeating global nearest-embedding matching is not a new proposal. The difference would be local token correspondence and an explicit same-legend hard-negative test. `strategies/s05_dinov2.py:27–45`; `notes/s05_dinov2.md:5–7`. | **0–3 row points**, with zero the prudent default expectation. **2–4 hours**. Low fitting cost, high risk that semantic embeddings discard the fine hatch distinctions that matter. |

The 12 GB GPU is not the limiting resource for these small experiments. The cached DINOv2-small weight file exists locally at approximately **88.25 MB**; S5’s archived synthetic run recorded about **0.30 GiB** peak reserved GPU memory. S17’s small boundary/matching system trained on CPU in minutes. These establish feasibility, not timings for the proposed token matcher. `out/s05_dinov2/synthetic/metrics.json`; `ml/synth_gulf/REPORT.md:72`.

**Do not equate “more synthetic” with generalisation.** S17 improved in-domain row accuracy **53.42→63.75%**, but reduced external composite **0.073→0.059**. S21 already expanded to **10,000 synthetic windows**; its report claims selection results only, with final evaluation pending. [`ml/synth_gulf/REPORT.md:7–12,54`](bakeoff/ml/synth_gulf/REPORT.md); [`ml/synth_v2/REPORT.md:33–36,72–75,129–130`](bakeoff/ml/synth_v2/REPORT.md)

**(D) The holdout experiment**

1. **Freeze the protocol before further hill-climbing.** Record code, preprocessing, augmentation ranges, model selection rule and annotation policy. The current 51-log review cannot become an untouched test retrospectively. Even clean refits retain design exposure; the S19 model card explicitly acknowledges this. `ml/combine/MODEL_CARD.md:30–40`.

2. **Run two separate outer evaluations.**

   | Evaluation | Exclude from fitting and selection | What it measures |
   |---|---|---|
   | **Log/source-group holdout** | All crops, synthetic donors, overlays, pseudo-labels and statistics from the held log’s well/scan lineage | Transfer to another source; report shared-template cases separately |
   | **Seven-fold family holdout** | Every log and derived asset from the held family | Transfer of the learned method across the existing family taxonomy |

   Tune only inside each training partition. Group MD/TVD versions, continuation scans and near-duplicates. Exact swatch hashes alone are insufficient for finding all related scans. Existing source-log separation explicitly does not establish well independence. `synth/README.md:374–390`.

   For the existing hand-engineered pipeline, call family holdout a **retrospective transfer test**: its rules were already developed with these families visible. A truly untouched style requires a new source outside that development history.

3. **Declare test-time support access.** Every method may read the held log’s legend: that is the task input. Compare:

   - **Frozen matching:** no parameter updates from the held legend or column.
   - **Legend-only adaptation:** a fixed training budget using synthetic samples from that legend, with no column labels or model-selected “correct” examples.

   Keep these results separate. Otherwise per-log synthetic training can quietly change the meaning of “held out.”

4. **Minimum human truth: an exploratory floor, then replication.**

   - **Seven complete legend annotations**, one per family: swatch boxes, printed names, rock/non-rock status and genuinely indistinguishable entries.
   - **Two independently sampled contiguous 1,000-row blocks per log:** **14,000 labelled rows** initially. Mark all contacts, blanks and ambiguous mixtures; isolated labelled windows cannot evaluate segmentation.
   - Use a declared dominant-rock/mixture policy and preserve uncertainty. Double-review ambiguous matches and thin contacts without displaying predictions.
   - Prefer **two independently sourced logs per family**—28,000 rows under the same scheme—before making family-level claims. Where the dataset lacks independent source groups, report that absence.

   Existing Neel truth can reduce annotation work: it contains **646 beds on two Gulf pages**, with documented QA and mixture caveats. It supplies depth within one family, not seven-family coverage. [`neel_dataset/human_labels/README.md:3–12,43–60`](neel_dataset/human_labels/README.md)

5. **Score the stages separately and end to end.**

   Report legend recall/name accuracy; matching with human legend entries and true bed interiors; then full interval prediction with automatic parsing and boundaries. Include row accuracy, per-class recall, boundary precision/recall, thin-bed recall, blank false positives, abstention coverage and runtime.

   Average by source, then family; retain worst-family results. Bootstrap source groups rather than crops. Score literal labels and predeclared visual-equivalence groups separately.

   Add blank scoring explicitly: the current harness ignores blank truth rows. Also report tighter boundary tolerances alongside the existing tolerance, which can reach 20 px and conceal errors on very thin beds. `SPEC.md:74–81`; `synth/CRITIQUE.md:50–53,73–75`.

Fit confidence only on inner held-out predictions. The current family-one-hot confidence model is fitted and displayed on the same selected, high-certainty LLM windows; it provides no validated confidence for a new family. `notes/conf_data_v4.py:34–58,110–116`.

**(E) What I could not verify**

- **No independent unseen-family result was located.** Existing family tables generally test held sources within represented families.
- I did **not rerun training or inference**. Archived scores use different parsers, truth tiers and aggregation rules; they are not one directly comparable leaderboard.
- The fresh v4 blind-review folder had no results or summary when inspected, so I could not validate fresh-window performance.
- I inspected saved legend sheets and representative column figures, not every full-resolution page. Dataset-wide frequencies of embedded text and isolated glyphs remain unverified.
- Exact swatch reuse is verified; complete well/scan lineage and near-duplicate overlap are not.
- LLM self-agreement was **57/70**, not independent correctness. Neither synthetic agreement nor LLM agreement resolves visually indistinguishable labels. `analysis/categories/llmrev/summary.json`
- All work was read-only, without network access or opening `.env` files.
