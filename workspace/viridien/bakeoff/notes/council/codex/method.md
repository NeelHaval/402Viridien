**(A) Top 10 findings, ranked by severity**

The blind LLM is useful for finding disagreements. The current hill-climb does **not** establish improved interval accuracy. All calculations below were read-only. Outputs changed during review; current-CSV figures refer to the **03:00 UTC, 8 November** snapshot.

1. **Critical — “Better” partly means accepting a broader answer.**
   V3 requires exact agreement; v4 receives credit if the judge’s answer is anywhere in its slash-separated group ([conf_data_v4.py:49](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:49)). Reconstructing the comparison from `llmrev/results.json` and `beds_csv/*.csv`: v3 matches **382/583 (65.5%)**; v4 matches **409/583 (70.2%)** with group membership, but **331/583 (56.8%)** with the harness’s label normalization. **78** v4 credits depend on accepting a group, including **54** predictions of `Clay-M / Clay-S / Clay-SACS`. The harness has no such membership rule ([score.py:10](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/harness/score.py:10)).
   **Fix:** Freeze one label contract for both versions. Report ambiguity-aware visual agreement separately. Resolve submission labels using an approved rule; expanding an answer set must not count as improved exact classification.

2. **Critical — The optimization target omits interval boundaries.**
   The specified score is `row_acc × boundary_f1`, with boundary tolerance capped at 20 pixels ([SPEC.md:74](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/SPEC.md:74)). V4 instead compares the label at each crop’s midpoint ([conf_data_v4.py:44](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:44)); **54/583** evaluated crops cross a predicted bed boundary. A correct midpoint therefore conceals misplaced contacts or erased thin beds. Using saved v4 CSVs and the scorer’s row/boundary rules, gold log06’s **labelled span, y=4409–10415**, scores **row accuracy 0.2105, boundary F1 0.3590, product 0.0755**. This is a diagnostic on that span, not the full competition evaluation.
   **Fix:** Make the headline \(S=\frac1L\sum_\ell \mathrm{rowAcc}_\ell\mathrm{boundaryF1}_\ell\), using fixed, independently labelled evaluation regions and equal log weights. That matches the existing harness’s equal-record aggregation when each log has one region ([run.py:84](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/harness/run.py:84)). Keep crop agreement diagnostic.

3. **Critical — The “blind” comparison is already development data.**
   This is explicit: a rejection threshold is described as **“tuned on the 700 LLM-judged windows”** ([legend_classify.py:113](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py:113)). `eval_blind.py` evaluates four parameter variants on that same set ([eval_blind.py:43](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/eval_blind.py:43)). The 700 unique crops cover **all 51 logs**. Hiding classifier answers from the judge prevents anchoring; it does not prevent the developers from overfitting its answers. A new crop from the same bed/log is weak evidence of generalization, and `llm_review.build` contains no exclusion of previous coordinates.
   **Fix:** Declare these 51 logs exposed development material. Group future splits by source log and related well/scan lineage. Use whole-family exclusion to test transfer. An honest never-tuned holdout requires newly reserved, previously uninspected material; retrospective splitting cannot erase exposure.

4. **High — Judge consistency is heterogeneous, and exact ambiguity is being confused with error.**
   Recomputed from `llmrev/results.json`:

   | Repeat comparison | Agreement |
   |---|---:|
   | All duplicate pairs | **57/70 = 81.4%**; Wilson 95% CI **70.8–88.8%** |
   | D-582 pairs | **8/17 = 47.1%** |
   | After treating D-582 Clay/Shale as one visual class | **66/70 = 94.3%** |
   | Both answers marked `high` | **51/51**; lower 95% bound **93.0%** |

   Nine of the 13 disagreements are Clay↔Shale swaps. Unique answers have **557 high, 134 medium, 9 low** certainty ratings. These ratings are not calibrated truth probabilities: on gold log06, **15/15** fully labelled, homogeneous review crops match organizer truth after harness normalization, including **13 marked medium**. Thus D-582’s low classifier agreement does **not** establish that the judge is wrong.

   There is also a systematic evidence restriction: sheets truncate crops to 320 displayed pixels after doubling ([llm_review.py:159](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/llm_review.py:159)). Bounding-box calculations show this discards the right-hand portion of **all 147 D-582 crops**, whose source widths are 246–297 pixels.

   **Fix:** Show full-width crops, allow explicitly ambiguous answers, and measure judge-versus-human errors by family and certainty. A biased judge would steer the hill-climb toward its naming, rendering and ambiguity preferences—even with perfect repeatability.

5. **High — Confidence is fitted and “validated” on the same selected answers.**
   The logistic model fits only high-certainty, LLM-labelled examples, using margin, agreement and family indicators; its calibration display evaluates the fitting observations ([conf_data_v4.py:56](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:56), [conf_data_v4.py:110](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:110)). Saved `conf.json` reports **461 fitted observations**, including a **186/186** successful highest band. That is resubstitution performance. Agreement is also a model feature and then multiplied into its output; the resulting number is not automatically a calibrated probability.

   Family one-hot is not inherently leakage, but here it can encode family-specific judge behavior. It provides no validated confidence for an unseen family. The confidence fit itself does not change bed labels, so it should not be blamed directly for the raw classification uplift.

   **Fix:** Fit and calibrate on separate log groups, evaluate held-out human correctness with Brier score/reliability plots, include every certainty stratum, and test both seen-family and unseen-family calibration.

6. **High — The sampling scheme does not support an unweighted population percentage.**
   The review contains **200 crops in each of three occupied confidence bands**, plus **100 rejected crops**, selected round-robin over logs ([llm_review.py:80](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/llm_review.py:80)). Consequently, D-582 contributes **21.0%** of review crops versus **3.5%** of saved v4 row-grid windows; Gulf contributes **38.1%** versus **59.2%**. Those are substantial distortions for a volume-weighted claim.

   Denominators also change: v3 has **382/600 = 63.7%** agreement over named sampled crops, **382/551 = 69.3%** conditional on both sides naming a rock, and **382/583 = 65.5%** in the comparison. Dropping rejects can reward abstention ([eval_blind.py:35](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/eval_blind.py:35)).

   The separate “2%” UI sample is deterministic, restricted to named windows and clamped to 8–30 ([conf_data_v4.py:91](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:91)): **958 samples**, with actual per-log fractions **0.50–53.3%**.

   **Fix:** Use equal-log interval scores for the headline, fixed family breakdowns, and recorded inclusion probabilities for weighted crop diagnostics. Report rejection coverage and errors together.

7. **High — Human evidence is smaller and less independent than the directory names suggest.**
   CSV counts: organizer gold has **17 intervals/6,006 labelled rows**; the owner’s human file has **three adjacent Clay intervals/92 rows**. `gt/human/labels.jsonl` records displayed classifier suggestions. Those labels are useful corrections, but not a representative blinded test.

   Neel supplies **646 intervals across two logs**, a much stronger resource. However, the dataset explicitly includes edits agreed by two Terra adjudicators, and some side-by-side mixtures retain an arbitrary dominant-label choice ([README.md:43](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/neel_dataset/human_labels/README.md:43)). Agreement with an LLM-edited reference is partly dependent evidence.

   **Fix:** Preserve original human labels, AI edits and human-adjudicated labels separately. Spend owner time on blinded contiguous regions across families, including blanks, contacts and ambiguous patterns. Keep a probability-sampled evaluation stream separate from actively selected failure corrections.

8. **High — Synthetic and external results need an explicit evidence ladder.**
   The harness synthetic generator tiles the classifier’s own legend swatches, draws a boundary line, and generally samples 90–499-row beds ([synth.py:15](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/harness/synth.py:15)). Success can demonstrate recognition of copied motifs and artificial contacts without solving real legend/column mismatch. The richer generator’s own [CRITIQUE.md:148](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/CRITIQUE.md:148) acknowledges source leakage and repeated-pixel dependence.

   External data also has a narrower interpretation: `external_data/kgs/386825/meta.json` describes truth from underlying vector tiles, including where overlays obscure them. Root and test copies share the same PDF hash; they are not independent wells.

   **Fix:** Use synthetic data for training, controlled failures and plumbing; source-disjoint external wells for transfer stress tests; untouched human-labelled challenge-like scans for performance claims. Split donor logs, motifs, overlays and fitted generator statistics before generation. Different seeds alone are not a meaningful holdout.

9. **High — A positive delta is not a statistically defensible “keep” rule.**
   Current CSV reconstruction gives **55 improvements and 28 regressions** among 583 paired crops. A naive exact McNemar test gives **p=0.0040**, but its independence and prespecified-comparison interpretation do not survive repeated tuning on these observations. The 700 unique crops represent only **520 predicted beds**, further clustered within logs.

   For planning only, observed paired discordance is \(q=83/583=0.1424\). The approximation
   \(n\approx(1.96+0.842)^2q/\delta^2\) gives **447 independent pairs for a 5-point improvement**, or **2,794 for 2 points**, at 80% power and two-sided 5% significance. An exploratory log bootstrap suggests roughly **1.78×** variance inflation—about **800 crops for 5 points**, before multiple-comparison allowance. Separately, estimating the observed raw repeat-disagreement rate within ±5 points requires roughly **233 independent duplicate pairs**, versus 70 now. None of these calculations identifies systematic judge error or powers the interval score.

   **Fix:** Preregister candidate batches; confirm on fresh validation groups. Test paired interval-score differences with well/log-cluster uncertainty and a prespecified allowance for multiple comparisons. Require both statistical evidence and a practical minimum gain.

10. **High — Mutable artifacts prevent reliable experiment attribution and stopping.**
    During this review, saved `compare_v3_v4.json` changed from **414 to 408** v4 matches; the 03:00 CSV snapshot reconstructed **409**. This may reflect an ongoing rebuild, but the files cannot be treated as one immutable experiment.

    **Fix:** Save an immutable experiment bundle containing code revision, complete `P`/`B`, data/split hashes, label policy, generator donors/seeds, judge model/prompt/rendering, calibration training groups, predictions, paired changes, failures and resource measurements. Record every attempted variant. Set a confirmation budget before starting—for example ten candidates—and stop when it is exhausted or further practically meaningful gains fail confirmation. Freeze the winner before opening the final test; changing it afterward requires a new test.

**(B) Proposed evaluation protocol — 14 lines**

1. Mark existing reviewed logs, inspected labels and synthetic donors as exposed development data.
2. Acquire/reserve genuinely unseen material; target 60/20/20 train/validation/test by whole well/scan group.
3. Keep every crop, bed, duplicate scan, donor asset and label derivative in its parent group.
4. On development data, run outer whole-family holdouts; tune only inside each remaining-family training pool.
5. Freeze test regions and sheets before prediction, independent of classifier confidence or rejection.
6. Label contiguous regions blindly; retain blanks, uncertainty, mixture conventions and independent human adjudication.
7. Train on synthetic data; validate on unseen donors/nuisances—this establishes controlled robustness only.
8. Evaluate source-deduplicated external wells separately—this establishes external transfer only.
9. Use held-out human real-log intervals for selection; retain LLM picks as audited weak supervision.
10. Headline: equal-log mean of `row_acc × boundary_f1`; also report family results, coverage, depth and resource limits.
11. Fit confidence on separate groups; evaluate calibration against held-out human correctness.
12. Log immutable experiment bundles and preregister confirmation candidates, minimum gain and comparison budget.
13. Keep only confirmed paired interval gains with multiplicity-adjusted cluster uncertainty; stop at the declared budget.
14. Freeze the winner, evaluate the never-tuned test once, and retire that test if its results guide further changes.

**(C) What I could not verify**

- `llmrev4/` was absent: no fresh-review results, repeatability, overlap or completeness could be checked.
- I verified the repository’s scoring specification, not an independently supplied organizer scoring implementation.
- The limited overlapping human truth cannot establish judge accuracy across all families. I did not independently adjudicate the scans or verify that every purported twin is visually identical.
- Historical experiment counts, prior human exposure and enforcement of synthetic donor splits were not established. Live artifact changes prevent assigning every output to one code/configuration version.
- I computed row/boundary diagnostics from saved CSVs without rerunning the pipeline. Depth accuracy, runtime, memory and the sample size needed for a powered **interval-score** test remain unverified.
