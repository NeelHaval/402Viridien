**Spend the next data budget on a frozen, blinded human audit and correcting the label vocabulary. Expanding the synthetic corpus now would amplify known mistakes.**

All computations below used local files, read-only Python, and stdout. No network or `.env` files were accessed.

**(A) Findings ranked by severity**

1. **Critical — the headline currently measures selected LLM agreement, not task accuracy.** `conf.json` reports **414/583 = 71.0%** agreement, after excluding **117/700** unique review windows where the LLM answered mixed/none. The current bed files label **115 of those 117** windows as a rock; those cases need evaluation, not exclusion. Sampling also balances confidence bands and logs rather than representing dataset prevalence. Meanwhile, the specified task score is **row accuracy × boundary F1**, which these window judgments cannot establish.
   Evidence: `bakeoff/analysis/categories/llmrev/results.json`; [sampling code](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/llm_review.py:60); [task metrics](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/SPEC.md:74).

   Confidence is fitted on **461 high-certainty LLM judgments**, then displayed against those same fitting observations. “100% agreement” in its highest band is therefore neither independent calibration nor human accuracy. [Confidence fitting and evaluation](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:39).

2. **Critical — the artefacts disagree about which model is being evaluated.** I counted **15,189 named beds** in current `beds/*.json`, versus **15,323** in `conf.json`; 25 logs have different counts. Recomputing the documented midpoint/twin agreement rule gives **408/583 = 70.0%**, not 414/583. The bed files are newer than the confidence summary. There is **no `llmrev4` directory** in this checkout.

   Freeze predictions, parameters, legend mappings, sampling manifest and hashes together before another hill-climb. Otherwise improvements cannot be attributed reliably. Evidence: `bakeoff/analysis/categories/{conf.json,beds/*.json,compare_v3_v4.json}`.

3. **Critical — synthetic “truth” contains non-rock classes and misses the dominant thin-bed regime.** In `synth_v2/synthetic_index.json`, **1,652 intervals**, representing **15.67% of Gulf filled rows**, are labelled `Oil_*`, `Gas_Good`, `Salted water` or `Dry`. These are being taught as lithology targets. S17’s synthetic corpus has the same problem: **15.52% of Gulf filled rows**.

   Both ML generators sample bed lengths uniformly from **48–349 pixels**, with random next labels excluding the previous label. Neither corpus contains an interior truth interval ≤30 pixels. By contrast, Neel’s two Gulf pages have:

   | Human-labelled page | Beds | Median thickness | Beds ≤30 pixels |
   |---|---:|---:|---:|
   | `60811405120000_1inch tvd mudlog` | 357 | 14 px | 275/357, **77.0%** |
   | Shenzi `60811405350100…` | 289 | 24 px | 175/289, **60.6%** |

   These pages do not establish every family’s distribution, but they directly contradict the ML generators’ Gulf distribution. Evidence: both `bakeoff/ml/synth_*/synthetic_index.json`; `neel_dataset/human_labels/truth/*.csv`; [generator](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/generator.py:156).

4. **High — substantial human work exists, but its provenance and coverage need consolidation.** The bakeoff human journal contains only **three adjacent decisions covering 92 rows**, all Clay in D-582 log02. Neel’s packaged dataset supplies **646 beds**, but both pages are Gulf, and the package applies AI-adjudicated relabels, insertions and deletions. Preserve those edits separately from original human decisions; two agreeing AI reviewers do not make an edit human truth. [Dataset provenance](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/neel_dataset/human_labels/README.md:41).

   There is also a useful Geoservices asset: `labeller-ux/labelling/labels/60808400140000_hadrian 2_masterlog_2in_10000-21962.json` contains **680 assigned layers**, versus **182** in the older `neel_labels` copy. Reconcile versions before requesting more labelling.

   My diagnostic rescore of current beds against Neel’s packaged intervals gives **92.3% and 98.5% row accuracy**. Nevertheless, Shenzi Siltstone accuracy is **75.1%** and Limestone **67.3%**; Halite occupies **55.4%** of its labelled rows. High row accuracy can hide weak minority classes. These are development diagnostics, not independent headline estimates.

5. **High — label equivalence is distorting the evidence.** Raw LLM repeat agreement is **57/70 = 81.4%**. All nine D-582 repeat disagreements are **Clay ↔ Shale**, the documented identical-picture pair. Collapsing just that equivalence gives **66/70 = 94.3%**, and D-582 becomes **17/17**.

   Thus, 81% is not an demonstrated LLM reliability ceiling. Conversely, 94% repeatability is not accuracy. Store per-legend equivalence explicitly and report both exact-entry and equivalence-aware results. Do not globally alias geologically different labels. Evidence: `llmrev/results.json`; [existing twin scoring](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/llm_summary.py:18).

6. **High — the newer synthetic pipeline has good safeguards, but the human-to-generator connection is incomplete.** Production `approved.json` contains **50 crops across six nonempty patterns**, from only **D-582 and Geoservices**, against **476 catalogue patterns**. Three “approved” entries contain no crops; **310 patterns have no compliant candidates**. An approval counter is not usable training coverage.

   Moreover, `statistics.observations()` reads queue suggestions, prefill, agreed consensus and gold—but **not `gt/human` or Neel’s dataset**. The inspected mock fit records 72,837 queue and 40,089 prefill input observations, versus 17 gold observations; these are overlapping input counts, not independent examples. The generator is largely learning machine-produced bed statistics. [Statistics inputs](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/statistics.py:83).

**(B) Active-learning plan with numbers**

Maintain separate **development acquisition** and **sealed audit** queues. Their purposes and sampling rules differ.

For the fixed 51-log collection, use randomly selected native row locations to estimate filled-row accuracy. Keep blank, unknown and unresolved outcomes. Report boundary F1 and the specified composite separately on completely annotated strips. Report family results alongside the overall result.

A practical first allocation is:

| Family | Logs | Share of column windows¹ | Random audit locations | Targeted development decisions |
|---|---:|---:|---:|---:|
| Gulf | 18 | 59.24% | 250 | 50 |
| Texas | 7 | 15.76% | 75 | 30 |
| MC765 | 4 | 9.28% | 40 | 5 |
| One-off | 5 | 6.07% | 40 | 20 |
| Geoservices | 5 | 5.27% | 40 | 40 |
| D-582 | 9 | 3.51% | 35 | 60 |
| MC761 | 3 | 0.87% | 20 | 5 |
| **Total** | **51** | **100%** | **500** | **210** |

¹Computed from `conf.json`’s full-column window counts; these are pixel-volume weights, not geological depth weights.

For the audit, sample within each family’s complete column extent, including predicted blanks. Record inclusion probabilities and use sampling weights. Estimate filled-row accuracy as a weighted ratio; also report false rock predictions on human-labelled blank rows. Do not replace difficult audit items with easier ones.

For development, prioritize disagreement, expected affected row volume, boundary uncertainty and diversity **per annotation minute**. Deduplicate nearby windows from the same bed: the 700 existing review windows map to only **519 current predicted beds**. Include confident agreement cases to detect shared model mistakes.

Start with these concrete targets; coordinates are native pixels, bottom exclusive:

| Priority target | Evidence and review action |
|---|---|
| **D-582 log04, window 9712–9760** | Review ID **465**: current bed **9318–9800** is Halite; high-certainty LLM says Siltstone. Label the containing interval and both contacts. |
| **D-582 log01, bed 21040–23296** | Current SIllty Sandstone, second Dolomite, chunk agreement **0.638**. Inspect separated positions and internal changes before accepting one label across this long bed. |
| **Geoservices DC802, bed 62147–71282** | Review ID **164**, window **67063–67111**: Marl versus high-certainty Clay Shale. Large potential row impact despite chunk agreement 1.0. |
| **Gulf MDN-1, bed 184903–187576** | Review ID **616**, window **185463–185511**: current bed is blank, LLM says Shale. Explicitly test false rejection. |
| **Texas Critter Creek 201, window 43160–43208** | ID **355**: BRECCIA versus MARLSTONE; bed agreement **0.601**. Check the analogous Critter Creek 202 case as a related example, not an independent well. |
| **One-off Shell AC0336, window 139129–139177** | ID **330**: Sand-MSS versus Sand-MS. Verify legend identity and distinguishability before treating it as a classifier failure. |

Evidence: `llmrev/results.json` joined to current `beds/*.json`. For D-582, resolve the Clay/Shale/SIllty Sandstone interpretation with human-reviewed native legend examples before propagating labels.

**How many for ±5 percentage points?**

For a **95% interval on a binary accuracy proportion**, the usual planning calculation is:

\[
n \approx \frac{1.96^2p(1-p)}{0.05^2}.
\]

- With unknown accuracy: **385 independent evaluable labels per family**, or **2,695 across seven families**.
- If independently established human accuracy is 80%, 90% or 95%, approximate requirements become **246, 139 or 73**. Do not substitute LLM agreement for that prior knowledge.
- The proposed 500-location allocation has an idealized worst-case overall half-width of **4.54 points** for an all-row binary outcome. Blank exclusions, unresolved labels, weighting and clustered acquisition can widen it. It does **not** deliver ±5 points per family.
- A design effect of two requires roughly **770 observations per family** for the conservative target. Thousands of nearby labels cannot establish generalization to new wells when a family contains only a few independent well lineages.
- These counts do **not** establish ±5-point precision for boundary F1 or the composite. Start with **six randomly located, fully annotated 600-row strips per family—42 strips total**; estimate uncertainty from the actual sampling design and expand where needed.

For hill-climbing, compare candidates on the **same human-labelled examples**, recording paired improvements and regressions. Candidate-disagreement sampling is useful for development; unweighted accuracy on that queue is not population accuracy. Keep the sealed audit out of threshold selection, synthetic fitting and LLM adjudication.

**Label format and tooling**

Use a hybrid:

- **Random point with context:** fastest way to estimate row accuracy; label the target row without assuming the whole 48-row crop is homogeneous.
- **Native intervals:** best development return when one label covers a coherent bed. Two boundary clicks plus a legend ID provide training patches, thickness and transition information.
- **Complete strips with boundary marks:** necessary for missed-bed and false-boundary evaluation. Reviewing only proposed contacts cannot measure boundary recall.

Keep the labeltool’s native-aspect crops, keyboard controls, timing and append-only undo journal. Keep labeller-ux’s sequential **F + rock** entry, legend editing, depth anchors and conflict handling. [Labeltool](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/README.md:19); [labeller-ux](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/labeller-ux/labelling/README.md:83).

Fix these before collecting evaluation labels:

- Add a **blind first pass**: hide predictions, confidence and preselected answers.
- Separate **blank, unknown rock, lateral mixture and unreviewed**. Existing Unknown/not-lithology handling conflates outcomes.
- Allow thin-bed insertion, splitting and merging without snapping to model proposals.
- Store image/legend hashes, native coordinates, legend-entry/equivalence IDs, reviewer, provenance, sampling probability and dataset role.
- Add **10% independent blind duplicate review**, then human adjudication of disagreements.
- Treat `/conf` and `/mix` “wrong” clicks as triage signals: their current controls do not supply corrected truth. [Current click behaviour](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/static/confreview.html:98).

The existing Spot-check mode estimates agreement only among confident suggestions. Its Wilson interval should not be presented as whole-dataset accuracy.

**LLM labels and fusion**

Use LLM judgments as **silver proposals**, after checking vocabulary validity and measuring reliability against blinded human development labels. Retain mixed/none, alternate choices and model/prompt versions.

A Dawid–Skene-style model is reasonable only with **human anchors**, regularized confusion matrices and enough coverage of the relevant confusion pairs. Estimate reliability by family where supported, shrinking sparse groups toward pooled estimates. Repeated calls to one model are dependent; the classifier confidence already fitted to that LLM is also not an independent vote.

Initially, simple audited rules are preferable: retain soft silver labels for supported agreement cases; send disagreements, unknowns and poorly supported classes to humans. Never let LLM consensus overwrite a human audit label automatically. Preserve Neel’s original labels and AI amendments as separate records.

**(C) Synthetic and outside-data verdict**

**The newer `bakeoff/synth` design is useful engineering infrastructure, not yet a representative dataset.** Keep its source-log separation, provenance, procedural rendering, clean/noisy pairs and explicit missing coverage.

Its remaining statistical limits are substantial:

- Thickness uses clipped log-normal fits plus an assumed **8% thin component**; rarity weights and quotas deliberately change prevalence. Transitions are first-order and mostly derived from machine suggestions. [Bed sampling](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/generate.py:53).
- Procedural textures use measured geometry and colour, but motif selection also contains label-dependent rules and can use unverified source appearances. Nine recorded round-trip checks validate selected texture measurements, not full distributional realism. [Renderer](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/procgen.py:20); `synth/out/procgen_roundtrip.json`.
- Nuisance probabilities are engineering choices. Column perturbations do not reproduce a shared page acquisition process affecting both legend and column. Blank gaps are reset to pure white after perturbations, making that negative class artificially clean. [Nuisances](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/nuisances.py:129).
- Source-log separation does not establish independent wells or artwork. S21 has only **one held source log per family**. [S21 report](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/REPORT.md:31).

**Good uses:** legend-conditioned matcher training, controlled hard negatives, boundary stress cases, artifact robustness and deterministic regression tests. Add genuinely thin beds, same-label drawn contacts, grid-only lines and non-rock overlays with correct targets.

**Never use synthetic scores for headline real-log accuracy, real-world confidence calibration or claims about unseen wells.** Known synthetic depths and reused donor legend caches also cannot validate depth OCR or fresh legend parsing. The existing [CRITIQUE](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/CRITIQUE.md:112) correctly identifies these limits.

There is already evidence against assuming transfer: S17 improved held synthetic Gulf matching **59.38% → 66.67%**, while real dev matching moved **39.76% → 39.54%**. Its external composite fell **0.073 → 0.059** versus S15. [Recorded ablations](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_gulf/REPORT.md:38).

For outside data, prioritize:

| Candidate | Plausible use | Acceptance condition |
|---|---|---|
| Pattern libraries | Rendering primitives and confusing-pair construction | Match each target legend’s appearance; do not assume universal symbol semantics. |
| Similar scanned logs with legends | Self-supervised pretraining and scan-artifact modelling | Demonstrated benefit on frozen target-family human development data. |
| Existing KGS/PDF-derived intervals | Geometry and matcher pretraining | Audit extraction and style differences; keep wells separate. |
| Numerical well logs, core photos, unrelated geology images | Little immediate value for this image-matching task | Defer without a specific, testable transfer hypothesis. |

The local KGS inventory is already sizeable: round2 records **21 accepted pages and 134 skipped pages** for one PDF, demonstrating selection effects in extraction. Use what is already present before gathering more. Evidence: `external_data/kgs/round2/summary.csv`.

**(D) Ordered build plan — approximately 16–18 person-hours**

These are planning estimates, not measured annotation throughput.

| Order | Deliverable | Engineering | Human review |
|---|---|---:|---:|
| 1 | Freeze predictions, hashes, metric definitions and separate development/audit manifests. Reconcile stale confidence outputs. | 0.75 h | — |
| 2 | Consolidate Neel/Hadrian versions; preserve AI edits separately; validate legend twins and remove non-rock training targets. | 1.25 h | 0.5 h |
| 3 | Add blind point/interval review, explicit blank/mixture states, sampling probabilities and duplicate assignment. | 2 h | — |
| 4 | Complete 210 targeted decisions and 42 random strips, with duplicate checks and adjudication. | — | 4–5 h |
| 5 | Build weighted human metrics, paired candidate comparisons and conservative silver-label reliability estimates. | 1.5 h | — |
| 6 | Fix synthetic thin beds, vocabulary and human-statistics ingestion; run one bounded augmentation ablation. | 2 h | — |
| 7 | Freeze the selected candidate; complete the 500-location audit and report actual intervals, failures and coverage. | 0.5 h | 3.5–4 h |

**Allocate zero hours to new outside-data collection in this first pass.** Expand only after the human evaluation identifies a missing appearance or artifact that existing data cannot supply.

**(E) What I could not verify**

- No fresh `llmrev4` results or production `synth/data/fake` corpus were present.
- I did not independently adjudicate geological identities, visually validate all annotations, or certify Hadrian’s 680 layers as reviewed human truth.
- Duplicate scans, shared well identities and historical training exposure remain unaudited.
- Full generator realism—especially transition distributions and correlated scan artifacts—has not been established by the available checks.
- A direct harness import failed because SciPy required a writable temporary directory. Human row-accuracy and boundary-F1 diagnostics were recomputed with NumPy using the inspected scorer’s normalization and tolerance; ARI and depth MAE were not rescored.
- UI behaviour and annotation speed were inspected through code and documentation, not a live browser session.
