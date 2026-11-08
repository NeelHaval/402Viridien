**(A) Top findings, ranked by severity**

**1. The current headline measures agreement with a reused LLM judge, not independently verified task performance.** I reproduced **408/583 = 70.0%** v4 agreement versus **382/583 = 65.5%** for v3. Gulf actually declined from **188/221 = 85.1%** to **185/221 = 83.7%**. These are the available artifacts’ numbers, which differ from the brief. The scorer also excludes LLM “mixed/none” cases from this comparison. [Comparison data](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/compare_v3_v4.json)

The specified score is **row accuracy × boundary F1**, averaged over evaluation windows. Window classification agreement cannot establish that score. Preserve the existing 13-window benchmark and distinguish it from any broader 51-page audit. [SPEC:74](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/SPEC.md:74), [aggregation:84](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/harness/run.py:84)

**2. Confidence is calibrated to the same judge and observations used for reporting.** The logistic model fits on **454 high-sureness LLM windows**; its displayed calibration is explicitly measured on its fitting set. Thus the **188/188 agreement** in the highest calibration band is training evidence. Multiplying the fitted probability by chunk agreement afterwards also needs separate validation. [Confidence fitting:39](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:39), [calibration display:110](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:110), [conf.json](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/conf.json)

**3. Synthetic “exact truth” currently includes non-rock classes.** In `synth_v2/synthetic_index.json`, I counted **1,530 of 52,937 intervals**, covering **275,005/9,420,844 = 2.92% of filled rows**, labelled `Oil_*`, `Gas_Good`, or `Salted water`. The generator trusts the parser’s `is_lithology` flag and then samples these entries as beds. Fix the ontology before generating or training more data. [Synthetic index](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/synthetic_index.json), [asset filtering:56](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/generator.py:56), [bed sampling:168](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/generator.py:168)

**4. Human data exists, but its independence and provenance are fragmented.**

- `bakeoff/gt/human` contains only **three adjacent decisions covering 92 rows**, all Clay, with classifier suggestions shown.
- Neel’s packaged dataset contains **646 beds across two pages**. Its builder applies AI-agreed relabels, insertions, deletions, and legend renames.
- The labeller-ux Hadrian file contains **680 assigned layers**, versus **182** in the older Neel snapshot. These are versions of one page, not additional independent examples.

Preserve the original human annotations and require human confirmation of AI-applied changes before calling the revised version human truth. [Human journal](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/gt/human/labels.jsonl), [Neel provenance:41](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/neel_dataset/human_labels/README.md:41), [applied fixes](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/neel_dataset/human_labels/qa/applied_fixes.json), [Hadrian annotations](<C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/labeller-ux/labelling/labels/60808400140000_hadrian 2_masterlog_2in_10000-21962.json>)

**5. The review representation hides segmentation uncertainty.** From the existing beds and v4 grid, I computed:

- **11,116/83,324 = 13.3%** of 48-row windows cross a **predicted** contact.
- **4,624 predicted beds** contain no grid-window midpoint.
- `conf.json` nevertheless reports **zero mixed windows**.

The exporter assigns each window its midpoint bed’s label. A confident label on such a window cannot certify its boundaries. These counts concern model hypotheses, not verified real contacts. [Window export:77](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py:77), [bed outputs](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/beds)

**6. Synthetic coverage and realism are much weaker than corpus size suggests.** I counted **476 catalogue patterns**, but only **six nonempty approved pattern lists containing 50 crops**. Separately, **453/472 texture models use unverified indexed crops**. More generated images amplify this uncertainty; they do not resolve it. [Catalogue](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/data/catalog.json), [approvals](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/data/approved.json), [texture statistics](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/data/texstats.json)

**(B) Active-learning plan with numbers**

**Separate learning from measurement.** My proposed first purchase is **400 unique human judgements**, plus geometry annotation:

| Family | Current v4/LLM agreement* | Sealed random audit | Random development | Active diagnosis |
|---|---:|---:|---:|---:|
| D-582 | 25/135, 18.5% | 40 | 10 | 18 |
| Geoservices | 41/57, 71.9% | 40 | 10 | 12 |
| Gulf | 185/221, 83.7% | 40 | 10 | 12 |
| Texas | 59/66, 89.4% | 40 | 10 | 4 |
| One-off | 26/32, 81.3% | 40 | 10 | 2 |
| MC765 | 57/57, 100% | 40 | 10 | 1 |
| MC761 | 15/15, 100% | 40 | 10 | 1 |
| **Total** | | **280** | **70** | **50** |

\*Computed from the comparison data above; these are acquisition signals, not human accuracy estimates.

Add **40 independently repeated human judgements**, followed by adjudication of disagreements, and **28 fully annotated 300–600-row strips**, four per family. Split the strips between development and sealed evaluation.

For the audit, sample from **all classifier states**, including predicted blanks. Record selection probabilities, refusals, and unresolved answers. Sample random positions rather than random predicted beds: otherwise a ten-row bed and a ten-thousand-row bed receive equal influence on a row-based metric. Use separate weights for the fixed benchmark and the wider corpus.

For subsequent acquisition, prioritize approximately:

`target weight × estimated uncertainty reduction ÷ annotation time`

Use random development observations to estimate uncertainty and cost. Cap repeated requests from the same bed, motif, or scan lineage.

**Label these diagnostic cases first.** Coordinates below come from joining blind-review rows to current beds; disagreement is a reason to inspect, not proof that the LLM is correct.

| Priority | Exact location | What the human resolves |
|---|---|---|
| 1 | D-582 log03, **y=5586**, bed **4989–5615**; log05, **y=8269**, bed **7968–8401** | Silty Sandstone versus Siltstone, and Halite versus Siltstone |
| 2 | Geoservices DC802, **y=67063**, bed **62147–71282** | Marl versus Clay Shale over a **9,135-row** prediction |
| 3 | Gulf MC822 #10, **y=20741**, bed **20003–21379** | Silty versus Sandy clay-shale; deduplicate the second reviewed crop at y=20549 |
| 4 | Neptune, **y=20891** | LLM “mixed” versus a named v4 window with confidence **0.993**; mark actual contacts |
| 5 | Rio Lavaca, **y=9261** | LLM “mixed” versus SILTSTONE with confidence **0.996** |

Source: [blind results](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/analysis/categories/llmrev/results.json), bed outputs, and `classified_v4/*.json`.

Before that queue, settle the D-582 **Clay/Shale visual-equivalence policy**. Nine of the thirteen duplicate-review disagreements are Clay versus Shale. Raw LLM self-agreement is **57/70 = 81.4%**; treating that documented D-582 equivalence consistently gives **66/70 = 94.3%**. This changes the interpretation of uncertainty substantially.

**How many labels for ±5 percentage points?**

For a binary correctness rate, at individual-family 95% confidence:

`n ≈ 1.96² × p(1−p) / 0.05²`

The conservative answer is **385 independent observations per family**, or **2,695 total**. This is ±5 **percentage points**, not relative percent.

- Correlation and unequal weights increase requirements. A provisional design effect of 1.5 implies about **578 observations/family**, pending pilot measurement.
- **40 audit labels/family is a pilot**, with roughly ±15 points worst-case precision.
- Perfect LLM agreement does not justify zero human allocation: even 15/15 correct observations would have a Wilson lower bound around **79.6%**.
- These counts do **not** guarantee ±5 points on `row_acc × boundary_f1`. Estimate filled-row correctness and boundary TP/FP/FN jointly from random spatial blocks, then propagate uncertainty through the product and aggregation. Do not count adjacent rows or contacts as independent labels.

**Use intervals for production truth; use windows for acquisition.** A human should label every interval inside a short strip, with context beyond its scored edges. Random strips expose missed contacts; reviewing only proposed boundaries cannot measure boundary recall.

Store exclusive pixel bounds, legend-entry IDs, equivalence group, explicit blank/text/unknown/mixed status, uncertain contact ranges, reviewer identity, time, suggestions shown, source version, sampling probability, and split. Preserve side-by-side composition where visible. Distinguish **unreviewed** from **verified blank**.

**Keep and fix the tooling:**

- Keep labeltool’s durable journal, undo, native coordinates, timing, and exclusion of narrowed candidate answers from hard truth. [Store:75](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/store.py:75)
- Add a blind mode with no preselected answer, confidence, or model overlay before submission. Existing review automatically selects suggestions.
- Replace “spot-check confident” as an overall audit: it samples only high-tier items. Its Wilson interval describes that selected population. [Queue selection:94](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/serve.py:94)
- Keep labeller-ux’s rapid bed-entry shortcuts, legend correction, calibration anchors, and save-conflict handling. Add annotation provenance and explicit reviewed masks to the shared export. [Labeller workflow:73](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/labeller-ux/labelling/README.md:73)
- Reconcile Neel snapshots by image/version identity before requesting more full-page work.

**(C) Synthetic/outside data verdict, including LLM silver**

**Synthetic data: useful engineering infrastructure; insufficient evidence of real-distribution fidelity.**

The two generators have materially different limitations:

| Aspect | Evidence | Verdict |
|---|---|---|
| ML bed thickness | Samples **48–349 rows**. Of **2,200** shorter indexed intervals, every one ends at the window edge. | Internal thin beds are absent. |
| Human thickness | Neel 1inch: median **14 px**, **85.4% <48 px**, **38 beds <6 px**. Shenzi: median **24 px**, **68.2% <48 px**. | Large mismatch on these labelled pages. |
| Main synthetic thickness | Clipped lognormal plus fixed **8%** thin component of **6–30 rows** | Better flexibility, but omits internal 1–5-row beds in the sampler. |
| Transitions | Main generator fits first-order transitions, then modifies sampling with rarity, variety, and quotas; ML generator chooses uniformly among other labels. | Neither demonstrates natural sequence fidelity. |
| Scan acquisition | I checked all 51 TIFF headers: **9 binary, 36 RGB, 6 palette**; compression tags **9 Group 4, 42 LZW**. | Binary handling is relevant; compression metadata does not establish visible artifact distributions. |

Sources: [ML sampling:168](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_v2/generator.py:168), [human interval CSVs](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/neel_dataset/human_labels/truth), [main sampling:53](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/generate.py:53).

The main generator’s fitted observations prioritize gold, agreed consensus, prefill, and queue suggestions; **they do not ingest the human-label directory**. Consequently, fitted geological statistics can reproduce model segmentation errors. [Observation sources:83](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/statistics.py:83)

Nuisance probabilities are independent engineering settings. Defaults include JPEG probability 0.20 and fading 0.35; degradation affects the column while donor legend pixels remain unchanged. Procedural/mixed mode does improve contacts by replacing several generic nuisances with measured models, but this remains conditional calibration. [Nuisances:8](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/nuisances.py:8), [procedural application:325](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/synth/generate.py:325)

Keep synthetic data for:

- Matcher training with phase, scale, and acquisition variation, after correcting labels.
- Boundary stress cases: 1/3/6/12/24-row beds, independent grid lines, same-label drawn contacts, blanks, mixtures, and uncertainty.
- Deterministic regression and format/calibration plumbing tests.

**Never use it for real-log headline accuracy, human-confidence calibration, or claims of unseen-well generalization.** Source-disjoint generation is worthwhile, but I verified identical train/held legend hashes in D-582 and MC765. Synthetic selection also already overstates transfer: S17’s synthetic matcher improvement accompanied dev row accuracy changing **39.76%→39.54%**. [S17 report:38](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_gulf/REPORT.md:38)

**Outside data: buy visual compatibility, not geological topic similarity.**

Prioritize other logs with comparable contractor artwork, track layout, legend-to-column mismatch, pixel scale, and scan mode. Unlabelled scans may support representation learning; small paired legend/column examples may support matcher training. Pattern libraries are rendering assets whose interpretation must remain tied to each log’s legend.

Do not assume numerical well logs or core photographs teach printed-symbol recognition. Existing outside data should first pass a small transfer experiment: real development improvement at fixed annotation/training cost. S17’s external composite score fell **0.073→0.059** despite higher row accuracy, which argues against spending broadly on outside data now. [External results:54](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/ml/synth_gulf/REPORT.md:54)

**LLM labels: silver only after human anchoring.**

Preserve blind numbered-legend review and duplicate crops. High confidence can support acquisition and training weights; repeatability does not establish correctness.

Use human development anchors to estimate family-level weak-label reliability with shrinkage. A Dawid–Skene-style model can combine LLM and classifier outputs, but shared prompts, artwork, and training history violate simple independence assumptions. Treat repeated calls from one model as correlated evidence, preserve abstentions, and emit soft targets when unresolved.

Keep human originals immutable. Store LLM proposals, classifier predictions, and adjudications separately. Require human approval of changes promoted to human truth. Seal audit labels from training, calibration, generator fitting, and hill-climbing; repeated evaluation turns any formerly blind set into development data.

**(D) Ordered one-page build plan**

These are planning estimates, not measured delivery times.

| Order | Deliverable | Person-hours |
|---|---|---:|
| 1 | Freeze current predictions and metric definition; inventory annotation versions and well/scan lineage; designate sealed audit data. | 1 |
| 2 | Add blind acquisition, explicit reviewed/blank states, sampling probabilities, and strip-based interval entry. | 2 |
| 3 | Reconcile the 646-bed export and Hadrian versions; human-confirm AI edits; resolve per-legend equivalences and non-rock classes. | 2 |
| 4 | Collect 400 unique judgements, 40 repeats, and 28 strips using the allocation above. | 5 |
| 5 | Build weighted human evaluation with joint boundary/row uncertainty; calibrate using development data only. | 2 |
| 6 | Run one focused classifier improvement cycle from diagnostic labels; compare paired development outcomes. | 2 |
| 7 | Repair synthetic semantics and thin-bed sampling; fit statistics from provenance-qualified development labels; run a small synthetic-mixture ablation. | 3 |
| 8 | Evaluate the frozen winner once on the sealed audit; publish score, uncertainty, coverage, and unresolved cases. | 1 |
| **Total** | **Pilot and one controlled improvement cycle** | **18** |

The five-hour annotation allowance assumes about 20 seconds per judgement and five minutes per strip. Measure actual throughput early. Defer new outside-data acquisition until existing-data improvements justify a specific transfer experiment.

**(E) What I could not verify**

- No `llmrev4` directory was present; fresh-v4 generalization remains unverified.
- I did not independently establish geological correctness, perceptual realism, or well/duplicate-scan independence.
- Available human pages cannot establish representative thickness, transitions, or artifact frequencies across all families.
- I reproduced row/boundary formulas in read-only NumPy: importing the official scorer failed because SciPy required a writable temporary directory. I did not compute ARI.
- I did not run browser/server workflows or training. All successful analysis was read-only, with no network or `.env` access.