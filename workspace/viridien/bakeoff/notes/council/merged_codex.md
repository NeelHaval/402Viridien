The council’s central conclusion holds: **the reported hill-climb does not establish improved interval accuracy.** I verified the major implementation defects, but two previously reported artifact failures have since been corrected.

Read all ten reports; used only read-only local checks. Ignored `llmrev4/`; no network, file writes, or `.env` access.

Report IDs: **A/S** = Astra/Sol; **m/a/d/g/f** = method/algo/data/gen/flywheel. Thus **Ad = dataA.md**, **Sd = data.md**; other Sol reports have `sol-` prefixes. Source abbreviations: [LC = legend_classify.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/legend_classify.py), [BED = beds.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/labeltool/beds.py), [CONF = conf_data_v4.py](C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff/notes/conf_data_v4.py). Other paths are relative to `bakeoff/` unless stated. Hours are estimates; overlapping work across plans is not additive.

**Merged findings, most severe first**

1. **Critical — selection measures the wrong objective with asymmetric grading.**
   **Raised:** all ten. **CONFIRMED:** `SPEC.md:74–81` requires row accuracy × boundary F1; `CONF:44–50` scores midpoint labels and accepts group membership only for v4. Recomputed: v3 **382/583**, v4 **408/583**, including **78 group-dependent credits**; v4 exact/normalized agreement is **330/583**. Another **117 mixed/none answers** are excluded, although v4 names 115. Gold log06’s original evaluation window gives **21.05% row accuracy, 0.3902 boundary F1, product 0.0821**.
   **Fix:** freeze identical label rules; select using independently labelled intervals, with crop agreement diagnostic only. **2–3 h.**

2. **Critical — development evidence is repeatedly treated as validation.**
   **Raised:** Am, Sm, Aa, Sa, Ag, Sg, Af, Sf. **CONFIRMED:** `LC:113` explicitly records tuning on the reviewed windows; `notes/eval_blind.py:43–47` compares four variants against them. I counted **700 unique crops spanning all 51 logs**. The current comparison has **55 gains and 29 losses**, but repeated adaptive testing prevents treating that as independent confirmation.
   **Fix:** permanently designate existing reviews as development; reserve genuinely protected sources, predeclare candidate/effect budgets, and compare paired source-level interval scores. **2–4 h**, excluding new labels.

3. **High — false twins erase class distinctions; NCC can ignore distinguishing marks.**
   **Raised:** Am, Sm, Aa, Sa, Ad, Sd, Ag, Sg. **CONFIRMED:** `LC:121–134,191–213`. I inspected Allegheny’s distinct M/W/P symbols; their bidirectional NCC values **0.913/0.803/0.791** nevertheless merge them. FREEMAN’s differently coloured marl symbols produce identical features. Two controlled swatches differing across **85.94%** of pixels both score NCC **1.0** because only their common corner is matched.
   **Fix:** preserve local entry IDs; compare complete motifs and coloured ink; permit only adjudicated visual equivalences with pairwise group consistency. **3–5 h.**

4. **High — rock/blank/unknown handling fails in both directions.**
   **Raised:** Aa, Sa, Ad, Sd, Ag, Sg. **CONFIRMED:** `LC:52–63,262–294`. All **nine D-582 Coal swatches** self-classify as blank; uniform greys **90–139** also become blank. Conversely, inspected Trail rows **1424–2159** contain grid-only paper but are labelled IGNEOUS; fresh crops reproduce this. Random noise also receives a named rock because `unknown=99`.
   **Fix:** establish track geometry globally, use complete paper/ink/fill modelling, and calibrate explicit unknown/grid states. **2–4 h.**

5. **High — greedy merging erases beds and discards confidence evidence.**
   **Raised:** Aa, Sa. **CONFIRMED:** `BED:137–188`. Controlled `A,B,B` chunks become one B bed; `A,A,blank` becomes A. Current outputs contain **90 named beds incorporating 6,336 explicitly blank rows**. Thin merging always chooses upward: **1,140 beds contain 78,393 rows absent from retained chunks**, sometimes retaining agreement 1.0. Same-label merging also removes independently detected contacts.
   **Fix:** preserve blank runs, contacts and all evidence; recompute labels/confidence after merges, then test decoding from full label costs. **3–5 h.**

6. **High — boundary summaries miss texture changes and displace edge contacts.**
   **Raised:** Aa, Sa, Ag, Sg. **CONFIRMED:** `BED:51–97`. A controlled vertical-to-diagonal motif change with identical row ink density produces **no boundary**. Colour contacts at rows **5/95** are detected at **13/86**.
   **Fix:** add texture-change evidence at several scales and truncated comparison bands near edges. **2–4 h.**

7. **High — damaged legend candidates cap every downstream method.**
   **Raised:** Aa, Sa, Ad, Sd, Ag, Sg. **CONFIRMED:** native cache plus inspected parser images. Callaway exposes **five compound entries containing multiple rocks**; Rio San Gabriel retains OCR fragments; Hadrian retains MDT and `I!S`. `LC:217` also rejects controlled legitimate phrases such as “Oil shale”; their current-corpus prevalence is unestablished.
   **Fix:** evaluate the existing v4 parser, repair swatch/caption ownership and entry roles, preserve printed names plus stable IDs, and replace substring vetoes with contextual filtering. **3–6 h.**

8. **High — caches prevent reliable unseen-log processing and experiment attribution.**
   **Raised:** Aa, Sa, Am, Sm, Af, Sf. **CONFIRMED:** `LC:154–188,220–250`; `BED:204–206`. A cache-missing identifier returns **an empty legend**. Configuration/image identities are omitted from relevant cache keys; existing bed outputs are skipped after algorithm changes.
   **Fix:** parse missing logs individually; fingerprint pixels, boxes, parser and configuration; bundle predictions and provenance immutably. **1–2 h.**

9. **High — confidence displays resubstitution agreement as calibration.**
   **Raised:** all ten. **CONFIRMED:** `CONF:34–58,72,110–116`. The current fit uses **454 selected high-sureness windows**, only **339 distinct beds across 47 logs**. The displayed **188/188** highest-band agreement uses fitting observations. Agreement is both an input and a subsequent multiplier.
   **Fix:** define human-correctness confidence; fit/calibrate on separate source groups, include all certainty levels, and validate unseen-family behaviour. **1–2 h**, excluding labels.

10. **High — review sampling and rendering restrict the claimed evidence.**
    **Raised:** Am, Sm, Af, Sf. **CONFIRMED:** `notes/llm_review.py:68–88,159–161`. Sampling contains **200 crops per occupied confidence band plus 100 mixed rejects**, but excludes the **11,754 classifier-blank windows**. All **147 D-582 originals**, widths 246–297, are clipped to 160 native pixels. Repeat agreement is **57/70**, or **66/70** after the documented Clay/Shale equivalence—not measured accuracy.
    **Fix:** show full-width crops and original legends; sample all states with recorded inclusion probabilities; human-audit judge errors. **1–2 h.**

11. **High — reference labels have limited independence and inconsistent semantics.**
    **Raised:** Am, Sm, Ad, Sd, Af, Sf. **CONFIRMED:** gold contains **17 intervals/6,006 rows**; root human truth is **three adjacent Clay decisions/92 rows**, with suggestions shown. Neel provides **646 intervals on two pages**, but its README:43–48 documents AI-applied amendments. Recomputed silver A/B overlap agreement is only **35,768/71,138 = 50.28%**; Rio Nueces A uses foreign legend names.
    **Fix:** retain human originals, AI proposals and adjudications separately; validate local vocabulary; collect blinded contiguous regions across families. **2–3 h** preparation, annotation below.

12. **High — synthetic truth is contaminated and unrepresentative.**
    **Raised:** Aa, Sa, Ad, Sd, Ag, Sg, Af, Sf. **CONFIRMED:** `ml/synth_v2/synthetic_index.json` contains **1,652 non-rock intervals/295,179 rows**, **3.13%** of filled rows. All **2,200 intervals under 48 px** terminate at the window edge; Neel’s pages have **85.4%/68.2%** beds under 48 px. D-582 and MC765 each reuse identical artwork across logs. Approved coverage is **50 crops/six patterns**; **453/472** texture profiles are unverified. `synth/statistics.py:83–127` omits human-label directories.
    **Fix:** repair identities first; split lineage before mining; ingest qualified human statistics; generate thin beds, realistic blanks and paired legend/column variation. **3–5 h.**

13. **Medium — review exports are not submission outputs.**
    **Raised:** Aa, Sa, Ad, Sd. **PARTLY:** `CONF:73–84` lacks the depth/cluster schema required by `SPEC.md:25–32` and omits **1,076 trailing grid rows**. A downstream submission adapter was not established.
    **Fix:** validate an explicit adapter, depth calibration, final tail handling and annotated-image export. **2–3 h.**

14. **Medium — external sources can duplicate each other and measure a different visual task.**
    **Raised:** Am, Sm, Af, Sf. **CONFIRMED:** KGS `386825/meta.json` and `at_005/386825/meta.json` share PDF hash/page; their metadata describes underlying-vector truth beneath overlays.
    **Fix:** deduplicate by source/well and audit visible versus underlying truth; report transfer separately. **1 h.**

15. **Medium — efficiency remains insufficiently demonstrated.**
    **Raised:** Aa, Sa. **PARTLY:** `harness/io.py:34–53` can fall back to whole-page decoding; `BED:94–96` and `CONF:27–30` contain repeated searches. Historical timings were not rerun.
    **Fix:** benchmark cold parsing, long-strip decoding, memory and inference; then index interval lookup and shortlist expensive matching. **1 h benchmarking; 2–3 h optimisation.**

**Reviewer disagreements and judgement**

No mutually exclusive core recommendations survived reconciliation. Boundary-first versus matcher-first proposals are priority differences; volume-weighted audits and equal-log scores estimate different quantities.

The substantive factual dispute is **Trail**: Ad could not confirm grid hallucination; Sd reported it. Native inspection and fresh classification support **Sd at rows 1424–2159**. This supports grid negatives, without invalidating correctly classified regions elsewhere.

Other apparent disagreements reconcile: gold products **0.0755/0.0821** use labelled-span/full evaluation windows; synthetic contamination **2.92%/3.13%** excludes/includes `Dry`.

**Consolidated evaluation/hill-climb protocol — ordered, ten lines**

1. Freeze exact labels, admissible equivalences, boundary tolerance and scorer version (**0.5 h**).
2. Group wells, scans, overlapping depths and derivative assets; hash manifests (**1–2 h**).
3. Keep existing 700 crops as development; reserve new protected sources for transfer claims (**0.5 h setup**).
4. Preserve the 13-window regression benchmark; add blinded random interval blocks (**0.5 h design; labels below**).
5. Select on mean per-log row-accuracy × boundary-F1; report family results separately (**1 h**).
6. Include blanks, unknowns, thin contacts, mixtures, failures and coverage diagnostics (**0.5–1 h**).
7. Fit confidence on separate grouped observations; assess held-out human correctness (**1–2 h**).
8. Predeclare useful gain, candidate budget and paired source-level uncertainty analysis (**1 h**).
9. Save every attempt’s configuration, predictions, costs and paired regressions (**0.5–1 h**).
10. Freeze the winner; open protected tests once; retire tests subsequently used for tuning (**0.5 h**).

**Bug-fix order**

| Rank | Deliverable | Hours |
|---:|---|---:|
| 1 | Correct legend candidates, local IDs and entry roles | 3–6 |
| 2 | Remove false twins; repair NCC coverage and coloured-ink features | 3–5 |
| 3 | Repair cropping, dark-fill handling and rejection | 2–4 |
| 4 | Preserve merge evidence, blank runs and drawn contacts | 3–5 |
| 5 | Repair texture/edge boundary detection | 2–4 |
| 6 | Add cache misses, fingerprints and immutable exports | 1–2 |
| 7 | Refit independent confidence; validate submission adapter | 3–5 |

**Generalisation plan**

| Rank | Experiment | Hours |
|---:|---|---:|
| 1 | Source-group and seven-family holdouts, excluding all derivative assets; call existing-family results retrospective | 2–3 setup |
| 2 | Compare automatic versus verified legends and predicted versus true boundaries to locate the bottleneck | 2–3 |
| 3 | Adapt photometry and motif scale per log; use depth calibration for duration priors | 2–4 |
| 4 | Test existing boundary CNN and direct-emission decoder with soft durations against repaired baseline | 4–8 |
| 5 | Test shared query/swatch metric learning with same-legend hard negatives | 4–8 |
| 6 | Time-box local DINO patch-token matching; defer OCR unless ambiguity remains | 2–4 |

Held-log legends are legitimate inference inputs. Report frozen inference separately from declared legend-only adaptation; count adaptation time.

**Data plan**

1. **First: repair reference semantics — 2–3 h.** Review seven complete legends, one per family; start with Callaway, Allegheny, Hadrian, D-582 and MC765 distinctions. Reconcile Neel/Hadrian versions and human-confirm AI amendments.
2. **Blinded pilot — 5–8 annotation h, plus 2 h tooling.** Collect **400 unique judgements**: 280 random audit positions (40/family), 70 random development positions, 50 targeted diagnoses; add **40 independent repeats**. Prioritise D-582 confusions, Geoservices giant-bed disagreements, false twins, blank grids and thin contacts.
3. **Geometry — included in that annotation allowance:** **28 complete 300–600-row strips**, four/family, divided between development and protected-label evaluation. Record every contact and blank, not only proposed beds. These are pilot counts: ±5-point binary accuracy needs approximately **385 independent observations/family**, more with clustering; this does not power the composite score.
4. **Synthetic — 3–5 h.** Fix contamination; derive statistics from qualified development labels; include 1/3/6/12/24-row beds, same-label contacts, independent grids, overlays and noisy blanks. Compare real-only versus real-plus-synthetic training at equal budgets.
5. **Outside data — 1 h local audit; zero collection hours initially.** Deduplicate existing KGS material and test one visual-compatibility hypothesis. Acquire new protected wells when required for genuine transfer evaluation.

**Appendix: refuted current claims**

- **REFUTED — high, Aa:** unlabelled `ok` beds and resulting sorting crash. `BED:157–163` now votes over non-OK states by rows; the controlled tie returns blank; current outputs contain **zero** invalid `ok` beds. **Remaining fix: 0 h.**
- **REFUTED — critical/high, Af, Am, Ad, Sd:** current stale/mismatched exports. All **51 JSON/CSV pairs now match**, with **15,992 intervals**, **15,186 named beds**, and **zero invalid CSV vocabulary rows**. Current confidence/comparison summaries agree. Historical observations may have been valid. **Remaining reconciliation: 0 h; provenance work remains above.**

Geological correctness, prospective transfer gains and full runtime remain unestablished. Row/boundary scores were replayed with NumPy after the full scorer import attempted a temporary-file operation that the read-only guard blocked.
