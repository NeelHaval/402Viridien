# 1. Verdict

Withholding unsupported tiles is sound; the recovery workflow needs revision before it can efficiently expand realistic training data.
The biggest problem is treating missing repeat evidence, random textures, damaged symbols and column/legend disagreements as the same remaking task.
Five trials support a donor-based Anhydrite recovery and a column-derived Sand correction; they do not support bulk approval of the 142 Failed entries.

# 2. Findings, ranked by impact

## 1. Entry counts hide how little column coverage this backlog can recover

**Severity: major.**

**Evidence.** The frozen board at **2026-11-08 03:16:09 UTC** contains 142 Failed, 2 Unsure and 1 Rejected training entries across 19 documents. All 145 have null `share_of_column_rows`: the board reads that field from the proposal, rather than calculating legend-entry weights (`data/genlab/fleet/board/board.py:304`).

I streamed the saved prefill, matched document names and labels using the lab's whitespace/case normalization, and recomputed the well split from its recorded group hash (`normalized.py:38`). This reproduces the live denominator of **1,969,941 guessed training rows**. Only **850 rows, 0.043149%**, belong to these entries:

| Priority by guessed training-row share | Entry | Rows | Share |
|---|---|---:|---:|
| 1 | `16c9064b18b77b61:5` Limestone | 826 | 0.041930% |
| 2 | `952caeadf1d8467c:7` Anhydrite | 24 | 0.001218% |
| Equal, diversity work | Other 143 entries | 0 | 0% |

These are **machine guesses, not occurrence truth**. Both nonzero entries' sampled guessed intervals fail the motif comparison: four Limestone probes and one Anhydrite probe score **0.000**. Thus 850 rows is a nominal coverage ceiling, not an expected real-data gain. I looked at the real/generated comparisons in `data/genlab/critique/s2-failed/test-1.png` and `test-2.png`; the guessed crops show other fills or boundaries.

The primary-cause classification, using every note and the six swatch contact sheets, is:

| Primary cause | Failed | Unsure/Rejected | Interpretation |
|---|---:|---:|---|
| Period/lattice underdetermined | 111 | 0 | Includes horizontal and combined-motif uncertainty, not just vertical spacing |
| Damage, lettering, or composite motifs | 14 | 0 | Includes Sand-SIS and the two mixed CONGL.SAND variants |
| Stochastic texture candidates | 17 | 0 | Seven Breccia, seven Conglomerate, Granite Wash, Fine sand, Salt-pepper sand |
| Explicit review conflict | 0 | 3 | Sand, Claystone and Calc. Dolomite |

This is a triage classification, not a claim that all 17 textures are statistically random. The complete per-entry notes, classifications and weights are in `data/genlab/critique/s2-failed/inventory.json`; the counting method is `data/genlab/critique/s2-failed/audit.py:33`. Coverage's existing weight definition is `genlab/score.py:224`.

**Fix.** Calculate per-document/per-entry weights in `score.coverage`, expose them on the board, and maintain separate queues for row coverage and class diversity. Persist the missing evidence and completed search results; rerun only when a new donor, source box or annotation arrives. Round IDs in this snapshot span 11–42, but do not by themselves establish how many attempts ran.

**Verify.** The board should reproduce the two weights above and explicitly mark zero-guess entries as unknown occurrence. A failed matching probe must not become a recovered geological row.

**Effort: S. Expected gain:** correct prioritization of 145 entries; prevents presenting class-count improvements as meaningful row-coverage gains.

## 2. The rejected Sand has usable column geometry; preserving legend spacing preserves the error

**Severity: major.**

**Evidence.** `data/genlab/proposals/5709fe9e86ce942b/legend-6.json:47` explicitly retains a **42-pixel** legend repeat despite all five listed column crops having a **14-pixel** repeat. The current-round reviewer rejects that difference (`legend-review-r12.json:3` in the same directory).

Trial 5 independently estimates a 14-pixel horizontal period in **all five** column crops. I extracted a 14 × 35 native hypothesis from the first crop and tested the other four:

| Model | RGB pixel match, all five crops | Ink Dice, all five crops |
|---|---:|---:|
| Rejected 42 × 35 tile | 0.8484–0.8755 | 0.3733–0.3791 |
| 14 × 35 column hypothesis | 1.0000 | 1.0000 |

The four additional crops were not used to construct the hypothesis. These are within-training-document checks, not held-out-well evaluation. Visually, the rejected strip has conspicuously wider dash gaps. Image looked at: `data/genlab/critique/s2-failed/test-5.png`; experiment: `data/genlab/critique/s2-failed/experiments.py:85`.

**Fix.** Request **one owner-labelled box** confirming which legend class these column dashes represent. If confirmed as Sand, retain the column's measured spacing and record the legend/column discrepancy instead of enforcing the legend template. Keep the other four crops for validation. The normalized snapshot contains **no owner-labelled record for any of the 145 target document/legend pairs**, so the supplied boxes do not establish class identity.

**Verify.** After class confirmation, a fresh reviewer should accept the native-width comparison, with approximately 1.0 RGB match and Dice on the four reserved crops.

**Effort: S plus one owner label. Expected gain:** one well-supported recovery among the three review conflicts; **zero additional saved-guessed rows**. Do not silently relabel other sand classes to increase this weight.

## 3. Borrowing can recover a small, measurable subset, but approved appearance is insufficient donor evidence

**Severity: major.**

**Evidence.** Existing borrowing accepts only human/verified donors, matching names, background distance ≤12, RGB agreement ≥0.92 and ink Dice ≥0.80 (`genlab/candidates.py:355`, `:373`). My sweep of current training tiles found **zero human/verified matches** passing those appearance gates, but **seven target entries** passing against agent-made or approved-swatch tiles. This measures appearance eligibility; it does not bypass existing tier, ambiguity or evidence rules.

Trial 1 supplies a concrete recovery: failed `952caeadf1d8467c:7` matches training donor `aeb414a4367bafd4:7` at **RGB 1.000 and Dice 1.000**. Its cited donor column box is only 22 rows high. Expanding read-only to **[1090,208692,1145,208756]** reveals 64 native rows: the whole crop matches at **1.000**, and the 29 overlapping rows separated by the proposed **35-pixel** period agree at **100%**. Failed `7966a10fba6fa984:11` also matches this donor at 1.000/1.000. Image looked at: `data/genlab/critique/s2-failed/test-1.png`; experiment: `experiments.py:55` in that directory.

Conversely, two of the seven high-scoring matches are Granite entries borrowing `17a317978b2c9139:14`. They reproduce the same **GR lettering** whose inclusion caused the targets to fail. The failed Granite note describes 250 unobserved phase pixels after lettering is masked (`data/genlab/proposals/38c9df683c1517bc/legend-14.json`, notes). Perfect swatch agreement cannot validate those hidden crosses. I looked at `data/genlab/critique/s2-failed/inventory-0.png` and the donor tile.

There is also a metric trap: 21 targets have a best same-name ink Dice ≥0.80, but **six best matches contain zero black-mask pixels on both sides**, returning 1.0 by definition (`genlab/candidates.py:141`, `:152`). Examples include red Tuff and orange Siltstone. Only **15** best matches contain nonempty black masks; seven already belong to the appearance-passing group, leaving **eight additional geometry leads**, not eight recoveries. Different native scales and colours remain unresolved.

**Fix.** Add an explicitly reviewed donor registry carrying real source boxes, measured axes, palette, scale and excluded overlays. Admit an agent-produced donor through that registry only after independent native-column evidence establishes its geometry. Start with the two Anhydrite targets; then investigate Sand-SIS, Others-A and Others-E. Preserve target colours and require an explicit native-scale fit before transferring a look-alike's period. Replace black-only symbol masks with palette-aware foreground masks; an empty foreground comparison should be unavailable, not evidence of matching symbols. Transfers recover appearance only; target thicknesses and transitions remain unmeasured.

**Verify.** Reproduce the expanded donor match, rerun both Anhydrite comparisons, and reject a GR-containing donor. Add a coloured-symbol case where the black masks are empty but the foreground motifs differ.

**Effort: M. Expected gain:** **two supported donor candidates**, with **three further candidates** if their donor evidence passes review; at most 24 saved-guessed rows, whose target interval currently mismatches. The remaining two of the seven are unsafe Granite transfers. The eight weaker geometry leads have no demonstrated recovery yield.

## 4. Random-pattern candidates need texture models, not proof of an exact lattice

**Severity: major.**

**Evidence.** Seventeen Failed entries fall into the texture-candidate group above. The Breccia failure, for example, demands a recoverable clast lattice (`data/genlab/proposals/28112eef4c3c645a/legend-21.json:16`). The current proposal loader represents every accepted proposal as a periodic tile, using image dimensions as both periods (`genlab/proposals.py:78`).

Trials 3–4 sample complete observed glyphs at native size with random placement and a one-pixel exclusion margin, stopping near the source ink fraction:

| Trial | Complete source glyphs | Source ink | Two generated ink fractions |
|---|---:|---:|---:|
| `28112eef4c3c645a:21` Breccia | 2 of 5 components | 18.779% | 19.200%, 18.900% |
| `57b1857d99779870:12` Fine sand | 25 of 31 components | 9.366% | 9.394%, 9.375% |

Both seed pairs produce different images without choosing a vertical period. I looked at `data/genlab/critique/s2-failed/test-3.png` and `test-4.png`, with real swatches and generated strips all enlarged exactly 3×. Fine sand is a credible texture hypothesis. Breccia visibly overuses two silhouettes and loses the source's faint background texture; density agreement alone is insufficient. Experiment: `data/genlab/critique/s2-failed/experiments.py:29`.

**Fix.** Support a stochastic proposal type in `proposals.load` and rendering: bootstrap glyph geometry and measured colours, fit density and nearest-neighbour distances, and model scan texture separately. For clasts, obtain more complete training glyphs before expanding shape diversity; do not invent the missing clipped outlines. Check autocorrelation before classifying a swatch as random, since irregular placement in one short crop does not prove randomness.

**Verify.** Compare multiple seeds against a fresh owner-labelled crop using component areas, outlines, spacing, palette and a blinded native-scale comparison. Confirm that longer outputs do not periodically repeat a short random seed tile.

**Effort: M. Expected gain:** **17 candidates for stochastic recovery**, with two previews demonstrated, **zero validated recoveries yet**, and zero saved-guessed-row gain. Fine sand is the strongest first candidate; Breccia needs more glyph evidence.

## 5. Symbol conventions supply priors; they cannot recover a document's missing pixel spacing

**Severity: major.**

**Evidence.** Fifty of the 111 period/lattice failures have ordinary named motifs suitable for a convention-based proposal: brick fills, salt grids, dots, dashes, triangles and related symbols. That is an eligible-preview count, not a success estimate. The [USGS/FGDC symbol standard](https://ngmdb.usgs.gov/fgdc_gds/geolsymstd/download.php) provides lithologic pattern families; applying their geometry to these scans remains an inference requiring document evidence.

Trial 2 demonstrates the ambiguity directly. Two diagnostic Limestone hypotheses with horizontal widths **54 and 108 pixels** each score **1.000** against the observed swatch while producing different joint spacing. Neither is an approved tile: both deliberately preserve the observed crop and expose the ambiguity and wrapping artifacts. All four sampled guessed column probes score **0.000**. I looked at `data/genlab/critique/s2-failed/test-2.png`; experiment: `data/genlab/critique/s2-failed/experiments.py:69`.

The existing seam implementation already says continuity is not proof of a geological repeat (`genlab/swatch.py:81`). Matching blank edges cannot recover unseen courses, and generic symbol conventions cannot decide whether a letter is repeated fill or an overprint.

**Fix.** Mark conventions as `prior-only`, with unknown axes and alternative spacings, and withhold them from the fidelity benchmark until verified. Truly useful owner requests are specific:

- **Sand `5709fe9e86ce942b:6`:** label one existing column box to resolve class identity, then use its measured geometry.
- **Unsure Claystone `5fbeaf2a0650e013:1007`:** one clean column box spanning repeated dash rows and stagger.
- **Unsure Calc. Dolomite `760ae0a5db33bede:31`:** one clean box exposing joints without the black rectangle; current swatch match is only **0.826265**, and there is no supplied column box (`legend-31.json:12` in that proposal directory).

For the 14 damaged/composite failures, request a box only if that motif actually occurs. It must contain complete motifs and enough extent to establish each missing axis. Where searches find no matching layer, request an **absent/unknown-occurrence decision** instead. One short owner box cannot manufacture an unobserved repeat.

**Verify.** Record observed versus borrowed versus assumed axes separately. A prior-only tile must never acquire measured-column status from a seamless preview or a high self-match.

**Effort: S for triage/schema, M for templates. Expected gain:** up to **50 conventional preview candidates**, **zero demonstrated faithful recoveries from conventions alone**. The three explicit review conflicts are the first owner-box requests; only Sand currently has a tested geometric correction. The 14 damaged/composite entries are additional evidence requests, not a promise that 14 boxes will recover them.

## 6. Cross-log recovery needs source lineage to prevent held-out-well leakage

**Severity: major.**

**Evidence.** The owner's legend groups contain **64 target-to-other-document member links to test wells**, all excluded before any donor image was read. Among the retained links, 21 links connect 14 targets to existing training cards. Results and excluded identities are recorded in `data/genlab/critique/s2-failed/lookalike-results.json`. These links are search leads, not proof of an interchangeable symbol.

The frozen generated cards contain **zero test-well targets**, and the board's training flags have **zero disagreements** with the split policy. Existing borrowing excludes test donors (`genlab/candidates.py:358`), and building excludes test targets (`genlab/build.py:93`).

However, proposal ingestion checks the tile's location and current-round approval, not the source well of its pixels or inferred period (`genlab/proposals.py:45`, `:78`). A concrete failure scenario is copying a test-well tile into a training proposal directory: its path and target split can pass while its source lineage is lost. **I found no evidence that this has happened.** The proposed borrowing expansion would make this missing lineage consequential.

**Fix.** Require donor document/legend IDs, real source boxes, source versions, group/split, and parameter lineage for transferred periods and glyph pools. Recheck every source group at ingestion and rebuild, including transitive donors. Use the owner legend service only for discovery; apply the train-only filter before image access or parameter fitting.

**Verify.** A fixture whose training proposal cites a test donor must be withheld even when its PNG is inside the training directory and its review is confident. Changing a donor's split or annotation must invalidate descendants.

**Effort: M. Expected gain:** blocks the demonstrated 64 held-out metadata leads from becoming generation inputs; no current leakage correction is claimed.

# 3. Quick wins

1. Populate and sort the backlog by guessed training-row weights; display the denominator and uncertainty.
2. Route the rejected Sand to one owner class-confirmation request with `test-5.png` attached as evidence.
3. Attach the expanded 64-row Anhydrite donor box to the two recovery candidates for review.
4. Assign all 145 entries an explicit recovery/evidence-needed status using the saved inventory.
5. Make empty black-mask Dice unavailable and retain RGB/palette checks for coloured motifs.

Each is under an hour of implementation or preparation; owner response time is excluded.

# 4. What is already good

Current-round review gating, explicit rejects and train-only target/donor rules correctly withhold unsupported work. Native-scale rendering and the existing quilt fallback for owner crops are useful foundations; preserve their provenance and do not turn machine proposals into evaluation truth.

# 5. Doubts

The 145-entry audit is a frozen snapshot; other agents continue changing proposals and code. The six contact sheets are triage evidence; the five individual trial sheets are the detailed comparisons. I used existing crop files or small windowed reads, never loaded an entire TIFF, and wrote only this report plus scratch evidence in its designated directory.

Zero guessed rows does not establish absence. I did not independently repeat every historical whole-column search, measure downstream classifier gains, or obtain owner labels. The donor sweep used current native tiles and explicit normalized names; the owner-group search broadened discovery but was not an exhaustive search over recolouring, scale changes or different names. Recovery estimates overlap and must not be summed: **two supported donor candidates; three more donor possibilities; eight weaker geometry leads; 17 stochastic candidates; 50 convention-only preview candidates; three priority owner confirmations.** Only new evidence can establish the remaining faithful-recovery yield. No conclusion here treats unfinished assembly or benchmarking work as a defect.
