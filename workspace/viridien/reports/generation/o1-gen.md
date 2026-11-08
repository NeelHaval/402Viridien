# o1-gen: fake pattern generation and the fake dataset

## 1. Verdict
- **The pattern side mostly works.** Most logs are vector-rendered with 2 colours, so a periodic repeat tile copied from real pixels is the right model. Using legend tiles alone, a plain pixel matcher picks the right legend entry for **65 of 79** held-out real boxes (82%; chance is 5.5%).
- **The dataset side is not yet fit to train a detector.** It has almost no variation inside a class. Of the "augmented" strips, 819/1048 train and 552/552 validation are pixel-identical to the clean ones. The held-out legend tiles skip the lab's seam gate, so 102/138 carry `seam_risk`.
- **Biggest problem:** the swatch-to-column transfer (period, seams, line weight, text in the box) is assumed exact and never varied. All 14 pixel-match failures are seam-risk legend tiles.

## 2. Findings (most impact first)

### F1. The dataset build reuses legend tiles that the lab would withhold (seams, text, mixed swatches). Severity: major
- **Evidence:**
  - `dataset.py:176-197` (`legend_pattern`) records `seam_risk` (line 189) but never gates on it. It also includes every legend entry (`dataset.py:294-297`), with no non-lithology exclusion.
  - In v4, **102/138** validation tiles have `seam_risk=True`. **All 14/14** boxes my pixel matcher gets wrong (`o1-gen/match.py`, `match.json`) use seam-risk tiles. Among the 65 correct boxes, 40 are seam-risk and 25 are not.
  - 14/90 coloured validation tiles contain whole near-white seam rows. The same check finds 0 of 242 coloured lab training tiles.
  - What `o1-gen/val-tiles.png` shows:
    - Text tiles: `5fbe:1008` "COAL", `7dc2:5` "SHALE COL", `7dc2:13` "MAEMAE", `7dc2:19` "CONGLOME".
    - Mixed swatches (the legend box spans several entries): `5fbe:0-4`.
    - White seam stripes: almost every `5fbe` tile.
    - Fluid/show symbols as candidates: `952c:21/22`, `aeb4:21/22`.
  - `o1-gen/zoom-claystone.png`: the 37×37 claystone tile wraps a frame remnant, so every period shows a doubled "=" dash that the real column never has. Claystone then ties with marl, 0.839 vs 0.841 (5 boxes lost).
  - `o1-gen/columns-real-vs-fake.png`: a v4 validation stacked column for 5fbeaf prints "COAL COAL COAL" as a layer, and its red layer has white seam lines.
- **Fix:**
  - Route `legend_pattern` through the same gates the lab uses: the seam test, non-lithology exclusions, and the near-white symbol rule.
  - For the remaining seam-risk entries, do not wrap the swatch interior. Use the legend-to-column period rule from F2.
  - Keep withheld entries in the candidate list as "unrenderable" so chance stays honest.
  - **Verify:** rerun `match.py` and the closed-set evaluation on a rebuilt v-next. Expect the claystone and 5fbeaf rows to move.
- **Effort:** S–M
- **Expected gain:** 5–9 boxes (+6–11 points of closed-set top-1), cleaner stacked columns, and no text taught as lithology.

### F2. The swatch-to-column transfer is treated as exact, but column period and line weight differ from the legend. Severity: major
- **Evidence:** `o1-gen/scale.py`, `scale.json` compare validation crops with their legend tiles (diagnostic only, no fitting).
  - Vertical period is off by more than 15% for 17% of classes (4 of 23). Smaller offsets are common:

    | Class | Real period | Tile period |
    |---|---:|---:|
    | 5fbe:1015 | 6 | 7 |
    | 5fbe:1023 | 6 | 7 |
    | 5fbe:1024 | 6 | 7 |
    | 5fbe:1000 | 12 | 14 |
    | 5fbe:1006 | 3 | 14 |
    | 5fbe:1021 | 3 | 7 |
    | d368:12 | 6 | 13 |
    | b9ba:19 | 8 | 9 |
    | b9ba:20 | 8 | 9 |

    The legend swatch's interior height includes a boundary row, so the wrapped tile is one row too tall per course.
  - Ink fraction (real ÷ tile), 90th percentile: 1.68. Examples: 7dc2:23 shale 0.167 vs 0.083, because the column prints 2-px dashes and the legend 1-px dashes (`o1-gen/scale-pairs.png`); 5fbe:1000 anhydrite 0.076 vs 0.015, because the column adds bedding lines the swatch lacks.
  - On the 2-colour vector logs (3515, 952c, aeb4) period and stroke match exactly. The problem is family-specific: scanned or older layouts.
  - The 142 Failed proposals ("swatch shows one course, so the vertical period can't be measured") are the same issue seen from the agent side.
- **Fix:**
  - Learn a per-family legend-to-column transfer on **training** docs that have both a column-learned tile and a swatch. That gives about 43 pairs in `state.json` (`tile_scores` has both column and legend_swatch for 43 cards). Record the period offset (e.g. interior−1), the stroke dilation and the colour shift.
  - Apply that rule to swatch-only and held-out entries.
  - Where the rule is uncertain, render the uncertainty: sample period in [interior−2, interior], stroke ±1 px dilate/erode, and ±10 RGB fill.
  - This also unblocks the "one course" Failed entries, without a hard measurement.
  - **Verify:** closed-set top-1 per family, and `scale.py` ratios on the training pairs (leave-one-doc-out).
- **Effort:** M
- **Expected gain:** fixes the scanned-family failures, which are 7 of the 14 misses (all of them in 5fbeaf), and recovers part of the 142 Failed entries.

### F3. The "augmented" arm is mostly a copy of clean, and its noise model is wrong for these documents. Severity: major
- **Evidence:**
  - In v4, 819/1048 train augmented strips are pixel-identical to their clean pair, and so are 552/552 validation strips (`augmentation_basis`: 1324 "unmeasured identity", 276 "document").
  - Every agent-made and swatch class (1168 + 528 of 2096 train strips) gets no augmentation at all (`dataset.py:410-419`: the pool comes only from measured crops of the same class).
  - Measured blur and noise are **0 at the 99th percentile** (981 measurements in `v4/realism.json`). `realism.py:46` estimates noise with MAD, and MAD collapses to 0 whenever more than half the pixels are exact.
  - Real crops on 21 of 27 docs have 2–12 unique colours; 5fbeaf, d368, 28112, b9ba and d2b5 have 18–2269.
  - The real nuisance is sparse rather than Gaussian: the share of pixels the tile cannot explain has median 5%, p90 16% and p99 31%. Generated nuisance is 0% for 78% of train strips, capped at 5% otherwise.
  - What that 5–30% actually is (`o1-gen/residuals.png`): the neighbouring layer inside the box, frame lines, and motif misfit (irregular sand dots not on the lattice). It is not scanner dirt.
- **Fix:** replace the per-class identity fallback with a **class-agnostic nuisance library** drawn from training docs of the same family:
  - partial neighbour layers at the top or bottom, 1–10 rows of another class from the same doc;
  - rails and frame lines at the measured offsets;
  - depth grid lines and text fragments cut from the non-column region of training pages;
  - for scanned families only, colour quantisation or JPEG blur;
  - stroke ±1 px.

  Draw the amounts from the measured `overlay_fraction` distribution (median 5%, p90 16%). Do not drive Gaussian noise from the MAD estimate; use a robust mean or a trimmed std instead.
  - **Verify:** the discriminator per family should fall toward 50%, and closed-set augmented accuracy must not drop.
- **Effort:** M
- **Expected gain:** this is what makes fakes useful for a detector. Today a CNN trained on them sees perfect, pixel-periodic tiles with zero contamination.

### F4. Repeat tiles are right for vector logs but too rigid for hand-placed or irregular fills. Severity: minor
- **Evidence:**
  - Training pairs reproduce exactly on vector docs: column-tile median score 1.0 vs legend-swatch 0.998 over 43 cards. In `residuals.png`, though, the 38c9 sandstone dots are almost all flagged as residual: the real dots are not on one lattice.
  - Only 2 of 389 cards use quilting (`learn.py:232-242`).
  - Real vertical phase is **random per layer** and not page-anchored: page-phase agreement is 2–5% on Hadrian's 290/199/157-box classes, and below 0.6 for all 16 classes with ≥3 boxes. Horizontal phase clusters in 2–3 modes. The generator's default "phase varies per layer" (`render.py:21-22`) is therefore correct, and `anchored=True` (13 cards) should stay rare.
- **Fix:** for dot/symbol fills whose residual covers more than 10% of a crop, add a "jittered lattice" renderer that places symbols cut from the tile with ±1–2 px jitter. Keep exact tiles for line, brick and dash fills.
- **Effort:** M
- **Expected gain:** realism on sand and silt classes. Low priority until F1–F3 are done.

### F5. Stacked columns have an unrealistic class prior and include agent and swatch tiles, despite the README. Severity: minor
- **Evidence:**
  - `dataset.py:536` builds `models` from all of a document's patterns. In v4, agent-made and swatch tiles fill most layers: for 4a056 the layer metadata shows 10 agent-made, 6 swatch and 1 human. The README says agent, borrowed and swatch sources are "excluded from stacked columns".
  - When no transition is known, the next layer is chosen uniformly (`render.py:66-68`).
  - Thickness and transitions mix in prefill guesses (`dataset.py:547-556`; about 71% correct on Hadrian).
  - Visible in `columns-real-vs-fake.png`: the real 5fbeaf column is mostly brown shale and yellow sand, while the fake cycles through 37 legend entries.
- **Fix:**
  - Either update the README or gate the sources.
  - Weight layer choice by observed per-document class frequency (human boxes first, prefill second), and fall back to family frequency rather than uniform.
- **Effort:** S
- **Expected gain:** detector priors and column-level metrics become meaningful.

### F6. Taxonomy: "name" is not a class, and the open-set number measures that. Severity: major for the evaluation, not for generation
- **Evidence:**
  - Open-set fake→real recognition is 4–11%, while closed-set within-document is 72–73% (matcher) and 82% (my pixel matcher).
  - The diag (`datasets/diag/report.md`) shows "anhydrite" has 6 colours across documents, so name centroids attract green shale.
  - The names include junk such as `claym`, `othersh` and `unlisted...` strings (v1 confusions).
- **Answer to the question:** the 4–11% reflects a **broken evaluation and representation**, not a broken pipeline. The fakes carry document-level identity well.
- **Fix:**
  - Report closed-set per document as the primary number.
  - Keep `doc:legend_id` as the class, with a curated normalized lithology name as a secondary head. A detector reads the page's own legend, so it needs swatch→column transfer, not cross-log names.
- **Effort:** S

## 3. Quick wins (each under an hour)
1. Gate `legend_pattern` on `seam_risk` and on the lab's non-lithology rules (`dataset.py:176-197`, `294-297`).
2. Add a text check on tiles: reject tiles whose connected components look like glyph rows. The 4 text tiles are listed in F1.
3. Rename the "augmented" variant when `augmentation_basis == 'unmeasured identity'`, or drop those rows from augmented-arm statistics.
4. Replace the MAD noise estimate in `realism.py:46` with a trimmed std on non-masked residuals.
5. Weight stacked-column layer choice by observed class frequency instead of uniform (`render.py:66-68`).

## 4. What is already good
- Tiles are copied from real pixels, not drawn: per-pixel median of period-aligned crops (`learn.py:191-317`). On vector logs they reproduce the column exactly.
- Random per-layer vertical phase matches the real data.
- Validation tiles come from the legend only, and the lab's own seam gate keeps training tiles free of whole near-white seam rows (0 of 242 coloured tiles). Do not loosen either.

## 5. Doubts
- The pixel-match metric (share of pixels within RGB distance 45) is lenient on mostly-white fills. d368 shale scores 0.61 for the true tile vs 0.63 for marlstone. An ink-weighted F1 would be fairer, and I did not rerun with it.
- The period and ink ratios cover 25 validation classes in 7 wells. The family rule in F2 is a hypothesis that needs a leave-one-doc-out check on training pairs.
- I did not open the 142 Failed proposals one by one, I did not check whether the assembler (`compose.py`/`noise.py`) already adds the nuisances proposed in F3, and I saw no owner Real/Wrong votes.
- Scratch scripts and images are in `data/genlab/critique/o1-gen/`: `match.py`, `scale.py`, `montage.py`, `pair.py`, `val-failures.png`, `val-tiles.png`, `zoom-*.png`, `scale-pairs.png`, `columns-real-vs-fake.png`, `reconstructed.png`, `residuals.png`.
