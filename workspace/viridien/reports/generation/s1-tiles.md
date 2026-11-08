## 1. Verdict
Useful for supervised exploration; not ready for automatic acceptance of column-training textures.
The biggest problem is that acceptance and realism checks can miss damaged symbols, embedded rails, and absent column evidence.
The sampled machine-verified tiles are strong; confidence should attach to demonstrated fidelity, not source or reviewer approval alone.

## 2. Findings, ranked by impact

**Audit basis.** Snapshot built 2026-11-08 03:07:57 UTC: 389 cards. I deliberately selected 30 across sources, documents, methods, and score extremes; this is a diagnostic sample, not an unbiased defect-rate estimate. I rendered and inspected all six `data/genlab/critique/s1-tiles/sheet-*.png` sheets: native tile, generated strip, supplied real crops, and legend, all at exactly 3x nearest-neighbour scale. Unlisted patterns have no legend. Crops used `learn.Crops(model.Images())`, with its cache redirected into this audit's scratch directory; no whole TIFF page was loaded.

Numbers below are medians of per-card median scores against the **supplied reference**, after the existing crop cleaning. Pixel match uses RGB distance <45 and best cyclic alignment; ink Dice uses the existing black-pixel mask. These are resemblance measurements, not classifier accuracy or independent validation.

| Source | All cards | Inspected | Cards with target-column references | Pixel match | Ink Dice |
|---|---:|---:|---:|---:|---:|
| Human | 65 | 6 | 6 | .946 | .937 |
| Machine-verified | 9 | 6 | 6 | .990 | .986 |
| Borrowed | 1 | 1 | 0; donor column instead | 1.000 | 1.000 |
| Swatch | 70 | 6 | 0; legend instead | .999 | 1.000 |
| Agent-made | 228 | 7 | 3; other four use legends | .943 | .933 |
| Unlisted | 16 | 4 | 4 | .996 | .990 |

Reproduction: `python -B -X utf8 data/genlab/critique/s1-tiles/audit.py`, then `diagnostics.py` in that directory. Snapshot, individual images, measurements, and diagnostic results are preserved there. Reference types must accompany these scores: swatch/donor scores do not measure target-column fidelity, and three inspected coloured patterns have no pixels in the black-ink mask.

### F1. Fidelity scores can approve missing symbols, while most cards have no realism score

**Severity: major.** Replacing each tile with its uniform background still scores >=.85 on **10/30** supplied references. Examples: Siltstone `c524c8e761b99c48:2` scores .950 with every dot removed; Fine sand `aea797844b2c0ce1:11` scores .914 blank. The swatch-selection threshold is .85 (`genlab/selection.py:38`). The auxiliary symbol metric also misses coloured marks: RECRYSTALLIZEDCALC, reconstructed ANHYDRITE, and the pale-blue unlisted bands each give **Dice 1.000 for a blank tile**, because `genlab/candidates.py:141` masks only pixels whose maximum RGB channel is <80.

Only 72/389 cards have realism scores; 64 reuse training crops and eight claim held-out crops (`snapshot.json:47`). `genlab/build.py:192` omits this check for agent-made, unlisted, borrowed, and swatch sources. A concrete admitted weak match is `d2b5e2c76cfa887c:u0`: its proposal reports column match .734; I measured median .762 pixel match and .582 black-ink Dice across five displayed crops. The generated X field visibly lacks the real crops' horizontal dropout bands. This is a scan-fidelity gap; it does not establish that the X glyph itself is wrong.

**Evidence:** `data/genlab/critique/s1-tiles/diagnostics.json:1659`; `measurements.json:2662` in the same directory. Images inspected: `s1-tiles/sheet-3.png`, `sheet-4.png`, `sheet-6.png` under this report's directory. Proposal acceptance checks made/confident status, not measured fidelity (`genlab/proposals.py:57`).

**Fix:** Score every source against its correctly typed reference. Add a blank-baseline improvement, background-relative coloured-mark overlap, and stroke/spacing checks; calibrate thresholds against owner Real/Wrong votes. Report clean-texture and scan-appearance fidelity separately so intentional dropout removal cannot silently count as realistic scanning. Verify that all ten blank controls fail and coloured blank controls score zero foreground overlap.

**Effort: M. Expected gain:** Stops background agreement and missing metrics from masquerading as pattern fidelity.

### F2. An antialiased frame rail becomes an interior texture feature

**Severity: major.** Human Shale gray `5fbeaf2a0650e013:1024` retains a continuous gray left-edge rail in its 59x12 tile: most rail pixels are RGB 119, beyond the <80 cleaning threshold (`genlab/learn.py:96`; `genlab/swatch.py:25`). Its generated sample has both the intentional frame at x=0 and an unwanted continuous dark column at **x=12**. Rendering the tile without any intentional frame produces an interior rail in **8/8 seeds**, at x=23,16,1,50,41,33,17,22. The model has no measured horizontal period, yet `genlab/render.py:22` randomizes horizontal phase and wraps it.

**Evidence:** `data/genlab/critique/s1-tiles/diagnostics.json:1934`; inspected `s1-tiles/human-defects.png` and `s1-tiles/rail-phases.png`.

**Fix:** Extend boundary detection to antialiased, background-relative neutral rails, retaining the existing protection for genuine striped fills. Represent x/y anchoring separately; a lane with no demonstrated horizontal repeat must not receive arbitrary horizontal phase or periodic wrapping. Verify this tile loses its neutral edge rail and no interior rail appears across the same eight seeds; retain the genuine horizontal Shale stripes.

**Effort: M. Expected gain:** Removes a strong false column-divider cue from synthetic training strips.

### F3. Median voting invents gray strokes and weakens native motifs

**Severity: major.** Human Clay/Shale `c524c8e761b99c48:0` has only RGB 0 and 192 in all three displayed source crops. Its tile introduces RGB **96 in 2.25% of pixels**, visibly creating recurring gray double-dashes. This follows the per-block/per-pixel median operations (`genlab/learn.py:120`, `genlab/learn.py:298`). Human CHALK `d368760651ded938:15` has tile black-ink occupancy .089 versus .153/.141 in its two crops. Its second crop scores **.952 pixel match but only .561 ink Dice**; the first scores .700/.447. The 12-pixel repeat cannot reproduce the observed drifting scan geometry: even the first crop's self-match at lag 12 is only .734.

**Evidence:** `data/genlab/critique/s1-tiles/measurements.json:64`, `measurements.json:445`, and `diagnostics.json:1841`. Inspected `s1-tiles/human-defects.png` and `s1-tiles/sheet-1.png`.

**Fix:** Choose a source-supported RGB medoid/mode instead of interpolating between black and background; validate complete motifs after voting. Require multi-cycle foreground agreement before forcing one integer repeat, falling back to a longer native patch/quilt when scan distortion defeats it. Verify the Clay/Shale output introduces no unsupported gray and the Chalk output preserves observed joint strokes across both crops. Do not simply assume a longer period is correct.

**Effort: M. Expected gain:** Preserves symbol contrast and topology instead of periodically replicating voting errors.

### F4. Different crop IDs are incorrectly treated as independent evidence

**Severity: major.** Fine sandstone `7966a10fba6fa984:1` advertises four held-out crops and perfect realism. **All four overlap learning crops; three are fully contained**, and the fourth shares 27/35 rows. For example, held crop y=1588..1616 lies inside training crop y=1578..1672 at identical x. `genlab/score.py:184` excludes IDs, not overlapping pixels. Selection also scores every crop before choosing the generator (`genlab/selection.py:35`), so the subsequent held-out set can already have influenced that choice.

**Evidence:** `data/genlab/critique/s1-tiles/diagnostics.json:171`; displayed overlapping evidence inspected in `s1-tiles/sheet-2.png`. I checked all eight cards claiming independent scores; this overlap failure occurred in one of them.

**Fix:** Reserve disjoint depth intervals before learning and model selection, with a guard band at least one repeat high. Exclude intersecting pixel rectangles, not just duplicate IDs, and report independent depth-range counts. Verify the four overlapping crops cease counting as held out and neither tile selection nor period estimation accesses the reserved intervals.

**Effort: M. Expected gain:** Makes realism scores useful for deciding which tiles generalize beyond their source pixels.

### F5. Legend fidelity is being used without evidence of column fidelity

**Severity: major.** **201/228 agent-made cards have swatch-only evidence**, alongside all 70 swatch cards; the borrowed card supplies donor evidence. Thus 272/389 cards lack supplied target-column evidence. All six inspected swatch cards' real panes are their legend boxes; four of seven agent-made cards do the same. This fallback is explicit in `genlab/proposals.py:63`; borrowed panes switch documents in `genlab/build.py:313`.

The distinction matters measurably: column-derived Clay Shale `926182367112f14a:0` matches its column crops at median **.975**, but its legend at **.529 pixel match / .278 ink Dice**. This card chose the useful column texture; a faithful legend reconstruction would not establish the same native column pattern. Conversely, the .999 sampled swatch score in the table establishes only legend resemblance.

**Evidence:** `data/genlab/critique/s1-tiles/snapshot.json:49` and the preserved reference boxes in `measurements.json`; inspected `s1-tiles/sheet-3.png`, `sheet-4.png`, `sheet-5.png`.

**Fix:** Keep legend-only patterns explicitly provisional for column training. Show target column, donor column, and legend as distinct reference types, and promote only after checking native spacing and motif identity against target-column evidence. Where the motif is absent or the swatch is one course high, retain uncertainty instead of inventing a measured vertical period. Verify that the 272 unsupported cards cannot be counted as validated column coverage.

**Effort: M. Expected gain:** Prevents high legend scores from hiding the legend-to-column domain gap.

### F6. Existing review sheets change pixel scale between evidence and fakes

**Severity: minor.** `genlab/candidates.py:447` resizes every real crop to the swatch's display width; previews are generated at 64 native pixels and resized to that same width (`genlab/candidates.py:449`, `genlab/candidates.py:452`). For the inspected 47-pixel-wide Clay/Shale column, real pixels are magnified **64/47 = 1.36 times** as much as generated pixels. The sheet therefore makes correct native spacing appear different.

**Evidence:** Inspected original review row, copied to `data/genlab/critique/s1-tiles/existing-review-row.png`, versus the fixed-scale `s1-tiles/sheet-1.png`.

**Fix:** Render every pane at a common integer pixel scale, pad unequal widths, and print native dimensions. Verify a one-pixel mark occupies the same display width in legend, evidence, and generated panes.

**Effort: S. Expected gain:** Makes period and scale judgments trustworthy during manual review.

## 3. Quick wins

- Label every real pane as target column, donor column, or legend; print the document ID.
- Change candidate sheets to fixed 3x pixel scale with padded panes.
- Add blank-baseline pixel and background-relative foreground scores to the review display.
- Stop counting overlapping rectangles as independent held-out crops; expose the overlap count.
- Temporarily withhold `5fbeaf2a0650e013:1024` pending removal of its reproduced interior rail.

Each is scoped to under an hour; the broader acceptance calibration and voting changes are not.

## 4. What is already good

Native-scale colours and motifs are excellent in the six sampled machine-verified cards; preserve the exclusive-match and ambiguity safeguards.
Source provenance, explicit reused-training flags, and current-round proposal approval are useful safeguards.
The snapshot has zero test-well cards and zero test-well borrowing donors under `data/normalized/split-policy.json`; preserve `genlab/build.py:93` and `genlab/build.py:119`.

## 5. Doubts

I did not establish target-column fidelity for legend-only entries, or a true vertical period from one-course swatches. No definite wrong lithology identity was established in these 30 cards. The sparse/coloured foreground masks limit the reported Dice scores. Owner labels and proposals are changing concurrently; the saved snapshot defines these counts. I did not rerun the full classifier evaluation or assess whether the unfinished assembler/noise work will restore the observed scan artifacts. Training-crop overlap above is separate from well splitting; no test-well leakage was found in the audited cards/donor links, but arbitrary proposal scripts' historical inputs were not traced.
