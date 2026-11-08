# Diagnosis and v1 to v4 controlled comparison

The low v1 recognition score was primarily a mismatch of task and feature representation, not a wrong page frame. V4 freezes v1 wells, 857 accepted boxes (79 validation), and 262 training tiles. Later lab edits are not in this comparison. Validation candidates are all 138 legend entries from seven documents. Both versions use 552 fakes per closed-set arm and 1,048 per open-set arm.

## Root causes and controlled evidence

- Cross-log names do not define a unique appearance. Six training anhydrite entries are red, magenta, pink and brown; all have 0% white pixels. Averaging their 64-bin colour histograms and repeat features produces a centroid that attracts unrelated green shale. The first green shale crop has squared distance 83.019 to anhydrite versus 96.390 to shale. Retaining document prototypes instead of averaging by name changes cross-log clean matcher from 7/79 to 29/79 and shale-to-anhydrite from 25 to 0. That diagnostic classifier is not substituted for the reported matcher. See anhydrite-training-prototypes.png.

- The original logistic input resizes all RGB boxes to 12x20, regardless of native height or aspect ratio. Median real dimensions are 70x89; fake strips were 240 rows. The sheets show thin black motif strokes becoming blurred colour bands. Keeping v1 samples fixed but using native colour/repeat features changes open-set logistic clean 8/79 to 29/79, augmented 4/79 to 32/79; closed-set clean 3/79 to 47/79, augmented 20/79 to 45/79. This isolates the representation fix from sample changes. V4 adopts these native features for recognition; the original RGB results remain in legacy_rgb_baseline. Discriminator and paired legend/column ablations retain their original representation.

- Held-well tiles were contaminated by column evidence: v1 has 21 validation classes marked column, 37 agent proposal, and only 8 legend swatch. Tile-source labels alone cannot prove the agent proposals swatch-only. V4 opens neither lab validation tiles nor quilts: all 138 validation classes are rebuilt from their legend boxes, and no validation residual field is used. The audit independently reconstructs all saved validation models from source-page legend pixels; a regression test changes real validation pixels and verifies unchanged tiles.

- V1 omitted accepted classes from its generated inventory: halite 3515004b6f643c87:2 (5 boxes), limestone 952caeadf1d8467c:5 (2 boxes). V4 includes all legend entries. The failure-04 sheet explicitly marks its true-class fake as a new swatch-derived reference, because no v1 fake existed.

- Old augmentation pooled residual RGB across lithologies and pasted up to 35% disagreement as corruption. Sheets 05/07/09 show incompatible lines/colour crossing motifs. V4 uses only same-class residuals for strips, caps residual density at 5%, blur at 0.7, noise at 6 and RGB drift at 12, and uses no residual pasting in stacked columns. Window heights vary 48/96/144/240 at native motif scale (scale factor 1.0). No evidence justified geometric rescaling. With native features held fixed, the combined sample changes change closed logistic clean 47/79 to 44/79 and augmented 45/79 to 52/79; open clean 29/79 to 27/79, augmented 32/79 to 31/79. These changes help the augmented within-log task, not every arm.

## Crop inspection

Fresh page-reader RGB equals the saved evaluation crop in 79/79 cases. All boxes are inside their documented column bounds. Width min/median/max = 49/70/95; height = 12/89/155. No crop hits the 480-row cap. There is no BGR/RGB or letterbox-padding mismatch: both real and fake inputs are RGB, and the original evaluator reads native file, not normalized_png. Native motif size is preserved until the old logistic resize.

Dark-column and dark-row occupancy flags occur in 73/79 and 30/79 boxes respectively. These are flags for inspection, not proof that every line is a frame: some horizontal strokes are the motif itself. I inspected all ten sheets: rails/occasional horizontal frame lines are visible; no obvious text or neighbouring lithology dominates the ten examples. There is no semantic OCR/mixed-layer ground truth for the remaining boxes. Removing frames leaves open matcher 7/79 and hurts raw-RGB logistic (8/79 to 1/79), so frame trimming was not adopted as a blanket fix.

True-class cyclic RGB match is median 0.95084 across the 72 boxes with a v1 true-class tile; seven have none. This is an upper-bound reconstruction check using tiles that may have learned from those crops, not held-out recognition. Zero of the 79 boxes is >95% white; anhydrite is not a blank sink.

## Before / after

V1 had no closed-set score. Its closed-set rows below are retrospective: the same swatch-only candidates, legacy 240-row fakes and original classifiers; augmented nuisances come from training wells only. V4 uses the new heights and native recognition features. No validation crop creates either version's closed-set candidates.

| Task | Model | Samples | Fake n per arm | Real n | V1 correct / top-1 % / macro % | V4 correct / top-1 % / macro % | Chance % |
|---|---|---|---:|---:|---|---|---:|
| Closed-set within document | matcher | clean | 552 | 79 | 57 / 72.152 / 68.0 | 57 / 72.152 / 68.0 | 5.543 |
| Closed-set within document | matcher | augmented | 552 | 79 | 58 / 73.418 / 69.0 | 58 / 73.418 / 69.0 | 5.543 |
| Closed-set within document | logistic | clean | 552 | 79 | 3 / 3.797 / 4.727 | 44 / 55.696 / 54.212 | 5.543 |
| Closed-set within document | logistic | augmented | 552 | 79 | 20 / 25.316 / 20.0 | 52 / 65.823 / 56.067 | 5.543 |
| Cross-log name generalisation | matcher | clean | 1048 | 79 | 7 / 8.861 / 10.863 | 7 / 8.861 / 10.863 | 1.187 |
| Cross-log name generalisation | matcher | augmented | 1048 | 79 | 7 / 8.861 / 10.863 | 7 / 8.861 / 10.863 | 1.187 |
| Cross-log name generalisation | logistic | clean | 1048 | 79 | 8 / 10.127 / 20.312 | 27 / 34.177 / 20.136 | 1.187 |
| Cross-log name generalisation | logistic | augmented | 1048 | 79 | 4 / 5.063 / 4.545 | 31 / 39.241 / 26.833 | 1.187 |

Closed-set macro averages all 25 observed document classes / 79 boxes; its macro chance is 5.347%. Candidate counts by document are 15, 37, 14, 13, 13, 23, 23. Box-weighted chance is 5.543%. Open-set macro retains the original 8 names / 71 boxes with at least three examples; top-1 includes all 79. There are 80 training names: 1/80=1.25% chance on the 75 boxes with seen names, and zero on four unseen-name boxes, giving 1.187% overall.

## Per-class failures (percent correct; all observed classes)

| Document:legend ID | Name | n | V1 open matcher | V4 closed matcher clean / aug | V4 closed logistic clean / aug | Evidence / next action |
|---|---|---:|---:|---|---|---|
| 3515004b6f643c87:1 | sandstone | 3 | 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | Known-legend task resolves cross-log confusion |
| 3515004b6f643c87:2 | halite | 5 | 0.0 | 0.0 / 0.0 | 100.0 / 100.0 | Missing v1 class; now swatch-only legend coverage |
| 3515004b6f643c87:3 | shale | 8 | 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | Known-legend task resolves cross-log confusion |
| 5fbeaf2a0650e013:1000 | anhydrite | 1 | 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | Known-legend task resolves cross-log confusion |
| 5fbeaf2a0650e013:1006 | claychokesand | 1 | 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | Inspect swatch-to-column motif/scale and similar legend entries; no validation-driven tile fitting |
| 5fbeaf2a0650e013:1015 | limestone | 3 | 0.0 | 100.0 / 100.0 | 33.333 / 66.667 | Matcher recognises motif; phase/repeat baseline or few samples remain limiting |
| 5fbeaf2a0650e013:1021 | sandstone | 1 | 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | Inspect swatch-to-column motif/scale and similar legend entries; no validation-driven tile fitting |
| 5fbeaf2a0650e013:1023 | shalecolored | 3 | 0.0 | 0.0 / 0.0 | 0.0 / 33.333 | Inspect swatch-to-column motif/scale and similar legend entries; no validation-driven tile fitting |
| 5fbeaf2a0650e013:1024 | shalegray | 2 | 50.0 | 0.0 / 0.0 | 0.0 / 0.0 | Inspect swatch-to-column motif/scale and similar legend entries; no validation-driven tile fitting |
| 5fbeaf2a0650e013:1028 | siltyshale | 3 | 0.0 | 100.0 / 100.0 | 33.333 / 0.0 | Matcher recognises motif; phase/repeat baseline or few samples remain limiting |
| 7dc28eda27f5c429:23 | shale | 6 | 0.0 | 100.0 / 100.0 | 0.0 / 66.667 | Matcher recognises motif; phase/repeat baseline or few samples remain limiting |
| 952caeadf1d8467c:0 | claystone | 4 | 0.0 | 0.0 / 25.0 | 25.0 / 75.0 | Inspect swatch-to-column motif/scale and similar legend entries; no validation-driven tile fitting |
| 952caeadf1d8467c:1 | shale | 1 | 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | Known-legend task resolves cross-log confusion |
| 952caeadf1d8467c:2 | marl | 1 | 0.0 | 100.0 / 100.0 | 0.0 / 0.0 | Matcher recognises motif; phase/repeat baseline or few samples remain limiting |
| 952caeadf1d8467c:5 | limestone | 2 | 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | Missing v1 class; now swatch-only legend coverage |
| 952caeadf1d8467c:8 | halite | 4 | 0.0 | 100.0 / 100.0 | 0.0 / 0.0 | Matcher recognises motif; phase/repeat baseline or few samples remain limiting |
| aeb414a4367bafd4:0 | claystone | 4 | 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | Inspect swatch-to-column motif/scale and similar legend entries; no validation-driven tile fitting |
| aeb414a4367bafd4:4 | siltstone | 1 | 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | Known-legend task resolves cross-log confusion |
| aeb414a4367bafd4:8 | halite | 2 | 0.0 | 100.0 / 100.0 | 100.0 / 0.0 | Matcher recognises motif; phase/repeat baseline or few samples remain limiting |
| b9ba393ab59601f2:19 | limestone | 3 | 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | Known-legend task resolves cross-log confusion |
| b9ba393ab59601f2:20 | shale | 11 | 0.0 | 100.0 / 100.0 | 63.636 / 100.0 | Known-legend task resolves cross-log confusion |
| d368760651ded938:12 | marlstone | 1 | 0.0 | 100.0 / 100.0 | 100.0 / 0.0 | Matcher recognises motif; phase/repeat baseline or few samples remain limiting |
| d368760651ded938:15 | chalk | 2 | 0.0 | 100.0 / 100.0 | 100.0 / 100.0 | Known-legend task resolves cross-log confusion |
| d368760651ded938:20 | shale | 2 | 0.0 | 0.0 / 0.0 | 0.0 / 100.0 | Inspect swatch-to-column motif/scale and similar legend entries; no validation-driven tile fitting |
| d368760651ded938:9 | siltstone | 5 | 100.0 | 100.0 / 100.0 | 100.0 / 60.0 | Matcher recognises motif; phase/repeat baseline or few samples remain limiting |

## Discriminator (50% chance)

| Version / arm | Train real + fake | Held real + fake | Accuracy % |
|---|---:|---:|---:|
| v1 / clean | 187 + 187 | 76 + 76 | 80.263 |
| v1 / augmented | 187 + 187 | 76 + 76 | 63.158 |
| v4 / clean | 187 + 187 | 79 + 79 | 70.253 |
| v4 / augmented | 187 + 187 | 79 + 79 | 55.696 |

The discriminator comparison is not a perfectly paired causal estimate: v4 supports three additional held real/fake pairs, replaces contaminated validation patterns with swatches, and changes sample windows and augmentation together. The held source categories are now all swatches, so per-source discriminator arms for human/agent-made have no held samples and correctly say too few.

## Verification

- python -X utf8 -m genlab.dataset build --baseline v1: PASS build v4 patterns 400 samples 3624.
- python -X utf8 -m genlab.dataset eval v4: PASS eval; both tasks saved. The original evaluator has only a two-line delegation to the extension in dataset.py, so the detached loop uses it too.
- python -X utf8 -m pytest genlab/tests/test_genlab.py -q: 39 passed in 43.94s.
- python -X utf8 genlab/tests/audit_dataset.py v4: PASS zero test samples; 276 independently reconstructed validation models; zero validation-crop leakage; 3624 hashes; 64 column masks; ZIP integrity; HTTP 200.
- python -X utf8 data/genlab/datasets/diag/check_tab.py: PASS desktop and phone, both tasks/chance, no horizontal overflow, no JS errors. Screenshots inspected alongside all ten failure sheets.

## Doubts and remaining work

Seven wells and 25 classes are a small correlated sample. Eight observed document classes have fewer than three boxes; reported closed-set macro includes them with low-n flags. Small or ambiguous swatches, colour-histogram collisions and native repeat estimates still fail. The 5% corruption cap and window heights are conservative proposals, not proven optima; native-feature open-set accuracy does not improve from these generation changes alone. No owner Real/Wrong votes exist. OCR and neighbour contamination were inspected visually only for the ten sheets. Cyclic reconstruction scores can be inflated by source reuse. V4 is paired to v1, while publication compares to live v3 and correctly flags changed human labels; promotion thresholds remain macro drop >2 points or discriminator rise >5 points. The tab displays the newest evaluated version separately from the promoted pointer.
