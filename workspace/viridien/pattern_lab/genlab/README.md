# Pattern generator lab

http://127.0.0.1:8781. Start it with `..\start-genlab.ps1` (add `-Restart` after code changes).

It checks the labeller's `data/reviews/*.json`, `data/normalized/current.json`,
candidate/verdict reports and `data/genlab/proposals/*/*.json` every 3 s.
Human sources must have a current normalized training record with no quality
hold. Test groups are excluded. Label and proposal changes rebuild only the
affected documents. Timing depends on the document and whether candidate
rescanning is needed; a live proposal update rebuilt one document in 2.3 s,
and the first build with 161 agent-made patterns took 63.3 s.
The lab never writes to the labeller's files. Everything it makes goes to `data/genlab/`.

Run `python -X utf8 -m genlab.candidates` to scan every document, or add `--doc ID`
to refresh one. Reports and review sheets live in `data/genlab/candidates/`, with
overall and per-document counts in `summary.json`. The build reads these cached
reports; it refreshes an existing report only when that document's accepted
training boxes or legend change. A missing report is scanned by the explicit command.

Each column probe is matched against every swatch in its own document, with
cyclic phase alignment and RGB distance below 45. Verification requires at least
90% matching pixels, an 8-point lead over the next entry, and three matches across
two non-overlapping depth ranges. Suggestions only choose places to look. A native
central lane of up to 64 pixels avoids edge annotations; 12 windows span the column,
plus up to 28 suggested intervals per entry. Evidence boxes record the compared lane.
The final machine model uses the same voting and suspect rejection as human models.

Tiers are `human` (accepted single-label training boxes), `verified` (exclusive column
evidence), `borrowed` (a matching learned human/verified tile from another document),
`swatch` (a clean swatch with passing continuity tests on both axes), `ambiguous`
(another entry matches at least 95% in both directions), `no`, and `not-lithology`.
Human approval takes priority over ambiguity; ambiguous machine entries are withheld
and list their twins. Non-lithology exclusions take priority over every source.
Human, verified, and borrowed patterns enter generation unless rejected. Swatch
patterns require a `confident` verdict; absent, unsure, and reject verdicts withhold them.
The labeller's current well split still applies: test wells remain in the candidate
reports for inspection, but test wells cannot generate or donate tiles.

Optional `data/genlab/verdicts/<doc_id>.json` files map legend IDs to
`{"verdict": "confident|unsure|reject", "note": "..."}`. Reject excludes every source;
unsure retains human, verified, and borrowed patterns with a visible flag. Candidate
and verdict edits trigger the server's normal rebuild. Cards and manifest entries
record `source` as `human`, `machine-verified`, `borrowed`, or `swatch`. The page can
filter by source. Borrowed records carry the donor document/legend IDs, native tile
file, colour match and donor evidence. Column checks use human boxes
only; machine evidence is not evaluation truth. For machine-only columns, sampled
crop heights describe the synthetic layers, not measured geological thicknesses.
Borrowed/swatches have no target-column evidence and do not establish layer
thicknesses or transitions for stacked columns.

## Agent fleet proposals

The lab also watches `data/genlab/proposals/*/*.json` (full relative name,
nanosecond mtime, and size). Fleet files are read only. A made proposal is
eligible only when its ID has a **confident** verdict in the review file for
the proposal's current `round`: `legend-review-rN.json` or
`unlisted-review-rN.json`. A new round immediately withholds its tile until
that round is approved; an old confident review cannot approve a remake.
Missing, unsure, rejected, skipped, failed, and incomplete proposals are withheld.

Source priority per document/legend entry is **human > machine-verified >
agent-made > borrowed > swatch**. Existing owner rejects and non-lithology
exclusions still apply. Test wells cannot generate, including from proposals.
Approved tiles use the same native RGB repeat renderer and 224 × 224
normalization/mask pipeline as the other tile sources. Agent sources have no
measured thicknesses and are excluded from stacked columns, like borrowed and
swatch sources. They do not become human evaluation truth or borrowing donors.

Unlisted proposals use string legend IDs (`u0`, `u1`, …), labels
`Unlisted: <description>`, and document-specific class IDs (`<doc_id>:u<n>`).
Their manifest records retain `looks_like`, `method`, `proposal_round`,
`pattern_id`, the review verdict/note, source boxes and normalization metadata.
The descriptive `looks_like` field never merges them into a legend class.
Cards show up to five real column crops, or the legend swatch when supplied
column boxes are absent, and identify the independent reviewer and method.

The source selector and header include **Agent-made** and **Not in legend**.
Fleet progress (`done/units`, `live`) is read on each state request, independently
of image rebuilds. Coverage includes agent-made legend entries in the existing
legend/guessed-row counts and lists unlisted patterns in their own row group.
Unlisted row shares are fleet estimates within each document; they are kept
separate from guessed legend row coverage to avoid double counting. Made
unlisted proposals awaiting approval remain visible in the coverage denominator.

The accuracy tab's **Reviewer accuracy** table counts the owner's latest
Real/Wrong vote per agent-made or unlisted strip, by source, method and both
together. It reports Real, Wrong, rated total and percent Real; unrated and
cleared votes do not count. New ratings retain source/method in `ratings.jsonl`
so superseded strips can still count. Older ratings can be attributed only
while their strip is present in the current manifest. Missing proposal methods
are shown as `unknown`, rather than inferred from the tile's appearance.

Each build records `rebuilt_documents` and `seconds` in state. A proposal change
invalidates only its document's rendering cache; unchanged document outputs
retain their mtimes. Polls may notice multiple fleet documents in one batch.

## Swatch and borrowing rules

`genlab/swatch.py` removes 1–3 pixel dark rails independently on all four sides,
including up to three pixels of exterior margin. A rail covers at least 90% of
its central length. Its thickness is compared with full ink runs in the interior:
recurrent fill stripes and solid dark fills are kept; an anomalous thin rail is
removed, and an extra rail touching a fill stripe leaves the normal stripe intact.
This avoids deleting a thick fill column simply because it is dark.

Native repeat overlap must have at least 96% pixels within RGB distance 35 and
variance correlation at least .82. An axis can use one complete swatch period if
its opposite edges agree at 96% within RGB distance 35. Repeats observed fewer
than twice also require matching edges on the extracted period. Without repeat
evidence, only the cleaned interior is wrapped, with no mirroring or blending;
failure on either axis records `seam_risk: true` and withholds a swatch candidate.
An edge test is evidence of continuity, not proof that a symbol is complete.

Borrowing requires zero exclusive column matches and no ambiguous twin. Names
are lowercased with punctuation and whitespace normalized. The complete explicit
synonym table is `Sand and Sandstone → Sandstone` and `Dolostone → Dolomite`;
`Clay-shale`, `Clay/Shale` and `Clay shale` already normalize identically. There is
no fuzzy matching, OCR correction, or collapsing of lithology modifiers. Donors
must be current-version human/verified native tiles from training documents and
must not be rejected. The full cleaned swatch must match the donor at >=.92 after
the best cyclic alignment, with RGB distance <20 and background distance <=12.
Borrowed entries always render the donor's learned tile, never a target swatch
substitute. The scanner resolves donors in a second pass, independent of scan order.
If later label edits change/revoke the donor model, generation withholds the
cached borrowed entry until the scanner rematches it to a current donor.

Conservative non-lithology exclusions:

- Names containing whole words show/shows, fluorescence, cut/cuts, stain/stains,
  fossil/fossils, accessory/accessories.
- Exact fluid names Oil/Gas (optionally Good/Moderate/Poor), Dry, Salted water.
- Exact Pyrite or Chert nodule(s), optionally followed by accessory/accessories.
  Bulk Chert, Coal, and Bitumen remain lithology candidates.
- A cleaned swatch >=97% white whose nonwhite symbol spans less than 55% of both
  dimensions. Blank white swatches are retained; a sparse distributed fill is retained.

Review sheets show the cleaned swatch at native 3x, evidence (up to four crops),
and two generated previews; every crop in a row has the same display width.
Borrowed rows show donor evidence. Previews on withheld rows are diagnostic only.
`data/genlab/candidates/review-queue.json` lists every verified/borrowed/swatch row.
`sheet_png` is relative to that directory, and `row_index` is zero-based within
that PNG. Reports also record the row's pixel offset/height and seam test details.
The summary records legacy swatch-only frame corrections separately from the
subset that passes both seam tests; baseline tiers persist across repeated scans.

## What it makes

Human and verified entries compare their column model with a seam-safe native
legend repeat against **every** real source crop. Both use the best cyclic RGB
pixel match (distance <45), after rail and neighbouring-layer cleaning. A swatch
must score at least .85; it wins for a quilt, fewer than two column voters, or a
score within .01 of the column model. Cards and manifest records retain
`tile_source` and both `tile_scores`. The held-out chip is separate: when there
are no independent crops, the page explicitly says the score reuses source crops.
Borrowing additionally requires at least .80 cyclic ink Dice so a large matching
background cannot hide differently sized or spaced symbols.
Borrowed cards retain their donor's `tile_source`; their source remains
`borrowed` and does not imply target-column evaluation.
Column overlap can establish a horizontal repeat wider than half an evidence
lane. Continuous stripes use a row majority to outvote thin scanner dropouts;
tile voting falls back to native reference pixels if it would erase the marks.
Machine evidence excludes neighbouring motifs using phase-aligned ink checks,
including cases where two layers share their background colour.

The raw dataset tile stays unmarked. `repeat-*.png` is a separate preview tiled
to the strip's native width and 48–240 pixels tall, with one repeat outlined in
red. It uses the same display width as generated strips.

Coverage counts all legend entries and the subset in training documents. Row
coverage weights saved prefill intervals by their pixel heights and maps log
names through candidate reports and labels through `learn.norm`. These numbers
are **estimated from machine guesses**, not geological truth. The table's row
share describes each label's proportion of all guessed column rows.

Swatch seam checks inspect per-scanline overlap as well as the average, reject
black edge runs that thicken on wrapping, and reject unexplained full scanlines
of another non-ink colour. Extraction takes a native interior window rather than
averaging two cycles into new gray colours. Failing swatches remain withheld;
reviewer rejects are never cleared by rebuilding.

`python -X utf8 -m genlab.stage3 capture`, `before`, `before-sheets`, and `audit`
preserve original models and render comparison sheets under `data/genlab/scratch`.
The audit renders withheld entries diagnostically without admitting them into
the generated dataset. Its metrics distinguish all-crop matches, independent
held-out matches, and borrowed entries that have no target-column evaluation.
After visually checking those sheets, `python -X utf8 -m genlab.stage3 finish`
writes the per-entry findings and the requested `data/genlab/jobs/rereview.json`.
Use `python -X utf8 -m genlab.candidates --workers 4` for an independent-process
rescan; donor matching still runs only after all local models have been saved.

- **One repeat tile per (document, legend label)** from the boxes you accepted with that single label.
  Rails at the column edge and rows from a neighbouring layer are cut away. The repeat height is pooled
  across up to 6 long boxes. Up to 12 boxes are aligned to each other and vote pixel by pixel (median),
  so one dirty box cannot change the result. Every output pixel comes from your scan crops.
- **Suspect labels**: boxes whose background colour or symbols disagree with the majority for that label.
  They are left out of the tile and listed on the card, so you can check them in the labeller.
- **Patterns with no measurable repeat** are patched together from real pieces (`generator/core.py`) and marked
  as such.
- **Stacked columns** for documents with at least two patterns of 3+ boxes. Layer thicknesses are drawn from
  your boxes, and the next label follows the label order in your boxes. Layers meet with no drawn line, as in
  the real logs. The frame rails are copied from the real column.

## How to judge it

- *Generated patterns*: your boxes and the generated strips, both at the same width and scale. The
  "pixel match" chip compares the tile with boxes it did **not** learn from. Mark strips Real or Wrong; your
  marks are saved to `data/genlab/ratings.jsonl` and counted on each card.
- *Generated columns*: a real stretch next to generated columns; tick "Show the true labels" to check the
  truth list. The check line trains a plain colour and repeat matcher on generated strips only, then labels
  your real boxes with it.
- *How accurate my labels are*: the saved machine guesses (`bakeoff/labeltool/data/prefill.jsonl` and the queue)
  scored against your labels. Your fully labelled Hadrian log is scored row by row. Every box you accept or
  reject in the queue is listed with its queue guess, its row-level guess and its box overlap.

## Dataset

`data/genlab/generated/` (or **Download dataset**): `manifest.json`, `p<legend>-<sig>-<n>.png` pattern strips,
`column-*.png` with a matching `column-*.json` layer list (`top`, `bottom`, `legend_id`, `label`, output pixel
rows). Each pattern also has a 224 × 224 RGB copy and a padding mask. The manifest records stable common label IDs, native and normalized paths, transform values, pixel hashes, and human source sample IDs with annotation versions. All outputs are synthetic training samples and start as unreviewed. Machine-verified entries keep their source designation and can have no human source IDs. Files from older label sets are removed when a pattern is rebuilt.

## Tests

`python -X utf8 -m pytest genlab/tests/test_genlab.py -q`, then `python -X utf8 genlab/tests/shots.py`
(headless page check while the server runs; screenshots go to `data/genlab/scratch/`).
It checks both viewport widths, all source filters, repeat-preview scale and
broken images, and captures three agent-made and two unlisted cards when
available. Each viewport holds one actual state response while it checks images,
so incoming fleet rounds cannot reset lazy image loading during the assertion.
After a complete scan, `python -X utf8 genlab/tests/audit_candidates.py` checks
queue locations, tier totals, donor identities/colours/native tiles, and legacy
seam corrections. Its detailed report is `data/genlab/scratch/candidate-audit.json`.

## Versioned datasets and the automatic loop

Open **Dataset** in the lab at http://127.0.0.1:8781. It shows version counts,
flat metric charts, regression flags, the latest evaluation tables, fleet
feedback, real/clean/augmented contact sheets, and a ZIP download for each
retained version. The original generated-pattern download still exists.

From the working folder:

```powershell
python -X utf8 -m genlab.dataset build
python -X utf8 -m genlab.dataset eval v1
python -X utf8 -m genlab.dataset loop
powershell -NoProfile -ExecutionPolicy Bypass -File genlab/start-dataset-loop.ps1 -Restart
```

The launcher is deliberately inside `genlab/`: this task permits source edits
only there. It uses `Invoke-CimMethod Win32_Process Create` to detach the loop,
surviving the launching shell. `-Restart` stops existing dataset loop processes.
The loop log is `data/genlab/datasets/loop.log`; its PID, last heartbeat, pending
changes and error are in `loop-state.json`. An OS lock prevents duplicate loops;
a separate writer lock prevents overlapping builds/evaluations.

Every 120 seconds it checks the active lab pattern signatures, reviews,
proposal/reviewer JSON files, normalized index/split policy, and owner ratings.
It builds when **10 or more distinct patterns were added, changed or removed**,
or **30 minutes have elapsed since the last build and anything changed**. It
builds immediately when there are no versions. Changes noticed during a build
remain pending for the next cycle. The loop does not start fleet agents.

Each `data/genlab/datasets/vN/` contains:

- `manifest.jsonl`, `stats.json`, `CARD.md`, and a snapshot of active lab metadata;
- saved model pixels, native clean/augmented strips, 224px letterboxes and valid
  padding masks, plus paired legend-only and saved-column-model ablations;
- stacked columns with clean copies, native uint16 label masks, JSON layer truth,
  measured thickness distributions and transition counts;
- `real.json` and accepted real crops, kept separate from synthetic training;
- fitted scan distributions and real RGB residual masks in `realism.json` and
  `overlays/`, plus per-source `sheets/`;
- `eval.json`, `eval.md`, and publication/regression metadata after evaluation.

Classes are stable `doc:legend_id` or `doc:uN`; canonical lithology names are
separate. The builder copies the lab's selected tiles and saved quilt models;
it never learns new column tiles. Ablation extracts the native legend repeat
with the existing swatch routine and uses cached pre-selection column models.

The labeller's exact fixed test-well rule is read from
`data/normalized/split-policy.json`. Test wells cannot contribute samples, real
crops, residuals, thicknesses or donors. Among its training wells, validation is
`SHA256('genlab-validation-v1:'+well)[:8] mod 5 == 0`. Borrowing from a validation
well also holds out every page/class of the dependent well. Training and
validation do not share tiles, residual pools or thickness/transition evidence.

For appearance fitting, native real crops are phase-aligned to the saved tile by
exact cyclic least-squares FFT. Blur is fitted over a small radius grid; RGB
drift and noise use robust residual estimates. Strong residual pixels retain
their real colours. Measurements sample within the document, then family from
saved prefill, then the same-split global pool. Each sample records its evidence
and fallback. Agent column references and machine boxes can measure appearance;
only human-accepted boxes can be evaluation truth. Detected rails stay at edges.
Crop requests cap at 480 rows and use `learn.Crops` under the image reader lock.
One page is open at a time; no whole TIFF page is loaded.

Thickness and label order combine human boxes with prefill intervals, explicitly
tagged as measurements rather than truth. Missing class thicknesses use their
family distribution. Family transition distributions back up missing outgoing
document transitions. Thin beds can be one row, and the final layer can be partial.

Evaluation trains the existing colour/repeat nearest-centroid matcher on fakes
only, plus CPU logistic regression when sklearn is installed. It compares
clean/augmented and matched legend/column ablations. Because entire wells are
held out, classifier targets are canonical lithology names; document-class IDs
are retained in per-class tables. Unseen names count as misses. Macro scores
include only names with at least three real boxes; smaller classes say **too
few**. The balanced discriminator trains on training wells and evaluates on
held wells, with separate source tables. Owner audits use the latest Real/Wrong
vote per file and retain historical source/method when recorded; no votes means
`n=0, too few`.

`CHANGELOG.md` and `history.json` preserve changes, counts and metric deltas.
Macro fake-to-real drops greater than 2 points, or discriminator rises greater
than 5 points, flag **REGRESSION**. Changed human validation truth is also flagged
as an unpaired comparison. `latest.json` and the portable text pointer `latest`
advance only after a complete evaluation with sufficient evidence and no
regression. Keep the last five versions; an empty `vN/keep` file or
`stats.json` `keep: true` protects a version. The latest passing version is also
protected. Pruned versions retain their metric history. `feedback.json` lists
poor classes, owner Wrong evidence/actions, and source-only classes ranked by
estimated column rows, including explicit unknown row weights.

Additional checks:

```powershell
python -X utf8 -m pytest genlab/tests/test_genlab.py -q
python -X utf8 genlab/tests/audit_dataset.py v1
powershell -NoProfile -ExecutionPolicy Bypass -File start-genlab.ps1 -Restart -NoBrowser
python -X utf8 genlab/tests/wait_ready.py
python -X utf8 genlab/tests/shots.py
```

The test suite builds two tiny datasets in a temporary directory and checks
stable IDs, test exclusion, well/donor isolation, unchanged augmentation labels
and column masks, machine-truth rejection, the 10-pattern trigger, exact cyclic
alignment, and Windows read retries. The artifact audit checks actual version
hashes, masks, split isolation, ZIP integrity and the HTTP download route.
Screenshots cover all four tabs at desktop and phone widths. Readiness retries
for up to ten minutes because a first rebuild can refresh changed documents.

### First evaluated version (v1)

328 patterns, 3,240 synthetic samples (64 columns), 1,220 measured crop windows;
112 patterns have real column references, 60 have human-accepted examples.
There are 857 accepted real boxes in the artifact, of which 79 belong to seven
validation wells. Training has 1,048 strips per main clean/augmented arm.

| Experiment | Matcher macro | Logistic macro | Evaluation n |
|---|---:|---:|---|
| Augmented fake to real | 10.863% | 4.545% | 79 boxes; 8 supported names / 71 boxes |
| Clean fake to real | 10.863% | 20.312% | Same 79 boxes |
| Legend-only paired ablation | 37.861% | 44.408% | 68 boxes; 6 supported names / 65 boxes |
| Real-learned paired ablation | 37.861% | 31.015% | Same 68 boxes |

The ablation has 43 matched training document classes and 172 training strips
per arm. The overall augmented matcher top-1 is 8.861% (7/79); logistic top-1
is 5.063% (4/79). The discriminator falls from 80.263% clean to 63.158%
augmented, on 76 real + 76 fake held crops, trained on 187 real + 187 fakes.
There are no owner ratings in this snapshot (`n=0`).

These are weak recognition results. Augmentation makes the distributions harder
to distinguish but harms the small logistic classifier; it has not established
reliable lithology recognition. The inspected sheets retain motifs and add real
rails, specks, scan lines and colour variation. Some residual fields repeat and
can contain tile mismatch rather than annotations. Disagreement above 35%
disables pasting, but does not prove the remaining masks are purely overlays.
Families without measured references use documented fallback pools. Most classes
still have no real reference, and swatch/unlisted/borrowed discriminator results
with no held real matches are **too few**, not evidence of success. The loop
surfaces these limits and sends evidence to the fleet through feedback only.
