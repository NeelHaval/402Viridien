# Part A — real pattern mining and review

Run from the bake-off folder. Python, NumPy, Pillow, OpenCV, PyTorch,
Transformers and pytest are already available. DINOv2 uses the benchmark's
cached `facebook/dinov2-small` weights, CUDA when available, and no downloads.
All code, caches, crops, sheets and review outputs for this part stay in `synth/`.
The real dataset and existing annotations are read only. Imported modules do
not write bytecode. No subprocesses are used, and credentials are never logged.

```powershell
python -X utf8 synth/patterns.py
python -X utf8 synth/mine.py --dry-run --workers 8
python -X utf8 synth/mine.py --max-calls 3000 --workers 8
python -X utf8 synth/sheets.py
python -X utf8 synth/review.py --host 127.0.0.1
```

Only the third command makes real JEV requests. Run that command in the main
session, where requests are permitted. `--no-jev` runs the same offline mining
pipeline without scheduling new verification. `--families "Gulf" "D-582"`
limits mining by case-insensitive family substring and preserves other families'
results. `--max-calls` defaults to 3000 and `--workers` to 8. Verification makes
up to three attempts, waiting 2/5 seconds before retries and cooling off for
10 seconds after an exhausted request. Eight consecutive failed attempts open
the shared circuit. Only successful responses are cached; legacy failure cache
records are ignored. Already running requests may finish. Every attempt counts
against the call budget, which is reserved under
a lock. Content-identical requests share a verification, including across
candidate patterns. Cached results consume no budget.

`--resume-scan` saves the completed offline scan before crop export and panel
construction, and reuses it when code, swatches, source intervals, budgets,
lookalikes and dataset file metadata still match. This flag can also be used
for the real JEV pass after offline inspection. Rendered native legend panels
are reused within a run; tests compare them pixel-for-pixel with the benchmark,
so verification cache keys remain unchanged.

## Catalogue and mining

`data/catalog.json` contains all patterns and legend types. A legend type is
the family and sorted set of normalized labels from `../analysis/swatches.json`,
exactly as in `make_mix_page.legend_types`: lowercase ASCII letters only, then
`sandandsandstone` becomes `sandstone`. No extra OCR correction is applied.
The signature is those labels joined with `|`, encoded as UTF-8. A pattern ID is
`family-slug-sha1(signature)[:6]/normalized-label`. Ordering of logs or swatches
does not change IDs. Shares are the sum of `label_mix` label share times column
height, divided by total dataset column rows. Ambiguous and unnamed rows stay
unallocated; zero estimated share does not remove a pattern. Targets start at
variety zero and are updated after mining using the required clamped formula.

Swatch thumbnails from analysis are copied into `data/swatches/`. Cached native
legend swatches are copied alongside them when available; their native pattern
period is measured by vertical autocorrelation. Otherwise the family's native
median period in `volume_budget.csv` is used. Height is `max(48, 3*period)`.

Sources are gold, explicitly agreed consensus, nonmocked prefill JEV interval
or cluster suggestions, high-tier colour suggestions, then embedding search.
Other silver and FFT suggestions are not treated as trusted seeds. Search
samples every log in the family, plus one-offs with the same legend vocabulary,
at approximately half the family's median
bed height. Detected horizontal contacts, blank transitions and colour changes
define search intervals; gold and consensus contacts also split them. A crop
stays at least eight rows inside its recorded interval. Detection is heuristic:
an unlabeled contact missed by the detector still needs visual review.

The full column is decoded through the harness in strips of at most 4000 rows;
the full page is never converted to RGB. Blank, horizontal-rule, vertical-grid
and likely text-line crops are rejected. Output crops retain full column width
and native pixel resolution. Full-query DINOv2 embeddings use the benchmark's
resize and normalization and are cached by content in `cache/embeddings.sqlite`.
References are native swatches tiled to query size. Every non-binary family is
gated in CIE Lab: quantized non-ink palette modes must occupy at least 8% of the
swatch, the candidate's main-fill share within deltaE 15 must be at least half
the swatch's, and its dominant non-ink colour must match that palette. Confirmed
gold/agreed crops calibrate family scan shifts only with at least five samples,
two labels, median residual <=5 deltaE and displacement <=25 deltaE. Unsupported
shifts remain zero and are reported. Similarity averages DINO cosine and a
symmetric Lab palette similarity. Ranking uses own similarity minus the best
other label in the same legend and requires a positive margin; listed lookalike
twins are excluded from rivals and candidates carry `twin: true`. Targets are
upper bounds, never quotas. `mining_report.json` records gate pass rates, counts,
calibration evidence, verification circuit state and per-pattern shortfalls.
Memory is bounded by retaining up to 80 distinct possibilities per log/pattern.
Their coordinates and provenance remain in `data/pools.json` for auditing.

Final selection enforces cosine similarity at most 0.97 between retained crops
and a **25% maximum from any one log when at least four logs contribute**;
otherwise it uses an even split across the available logs. Both the verification
set and final set (at most the pattern target, capped at 30 for review) obey
that rule. Sparse patterns remain visible with raw candidate counts and
coordinates in `pools.json`. `no_candidates` lists patterns without a
compliant shortlist; `no_raw_candidates` distinguishes actual discovery gaps.

Verification reuses `classify.data.panel` and the benchmark prompt: gallery for
at most 15 rocks, choice otherwise. Choice panels retain the original printed
native legend cells using cached parser coordinates. Each candidate records
the predicted label, normalized probability, agreement, method and cache key.
Unavailable/budget-exhausted candidates are explicitly marked. Verification
agreement is recorded for review; ranking remains the contrastive margin. Variety is
the normalized embedding variance, `1 - ||mean(unit embeddings)||²`.

## Sheets and reviewer input

`data/sheets/<pid_slug>.png` uses an unstretched tiled swatch in its header.
Every candidate in a pattern has the same isotropic display scale. Sheets are
at most 1600px wide and split into two pages for taller grids; numbers continue
across pages. `data/sheets/index.json` is the authoritative mapping from pattern
and number to log, native coordinates, crop file, source and sheet position.
All stored file paths are relative to `synth/`.

A vision reviewer writes `data/selected.json` after inspecting these sheets:

```json
{
  "family-slug-abcdef/shale": {
    "picks": [1, 4, 8],
    "notes": "Clear repeated hatch, few scanning artifacts.",
    "issues": ["Example 4 may include a modifier."]
  }
}
```

The example ID above is illustrative. Use actual IDs and numbers from the
index. The tests generate mock selections in their own fixture directories;
production selections are never fabricated. Restart the review server after
changing selections or regenerating sheets. Keep sheet numbering fixed while
reviewing an existing selection.

## Human review

Open `http://127.0.0.1:8770`. Use `--host 0.0.0.0` to review from a phone on
the same network, then open the machine's LAN address on port 8770. The stdlib
server serves the UI and data images; it does not launch a browser. It loads
only the catalogue summary and first pattern initially and prefetches the next
three patterns, including their selected images. The page is plain HTML/JS,
with a tiled swatch, true-aspect examples, notes, issues, folded alternatives,
large touch controls and family progress bars. No frontend build is required.

| Input | Action |
| --- | --- |
| Y / Fine | Approve the remaining picks and advance |
| N / Bad | Mark the pattern bad and advance |
| R / Re-pick | Request another selection and advance |
| Tap an example | Toggle inclusion; save immediately |
| Number, Enter | Toggle example 1–30; a 650ms pause also confirms |
| J / K / arrows | Next / previous pattern |
| F | Focus the all/unreviewed/flagged/family filter |
| U / Undo | Undo the latest active decision on this pattern |

Every edit and decision appends to `data/review.jsonl`, with an ID and timestamp.
Undo also appends an event, retaining the audit history. Last active decision
wins. Only a pattern with status `approved` contributes to `data/approved.json`:

```json
{"pid": [{"file": "data/approved/pid-slug/01.png", "log": "log-name", "top_y": 100, "bottom_y": 160, "col_w": 60, "h": 60}]}
```

Approved native crops are copied into `data/approved/<pid_slug>/NN.png`.
Changes, rejection and undo rebuild the manifest and remove stale copied
crops. Server startup rebuilds from the journal, so a missing/stale manifest
is recoverable. Selection alone never constitutes human approval. The review
server is intended for a trusted local network; there is no account system.

## Validation

```powershell
python -B -m pytest -q synth/test_mining.py
```

All JEV tests inject a mock client. Tests cover stable IDs; strict diversity and
cross-log cosine deduplication; interval margins; blank/grid/repeat detection;
native crop preservation through the complete mocked mining pipeline; sheet
pixels, numbering and splitting; exact benchmark gallery/choice selection;
verification budget, successful caching, retries and the eight-failure circuit; review save, undo, restart and
approved-file rebuilding; and a localhost first-screen fetch under one second.
Tests use ordinary directories under `synth/cache/test-cases/` because pytest's
restricted Windows temporary-directory ACLs do not work in this sandbox.
Pytest's optional cache is disabled by `synth/pytest.ini` for the same reason.

The all-log dry-run report is `data/mining_report.json`. Dry-run executes real
offline embedding/mining and writes real crops; only JEV is omitted. Its call
count is the number of unique uncached verification requests, capped by
`--max-calls`, rather than the raw number of pattern/candidate pairs.

Historical validation before colour/contrastive filtering: **18 tests passed**. Catalogue and final
dry-run covered all **51 logs**, **29 legend types**, and **476 patterns**, with
no log errors. The dry-run schedules **118 unique JEV calls** for **193 candidate
entries** and makes **zero** external calls. It produced 476 sheets. All patterns
have raw possibilities; 441 have no compliant shortlist after diversity and
deduplication. Local server startup plus HTML, catalogue summary, first pattern
and swatch payloads took **45.5 ms**; browser rendering was not timed.

| Family | Patterns | Zero compliant candidates |
| --- | ---: | ---: |
| D-582 1-bit grid | 10 | 9 |
| Geoservices grid | 75 | 75 |
| Gulf 3-panel | 190 | 178 |
| MC761 mineral/lithology | 11 | 11 |
| MC765 colour grid | 23 | 1 |
| One-off layout | 102 | 102 |
| Texas 'Rock Types' box | 65 | 65 |

For the current miner, run `python -X utf8 -B -m pytest -q synth/test_mining.py`.
The all-log offline run is `python -X utf8 -B synth/mine.py --no-jev --dry-run --workers 8`,
followed by `python -X utf8 -B synth/sheets.py`. Run
`python -X utf8 -B synth/mining_check.py --baseline` **before** replacing an old
candidate manifest to retain comparable native-colour rates, then run
`python -X utf8 -B synth/mining_check.py` afterwards. It writes before/after
audits, adds output-colour and confirmed-sample calibration rates to the mining
report, and builds `out/mining_check.png`: two seeded random populated patterns
per colour family, with a native swatch and up to eight candidates at equal
display height and native aspect ratio. Sparse/empty patterns are counted in
the JSON report. Native-fill RGB checks use 32-level bins, exclude near-black
swatch ink, and compare the crop's top two modes within 48 per channel; the
separate Lab audit uses the actual mining gate. These are different measures.

# Part B — synthetic columns and offline benchmark

## Procedural texture calibration and blind comparison

Run these commands from the bake-off root, without network access:

```powershell
python -B -m synth.texstats
python -B -m synth.compare
python -B -m synth.audit_compare
python -B -m unittest synth.test_procgen synth.test_generate -v
```

Open `synth/out/compare.html` directly. It is a single portable HTML file with
embedded PNGs, styles and JavaScript. Choose the real A/B chunk to reveal the
answer. Scores persist in localStorage and export as JSON. The expandable
inspection panels show four real/generated samples at native 1x and 3x,
the native legend, three contact examples where annotated contacts exist,
and measured differences highlighted above 20%. Shortages and repeated
contacts are stated. Geoservices currently has an isolated consensus interval
but no adjacent labelled contact pair; its contact panel says unavailable.
Contact examples may use other patterns from the same family, explicitly
identified as family examples. They are never relabelled as the target pattern.

`data/texstats.json` records every indexed crop pattern plus supported gold and
agreed-consensus native intervals. Its input manifest freezes the index hash,
distinct source files and counts, so another mining run cannot silently change
the calibration or copy-correlation test. Rebuilding calibrates a new snapshot.
Duplicate crop files are measured once, even when assigned to several pids.
The source priority is labelled native intervals, then unverified indexed
crops. Unverified candidates are not human approvals. Large colour conflicts
with the legend are recorded and shown. No search-pool label is promoted to
truth. Gold replaces consensus for that source log. Full native strips, bounded
to 4,000 rows, separately measure family rails and widely spaced grid rules.

The schema includes censored horizontal/vertical run distributions, peak-row
dash lengths, row spacing, stagger, centroid wobble, edge variation, CIE Lab
fill/ink moments, radial fill-noise power spectra, colour bands, quantization,
native/swatch scale measurements, source coordinates and contact profiles.
Colour moments reject ink fringes and adjacent-bed contamination. Full-width
frequent hatches are distinguished from sparse depth rules. Contact sharpness
is a local 10–90% feature-profile width; indistinguishable profiles stay
unresolved. Cut-share is an ink-discontinuity proxy at the annotated contact,
not a verified reconstruction of every original pen stroke. Slivers are beds
at most six native rows. Boundary fallbacks to family statistics are explicit.

`procgen.render_pattern(pid, height, width, seed)` and
`procgen.render_column(spec, seed)` return native-resolution uint8 RGB arrays.
Column specs contain `width` and `beds`, with each bed specifying `pid` and
`height`, or contiguous `top_y`/`bottom_y`. One-row slivers and partial motif
rows are supported. A missing pid renders a white interval. Contacts clip
strokes without blending, with measured probabilities of a drawn rule.
Only numeric distributions, geometry and random noise reach the renderer;
it never opens crops or swatch images. Motif primitives are approximate for
complex symbols, and a conflicting unverified crop cannot establish a rock's
correct visual identity. The comparison matches each source appearance's
parameters to its generated counterpart, so random selection among conflicting
source appearances does not itself disclose the answer.

Use `--texture procedural|quilt|mix` with `synth.generate`; the default is
`mix`, which selects procedural or quilt per bed. Explicit `quilt` preserves
the original quilting path. Procedural mode fails clearly for unsupported pids;
mix can quilt them. Metadata records each bed's mode and parameter-source logs,
and `procedural_models.json` freezes the fitted numeric models. Procedural
parameters are restricted to the requested train/test source pool; when the
calibration snapshot has no matching source, parameters are measured from
that pool's approved crops. No source pixels are pasted in procedural beds.
The run signature includes the calibration SHA-256. Measured hard contacts
replace default gradational nuisances in procedural/mix runs; explicit nuisance
overrides remain available for ablations.

The tests cover all calibrated pids for determinism and blocked source-image
I/O; representative fields from every family are searched against **every
distinct source crop in the frozen snapshot**, using translation-maximized
grayscale NCC with threshold 0.985. This rejects near copies while allowing
shared periodic motifs; NCC alone is not a proof of independence. Quantitative
round-trip gates cover nine measured hatch/fill patterns: 35% dash/ink tolerance,
30% gap, 20% spacing, and Lab distance 12. The page's stricter 20% flags remain
visible. Tests also check abrupt contacts, one-row beds and RGB versus 1-bit
quantization. Artifacts include NCC and round-trip JSON reports and the complete
1x/3x PIL contact sheet `out/compare.png`. `audit_compare` executes the actual
score JavaScript in Qt's JS engine with a DOM adapter, checking click, reload,
reset and JSON export, and checks every embedded image's aspect ratio. Its
PIL phone/desktop previews approximate the UI; browser layout is not certified.

Part B reads Part A's catalogue and human-approved crop manifest. It never
changes the catalogue, review decisions, strategies, harness, real annotations,
or dataset. All derived outputs and redirected caches stay under `synth/`.

After human review has produced `synth/data/approved.json`, run these exact
commands from the bake-off root:

```powershell
python -B -m unittest synth.test_generate -v
python -B -m synth.generate --catalog synth/data/catalog.json --approved synth/data/approved.json --output synth/data/fake --n-logs 120 --seed 42
python -B -m synth.bench_synth --root synth/data/fake --out synth/out --device cuda
```

Use `--device cpu` when CUDA is unavailable. The final command discovers every
strategy in `strategies/` and every offline classifier listed by
`classify.models.LOCAL`: colour, FFT hatch, DINOv2, SigLIP and the synthetic CNN.
It also reports the existing `classify.bench` fixed FFT local-only hybrid
fallback, without selecting a local model or threshold on test truth. Offline
classifiers use fixed 32-row patches and merge adjacent predicted labels; they
are never given true bed boundaries. Change `--step` for a declared resolution
ablation. Module imports do not write bytecode.

To reproduce the six-log mock sample and its two cheap strategy evaluations:

```powershell
python -B -m synth.generate --mock --output synth/data/mock_demo --n-logs 6 --pool test --seed 17 --min-height 2000 --max-height 2400 --clean-fraction 0.34
python -B -m synth.bench_synth --root synth/data/mock_demo --out synth/out --strategies s01 s03 --classifiers none
```

The mock fixture uses native full-width gold log06 crops, agreed-consensus
log02 crops and their real legend swatches. It writes `data/mock/catalog.json`
and `data/mock/approved.json`; it never fabricates production approvals. Every
mock log and report is marked. Three D-582 labels and one held-out source do
not represent the other families or the complete real task.

## Fit, allocation and crop handling

The fit resolves overlapping interval suggestions with gold above explicitly
agreed consensus, above prefill best guesses, above queue suggestions. Adjacent
equal labels merge. Only touching intervals contribute transitions; uncovered
stretches do not create artificial label transitions. Test source logs are
excluded from the fit. Label row counts have +1 smoothing; transition counts
have +0.5 smoothing with a zero diagonal. A one-label type necessarily has no
nonself transitions. Family/label log-normal thickness models use observed
p1–p99 clipping. Family observations and analysis summaries provide explicitly
flagged fallbacks. A separate thin component produces 6–30-row beds; it is an
assumption outside those clipped bulk distributions.

Log allocation is proportional to the square root of the type's filled-row
share from `label_mix.json`, with at least two logs per represented type.
`--n-logs` is an exact total and defaults to 120. A request smaller than twice
the number of represented types fails. Heights default to 2,000–8,000 rows.
Row shares are divided by expected thickness to obtain bed probabilities,
then combined with fitted transitions, rare-pattern and variety weights.
Coverage quotas require each approved pid at least five times overall,
distributed across supported pools in proportion to their fake-log counts.
Singleton patterns get all five placements in training. These choices deliberately
change natural prevalence; overall scores are equal fake-log means.

Beds vertically quilt approved native crops using a row-similarity start
search and a connected minimum-error overlap seam. Width fitting crops or
tiles pixels, with random horizontal phase. Persistent ink in the outer 8%
is treated as a possible source rail and trimmed before phase shifting; the
trim is recorded. This prevents source borders becoming interior black bars.
No separate horizontal/vertical stretch is permitted. Only when donor width
differs from source width by more than 15% is one affine scale applied equally
to both axes; every such exception is recorded. Output canvas height rounding
does not change the affine scale. `source_crops` records source log, source
file, native interval, overlap, phase, trim, isotropic scale and output rows.

## Source hold-out and missing coverage

The split targets 70/30 by source log, never by crop. One global source-log
assignment also prevents cross-pid source leakage. Deterministic balancing
prefers nonempty pools for multi-source patterns; integer rounding and shared
source incidence can prevent exact 70/30 for each pattern. Singleton patterns
stay in training and appear in `unavailable_test_patterns`. They are not
silently reused in testing. Test fake logs use only test crops; their donor
legends and real overlay assets also come from test sources.

`data/split.json` is the production split. Custom input manifests get
`split.json` next to their approved file, and each fake-run directory includes
its own immutable-for-that-run copy plus `fitted_models.json`. A mixed run
requires at least one train and test source for every represented legend type;
otherwise generation explains the missing coverage and stops. Approve more
source logs for that type, or use `--pool train` for a training-only run. A
test-only run additionally requires held-out coverage for every approved pid.
The benchmark always refuses to score train logs. Crop-source separation does
not prove different wells or eliminate duplicate-scan leakage.

Seeded reruns with identical arguments can regenerate a directory. A different
configuration must use a different `--output`; old runs are never recursively
deleted. `run.json` marks an in-progress run and the bench rejects incomplete
generation. It can be completed by rerunning the same configuration.

## Nuisances, truth and files

Every nuisance has its own recorded probability, selection and applied flag.
Parameters and source coordinates distinguish a selected but unavailable
overlay from an actual perturbation. Connected ink components in donor
non-lithology side strips supply text/annotation/symbol overlays; classification
of those components is heuristic, not OCR-verified semantics. Grid pixels and
spacing come from real donor strips. Borders, blank gaps, 4–30-row dithered
contacts, thin alternations, optional contact lines, colour changes, fading,
noise, JPEG and row-preserving shear below 0.3 degrees are independent draws.
D-582 and other 1-bit donors finish with 1-bit output. LZW is a lossless TIFF
storage/decoder test, not a claim of visible compression damage. Native source
crops may already contain incidental nuisances.

`--clean-fraction` defaults to 0.2 and allocates a deterministic clean quota
within each type/pool, with at least one clean log when that pool has three
or more logs. Clean 1-bit logs still require binary representation. Supply
`--probabilities synth/data/my_probabilities.json` to override any probability;
use the names in `nuisances.PROBABILITIES`. For an isolated nuisance ablation,
set all optional probabilities to zero and the desired nuisance to one.

Each `data/fake/<name>/` contains:

- `image.png` or `image.tif`, with a copied native donor legend on the page;
- `boxes.json` with the real `bounding_boxes` wrapper and all three box keys;
- `legend.png`, donor legend parser caches and `legend_entries.json`;
- `truth.csv`: `top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id`;
- `meta.json`: seed, pool, family, dimensions, calibration, all nuisance flags,
  bed plan and every source crop/transform;
- `image_clean.png`: the same bed plan and textures without image nuisances.

Truth intervals exactly tile the column with exclusive bottoms. Blank gaps
are explicit intervals with an empty `lithology` and cluster -1; this makes
tiling auditable while `harness.score` ignores those pixels. They differ from
the real strategy convention of omitting blank rows, with identical scoring
semantics. The boundary truth for a gradient is its nominal contact centre.
Adjacent same-label texture placements merge in scored truth unless a contact
line was actually drawn. `meta.beds` retains every sampled bed/placement and
its sources even when truth merges their indistinguishable labels.
Depths use a family ft/px draw and a synthetic linear offset, rounded to 0.1 ft.

## Adapter, report and interpretation

`io_synth` exports `Log`, `list_logs`, `open_log`, `read_rgb` and CSV helpers.
`Log` is the real harness class. Its reversible `patched` context changes
`harness.io` and runner lookup in-process and redirects all runtime writes:

```python
from synth.io_synth import patched
from harness import run
import sys

with patched('synth/data/fake', pool='test'):
    sys.argv = ['harness.run', '--strategy', 's01', '--set', 'eval', '--gt', 'gold']
    run.main()
```

The context is not thread-safe. Donor OCR/parser caches are shifted to the
copied legend position and the known synthetic linear depth scale is a warm
calibration cache. The benchmark therefore measures lithology/segmentation
conditional on those inputs, not fresh legend OCR or depth-estimation quality.
Clear or separately evaluate those caches when investigating those stages.

`out/results.csv` contains every method/log/condition and explicit availability.
`out/summary.csv` has family, nuisance and overall means and 95% CIs;
`out/breaks.csv` has accuracy drops. `out/report.html` is standalone, flat and
scrollable on a phone. Native-size column PNGs are in `out/previews/`.
`out/predictions/` holds challenge-format predictions. Algorithm errors score
zero; missing dependencies/weights and unavailable paid responses are NA with
coverage reported. No exception text or credentials are stored.

CIs bootstrap whole fake logs 2,000 times, conditional on the source library;
they do not estimate uncertainty across independent real wells. One-log groups
have no CI. The report shows contributing source counts. “Versus clean logs”
compares within family and needs observed clean logs. The paired image drop
uses identical beds/textures and removes all image nuisances together; it is
not an isolated causal effect of the named nuisance. Structural interbeds,
blank gaps and intrinsic 1-bit output stay in paired controls, so their paired
drop is NA. Read `CRITIQUE.md` before choosing a pipeline from these numbers.

## Paid methods stay offline

Paid strategies are listed but disabled by default. `--include-paid` enables
only capped offline response replay for JEV, Terra and their paid strategies;
it does not enable network or subprocesses. No API key is read. Every run
installs a reversible network/process guard. `--paid-cap 20` caps replay lookups,
including misses, across all methods and both conditions. External calls are
always zero and missing responses are unavailable rather than scored guesses.

```powershell
python -B -m synth.bench_synth --root synth/data/fake --out synth/out/paid_replay --include-paid --paid-cap 20 --paid-cache synth/data/paid_cache --device cuda
```

Replay files are `<method>_<request-sha256>.json`, containing a provider's
decoded response. `offline.Replay` hashes sorted JSON request signatures;
`out/paid_replay.json` records requested keys and cache hits. The strategy JEV
signature is `{state, assets, questions}`; strategy Terra uses
`{image_sha256, prompt}`. Patch classifier signatures are
`{images: [sha256(native RGB bytes), ...], vocabulary: [...]}`. Replay/import
responses only from authorized prior runs; the bench itself cannot fetch them.
