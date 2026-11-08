# Lithology label and recheck tool

From the bakeoff working folder:

```powershell
python -X utf8 -B labeltool/build_queue.py --workers 3
python -X utf8 -B labeltool/serve.py --port 18765
```

Open **http://127.0.0.1:18765/**. Port 8765 is occupied by the machine's existing
Helm service. The tool defaults to 8765 when `--port` is omitted; `python
labeltool/serve.py` works when that port is free. The queue is already built, so
normally only the second command is needed. Use
`--host 0.0.0.0` to listen on the team VM's network interfaces; choose another
port with `--port`. The HTTP server uses only the Python standard library;
TIFF decoding and preprocessing use the existing harness, Pillow, NumPy, OpenCV,
and the S11 descriptor's installed dependencies. There are no CDN requests or API calls.

## Review

The suggested label is selected automatically. Pick a legend tile or type the
exact printed label, then accept. The original swatch pattern is shown with its
aspect ratio preserved. Native image crops have 150 rows of context on each side,
blue interval brackets, and a row ruler. 1×, 2× and 4× multiply both dimensions
equally. Scroll long crops. The narrow full-column overview is deliberately
compressed vertically; the working crop retains its true proportions.

| Key | Action |
| --- | --- |
| Enter | Save selected label and edited boundaries; advance |
| 1–9, a–z | Select the corresponding legend tile |
| U | Save Unknown / not lithology; advance |
| B | Toggle boundary editing: drag a handle, tap the nearer boundary, or edit row fields |
| Space | Skip for this browser session |
| Z | Undo the latest saved decision, including across restarts |
| ? | Help |

Uppercase U/B/Z always invoke their actions. Lowercase u/b/z choose their tiles
when those tiles exist, and otherwise invoke the actions. Use Shift+letter to
disambiguate. Buttons support touch. Skips can be revisited when a queue is empty
or by selecting a mode again. Clicking a log in Progress enters Walk mode.

**Uncertain first** reviews low, then medium, then high tiers, round-robin across
logs within each tier. **Spot-check** samples high-tier items in a shuffled random
order without replacement. Its live statistic counts the shown suggestion as
right only when its label is retained and its boundaries are unchanged, and shows
a Wilson 95% interval. **Disputed** selects explicitly disputed consensus rows.
**Walk a log** reviews one log from top to bottom. Gold log06 is excluded from all
four modes; its agreement table is separate in Stats.

The next three crops are prefetched. Cached PNGs live in `labeltool/cache/`.
**Refresh sources** discovers newly written consensus, S, T and vocabulary files;
it recalculates live queue confidence without touching other jobs' outputs.

## Labels and scores

`gt/human/labels.jsonl` is an append-only, flushed decision journal. Each label
records native edited rows, item ID, source suggestion shown, boundary-edit flag,
mode, elapsed milliseconds and UTC timestamp. Undo appends a cancelling event.
Each save/undo rebuilds `gt/human/<log>.csv`; overlapping rows use the last active
decision, trimming earlier decisions as necessary. Restarting recovers CSVs from
the journal. CSV fields are
`top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id`; depths use the harness
calibration cached per log. If OCR calibration fails, depth fields stay empty and
the failure is recorded in `labeltool/data/summary.json`, never invented.

```powershell
python -X utf8 -B labeltool/score_human.py
python -X utf8 -B labeltool/score_human.py --json
```

The same table appears in **Stats**. Strategies under `out/<strategy>/`, silver A/B,
and available labelling consensus are compared only on human-labelled rows.
Missing predictions count as wrong. `covered_rows` distinguishes source coverage
from the common denominator `n_rows`. Boundary hits are one-to-one matches with
the harness's 4–20 native-row tolerance; source boundaries wholly in unreviewed
gaps are excluded. Reviewed span edges are treated as review boundaries. Scores
measure the rows you chose to review; only the randomized Spot-check estimates
agreement on confident items. Repeated active decisions for an item count once
in progress and the spot-check statistic.

## Queue and isolation

`labeltool/data/queue.jsonl` covers the entire interpreted lithology box for every
one of the 51 annotated TIFFs, including blanks. A cheap change-point pass pools
four native rows and at most 96 columns; it combines colour, ink and horizontal
texture changes with available machine boundaries, merges slivers under 12 native
rows, and divides very long constant runs into reviewable crops of at most 1600
rows. Native-scale centre patches provide additional S11 FFT estimates; exact
saved S11 CSV predictions are retained separately wherever available. Colour Lab
estimation duplicates `../scripts/label_mix.py`'s masking and dominant-colour method.

The original builder's high tier requires d1 ≤ 12, margin ≥ 15 and agreement of every available source.
Any disagreement, hatch-only/grey/blank colour, colour outside the legend, or
missing parsed swatches is low. Remaining weaker colour matches are medium.
Unavailable model calls supply no fabricated suggestions. Gold is marked truth.
Family names match the existing analysis's seven layout families.

## BEST-GUESS prefill

Run these in the main session with network access, from the working folder:

```powershell
python -X utf8 -B labeltool/prefill.py --dry-run
python -X utf8 -B labeltool/prefill.py --validate-gold --max-calls 4000 --workers 8
python -X utf8 -B labeltool/prefill.py --logs all --max-calls 4000 --workers 8
python -X utf8 -B labeltool/prefill.py --merge
python -X utf8 -B labeltool/serve.py
```

Stop/restart the server after merge or undo. The best guess is selected immediately;
its source chip identifies e.g. **JEV x3 agree**, **JEV interval**, or **Sol+Terra**.
Enter accepts the selected label using the existing decision journal.

The prefill reads **harness.legend_v2** legends through the isolated harness cache.
Each interval supplies a native-scale centre patch, at least 24 rows and twice the
largest estimated legend pattern repeat, even when the interval itself is shorter.
Cached DINOv2 embeddings (GPU when available) are combined with a Lab histogram.
For large logs, deterministic MiniBatchKMeans compresses features to at most 256
microclusters before Ward agglomeration into at most 20 groups. Robust distance
outliers form a noise bucket; they never inherit a cluster majority.

Each cluster supplies the squared-distance medoid and two spread-out members
(fewer for tiny clusters). JEV uses the benchmark's named cropped gallery for
at most 15 rock entries and its choice panel for larger legends. A colour legend
with a consistent close fit across every member (d1 ≤ 12, margin ≥ 15) needs no
JEV call. Disagreements and noise receive individual calls, longest intervals
first, with at most 400 extra calls per log and the global cap enforced across
workers. All-different votes have no propagated label. Missing API results retain
the available fallbacks. Three API failures open a circuit; no automatic retries.

The default dry-run reserves **3,017 representative + 983 extra = 4,000 calls**
across the current 51 logs. It reports conservative per-log/family quotas without
loading models or making calls. Colour shortcuts, fewer clusters and content cache
hits reduce actual usage; unused quotas are not spent automatically. The validation
command has its own separate budget (up to 60 representative + 160 individual
calls for the current log06 queue). Monetary cost depends on the account's tariff;
the summaries record actual request counts. `--logs log06` or a space-separated
list of log names selects a subset. Cached models must already be installed:
model loading is local-only and never downloads weights.

Precedence is gold → fully covered, agreed Sol+Terra consensus → individual JEV →
cluster JEV → colour → S11. Every available source, including disputed consensus
and representative votes, remains in `suggestions[]` with its tier and reason.
High requires the selected JEV label to match an independent source, or three
unanimous representatives with minimum winning probability ≥ 0.8. These are
agreement heuristics; JEV's probabilities are not calibrated accuracy estimates.
Uncertain-first ordering remains low, medium, high, round-robin across logs.

Results land in `labeltool/data/prefill.jsonl`; features and request results live
under `labeltool/cache/prefill/`. Partial-log runs retain other logs' prior output;
reruns reuse successful requests and cached features. `calls` counts requests
directly charged to that interval; `cluster_calls` repeats the shared cluster cost
for reference and must not be summed. The adjacent summary JSON records the run
total. Nothing writes into the dataset or other jobs' output folders.

`--validate-gold` runs the pipeline on log06 with **gold and consensus withheld
as prediction inputs**, then uses `harness.score.score` to report `row_acc` on
gold-covered rows and calls used. Its separate `prefill.gold.jsonl` and summary
cannot overwrite the all-log prefill. The normal all-log run restores gold's
precedence; gold remains excluded from human review.

Merge creates an immutable, byte-for-byte `queue.orig.jsonl` before atomically
replacing the queue; repeated merges start from that original. Undo restores it:

```powershell
python -X utf8 -B labeltool/prefill.py --unmerge
```

Keep `queue.orig.jsonl` until finished: later merges intentionally use its IDs and
boundaries. Regenerating the queue requires a fresh prefill/backup pair.

Offline verification never needs credentials, child browser processes or network:

```powershell
python -X utf8 -B -m pytest labeltool/test_prefill.py -q -p no:cacheprovider
$env:LABELTOOL_PREFILL_MOCK='1'
python -X utf8 -B labeltool/prefill.py --validate-gold --max-calls 80 --workers 8
Remove-Item Env:LABELTOOL_PREFILL_MOCK
```

The explicit mock replaces both JEV and the embedding model, uses separate caches,
and marks outputs `mock: true`. Mock accuracy is not a performance measurement;
mock output is rejected by merge. Tests also inject a fake shared-client interface
to exercise real gallery rendering, probability parsing, caching and concurrency.

The builder checks for `harness/legend_v2.py` for each log and prefers it.
Shared caches are read and copied into `labeltool/cache/harness/` before use;
all harness cache writes are redirected there. Each build worker owns its OCR
instance. Nothing is written to harness, strategies, other GT folders or dataset.
`-B` also prevents imports from writing bytecode into those folders.

## Verification

```powershell
python -X utf8 -B -m pytest labeltool/test_labeltool.py -q -p no:cacheprovider
# With the server running:
python -X utf8 -B labeltool/verify_queue.py
python -X utf8 -B labeltool/validate_api.py --base http://127.0.0.1:18765
python -X utf8 -B labeltool/check_browser.py --base http://127.0.0.1:18765
# Optional actual Chromium fallback when Edge is denied by the Windows sandbox:
python -X utf8 -B labeltool/check_qt_browser.py --base http://127.0.0.1:18765
# Stop the server before removing only the marked API test events:
python -X utf8 -B labeltool/validate_api.py --cleanup
```

Tests use two real logs and tiny independent interval fixtures. API validation
checks the running server, edits, CSVs, stats, undo, randomized spot checks and
latency. Browser validation attempts headless Edge at desktop 1440×1000 and phone
390×844, then tests aspect ratio, zoom, keyboard controls and Stats with Playwright.
Screenshots, when the Windows sandbox permits Edge to launch, are written under
`labeltool/screenshots/`. Browser failures are recorded explicitly in
`browser-validation.json` and `browser-error.log`; no replacement image is
fabricated. `validation.json`, `pytest.log`, `data/summary.json` and `build.log`
retain measured results. The first-page API measurement includes fetching an item
and its actual crop; it is distinct from browser first-render latency.
The Qt fallback uses offscreen Qt WebEngine Chromium, saves `desktop-qt.png` and
`phone-qt.png`, and records its own timing, aspect, keyboard and layout checks in
`qt-ui-validation.json`. Those screenshots are explicitly not Edge screenshots.

Known limits: generic reduced-scale boundaries need human adjustment for thin
beds; OCR legend names and depth calibration need checking; FFT centre estimates
can miss mixed patterns in long intervals; tiny colour patches can be ambiguous.
Queue coverage includes blank/non-lithology spans, which can be marked Unknown.
This is a trusted local/team tool with one person's global undo, without accounts
or concurrent-user assignment. The server intentionally restricts image requests
to known logs and crops of at most 10,000 native rows.

## Measured on this machine

The completed all-log build took **450.561 seconds (7m 31s)**, reusing available
OCR caches; this is not a cold-cache benchmark. All 51 logs have v2 legends and
harness depth calibrations, with zero build errors. There are 73,567 reviewable
intervals (52,372 low; 9,731 medium; 11,464 high), plus 160 held-out gold intervals.

Nine pytest tests pass. The real API validation passed all four modes, stable
round-robin across successive batches, edited save, CSV rebuild, idempotency,
stats, gold exclusion, validation errors and undo. After the progress-index
optimization, page + item + crop fetched in 283.7 ms; subsequent item API calls
took 98.2 ms median / 109.4 ms maximum. The real Qt Chromium UI measured 219 ms
to the complete first item, 11.2 ms for a prefetched transition and 32 ms for
Enter-save-and-advance. Earlier uncached UI loading also completed below 2 s.
Desktop/phone aspect, overflow, boundary fields, zoom, keyboard pick/Enter/undo
and Stats checks pass. The saved screenshots were inspected and phone overflow
and crop prefetch order corrected. Headless Edge and Playwright could not launch:
Windows denied their crashpad/pipe creation with `Access is denied (WinError 5)`.
The inspected screenshots are Qt Chromium output, not Edge output.
