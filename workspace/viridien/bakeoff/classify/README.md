Run from the bakeoff folder. The larger benchmark is prepared with:

```
python -X utf8 classify/sample2.py --dry-run
python -X utf8 classify/sample2.py
python -X utf8 -m unittest classify.test_sample2 -v
python -X utf8 classify/sample2.py --label --bench --max-calls 124 --workers 6
```

The first two commands make no model calls. Dry-run prints family counts, tier
and height allocations, every log's count, sheet counts and remaining quota.
The second command writes `out/sample2.jsonl` and `out/sample2_plan.json`.
The fixed seed is 5822026. Height strata are family-wide empirical terciles.
Sampling maximises log coverage, balances available tiers, balances log counts,
then balances height strata within tiers. Missing tiers redistribute their quota.

The current queue supports 405 patches across all 50 non-gold logs. MC761 can
provide only 45 because its three logs are each capped at 15. The other six
families provide 60 each. Keeping sheets within one log requires 71 sheets and
142 reference calls, exceeding the 124 calls remaining at ledger usage 96/220.
The cap therefore leaves a partial reference set. It may also leave no quota
for Terra classification; use separate `--label --max-calls N` and
`--bench --max-calls N` invocations to reserve a portion for classification.
`--max-calls` limits NEW Codex launches in that invocation, across both stages;
the shared ledger's limit is always at most 220. JEV requests use their existing
separate audit, not the Codex call cap.

Sol/high and Terra/medium independently read the same sheet without seeing each
other's answers. Sheets show complete native intervals, bracketed outside the
source pixels, plus 40 rows of context on each side (clipped at column bounds)
and the original legend. Vocabulary and local swatches use legend_v2. Only exact,
non-unknown agreements become `out/sample2_reference.jsonl`. Agreement rates
use completed pairs as their denominator; the audit also records planned counts.
`out/sample2_status.json` distinguishes a finished pass from a quota/failure stop.

All reader responses, including failures, are cached by image/prompt/model/effort
hash in `out/sample2_responses/`. Failed responses are never retried. New calls
stop after a failure; in-flight answers are saved. Reference and SAMPLE2 Terra
calls share `labelling/out/calls.json` with the original labelling job, including
failed launches. A file lock protects reservations and completion updates.
An interrupted process may leave `calls.sample2.lock`; remove it only after
confirming no labelling or SAMPLE2 process is still running.

SAMPLE2 Terra uses a fixed hash-based 25% of references per family, rounded down.
Unselected patches are absent from its denominator. All other classifiers use
every agreed patch; the hybrid remains calibrated on TRUE. If no TRUE results
exist, the hybrid uses a fixed FFT policy without JEV escalation. Existing TRUE
and CONSENSUS result rows are retained by `--bench`. SAMPLE2 can also be run with
`python -X utf8 classify/bench.py --sample2`.

The report leads with SAMPLE2 and includes planned/paired/agreed coverage, log
counts, 95% Wilson top-1 intervals, top-3, macro F1, latency and calls. A family
gets a classifier recommendation only when its best top-1 Wilson lower bound
exceeds the runner-up's point estimate; otherwise the recommendation is `tie`.
These patch intervals do not account for correlation within logs. Machine
agreement is not independent gold, and Terra shares a model with the readers.

The original benchmark remains available with `python -X utf8 classify/bench.py`.

The default run evaluates TRUE, consumes agreed CONSENSUS intervals that exist,
and polls every three minutes for up to 100 minutes if the upstream summary is
absent. The manifest is read from `../analysis/mix_manifest.json`.

Useful commands:

```
python -X utf8 classify/bench.py --true-only
python -X utf8 classify/bench.py --retry-failed
python -X utf8 classify/bench.py --report-only
python -X utf8 classify/check.py
python -X utf8 classify/render_check.py
```

`--retry-failed` reuses successful outputs and retries unavailable ones. Previous
API responses remain archived, and the persistent Terra ledger enforces a total
of at most 60 CLI invocations across reruns. Without this flag failures are
cached too. No credentials are printed or cached.

`out/results.csv` has one row per selected patch and classifier. Unavailable
classifiers have empty predictions, probabilities `{}`, and a status reason.
`out/summary.csv` includes all requested CONSENSUS families, using n=0 and
empty metrics if upstream truth is unavailable. `n_scored` counts returned
predictions. `rank_of_truth` is 1-based; local probabilities are uncalibrated
softmax similarities. Macro-F1 averages classes observed in truth. If some
predictions fail, those rows count wrong; an entirely unavailable cell is NA.

`out/api_cache/` retains response JSON and input PNGs, `out/calls.jsonl` records
actual attempts, and `out/prediction_cache/` retains original timing and memory
measurements. `out/run.json` records hardware, upstream status, costs and caveats.
The CPU/GPU startup, legend embeddings and swatch-only CNN training are included
in classifier latency, amortized over patches. Shared TIFF/legend preparation
is excluded. API failure times are not successful inference measurements.

The hybrid uses DINOv2 or whichever local wins on TRUE, a top-two probability
margin gate tuned only on TRUE, and fixed JEV choice. Its TRUE score is fitted.
Without valid TRUE JEV responses, the zero-call tie-break gives a local-only
policy; no hybrid gain can be inferred.

`report.html` is self-contained, with embedded WebP images. `out/heatmap.png`
separates SAMPLE2, TRUE and CONSENSUS. `out/examples_true_d582.png` contains six TRUE
examples; `examples_<manifest-key>.png` contains consensus examples when truth
exists, or a clearly labelled n=0 placeholder otherwise. `out/edge_check.json`
records the requested 390×844 headless Edge attempt and whether a screenshot
was actually produced. An Edge exit code of zero alone is not a passing check.

Current completion state and measured availability are recorded in `out/run.json`
and `out/summary.csv`; no screenshot or remote execution is needed for the offline
SAMPLE2 tests.
