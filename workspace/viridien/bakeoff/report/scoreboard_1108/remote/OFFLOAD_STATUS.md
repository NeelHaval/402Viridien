# S20 CPU work on boat

This directory holds a separate S20 result shard. It does not replace the local
runner checkpoint. The shared scorer is harness/score.py with min_layer_px=0.
This is not the other packet's interval-v2 scoring contract.

All 58 remote S20 inference cases have returned and have validated primary
shared depths. No pixel-only diagnostic remains pending. The 51 DEV windows
were promoted from cached intervals after their exact local contexts became
available. The separate unQA Hadrian view remains in its own suite. Original VM
receipts, CSVs, images and portable HTML are in
the numbered directories. validation.json records local strict-score replay.
policy_scores.json uses the owner's current grading_policy module on saved
CSV copies. It does not alter the original labels or run inference again.

The final 54 S20 inference cases completed in this bounded job:

    /home/user/viridian402-jobs/v4-cpu-20261108T063000Z-f785e5/scoreboard-s20-remaining

The saved timeout process was PID 2228168, started at 06:54:05 UTC. It acquired
the lock at 07:01:30.719 UTC after an observed 445.719-second queue wait. It
finished at 07:10:27 UTC. All 54 case exit receipts and the batch job.exit are
zero. The process is gone, and the exit observer found no private case workers.
The original timeout-process return code was not captured by its non-parent
pidfd observer; no code is inferred. The actual batch receipt is retained.
There was no retry or second waiter. The lock wait limit was 1200 seconds, the
case compute limit was 300 seconds, and the whole job limit was 1800 seconds.
Queue time and inference time are separate measures.

The batch started with Shenzi, Hadrian and Rio Nueces. Cases without a
validated initial shared depth calibration produce separate pixel-only
diagnostics. enrich_shard.py can add a later validated shared calibration to
their cached interval edges through harness.run._clean. It never calls the
classifier. It retains the original pixel CSV, model output, calibration
signature and an enrichment receipt. Before promotion, pending diagnostics did
not count as primary scored records. The final promotion pass preserved all
contacts, labels and cluster IDs exactly; only the two depth fields were filled.

The final key check found all 68 S20 cases: 58 remote and 10 retained local,
with no missing, extra or duplicate keys. All 58 primary strict scores replayed
from their saved CSVs, and no image render failure was found. Portable case
status captions were refreshed after the last promotion. The final consumer
manifest rebind is complete, at digest
1da7b757c9846b7cf1b7d79a1f01519195d3011a6c4041ac02c05f6798795301.
It does not change or replace original execution receipts. The final check
replayed 58 strict scores, found no render failures and no pending diagnostics.

Inputs are lossless, full-width native crops at the exact source coordinates.
The original image, annotation and truth hashes are kept in each receipt, as
are the native RGB hash, crop dimensions and original boxes. The frozen v4
native swatch cache and source hashes are unchanged. The CPU environment is the
existing private packet environment and is read only. It uses NumPy 2.2.6;
our local Python uses NumPy 1.26.4. The environments are not identical.

The one MC765 parity check produced exactly equal interval dictionaries and
strict scores on local and VM runs of the same decoded crop. This proves that
fixture only. It does not prove bitwise equivalence for all data or packages.
Remote timing starts from decoded window pixels; it cannot be used as a direct
speed comparison against local source-page loading or CUDA execution.

Each portable report has a readable detail image for the fixed first 1200
source rows, or the whole window if shorter, and a link to the full overview.
The rendering pass does not change inference, source hashes or original VM
result receipts. Its own receipt records the exact detail rows and PNG hash.

The full-column MC765 readiness case ran once after the 54-case batch, under
the same lock and a 300-second compute limit. Its supervisory whole exit and
case exit receipts are zero. It used box [1134,1623,1171,36273], a 34,650 by 37
pixel crop, and a shared calibration with 63 points. It took 1.063307 seconds
of decoded-crop inference and used 179,300 KiB peak RSS. Its 83 intervals cover
28,385 rows; sorted, nonoverlapping, in-bounds and finite-depth checks succeeded.
It is outside the 68-case matrix. Descriptive scoring used only approved truth
rows [7623,13623] and gave strict score zero. Artifacts are under
../readiness/mc765_s20/. No wider truth or sealed external split was opened.
No further VM worker was queued after this case.

S19 stays local. The inspected VM environment has torch 2.15.0+cpu and NumPy
2.5.4, and no cached torch 2.12 wheel was found. It is not the required pinned
environment. No shared environment was changed and no S19 case was reserved.
