# Integrated-main review — 8 November 2026

Reviewed main: `b790a6b`, which merged the adapter, export, visual evidence and
local-download work while the earlier demo review was underway. Final tested
code: `8db8b6b82453236fee119940062444d527ccd6d0`. The earlier
[0072114 measurements](RETEST-0072114.md) are historical; the results below
cover the integrated code and fixes. [Machine-readable evidence](retest-b790a6b.json)
records versions, seeds, source fingerprints, counts and model diagnostics.

## Final executable evidence

| Check | Result | Wall time / sampled tree RSS |
| --- | --- | --- |
| Complete software suite, quick/50 iterations, core + demo regressions | **841 passed, 5 skipped, 0 failed/errors** | 96.91 s / 364.85 MiB |
| Clean upgraded core environment, same core checks | **494 passed, 5 skipped, 0 failed/errors** | 90.86 s / 286.06 MiB |
| Second-seed soak, 3,000 iterations per seeded loop | **260 passed, 2 skipped, 0 failed/errors** | 23.75 s / 238.30 MiB |
| Real OCR crop through layer 1 and backend pipeline | **1 passed**, not skipped | 11.44 s including test setup |
| Browser scenarios against built frontend, mocked API | **7/7 passed**, no uncaught errors | JSON contains case timings |
| TypeScript, frontend build, package build, `pip check`, actionlint | Passed | — |

The complete software run collected 846 cases and explicitly deselected the
real-OCR case; it was executed separately. Skips are unavailable Windows
symlink privileges. Hardlink tests ran. The core and soak repeat subsets;
do not sum them as unique coverage. Final runner records confirm unchanged
source and no unconfirmed child cleanup. The clean core uses NumPy 2.2.6;
the demo uses NumPy 1.26.4 in its separate locked environment.

Earlier broad runs were invalidated by changes during review. One extended
local attempt timed out and is **not** counted as passing. Low available RAM
was observed afterward; the precise stall cause was not isolated. Those raw
attempts remain in ignored `.runs/`. Passing evidence here is the final full
quick run, selected second-seed soak, focused tests and separate real smoke.

## Defects fixed

The [demo review](RETEST-0072114.md) covers upload races, conflicting bbox input,
request-body limits, queue admission/rollback, invalid CSV values, thin-bed
depth precision, aggregation overflow, report complexity and frontend polling.
The newly integrated code added these further reproductions and fixes:

| Boundary | Defect and corrected behavior |
| --- | --- |
| Candidate features | A valid component cap of one produced an empty pair-distance mean/NaNs. It now returns finite features; 56 new candidate edge cases cover geometry, caps, privacy and determinism. |
| Visual publication | Linked report/evidence targets could overwrite outside files or immutable score artifacts. Atomic publication rejects symlink/reparse parents and hardlinked destinations. |
| Image links | Raw local filenames containing URL syntax could render the wrong image or become remote/executable links. PNG-only portable relative paths are validated and URL-encoded. |
| Saved score consistency | Matching boundaries alone could conceal changed predicted labels. Evidence recomputes supervised metrics/confusion/depth against verified sources and marks discrepancies inconsistent. |
| Rendering costs | Quadratic diagnostic pairing/span scans became a heap with cached spans. Explicit interval/candidate/CSV/image budgets prevent uncontrolled expansion. Disabled previews avoid new decoding; timings include report creation. |
| Reviewed labels | Duplicate legend IDs, case-folded/device filenames and off-page geometry could create wrong or nonportable truth/provenance. Strict schemas reject them before publication. |
| Crop lineage | A same-size arbitrary crop could claim a catalog parent. Export now compares bounded RGB pixels and parent column annotation, with consistent source snapshots. |
| Export batch memory | Retained images and documents could grow into many GiB. Limits now apply before the next decode/parse: 128 MiB accumulated RGB, 32 MiB cumulative source JSON, 16 MiB per JSON and 1,000 windows. Split larger exports into batches. |
| Policy integrity | Local overlay/policy changes could evade registry-string checks. Metadata is anchored to committed SHA/size pins and exact POLICY bytes in the pinned cached ZIP. |
| Protected directories | Development and demo/check/run inputs now reject scorer-only directories and descendants before creating files or invoking a model. Held-out references still require explicit opt-in and a separate scorer location. |
| Download/local workflow | Portable asset/cache paths and bounded lock entries, unique pending files and redacted download failures are checked with tiny mocks. Software checks use the core stress runner without requiring demo dependencies. |
| Windows cleanup | Exited/reused PIDs are filtered by process identity before waiting, so unrelated new processes cannot become false cleanup failures. |

Independent Astra review found the protected-demo and cumulative-memory gaps;
their small regression fixtures confirm rejection before input generation,
extra decoding or extra parsing. No private dataset download, training or model
API service was used for these guard tests.

## Dependency advisories

The audit reported hits in five older core pins. Repeated advisory IDs/aliases
were present, so raw hit counts should not be interpreted as distinct exploits.

| Package | Previous | Updated |
| --- | --- | --- |
| Pillow | 11.3.0 | 13.0.0 |
| pytest | 8.4.2 | 9.2.0 |
| setuptools | 80.9.0 | 84.1.0 |
| wheel | 0.45.1 | 0.46.2 |
| Pygments | 2.19.2 | 2.21.0 |

After updates, `pip-audit` reported no known vulnerabilities and no skipped
packages for the scanned core pins, legacy pins, demo resolved lock and the
clean core environment's public resolved dependencies. The private editable
project was excluded from that public registry scan; its source was tested
directly. `npm audit` reported zero advisories for the frontend lockfile.
These are registry snapshots, not proof that every package or application path
is safe. The demo lock also aligns both OpenCV distributions below 4.11.

## Paired constructed model probes

Both models completed the same nine procedural cases: seed 2026, height 800,
one case per degradation, `supplied_legend`. The strict comparison command
accepted dataset/scorer/mode/case fingerprints for every pair.

| Degradation | Configured baseline interval score | Notebook ink candidate |
| --- | ---: | ---: |
| Clean | 0.1328 | 0.9675 |
| Blur | 0.0000 | 0.9675 |
| Noise | 0.1241 | 0.9675 |
| Low contrast | 0.0000 | 0.9675 |
| JPEG | 0.1143 | 0.8042 |
| Horizontal shear | 0.1328 | 0.9675 |
| Downsample | 0.0000 | 0.8042 |
| Occlusion | 0.1328 | 0.9675 |
| Mixed | 0.0000 | 0.4738 |

The candidate met the illustrative 0.20 threshold in 9/9; the baseline missed it
in 9/9, while both passed operational checks. The baseline command deliberately
returns exit 1 for that accuracy gate. These dependent constructed examples use
extra public legend assistance. They do **not** establish a real-data winner,
fresh held-out accuracy or official challenge performance. Short white gaps
remain a documented candidate limitation.

## Reproduce and remaining limits

See [the guide](README.md) for separate core/demo environments and browser setup.

```sh
python stress_testing/run_stress.py --profile quick --include-regressions --include-demo --timeout 600
python stress_testing/run_stress.py --profile soak --seed 9122026 --suite metrics --suite adapter --suite visual --suite exchange --suite demo-config --suite demo-pipeline --suite demo-services
python scripts/local_test.py --no-bootstrap --check
```

Use the opt-in real-model command from the guide or the backend's existing
marked test. CI runs core and demo software suites plus browser probes on the
merge context. Native image/OCR work still has no hard whole-job memory/deadline
isolation; queue admission does not bound all concurrent decode memory. Atomic
file publication assumes exclusive caller control against hostile races.
All 54 scans and unseen-well accuracy were not benchmarked. Browser map/API
responses are mocked, and whole-process resource measurements are sampled.
