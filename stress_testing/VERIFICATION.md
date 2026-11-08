# Executed verification — 8 November 2026

These are actual local results on Windows 11, Python 3.13.4, NumPy 2.2.6,
Pillow 11.3.0, pytest 8.4.2 and psutil 7.0.0. GitHub Actions uses Python 3.12.
Machine-readable measurements and the tested source-file fingerprints are in
[verification.json](verification.json). Timings include concurrent verification
load and should not be treated as production throughput estimates.

| Command/check | Result | Time | Sampled process-tree peak |
| --- | --- | ---: | ---: |
| `run_stress.py --profile extended --include-regressions` | 327 passed, 3 skipped, 0 failed/errors; 500 iterations per seeded loop | 156.59 s | 281.44 MiB |
| `run_stress.py --profile soak --suite metrics --seed 9122026` | 13 passed, 0 skipped/failed/errors; 3,000 iterations per seeded loop | 6.10 s | 116.60 MiB |
| Built source archive, `--suite notebook --suite harness --iterations 1` | 30 passed, 0 skipped/failed/errors | 4.92 s | 209.34 MiB |
| Normal regression suite separately | 172 passed | 24.35 s | Not measured |
| Wheel + source archive build | Passed; scripts/guide/notebook included, `.runs/` excluded | — | — |
| `actionlint .github/workflows/stress.yml` and `git diff --check` | Passed | — | — |

The extended run includes the 172 normal regressions and 158 isolated stress
cases (155 passed, 3 skipped). The additional runs repeat some checks; do not
sum them as unique coverage. All three skips are temporary symlink creation
privileges unavailable to this Windows process. Their owned-directory link
escape/packaging behavior still needs execution on a symlink-capable host.
The final extended and metric soak reports confirm unchanged source during
execution and no unconfirmed descendant cleanup.

## What the model stress probes exposed

The unaided procedural baseline completed all nine degradations at height 800.
Its interval score was zero throughout because it emits `Unknown` without a
legend/OCR naming mechanism. No accuracy gate was configured for that run;
operational success is not an accuracy claim.

A separate `supplied_legend`, height 128, seed 2026 diagnostic deliberately set
`--min-score 0.20`. All nine operational checks passed, but four accuracy gates
failed, so the command correctly exited 1:

| Degradation | Interval score | Meets 0.20? |
| --- | ---: | --- |
| Clean | 0.3188 | Yes |
| Blur | 0.1289 | No |
| Noise | 0.2983 | Yes |
| Low contrast | 0.0000 | No |
| JPEG | 0.3281 | Yes |
| Horizontal shear | 0.3188 | Yes |
| Downsample | 0.1309 | No |
| Occlusion | 0.3188 | Yes |
| Mixed | 0.0000 | No |

These are one constructed case per degradation, with an assisted legend. They
are useful failure probes, not estimates of real scanned-log generalization.
Repeat with frozen human/external manifests and report those separately before
claiming model improvement.

## Defects reproduced and corrected

- Malformed top-level manifest/annotation JSON and non-object records now receive
  controlled validation errors.
- Saved result validation rejects removed per-class aggregates/entries.
- Worker responses have log/output budgets, bounded diagnostic tails, validated
  error fields and handling for deeply nested JSON. Cleanup races do not erase
  the report; unconfirmed live descendants prevent a pass.
- Dataset packaging rejects symlinked references that would break portable
  manifests. Extraction rechecks post-staging destinations and rejects case-folded
  file/parent conflicts before commit.
- Dataset/model metadata cannot create raw Markdown headings, images or HTML in
  the checked evaluation and model-stress reports.
- The harness rejects missing/truncated/contradictory JUnit, all-skipped runs and
  changed source instead of producing a false passing result.

Known scope limits, including legacy notebook malformed-record handling,
sampled resource limits, trusted plugins, local archive race boundaries and the
absence of a dependency CVE audit, remain explicit in [README.md](README.md).

Raw reports are local ignored artifacts under `stress_testing/.runs/`; reproduce
the commands above to create your own JUnit, logs and provenance. The curated
JSON preserves the evidence needed to compare tested source with a checkout.
