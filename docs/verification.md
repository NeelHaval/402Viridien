# Executed verification

For the subsequent v0.2.0 hardening loop, see [172-test contributor/package verification](review-loop.md) and [fresh model statistics](benchmarks/2026-11-08.md). The records below describe initial setup verification.

Date: 8 November 2026. Implementation used three GPT-6.1 Sol builders/progress agents and two independent GPT-6 Astra reviewers. Reviews found and corrected integration, leakage, resource-reading, packaging and reporting issues before these final checks. Historical findings remain in the review documents with their original scope.

## Hosted evidence

The code at `d68907b` passed [Windows and Ubuntu CI on Python 3.12](https://github.com/NeelHaval/viridian402/actions/runs/42446350561), including tests, package builds and the demo workflow. The original Windows-only temporary path assertion failure was corrected to compare resolved paths; it did not change imported dataset bytes.

These manual workflows completed successfully and published result artifacts:

| Run | Purpose |
|---|---|
| [Human labels](https://github.com/NeelHaval/viridian402/actions/runs/42446370056) | Actual two-page human interval snapshot, default boundary/cluster baseline |
| [Real crops](https://github.com/NeelHaval/viridian402/actions/runs/42446371695) | PR54 fixed 907-train/112-test crop classifier |
| [Organizer S11](https://github.com/NeelHaval/viridian402/actions/runs/42446373340) | Optional OCR/FFT dependency installation and frozen existing-model adapter |
| [Stress](https://github.com/NeelHaval/viridian402/actions/runs/42446375392) | Deterministic real-interval-derived noisy synthetic construction tests |

A successful run without a minimum score configured establishes execution/nonempty output, not acceptable accuracy. Download its artifact to inspect metrics, errors, predictions and annotated previews. The default baseline returns Unknown names without legend assistance. The crop prototype's local exact-set accuracy was 8.93%; its purpose is a reproducible starting point, not a competitive result.

## Local and packaging checks

- **65 tests passed** on Windows/Python 3.13.4 after PR54 integration and the cross-task training-well alias regression.
- The complete imported interval catalog validated **84 windows**, with 36 supervised and 48 unlabelled cases.
- The PR54 audit verified **2,961 assets** and fixed memberships: 907 usable train, 112 test, 9 holds; test statuses 60 source-prospective and 52 retrospective.
- Fresh authenticated downloads from the new repo's private evaluation and crop assets passed size/range/SHA-256 verification and extracted successfully. GitHub's server-reported SHA-256 digests match all three locked assets, including the optional 905 MB remaining-scan archive.
- Independent rebuilt wheel install and extracted-source tests passed. The source package includes scripts/configs/docs/constraints needed by its tests and examples.
- `actionlint` v1.7.12 reported no workflow errors. Its downloaded executable was checked against the upstream release checksum. `pip check` reported no broken installed requirements.
- Reviewers additionally checked 1,000 randomized interval examples against a raster accuracy oracle, import-time timeouts, observed descendant cleanup, held rows, duplicate pixels, synthetic long-bed clipping and truthful unlabelled metrics.

## Scope limits

The entire 48-case unlabelled challenge suite was packaged and validated, but not exhaustively benchmarked here. Sealed external tests are unavailable. Optional learned models beyond the frozen S11 adapter need their own implementation/checkpoint provenance and evaluation. GPU/paid API cost instrumentation is not provided. Python model plugins are trusted code, not a security sandbox. Package constraints and recorded installed versions support reproducibility; they are not a hash-locked container image.

Per-window challenge-format CSVs can omit depths when no valid calibration is available. They and their image overlays support experiments; a final competition submission still needs full document coverage and verified depths.
