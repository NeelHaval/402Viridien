# Figure sources and versions

These three figures use existing saved bytes. No inference, training, API request, or truth overlay was added.

| Figure | Source and transformation | Interpretation |
|---|---|---|
| `01-released-pages-s19.png` | All three approved released pages; fixed middle 768-row excerpt; crop x=0..568 of each existing source/S19 render, no rescale | Saved standalone predictions, no interval truth, S19 depth uncalibrated |
| `02-fixture-legend.png` | Exact fixture crop `[30,1700,650,2100]`, with saved CSV label/cluster pairs | Generated fixture, historical OCR app; no claim about final Strata |
| `03-fixture-source-prediction.png` | Fixture crop `[40,380,390,1620]`, resized to 235×831; boundaries and text from saved CSV | Generated regression check; not project accuracy |

The first figure removes only the separate Benedek panel. The saved S19 intervals and source pixels stay at equal vertical scale.
Its case IDs are `c27f24013e5cc155`, `49ab92322aa76b26`, and `fc5af26c8937097c`.
The existing selection used fixed excerpts without reference labels. Sparse output remains visible.
The older gallery's generic depth sentence is not evidence of S19 calibration; these saved S19 files have missing depth values.

The generated scan source SHA-256 is `4280a39e932fdf0e0d0a9ac1bca02b7372e41aeb32bf9ef97ffa551d747f109c`.
Its boxes SHA-256 is `ab8fd492868643be0e5f107bd07b1e42f6c037bd3eec75fc3e07952fa2e3fd3c`.
Its original receipt SHA-256 is `168c8334486c3caf2302ffce873c113e1daeaf30546e17e7bc448c6d3bd126ec`.
The generator/HTTP test SHA-256 is `1df5ed5e86c7cde917a58579fe6ebb3345c160dae58c14521df30f2feb9bb20e`.

Historical app source commit: `fcb8b319604200b999902b44753afb55b7da133d`.
Historical app image: `stratum:fcb8b3196042`.
Image digest: `sha256:5b938b2669b04dcf86220b6c3bff712904d45cf4fcc29848b9df8ec8bbd65451`.
This fixture was run through VM-local HTTP in an isolated container. It was not a browser test.
Core candidate `559c99cb4e24` and final Outfit/Strata/provider acceptance are separate.

The three internal fixture boundaries were checked from saved rows: drawn `700,1000,1300`; saved `696,1000,1304`.
Maximum absolute difference: 4 pixels. All four saved label strings match the fixture swatches.
This says nothing about unseen-scan accuracy. Only this fixture uses the reported feet calibration.

Read [evidence.json](evidence.json) for complete source/output hashes, crop coordinates, row ranges, and image dimensions.
[fixture-api.csv](fixture-api.csv) is the exact saved CSV text. [fixture-receipts.json](fixture-receipts.json) contains only this generated fixture's API responses.
[fixture-harness.csv](fixture-harness.csv) changes only column order and the lower-depth alias for the pinned exporter.

## Rebuild the figures from approved saved evidence

Use Python with Pillow and the same Arial/Arial Bold fonts. Supply the existing evidence workspace explicitly:

```sh
python -X utf8 docs/showcase/build_showcase.py --source-root APPROVED_WORKSPACE --review-dir PRIVATE_SCRATCH/review
```

The script expects the original `management/vultr-setup/test-swarm/ocr` and the approved three-page comparison under `viridien_work/bakeoff/report/scoreboard_1108/benedek_comparison`.
It checks fixture pins, uses recorded crops, and makes opaque white PNGs below 500 KB each.
`--font-dir` can point to the same font files on another platform; different fonts/Pillow versions may change PNG hashes.
No source scans or private model files are downloaded. A local contact sheet goes only to the explicit scratch path.
The output adapter is copied from the existing source without changes and has a separate pin in `docs/export-adapter.sha256`.
