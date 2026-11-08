# Shared harness validation

Implemented the nine shared API modules, `harness/_dummy.py`, the reproducible
`harness/_check.py` calibration audit, and `harness/test_score.py`.

Commands executed from the bake-off root:

```
python -m unittest harness.test_score -v
python -m harness._check
python -m harness.run --strategy _dummy --set eval --gt gold
python -m harness.run --strategy _dummy --set synthetic
python -m harness.report
```

Seven tests pass: perfect agreement including case/space normalization, shifted
boundaries inside/outside tolerance, blank truth/predictions, one-to-one matching,
matched-boundary depth errors and absent depths, window clipping, and loading a
silver-style CSV with only top_y/bottom_y/lithology. Perfect gold self-prediction
also returns row_acc = boundary_f1 = cluster_ari = score = 1, depth MAE = 0.
Seeded synthetic generation is repeatable and its self-prediction score is 1.

The eval run writes 13 CSV/PNG pairs, scores only the supplied log06 gold truth,
and renders the 12 logs without gold truth. The synthetic run writes and scores
39 CSV/PNG pairs. Both finish with zero crashes and zero API calls. The trivial
one-interval baseline has no internal boundaries, so its boundary F1 and score
are zero. Full metrics are in `out/_dummy/metrics.json` and
`out/_dummy/synthetic/metrics.json`.

All seven required legend debug figures were opened in the image viewer; see
[legend_quality.md](legend_quality.md). Calibration measurements for all 13 logs,
plus the tested GPU EasyOCR alternative on log06, are in
[depth_check.md](depth_check.md). `perfect_pipeline.png` was opened to check source,
prediction, GT, green/gray error strip, depth labels, legend highlights and stages.

TIFF segment decoding avoids Pillow's hidden full-page load for supported striped
TIFFs. Its legend crops match Pillow pixel-for-pixel on all 13 eval logs, including
palette lookup and 1-bit fallbacks. Crops convert to RGB at no more than 4000 rows
per operation. Rio Nueces' strategy-call process peak dropped from about 1.45 GB
to 0.69 GB after adding segment decoding. Process RSS includes shared libraries,
OCR models and the currently open source; it is not incremental strategy RAM.

The dashboard uses only `out/*/metrics.json` so there is one leaderboard/scatter
entry per strategy. Synthetic metrics remain separate. Image links are relative
and URL-encoded, including spaces and the `#` in gb506#2 filenames.

Known limits:

- OCR spellings are preserved and some minor label errors remain (see audit).
- Sparse UNKNOWN question-mark keys in the Rock Types legends are omitted.
- Unsupported TIFF compression or very large strips use Pillow's native-mode
  page allocation, followed by per-crop RGB conversion. The unsupported CCITT
  logs stay in 1-bit native storage rather than allocating a full RGB page.
- GPU peaks measure the PyTorch reserved allocator; non-PyTorch GPU allocations
  are not included. RAM is sampled every 20 ms and includes child processes.
- Calibrations fit one affine depth scale per page; nonlinear scan warping is
  represented by the reported residuals rather than a piecewise calibration.
- The 51-log full mode and consensus mode are implemented but were not run as
  full dataset experiments. Evaluation and synthetic modes were exercised.

No files in `strategies/`, `gt/`, the source dataset, or root `eval_set.json` were
edited. Synthetic images/truth are generated in memory; predictions go to `out/`.
