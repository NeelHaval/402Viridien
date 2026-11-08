# Historical repository guide

This is the earlier guide. Use [START_HERE.md](../START_HERE.md) for current setup.

# Viridien challenge: lithology segmentation and clustering

Work so far on "Decoding the Subsurface". For each scanned mud log, the goal is a CSV of contiguous intervals (`top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id`) plus an annotated image.

## Start here

| file | what it is |
|---|---|
| `artifact/lithology_explainer.html` | Visual explainer for the dataset, all the strategies, bounding-box examples and their trade-offs. Images are embedded, so it opens offline. |
| `analysis/label_mix_examples.html` | The share of each label per legend, a difficulty score per legend family and one worked example per family. |
| `lithology_explainer.ipynb` | Notebook with the same explanations, code and HQ images (30 MB). |
| `research/SYNTHESIS.md` | State-of-the-art research on each pipeline stage, with a ranked list of upgrades. |
| `bakeoff/SPEC.md` | The bake-off harness spec. |

## Layout

- `bakeoff/harness/`: loading, legend parsing, depth calibration (RapidOCR + RANSAC), scoring, synthetic data and the runner. Run from `bakeoff/` with `python -m harness.run --strategy s11 --set eval --gt gold` (other `--gt` values: `silver/A`, `silver/B`).
- `bakeoff/strategies/`: 11 strategies `s01`-`s11` plus `b01_jev_boxes` (JEV vision boxes). JEV needs `JEV_API_KEY` in your environment; it is never stored here.
- `bakeoff/gt/`: the ground truth. `gold` is the organisers' example CSV for log06, the only true answer. `silver/A` and `silver/B` were made by AI labellers and are not truth.
- `bakeoff/out/`: per-run CSVs and `metrics.json` files. The annotated PNGs are left out (about 300 MB); rerun a strategy to regenerate them.
- `scripts/`: figure, notebook, page and label-mix builders. `analysis/`: label-mix and difficulty CSVs.

## Not included

- The challenge dataset (TIFFs): get it from the organisers' zip.
- Renders of the dataset pages used by labeller B (`notes/label_B/views`).

## Status (7 Nov 2026)

On log06, the best boundary F1 is about 0.85, but naming the 1-bit hatch rocks still fails for every strategy.

The fair re-run of all strategies against gold and both silver sets is complete: `bakeoff/out/_rerun/summary.csv`.

Main known gaps:
- The legend parser fails on the Gulf 3-panel layout, which is a third of the column length.
- There is no hatch-pattern naming yet.
- The label-mix estimator ignores grey fills.

## Labelling, benchmark and annotation tool (2026-11-07)
- `bakeoff/harness/legend_v2.py`: legend parser that now reads all 51 legends (167 more swatches on 11 logs).
- `bakeoff/labelling/`: Sol+Terra LLM labeller (`llm_label.py`), per-window consensus in `out/<key>/`, `review.html`.
- `bakeoff/classify/`: classifier benchmark (colour, FFT, DINOv2, SigLIP, CNN, JEV choice/gallery, Terra); `report.html`.
- `bakeoff/labeltool/`: human labelling/recheck tool (`python labeltool/serve.py`, port 8765). The 47 MB `data/queue.jsonl` is not committed; rebuild it with `python -X utf8 labeltool/build_queue.py`. Best-guess prefill is in progress.
- `analysis/volume_budget.csv`, `scripts/volume_budget*.py`: rows to analyse and accuracy needed per legend type.
- Raw API response caches are not committed.
