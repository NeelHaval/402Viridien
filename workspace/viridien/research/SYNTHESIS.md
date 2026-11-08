# Mud-log pipeline synthesis

The first three upgrades below total roughly 15 hours, leaving time to integrate and evaluate. Rankings are engineering judgments, not measured gains: none of the sources establishes performance on these 51 logs. `r1.txt` was absent; the identical task context in `r1_layout.txt` was used.

## Spot-check: 10 consequential claims

VERIFIED means the cited source supports the specific claim, not that its benchmark transfers to mud logs. Model files were listed online, not downloaded or executed.

| Claim in notes | Check |
|---|---|
| R1: PP-DocLayout-S 70.9 mAP@0.5, Apache-2.0 | **VERIFIED**: [card](https://huggingface.co/PaddlePaddle/PP-DocLayout-S) reports 70.9 on its 500-image document dataset and lists the licence. |
| R8: Florence-2-large-ft TextVQA 81.5 | **WRONG**: [Microsoft's table](https://huggingface.co/microsoft/Florence-2-large) reports 73.5 TextVQA and 81.7 VQAv2. |
| R1: Qwen2.5-VL-3B licence is non-commercial | **VERIFIED**: the checkpoint's [licence](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) limits use to research/evaluation without a commercial licence. |
| R8: Qwen2.5-VL-3B DocVQA 93.9 on its official card | **UNVERIFIABLE**: [checkpoint card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) does not show that number; this does not disprove it elsewhere. |
| R8: Phi-4 multimodal DocVQA 93.2 | **VERIFIED**: [Microsoft card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) reports 93.2; vendor benchmark, not hatch recognition. |
| R8: SmolVLM2-2.2B DocVQA 79.98, Apache-2.0 | **VERIFIED**: both appear on its [card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct). |
| R7: `facebook/dinov2-small` downloadable, Apache-2.0 | **VERIFIED**: [model card](https://huggingface.co/facebook/dinov2-small) lists the checkpoint, files and licence; local download untested. |
| R2: `ruptures` licence unknown | **VERIFIED** as now resolved: [upstream](https://github.com/deepcharles/ruptures) specifies BSD-2-Clause. |
| R1: DocLayout-YOLO inference repository AGPL-3.0 | **VERIFIED**: [repository](https://github.com/opendatalab/DocLayout-YOLO) lists AGPL-3.0; checkpoint terms are separate. |
| R6: pyLogDigitizer source is AGPL-3.0 | **VERIFIED** for the [repository](https://github.com/GeoLarryLai/pyLogDigitizer); R5 reports conflicting PyPI metadata, so resolve before reuse. |

## Ranked upgrades

Steps: 1 boxes, 2 legend, 3 boundaries, 4 grouping, 5 naming, 6 depth, 7 efficiency. Hours are estimates; arrows denote hypothesized task gains. Evidence is the cited note's method/source, not a measured gain. Rank accounts for impact, implementation time, dependencies and the optional nature of B1.

| Rank | Upgrade | Step | Improves | Expected gain and evidence | Hours | Risk |
|---:|---|---|---|---|---:|---|
| 1 | Sparse numeric OCR + tick-checked RANSAC | 6 | S10 | Accuracy ↑: robust outlier fitting, R5/R6; no log benchmark | 4–5 | False coherent scale |
| 2 | FFT hatch + Lab swatch matching, reject unknown | 5 | S1, S3, S10 | Accuracy ↑ for phase-shifted same-colour patterns, R3/R6 | 5–6 | Tiny swatches, overprint |
| 3 | Legend geometry + OCR pairing | 2 | S8, S9, S10 | Accuracy ↑ and calls ↓: row geometry, R4 | 4–5 | Nonrectangular symbols |
| 4 | Batched FP16, cache swatch embeddings | 5, 7 | S5, S6 | Speed ↑, cost ↓: reuse existing encoders, R7 | 2–4 | FP16 ranking drift |
| 5 | Bounded TIFF bands and native coordinates | 1–7 | S1–S10, B1 | RAM ↓; speed potentially ↑: demand-driven decode, R7 | 3–5 | Compressed-strip I/O |
| 6 | Confidence-gated existing JEV | 2, 5 | S8–S10 | Calls/cost ↓: one representative per ambiguous cluster, R7 | 2–3 | Confident cheap-stage errors |
| 7 | Multiscale LBP/Gabor + PELT | 3 | S1, S2, S4 | Boundary accuracy ↑ for hatches: texture descriptors, R2 | 3–6 | Lines, thin beds |
| 8 | DINO patch prototypes + texture voting | 5 | S5, S10 | Accuracy ↑: local motif votes, R3 | 2–5 | Patch resolution |
| 9 | HDBSCAN with unknown/noise | 4 | S4, S10 | Accuracy ↑: avoids fixed k, R2 | 2–4 | Singleton classes |
| 10 | Scattering features + L2 PELT | 3–4 | S2, S4 | Accuracy ↑: orientation/frequency features, R2 | 6–10 | Windows setup, runtime |
| 11 | Rule proposals + JEV tie-break | 1 | B1, S8 | Box recall ↑: native-width geometry, R1 | 6–10 | Unruled columns; bonus only |
| 12 | PP-OCRv5 on difficult crops | 2, 6 | S10 | Accuracy ↑: vendor 85.25% English recognition, R4; domain differs | 3–5 | Paddle Windows/CUDA |

## Build next

The harness is not present in this research folder. Filenames below are proposed: inspect its registry and existing data contracts first, then register the indicated IDs without altering baselines. Commands must be run separately with `--set synthetic` and `--set eval`; `--gt gold` is evaluation only. Retain original y coordinates and the required interval schema.

1. **`depth_ransac`** — Add `harness/strategies/depth_ransac.py`, wrapping S10; reuse RapidOCR/EasyOCR, add `numpy`, `opencv-python`, `scikit-learn`. OCR native-resolution depth bands, record numeric anchors and global y; find nearby tick lines. Fit `RANSACRegressor` for depth versus y, refit inliers, check increasing depth, tick spacing and at least three anchors; report residuals and unknown depth on failure. Test `python -m harness.run --strategy depth_ransac --set synthetic --gt gold` and repeat with `eval`. Compare endpoint depth error, inlier overlays and runtime against S10; test bad digits and scale breaks.

2. **`fft_legend`** — Add `harness/strategies/fft_legend.py`, wrapping S10 boundaries; use `numpy`, `scipy`, `opencv-python`. Crop swatch interiors and several interval windows, preserve scale and orientation, remove obvious borders, Hann-window FFT magnitudes, compare directional frequency bins plus ink fraction and Lab colour; suppress colour for 1-bit. Aggregate by median, expose top-two margin and unknown. Test `python -m harness.run --strategy fft_legend --set synthetic --gt gold` and repeat with `eval`; compare S1/S3/S10 depth-weighted accuracy, per-class accuracy, unknown rate and phase-shift/overprint cases.

3. **`legend_pairs`** — Add `harness/strategies/legend_pairs.py`, wrapping S10; reuse existing OCR plus `opencv-python` and `scipy`. OCR the provided legend at native and 3× scales; find rectangle and non-text component candidates, merge multiline labels, pair using vertical alignment and Hungarian assignment with unmatched options. Keep boxes, raw text, confidence and rock/modifier/unknown; send ambiguous items to existing JEV only. Test `python -m harness.run --strategy legend_pairs --set synthetic --gt gold` and repeat with `eval`; audit pairing overlays, rock/modifier errors and calls against S10.

Evaluate synthetic cases separately from the single 17-interval labelled example; manually inspect unseen monochrome, palette, blank and same-colour logs. Report boundary error, depth-weighted accuracy, wall time and peak RAM/VRAM. Do not tune on the entire evaluation set.

## Disagreements and decisions

- **Local VLM as primary versus adjudicator:** R8 prefers Qwen; R4/R7 favour OCR and cheap matching first. The latter has the stronger task fit: document QA scores do not validate hatch matching, and Qwen's checkpoint has research-only terms. Compare per-cluster only if time remains.
- **Scattering versus conventional texture:** R2 favours scattering; R3/R6 favour FFT/LBP. Neither has target-domain evidence. Start FFT for direct phase-tolerant legend matching, then test scattering for boundaries if time remains; these address different steps.
- **Whole-page Florence proposals versus native bands:** R8 suggests a downscaled page for proposals; R1's geometry supports native-width bands. A 63-pixel strip becomes about 1.3 pixels on a 50,000-to-1,024-pixel reduction, so full-page proposals are unreliable for narrow tracks.
- **TIFF memmap:** R5 suggests it for uncompressed/tiled images; R7 correctly limits direct mapping to compatible contiguous uncompressed layouts. Inspect each TIFF; compressed or tiled data may need segment selection and codecs.
- **OCR engine:** R5 proposes PP-OCR first; R6 reuses existing OCR. With Windows setup and no domain comparison, reuse existing engines plus robust calibration first; A/B PP-OCR on failures.
