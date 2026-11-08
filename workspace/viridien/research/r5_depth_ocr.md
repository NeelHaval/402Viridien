# R5 — depth OCR and row→depth calibration (research 2026-11-07)

## Decision

Do not OCR the whole 2,600–313,800 px TIFF. Treat the supplied depth box as a one-dimensional calibration problem: find horizontal tick/grid evidence, OCR only label candidates beside major ticks, and fit depth from many redundant anchors. This is the best route to sub-foot *mapping* error; exact sub-foot output also requires at least ~2–3 pixels/ft and a scan with no local stretch. "<1 s/log" is plausible only after TIFF decode / with region reads and batched crops; decoding a 500-Mpx TIFF alone can exceed it. Measure both decode and calibration time separately.

| Candidate | Source (opened; year) | Step | Why it may beat current bake-off | Downloadable now? weights + licence | Cost / size | Plug-in | Evidence actually seen |
|---|---|---|---|---|---|---:|---|
| PP-OCRv5 English/mobile **recognition only** | [official docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md), [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Cui_PP-OCRv5_A_Specialized_5M-Parameter_Model_Rivaling_Billion-Parameter_Vision-Language_Models_on_CVPR_2026_paper.pdf) (2026) | Read pre-cropped depth labels | Modern compact recognizer; unlike EasyOCR/RapidOCR full pipeline, no detector sees clutter. Batch 20–100 label strips; parse digits/decimal/sign afterwards. | Yes; official inference/training links; code Apache-2.0 ([licence](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)). | `en_PP-OCRv5_mobile_rec`: 7.5 MB, reported 85.25 average accuracy. General mobile: 16 MB; official static-rec benchmark 9.76 ms/crop, ONNX 4.91 ms/crop (not our 4080). | 3–5 h | Official score/speed, but not digit-only/mud-log data. CVPR reports 5M parameters and OmniDocBench mean edit distance 0.067; **no well-log benchmark**. |
| docTR CRNN-Mobile recognition-only | [model zoo](https://mindee.github.io/doctr/latest/models/recognition.html) (2026) | Fast fallback OCR | PyTorch-native, easy batching on existing Torch CUDA; CTC model is the appropriate cheap comparator for clean, aligned printed numbers. | Yes, pretrained; Apache-2.0 ([repo](https://github.com/mindee/doctr)). | 2.1M; docTR reports 0.05 sec/iteration (batch 64, i7-11800H). | 2–3 h | FUNSD/CORD exact word accuracy 87.25/93.91%; not well logs. |
| docTR PARSeq | [paper](https://arxiv.org/abs/2207.06966), [docTR benchmark](https://mindee.github.io/doctr/latest/models/recognition.html) (2022/26) | Rescue OCR for dirty/overprinted labels | Stronger context-aware recognizer; use only on PP-OCR low confidence/disagreement, not all labels. | Yes, weights; Apache-2.0 ([reference repo](https://github.com/baudm/parseq)). | 23.8M; docTR CPU benchmark 2.2 sec/iteration at batch 64. | 3–5 h | Paper: 91.9% STR accuracy (synthetic training), 96.0% with real data. docTR: 88.53/95.56% exact; domain mismatch. |
| TrOCR-small-printed | [HF model card](https://huggingface.co/microsoft/trocr-small-printed), [paper/results](https://github.com/microsoft/unilm/tree/master/trocr) (2021–23) | Last-resort label reader | Autoregressive vision/text priors can repair a damaged multi-digit crop. It is a useful oracle to arbitrate two cheap models, not a primary path. | Weights downloadable (Safetensors); **licence not stated on the opened model card—verify before use**. | 61.4M params, F32; published small model 95.86 SROIE F1. | 2–4 h | Real benchmark above, but checkpoint is fine-tuned on receipts (SROIE), not depth labels. |
| Horizontal tick/grid detector: threshold + morphology/CC; optional HoughLinesP | [OpenCV feature docs](https://docs.opencv.org/5.0/main_modules/imgproc_feature.html) (2026) | Anchor proposals, scale/break evidence | Deterministic O(H×Wdepth), no weights, and every minor tick supplies spacing even when text fails. Restrict to short horizontal components near the scale edge; Hough is optional because CC/morphology is faster here. | Yes; OpenCV Apache-2.0 (verify installed version). | CPU; no model. | 2–4 h | OpenCV documents binary-input Hough line detection; **no mud-log accuracy published**. |
| RANSAC affine fit + weighted least-squares refit | [scikit-learn docs](https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.RANSACRegressor.html) (2026) | Convert anchor y→depth; reject OCR errors | Directly tolerates a few misread labels, header/footer text and false tick associations. Use OCR confidence and label/tick alignment only after RANSAC selects inliers. | Yes; BSD-3-Clause (sklearn project; verify dependency licence if shipping). | CPU, tiny. | 1–2 h | Docs: robust estimation from inlier subset; threshold controls inliers. **No task benchmark.** |
| Theil–Sen slope + residual-change segmentation (piecewise affine) | [SciPy docs](https://scipy.github.io/devdocs/reference/generated/scipy.stats.theilslopes.html) (2026) | Stable slope / detect scans stitched or scale reset | Median pairwise slope is a cheap deterministic cross-check on RANSAC. Split only when persistent residual jump/slope change survives a minimum-anchor test; prevent fitting OCR noise. | Yes; BSD-style SciPy licence (verify release bundle). | CPU; O(n²), trivial for tens of anchors. | 2–4 h | SciPy states it is robust and uses median pair slopes; 95% slope interval available. **No log benchmark.** |
| pyLogDigitizer (domain reference, not dependency) | [GitHub](https://github.com/GeoLarryLai/pyLogDigitizer) (2026) | Design/QC reference | Only located live tool explicitly claiming TIFF well-log OCR, depth/scale calibration, grids and deskew. Borrow its anchor/QC idea, not its stack. | Source yes, but GitHub says AGPL-3.0 while PyPI says Apache-2.0 and pre-alpha: **licence conflict; do not embed**. | Optional Rust; unknown OCR/model cost. | 4–8 h to inspect | Claims only; repository warns experimental/not fully validated; no published depth-OCR numbers located. |

## Top 3 recommendations

### 1. PP-OCRv5 sparse anchors + tick-first RANSAC (ship first)

1. Decode only the depth ROI in 2,048–8,192-row bands (`tifffile.memmap` if uncompressed/tiled; otherwise `aszarr`/Pillow crop); retain native y offsets.
2. Gray-scale, CLAHE only if local contrast is poor, Sauvola/adaptive threshold; remove long vertical rulings with a 1×80 opening.
3. In a narrow strip on the tick side, use a horizontal 15–60×1 opening and connected components; retain near-horizontal components and cluster their centre-y values.
4. Estimate dominant tick period from differences/autocorrelation; label every 5th/10th candidate as a likely major tick and crop the adjacent number zone.
5. Run `paddleocr` v3 `TextRecognition(model_name="en_PP-OCRv5_mobile_rec")` on all crops as batches; do **not** run text detection.
6. Normalise `O→0, I/l→1, S→5` only when the whole string matches a numeric grammar after substitution; retain raw text and confidence.
7. Associate each parsed label to nearest major tick (or its text baseline); fit `depth = a*y+b` with `RANSACRegressor(LinearRegression(), residual_threshold=1.0)` in depth units.
8. Refit weighted least squares on inliers; report max/median residual, inlier fraction, slope (depth/px), tick-period CV and extrapolation distance.
9. Reject/flag if slope ≤0, fewer than 3 anchors, residual >0.5 ft (or `0.25*major_increment`), or inferred labelled increments disagree.

Watch: PP-OCR's dictionary is bound to weights—official guidance says replacing it is not a digit whitelist without fine-tuning. Constrain in post-processing and geometry, not by editing its dictionary.

### 2. Make calibration redundant: ticks + robust affine/piecewise model

1. Build an ordered table `(y_tick, y_text, parsed_depth, OCR_conf, major_score)`; retain all candidates, including low-confidence readings.
2. Make initial `a` from `scipy.stats.theilslopes(depth, y)` on high-confidence anchors; it is a deterministic diagnostic.
3. RANSAC fit using anchor pairs; choose residual threshold in **feet** (start 0.5 ft) and `random_state=0`.
4. Require increasing depth with y and compare `a*median(minor_tick_gap)` with allowed divisors of observed labelled depth gaps.
5. Refit only RANSAC inliers using weighted `numpy.polyfit(..., w=confidence*major_score)`; preserve RANSAC mask for audit.
6. Compute residuals ordered by y; consider a breakpoint only with ≥3 accepted anchors on each side and a sustained jump >2× robust MAD.
7. Fit two affine segments, accept only if BIC/held-out residual materially improves and both retain positive slopes; otherwise use one line.
8. Emit `depth_at_y`, uncertainty (local residual MAD + slope CI proxy), anchors and QC overlay, rather than silently guessing.

Watch: a piecewise fit will happily explain one mis-OCR. Require a visible grid/tick discontinuity or multiple agreeing anchors; most scans should stay globally affine.

### 3. Two-stage rescue ensemble, not a heavy default

1. Install/use existing Torch stack with `python-doctr`; instantiate `recognition_predictor("crnn_mobilenet_v3_small", pretrained=True)` once on CUDA.
2. Send only PP-OCR empty/low-confidence crops plus their 2× upscaled, binarised variants to CRNN in a single batch.
3. If numeric outputs agree, accept; if they differ by a one-character confusion, defer to the model with higher calibrated confidence.
4. For the remaining ≤5 labels, call `recognition_predictor("parseq", pretrained=True)`; it supports crop-only inference.
5. Accept a rescue result only if it becomes a RANSAC inlier and restores an expected numeric increment; geometry is the final arbiter.
6. Log per-engine strings/confidences for a small future labelled calibration set; do not train from the one 17-interval example.

Watch: docTR weights use a broad Latin vocabulary and reported confidence is not calibrated across engines. Never compare raw PP-OCR and docTR confidence values without calibration.

## Do not bother with

* Full-page PP-OCR/docTR detection, VLM OCR, or an LLM pass: depth ROI geometry is already known, and huge vertical TIFFs make detector tiling and VLM latency the wrong cost centre. PP-OCRv5 itself notes VLM localisation/hallucination weaknesses.
* TrOCR as the main reader: 61.4M autoregressive model is overkill for sparse printed digits; use it only for evidence-rich disagreements. Its opened model card has no declared licence.
* Training a digit recognizer from the single labelled log: labels are interval annotations, not digit crops, so it supplies essentially no supervised OCR signal. Synthetic digits can help later, but first collect high-confidence anchors for self-training.
* Template matching each digit/tick: font, anti-aliasing, rotation and line overprint change across scans; it also cannot exploit the global monotonic/constant-spacing constraints.
* Blind OCR correction by rounding to an assumed increment: it can make a plausible but shifted log. Rounding is acceptable only after an independently validated affine calibration, and must be recorded as derived.
