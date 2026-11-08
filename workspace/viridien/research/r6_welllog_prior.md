# R6 — prior work on digitising scanned well logs

Research snapshot: 7 November 2026. **Best immediate bet:** use the legend as the local ground truth, augment colour with phase-insensitive hatch features, and reuse raster-log depth-calibration ideas. I found no validated, downloadable, zero-shot model specifically for extracting patterned lithology intervals from mud-log rasters. Curve tracing, prediction from numerical logs, and rock-core photography are different tasks; their reported scores do not establish performance on this challenge.

Costs and integration times below are **my estimates**, not published timings; model sizes are marked when unavailable. “No weights” means an algorithm/library, not a missing checkpoint. Verify package compatibility with Python 3.12 locally.

| Method | Source, year | Step | Why it might improve the bake-off | Download / licence | Cost / size | Effort | Evidence actually seen |
|---|---|---|---|---|---|---|---|
| ImageToLithology | [Repository](https://github.com/giecaruff/ImageToLithology), year unverified | 3–5 | Maps colour to lithology and converts image to layers; reverse rendering gives an easy QC loop. Adapt its per-log colour dictionary and reject duplicate-colour ambiguity. | Yes; Apache-2.0; no weights. | CPU; size not stated. | 2–4 h | Synthetic sample and expected outputs only; no reported real-log accuracy. Its sample assigns identical colours to multiple lithologies. |
| Texture + mixture clustering | [Jeong et al.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JB018204), 2020 | 3–5 | Core images: RGB + local entropy, depth averaging, Gaussian mixture clusters. Adapt with orientation and periodicity for phase-shifted hatches; complements dominant colour and generic embeddings. | Reimplement; paper-specific code/weights not verified. | CPU, no model weights; workload scales with strip pixels. | 4–8 h | Paper describes unsupervised core-image facies, not scanned hatch symbols; no transferable lithology-column score verified. |
| stratapy symbol library | [Paper](https://www.nature.com/articles/s41598-026-58501-2), [code](https://github.com/Jack6228/stratapy), 2026 | 2, 4–5 | Over 200 patterns, including USGS-derived black-and-white patterns: a source of extra synthetic negatives and robustness tests; per-log legend remains authoritative. | Code downloadable; BSD-3-Clause; no weights. Check individual pattern provenance/terms before redistribution. | CPU rendering; model size 0; asset size unverified. | 3–6 h | Rendering library, not recognition benchmark. Asset appearance and scale match to these 51 scans unverified. |
| pyLogDigitizer | [Repository](https://github.com/GeoLarryLai/pyLogDigitizer), 2026 | 1, 6 | Track/grid detection, deskew, OCR depth calibration and manual corrections; potentially saves calibration implementation. Its DP tracer is for curves, not rock bands. | Source/PyPI advertised; AGPL-3.0; optional U-Net needs a trained checkpoint; pretrained weights not verified. | CPU conventional path; size and Python 3.12/Windows support unverified. | 3–6 h to evaluate | Authors show held-out Kansas curve comparisons but numeric scores not verified; no lithology accuracy. |
| Landmark raster pipeline | [Patent application](https://patents.google.com/patent/US20260103975A1/en), 2026 | 1, 2, 6 | Contours propose track/header boxes, VQA interprets metadata, text-guided masks isolate curves. Reuse the proposal-and-check principle with existing JEV. | Patent text public; implementation/weights/licence not found; application pending per source. | CPU proposals + API calls; model size unreported. | 4–8 h for simple proposals | Claim only; curve extraction, no interval or box benchmark. |
| VeerNet | [Published paper](https://www.mdpi.com/2313-433X/9/7/136/pdf?version=1688693413), 2023; [preprint](https://arxiv.org/abs/2210.05597), 2022 | 3, curves only | U-Net-like downsampling and synthetic training cope with tall raster tracks and grids; useful augmentation ideas, poor fit for rock swatches. | Paper says code available; usable checkpoint and code licence **unverified**. | GPU training; parameter count/runtime unverified. | 12–24+ h to adapt | Preprint: overall curve F1 35%, IoU 30%, GR Pearson 0.62; not lithology results. |
| Manual digitisation/QC | [BGS review](https://nora.nerc.ac.uk/id/eprint/530055/1/OR21006.pdf), 2021 | 3, 6 | Display raster aligned to calibrated depth; boundaries can be inserted and dragged. Useful only for low-confidence exceptions and the single labelled example. | WellCAD is commercial; licence/pricing unverified; no weights. | Human time; hardware/size unverified. | 2–4 h for a simple overlay | Describes a manual workflow for interpreting geophysical logs, no automatic extraction accuracy. |

## Top three: practical recipes

### 1. Legend-conditioned colour + hatch descriptors (4–8 h)

Adapt the feature-and-cluster idea from Jeong and ImageToLithology; orientation/periodicity below is a proposed extension, not a reported result.

1. Read swatch interiors and the supplied lithology ROI at native resolution with `tifffile` or `PIL`; convert palette/1-bit images consistently to RGB.
2. Mask swatch borders and likely text/long straight lines with `opencv-python`; retain an unmasked fallback because hatch lines can resemble grid lines.
3. For sliding windows of e.g. 32, 64 and 128 rows, calculate Lab colour histograms, local binary pattern histograms (`skimage.feature.local_binary_pattern`), and coarse gradient-orientation histograms (`cv2.Sobel`).
4. Add normalized 2-D FFT power or autocorrelation descriptors on resized patches to tolerate hatch phase shifts; keep colour and texture distances separate.
5. Compute the same features on swatch interiors; compare using normalized distances, with both texture-heavy and colour-heavy weights. Calibrate thresholds from augmented swatches, not the 17 intervals alone.
6. Decode window scores using the existing HMM/PELT; pool adjacent windows, refine changes at native resolution, and output unknown when all matches or the top-two margin are poor.
7. Test against colour runs, existing hand features and NCC on all 17 labelled intervals, plus visual review of held-out monochrome and same-colour examples. **Failure:** tiny swatches contain less than one pattern period; seams, text and thin beds bias features. Do not force ambiguous same-colour classes.

### 2. Robust depth calibration and audit (3–6 h)

Use pyLogDigitizer's calibration workflow as a reference; its public high-level example is a curve workflow, not a guaranteed depth-only API.

1. Crop the provided depth box; process in native-resolution overlapping vertical tiles with `tifffile`/`PIL`, preserving each tile's global y offset.
2. Run existing `rapidocr` or `easyocr` on candidate text strips; record numeric values, confidence and full-image y centres.
3. Discard impossible values, duplicates and text outside the depth column; keep both feet/metres and sign hypotheses until checked.
4. Fit `sklearn.linear_model.RANSACRegressor` to `depth = a*y + b`; enforce monotonic increasing depth and score residuals in pixel-equivalent units.
5. Inspect residual versus y for scale changes or stitched scans; fit piecewise linear segments only when multiple anchors support each segment.
6. Convert every interval endpoint from original image y to depth, leaving depth unknown if there are too few consistent anchors; preserve y and render anchor/inlier overlays.
7. **Failure:** repeated scales, OCR digit swaps, depth-axis reversal and scan stretching can produce a convincing wrong fit. Verify against independent ticks and labelled endpoints.

### 3. Pattern-library stress tests and box proposals (5–10 h)

Combine stratapy's pattern diversity with the proposal-then-verification idea in the Landmark application; this is a new harness, not their released system.

1. Inspect `stratapy` pattern assets and licences; render selected patterns in memory with `matplotlib`/`stratapy`, or use the existing legend-swatches synthetic generator.
2. Generate held-out synthetic columns with varying pattern phase, scale, colour, thin beds, text, line overlays, scan noise and blank stretches; save the exact interval truth.
3. Use these only to compare descriptor weights and rejection behavior; do not treat synthetic accuracy as real-log accuracy.
4. For optional box detection, downsample **in tiles**, use `cv2` line morphology and contours to propose vertical tracks; preserve global coordinates and group aligned candidates.
5. Crop headers and legend candidates at native resolution; use existing JEV typed-choice calls to identify depth/lithology/legend among candidates, with an explicit none-of-the-above option.
6. Score on the three provided boxes: intersection-over-union, recall and missed-box rate; compare OCR anchors and extracted intervals with supplied-box runs.
7. **Failure:** generic library patterns differ from local legends; long empty columns and truncated borders can defeat contours, while tiny headers defeat VQA. Fall back to supplied boxes when available.

## Do not prioritise

- **FORCE/Kaggle-style lithology predictors:** they classify numerical geophysical measurements, not printed lithology patterns. [FORCE-based example](https://isprs-archives.copernicus.org/articles/XLIX-B2-2026/729/2026/) explicitly uses logging sequences; scores are not transferable.
- **VeerNet or full image-segmentation retraining:** curve labels and a 35% F1 do not justify training a lithology segmentation network with only one annotated log. The swatch-trained CNN already tests the cheaper synthetic route.
- **SLB Automated Lithology:** [SLB describes](https://www.slb.com/products-and-services/innovating-in-oil-and-gas/well-construction/measurements/logging-while-drilling-services/automated-lithology) imaging and describing physical cuttings, not converting legacy raster lithology columns; commercial availability and price unverified.
- **Whole-image VLM passes:** a 313,800-pixel image must be downsampled so much that thin beds and small type disappear; per-cluster VLM and typed box calls are already being tested. No comparable end-to-end mud-log accuracy was verified.

Measure boundary error in pixels/depth, depth-weighted lithology accuracy, unknown rate, runtime and peak RAM; report the 17-interval example separately from any manually reviewed unseen logs. Confirm that the TIFF reader supports windowed access for each compression/layout: a narrow crop alone does not guarantee that a huge TIFF is not decoded in full.
