# R1 — automatic mud-log box location

**Recommendation:** stream native-resolution horizontal bands, find persistent vertical rules, and rank candidate strips using sparse OCR and header/legend evidence. Use the existing JEV box/choice API only for low-confidence cases. This is a hypothesis, not a measured result on these 51 logs. Resizing a 1,500 × 50,000 page to 1,024 high makes a 63 px strip roughly 1.3 px wide; a whole-page detector will lose it. A 1,500 × 313,800 RGB decode alone is about 1.41 GB.

Times below are integration estimates, not measurements. Published metrics are from the linked authors and are not mud-log results; speeds use their hardware, not the laptop. Windows/Python 3.12/CUDA compatibility and model accuracy on these logs remain unverified.

| Method | Source (year) | Step | Possible advantage | Download / licence | Cost and size | Effort | Evidence |
|---|---|---|---|---|---|---|---|
| Streaming morphology, Hough, x profiles + OCR | [OpenCV morphology](https://docs.opencv.org/4.13.0/dd/dd7/tutorial_morph_lines_detection.html) (classic; docs 2026), [pyvips](https://github.com/libvips/pyvips) | 1, 6 | Keeps narrow rules at native width; sparse numeric OCR distinguishes depth | OpenCV Apache-2.0; pyvips/libvips LGPL-2.1-or-later; no weights | CPU, bounded band memory; linear in pixels; model 0 | 5–10 h | Source shows line extraction; mud-log accuracy and speed unverified |
| JEV semantic tie-break on proposals | [OpenCV proposals](https://docs.opencv.org/4.13.0/dd/dd7/tutorial_morph_lines_detection.html) (2026) | 1 | Existing typed choice/box calls can recognize headers and swatches after geometry reduces choices | Already available per task; API model/terms and local weights unverified | User reports ~0.4 s/call; network/model size unverified | 3–6 h | Task report only; no independent benchmark verified |
| PP-DocLayout-S / RT-DETR-L plus-L | [Paddle docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.en.md) and [S weights](https://huggingface.co/PaddlePaddle/PP-DocLayout-S) (2025) | 1 | Generic table/image proposals for legend on cropped header | S downloadable Apache-2.0; plus-L weights/terms verify separately | S 4.834 MB, 11.54 ms GPU / 18.53 ms CPU; plus-L 126.01 MB, 53.03 / 634.62 ms, vendor standard mode | 4–8 h | Vendor self-built layout mAP@0.5: S 70.9, plus-L 83.2; distinct datasets/classes, not comparable directly |
| DocLayout-YOLO | [Repo](https://github.com/opendatalab/DocLayout-YOLO), [weights](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) (2024) | 1 | Detects document tables/images to propose a legend area | Yes; weights card Apache-2.0, inference repo AGPL-3.0 | GPU preferred; size/latency unverified | 3–6 h | Document-layout claim only here; no target-class benchmark |
| OWLv2 base | [Model card](https://huggingface.co/google/owlv2-base-patch16-ensemble) (2023) | 1 | Text prompts for legend and local column crops, no training | Yes, Apache-2.0 | 0.2B parameters; GPU preferred; laptop latency unverified | 3–6 h | Zero-shot detection capability documented; mud-log performance unverified |
| YOLO-World V2.1 | [Official repo](https://github.com/AILab-CVC/YOLO-World) (2024/25) | 1 | Cached text vocabulary could speed repeated proposals | Yes; repo GPL-3.0; separate commercial licence offered; individual weights terms verify | S, 640 px; size and Windows latency unverified | 6–12 h | Authors: S 18.5 LVIS AP; no mud-log result |
| Grounding DINO 1.5/1.6 | [Official API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API) (2024/25) | 1 | Text-conditioned proposals | Hosted API/token; downloadable 1.5/1.6 weights and licence unverified; API client Apache-2.0 | Cloud latency, pricing and size unverified | 4–8 h | Authors: 1.6 Pro 55.4 COCO AP; irrelevant-domain benchmark |
| Florence-2 base | [Model card](https://huggingface.co/microsoft/Florence-2-base) (2024) | 1–2 | Phrase grounding and OCR-type outputs on header crops | Yes, MIT | 0.23B parameters; GPU preferred; runtime unverified | 4–8 h | Grounding capability documented; mud-log performance unverified |
| Qwen2.5-VL-3B | [Model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [licence](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) (2025) | 1–2 | Semantic box selection when headers are ambiguous | Yes; linked licence is Qwen research, non-commercial only; check current checkpoint terms | 3B; ~6 GB FP16 weights alone; GPU latency/total VRAM unverified | 5–10 h | Grounding claimed; target accuracy unverified |
| SAM 2 / SAM 3 | [SAM 2](https://github.com/facebookresearch/sam2) (2024), [SAM 3](https://github.com/facebookresearch/sam3) (2025) | 1 | Refine a supplied local box or segment a text-prompted legend | SAM 2 checkpoints Apache-2.0; SAM 3 gated access and SAM licence | GPU; size and laptop runtime unverified | 6–12 h | General segmentation only; no evidence that semantic columns form masks |
| DiT / LayoutLMv3 | [DiT](https://github.com/microsoft/unilm/tree/master/dit), [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) (2022) | 1 | Document-specific representations | Yes; DiT checkpoint terms unverified; LayoutLMv3 base CC-BY-NC-SA-4.0 | DiT detector size/runtime unverified; LayoutLMv3 0.1B | 8–20 h | DiT-base Mask R-CNN 0.935 PubLayNet mAP; unrelated classes; LayoutLMv3 base needs task head |

## Top three implementation recipes

### 1. Stream rules + depth evidence; CPU first

1. Use `pyvips.Image.new_from_file(path, access="sequential")`; check TIFF decoder and 1-bit/palette conversion on all formats. If band random-access is required, test `access="random"` and actual peak RAM/I/O; TIFF strip compression can force extra decode.
2. Process full-width 2,048–4,096-row bands, with 64-row overlap; convert to grayscale and threshold adaptively. Preserve `(x,y)` offsets and the original pixel grid.
3. In `opencv-python`, run vertical `morphologyEx(..., MORPH_OPEN, kernel=(1, 40–120))`; optionally `HoughLinesP` to bridge skew/broken rules. Sum response by x, merge peaks within 2–5 px, and accumulate per-band support, not raw ink alone.
4. Pair stable adjacent rules, including plausible 29–297 px lithology widths; permit unruled candidates and shifted/skewed borders. Do not assume lithology lies on a particular side of depth.
5. Run existing RapidOCR/EasyOCR on 4–8 representative crops per candidate. Score depth by multiple numeric labels that fit a monotonic linear row→depth relation with RANSAC; penalize inconsistent units and overprints.
6. Score lithology by repeated boundaries, bandwise colour/texture variability, adjacency to plausible depth, and header text. Blank bands must not terminate a column.
7. Derive vertical extent from rule continuity and header/terminal evidence; report full-height candidate with uncertainty rather than inventing an endpoint. Return original-coordinate `[x0,y0,x1,y1)` boxes.
8. Search header/footer and isolated non-body bands for legend words and multiple swatch+label rows; group OCR boxes spatially and expand to include swatches. Search additional bands if absent.
9. **Watch:** decorative rules, numeric non-depth columns, no borders, rotation, and legends mid-page. A true depth column can be sparse; low evidence should trigger fallback, not automatic rejection.

### 2. Existing JEV on short-listed local crops

1. Render top 2–4 candidate strips with margins at native width and 1,024–2,048 px height; include top/header, middle, and any legend-like area separately.
2. Label candidates A/B/C in a contact sheet; preserve a mapping from displayed coordinates and crop/resize transforms to original pixels.
3. Use the existing typed-choice interface to ask which contains depth numbers, lithology patterns, or legend swatches; include `none` and `uncertain` options.
4. For legend, use the existing box question on the likely header crop, asking for the enclosing swatches and labels, including non-rock symbols.
5. Map returned crop-local box coordinates through inverse scaling and add crop offset. Confirm API coordinate convention in the existing harness; do not assume normalized coordinates.
6. Reconcile choices across crops with rule persistence and monotonic numeric OCR; ask again only when scores disagree. Use geometry to extend column x boundaries across the page.
7. Cache calls by crop hash, batch where supported, and record call count, errors, and confidence; do not send the 50,000 px page.
8. **Watch:** model hallucination, tiny labels, crop-dependent answers, API availability and data-sharing terms. ~0.4 s/call is the user's reported figure, not an independently measured guarantee.

### 3. Lightweight layout proposals for legend recovery

1. Try `from paddleocr import LayoutDetection; model = LayoutDetection(model_name="PP-DocLayout-S")` using the [documented API](https://huggingface.co/PaddlePaddle/PP-DocLayout-S); installation and platform compatibility need testing.
2. Call `model.predict(input=crop, batch_size=1)` on native-resolution header/footer tiles and any suspicious mid-page tiles; include tile overlap and retain offsets.
3. Keep table/image/text proposals as generic regions, not as definitive legend or lithology labels.
4. Merge overlapping detections across tiles; rerank via OCR hits such as `legend`, `lithology`, `shale`, and `sandstone`, plus repeated small swatches beside text.
5. Expand the best proposal to include neighboring swatch and label rows; clip to image bounds and optionally use JEV to choose among candidates.
6. If S misses densely packed content, compare `PP-DocLayout_plus-L` on the same small tiles; do not assume its vendor score transfers to these logs.
7. **Watch:** no trained legend class, missed tiny text, and a table proposal that covers the whole log. No candidate is a valid result; then search other bands or flag for review.

## Evaluation and exclusions

Use the supplied boxes as held-out **evaluation** ground truth, not inference inputs. Report per-class recall at IoU 0.5, median IoU, x-edge error, lithology-width coverage, failure count, wall time, peak RAM/VRAM and JEV calls. Long boxes can have good IoU despite wrong x boundaries. Tune on the labelled example or leave-one-log-out; flag tuning against the 51 boxes. Test 1-bit separately. The fastest reliable option remains unproven.

Avoid full-page resizing, dense VLM tiling, SAM over the entire page, or training a detector on one example: resolution, repeated inference, mask semantics, and supervision are poor fits. LayoutLMv3 base lacks a task-specific detector and restricts commercial use; DiT targets conventional classes. Molmo ([model](https://huggingface.co/allenai/Molmo-7B-D-0924), 2024) points but needs box reconstruction; 7B fit on 12 GB is unverified. Grounding DINO 1.5/1.6 and SAM 3 add access dependencies; use only if the existing fallback fails. No benchmark establishes geological layout accuracy.
