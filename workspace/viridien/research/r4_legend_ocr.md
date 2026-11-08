# R4 — legend parsing / OCR research (checked 2026-11-07)

**Bottom line.** Treat a legend as a small spatial database, not a page to be
converted to Markdown: detect/read text at 2–4x scale, find candidate swatches
with connected components/contours, then pair by geometry.  Use a VLM only as a
cheap *adjudicator* for uncertain label meaning (rock versus modifier) or an
OCR disagreement.  None of the published benchmarks is a mud-log legend or
reports tiny scanned-label CER, so claims of superiority for this exact problem
are **unverified**.

| candidate method | source (opened; year) | pipeline step | why it could beat current bake-off | downloadable now? weights / licence | cost / size (12-GB GPU) | plug-in | evidence quality |
|---|---|---|---|---|---|---:|---|
| PP-OCRv5 English/mobile + DB detector | [official model table](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md) (2025), [report](https://arxiv.org/abs/2603.24373) (2026) | text boxes, OCR | Purpose-built recognizer, word/line boxes and confidence; crop/upsample rather than ask a VLM to invent reading order. | Yes; PaddleOCR models; Apache-2.0 code (weight licence not separately verified). | mobile rec 7.5 MB; official GPU rec time not reported for English-only; generic mobile 5.43 ms, server 8.46 ms/crop. Detector dominates. | 3 h | Official internal English accuracy: mobile 85.25%; not a tiny-text benchmark. |
| PaddleOCR-VL-0.9B/1.5 | [official repo](https://github.com/PaddlePaddle/PaddleOCR) (2025–26), [paper](https://arxiv.org/abs/2601.21957) (2026) | one-shot layout/text fallback | Dynamic resolution and layout are useful if a legend is irregular; can emit structured content. Overkill for 10 labels. | Yes via PaddleOCR/HF; code Apache-2.0, exact weight terms **unverified**. | 0.9B: ~1.8 GB fp16 weights plus image/KV; likely fits one cropped legend, **not benchmarked on 12 GB**. | 6–10 h | Author OmniDocBench v1.5 94.93 overall; page parsing, not labels. |
| docTR (FAST/DBNet + PARSeq) | [model zoo](https://mindee.github.io/doctr/latest/models/index.html) (2026), [repo](https://github.com/mindee/doctr) | OCR alternative | Pure PyTorch, easy detector/recognizer swapping; PARSeq may rescue a crop where PP-OCR fails. | Yes, pretrained; Apache-2.0. | Small CNN/transformer components; CPU-capable; no comparable GPU crop latency published. | 3 h | Public FUNSD/CORD tables exist, but docs warn they are only a starting point; no relevant tiny scan result. |
| Surya 2 | [official README](https://github.com/datalab-to/surya/blob/master/README.md?plain=1) (2026) | full legend OCR/layout | Produces HTML/layout/reading order in one pass; could remove pairing engineering. | Yes; code Apache-2.0, weights modified OpenRAIL-M (commercial restriction). | 650M; official vLLM configs start at 16-GB T4 / 24-GB A10; unsafe fit on 12 GB. | 8 h | 83.3% olmOCR-bench (author); 5.35 pages/s on 32-GB 5090, not comparable crop latency. |
| GOT-OCR 2.0 | [official repo](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/) (2024), [paper](https://arxiv.org/abs/2409.01704) | crop OCR fallback | Explicit fine-grained box mode is attractive for a single difficult label. | Checkpoint downloadable; checkpoint licence **not stated/verified**. | ~0.6B class; legacy Python 3.10/CUDA 11.8 + FlashAttention instructions; awkward on Windows/Py3.12. | 6 h | Claims only for this task; repo says multi-crop is not batch inference. |
| Florence-2-large-ft | [Microsoft model card](https://huggingface.co/microsoft/Florence-2-large) (2024) | OCR-with-region / visual verifier | Has `<OCR_WITH_REGION>` and grounding in 0.77B, so useful to propose text boxes plus a visual answer. | Yes; model card lists 0.77B, licence MIT. | ~1.5 GB fp16 weights; likely fits; autoregressive, so slower than PP-OCR per label. | 4 h | General-task model; no scanned-tiny-text number observed. |
| olmOCR-2 | [official repo](https://github.com/allenai/olmocr) (2025) | high-quality page fallback | Strong document transcription and an unusually useful 7,000-case benchmark. | Yes; code/model/data presented as open (check exact chosen checkpoint before use). | 7B: does not safely fit fp16; 4-bit is feasible but slow/complex on 12 GB. | 8 h | Benchmark exists, but is document linearization, not swatch relation extraction. |
| Qwen2.5-VL-3B-Instruct | [official repo/results](https://github.com/QwenLM-corp/Qwen2.5-VL/blob/main/README.md) (2025), [report](https://arxiv.org/abs/2502.13923) | semantic rock/modifier decision; dispute resolver | Prompt it with crop + OCR candidates: it can use words and symbol appearance, unlike lexical rules. | Yes, HF/modelscope; Qwen licence (not Apache) — verify competition compatibility. | 3B: 4-bit weights ~1.5–2 GB plus KV; fits 12 GB; expect seconds/crop, no official latency. | 4 h | Author 3B DocVQA 93.9; no legend or tiny text evaluation. |
| SmolDocling-256M | [paper](https://arxiv.org/abs/2503.11576) (2025), [model card](https://huggingface.co/docling-project/SmolDocling-256M-preview) | structured legend fallback | Smallest VLM that emits location-aware DocTags; can test full legend at negligible VRAM. | Yes; Apache-2.0. | 256M (~0.5 GB fp16); likely sub-second crop, **no published RTX-4080 latency**. | 4 h | Paper says it beat GOT/Qwen2.5-VL on its full-page OCR sets; not independent. |
| MinerU2.5/Pro | [ACL paper](https://aclanthology.org/2026.acl-industry.3.pdf) (2026), [repo](https://github.com/opendatalab/mineru) | page parser fallback | Very good high-resolution document parsing, could cope with noisy legend scans. | Yes; current terms changed across releases—verify exact Pro model. | 1.2B; feasible quantized, but sizeable stack and page-oriented. | 8–12 h | OmniDocBench 90.67 (paper; earlier model); not a small-label metric. |
| Mistral OCR 4 API | [API docs](https://docs.mistral.ai/api/endpoint/ocr) (2026), [pricing](https://mistral.ai/news/ocr-4/) | cloud adjudicator / JSON output | Returns blocks/boxes/confidence and supports schema-guided annotation: excellent zero-code independent second opinion. | API only, not downloadable. | $4/1,000 pages ($2 batch): 51 legend crops about $0.20, plus network; no local GPU. | 2 h | Vendor price verified; advertised quality, no public tiny-scan number observed. |
| Moondream 2/3 | [official models page](https://moondream.ai/models) (2026) | semantic classifier only | Small/local visual QA and grounding, not a primary transcriber. | Yes (verify current model terms); Moondream 2 supports int4. | v2 1.9B; v3 9B MoE/2B active. Fits quantized; generative latency. | 3 h | Official OCRBench 61.2, well below an OCR-first choice. |
| Geometry + bipartite matching (recommended pairing layer) | [KVP10k task paper](https://arxiv.org/abs/2405.00505) (2024) | swatch detection, pairing | No training: exploits legend convention (same row, nearest right/left text), survives unknown symbols, returns auditable confidence. | N/A; OpenCV/SciPy. | CPU milliseconds. | 4 h | KVP10k establishes KVP as a hard task; no legend-specific benchmark. |

## Top three recommendations

### 1. PP-OCRv5 first, deterministic swatch/text pairing (ship this)

1. `pip install paddleocr paddlepaddle-gpu opencv-python scipy rapidfuzz` (validate the CUDA/Paddle wheel first).
2. Crop only the supplied legend; make `cv2.resize(..., fx=3, fy=3, INTER_CUBIC)` plus a CLAHE/gray alternate.
3. Run `PaddleOCR(lang="en", ocr_version="PP-OCRv5")` on both; retain polygon, text and confidence; choose the higher-confidence normalized string per overlapping line.
4. Find swatches on original pixels: grayscale/HSV threshold variants, morphology close, `findContours`, retain rectangles/squares and patterned rectangles by border/area/aspect; never require uniform fill.
5. Merge OCR words into lines; score every swatch--line pair: vertical overlap, centre-y distance, horizontal direction, and no intervening swatch.
6. Solve one-to-one pairs with `scipy.optimize.linear_sum_assignment`; permit stacked multi-line labels by merging adjacent aligned lines.
7. Classify label strings against a rock lexicon (`shale`, `siltstone`, `sand(stone)`, `limestone`, `coal`, etc.); preserve unknowns and flag rather than discard.
8. Save `swatch_bbox,label_bbox,text,pair_score,ocr_conf,rock_flag` for visual review and later swatch matching.

Watch: a colour key may use a line/point/fossil icon rather than a closed rectangle. Add a second “non-text connected component aligned with label” candidate class; do not make rectangles a hard gate.

### 2. Qwen2.5-VL-3B as an offline semantic/repair gate, never the main OCR

1. Install `transformers accelerate bitsandbytes qwen-vl-utils`; load `Qwen/Qwen2.5-VL-3B-Instruct` in 4-bit (`BitsAndBytesConfig(load_in_4bit=True)`).
2. Send a **single 768–1200 px-upscaled paired item** (swatch + its OCR line), not the whole tall log.
3. Use temperature 0 and demand JSON: `{"transcription":"", "class":"rock|modifier|drilling|fossil|other", "canonical_rock":null, "confidence":0..1}`.
4. Include PP-OCR candidate text and a closed allowed rock vocabulary in the prompt; ask it to copy exact label, not infer a lithology.
5. Invoke only if OCR confidence <0.85, pair score is low, or lexical rock decision is unknown; cache SHA-256(crop+prompt).
6. Accept a changed spelling only when Qwen confidence is high and Levenshtein distance to a vocabulary entry improves; otherwise retain PP-OCR.

Watch: VLMs confidently normalize/ hallucinate (“Sandstone”) from a near-illegible label. Its semantic class is useful; its free-form transcription is not ground truth.

### 3. Mistral OCR 4 as the independent, low-volume escalation path

1. `pip install mistralai`; encode an upscaled PNG legend/item as a data URL (or upload per the [OCR API](https://docs.mistral.ai/api/endpoint/ocr)).
2. Call `client.ocr.process(model="mistral-ocr-latest", document=..., include_blocks=True, confidence_scores_granularity="word")`.
3. For uncertain whole legends set `document_annotation_format` to JSON schema and request `[{swatch_description,label,kind}]`; keep the native blocks too.
4. Match returned blocks to OpenCV swatches geometrically; schema output is only a cross-check.
5. Escalate only the 1–3 ambiguous items per log; log response/model version and use its word confidence as a vote, not an oracle.

Watch: network/API availability, scan confidentiality, and paragraph-level blocks can merge neighbouring labels. The advertised $4/1,000-page rate makes cost negligible here, but latency is not harness-deterministic.

## Do not bother (for this bake-off)

* **Surya, olmOCR and MinerU** as the default: page-to-Markdown systems spend seconds and substantial VRAM on a 5–30-label crop. Surya's official inference presets start at 16 GB and its weights restrict larger commercial users; olmOCR is 7B. MinerU's excellent OmniDocBench score is a different task.
* **PaddleOCR-VL** as default: strong paper score, but the 0.9B autoregressive parser gives no reason to beat PP-OCR on short, upscaled English labels with known geometry. Test it only on failures.
* **GOT-OCR 2.0**: old environment, no stated checkpoint licence, and its own instructions say multi-crop mode is not batch inference.
* **Florence-2 / SmolDocling / Moondream** as primary OCR: attractive sizes, but no observed evidence on degraded tiny labels. SmolDocling is worth a one-hour A/B experiment; do not let its full-page DocTags replace boxes + matching.
* A learned key-value/relation model: KVP10k/form datasets do not contain repeated geological swatches, and one 17-interval labelled log is not enough to train a domain relation extractor. Geometry gives the necessary relation with no labels.

