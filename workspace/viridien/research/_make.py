import pathlib
P = pathlib.Path(__file__).parent
CTX = """sandbox=workspace-write
You are a research agent. Use live web search heavily (papers, arXiv, GitHub, Hugging Face, benchmarks, docs) and open
the sources; do not answer from memory. Today is 2026-11-07: prefer 2024-2026 work, but include older classics if they
are still the best cheap option. Working folder: C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/research
Write ONE file, research/<ID>.md (ID given below), and nothing else. Do not run heavy downloads or installs.

THE TASK WE ARE SOLVING (Viridien HackTheBurgh challenge)
Input: 51 scanned mud logs (TIFF), 2,600 to 313,800 px tall (median 44,550), about 1,000-1,700 px wide; 9 are 1-bit
black-and-white (rock types shown only by hatch patterns), 36 RGB, 6 palette. Each comes with three boxes: the depth
column (printed depth numbers), the interpreted lithology column (29-297 px wide, median 63, a vertical strip filled
with colour and/or hatch patterns), and the legend (swatches + labels, e.g. Shale, Siltstone, Sand and Sandstone,
Limestone, Coal; mixed with non-rock modifier/drilling symbols). Output per log: contiguous intervals
top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id (+ an annotated image). Judged on accuracy and efficiency.
Bonus: find the three boxes automatically. Only ONE labelled example exists (17 intervals), so methods must be
zero-shot, few-shot from the legend swatches, or self-supervised. Hard cases: pattern-only rocks, same colour
different rock, tiny legend text, legends mixing modifier symbols, long blank stretches, overprinted lines/text.
Pipeline: (1) locate boxes, (2) parse legend (swatch crops + OCR labels, rock vs non-rock), (3) segment the column
into intervals along depth, (4) cluster intervals by visual pattern, (5) name clusters by matching legend swatches,
(6) OCR the depth column and fit row->depth, (7) run fast on huge images.
Hardware: one RTX 4080 Laptop (12 GB), Windows, Python 3.12, torch 2.12+CUDA. Already tried / in our bake-off:
dominant-colour runs, hand features + ruptures PELT, OpenCV template NCC, k-means + HMM Viterbi, DINOv2-small,
SigLIP-base, a small CNN trained on synthetic columns made from legend swatches, the Typesafe JEV API (typed choice
and box questions over images, ~0.4 s/call), a GPT vision LLM per cluster, rapidocr/easyocr for text.

WHAT TO PRODUCE in <ID>.md (be concrete, under 1,500 words):
1. A table of candidate methods for YOUR topic: name | source (URL, year) | which pipeline step | why it could beat
   what we have (accuracy / speed / cost) | downloadable now? (weights + licence) | GPU/CPU cost, model size | effort to
   plug in (hours) | evidence quality (benchmark numbers you actually saw, or "claim only").
2. Your top 3 recommendations, each with a 5-15 line implementation recipe for our harness (python, package names,
   model ids) and the failure mode to watch.
3. What NOT to bother with and why (hyped but wrong fit, too slow, licence, needs training data we don't have).
Mark anything you could not verify. Every row must have a real URL you opened.

YOUR TOPIC:
"""
T = {
 "r1_layout": "Bonus step (1): locating the depth column, lithology column and legend on the page without annotations. Document layout detection (DocLayout-YOLO, PP-DocLayout / PP-Structure, LayoutLMv3, DiT, RT-DETR layout models), open-vocabulary detectors (Grounding DINO 1.5/1.6, OWLv2, YOLO-World, Florence-2, Molmo/Qwen-VL pointing and grounding), SAM 2 / SAM 3 promptable segmentation, and classical ruled-line / table-structure detection (Hough, morphology, column profiles). What is the fastest reliable way for narrow tall columns on a 1,500 x 50,000 px page?",
 "r2_segmentation": "Step (3)-(4): finding interval boundaries and grouping intervals along a long 1-D strip of textured image. Change-point detection (PELT, binary segmentation, kernel CPD, BOCPD, deep/learned change-point, ClaSP, FLOSS), texture descriptors (Gabor, LBP, wavelet scattering, Fourier period), unsupervised texture segmentation, and fast 1-D segmentation of DINO/CLIP feature sequences. Also how to pick the number of clusters robustly. Speed on 300k-row strips matters.",
 "r3_symbol_matching": "Step (5): naming each interval by matching it to the legend swatches: few-shot / one-shot pattern and symbol recognition. Include the DARPA/USGS AI for Critical Mineral Assessment (CriticalMAAS) map feature extraction challenge and its winning methods (legend-swatch-to-map-polygon segmentation, e.g. LDTR, DIGMAPPER, Jataware/ISI work), geological map legend matching, hatch-pattern recognition, Siamese / prototype networks, DINOv2 / DINOv3 / RADIO / SigLIP 2 patch matching, scale- and phase-invariant template matching (FFT, Fourier-Mellin), and training on synthetic data built from swatches.",
 "r4_legend_ocr": "Step (2): parsing the legend: finding swatch rectangles, pairing them with their label text, reading small/degraded text, and telling rock types from modifier/drilling/fossil symbols. Compare OCR and doc-VLM options: PaddleOCR PP-OCRv5 / PaddleOCR-VL, docTR, Surya, GOT-OCR 2.0, Florence-2, olmOCR, Qwen2.5/3-VL small, SmolDocling, MinerU, Mistral OCR, moondream; speed per crop on a 12 GB GPU and accuracy on tiny text. Also key-value / legend-item pairing approaches.",
 "r5_depth_ocr": "Step (6): reading the depth column and fitting a row->depth mapping. Fast digit OCR (PP-OCRv5 rec only, CRNN/PARSeq, TrOCR-small, docTR), detecting tick marks, robust fitting (RANSAC, Theil-Sen, piecewise linear for scale changes and breaks), sanity checks (monotonic, constant spacing), and how well-log / mud-log digitisation tools do this. Aim: sub-foot accuracy at under 1 s per log.",
 "r6_welllog_prior": "Prior work specifically on digitising scanned well logs and mud logs: papers, theses, companies and open-source tools (raster log digitisation, lithology column extraction, curve digitisation, e.g. work from Viridien/CGG, Schlumberger/SLB, Halliburton, Kaggle/FORCE competitions, SPE/SEG/EAGE papers 2018-2026, GitHub repos). What pipelines did they use, what accuracy did they report, and which ideas can we copy today?",
 "r7_efficiency": "Step (7): running fast and cheap on huge TIFFs and many crops. Image I/O (pyvips/libvips streaming, tifffile tiled/strip reads, memory maps, PIL crop-before-convert costs), downsampling choices, GPU batching, ONNX Runtime / TensorRT / torch.compile, fp16/int8 quantisation of DINOv2/SigLIP, caching embeddings, cascades (cheap model first, expensive only on ties), and cost comparison: local small models vs API calls (JEV, GPT vision, Gemini Flash, Claude Haiku) per log. Give numbers.",
 "r8_small_vlms": "Small open vision-language models and structured-output VLM APIs that could replace or back up our paid calls for (a) naming an interval crop given the legend image and (b) giving boxes: Qwen2.5-VL/Qwen3-VL (2B-8B), InternVL3, MiniCPM-V, Gemma 3, SmolVLM2, Moondream 2/3, Florence-2, Phi-4-multimodal, Molmo; plus constrained decoding (outlines, xgrammar) for JSON choices. VRAM on a 12 GB GPU, latency per image, and any benchmark on fine pattern/texture discrimination or chart/diagram reading.",
}
for k, v in T.items():
    (P / f"{k}.txt").write_text(CTX.replace("<ID>", k) + v + "\n", encoding="utf8")
print(sorted(T))
