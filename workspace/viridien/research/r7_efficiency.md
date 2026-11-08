# R7 — huge TIFFs and inference efficiency

Research date: 2026-11-07. Estimates below are engineering estimates, not measurements on this laptop or these TIFFs. No downloads or installs were performed. Hours are integration estimates. “Claim” means no relevant benchmark was verified.

| Method and opened source (year) | Step | Why it might beat current setup | Download / licence | Cost and size | Effort | Evidence |
|---|---|---|---|---|---|---|
| [libvips / pyvips](https://libvips.github.io/pyvips/intro.html) (docs, accessed 2026) | 1–7 | Demand-driven crops and sequential scan limit peak RAM; do not materialize entire RGB page | Yes; no weights; libvips LGPL-2.1-or-later, pyvips MIT; Windows native DLL required | CPU; bounded buffers, TIFF-layout dependent | 3–5 h | Architecture/docs; speed on our files unverified. [Library benchmark](https://github.com/libvips/libvips/wiki/Speed_and_Memory_Use) uses a different workload. |
| [tifffile selection / Zarr](https://www.cgohlke.com/docs/tifffile/) (2026) | 1–7 | Reads intersecting TIFF segments; direct memmap where compatible; enables repeated strip reads | Yes; BSD-3-Clause; codecs/Zarr as needed; no weights | CPU, one or more compressed tiles/strips per selection | 3–5 h | Documented segment selection; actual compression/layout unknown. |
| [Pillow crop](https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html) (2026) | 1–7 | Simple baseline; crop before RGB conversion reduces converted output | Yes; HPND licence; no weights | CPU; `crop()` invokes `load()`, potentially decoding whole page first | 0.5 h | Source code; no random-access guarantee. |
| [DINOv2-small](https://huggingface.co/facebook/dinov2-small) batched FP16 (2023) | 4–5, 7 | Reuse existing encoder, batch crops and cache vectors; no retraining | Yes, Apache-2.0 weights | 22.1M parameters; ~88 MB FP32 / 44 MB FP16 weights, plus activations; GPU time unmeasured | 2–4 h | Model card reports 22.1M; throughput/accuracy claim only for logs. |
| [SigLIP-base](https://huggingface.co/google/siglip-base-patch16-224/tree/main) batched FP16 (2024) | 4–5, 7 | Cache legend embeddings once; use it only for uncertain DINO/texture matches | Yes, Apache-2.0 weights | GPU; full checkpoint size and latency unverified; vision-only execution can avoid text tower | 2–4 h | Model card supports weights; log speed and accuracy claim only. |
| [ONNX Runtime TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) (docs, accessed 2026) | 4–5, 7 | FP16 engine and cached fixed-shape batches may reduce GPU latency | ORT MIT; TensorRT NVIDIA proprietary SDK terms; export existing Apache weights | GPU, plus engine/workspace; engine size and speed unmeasured | 6–12 h | Vendor performance claim; CUDA fallback required for unsupported ops; CUDA/cuDNN/TRT versions must match. |
| [INT8 TensorRT calibration](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) (docs, accessed 2026) | 4–5, 7 | Potential faster/smaller encoder if calibration preserves pattern similarities | ORT MIT; TRT terms; generate from existing weights | GPU, calibration and engine build; INT8 speed unmeasured | 8–16 h | Quantization procedure documented; no DINO/SigLIP log benchmark verified. |
| [Jev visual choice/box](https://madewithjev.com/jev-images) (2026) | 1, 2, 5, 7 | Group several questions per image, send only ambiguous clusters | Hosted proprietary; no downloadable weights; image mode preview | ~0.4 s/call observed by user, not independently measured; cost below | 1–3 h | User timing; third-party documentation for preview. Accuracy on logs unverified. |

## Top 3 recommendations

### 1. Decode a bounded region, preserving native pattern resolution

1. Inspect `tifffile.TiffFile(path).pages[0]`: `shape`, `dtype`, `compression`, `is_tiled`, `rowsperstrip`, `photometric`, `colormap` and orientation.
2. Try `pyvips.Image.new_from_file(path, access='sequential')`; crop a union of the three x-ranges before materializing, then sweep y in increasing 2048–8192-row chunks with overlap at least the largest analysis window.
3. For non-monotone legend/OCR access, open a separate `access='random'` handle; do not request earlier rows from a sequential decoder.
4. Alternative: `tifffile.imread(path, selection=(slice(y0,y1), slice(x0,x1), ...))` using axes/shape appropriate to the page. Requires compatible `zarr` and `imagecodecs`; pin working versions.
5. Attempt `tifffile.memmap(path)` only for compatible contiguous, uncompressed, non-tiled data; compressed/tiled pages cannot be treated as direct mapped pixels. `out='memmap'` can instead decode to a temporary file.
6. Expand palette through its colormap and 1-bit correctly; test against a trusted reference. Preserve original y offsets. Never resize the whole tall page to a single small image.
7. For box search use overlapping thumbnails, retain native crops for depth OCR, boundary refinement and hatch matching. Anti-aliased area shrinking for colour; compare native or lightly resized grayscale for hatch. Avoid narrowing a 29–63 px strip to a few pixels.
8. **Watch:** a narrow crop of a full-width compressed strip still decodes that strip; a single huge strip may require substantial I/O. Time cold/warm reads and peak RAM on RGB, palette and 1-bit examples; verify orientation and seam equality.

### 2. Batch and cache existing encoders

1. Use `transformers.AutoImageProcessor` and `AutoModel.from_pretrained('facebook/dinov2-small')`; use existing SigLIP model only where helpful.
2. Produce native-scale or padded square crops at fixed model input dimensions; preserve aspect ratio and keep hatch samples sufficiently large. Apply identical preprocessing to swatches and column crops.
3. Run `model.eval().cuda()` in `torch.inference_mode()` and `torch.autocast('cuda', dtype=torch.float16)`; use the same embedding pooling as the existing baseline.
4. Start batches 16/32/64; use pinned host memory and nonblocking transfer, reducing batch on OOM. Warm up; synchronize CUDA around timing, report end-to-end crops/s and peak VRAM.
5. L2-normalize embeddings, cache per swatch and per representative crop, and batch cosine matching; e.g. `float16` cache for 384-dimensional DINO vectors is 768 bytes/vector.
6. Key cache by image content/version, coordinates, model revision, preprocessing, dtype and pooling; never share a legend across different logs without explicit identity.
7. Compare FP16 with FP32 on swatch ranking, interval boundaries and the 17 labeled intervals; sample other logs manually. **Watch:** tiny hatches and near-ties can reorder even when average cosine error is small.
8. Only if inference remains dominant, export fixed shapes to ONNX, benchmark CUDA EP then TensorRT EP with FP16, engine caching and CUDA fallback; verify active providers and actual nodes. [CUDA/cuDNN compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) can block Windows setup.

### 3. Confidence-gated cascade

1. For each candidate interval, compute colour, blankness, texture/hatch features and nearest legend distances on CPU; use the existing synthetic CNN where it performs well.
2. Accept only if the best match is clear and agrees with an independent texture/colour check; never classify blank or modifier-only regions as rock automatically.
3. Batch ambiguous crops through cached DINOv2, then SigLIP or a Jev visual choice only when distances remain close or modalities disagree.
4. Send a representative interval plus labeled swatch montage in one call; batch multiple questions about the same visual context where supported. Keep original coordinates outside the model.
5. Tune thresholds on held-out synthetic swatches plus the labeled example; report escalation rate and audit high-confidence mistakes, especially 1-bit same-colour rocks.
6. Preserve an explicit unknown/low-confidence result when no legend swatch matches. **Watch:** scores are not automatically calibrated across logs; a confidently wrong cheap stage prevents escalation.

## Illustrative cost per log

Assume **20 independent decisions**, each with one 320×320 image, 200 text input tokens and 50 output tokens; no retries, caching, tax or network cost. These are comparable call counts, not equivalent accuracy or actual token bills. List USD prices per million tokens; 51-log totals multiply by 51.

| Route | Assumptions | USD/log; 51 logs |
|---|---|---|
| Local existing model | Illustrative 60 s end-to-end at 250 W wall power, electricity $0.30/kWh; no API fee | $0.00125; $0.06375. Runtime and wall power unmeasured; excludes hardware. |
| Jev | [Direct price](https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost) $0.042/M input, output free. Assume **458 total input tokens/call**, including image; actual image billing unverified | $0.000385; $0.0196. At user-observed 0.4 s sequentially: ~8 s/log, excluding overhead. |
| GPT-4.1 mini | [$0.40/$1.60](https://developers.openai.com/api/docs/models/gpt-4.1-mini); [320² image = 100 32px patches × 1.62 = 162 tokens](https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses) | $0.00450; $0.229. Model is deprecated; confirm availability. |
| Gemini 2.5 Flash | Google search result states $0.30/$2.50 and 258 image tokens for ≤384²; [pricing](https://ai.google.dev/gemini-api/docs/pricing), [tokens](https://ai.google.dev/gemini-api/docs/tokens) | $0.00775; $0.395. The source pages could not be opened by the browser; rates/token rule **unverified beyond search extracts**. Thinking output may add cost. |
| Claude Haiku 4.5 | [$1/$5](https://platform.claude.com/docs/en/about-claude/pricing); [image estimate](https://platform.claude.com/docs/en/build-with-claude/vision) 320×320/750 ≈ 137 tokens | $0.01174; $0.599. Actual tokenization and thinking usage may differ. |

Use returned usage rather than assumed image tokens; more images, resolution, prompt context, retries and output change costs. Hosted multi-question calls can amortize input and latency; 20 calls is illustrative, not a recommendation.

## Do not spend hackathon time on

- Whole-page Pillow `convert('RGB')` or a single global thumbnail: peak memory and lost hatch/depth detail; `crop()` itself may load the page.
- INT8 before measuring FP16: calibration and ranking drift can cost more than the saved milliseconds; ONNX GPU INT8 generally needs a compatible TensorRT path, not merely a quantized CPU graph.
- `torch.compile` as a prerequisite: compilation and Windows GPU backend setup can exceed a 51-log run's savings; [Windows CPU/XPU guide](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) does not establish support for this CUDA configuration. Benchmark cold start and steady state if already available.
- One API call per sliding window or full-length image: network latency and image-token volume scale badly, and vertical downsampling destroys detail. Use representatives and ambiguity routing.
