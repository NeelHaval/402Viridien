# R8 — small VLMs for legend-conditioned lithology and boxes

**Reality check (checked 2026-11-07).** Use VLMs as *adjudicators*, not per-row
segmenters: crop an interval and legend/contact sheet, then ask for one allowed label.
No source opened reports RTX 4080 Laptop latency or hatch-log accuracy. Latency is
therefore **unverified**: measure fixed 768px crop, 512px legend, 32 tokens. Storage
figures are weights only; visual tokens/KV cache can still exhaust 12 GB. Never feed a
whole 313k-px TIFF.

| Candidate method | Source opened (year) | Pipeline step | Why it may beat current bake-off | Downloadable now? | 12-GB cost / latency | Plug-in | Evidence quality |
|---|---|---|---|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | [official card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (2025) | interval+legend naming; OCR sanity check | Strong small document model; accepts multiple images, so preserve a separate high-res swatch/interval rather than a destructive montage. | Yes, weights; Qwen licence—**verify exact card terms before release**. Official AWQ also released. | 3B: >=6 GB BF16; practical 4-bit ~4–6 GB + image cache. 4080 latency unverified. | 4–8 h | Card: DocVQA 93.9, InfoVQA 77.1, AI2D 81.5 (3B). Relevant, but not texture proof. |
| **Qwen3-VL-4B Instruct** | [official repo](https://github.com/QwenLM/Qwen3-VL) (2025) | same; optional box questions | Newer fine-detail/OCR; 2B/4B/8B permits an on-machine sweep. | Yes; Apache-2.0 repo/weights. | 4B: >=8 GB BF16; use 4-bit. Latency unverified. | 6–12 h | Official claims only; no hatch result verified. |
| **InternVL3-2B / 8B** | [official card](https://huggingface.co/OpenGVLab/InternVL3-8B) (2025) | naming; multi-crop | Dynamic high-res tiling may help tiny text and narrow patterned strips. | Yes; MIT (Apache-2 Qwen component). | 2B BF16 likely fits; 8B needs 4-bit. Latency unverified. | 8–16 h | Paper: 2B MMBench 81.3, MTVQA 26.7; not lithology. |
| **MiniCPM-V 4.5** | [official card](https://huggingface.co/openbmb/MiniCPM-V-4_5) (2025) | high-res legend OCR + naming | Explicit multi-image/in-context support and vendor claims of unusually efficient high-res vision. | Yes; Apache-2.0. | 4.5B: BF16 borderline after images; 4-bit likely. Latency claim relative to Qwen, not 4080. | 8–16 h (custom chat API) | Paper claims 46.7% memory and 8.7% inference time of Qwen2.5-VL-7B on VideoMME—different workload. |
| **Phi-4-multimodal-instruct** | [official card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) (2025) | high-confidence second opinion; label/OCR | Best directly reported small-model document/diagram figures I found; useful only on uncertain clusters. | Yes; MIT. | 5.6B: >=11.2 GB BF16 before cache: use 4-bit, low visual-token budget; latency unverified. | 8–16 h | Card zero-shot: DocVQA 93.2, OCRBench 84.4, ChartQA 81.4, AI2D 82.3. Vendor evaluation. |
| **SmolVLM2-2.2B** | [official card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) (2025) | cheap all-cluster triage / fallback | Full Apache stack, 5.2-GB reported video GPU use; leaves headroom for a large crop and batch. | Yes; Apache-2.0. | 2.2B: ~4.4 GB BF16; reported 5.2 GB for video inference; latency unverified. | 3–6 h | Card: OCRBench 72.9, ChartQA 68.84, DocVQA 79.98: speed baseline, not best classifier. |
| **Gemma 3 4B IT** | [official card](https://huggingface.co/google/gemma-3-4b-it) (2025) | alternate naming | Good general vision, but weaker document numbers than Qwen/Phi; a controlled backup only. | Gated download; Gemma terms, not OSI open—accept terms. | >=8 GB BF16; 4-bit preferred. Latency unverified. | 4–8 h | Card: DocVQA 72.8, AI2D 63.2, ChartQA 74.7 (4B): vendor numbers. |
| **Moondream 3 Preview / 2** | [official docs](https://docs.moondream.ai/) (2025) | box/point proposal | Native detect/point/segment may help find three boxes. | Downloadable, but 3 is BSL 1.1/no third-party service; check 2 terms. | 3 is 9B MoE/2B active; memory/latency unverified. | 6–12 h | Vendor: DocVQA 79.3, ChartQA 77.5; no log test. |
| **Florence-2-large-ft** | [official card](https://huggingface.co/microsoft/Florence-2-large) (2024) | propose legend/depth/lithology boxes, OCR/grounding | A compact promptable detector/grounder; much better fit for box proposals than semantic lithology naming. | Yes; MIT; card download 3.12 GB. | 0.77B-ish model, 3.12-GB repository; likely easy on 12 GB, latency unverified. | 4–8 h | CVPR paper: TextVQA 81.5, plus RefCOCO grounding gains. Not trained to identify arbitrary column semantics. |
| **Molmo-7B-D** | [official card](https://huggingface.co/allenai/Molmo-7B-D-0924) (2024) | box/point QA and adjudication | Strong pointing/grounded-description research, fully open; worth a small box A/B only. | Yes; Apache-2.0. | 8B card, FP32 files: must quantize; BF16/4-bit peak **unverified**. | 12–20 h (`trust_remote_code`) | Official README reports 77.3 11-benchmark average; card says evaluated with FP32 weights/autocast—poor 12-GB fit. |
| **vLLM structured outputs + XGrammar** | [vLLM docs](https://docs.vllm.ai/en/latest/features/structured_outputs/) / [XGrammar](https://github.com/mlc-ai/xgrammar) (2024–25) | exact allowed label; JSON boxes | Removes malformed JSON and forces a label from legend-derived choices; prevents a formatting retry/cost. | Yes; Apache-2.0 components. | No model cost; vLLM Windows support/actual overhead **not verified**. Best via WSL2/Linux. | 2–6 h after serving works | Docs guarantee choice is exactly one offered; XGrammar claims 100% structural correctness, not semantic correctness. |
| **Outlines** | [official repo](https://github.com/dottxt-ai/outlines) (2025) | Transformers-side JSON/regex fallback | Same schema/choice idea when not serving vLLM; integrate only after confirming the chosen multimodal wrapper works. | Yes; Apache-2.0. | CPU-side constraint overhead; 4080 figures unverified. | 4–12 h | Project documents transformers/vLLM/Ollama support; formatting guarantee only, no interval benchmark. |

## Top three recommendations

### 1. Qwen2.5-VL-3B: primary local legend-conditioned classifier

1. `pip install -U transformers accelerate qwen-vl-utils pillow`; do **not** install or download during the contest run.
2. Load `Qwen/Qwen2.5-VL-3B-Instruct` in `torch.float16`, `device_map="cuda"`; use its `AutoProcessor`.
3. Per visual cluster select 2–4 crops with 10 px horizontal context; resize each longest edge to 768–1024 and retain greyscale for 1-bit logs.
4. Send the legend crop first and contact-sheet of only OCR-confirmed rock swatches second, then the interval crops as separate images.
5. Prompt: “Choose exactly one ID from `R0…Rk`; compare fill/hatch, ignore lines/text/modifier symbols; return only ID.” `do_sample=False`, `max_new_tokens=4`.
6. Ask each cluster once, not every interval; save raw response, confidence explanation in a separate optional pass, and the selected swatch ID.
7. Map ID to normalized legend text in Python; never let the VLM invent lithology spelling.
8. A/B threshold: use only if held-out crops from the labelled log beat your SigLIP/DINO nearest-swatch baseline.

**Watch:** fine hatch frequency vanishes under resizing and Qwen can privilege the printed label over swatch appearance. Give a high-resolution swatch crop, strip all legend text for the visual pass, and include text only in a second naming pass.

### 2. Florence-2-large-ft: low-cost three-box proposer, not rock classifier

1. `pip install -U transformers accelerate timm einops`; load `microsoft/Florence-2-large-ft` with `trust_remote_code=True`.
2. Downscale a full page to 1500–2000 px height for proposal only; keep original coordinates and remap results.
3. Run `<CAPTION_TO_PHRASE_GROUNDING>` with phrases `"vertical lithology column"`, `"depth numbers column"`, and `"legend colour swatches"` individually.
4. Parse returned polygons/boxes using `processor.post_process_generation(..., task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=...)`.
5. Reject boxes by geometry: lithology is long/narrow; depth has OCR digit density; legend has many small colour components plus text.
6. Fuse with your existing page heuristics; use it only to rank 5 candidates per box, then let the typed JEV box call or deterministic scoring choose.
7. Test labelled-example rotation/noise/downscale before enabling the bonus.

**Watch:** phrase grounding was not trained for document-layout roles and may box an entire page. It is a proposal generator; never replace geometry/OCR validation with it.

### 3. Phi-4-MM (uncertain cases) plus constrained choice output

1. Use `microsoft/Phi-4-multimodal-instruct` only for clusters where Qwen’s answer disagrees with nearest-swatch similarity or confidence margin is low.
2. Start with `transformers`, BF16 only if peak fits; otherwise test an NF4/4-bit loader in a clean environment—Windows compatibility and quality are **unverified**.
3. Provide one legend contact sheet and one interval crop; cap to 32 output tokens and greedy decoding.
4. If WSL2/Linux vLLM is available, serve the selected VLM and send `extra_body={"structured_outputs":{"choice": rock_ids}}`.
5. For boxes require a small JSON schema `{name, x0, y0, x1, y1}`; validate bounds and ordering in Python.
6. On native Windows without a working VLM vLLM build, request only `Rk`, strip/validate it locally, then retry once with a stronger closed-choice prompt—do not pretend JSON is constrained.
7. Log semantic accuracy separately from JSON validity; constrained decoding cannot fix a wrong visual match.

**Watch:** at 5.6B Phi can exceed 12 GB once visual tokens/KV cache are included; an OOM or aggressive resize negates its benchmark advantage. It is an adjudicator, not the default.

## Do not bother (for this challenge)

- **Qwen3-VL-8B, InternVL3-8B, Molmo-7B at BF16:** weight storage alone nearly/overfills 12 GB before image cache. Quantized experiments are possible but not the first efficient path.
- **Gemma 3 4B as the primary:** gated terms and materially lower published DocVQA than Qwen2.5/Phi; no compensating pattern evidence found.
- **Moondream 3 for the core submission:** intriguing native grounding, but Preview/BSL restrictions and no independently verified log texture result make it bonus-only.
- **Free-form JSON / chain-of-thought:** it wastes tokens and produces brittle parsing. Use a one-token label choice; use XGrammar/vLLM only for syntax, with Python validating coordinates.
- **Whole-page VLM prompting or fine-tuning:** the raw height and one 17-interval label make both slow and under-supervised. Segment/cluster first, then make tens of VLM calls rather than millions of pixels of context.
