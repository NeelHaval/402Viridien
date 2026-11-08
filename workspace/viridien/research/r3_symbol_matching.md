# R3 — Matching lithology intervals to legend swatches

Research checked 2026-11-07. **Start with spectral texture + colour, then use existing DINOv2 patch features.** Train prototypes only if they improve held-out real examples. Times are estimates; inference and Windows/Python compatibility are unverified. No verified mud-log benchmark.

## Candidates

All rows concern step 5 unless noted. “Available” refers to upstream links, not a download or local test.

| Method and opened source (year) | Step | Potential advantage over existing bake-off | Downloadable now; weights and licence | Cost / size | Effort | Evidence actually observed |
|---|---|---|---|---|---|---|
| [FFT texture + log-polar registration](https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html) (1996 algorithm; current docs) | 5 | Fourier magnitude tolerates hatch phase; preserve direction and frequency unlike colour; bounded scale search supplements NCC | No weights; scikit-image BSD-3-Clause | CPU; small arrays, FFT O(n log n); negligible model | 3–6 h | Docs recover 53.75° vs 53.7° and 2.198 vs 2.2 on transformed retina; not a texture benchmark. |
| [Uncharted hybrid](https://uncharted.software/blog/uncharted-earns-another-spot-on-the-leaderboard-in-ai-competition-by-darpa-and-usgs/) (2022) | 5; also 2–3 | Per-map colour, high-pass filtering and noise removal target similar fills; multiple cues beat single NCC in principle | Description public; matching weights/code/licence **not verified** | CPU/hybrid; size and runtime unverified | 4–8 h for inspired version | Team reports first in point and second in polygon categories; point uses multi-rotation cross-correlation. No mud-log score. |
| [LOAM](https://github.com/Fandel-Lin/LOAM) (2023) | 3, 5 | Legend-conditioned map/key bitmap + CNN handles local map-specific cues | Code and checkpoint directory public; checkpoint usability and licence **unverified** | GPU; model size/runtime unverified | 12–24 h | Domain-specific map method; transfer to narrow vertical strips unverified. |
| [DIGMAPPER/TOPAZ](https://arxiv.org/abs/2506.16006) (2025) | 2, 3, 5 | Joint map polygon and legend association suggests swatch-conditioned segmentation | Paper public; usable TOPAZ checkpoint/code and licence **unverified** | GPU; size/runtime unverified | 16–40+ h | Weighted median polygon F1: 0.98/0.77/0.28 on excellent/good/fair categories, 2/5/4 maps respectively; not independent mud-log performance. |
| [Prototype / Siamese metric](https://arxiv.org/abs/1703.05175) (2017) + swatch synthesis | 5 | Compare support/query augmentations instead of fixed-class synthetic CNN; supports new per-log classes and multiple crops | Algorithm public; train own weights; model licence not applicable | Tiny CNN, e.g. 0.1–2M parameters by design; CPU inference / GPU training, unmeasured | 6–12 h | Few-shot prototype principle demonstrated on Omniglot and miniImageNet; synthetic-to-real mud-log improvement **unverified**. |
| [DINOv2 patch prototypes](https://github.com/NVlabs/RADIO/blob/main/README.md) (2023) | 5 | Reuse already-loaded small encoder; distribution of local patch similarities can retain motifs lost in one global embedding | Existing DINOv2 weights; Apache-2.0 per upstream DINOv2 release; availability already established by user | ViT-S ~22M parameters; batched GPU; latency unmeasured | 2–5 h | DINOv2 baseline already tried; incremental benefit of patch distribution is a hypothesis. |
| [DINOv3 ViT-S/16](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) (2025) | 5 | Improved dense features may distinguish local hatch; compare same retrieval protocol | Weights released, access may be gated; [DINOv3 licence](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/blob/refs%2Fpr%2F1/LICENSE.md), use restrictions | ~21M parameters; GPU batches, runtime unmeasured | 3–6 h plus access | Meta claims stronger dense features; no mud-log test. Verify access before relying on it. |
| [SigLIP 2 base/16](https://huggingface.co/google/siglip2-base-patch16-224/blob/02c35f2c035e0ed4a367fb10a892c1fe2a3f364e/model.safetensors) (2025) | 5 | Improved dense features over SigLIP, batched image-image retrieval | Public weights, Apache-2.0 | Full checkpoint 1.5 GB; base vision tower ~86M parameters; GPU batch latency unmeasured | 3–6 h | [Paper](https://arxiv.org/abs/2502.14786) claims improvements over SigLIP across scales; no hatch benchmark. |
| [RADIOv2.5-B](https://github.com/NVlabs/RADIO/blob/main/README.md) (2024) | 5 | Distilled DINO/CLIP/SAM features may improve dense correspondence | Public weights linked from repo; RADIO licence **non-commercial**; C-RADIO has separate NVIDIA Open Model licence | ViT-B ~86M parameters; GPU, runtime unverified | 4–8 h | Authors claim strong dense features; no mud-log result. Licence may rule out deployment. |

The 2022 DARPA/USGS **feature-extraction** winner was USC ISI/University of Minnesota; Jataware placed second in **georeferencing** ([winner report](https://www.executivegov.com/articles/darpa-usgs-name-winners-of-mineral-assessment-competition-stefanie-tompkins); page open failed, ranking from search excerpt, independently unverified). Uncharted's category placements are self-reported. [USGS](https://www.usgs.gov/publications/extracting-data-maps-lessons-learned-artificial-intelligence-critical-mineral) reports prompt-based median polygon F1 0.77, line 0.56, point 0.35; these are map extraction scores. DIGMAPPER is later work, not established as the winning submission. Its [LDTR](https://github.com/knowledge-computing/LDTR) extracts **fault-line graphs**, not polygon fills. The precise winning ISI method and Jataware feature-matching implementation/checkpoints were not verified. Each map's legend is local, per the [competition FAQ](https://criticalminerals.darpa.mil/FAQs).

## Top three implementation recipes

### 1. Phase-tolerant texture + colour, with rejection

1. In `numpy`, `opencv-python` and `scipy`, crop each legend swatch **inside** its border; retain only rock-labelled items and their OCR confidence.
2. Sample up to 12 evenly spaced, interior square windows per predicted interval, away from boundaries; cache descriptors once per window and swatch.
3. Match physical pixel scale first; compare a small bounded scale grid, e.g. 0.8–1.25, without rotating by default: hatch angle can encode rock type.
4. For each patch, subtract background, remove obvious border rules, apply a Hann window, then compute normalized `abs(np.fft.fft2(...))`; mask DC and compare directional frequency bins with cosine distance.
5. Combine spectrum with ink fraction and robust Lab median/histogram distance; colour weight zero for 1-bit scans. Handle blank swatches separately.
6. Optionally use `skimage.transform.warp_polar` + `skimage.registration.phase_cross_correlation` on magnitude spectra to estimate small scale/rotation differences; verify a bounded transformation with correlation. Magnitude comparison is phase-tolerant, but discards phase information.
7. Aggregate clean-window scores by median per interval; exclude text/lines where detectable. Return top two, margin and `unknown` if absolute score or margin fails thresholds, calibrated on labelled example plus manually checked hard cases.
8. Failure: sparse or tiny swatches lack enough hatch periods; stripes with similar frequency, blank windows and overprint can collide. Do not force a label.

### 2. Local DINOv2 prototypes and robust patch voting

1. Reuse `facebook/dinov2-small` in `transformers` or the existing DINOv2 harness; call its vision encoder with `output_hidden_states=True`.
2. Make multiple swatch crops and interval windows at matched native scale, padding rather than stretching; repeat grayscale into RGB. Batch with `torch.inference_mode()` on CUDA.
3. Remove the CLS token and any register tokens before collecting patch tokens; check actual sequence shape for the selected checkpoint.
4. L2-normalize local token vectors; cache a set per swatch. Match each interval token against its nearest swatch token, then aggregate a trimmed mean or median over clean windows.
5. Score both directions or compare token-distribution quantiles to avoid one common background token dominating. Blend ranks with recipe 1's colour/texture score.
6. Keep multiple swatches per lithology and aggregate by class; preserve a separate unknown option and top-two margin. Evaluate versus existing global DINOv2 embeddings at equal crops.
7. If necessary, substitute `google/siglip2-base-patch16-224` and `AutoModel`/`AutoProcessor`; use the vision model's `last_hidden_state` and validate its actual token layout. DINOv3 is a later gated ablation.
8. Failure: 14/16-pixel patches can erase thin strokes, and natural-image features may ignore hatches. Tile at multiple native scales and measure, not assume improvement.

### 3. Swatch-episode prototypes, trained on synthetic variation

1. Reuse the synthetic-column generator and small CNN already in the bake-off; add metric learning rather than replacing the baseline blindly.
2. For each log, create support/query crops from each rock swatch with random phase translations, bounded scaling, scan noise, mild blur, colour drift and simulated text/rule occlusion.
3. Use only plausible small rotations: directional hatches can denote different classes. For 1-bit samples simulate thresholding, preserving separate blank classes.
4. With PyTorch, train an embedding CNN episodically: in each episode average support embeddings per swatch, compute negative squared Euclidean distances for queries, and optimize cross-entropy.
5. At inference average several deterministic swatch-crop embeddings into prototypes; score multiple clean interval windows and take a robust median. Cache prototypes per log.
6. Split validation by **real log**, not synthetic crops from the same swatch; also report held-out manually labelled real interval windows and per-rock accuracy.
7. Compare directly with the existing synthetic fixed-class CNN at the same data budget; if no real-data gain, retain the simpler baseline.
8. Failure: synthetic variants can reproduce the legend's artefacts but miss actual print-scale changes; classes with near-identical swatches or too few motif repeats are intrinsically ambiguous.

For all three: preserve upstream `top_y`, `bottom_y`, depths and `cluster_id`; vote by cluster only when windows agree. Review disagreements with the existing JEV/vision fallback. Report top-1 accuracy on 17 labelled intervals, macro per-class accuracy, unknown rate, latency and peak VRAM; estimates are uncertain. Label additional real 1-bit, same-colour hatch and blank windows before tuning. Synthetic validation is not independent real evidence.

## Do not prioritise

- Full map polygon or fault pipelines: LOAM/TOPAZ require adaptation and LDTR addresses lines; their map benchmarks do not establish improvement for a 63-pixel median strip.
- Large foundation models or a vision LLM per window: repeated crops raise latency and cost; run the existing LLM/JEV only on ambiguous clusters. Avoid DINOv3-7B on 12 GB as a first experiment.
- Unconstrained rotation invariance, forcing a single nearest legend, or colour-only matching: these can merge distinct hatches, modifier overlays, unknowns and blank stretches.
- Training a new generic Siamese network from scratch or large supervised map model without independent real validation: one swatch per class and synthetic crops do not establish generalization. RADIO's non-commercial licence also limits its use.
