# R2 — long-strip interval segmentation and grouping

**Decision:** do not add another generic colour-run or CNN experiment. The most promising cheap delta is a fixed multiscale *texture* embedding (wavelet scattering) sampled down the column, then an offline CPD method. It directly represents hatch frequency/orientation without needing the one labelled log. Treat all timings below as estimates for the 4080 / 300k-row worst case, not published benchmarks.

## Candidates

| Method | Source (opened; year) | Step | Why it may beat current bake-off | Downloadable now? | Cost / size | Plug-in | Evidence quality |
|---|---|---|---|---|---|---|---|
| 2-D wavelet scattering patches + L2-PELT | [Kymatio](https://github.com/kymatio/kymatio) (2019–); [PELT](https://github.com/deepcharles/ruptures/blob/master/src/ruptures/detection/pelt.py) (2018–) | 3, 4 | Fixed multiscale orientation/frequency coefficients should distinguish monochrome hatches and same-colour rocks; no training or internet weights. PELT accepts multivariate input and has `jump` subsampling. | Yes: `kymatio`, `ruptures`; Kymatio BSD-3; ruptures licence **not verified**. No weights. | 0 learned parameters; GPU batched FFT; estimate 1–4 min/log at 8-px stride, <2 GB. | 6–10 h | Kymatio documents fixed-filter, translation-invariant 1/2/3-D GPU transforms; no mud-log benchmark. **Claim only for this task.** |
| ClaSP on a compact multivariate texture sequence | [paper](https://arxiv.org/abs/2207.13987) (2022); [ClaSPy](https://github.com/ermshaua/claspy) (current) | 3 | Self-supervised split classifier detects distribution/shape shifts, chooses count/location autonomously, rather than requiring a PELT penalty. Good independent validator of scattering-PELT. | Yes: `pip install claspy`, BSD-3; no weights. | CPU/Numba; 0 learned parameters. First use a 16–32-D PCA sequence at 16–32 px stride; estimate minutes, verify locally. | 3–5 h | Authors report lowest segmentation error and 80/98 best segmentations vs five methods; reproduction reports 0.9% error. Not imagery. |
| Multiscale uniform-LBP + Gabor-energy sequence + PELT | [LBP](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html) (docs 2026); [Gabor](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html) (docs 2026) | 3, 5 | Very fast CPU fallback and legend-comparable descriptor; LBP bins and filter energy ignore absolute colour, ideal for 1-bit hatch. Add Lab mean/variance for coloured logs. | Yes: `scikit-image`; licence **not verified here**; no weights. | CPU convolution, ~30–80 features; estimate <1 min at 16-px stride. | 3–6 h | Docs show LBP histograms classify texture and Gabor mean/variance features; their Gabor example misclassifies one rotated grass sample. **Weak task evidence.** |
| FLOSS/FLUSS matrix-profile regimes | [STUMPY](https://github.com/stumpy-dev/stumpy/blob/main/README.rst) (2017–); [tutorial](https://stumpy.readthedocs.io/en/latest/Tutorial_Semantic_Segmentation.html) | 3 | Finds repeated regimes and has `gpu_stump`; useful where the same lithology recurs far apart. | Yes: `pip install stumpy`; licence **not verified here**; no weights. | GPU supported; matrix-profile memory/time can be large; only try on scalar PCA-1 sequence downsampled to <=20k. | 4–6 h | Original method is explicitly semantic segmentation; no inspected quantitative result for logs. |
| RBF-kernel PELT / KernelCPD | [ruptures RBF cost](https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/costs/costrbf.html) (2018 docs; method 2012) | 3 | Nonlinear mean change can separate feature distributions that L2 misses. | Yes: `ruptures`; licence **not verified**; no weights. | CPU; RBF Gram matrix is impractical at 19k–37k samples. | 1–2 h | Kernel mean-change formulation is documented; no relevant benchmark. Use only <=5k coarse points. |
| BOCPD / Fast-BOCPD | [bocd](https://github.com/fiannai/bocd) (2026); [Fast-BOCPD](https://fast-bocpd.readthedocs.io/en/latest/) (2025) | 3 | Streams tiles and offers Student-t outlier robustness; could emit candidate boundaries before image ends. | `bocd` MIT; Fast-BOCPD MIT; no weights. | CPU; bounded max run length, low memory. | 4–8 h | Package says vectorised/log-space; Fast-BOCPD has no inspected accuracy benchmark. **Claim only.** |
| HDBSCAN over interval descriptors | [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) (2017–) | 4 | Selects a stable cluster count, supports noise, and reports persistence/soft membership—much safer than fixed-k k-means for modifier/blank intervals. | Yes: `hdbscan`, BSD-3; no weights. | CPU; hundreds of intervals, negligible. | 2–4 h | Its stability-based extraction and persistence are documented; no lithology benchmark. |
| TIRE / TS-CP2 learned CPD | [TIRE](https://github.com/deryckt/TIRE) (2021); [TS-CP2](https://arxiv.org/abs/2011.14097) (2021) | 3 | Could learn subtle frequency changes from this corpus. | TIRE code exists but licence/weights **not verified**; TS-CP2 code/weights **not verified**. | Requires per-log/network training; GPU but unknown size. | 12–24 h | TS-CP2 reports +79.4% F1 over handcrafted and +17% over deep baselines on three time-series sets—not document textures. |

## Top 3 recommendations

### 1. Scattering → PCA/robust scale → L2-PELT, then HDBSCAN

1. Add `kymatio`, `ruptures`, `scikit-learn`, `hdbscan`; retain TIFF memmap/tiled reads—never construct the entire RGB strip on GPU.
2. Crop only the lithology x-box; remove a 2–4 px edge margin and use gray plus Lab a/b (or gray only for 1-bit).
3. For centres every 8 px, form 128×64 patches (resize/pad horizontally only); batch 128–512 into `kymatio.torch.Scattering2D(J=3, L=8, shape=(128,64))` on CUDA.
4. Spatial-average coefficients; append robust Lab mean/std and replace non-finite values. Fit PCA(16, whiten=True) on at most 20k sampled rows; transform all rows.
5. Set `min_size=ceil(80/8)` and run `rpt.Pelt(model="l2", min_size=..., jump=1).fit_predict(Z, pen=...)` over a small log-spaced penalty sweep.
6. Choose the least-complex solution within 5% of the minimum held-out (alternating blocks) reconstruction SSE; convert indices ×8 to `top_y/bottom_y`.
7. Locally refine each boundary ±32 px using the maximum adjacent-window cosine/chi-square descriptor distance, then merge intervals shorter than 24 px.
8. Describe each final interval by median scattering/Lab features; use `hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method="eom")` and retain `probabilities_`/persistence.
9. Match stable cluster medoids to identically embedded legend swatches; leave low-confidence/noise as `unknown/modifier` rather than forcing rock.

**Watch:** horizontal resizing can alter hatch pitch, and scattering can treat printed labels/grid lines as texture. Use an ink/text mask or median aggregation and compare original vs 0/90-degree rotation only where legend orientation permits.

### 2. ClaSP as boundary-count-free cross-check / ensemble

1. Reuse the 16-D scattering/PCA sequence above, but subsample to 16 px (<=18,750 points at 300k rows).
2. `from claspy.segmentation import BinaryClaSPSegmentation`; pass `Z16` as the documented 2-D multivariate array.
3. Start `BinaryClaSPSegmentation().fit_predict(Z16)`; map returned indices ×16 to y.
4. Run separately on gray texture-only features and Lab-colour features; union only boundaries within ±64 px that both modalities support, otherwise keep the higher local two-window distance.
5. Merge candidates with scattering-PELT: snap PELT cuts to a ClaSP cut inside ±64 px; retain isolated ClaSP cuts only above the 90th percentile local contrast.
6. Preserve ClaSP score profiles for the annotated debug image and tune only `window_size` from expected minimum bed thickness (try 64, 128, 256 px / 16).

**Watch:** ClaSP’s published benchmark is ordinary time series, and its own docs advise supplying only necessary dimensions. PCA channels dominated by colour/text will create false transitions; do not feed raw patch pixels or >32 channels.

### 3. HDBSCAN and legend-aware robust cluster selection

1. Segment with the best existing/new boundary proposal; build one descriptor per interval using median/10–90% quantiles of scattering/LBP/Lab row features.
2. Standardise on in-column nonblank rows; L2-normalise descriptor blocks so colour cannot swamp texture.
3. `hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean", cluster_selection_method="eom", allow_single_cluster=True).fit(X)`.
4. Use `labels_`, `probabilities_`, and `cluster_persistence_`; label `-1` and probability <0.35 unknown rather than inventing a lithology.
5. For each cluster, compare its medoid against every parsed **rock** swatch using the same descriptor and a calibrated cosine distance; require a margin over the best non-rock symbol.
6. If all intervals become one cluster, fall back to `cluster_selection_method="leaf"`; if many singletons, raise `min_cluster_size` to 3 and merge adjacent same-legend assignments.

**Watch:** a log with one occurrence per lithology is fundamentally unclusterable; legend nearest-neighbour must still name individual intervals. HDBSCAN is a grouping/uncertainty layer, not a boundary detector.

## Do not spend bake-off time on

- **Full-resolution RBF/kernel PELT:** the RBF cost needs pairwise distances; it is the wrong memory curve for 19k–37k feature rows. Keep it only as a very coarse diagnostic.
- **BOCPD, FLOSS/GPU-STUMP, or Fast-BOCPD as the primary:** streaming is irrelevant for an already complete TIFF, their crucial window/hazard choice is another unlabelled hyperparameter, and matrix profiles need aggressive downsampling. Use only to arbitrate a difficult log.
- **TIRE, TS-CP2, new DINO/CLIP finetuning, SAM, or an LLM at row/patch level:** impressive CPD claims are not on scanned hatch textures; they need per-domain optimisation/training, add GPU cost, and do not solve legend matching. The one 17-interval label cannot validate them without leakage.
- **Selecting k with silhouette/elbow or a Dirichlet-process mixture:** it forces every blank/modifier into a rock and makes the answer sensitive to rare lithologies. Prefer HDBSCAN’s stable clusters plus legend distance and an explicit unknown state.
