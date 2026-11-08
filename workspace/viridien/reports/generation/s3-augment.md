# 1. Verdict

Useful prototype; scan realism is not yet fit to certify synthetic-to-real equivalence.
The biggest demonstrated problem is missing column rails: an exact-class, rail-only discriminator separates current fakes at 96.9%.
The historical 63.2% score is reproducible, but neither it nor the newer 57.0% score rules out strong visible shortcuts.

# 2. Findings, ranked by impact

Measurements use published v1/v5 native PNGs, accepted real boxes, and the current assembler library. No test-well pixels were read. TIFF inspection was header-only; images were processed one crop at a time. Scripts, numerical outputs, source fingerprints and numbered source snapshots are in `data/genlab/critique/s3-augment/`. Builders were changing assembler code/library during review; counts below refer to the saved probe outputs.

## 1. Strip generation omits the frame that dominates real crops

**Severity: major.**

**Evidence.** In exact document/class matches, 60/64 v5 validation real crops have an outermost column that is dark along >80% of its height; 0/64 corresponding augmented fakes do. A one-feature logistic classifier, fitted on training wells, scores **124/128 = 96.875%** on these validation images. The feature is simply maximum dark occupancy of the left/right edge, with dark defined as maximum RGB <80. Each pair uses the same native height; widths remain native. This is a separate diagnostic, not the published discriminator.

`genlab/dataset.py:416` calls `render.layer` directly; it bypasses the frame-aware `render.sample` at `genlab/render.py:40`. Rails reach augmented strips only through a fitted measurement (`genlab/realism.py:81`). Validation measurements are correctly excluded (`genlab/dataset.py:370`), but the class-only pool at `genlab/dataset.py:410` consequently supplies no measurement or frame for those classes.

I looked at [the six exact-class real/clean/augmented comparisons](s3-augment/real-clean-augmented.png), all enlarged by the same integer factor. The first green shale pair shows strong real rails and frameless fakes. Source PNGs and native crop dimensions are recorded in `s3-augment/image-sources.json`; statistics are in `s3-augment/probes.json`.

The original discriminator is a weak aggregate check of this defect. I reproduced **96/152 = 63.158%** for v1 and **90/158 = 56.962%** for v5. A rail-only probe on its original name-matched pairs scores 63.8% and 87.3%, respectively. However, removing three edge pixels from both domains changes the original discriminator to 69.1% for v1 and 53.8% for v5. Thus rails are sufficient for discrimination, but do **not** explain all of v1's result. Its 12×20 bilinear representation (`genlab/dataset_eval.py:61`) and name-based matching (`genlab/dataset_eval.py:107`) mix several differences. Native height/width/aspect alone separate v1 at 96.1%. See `s3-augment/audit.json` and `s3-augment/ablations.json`.

**Fix.** Give clean and augmented strips an explicit frame policy in `DatasetBuilder.build`, independent of motif residual availability. Fit rail width, ink and presence distributions from training wells, conditional on scan family; preserve a deliberate frameless arm for cropped-interior classification. Never estimate held-well rail parameters from evaluation column pixels. Add exact-class, native-size rail and interior probes alongside the existing discriminator.

**Verify.** Rebuild and repeat the 64-pair diagnostic; close the 60-versus-0 occupancy discrepancy and require the rail-only probe to approach chance without damaging interior recognition. Report results by well.

**Effort: M. Expected gain:** removes a shortcut currently supporting 96.9% discrimination and makes frame-sensitive detector training more representative; downstream accuracy gain remains unmeasured.

## 2. Identity augmentation preserves a twofold stroke-weight error

**Severity: major.**

**Evidence.** For validation class `7dc28eda27f5c429:23`, I measured vertical black runs after removing three edge columns, at native scale: **all 344 real runs are 2 pixels thick; all 344 clean and augmented fake runs are 1 pixel thick**. Interior dark occupancy is **16.67% real versus 8.33% fake**. Both domains have exactly two RGB colours, so neither Gaussian blur nor added grain is supported as the explanation. I inspected [the matching x6 pixel enlargement](s3-augment/stroke-width.png), which also shows differences in dash geometry. The run measurement isolates the thickness defect; it does not prove all geometry is otherwise correct.

Across v5, **1,324/1,600 augmented strips have no measurement**, and **1,369/1,600 (85.6%) have identity augmentation**: no blur, noise, drift, pasted pixels or rails. Same-class-only lookup at `genlab/dataset.py:410` falls through to identity at `genlab/realism.py:60`. The example's clean and augmented panels are identical. Evidence: `s3-augment/ablations.json`, `s3-augment/audit.json`.

**Fix.** Separate motif learning from scan-style calibration. Measure stroke thickness and edge profiles on training pages relative to their legend swatches, and fit a family-conditioned legend-to-column style prior. Apply controlled ink expansion or native-resolution rerendering where supported. Preserve the current prohibition on fitting validation-column residuals. Record unmeasured augmentation explicitly and distinguish it from a successful measured identity fit.

**Verify.** On unseen training-well calibration folds, compare native ink-run distributions and edge profiles before and after correction; then check held validation without refitting. Include the demonstrated 1-versus-2-pixel case and monitor class recognition for merged neighbouring marks.

**Effort: M. Expected gain:** corrects the demonstrated twofold ink-coverage deficit; improves stroke realism where legend and column printing weights differ. Identity itself is appropriate when the measured style already matches.

## 3. The 5% overlay cap converts coherent residuals into artificial pepper noise

**Severity: major.**

**Evidence.** I exercised the current `dataset_augment` on saved training residual `c72589f5d566d9b5563875b8` (`5709fe9e86ce942b:1`, source box `[532,28581,575,28595]`). Its usable mask contains **210 pixels in 2 connected components**, largest 119 pixels. The cap leaves **30 pixels in 27 components**, largest 2 pixels; **80% of retained pixels are isolated singletons**. The code chooses individual pixels uniformly at `genlab/dataset.py:112`, rather than retaining connected ink structures.

I inspected [source, original mask, capped mask and neutral-fill application](s3-augment/residual-thinning.png). The source contains motif/background disagreement, so this is not evidence of verified text. It demonstrates why an unverified residual cannot safely stand in for text or curves: the current policy further destroys its spatial structure. Numbers and actual augmentation parameters are in `s3-augment/probes.json`.

**Fix.** Select or reject whole connected components under an area budget. Reject broad colour/motif disagreements; require darker-than-predicted ink and component review before assigning text/curve semantics. The assembler's darker-ink selection at `genlab/assemble/noise.py:270` is a useful starting point. Place retained components once, preserving native connectivity, instead of periodically tiling a residual mask (`genlab/realism.py:74`).

**Verify.** Re-run this exact fixture and a small inspected component gallery. The cap should reject this large ambiguous field or retain coherent structures; it should not transform two structures into 27 specks. Separately measure coverage of actual text and curves before claiming those nuisances are represented.

**Effort: M. Expected gain:** removes the demonstrated artificial singleton texture and preserves the topology needed to train overlay robustness.

## 4. Assembler scan noise repeats exactly inside supposedly nonrepeating blocks

**Severity: major.**

**Evidence.** `genlab/assemble/compose.py:429` and `:430` wrap one residual field modulo its height and width within each 512-row output block. Randomizing the origin once per block does not stop repetition inside it.

Using the live `Composer.appearance` with actual residual `127c61078b9e146a86dd`, I isolated scan noise on a constant RGB-128 fill, disabling overlays, boundaries and rails. The measured field is **117×76 pixels**. Within the first 512 output rows, **100% of RGB pixels equal the pixels 117 rows later**, despite nonzero output standard deviation **0.825 grey levels**. This is an isolated component experiment, not a claim that a whole generated lithology column repeats identically. I inspected [the output with 117-row markers](s3-augment/assembler-period.png); `s3-augment/probes.json` records the parameters.

**Fix.** In `Composer.appearance`, sample smaller independently chosen valid blocks across multiple training residuals, with overlap/blending, or fit a family-specific correlated-noise model to flat-region residual spectra. Preserve any supported scanner/block coordinate structure explicitly. Do not repeat an entire source field to fill a larger block.

**Verify.** On constant fills, test lag equality and autocorrelation at every donor height and width, plus discontinuities at the 512-row join. Compare flat-region spectra with independent training-well crops. Require disappearance of exact donor-period copies without introducing seams.

**Effort: M. Expected gain:** removes the demonstrated deterministic scan-texture fingerprint from long synthetic columns.

## 5. Noise fitting checks source wells but omits the tile's donor ancestry

**Severity: major; potential test leakage, not an observed contaminated output.**

**Evidence.** `genlab/assemble/noise.py:370` accepts a card when its document and `real_doc_id` are training wells, then opens its generated model at `:374`. It never checks `card['donor']`. Conversely, `genlab/assemble/compose.py:358` explicitly rejects test-well donors. A fitted residual depends on both the real crop and the model tile, so checking only the real-pixel source is insufficient.

A predicate fixture using training document `5709fe9e86ce942b` and test donor `5087cb9d7b905018` passes the noise-fit gate and fails the Composer gate. This uses the actual split-policy hash; it does not read the test image. See `s3-augment/donor-guard.json`. A cached library fitted through that gap can later pass Composer's source-document checks because measurement records do not retain the tile donor.

The actual audited library has **0/354 measurements with a direct test source/document**. Published v1/v5 manifests and measurement document IDs also have zero direct test inputs; the live board's one explicit donor is not a test well. Thus I found a concrete unguarded path, **not evidence that it has already leaked**. Split checks are in `s3-augment/split-audit.json`.

**Fix.** Share a recursive model-ancestry training guard between noise fitting, dataset snapshots and composition. Reject unknown ancestry or any test-well ancestor before loading a tile. Persist model/donor document IDs and fingerprints with each residual and validate them when loading cached libraries.

**Verify.** Supply the fixture card to the guarded noise fitter and require rejection before model loading or measurement; audit every persisted ancestor against `data/normalized/split-policy.json`.

**Effort: S for the direct-donor guard; M for persisted recursive ancestry. Expected gain:** closes a route by which held-out pixels can influence generation through fitted residuals, even when the pasted RGB comes from a training well.

# 3. Quick wins

- Add rail occupancy and native ink-run summaries to the realism report; retain the existing discriminator for comparison.
- Label augmented identities and report measured-versus-unmeasured coverage separately by split and family.
- Reject oversized ambiguous residual components instead of randomly thinning their pixels.
- Add the missing direct-donor split check to `noise.fit` before opening a model.
- Include constant-fill donor-period checks in the assembler's scan-noise verification.

Each is scoped to less than an hour; the broader fixes above need calibration and verification.

# 4. What is already good

Native motif scale and explicit real/synthetic provenance are valuable. V5 correctly excludes validation residual fitting: v1 saved 239 validation measurements, v5 saves none. The assembler already keeps rails at edges and separates boundary appearance; its inspected `data/genlab/assemble/out/iteration-2/sheet-5709fe9e86ce942b.png` shows much better rail continuity than dataset strips.

# 5. Doubts

- The 32 permitted training TIFF headers are all **LZW**, comprising 28 RGB and 4 palette images, all 8-bit. That does not exclude JPEG artifacts inherited before TIFF encoding. I found no evidence supporting blanket JPEG recompression. Spectrum-only features on size-matched v1 validation pairs give AUC **0.474**; they do not establish noise spectrum as the cause of 63.2% discrimination.
- Robust incremental blur is zero in **978/981** v5 training fits; noise is nonzero in only **5/981**. Many inspected crops have discrete palettes and sharp edges. These measurements do not justify adding Gaussian grain or blur everywhere. Colour summaries on name-matched pairs are confounded by document/class appearance; I do not claim a measured scanner colour-drift defect from them.
- I have not measured the prevalence of semantically verified text, handwriting or curve crossings. Residual connected-component labels are heuristics, and the inspected residual example is not verified text. Full-page overlay semantics and detector robustness therefore remain unvalidated, rather than proven broken.
- The probe samples are small and clustered by well. No confidence interval or downstream detector gain is claimed. Current assembler files/library were being refined during review; reproduction inputs and source snapshots are saved, but these are not a frozen end-to-end assembler release.
