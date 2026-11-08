# S6 SigLIP

Uses cached `google/siglip-base-patch16-224` image and text encoders. The lithology column is sampled in overlapping 25-row strips (12-row step), with batched image inference. Legend swatches are tiled to square images and embedded; text prompts use `a geological lithology symbol for <label>`. Per-strip scores are an equal blend of image-to-swatch and image-to-text cosine similarity. A five-strip moving average suppresses short label flips. Predicted label changes define interval boundaries; output includes four pipeline stages.

## Tuning and results

Compared image/text weights 0.78/0.22 and 0.5/0.5 on the first deterministic log06 synthetic column. Equal weights improved row accuracy from 0.253 to 0.324 and boundary F1 from 0.250 to 0.367, so the equal blend was retained. This is a single synthetic sample comparison, not a broad weight sweep.

Final `eval --gt gold`: 13 windows, no crashes, one scored window. On log06, row accuracy 0.0105, boundary F1 0.1985, cluster ARI 0.3314, score 0.00208, and depth MAE 0.462 ft. The output image shows frequent short intervals and a severe naming collapse toward Silty Sandstone on the striped regions. The other 12 windows have no gold truth and therefore only provide runtime/resource measurements.

Mean time was 8.37 seconds/window; mean peak reserved GPU memory was 1.07 GiB. Per-window peak RSS averaged 1.81 GiB. All windows completed within the 60-second budget.

## Limits / next steps

SigLIP embeddings capture broad appearance but do not reliably distinguish the fine repeated hatch patterns. Test longer vertical context and multi-scale embeddings, normalize image/text similarities per strip before blending, and evaluate the weight across all synthetic columns. A dedicated boundary detector could replace label flips as the interval segmentation cue.
