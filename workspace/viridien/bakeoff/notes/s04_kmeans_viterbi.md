# S4 tile k-means + Viterbi

The strategy describes each swatch-height tile with a normalized 8×4×4 Lab
histogram, uniform 8-neighbour LBP histogram, and dark-pixel fraction. It marks
near-white tiles blank, standardizes the remaining descriptors, selects k from
2–10 using silhouette score, and decodes the cluster emissions with a
self-transition penalty. Each decoded cluster is assigned the closest legend
swatch in the standardized descriptor space. Tile state changes are shifted to
the strongest local row-mean/ink transition. The runner gets source, tile
cluster, and decoded interval stage images.

## Runner results

- Synthetic: 39/39 windows completed; mean score 0.783, row accuracy 0.915,
  boundary F1 0.849, cluster ARI 0.899, and 1.33 s/window.
- Gold eval: all 13 windows completed. Only log06 has gold truth; its row
  accuracy and score were 0, boundary F1 0.429, cluster ARI 0.637, and depth
  MAE 0.57 ft. Mean run time was 1.97 s/window (all windows below the 60 s
  budget); no crashes or API calls.

The approach handles blank gaps well and reproduces several strongly distinct
patterns on synthetic columns. It confuses similar low-contrast monochrome
textures, especially in log06, where the matched interval boundaries still
landed reasonably but cluster-to-swatch names were wrong. The silhouette
criterion can also overfit synthetic descriptor groups and produce missed or
merged boundaries. A next step would be a multi-scale spatial texture
descriptor and stronger direct swatch calibration, while keeping Viterbi for
continuity.
