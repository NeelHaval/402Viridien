# S2 hand-made features + PELT

For each downsampled row, the strategy pools Lab colour, dark-ink fraction, and horizontal/vertical Sobel stroke activity across the lithology column. It smooths these profiles over a short vertical neighborhood, standardizes them, and uses L2 PELT to propose interval boundaries. Segment descriptors add within-interval variation and a simple vertical ink autocorrelation period. Agglomerative clustering assigns recurring visual regimes; each segment is named by its nearest lithology swatch descriptor. Nearly blank white segments are omitted.

`segment(log, window, ctx)` is the reusable boundary function. The runner output includes the cropped column, row-feature map, PELT boundary view, and cluster view.

Synthetic tuning found acceptable boundary recovery, but lithology naming remains sensitive to short segments and visually similar swatches. The method is CPU-only and took roughly 1–3 seconds per 4000-row synthetic window after reducing PELT's candidate points with `jump=4`. It does not yet separate close patterns reliably on real scans; inspect the synthetic/eval PNGs for errors before relying on labels.
