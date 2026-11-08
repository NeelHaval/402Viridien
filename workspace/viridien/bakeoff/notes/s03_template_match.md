# S3 template matching

Uses each lithology swatch as a horizontally repeated pattern, scales it to the lithology-column width, and compares a small scale bank against the column with grayscale normalized cross-correlation. A color-mean term breaks ties; tile winners are merged, and near-white blank tiles are omitted. The output cluster ID is the legend entry ID. Stage images show the sampled column, correlation map, and assigned intervals.

Synthetic tuning run: mean row accuracy 0.294, boundary F1 0.120, ARI 0.299; mean strategy time 3.84 s per 4,000-row column. Strongest results were on distinctive, high-contrast patterns. Weaknesses include low accuracy and imprecise boundaries when swatches are sparse, similar, or contain symbols; the log06 symbol legend is an especially clear failure. Scale variation, local score-based boundary refinement, and explicit separation of lithology textures from symbols remain possible improvements.
