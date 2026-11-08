# S1 colour runs

For each row, sample the central 70% of the lithology column and use the most common 4-bit-per-channel RGB bin after filtering white and near-black pixels. Match its Lab value to lithology legend swatches. Short runs are merged to suppress isolated pattern marks. When the legend has no chromatic swatches, use row ink density and horizontal/vertical Sobel edge energy to select a swatch. Cluster IDs follow the matched legend entry.

The pipeline figure shows the column crop, modal row colour, and final interval colouring. The implementation is CPU-only and performs no OCR or per-log tuning.
