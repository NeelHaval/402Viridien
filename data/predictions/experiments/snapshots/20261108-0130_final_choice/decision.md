# Classification method choice

- time: 2026-11-08T01:30:51
- chosen: **baseline**, the Section 4 naming as it stands (split mixed segments, then nearest swatch by colour + ink + run-length texture)
- evidence: snapshots/20261108-0058_it1_method_sweep (iteration 1, ten methods); iteration 2 (split sweep) was interrupted and is not compared
- why: tied best on naming alone (oracle: 98.2% of rows, 95.3% of layers) and clearly best on what ships
  (pipeline: macro-F1 0.408 on the hadrian page vs 0.368 next, 67.7% of rows vs 62.7%); it is the only
  method that splits segments, which recovers boundaries Section 2 misses
- caveats: tuned and scored on the same two test pages; pipeline numbers use the Section 2 features in the
  notebook at the time (r, g, b, ink, htr, vtr); the mudlog page's labels ("unknown", "13") cannot be scored
- to restore the experiments: paste the cells from section_4_6_experiment_cells.py back into 4.6

|                                        |   pipeline page_f1 |   pipeline pixel_acc |   oracle pixel_acc |   oracle segment_acc |   oracle page_f1 |   pipeline macro_f1 |   oracle macro_f1 |   pipeline segment_acc |
|:---------------------------------------|-------------------:|---------------------:|-------------------:|---------------------:|-----------------:|--------------------:|------------------:|-----------------------:|
| ('it1_method_sweep', 'baseline')       |              0.408 |                0.677 |              0.982 |                0.953 |            0.489 |               0.218 |             0.349 |                  0.481 |
| ('it1_method_sweep', 'fft_texture')    |              0.368 |                0.533 |              0.981 |                0.947 |            0.544 |               0.151 |             0.377 |                  0.412 |
| ('it1_method_sweep', 'window_vote')    |              0.336 |                0.545 |              0.981 |                0.952 |            0.488 |               0.168 |             0.349 |                  0.459 |
| ('it1_method_sweep', 'colour_heavy')   |              0.33  |                0.533 |              0.982 |                0.953 |            0.489 |               0.165 |             0.349 |                  0.411 |
| ('it1_method_sweep', 'no_split')       |              0.33  |                0.627 |              0.982 |                0.953 |            0.489 |               0.176 |             0.349 |                  0.411 |
| ('it1_method_sweep', 'colour_hist')    |              0.303 |                0.546 |              0.975 |                0.924 |            0.497 |               0.16  |             0.344 |                  0.459 |
| ('it1_method_sweep', 'scale_matched')  |              0.293 |                0.533 |              0.979 |                0.941 |            0.487 |               0.155 |             0.348 |                  0.411 |
| ('it1_method_sweep', 'column_adapted') |              0.293 |                0.533 |              0.982 |                0.953 |            0.489 |               0.147 |             0.349 |                  0.411 |
| ('it1_method_sweep', 'colour_only')    |              0.217 |                0.473 |              0.771 |                0.427 |            0.249 |               0.145 |             0.181 |                  0.302 |
| ('it1_method_sweep', 'texture_only')   |              0.177 |                0.293 |              0.698 |                0.507 |            0.237 |               0.097 |             0.19  |                  0.092 |
