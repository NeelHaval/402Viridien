# it1_method_sweep

Ten naming methods on the same segments and legend.

- time: 2026-11-08T00:58:06
- legend source: detect; Section 2 segments: cnn_clustered; test pages: 2
- primary metric: oracle macro_f1 (page_f1 = macro-F1 on 60808400140000_hadrian 2_masterlog_2in_10000-21962); guard: oracle pixel accuracy
- metrics from eval_cells/evaluate_segment_classes_cell.py; background excluded from macro means

| method         |   oracle page_f1 |   pipeline page_f1 |   oracle macro_f1 |   pipeline macro_f1 |   oracle pixel_acc |   pipeline pixel_acc |   oracle segment_acc |   pipeline segment_acc |   oracle macro_f1 vs baseline |   oracle pixel_acc vs baseline |
|:---------------|-----------------:|-------------------:|------------------:|--------------------:|-------------------:|---------------------:|---------------------:|-----------------------:|------------------------------:|-------------------------------:|
| fft_texture    |           0.5439 |             0.3678 |            0.3765 |              0.1515 |             0.9807 |               0.5332 |               0.9472 |                 0.412  |                        0.0275 |                        -0.001  |
| baseline       |           0.4887 |             0.4081 |            0.3491 |              0.2177 |             0.9817 |               0.6767 |               0.9531 |                 0.4809 |                        0      |                         0      |
| colour_heavy   |           0.4887 |             0.33   |            0.3491 |              0.165  |             0.9817 |               0.533  |               0.9531 |                 0.4106 |                        0      |                         0      |
| column_adapted |           0.4887 |             0.2933 |            0.3491 |              0.1467 |             0.9817 |               0.533  |               0.9531 |                 0.4106 |                        0      |                         0      |
| no_split       |           0.4887 |             0.33   |            0.3491 |              0.176  |             0.9817 |               0.6268 |               0.9531 |                 0.4106 |                        0      |                         0      |
| window_vote    |           0.4884 |             0.3363 |            0.3489 |              0.1681 |             0.9811 |               0.5454 |               0.9516 |                 0.4589 |                       -0.0002 |                        -0.0006 |
| scale_matched  |           0.4872 |             0.2933 |            0.348  |              0.1553 |             0.9789 |               0.533  |               0.9413 |                 0.4106 |                       -0.0011 |                        -0.0028 |
| colour_hist    |           0.4968 |             0.3025 |            0.344  |              0.1602 |             0.9751 |               0.5455 |               0.9238 |                 0.4589 |                       -0.0051 |                        -0.0066 |
| texture_only   |           0.2369 |             0.1767 |            0.1896 |              0.0972 |             0.6985 |               0.2934 |               0.5073 |                 0.0924 |                       -0.1595 |                        -0.2832 |
| colour_only    |           0.2492 |             0.2172 |            0.1812 |              0.1448 |             0.771  |               0.4726 |               0.4267 |                 0.3021 |                       -0.1678 |                        -0.2107 |
