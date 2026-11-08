# Failures and denominators (read with REPORT.md)

Every attempted page stays in failure_rate. Other averages skip missing values; their page counts are in averaged_metric_denominators. Depth metrics are unavailable (not zero) on pages whose depth scale could not be read.

## clean run: 520 cases

| Family | Strategy | Attempted | Exec. failures (class) | Completed | Empty | Scorable | Depth available | Averages taken over (assigned / short-bed / overseg.) |
|---|---|---|---|---|---|---|---|---|
| All | s11_fft_legend | 130 | 32 (depth_calibration_unavailable 32) | 98 | 25 | 73 | 98 (else unavailable) | 130 / 79 / 69 |
| All | s17_synth_gulf | 130 | 32 (depth_calibration_unavailable 32) | 98 | 22 | 76 | 98 (else unavailable) | 130 / 79 / 69 |
| All | s19_combine | 130 | 32 (depth_calibration_unavailable 32) | 98 | 17 | 81 | 98 (else unavailable) | 130 / 81 / 77 |
| All | s22_boundary_v2 | 130 | 32 (depth_calibration_unavailable 32) | 98 | 22 | 76 | 98 (else unavailable) | 130 / 79 / 75 |
| D-582 1-bit grid | s11_fft_legend | 12 | 1 (depth_calibration_unavailable 1) | 11 | 5 | 6 | 11 (else unavailable) | 12 / 8 / 8 |
| D-582 1-bit grid | s17_synth_gulf | 12 | 1 (depth_calibration_unavailable 1) | 11 | 4 | 7 | 11 (else unavailable) | 12 / 8 / 4 |
| D-582 1-bit grid | s19_combine | 12 | 1 (depth_calibration_unavailable 1) | 11 | 2 | 9 | 11 (else unavailable) | 12 / 9 / 9 |
| D-582 1-bit grid | s22_boundary_v2 | 12 | 1 (depth_calibration_unavailable 1) | 11 | 4 | 7 | 11 (else unavailable) | 12 / 8 / 7 |
| Geoservices grid | s11_fft_legend | 8 | 0 (-) | 8 | 0 | 8 | 8 (else unavailable) | 8 / 8 / 6 |
| Geoservices grid | s17_synth_gulf | 8 | 0 (-) | 8 | 0 | 8 | 8 (else unavailable) | 8 / 8 / 7 |
| Geoservices grid | s19_combine | 8 | 0 (-) | 8 | 0 | 8 | 8 (else unavailable) | 8 / 8 / 7 |
| Geoservices grid | s22_boundary_v2 | 8 | 0 (-) | 8 | 0 | 8 | 8 (else unavailable) | 8 / 8 / 8 |
| Gulf 3-panel | s11_fft_legend | 60 | 22 (depth_calibration_unavailable 22) | 38 | 10 | 28 | 38 (else unavailable) | 60 / 30 / 28 |
| Gulf 3-panel | s17_synth_gulf | 60 | 22 (depth_calibration_unavailable 22) | 38 | 8 | 30 | 38 (else unavailable) | 60 / 30 / 30 |
| Gulf 3-panel | s19_combine | 60 | 22 (depth_calibration_unavailable 22) | 38 | 8 | 30 | 38 (else unavailable) | 60 / 30 / 30 |
| Gulf 3-panel | s22_boundary_v2 | 60 | 22 (depth_calibration_unavailable 22) | 38 | 8 | 30 | 38 (else unavailable) | 60 / 30 / 30 |
| MC761 mineral/lithology | s11_fft_legend | 7 | 0 (-) | 7 | 7 | 0 | 7 (else unavailable) | 7 / 0 / 0 |
| MC761 mineral/lithology | s17_synth_gulf | 7 | 0 (-) | 7 | 7 | 0 | 7 (else unavailable) | 7 / 0 / 0 |
| MC761 mineral/lithology | s19_combine | 7 | 0 (-) | 7 | 7 | 0 | 7 (else unavailable) | 7 / 0 / 0 |
| MC761 mineral/lithology | s22_boundary_v2 | 7 | 0 (-) | 7 | 7 | 0 | 7 (else unavailable) | 7 / 0 / 0 |
| MC765 colour grid | s11_fft_legend | 25 | 4 (depth_calibration_unavailable 4) | 21 | 0 | 21 | 21 (else unavailable) | 25 / 21 / 18 |
| MC765 colour grid | s17_synth_gulf | 25 | 4 (depth_calibration_unavailable 4) | 21 | 0 | 21 | 21 (else unavailable) | 25 / 21 / 20 |
| MC765 colour grid | s19_combine | 25 | 4 (depth_calibration_unavailable 4) | 21 | 0 | 21 | 21 (else unavailable) | 25 / 21 / 21 |
| MC765 colour grid | s22_boundary_v2 | 25 | 4 (depth_calibration_unavailable 4) | 21 | 0 | 21 | 21 (else unavailable) | 25 / 21 / 20 |
| One-off layout | s11_fft_legend | 5 | 1 (depth_calibration_unavailable 1) | 4 | 3 | 1 | 4 (else unavailable) | 5 / 3 / 1 |
| One-off layout | s17_synth_gulf | 5 | 1 (depth_calibration_unavailable 1) | 4 | 3 | 1 | 4 (else unavailable) | 5 / 3 / 1 |
| One-off layout | s19_combine | 5 | 1 (depth_calibration_unavailable 1) | 4 | 0 | 4 | 4 (else unavailable) | 5 / 4 / 3 |
| One-off layout | s22_boundary_v2 | 5 | 1 (depth_calibration_unavailable 1) | 4 | 3 | 1 | 4 (else unavailable) | 5 / 3 / 1 |
| Texas 'Rock Types' box | s11_fft_legend | 13 | 4 (depth_calibration_unavailable 4) | 9 | 0 | 9 | 9 (else unavailable) | 13 / 9 / 8 |
| Texas 'Rock Types' box | s17_synth_gulf | 13 | 4 (depth_calibration_unavailable 4) | 9 | 0 | 9 | 9 (else unavailable) | 13 / 9 / 7 |
| Texas 'Rock Types' box | s19_combine | 13 | 4 (depth_calibration_unavailable 4) | 9 | 0 | 9 | 9 (else unavailable) | 13 / 9 / 7 |
| Texas 'Rock Types' box | s22_boundary_v2 | 13 | 4 (depth_calibration_unavailable 4) | 9 | 0 | 9 | 9 (else unavailable) | 13 / 9 / 9 |

Recorded errors:

- 25 x s11_fft_legend|ValueError: Depth calibration extrapolates above zero at column top
- 25 x s17_synth_gulf|ValueError: Depth calibration extrapolates above zero at column top
- 25 x s19_combine|ValueError: Depth calibration extrapolates above zero at column top
- 25 x s22_boundary_v2|ValueError: Depth calibration extrapolates above zero at column top
- 6 x s11_fft_legend|ValueError: No consistent increasing depth calibration
- 6 x s17_synth_gulf|ValueError: No consistent increasing depth calibration
- 6 x s19_combine|ValueError: No consistent increasing depth calibration
- 6 x s22_boundary_v2|ValueError: No consistent increasing depth calibration
- 1 x s11_fft_legend|ValueError: Fewer than two numeric depth anchors found
- 1 x s17_synth_gulf|ValueError: Fewer than two numeric depth anchors found
- 1 x s19_combine|ValueError: Fewer than two numeric depth anchors found
- 1 x s22_boundary_v2|ValueError: Fewer than two numeric depth anchors found
