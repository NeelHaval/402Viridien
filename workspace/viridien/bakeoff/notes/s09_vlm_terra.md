# S9 Terra vision model per cluster

The strategy extracts the requested lithology-column window, proposes boundaries from smoothed row color and texture descriptors, and groups bands by k-means. It builds one representative crop plus numbered lithology swatches per cluster, calls Codex for each cluster, and makes a second overview call for a log-wide mapping. Results are cached by PNG SHA-256 under `out/_cache/s09_vlm_terra/`.

If no valid model mapping is available, the strategy assigns clusters by RGB, grayscale, and gradient-orientation histogram distance to legend swatches. The runner's synthetic check showed highly variable results: boundaries sometimes followed pattern changes inside a single interval, and histogram matching often selected the wrong lithology. The local Codex CLI returned an access-denied initialization error in this environment, so model-derived answers could not be confirmed; `_api_calls` counts attempted invocations.

The visual output includes the source crop, proposed boundaries, cluster tinting, and numbered swatches. The primary remaining work is to improve boundary precision and confirm the configured vision model can start under the runner's account environment.
