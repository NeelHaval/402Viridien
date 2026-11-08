# S08 JEV

The strategy extracts the requested column in strips via the harness, uses S2's segmenter when available and otherwise detects changes in local colour distributions and horizontal ink lines. Segment histograms are matched to lithology swatches; populated groups are presented in batches of up to eight questions with representative crops and a numbered swatch mosaic, subject to the time budget. When JEV is unavailable, the local swatch match is retained. A detected broken HTTPS proxy is suppressed for the remainder of the process after the configured retries.

Images sent to JEV are fitted and padded to the required 512–1024 pixel dimensions. The API key is read at runtime from the environment or the specified .env file and is never stored in output. Requests are counted, including retry attempts.

In this environment the configured HTTPS proxy refused connections, so the JEV result and the relative value of a swatch mosaic could not be evaluated. The fallback's colour histograms can confuse similar monochrome patterns, and its boundaries are approximate when S2 is absent. Adding orientation and texture features and comparing the crop-only and mosaic prompts with an accessible API are promising next steps.
