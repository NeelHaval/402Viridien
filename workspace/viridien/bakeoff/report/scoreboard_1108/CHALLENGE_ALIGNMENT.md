# Viridien challenge alignment and export handoff

This is preparation for a later pipeline PR. The challenge materials supplied in the project do not state a numerical judging formula, and these files do not establish an official winner or grade.

## Challenge sources

- [Viridien presentation](../../../../viridien_challenge/Viridien_HTB_challenge/Viridien_HackTheBurgh_Presentation.pptx), slide 10, asks teams to identify lithological intervals and boundaries, group similar visual patterns, determine top and base depths, and match groups to the legend. Locating the three boxes without supplied bounding boxes is a bonus.
- The same presentation, slide 13, says **“For each document”** provide structured CSV results and an annotated image. It names the lower depth `base_depth` and lists cluster IDs, pixel boundaries and lithology. The annotated image must show boundaries, depth estimates and cluster/lithology labels.
- Slide 14 states **“Solutions will be judged on correctness, and efficiency.”** It asks the demo to explain the approach, results and limitations, runtime and resource requirements, and notes generalisation, simplicity and automation. The shorter [challenge deck](../../../../viridien_challenge/Viridien_HTB_challenge/hack27/HTB_Challenge_Slides.pptx), slides 4, 7 and 8, repeats these points.
- The [provided notebook](../../../../viridien_challenge/Viridien_HTB_challenge/hack27/project/example_notebook.ipynb), cells 9–10, instead uses `bottom_depth` and reads `cluster_id` as optional. Its [example CSV](../../../../viridien_challenge/Viridien_HTB_challenge/hack27/project/example/example.csv) also uses `bottom_depth`. The notebook uses full-image pixel coordinates. The slide calls for a cluster ID, so the adapter requires one.

Ask the organisers which lower-depth header and submission schema they will accept. The two formats in this handoff make the published discrepancy visible; neither is represented as the confirmed final submission format.

The local row accuracy ignores ground-truth blank rows. Its product with boundary F1 does not directly include depth error, cluster ARI, runtime or memory; those values are reported separately. Results on selected windows do not establish full-document readiness. The [readiness preview](readiness_preview.json) audits only the saved checkpoint and recorded annotation geometry; it is incomplete and can be refreshed with `python -X utf8 report/scoreboard_1108/scan_readiness.py` from `bakeoff/`.

## Official requirements and local measurement

| Topic | Challenge materials | Local implementation or policy |
|---|---|---|
| Overall grade | Correctness and efficiency; no weights, formula or thresholds published in the provided decks or notebook. | The harness score is `row_acc × boundary_f1`; it is a local proxy, not a verified challenge grade. Report row accuracy, boundaries, clusters, depth, time and resources separately too. |
| Lithology names | Match intervals to the supplied legend. No Clay/Shale equivalence is specified. | The owner requests symmetric full class credit among Clay, Shale, Clay/Shale and Clay-Shale. That is a local grading rule, not a verified Viridien rule. Preserve raw captions in exports. |
| Depth field | Slides: `base_depth`. Notebook and example: `bottom_depth`. | Two explicit CSVs are produced; neither silently adds the other alias. Confirm the submission format with the organisers. |
| Clusters | Slide expects IDs for visually similar patterns. | Export keeps the existing IDs unchanged; schema validation alone does not establish that clusters are correct. |
| Image | Annotated image with boundaries, depth and labels for each document. | Existing log06 comparison images are review artifacts. Check a final per-document submission image against the slide requirements. |
| Generalisation and resources | Unseen documents, automatic predictions, runtime, hardware and memory matter. | One saved evaluation window and its timings are evidence for that run only; full-document and held-out readiness need separate checks. |

## Export adapter and real example

From `bakeoff/`, run `python -X utf8 -m strategies.challenge_export SOURCE.csv --out-dir DESTINATION --image-height HEIGHT`. The image height is optional; provide it to check the upper pixel bound. The adapter checks required fields, integer nonnegative and sorted nonoverlapping full-image intervals, nonempty labels, integer cluster IDs, and finite boundary depths. The local harness spec requires integer cluster IDs; the slide example uses letters, so verify the desired output ID convention with the organisers. The adapter preserves raw label, cluster, pixel and depth strings. It does not perform inference, change grades or claim that its input labels are correct.

The completed S21 log06 evaluation export is in [export_example](export_example/):

- [Notebook-format CSV](export_example/6b663182fbfb4f87.notebook.csv): `top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id`.
- [Slide-format CSV](export_example/6b663182fbfb4f87.slide.csv): `top_y,bottom_y,lithology,cluster_id,top_depth,base_depth`.
- [Manifest](export_example/6b663182fbfb4f87.manifest.json): source path and SHA-256, output hashes, schemas, row count and the 12,050-pixel image-height check.

The source is the saved S21 prediction at [out/scoreboard_1108/s21_v4dino/6b663182fbfb4f87.csv](../../out/scoreboard_1108/s21_v4dino/6b663182fbfb4f87.csv). All 16 rows round-trip to both exports with the hashes verified. This log06 evaluation window spans the supplied interpreted column for this document; it is not evidence that every document has been processed or that the result meets the challenge quality standard.

## Legend coverage for new documents

The frozen v4 adapters use the saved native swatch cache. The final validated DEV suite has no native entries on all 51 cases / 46 logs for v4, baseline DINO and S23. Their grades on this suite are pipeline coverage failures. These results do not isolate the DINO matcher. The complete counts are in the final cache-status table.

The next production PR must supply legend swatches for a new document before matching its beds. Use the production legend parser, retain native swatch pixels and printed captions, and validate the input. Record a clear failure when no usable legend is found. A new legend input path changes inference and needs its own same-contract evaluation. The Clay/Shale grading helper must not grant credit for Unknown or conceal a missing legend. The current comparison keeps the frozen adapters as they were; it does not change the live labeltool parser.

## Full-scan readiness checklist

- [x] One saved real log06 CSV has finite depths, valid ordered intervals, labels and cluster IDs; both explicit exports round-trip to it. These checks cover format and coordinates, not geological correctness.
- [ ] Produce and validate an interval CSV for **each** required document, with full-image coordinates and the confirmed lower-depth field. Test full-height input handling, bounds, gaps and failure reporting for every document.
- [ ] Generate and inspect a submission image for each document with visible boundaries, boundary depth estimates and cluster/lithology labels. The [log06 review](log06_dino_comparison.html) is a comparison artifact, not a verified submission package.
- [ ] Confirm legend matching, visually meaningful cluster IDs, boundary placement and depth values against appropriate development evidence; report uncertainty and failures.
- [ ] Measure full-run time, hardware, peak memory and model-loading conditions; report generalisation and limitations. Ensure final predictions require no manual correction.
- [ ] Obtain organiser clarification on the lower-depth header and any official grading or synonym rules. Keep strict and owner-equivalence local results clearly labelled until then.

Adapter checks: `python -X utf8 -m strategies.test_challenge_export` from `bakeoff/`. The tests cover both schema round-trips and reject missing/nonfinite depth, malformed coordinates, overlap, missing or noninteger cluster IDs and out-of-image intervals.


## Final evaluated coverage

The final comparison contains 272 records: 68 cases for each of four candidates. Both strict and owner-equivalent saved-CSV replay passed, with 271 successful primary predictions, no pixel-only diagnostics and zero audit errors. One formal S19 Shenzi whole-attempt timeout remains a zero-score failure. Its inference seconds are unknown; the 600.705492-second whole-attempt receipt is not an inference timing.

The final [readiness diagnostic](readiness_preview.json) reads merged [scores.json](scores.json). It reports complete record coverage and keeps submission readiness unverified. Each candidate covers 61 distinct source images in this matrix; 44 images have at least one full interpreted-lithology-box record. S19 has 67 nonempty valid CSV records and one empty unverified record; each v4/DINO candidate has 68 nonempty valid CSV records. Unknown-filled output can pass CSV checks while failing label coverage. These counts do not establish complete-log correctness or a submission package.

S19 is the recorded development accuracy/coverage default. This matrix does not certify production readiness or the owner's later 1–2 minute evaluation goal. The separate 120-second speed gate and live cache pilot are not part of these grades. External test and verify remain closed.
