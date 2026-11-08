# Jev image pipeline: coverage and matched tests

Measured on 8 November 2026. All model outputs remain unapproved. No live reviews were changed.

## Coverage

- The dataset catalog has 51 documents. The saved audit has 350 new accepted boxes from 44 documents and 70 visual pattern sets.
- The 678 supplied boxes come from Hadrian 2 and contain four labels. They were excluded from the large-grid benchmark.
- The old 128-crop test covered 14 documents, 51 of the 70 reviewed sets, and only three of the seven observed layout families. Six documents supplied 120 crops (93.75%).
- The new test has 80 crops from 23 documents and all seven families: 48 truth-bed interiors, 12 known boundaries, and 20 accepted crops from 20 other documents.
- The three truth logs are Neel 1-inch, Shenzi, and D-582 log06. The first two have reviewed truth; the last has gold truth. Hadrian was excluded from this truth test because its supplied labels have known quality issues.
- Truth-bed sampling gives labels similar weight. It does not weight long layers by pixel area. The 20 saved reviews are a separate agreement test, not independent gold.
- The 80 test boxes have no spatial overlap with the earlier 128-crop or 60-crop packs. Inputs and answer keys were hashed before API design tests.
- Only 32 old crops were used to choose the input format. The three truth logs were already known to local model work. This is a new-window test, not a claim of generalisation to unseen wells. Shenzi truth may also retain model boundary anchoring.

## Matched crop results

Exact results require one correct current source-legend ID. A two-label answer does not receive exact credit. Unknown or unmapped outputs also receive no exact credit.

| Method | All 68 single-pattern crops | Truth interiors (48) | New reviewed crops (20) | Two-label answers on all 68 |
|---|---:|---:|---:|---:|
| Jev, new pipeline | 65/68 (95.6%) | 45/48 | 20/20 | 0 |
| Jev, old grid | 55/68 (80.9%) | 36/48 | 19/20 | 0 |
| Cached v4 | 55/68 (80.9%) | 39/48 | 16/20 | 5 |
| DINOv2 prototype | 49/68 (72.1%) | 36/48 | 13/20 | 10 |
| DINO + hand prototype | 52/68 (76.5%) | 36/48 | 16/20 | 9 |

- v4 has five two-label answers; all five contain the truth. Fusion has nine; all nine contain the truth. Count these as unresolved, rather than wrong visual matches. With this looser diagnostic, v4 contains the truth in 60/68 and fusion in 61/68. This is not exact classification accuracy.
- The new Jev format improves exact agreement by 10/68 over its old grid on these same pixels. This is a change to several parts at once. It does not isolate which part caused the gain.
- The old grid hit the normal context limit for its last batch and used the documented long-context retry. The first two calls were reused during recovery. Its combined wall time is not a fresh latency measurement.
- The new Jev run returned all 80 rows in 5.37 s, with 35 fresh requests, four concurrent requests, no failed calls, and model jev-1.14.0. This includes card creation, requests and parsing. It excludes TIFF crop extraction and local candidate ranking. It is one run, not a latency guarantee.
- DINO baseline timing: 20.12 s to load the model, then 10.43 s for hand features and DINO inference on 240 crops (new, development, and old). It used CUDA. These scopes differ from Jev, so they are not a direct speed ratio.
- Crop classification does not measure layer detection, row accuracy, or boundary F1. This result does not establish a higher end-to-end challenge score.

## Errors and review checks

- Jev has three exact-ID errors. Two are D-582 Shale versus Clay. The source legend draws these symbols alike. The third is Neel Sandstone versus Siltstone. All three went to review.
- Jev marks eight boundary crops as mixed and four as bad crops. All 12 go to review. It flags 12 of the 48 truth interiors too: 11 are uncertain D-582 symbols and one is another quality flag.
- At the thresholds fixed before the test, only 13/68 single-pattern crops qualify as strong suggestions. All 13 match the saved truth. This small sample does not justify automatic acceptance.
- Candidate recall is 68/68 on the test and 30/32 on development. A shortlist can omit the true label. Keep the full source legend available and use the fit check to send weak cases to a person.
- D-582 remains the hard layout. New Jev gives 14/16 exact labels, cached v4 8/16, and fusion 4/16 plus five two-label answers. Source hatch size differs from the column hatch size. The test is class-balanced and uses 72-pixel windows; earlier DINO reports used a different, often shale-heavy window sample. These results are not interchangeable.

## Implemented Jev path

1. Keep native source pixels, source coordinates, source document, legend IDs, and hashes.
2. Group by document. Rank active legend symbols by local hatch features and background colour. Keep the top four hatch options plus the top two colour options.
3. Send at most four queries per document request. Use source swatch images directly in Choice criteria. Upload identical images once with multipart.
4. In the same request, ask whether any candidate fits and whether the crop is mixed, unclear, or unusable. Use the original crop for the quality question.
5. Send a result to review unless fit is at least 0.9, top probability at least 0.8, confidence at least 0.9, and quality is single-pattern. These are uncalibrated review rules, not probabilities of correctness.
6. Keep weak candidates, twins, boundary changes and missing legend options in the human queue. Keep all generated results separate from human labels and synthetic-data approval.

The previous clustering experiment remains separate. This new test does not measure propagation of a cluster representative to all members. Do not use its 65/68 result to approve an entire cluster.

## Development tests

| Format | Exact agreement on 32 old crops | Measured run time |
|---|---:|---:|
| raw | 25/32 | 3.93 s |
| tile | 24/32 | 9.26 s |
| criteria | 25/32 | 2.28 s |
| grid | 24/32 | 3.43 s |

The criteria format tied raw full-legend cards and used less time. Native-pixel tiling did not help Jev. Settings were frozen before the new test. These are single-run observations.

## Official guidance used

- [Images](https://docs.typesafe.ai/images): images can define Choice options; repeated references expand image context; state reuse and multipart reduce duplicate transfer. The API has size limits and an image feature header.
- [Companion Noul](https://docs.typesafe.ai/patterns/companion-noul): a Choice always picks an option. Pair it with a fit check.
- [Parallel questions](https://docs.typesafe.ai/patterns/fan-out): ask all needed questions in one request, then choose the route in code.
- [Models and context](https://docs.typesafe.ai/models): pin the model version; avoid long context when normal context fits. The docs warn that long context is slower and less accurate.
- [Confidence cookbook](https://docs.typesafe.ai/cookbooks/classification_using_confidence): check confidence on data from the target task. The cookbook does not calibrate our geology inputs.

## Reproduce

Run from the project root with Python UTF-8 enabled. The prepared pack is immutable. The configured project Jev credential is used without logging it.

```text
python -X utf8 viridien_work/pattern_lab/jev_optimize.py rank
python -X utf8 viridien_work/pattern_lab/jev_optimize.py dev --variant criteria
python -X utf8 viridien_work/pattern_lab/jev_optimize.py test
python -X utf8 viridien_work/pattern_lab/jev_optimize.py report
python -X utf8 viridien_work/pattern_lab/jev_report.py
```

Repeated API jobs use content-and-prompt caches. A cached rerun cannot establish fresh API latency. Do not refit the frozen pipeline using this test and then call the same cases a new holdout.
