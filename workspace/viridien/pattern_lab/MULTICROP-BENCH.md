# Larger grids and adaptive cluster test

The test files are in `data/multicrop-bench-20261108/`. No live human label
was changed. The review page keeps its existing human confirmation step.

The sample contains 128 unique saved review crops from 14 documents. It is
a selected development sample, with most crops from six documents, six D-582
crops, a boundary-risk crop, and another layout. It is not a random sample
or an independent final test. Only one primary source legend ID is scored.
Mixed patterns can be flagged, but there is no full boundary ground truth.

All four models received the same native crop pixels and document-specific
legends. The grids used 32, 64, or 128 crops on a maximum 1024 px image.
Higher counts make each motif smaller. Each native model/size condition used
a separate blind agent session. Human labels and other model outputs were
hidden. Jev used its image API and named Choice questions for the same cells.
The returned Jev model ID was `jev-1.14.0`.

| Model | 32 per grid | 64 per grid | 128 per grid |
|---|---:|---:|---:|
| GPT-6.1 Terra | 121/128 | 122/128 | 119/128 |
| GPT-6.1 Sol | 121/128 | 123/128 | 125/128 |
| GPT-6 Luna | 93/128 | 78/128 | 113/128 |
| Jev 1.14 | 116/128 | 110/128 | 78/128 |

These are exact agreements with saved human labels, not verified accuracy.
One sample, P037, is `57a129aee41f9852b4ff4ea7`. The earlier independent Opus
audit found its CLAYSTONE label wrong and its pixels identical to a LIMESTONE
crop. We kept the original reference unchanged for the table. The cluster
pass exposed the same conflict. Similar shale/shale-grey choices also need
manual checks. No model output was used to correct its own reference labels.

## Timing limits

These are one-run measurements under concurrent work. Native timing starts
at the first tool and ends at matrix validation. It includes reading panels,
tool overhead, reasoning, and writing the output. It excludes model startup
before the first tool. Jev timing covers asset encoding, sequential direct
HTTP calls, conversion, and validation. These are useful workflow timings,
not a fair comparison of raw model API latency. Native token usage and cost
were not captured. Jev usage is saved in the JSON results.

| Model | 32 per grid, seconds | 64 per grid, seconds | 128 per grid, seconds |
|---|---:|---:|---:|
| Terra | not measured | 22.09 | 16.60 |
| Sol | 44.50 | 81.23 | 74.22 |
| Luna | 57.43 | 97.67 | 85.16 |
| Jev | 4.16 | 5.19 | 2.94 |

Terra-32 lost its timer when the first Python command failed in Git Bash.
Its labels still passed parent validation. No late timer was invented.
Luna-64 needed matrix nesting and source-ID repair; its time includes that
repair. Final matrices passed validation. The validator rejects foreign IDs
per document, stale hashes, wrong order, missing cells, and wrong shapes.

## Cluster-first and larger-detail pass

`adaptive_cluster_bench.py` reuses the existing pixel descriptor. It uses no
human crop labels for grouping or routing. It first groups by document and
broad background colour. Within each group, a complete-link distance bound
of 0.30 prevents a similarity chain from joining different motifs. The
medoid becomes the representative. Different documents never share an
inferred label solely because their pixels look similar.

The 128 crops formed 51 groups in 0.68 seconds, including descriptor work
and representative-grid preparation. Representative results were projected
to members for this experiment only:

| Representative model | Model queries | Projected agreement | Observed workflow seconds |
|---|---:|---:|---:|
| Jev | 51 | 115/128 | 2.66 |
| Terra | 51 | 121/128 | 21.32 |
| Sol | 51 | 121/128 | 57.01 |

The adaptive variant screened with Jev, then sent 71 crops to a larger-detail
Sol pass. Routes used low model confidence/probability, quality flags,
similar legend descriptors, within-group variation, and possible internal
texture boundaries. Thresholds were fixed before reading human labels.

Each detail panel contains four query crops and up to two unlabelled visual
neighbours per query. The neighbours can have different labels. The full
source legend stays available. Native pixels are enlarged without invented
detail. The model labels the query only. This tests both increased display
detail and extra comparison examples; it does not isolate their effects.

The detail pass rescued six labels and introduced no new disagreements
relative to the initial cluster screen. Final agreement was 121/128, with
nine review flags. It did not beat flat Sol at 125/128. Recorded stage times
sum to about 84.8 seconds: 0.68 preparation, 2.66 Jev, 2.51 routing/rendering,
and 78.92 Sol. This is a sum of separately timed stages, not one continuous
end-to-end timing. Routing time came from a deterministic repeat; input
hashes were checked again before scoring.

One group had conflicting saved human labels. It contains P037, the known
CLAYSTONE/LIMESTONE reference error. This is not proof of a visual cluster
failure. Representative classification can still be wrong and spread an
error through an otherwise consistent group.

## Working choice

- Terra at 64 crops is a useful fast pre-label pass on clean boxes.
- Sol at 128 had the best observed label agreement in this sample.
- Jev at 32 is useful for quick screening; avoid its 128-crop setting here.
- Luna was inconsistent and often confident on wrong labels in this test.
- Keep clusters for ordering, caching, and selecting comparison examples.
  Do not copy a cluster label into approved training or test data without
  review. Keep larger-detail small grids for boundaries and look-alikes.
- The adaptive variant remains experimental. It was not made the live
  default, because the measured quality and speed did not improve together.

## Files and repeat checks

`multicrop_bench.py prepare` creates blind packets and a private answer key.
The supplied native runs use `begin --run NAME` and `finalize --run NAME`.
`multicrop_bench.py score` checks the matrices against the frozen reference.
`jev_grid_bench.py` runs the direct API benchmark with the configured project
credential. It stores no credential in result files. `adaptive_cluster_bench.py`
provides `prepare`, `route`, and `score` stages. Its stage-1 Jev request uses
the same adapter with `--packet` and `--output`.

The blind packets, raw matrices, input hashes, timings, routing reasons,
API usage, and disagreements remain available. `comparison.html` provides
the interactive comparison. Keep these files separate from approved labels.

The adapter follows the [TypeSafe image documentation](https://docs.typesafe.ai/images):
images are sent once in state, the image header is enabled, and each crop
gets structured label and quality questions. The image interface is marked
experimental by TypeSafe. Its limits and behaviour can change.
