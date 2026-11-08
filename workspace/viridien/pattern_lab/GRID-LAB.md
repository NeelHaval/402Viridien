# Terra grid pipeline

This pipeline proposes labels for human review. It does not accept annotations.
The generator continues to use approved, quality-accepted training sources.
Terra proposals do not enter the real test set or become human test truth.

Each batch contains crops from one document and that document's numbered
legend. Crops keep their aspect ratio. White padding fills the unused space.
The pipeline keeps native source boxes and source versions for each cell.
Images have a maximum edge of 1024 px. Large legends use separate panels.

The model returns a row-major matrix. Each cell contains `item_id`,
`legend_ids`, `status`, `confidence`, and a short visual reason. The batch
response also contains `batch_id` and the exact `input_hash` from its request.
IDs must come from that document's active legend. Padding cells must be null.
The validator rejects missing cells, wrong order, foreign IDs, duplicate IDs,
bad shapes, and stale input hashes. Missing or invalid batches enter a retry
list. Valid results are cached by image content, geometry, vocabulary, model,
and response contract. This avoids repeated work on unchanged inputs.

From the project root:

```powershell
python -X utf8 viridien_work/pattern_lab/grid_lab.py prepare --remaining-only --batch-size 8
```

For a new round, add `--out-dir PATH` to preparation, collection, and
publication. Use a new directory to keep the first benchmark unchanged.
The default path below refers to the completed first round.

Read `data/terra-grid-20261108/blind/instructions.txt` and send only the pending
blind run files and their images to GPT-6.1 Terra. Keep `private/` and live
human reviews out of the model context. The run files specify the matrix
shape and allowed IDs. Save responses as the stated result files. Then run:

```powershell
python -X utf8 viridien_work/pattern_lab/grid_lab.py collect
python -X utf8 viridien_work/pattern_lab/grid_publish.py
```

`pending-batches.json` lists uncached inputs. `retry-batches.json` lists missing
or invalid outputs. `terra-results.json` keeps validated matrices. The cache
contains validated batches. Publishing changes the suggestion pack only. It
checks the live task version and keeps saved or changed annotations. A backup
of the original bed-model pack supports repeatable publication. Model and
source provenance are stored when a human accepts a suggestion in the page.
After publication, reload the review page to load the new pack.

Use the default of 8 crops for new work. The renderer supports 4, 8, 16,
32, 64, and 128 crops. Keep large grids for already clean boxes. The separate
[larger-grid benchmark](MULTICROP-BENCH.md) tests 32, 64, and 128 crops with
Terra, Sol, Luna, and Jev. Smaller follow-up grids
are useful for boundaries, low-confidence answers, and model disagreements.
The first experiment used native agent sessions, with several panels in each
session. It did not measure independent API latency, token use, or cost.

## First blind test

The test used 60 unique crops from saved human reviews. Forty-eight came from
three documents with many reviews. Twelve added other layouts, including
six D-582 crops. This is a purposive sample, not a random accuracy estimate.
The three grid-size conditions ran in separate Terra sessions. They did not
read the human labels or each other's answers.

- Terra returned the same exact source legend IDs on 55 of 60 crops.
- Among 53 high-confidence clean matches, 52 agreed with the saved label.
- All 48 response batches passed matrix and legend-ID validation.
- The 12 crops shared by the 4-, 8-, and 16-crop conditions had 11 exact
  agreements in every condition. No label changed across those conditions.
- The smaller grids flagged more boundary issues on those same crops:
  2 flags at size 4, 1 at size 8, and 0 at size 16.

These are agreements with one saved human review, not established accuracy.
One disagreement is a clear mixed shale/marl box which the saved label calls
shale. Two D-582 shale crops were called silty sandstone. One shale/shale-grey
pair has similar symbols. Another D-582 crop remained uncertain. No disputed
human label was changed by this experiment.

The pending-work run made 70 proposals: 46 matches, 7 mixed crops,
10 uncertain crops, and 7 poor crops. Of the 46 matches, 41 were high
confidence and 5 were medium confidence. Only a high-confidence clean match
can use the quick-confirm action. A human must still check the source legend.
Set 70 contains six crops marked poor, mostly blank or frame lines. Do not
collect five blank examples to meet a count target.

Results and private scoring inputs are in `data/terra-grid-20261108/`.
`report.json` contains counts, disagreements, and the paired comparison.
`disagreements-contact.png` compares all five disagreements with both legends.
`publish.json` records suggestion publication and annotation-change counts.

The five-example target is a coverage goal. It is not a minimum required to
start source-based generation. Prefer clean motifs and different documents
over repeated boxes that only increase the count. Continue to keep real
test sources separate from generator sources.
