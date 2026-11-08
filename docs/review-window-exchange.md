# Exchange reviewed windows

Use `scripts/export_review_windows.py` to export saved `pattern_lab` reviews.
The helper uses the existing manifest validator and CSV writer. Use the existing
`scripts/datasets.py pack` command to make the ZIP. No service is needed.

## Source format

The source labeller saves one document JSON. It has `id`, `image_name`, `width`,
`height`, `boxes`, `legend`, `segments`, and `revision`. Each legend entry has an
integer `id` and a `label_printed`. Each segment has an `id`, a page-coordinate
`box`, `legend_ids`, `status`, and `approved`. It can also have reviewer fields,
comments, flags, a suggestion, a confidence tier, and reasons.

These fields come from `pattern_lab/model.py`, its `Store.validate` method, and
the saved documents in `pattern_lab/data/genlab/label-windows/after`. The prepared
window crops are in `label-windows/terra/blind/<document-id>/native.png`.
`terra/windows.json` states their page boxes. Do not read calibration truth,
private answers, or sealed files to make an exchange.

The first five prepared windows are 1024 rows high:

| Well | Document ID | Original page box | Source split |
| --- | --- | --- | --- |
| Ardennes | `1bac8dda58a39eaf` | `[1167,246482,1230,247506]` | train |
| MC761 | `9da74bcde9aee789` | `[1290,8147,1369,9171]` | train |
| MC765 | `4a056c8ff25cf749` | `[1134,8348,1171,9372]` | train |
| Trail Mtn | `d2b5e2c76cfa887c` | `[335,13399,391,14423]` | train |
| Rio Lavaca | `28112eef4c3c645a` | `[661,44257,731,45281]` | train |

The source split audit states `train` for both source split rules. The existing
source importer can assign challenge cases to `dev`. Do not change a split to
make validation pass. Supply the agreed parent catalog and resolve any conflict
before a release. Keep the source split audit in the exposure history.

## Input file

Write an exchange JSON. Paths are relative to that file. Use the exact saved
document and crop SHA-256 values. Do not copy a full TIFF. The crop must have its
native width and height and must span the full lithology column.

```json
{
  "schema_version": 1,
  "name": "review-windows-v1",
  "windows": [{
    "id": "mc765-y8348-9372",
    "document_id": "4a056c8ff25cf749",
    "document": "after/4a056c8ff25cf749.json",
    "image": "terra/blind/4a056c8ff25cf749/native.png",
    "box": [1134, 8348, 1171, 9372],
    "document_sha256": "REPLACE_WITH_SAVED_DOCUMENT_SHA256",
    "window_image_sha256": "REPLACE_WITH_NATIVE_PNG_SHA256",
    "parent": {
      "id": "REPLACE_WITH_PARENT_CATALOG_RECORD_ID",
      "well": "MC765",
      "group": "well:mc765_p011",
      "split": "train",
      "image_sha256": null,
      "exposure_history": [{
        "kind": "visible_development",
        "source_normalized_split": "train",
        "source_generator_split": "train",
        "note": "Model suggestions were prepared and shown for review."
      }]
    }
  }]
}
```

Use `viridian402.data.sha256` to get the hashes. A missing or changed input hash
stops the export. A missing parent full-page image hash stays null. The helper
gets that hash from the parent catalog when `--catalog` is supplied. It does not
claim that a crop hash is a full-page hash.

For a local check, omit `--catalog`. The output will state
`catalog_validation: "not_run"`. For a release, supply the existing combined
train/dev catalog. The helper checks parent IDs, groups, splits, image hashes,
page names and sizes. It also validates the parent records and new windows
together with `load_manifest`. Test/verify catalogs are refused before any truth
file is opened. Use metadata from the full experiment catalog in the release
review to check other split assignments. Do not open sealed truth for this step.

## Human review gate

Suggestions do not become truth. An approved flag alone does not create truth.
After a human reviews the saved document, add this block to the window entry:

```json
"review": {
  "kind": "human",
  "reviewer": "REPLACE_WITH_REVIEWER_NAME",
  "reviewed_at": "2026-11-08T12:00:00Z",
  "document_sha256": "REPLACE_WITH_REVIEWED_DOCUMENT_SHA256"
}
```

This is a human review record. A model must not add it on the reviewer's behalf.
It applies only to that saved document hash. A new save needs a new review.
The helper exports only segments with `status: "approved"`, `approved: true`,
one valid legend ID, and a full-column box. Rejected segments, suggestions,
unknown regions, flags, explicit uncertainty, and lateral mixtures have no CSV
row. The helper does not choose a dominant rock for a mixture. Two accepted beds
that overlap cause an error. No depth is inferred from the page calibration.

Without usable reviewed beds, a record has `truth: null` and
`source: "external_unlabelled"`. With reviewed beds, it has `source: "human"`
and explicit partial truth coverage. Gaps have no truth. This agrees with the
existing manifest contract. The source legend's printed names are kept as saved.

## Coordinates, history, and ZIP contents

Output images are small RGB PNG windows. Their boxes and CSV rows use local
coordinates. Add the original box's left and top values to get page coordinates.
For example, MC765 local row 24 is page row `24 + 8348 = 8372`.
`original_coordinates` records this mapping and the original page dimensions.
All pixel boxes use exclusive right and bottom edges.

`provenance/<id>.json` keeps the unchanged source segment boxes, reviewer fields,
comments, reasons, tiers, flags, legend, revision, parent metadata, and source
hashes. It states why each bed did or did not enter truth. It includes only
segments that touch the window. It keeps the source depth calibration as evidence.

Each window keeps its parent well, group, and split. `lineage` keeps the parent ID
and image hash. Exposure history is retained and extended. Every output states
`fresh_holdout: false`. Parent catalog tags and review history are also kept in
provenance. Existing synthetic generation inherits the window group and split.
These windows and their descendants cannot become unseen verification cases.
Keep these history fields in later exports; split checks do not replace history.
The existing generator records the parent ID and hashes. It does not copy the
full exposure history to each child. Keep this export and its parent catalog
with any descendants so that their review history remains available.

`source-snapshot.json` contains the exchange hash, source hashes, catalog hash,
validation status, and limits. The existing pack command includes this file and
all files under `provenance/`, in addition to manifest assets. Failed validation
leaves no export directory. The helper refuses to overwrite an existing export.

## Export and prepare a private release

Run these commands from the repository. Use a new output directory each time.
First install this checkout with `python -X utf8 -m pip install -e . --no-deps`
if the package is not on the Python import path. Install the project dependencies
first if they are not already available.

```sh
python -X utf8 scripts/export_review_windows.py --windows data/local/windows.json --catalog data/local/parents/all.json --output data/local/review-windows-v1
python -X utf8 -m viridian402 validate --manifest data/local/review-windows-v1/all.json
python -X utf8 scripts/datasets.py pack data/local/review-windows-v1/all.json --output data/local/review-windows-v1.zip
```

The pack command prints the ZIP `sha256`, byte `size`, `asset_name`, and manifest
names. Keep that output as the release metadata. Keep the ZIP private and local
until its data review is complete. This helper does not upload or publish data.
Do not edit accepted labels or add a dataset lock entry with a made-up asset ID.
A later authorized release must record its real asset ID, hash, and size under
the existing dataset lock contract.

Run the focused checks after the last code change:

```sh
python -X utf8 -m pytest tests/test_export_review_windows.py tests/test_bundles.py -q
```

The tests use tiny fixture images. They check the review gate, ignored ambiguous
regions, coordinate offsets, source hashes, split guards, failed export cleanup,
and the existing ZIP round trip. They do not measure label accuracy.
