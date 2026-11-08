# Saved-output validation and export

This is the S19 owner's existing `strategies/challenge_export.py`, copied without changes to `scripts/challenge_export.py`.
SHA-256: `37ef9f02e5262aa517c9f8944735d7d30c373f7b9afed9cce4000d3d19f1e259`.
The adapter runs no model, reads no reference labels, and makes no unit conversion.

From the repository root, check the pin before each export. Stop if either command fails:

```sh
sha256sum --check docs/export-adapter.sha256
python -X utf8 scripts/challenge_export.py approved-raw.csv --out-dir PRIVATE_SCRATCH/export --image-height FULL_IMAGE_HEIGHT
```

Use the original scan height in pixels. Keep real outputs outside tracked source.
The raw header is `top_y,bottom_y,lithology,cluster_id,top_depth,bottom_depth`.
The app's `base_depth` must be explicitly mapped to raw `bottom_depth`, without changing its value.

The adapter writes:

- `*.notebook.csv`: `top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id`.
- `*.slide.csv`: `top_y,bottom_y,lithology,cluster_id,top_depth,base_depth`.
- `*.manifest.json`: the input hash, row count, schemas, and output hashes for this one export.

`base_depth` equals `bottom_depth`, including the original numeric string. No lithology relabelling occurs.
Missing/nonfinite depths, malformed rows, invalid cluster IDs, overlaps, negative/out-of-order coordinates,
and intervals beyond the supplied image height are rejected. Input changes during reading are also rejected.
Empty output is not a completeness check: this adapter does not require a nonzero row count.

Use this generated example without inference:

```sh
sha256sum --check docs/export-adapter.sha256
python -X utf8 scripts/challenge_export.py docs/showcase/fixture-harness.csv --out-dir PRIVATE_SCRATCH/fixture-export --image-height 2200
```

The fixture input only renames saved `base_depth` to `bottom_depth` and orders fields.
The unchanged API CSV is at `docs/showcase/fixture-api.csv`.
An official document still needs source identity, a calibration/unit/datum receipt, and an annotated column.
Annotations must show boundaries, depths, and lithology/cluster labels. Format success does not prove accuracy.
The automatic outputs for the three released documents are in [`outputs/official/`](../outputs/official/). To produce the same layout for new pages, see "Running on new verification data" in the [README](../README.md).
