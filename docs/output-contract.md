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
The three released S19 comparison files initially lacked depths; they stay blocked until their owner supplies calibration.

The package owner alone writes the portable submission bundle. The tool is installed at
`/srv/viridien/submission/20261108/bundle/tools/verify_package.py`, with its S: bundle mirror.
The owner reported `--integrity-only`: `INTEGRITY_OK files31 errors0`, and a changed-artifact
copy was rejected. That proves the reported copy/hash check, not submission readiness.
The final package gate is:

```sh
python tools/verify_package.py --strict
```

Run it from the portable bundle. The tool is not part of this source checkout.
The owner's reported `--strict` result is blocked by 14 missing gates, including accepted
S19/app/UI identity, preload, live providers, public source, and official depth outputs.
`--integrity-only` checks hashes/copies; it is not a submission-readiness result.
Changed source or CSV files must pass the package owner's current strict gate again.

## Current source handoff

D is building a combined candidate with S19, upload/queue/result-identity fixes, both phone
fixes, partial CSV/annotation exports, and the safe provider patch. This document does not
assert that the build or final app acceptance has passed. D alone supplies that receipt.
The fresh public source packet still identifies core candidate `559c99cb4e24`; it must be
refreshed only from D's accepted source/archive/hash receipt before it represents the final app.
The existing repository remains private. Public source requires a new reviewed allowlist
with fresh history and an HTTPS clone check without sign-in. Private README images and
old research/import history must not be copied to that public repository by default.
