# Dataset review release, 8 November 2026

This is a private dataset snapshot for `NeelHaval/viridian402`.
Use `DATASET.json` as the release manifest. All data paths in that manifest
are relative to its directory. The source metadata is retained without edits.
Some source metadata has archived Windows paths. Use the relative paths in
`DATASET.json` when you run an algorithm on another computer.

## Contents

| Selection | Records | Use |
|---|---:|---|
| `synthetic` | 82 pages; 69 ready, 13 blocked | Construction-truth diagnostics. One base well per family. |
| `human-crops` | 27 training pages; 458 approved boxes | Sparse lithology checks. Ignore all pixels outside approved boxes. |
| `ext_dev` | 51 windows | Development reference. |
| `ext_shared_verify` | 8 windows | Shared reference check. |
| `ext_test` | 20 windows | Held-out reference check. |
| `mirror_lake` | 32 windows from one well | Held-out reference check. |
| `djs_idaho_test` | 25 windows from one well | Held-out reference check. Depth was unavailable on 24 windows in the saved run. |
| `frozen130_unlabelled` | 130 pages | Label-free diagnostics. Depth was unavailable on 32 pages in the saved clean run. |

The counts mix pages and windows. They are not counts of independent wells.
No permanently sealed images or labels are included.

The current synthetic manifest blocks five empty Gulf pages and eight MC765
pages. The MC765 boundaries came from AI labels of a base page. They are not
independent construction truth. Earlier notes that say 77 ready pages are
superseded by the 69-page selection in this snapshot.

## Download and check

Download the release assets with an authenticated GitHub CLI:

```sh
gh release download dataset-review-20261108-v1 -R NeelHaval/viridian402 --dir dataset-review
python -X utf8 dataset-review/verify_release.py --root dataset-review --archives
```

Check `SHA256SUMS.txt` with your checksum tool before use. The Python check
then reads all data members in the ZIP files and checks their sizes and hashes.
It does not extract or run the algorithm.

Extract the data ZIP files into one directory. Keep `DATASET.json` and
`verify_release.py` in that directory. ZIP members keep the original project
directory layout. File paths can include spaces and `#` characters.

Keep `outside-heldout-references-*.zip` in a separate scorer directory.
Extract it into the same layout only in a scoring environment. Do not give
held-out reference labels to the prediction model. The other held-out ZIP
contains the input images, annotation boxes and selection metadata.

For a complete extracted copy, check the files with:

```sh
python -X utf8 verify_release.py --root .
```

## Run an algorithm

Read the `records` array in `DATASET.json`. Select a `selection`, then keep
records with `ready=true`. Resolve `image` and `annotation` against the
manifest directory. Give only these public inputs and the selected row
window to the prediction algorithm. Keep `truth` and `ignore` for the scorer.

TIFF images and annotation JSON files are byte-for-byte source copies.
Synthetic interval CSV files use the challenge-style columns. Human truth
uses sparse box JSON. The truth formats remain distinct in the manifest.
The instruction ZIP also contains the existing evaluation adapters,
ignore-mask scorer and report code. Those source snapshots use the original
project harness and model dependencies. They are not a new installed model.

Synthetic scoring must remove ignored rows from all denominators. Contacts
at ignore-mask edges must not count as true boundaries. Report results by
family and base well. Do not report a pooled synthetic accuracy score.
Human crop checks provide no full boundary or depth score. Outside reference
labels came from programmatic extraction and audits. They are not
human-certified truth. Unlabelled pages provide no accuracy measurement.

The source data licences have not been verified for redistribution. This
release retains the existing private repository access. It grants no new
licence to the source data. Keep all attribution and lineage records.

This publication does not run model inference. The manifest, source hashes,
archive hashes and uploaded GitHub asset digests are checked separately.
