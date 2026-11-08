# Local Viridien workspace

This directory contains the source tools, experiment records and shared examples
from the local HackTheBurgh workspace. Start with the main repository's
[local testing guide](../docs/local-testing.md) for the current evaluator.

The import starts from `silmoon04/hacktheburgh26` commit
`3f883fc639bcb5861feb5879b9b0a0efef583301`, then adds the local source updates.
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records the source and SHA-256 of
each copied file. The listed adapted files have small changes for this location.

## Find the work

| Path | Contents |
| --- | --- |
| [viridien/START_HERE.md](viridien/START_HERE.md) | Original setup and tool guide. Run its commands from this `workspace` directory. |
| [viridien/bakeoff/](viridien/bakeoff/) | Benchmark harness, S11 and later strategy experiments, outside-source converters, score reports and their source snapshots. |
| [viridien/pattern_lab/](viridien/pattern_lab/) | Pattern review, crop export, generator and assembly fixes, grid and JEV experiment tools. |
| [labelling/](labelling/) | Team interval editor and saved label files, including the two local label updates. |
| [final_pipeline/](final_pipeline/) | Original standalone output validator and its checks. |
| [management/dataset-review/](management/dataset-review/) | Dataset catalog, split checks and review UI source. Large generated catalogs stay in the original data release. |
| [management/dataset-publication/](management/dataset-publication/) | Dataset release builder, verifier and advisory source. |
| [colab/HTB2026.ipynb](colab/HTB2026.ipynb) | Original challenge notebook. |

These tools retain their original paths and dependencies. Some experiment and
compute scripts need the original local datasets or host setup. Their presence
does not register a new model in `src/viridian402`.

## Download the fixed source bundles

The five source bundles are available in this private repository's
[workspace release](https://github.com/NeelHaval/viridian402/releases/tag/workspace-20261108-v1).
Their ZIP bytes and checksums match the original release. The catalog points to
`NeelHaval/viridian402`, so the downloader does not need access to the old repo.

From the repository root:

```sh
cd workspace
python -X utf8 viridien/tools/workspace.py download real
python -X utf8 viridien/tools/workspace.py verify --require-assets
```

The other bundle names are `lab`, `generated`, `synthetic-v5` and `models`.
Install only the bundle that the selected tool needs. The downloader checks its
size, checksum and paths before extraction. It keeps different existing files.
[catalog.source.json](viridien/data/catalog.source.json) retains the original
release metadata.

## Use the current data policy

The current [local testing guide](../docs/local-testing.md), checked exclusions
and release notices govern new work. Archived source guides describe earlier
experiments. Their split names and approval fields do not grant new training,
tuning, API calls or inference. In particular, the old crop eligibility counts
must be read with the current protected-well exclusions.

Keep construction truth, human crop checks, programmatic outside references and
unlabelled diagnostics separate. The archived score reports retain their original
data and metric versions. They do not establish a new accuracy result.

Private authentication files, reviewer links, live databases, task dashboards,
raw logs, original scans and permanently sealed pools are excluded. The separate
`htb13-brief` research site is outside this project import.

See [IMPORT_CHECKS.md](IMPORT_CHECKS.md) for checks run on this import. The
source's [earlier checks](viridien/docs/CHECKS.md) retain known experimental
failures and must not be read as checks of this new snapshot.

Run `python -X utf8 workspace/check_import.py` from the repository root to
check the copied file hashes and Python syntax. Add `--bundle-dir DIRECTORY`
to check the five downloaded ZIP files against the destination catalog.
