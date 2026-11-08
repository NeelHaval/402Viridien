# Inspect and generate synthetic patterns

Generated labels are exact for the generated pixels. They do not establish real
log accuracy. Read [status](STATUS.md) and [the critiques](../reports/generation/).

## Inspect the frozen output

From the repository root:

```bash
python viridien/tools/workspace.py download generated
```

Files go under `pattern_lab/data/genlab/`. `generated/manifest.json` links each
sample to its document, legend entry, source boxes, tile method, donor, and review.
Native images and padded copies stay separate. `PUBLICATION.json` marks the
bundle as experimental. Count distinct `pattern_id` values to count patterns;
four rendered variants are not four independent sources.

## Run the generator lab

Install the seed and set the original scan path first:

```bash
python viridien/tools/workspace.py download lab
python viridien/tools/workspace.py download generated
cd viridien/pattern_lab
python -X utf8 -m genlab.server --host 127.0.0.1 --port 8781
```

Open <http://127.0.0.1:8781>. The lab compares real crops, tiles, strips, and
columns. It rebuilds when local accepted labels change. A cold build can take
longer than an update. For one build, run `python -X utf8 -m genlab.build` from
the same directory.

The builder uses eligible training crops, machine evidence, reviewed proposals,
donors, and swatches. These sources have different evidence levels. A confident
AI review is not a human acceptance. Check native pattern spacing and line
continuity against real column crops. The
[generator README](../pattern_lab/genlab/README.md) gives the detailed rules.

## Historical synthetic v5

```bash
python viridien/tools/workspace.py download synthetic-v5
```

This installs `viridien/archives/synthetic-v5/`. It is separate from current
training data. `manifest.jsonl` describes samples; `real.json` contains separate
real evaluation records. The original card and scores remain historical.

Later reviews found an exact-pixel quarantine gap, rails and seams, unlabelled
areas treated as background, and unreliable bed-thickness statistics. These
findings supersede earlier claims of a clean split. Fix work is included, but
this publication does not certify the complete dataset as fixed.

## Assembly, board, and atlas

`genlab/assemble/` has thickness, layout, noise, and benchmark experiments. Use
complete labelled intervals for bed statistics. Fixed-height pattern boxes are
not complete beds. Unknown rows need an ignore state.

The read-only board and atlas retain their original source paths. Run each in a
separate terminal from `viridien/pattern_lab/`:

```bash
python data/genlab/fleet/board/board.py
python data/genlab/atlas/atlas.py
```

They use ports 8782 and 8783 and expect local lab outputs. A fresh checkout has
no active maker fleet or job history. Counts do not describe jobs on another host.

Audit and maker-fleet support scripts are also preserved under
`pattern_lab/data/genlab/critique/`, `fleet/`, and `datasets/diag/`. They retain
their original paths for traceability. They can require the original host's
agent tools or local measurements. Installing a bundle does not run them.
