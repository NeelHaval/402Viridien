# Run locally

Use Python 3.12. For private downloads, install GitHub CLI, run `gh auth login`,
and use an account with access to `NeelHaval/viridian402`. A token with repository
read access in `GH_TOKEN` also works. Do not put tokens in config files.

## Get the code without the large scans

Run these commands in PowerShell, bash, or another terminal:

```sh
gh repo clone NeelHaval/viridian402 -- --filter=blob:none --sparse
cd viridian402
git sparse-checkout set src scripts configs docs tests examples stress_testing .github notebooks data/provenance
python scripts/local_test.py
```

The last command creates `.venv-local-test`, installs the constrained package,
generates one small manufactured example, and validates its inputs. It does
not run a model. Later calls reuse the environment. First setup needs internet
access for Python packages. The repository contains large scans and weights;
a full clone can be much larger than this sparse checkout.

## Download and check the real development data

```sh
python scripts/local_test.py --suite quick-dev --check
```

This downloads about 51 MB, verifies the pinned asset hashes and extracted
source hashes, loads the well exclusion overlay and v2 policy, and prepares
two development windows from distinct wells. It validates the native evaluator
manifest without running inference. Downloads are cached under the chosen
data directory. New downloads require private repository access.

Run an algorithm only when you intend to request that evaluation:

```sh
python scripts/local_test.py --suite quick-dev --run
python scripts/local_test.py --suite dev --run --model your_package:predict --config configs/your-model.json
```

`quick-dev` uses two fixed windows. `dev` uses the 51 outside DEV windows.
The default model is the weak built-in baseline. A successful run establishes
execution, not useful rock-label accuracy. Each command prints the HTML report
and Markdown summary paths. The report includes source images, predicted and
reference intervals, failures and resource measurements. Per-case timeout is
60 seconds by default; use `--timeout` to change it.

Use `--data-dir PATH` to put data and the download cache on a drive with enough
space. Use `--output PATH` for reports. The helper checks free space before each
large archive. Installation and download time are separate from model run time.
No two-minute full-dataset runtime is promised.

## Download other frozen selections

```sh
python scripts/local_test.py --download human
python scripts/local_test.py --download synthetic
python scripts/local_test.py --download unlabelled
python scripts/local_test.py --download all --data-dir YOUR_LARGE_DATA_DIRECTORY
```

`all` downloads the frozen v1 payloads, including held-out inputs, and the checked
metadata notices. It excludes held-out reference labels. Allow roughly 6 GB of
free space for the full download and extraction. Large archives remain private.

The human archive retains its immutable 27-page, 458-box history. Use only the
derived `metadata/human-eligible.json` selection of 15 pages and 422 approved
boxes for the stated human diagnostic. The other 12 pages and 36 boxes are
excluded by the protected-well policy. Human truth is sparse box JSON; it is
not a full interval/depth reference. The interval runner does not silently
convert these boxes into full-page truth. Synthetic downloads retain ignore
masks and their original scoring instructions. They are not fed into the
interval runner without the required mask-aware scorer.

Held-out reference labels require a separate scorer directory and explicit opt-in:

```sh
python scripts/local_test.py --download heldout-references --allow-heldout-references --data-dir data/scorer-only
```

Never give that directory to a prediction model. This command does not run or
train a model. Existing original release assets and historical flags are kept;
the exclusion overlay supplies the effective permission correction.

## Software checks

```sh
python scripts/local_test.py --software-tests
python stress_testing/run_stress.py --profile quick
```

The first command installs the test extra and runs the regression and stress
checks for the core evaluator, adapters, reports and exchange tools. The second
writes separate bounded stress evidence. The web demo needs its separate
locked environment and `--include-demo`; see the
[application stress guide](../stress_testing/README.md#test-the-current-backend-and-frontend).
These use
manufactured fixtures and need no private dataset downloads. Optional science
dependencies can produce explicit skips; see the test output. Use
`--no-bootstrap` only when the current Python environment already has the
package dependencies. This option is mainly for CI and developers.

## Dataset status

The download lock separates the frozen payload version from metadata v2.
The metadata update does not replace the original images, annotations or truth.
DJS v2 still has five unavailable depth calibrations and one wrong fit; the old
frozen DJS data is not presented as a checked v2 payload. Prior truth-overlay
exposure remains recorded. The original and revised 130-page sets are paired
versions of the same pages; their depth-unit issue is OPEN. Synthetic v2 remains
unaccepted and its gate is OPEN.

DEV `autoLabelAllowed` is false. This helper supplies no auto-labelling, JEV or
model-API dispatcher. The `--run` flag is an explicit local evaluation request;
metadata refreshes never trigger it. Protected wells and descendants remain
blocked for training, tuning, label generation and review confirmation. Missing
identity is denied. Silver labels remain unreviewed source-A-on-ties labels.
Missing depth and empty truth cannot supply the affected accuracy metrics.
These development checks do not establish independent or official accuracy.
