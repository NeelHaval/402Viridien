# Import checks

These checks ran on Windows with Python 3.12.7. The source checkout and temporary
files were on drive S because drive C had no free space. The active evaluator
used `PYTHONPATH=src`. No model inference or training ran for this import.

| Check | Command and working directory | Actual final summary |
| --- | --- | --- |
| Active repository regression and stress checks | `python -X utf8 -m pytest -q tests stress_testing/checks` from the repository root; a temporary directory was supplied with `--basetemp` | `375 passed in 230.08s (0:03:50)` |
| Workspace split and release installer boundaries | `python -X utf8 -m unittest discover -s viridien/tools -p "test_*.py"` from `workspace/` | `Ran 10 tests in 0.371s`; `OK` |
| Source harness, converters, grading availability, label identity and challenge export | `python -X utf8 -B -m pytest -q harness strategies/test_grading_policy.py strategies/test_legend_identity.py strategies/test_grading_availability_v2.py strategies/test_challenge_export.py external/test_build_testset.py external/test_kgs_extract.py external/test_scanify.py` from `workspace/viridien/bakeoff/`, with `PYTHONPATH=.` and a temporary `--basetemp` directory | `187 passed, 9 skipped in 50.59s` |
| Pattern editor JavaScript logic | `node --test tests/test_queue_navigation.cjs tests/test_suggestions.cjs tests/test_verification_model.cjs tests/test_viewport.cjs` from `workspace/viridien/pattern_lab/` | `tests 22`; `pass 22`; `fail 0` |
| Saved score release bindings | SHA-256 checks for active scores, preserved original scores, the external version notice and the JEV checkpoint named in `SCORE_RELEASE.json` | `Score release bindings checked: 4; errors: 0` |

The nine skipped source checks need an optional original source PDF. The
downloader regression checks that the catalog selects `NeelHaval/viridian402`
and that a downloaded archive passes checksum and path checks before extraction.

## Saved label status

From `workspace/labelling/`, with the original `LITHOLOGY_DATASET` configured:

```sh
python -X utf8 scripts/validate_labels.py "60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev" "60811405120000_1inch tvd mudlog"
```

The actual summary was `1/2 pages valid` (exit 1). The partial GC743/DC141 page
has two layers and one error: `box without a rock / legend_id not in legend` at
row 7988. The 356-layer TVD mudlog has zero errors and 14 coverage warnings.
The imported JSON files retain the saved annotations. No class or missing label
was inferred during the copy.

## Source and artifact integrity

`python -X utf8 workspace/check_import.py --bundle-dir DIRECTORY` checks every
copied file against `SOURCE_MANIFEST.json`, parses all copied Python files and
checks the size and SHA-256 of each of the five fixed ZIP files. The import
preserves file bytes through `workspace/** -text` in `.gitattributes`.

The actual import audit summary was
`Files: 4682; Python syntax: 743; bundles: 5; errors: 0` (exit 0). The credential
scan checked 2,904 text files and found zero flagged files.

The source scanner checks text files for private-key blocks, GitHub and API
credential formats, private review links and credentials embedded in URLs. It
prints file names only when it finds a match; it does not print matched values.

## Not run

- Full generator and learned-model suites: not run for this import. The earlier
  source [check record](viridien/docs/CHECKS.md) lists inherited experimental
  failures. It is historical evidence, not a current passing result.
- Model training, API labelling and new dataset inference: not run.
- Browser interaction or a visual review of every archived report: not run.
- Sealed verification and a clean dependency installation: not run.

The archived source retains experimental behavior and host-specific scripts.
The current evaluator and checked dataset policy remain in the repository root.
