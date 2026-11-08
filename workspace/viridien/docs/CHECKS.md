# Publication checks

These checks ran in the copied GitHub workspace on Windows with Python 3.12.7.
They check publication integrity and selected tool behaviour. They do not
establish a new model accuracy result.

| Check | Command or method | Recorded result |
| --- | --- | --- |
| Repository gate | `bash scripts/test.sh` from the repository root | `Tooling and detected project tests passed.` |
| Split and installation boundaries | `python -X utf8 -m unittest discover -s viridien/tools -p 'test_*.py'` | `Ran 9 tests`; `OK` |
| Real image integrity | `python -X utf8 viridien/tools/workspace.py verify --require-assets` | `Images: 2961 checked`; `PASS: real data splits, quality holds, source bounds and generator permissions` |
| Python source syntax | Parse each copied Python file with `ast.parse` | `Python syntax files: 406 errors: 0` |
| Shared data file hashes | SHA-256 against each subset's `SHARED_TAG.json` | 53 development files and 36 check-subset files; zero hash errors |
| Release bundle integrity | Check ZIP checksum, CRC, unique members, linked pattern assets, and accepted annotation versions | `AUDIT ERRORS: 0` across all five bundles |
| Labeller startup | Start a separate local process from the downloaded seed; GET health, document list, and review page; stop that process | `PASS: fresh-seed lab health, document list, and review page HTTP responses` |
| JavaScript logic | `node --test tests/test_queue_navigation.cjs tests/test_suggestions.cjs tests/test_verification_model.cjs tests/test_viewport.cjs` from `pattern_lab/` | `tests 21`, `pass 21`, `fail 0` |
| Core harness and converters | `python -X utf8 -m pytest -q harness external/test_build_testset.py external/test_kgs_extract.py external/test_scanify.py --disable-warnings` from `bakeoff/` | `123 passed, 9 skipped in 47.62s` |
| Documented example | `python -m harness.bench --strategy s11 --suite ext_shared --workers 1 --no-cache` from `bakeoff/` | 17 of 17 windows completed; zero failures |

## Experimental suite failures

Command from `viridien/pattern_lab/`:

```bash
python -X utf8 -m pytest -q tests generator/test_core.py genlab/tests --disable-warnings
```

Actual final line:

```text
12 failed, 169 passed, 2 skipped, 8 errors in 116.41s (0:01:56)
```

This inherited experimental suite does not pass. The failures cover generator
source bridging and caches, column-transition sampling and fallback, scanline
composition, atlas inputs, and coverage weighting. The eight errors include
missing assembler measurements and fixtures without the normalized eligibility
index now required by the dataset builder. These failures are retained and listed;
the source and synthetic downloads are not presented as a validated system.

Failed tests:

```text
tests/test_normalized.py::test_generator_bridge_uses_only_train_sources_and_exports_same_label_ids
genlab/tests/test_assemble_bench.py::test_seeded_suite_orchestration_without_private_compose_imports
genlab/tests/test_assemble_compose.py::test_fixed_transitions_chisquare_2000_layers
genlab/tests/test_assemble_compose.py::test_zones_condition_second_order_and_gaps_tile
genlab/tests/test_assemble_compose.py::test_unavailable_classes_are_renormalized
genlab/tests/test_assemble_compose.py::test_zone_composition_breaks_inapplicable_transition_cycle
genlab/tests/test_assemble_compose.py::test_only_exact_names_can_substitute_within_family
genlab/tests/test_assemble_compose.py::test_sparse_document_backs_off_family
genlab/tests/test_assemble_compose.py::test_scanline_windows_match_small_native_image
genlab/tests/test_atlas.py::test_all_861_board_entries_belong_to_exactly_one_symbol
genlab/tests/test_genlab.py::test_build_uses_cache_verdicts_and_human_override
genlab/tests/test_genlab.py::test_coverage_weights_rows_and_separates_training_documents
```

## Not run

- Browser interaction and visual regression: not run for this publication.
- Full model training or a fresh accuracy benchmark: not run.
- Sealed verification: not run.
- Installation in a clean Python environment: not run. Package ranges and the
  tested host's available environment are not a dependency lock.

## Final command summaries

```text
Repository gate: Tooling and detected project tests passed.
Gate unit groups: Ran 16 tests, OK (skipped=2); Ran 9 tests, OK.
Real images: Images: 2961 checked
Bundle audit: AUDIT ERRORS: 0
Core harness and converters: 123 passed, 9 skipped in 47.62s
JavaScript: tests 21; pass 21; fail 0
Shared example: 17 windows, 3 wells, failures 0
```

The nine skipped converter checks require an optional source PDF. The skipped
experimental tests and the failures above are separate. A PNG input fallback
and legacy package imports were corrected during publication.
