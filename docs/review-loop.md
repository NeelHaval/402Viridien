# Independent contributor and shipping review

GPT-6 Astra independently reviewed the contributor path, workflow inputs, packaging, offline doctor behavior, archive handling and crop attribution during the 2026-11-08 `improvement/deep-review` fix loop. Metric/data-oracle review and real-model benchmark interpretation were assigned separately. This review made no implementation changes; it added `tests/test_shipping_deep.py` and this evidence record.

## Findings fixed and rechecked

| Finding | Fix and observed verification |
| --- | --- |
| Source distribution omitted the new contributor guide, Python model template and PR template | Manifest includes all three; inspected their presence in the built v0.2.0 source archive. |
| Doctor accepted Windows BOM-prefixed JSON while evaluation/classification rejected it | Config readers now agree. Both direct CLI regressions pass; workflow tests also cover BOM configs/locks. |
| One malformed nested crop summary crashed recursive run listing | Invalid crop schemas are normalized/rejected. Null, array and numeric summary regressions pass without hiding valid runs. |
| Combined report links were relative to individual runs | Downloaded combined links now resolve from the artifact root. An actual selected-case run produced two links, both verified to exist; the GitHub summary displays labels instead of broken local links. |
| Release tag could disagree with package version | Release workflow checks the tag against `pyproject.toml`; the v0.2.0 check succeeds. |
| Custom crop history could be confused with the helper prototype's train-only history | Summary distinguishes custom-model declarations/unknown history, and the loader checks that history against recorded model/config. |

The eight independent shipping regression cases additionally confirm that CRC corruption writes no destination files, conflicting extraction preserves local edits before committing any new file, and packing cannot overwrite an input manifest. Existing validation covers traversal, symlinks, expanded-size limits and nonportable Windows names.

## Executed contributor path

A separate checkout-shaped directory, with no private data bundles, received a copy of `examples/models/template_model.py` as `src/shipping_template/__init__.py`. Pip built and installed that checkout and the template into a separate target. Explicitly checked imports came from the installed target, not the development source tree. Dependencies were reused from the existing Python 3.13.4 environment; this is package isolation, not a newly resolved dependency environment.

The installed model passed `doctor --suite demo --model shipping_template:predict --config configs/baseline.json`. All missing real-data bundles remained informational. The workflow's demo `--fetch-data` path completed without fetching. A manifest override to `examples/demo/manifest.json` and selection `{"case_ids":["demo-001"],"max_worker_rss_mb":256}` ran exactly one requested case, completed 1/1, and appeared in recursive run listing. Local evidence is under `.local/deep-shipping-clean/runs/demo/20261108T044526Z-24036b6274`.

Doctor discovers plugin files without importing them; an import-raising plugin regression confirms this behavior. It imports known dependencies and audits requested local inputs. Selected workflow config/manifest paths resolve inside the checkout, filter JSON has an explicit schema, and bundle overrides select only recorded lock entries. Workflow inputs are environment data, not interpolated shell code. The GitHub read token is confined to the fetch step and absent from the evaluation step; worker environment handling is tested separately.

## Verification scope

- Windows / Python 3.13.4: final working-tree suite **172 passed**.
- Final v0.2.0 source distribution: extracted to a fresh directory, **172 passed**. Required contributor/release resources are present.
- Final wheel installed into a separate target: **172 passed** with pytest's source-path override disabled; its import path was verified and all packaged runtime Python files matched the reviewed checkout. Artifacts are under `.local/deep-shipping-final/`.
- All three GitHub workflows pass local `actionlint`; the release version check accepts `v0.2.0`.
- Offline build verification used pinned setuptools 80.9.0 and wheel 0.45.1 in an isolated local build-dependency directory. They were downloaded through automatic approval after restricted execution could not resolve them; no user confirmation was needed and the project virtual environment was not modified.

No unresolved contributor/shipping blocker remained in the reviewed local paths. Remote CI execution, authenticated dataset download and release publication for this new revision require their own evidence; local checks do not establish those outcomes. Existing legacy constraints and build-system ranges do not provide fully hash-locked transitive environments. Passing execution checks and the template's zero accuracy are not claims of useful geological prediction.

Some Windows runs printed a `pyreadline3` console-destructor warning after pytest completed; exit status was zero and all tests passed. This optional dependency warning was not counted as a model or test failure.
