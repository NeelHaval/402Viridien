# Availability v2 report

The prior strategy default is withdrawn. No replacement champion is promoted. Changes in metric means come from reference availability and denominator corrections, not a new model run.

The report uses the separately audited derived_scores.json. The parent audit checked the final derived bytes. The renderer also checks the bound data audit, exact baseline case keys, source pins, metric availability, null reasons, and summary denominators before it writes the page.

The main series is qualified_development, shown as eligible development reference. It excludes all unreviewed silver lithology, ARI, and product values, and all Hadrian values without QA. Silver boundary F1 remains a source-A contact diagnostic. Supplied DEV references remain development data. Neither this name nor a passing audit certifies independent human truth or unseen wells.

The reference_diagnostic series retains valid source-A silver diagnostics. Empty Neptune has no supervised metrics. The two vocabulary-invalid cases cannot supply lithology, ARI, or product metrics. Missing depth truth and unmatched prediction depth are unavailable, never zero depth error. Failed predictions with valid supervision retain the derived zero failure metrics.

Strict labels and the owner-configured symmetric Clay/Shale equivalence remain separate. The shared scorer, raw labels, predictions, baseline scores, and baseline raw results are unchanged. The exact original HTML is in index_baseline_v1.html, with its copy hash in baseline_v1_preservation_receipt.json. The archive contains the withdrawn claim.

No protected truth image or label was opened for this report. No new truth overlay was generated. Image status is not-generated_due_to_owner_hold. This is not a render failure. Historical vocabulary-invalid truth images must not be treated as approved target labels.

The phone check used a 390 px page width. The page had no console messages. The saved contact sheet is 1020 by 900 px. It shows the heading, the availability chart, and the first metric tables. The tables scroll horizontally. It contains no protected source images.

Commands from the workspace root:

    python -X utf8 viridien_work/bakeoff/report/scoreboard_1108/availability_v2/build_report.py
    python -X utf8 -m pytest viridien_work/bakeoff/report/scoreboard_1108/availability_v2/test_report_contract.py viridien_work/bakeoff/report/scoreboard_1108/availability_v2/test_report_audit.py -q

The test summary and the final HTML/data/audit hashes are reported in the parent handoff and report_receipt.json. The report does not claim that the official challenge uses this local product, that an untested hybrid wins, or that the frozen matrix meets the later 1–2 minute latency goal.
