# Independent implementation review

Reviewed on 2026-11-08 by the delegated Astra reviewer. The review covered the original eight-slide challenge deck, the existing bake-off specification, human-label provenance, outside-data limitations, and the new implementation. Review findings were sent to the implementation owners while they worked; the reviewer did not edit implementation code.

## Corrections verified

- Full-page truth CSVs now score subwindows without rejecting intervals elsewhere on the page. Labelled windows without any truth-filled pixels are rejected.
- Models receive validated image dimensions and public annotations. Truth paths, human bed annotations and split metadata are excluded from requests. Full-image and assisted-legend runs remain separate experiments.
- Nonfinite timeouts and thresholds are rejected. Python and NumPy random seeds are set in each worker. Package import occurs inside the timed worker, and observed child processes are cleaned up after successful exits as well as timeouts.
- Preview decoding uses the bounded reader. Decode estimates account for 16/32-bit image modes. Missing cluster IDs are rejected instead of inventing a distinct cluster for every interval.
- Real-pattern synthesis clips source beds to the selected window. This handles Shenzi windows entirely inside the long Halite bed. Derivatives retain parent group, split and source hashes.
- Unknown depth datums do not produce a comparable depth-error metric. The Flat Bay manifest's feet conversion takes precedence over the outdated README claim about metre-valued truth.
- Comparisons check dataset, case set, input mode, scorer version and scorer hash. Saved case counts and checksums detect truncation or edits. Paired group-bootstrap intervals preserve dependence between windows from one well.
- Release publishing has the Actions read permission required to retrieve tested build artifacts. Archive traversal tests account for Windows ZIP path normalization.

## Verification evidence

The reviewed working tree passed **48 tests** on Windows/Python 3.13. The CI configuration separately targets Python 3.12 on Windows and Linux; successful hosted runs should be checked in Actions.

Independent checks beyond the repository suite:

- 1,000 randomized sorted/gapped interval examples matched a simple raster row-accuracy oracle.
- A plugin whose package initializer sleeps for 20 seconds was stopped and recorded as a failure in 0.765 seconds with a 0.5-second worker timeout. Coordinator startup/provenance accounts for the additional wall time.
- A successful plugin spawned a sleeping subprocess; the child no longer existed after evaluation returned.
- The actual imported manifest contained 84 records. Inspection confirmed the three human windows with no complete bed inside the window that motivated the synthetic clipping fix.

## Interpretation and remaining limits

The small baseline tests segmentation and clustering but names rocks `Unknown` without supplied legend assistance. Synthetic assisted scores are construction tests, not evidence of real-page OCR or unseen geological generalisation. The shared diagnostic well was used in prior tuning and remains explicitly unsuitable for a fresh holdout claim.

Plugins are trusted Python code. Removing truth fields from requests and stripping credentials prevents accidental leakage; it does not stop a plugin deliberately reading local files. Sampled RSS can miss short peaks. Child cleanup covers observed descendants, not a hostile process deliberately escaping supervision. GPU memory and paid API costs are not instrumented.

Prediction/submission CSVs use the challenge field names, but evaluations operate on windows and may lack depths. Previews show a bounded source crop beside full-window prediction/truth strips. These are experiment diagnostics; a complete per-document challenge submission still needs full requested coverage, valid depths and suitable annotated images.

This review did not certify every optional OCR dependency, model's accuracy, external checkpoint history or remote deployment. Model config files and recorded hashes make those inputs inspectable; undeclared training history and dynamically imported external helpers still require the model author's disclosure. Review the latest hosted CI/evaluation results and actual failure cases before selecting a model.
