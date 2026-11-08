# Final lane review resolution

Both read-only reviews were read before the final run. No sealed data were opened.
Only split declarations were read as metadata. There was no search for sealed data.

| Finding | Action and limit |
|---|---|
| 1. Old boundary and Emil weights | Confirmed by model hashes and saved fit records. S13-S18 and deployed S20 remain historical comparisons. S19, S21 and S22 have new five-well fits. S12 is outside the requested table. |
| 2. Numbered feature and fold caches | Confirmed. No old boundary, matcher or S18 positional cache was used for the choice or a new fit. Legacy boundary and matcher cache files are outside final-lane ownership. Their training commands remain unsafe for reuse. The S18 prepare and select commands now refuse finalized experiments or unversioned interval caches. A safe full legacy rebuild needs its owning lane. |
| 3. Benchmark dependency hash | Fixed. The digest includes ML code, deployed models and selected settings. Prediction, strategy and preprocessing paths use it. A test changes an imported helper and a model file and checks that the digest changes. Final fingerprints supersede earlier lane READY hashes for the changed shared code. |
| 4. Depth cache identity and wrong scale | Fixed. The depth signature hashes the exact track and unit-context pixels in bounded strips. Equal-support fits rank residual before span. An OCR-only regression checks the correct scale. S19 also retains its general weak-fit check. No reserved image or truth was used to set this repair. |
| 5. Unit precedence | Fixed. Explicit “Depth in feet” or “Depth in metres” has priority over ROP units. Both cases have invented-data checks. |
| 6. Invalid and conflicting depths | Fixed. The runner and scorer reject non-finite values and inconsistent shared-boundary depths. NaN, infinity and a 950-ft conflict have checks. |
| 7. Failure means and plot errors | Fixed. Legacy scored failures set all main quality metrics to zero. Both runners record plot failures separately from prediction quality. An invented render failure keeps a perfect prediction score. |
| 8. Repeated truth-read failure | Fixed. The legacy outer exception path does not read truth again. A two-item invented bad-CSV run records both failures and writes its summary. |
| 9. Consensus ties | Fixed. Both runners sort labeler paths before first-vote tie resolution. No reserved truth was read for this change. |
| 10. Joint blank gaps | Fixed. Split the DP into nonblank spans before applying rock duration. An eight-pixel blank stays blank with a 16-pixel minimum bed. |
| 11. Rule preservation | Fixed. Restored cuts must be far enough from the previous cut and the bottom. Strong preservation requires high probability. A pair of close rules cannot create a bed below the minimum. |
| 12. Restored probability | Fixed. Each restored row gets the probability at its own top. A check uses probabilities 0.2 and 0.95. |
| 13. Paired source mismatch | Fixed. Comparisons reject different dataset or truth fingerprints when supplied. The benchmark records both. Invented pairs check both mismatch cases. |
| 14. Old nested diagnostic | Confirmed. It used a search and base fits with outer-well exposure. It is not independent nested CV. It was not used for this choice. New lane CV results are also selection estimates, not independent nested CV. |

The leak review also confirms old matcher, parser, synthetic-model and confidence
selection exposure to Forsyth. None of those inherited assets was used to train
S19. S20 deliberately still uses inherited S18 weights. Its clean parser CV does
not make its deployed strategy clean. Historical final runs already exist. New
weights do not remove inherited design exposure or prove an untouched blind test.

Current dev identities were checked against all three clean lane indexes.
Synthetic train and held source IDs were checked for overlap and eval exclusion.
Clean deployed boundary metadata list only the five allowed wells. S19's twelve
fold/all fit hashes were checked before the final run. See ml/final/audit.json
and selection_recheck.json for the records.

The final benchmark now supports four workers with one compute thread each.
It caps nested BLAS decorators, torch intra-op and inter-op pools, OpenCV and
RapidOCR. Shared verify permits only ForsythB2-17. Permanent sealed identifiers
cannot be opened with the unseal flag.

An inherited combine test expects the old shared depth fitter to return a bad
slope greater than 3. The fixed fitter returns about 0.13323. That test belongs
to another lane and was left unchanged. Its initial failure is in lane_tests.log.
The final check deselects that obsolete assertion and runs an owned replacement
that requires the correct scale. This is a test expectation conflict, not a
remaining scale defect.

## Final resource finding

The full four-worker run peaked at 5.423 GiB total process-tree RSS. It did not
meet the 4 GiB limit. The earlier dev smoke check peaked at 2.301 GiB. No
strategy, model, threshold or code was changed after the freeze. There was no
held-out rerun. This resource failure is retained in the report and machine table.
