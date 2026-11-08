Availability v2 is a separate derived view. Baseline scores, truth, predictions and shared scorer are unchanged.

Use audit.json to bind derived_scores.json and the pinned inputs. reference_diagnostic retains qualified source-A diagnostics; qualified_development excludes lithology/ARI/product for all twelve unreviewed silver rasters and excludes Hadrian unQA. Silver contact scores remain source-A diagnostics. This means development eligibility, not unexposed validation. DEV and development gold remain development data. No champion is declared or promoted.

A missing metric is null with reasons and explicit denominators. Failed predictions on valid supervision remain zero for score, row accuracy, boundary F1 and ARI. Missing depth is never zero MAE. Labels, interval contacts, clusters and depth values remain in the referenced frozen CSVs. The original strict and four-name owner-equivalent raw metrics are retained per record. No extra spelling or geology mapping was added.
