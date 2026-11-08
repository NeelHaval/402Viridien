# Review of the synthetic benchmark and the pipeline it tunes

This is a controlled stress test of an approved texture library, not a substitute
for independent real-log validation. Human approval establishes that a crop is a
reasonable example; it does not establish that it is representative of a well,
all scans of that well, or an entire legend family. The mock run is a software
smoke test using three D-582 labels from gold/agreed-consensus intervals and real
donor swatches. It supports no general performance claim.

## 1. Circularity and reader preference

**Risk.** JEV/LLM proposals and consensus generated with related readers influence
which examples reach approval, their labels, and the fitted bed statistics. A
classifier agreeing with those readers may score well without identifying the
real rock. Agreement among dependent readers is not independent ground truth.

**Check.** Audit provenance and best-source counts in prefill and the review
history. Compare JEV-selected, purely local-selected, and independently sampled
human-labelled real beds. Have a reviewer label a blinded random sample without
seeing model suggestions; keep those source logs out of tuning.

**Look at.** Crops where a human changed the proposed label, cases all readers
agree on but a geologist rejects, and whether JEV's advantage persists on the
blinded sample. Treat a >10 percentage-point accuracy difference between
JEV-selected and independently sampled examples as a reason to stop tuning.

## 2. Selection of the clearest stretches

**Risk.** Representative mining, agreement filtering, and approval favour clean,
long, periodic stretches. Unclear ink, partial fills, mixed percentages and beds
under twelve rows are underrepresented. Nuisance overlays cannot recover missing
geological diversity. Rare-pattern quotas also distort natural prevalence.

**Check.** Compare approval rates, widths, thickness quantiles, ink coverage,
colour distributions and source-log counts against the queue and volume budget.
Review rejected and disputed examples, not just sheets of approved ones.

**Look at.** Labels occurring in only one source log, overrepresented clear
examples, and six-row contacts. A real random-sample accuracy >10 points below
approved-crop accuracy is evidence that selection dominates the benchmark.

## 3. Beds absent from the catalogue

**Risk.** Unknown rocks, hydrocarbon shows, modifiers, annotations, unfilled
percentages and colours absent from the legend are not new rock classes here.
The generator makes beds from approved labels only. Overlaying a symbol does
not model a genuinely new compound lithology. Gold/consensus unknowns are excluded
from the fit, so benchmark coverage is conditional on known labels.

**Check.** Measure out-of-vocabulary and abstention rates on untouched real logs.
Sample rows excluded by catalogue matching. Score blank detection separately:
the harness deliberately ignores empty-label truth pixels and can therefore
reward a strategy that hallucinates rock through blank paper.

**Look at.** Unknowns near shows panels, nonlegend fill colours, and cases where
the correct action is abstention. More than 5% unrepresented filled real rows
should block claims that a synthetic score measures the full task.

## 4. Boundary realism

**Risk.** A first-order model forgets sedimentary sequences, long-range structure,
repeated formations and depth-dependent composition. Log-normal thicknesses are
an approximation; family fallbacks and a separate 6–30-row component are explicit
assumptions. Coverage quotas can force unnatural short contacts. Quilting seams
can create or hide boundaries. Genuine depth grids and borders can already exist
inside approved crops before any synthetic nuisance is drawn.

**Check.** Compare real and fake thickness CDFs, transition counts, thin-bed rates,
line/contact alignment, and two-/three-bed motifs. Run a blinded real/fake contact
review. Inspect native-scale seam crops and `source_crops.overlap_rows`.

**Look at.** Repeated motifs, disappearing short beds, abrupt colour edges, and
whether a boundary detector wins by finding quilting seams. A >10-point boundary
F1 change after changing quilting overlap or disabling drawn lines warrants
investigation. The harness's 4–20-row tolerance can hide errors on six-row beds.

## 5. Legend versus column mismatch

**Risk.** The same label can have different print scale, hatch, or colour in the
legend and column. A copied donor legend retains actual donor mismatch but does
not reproduce page-wide acquisition changes consistently: image nuisances affect
the column while the legend stays intact. A width difference above 15% causes a
flagged isotropic scale exception. Persistent edge rails are trimmed before
horizontal phase shifts, which can also remove real edge-adjacent ink.

**Check.** Stratify by width exception, donor, scan mode and native legend/column
descriptor distance. Compare raw donor crops and the clean control. Separately
run fresh legend OCR on the copied region.

**Look at.** `meta.json`, native previews, source rails and the actual copied
`legend.png`. Warm donor parser caches bypass fresh OCR; these scores do not
validate legend discovery. A >10-point gain from warm versus fresh parsing on
real logs would be a major deployment limitation.

## 6. Normalisation, aliases and cluster meaning

**Risk.** `harness.score.normalize` removes whitespace, folds case, and aliases
“Sand and Sandstone” to “Sandstone”. The mining normaliser also strips punctuation;
those rules differ. OCR spellings such as “SIllty” and “Anhydrlte” survive. One
label may cover several visual patterns, and different labels may be identical.
The scorer's cluster ARI compares clusters with rock labels, not catalogue pid.

**Check.** Export unmatched labels and normalisation collisions. Verify legend
spelling against gold and consensus before fitting; inspect original labels
alongside normalised ones. Evaluate visual clustering separately if that is the
desired scientific question.

**Look at.** Confusion pairs and OCR aliases, especially rock names containing
hyphens or punctuation. More than 1% filled rows changing correctness solely
through aliases, or any many-rock alias collision, should stop score comparison.

## 7. Depth calibration

**Risk.** Synthetic depths are exactly linear, with positive ft/px drawn from
training-family measurements. Real scans have breaks, repeated scales, OCR
errors, MD/TVD ambiguity, stretches and local distortions. Supplying known
synthetic calibration makes depth performance optimistic. Truth depths are
rounded to a tenth of a foot, which matters for tiny beds.

**Check.** On independent real pages, compare calibration anchors to manually
read depth marks, including near scale changes. Report calibration failures and
depth MAE separately from lithology accuracy. Verify ft/px against volume logs.

**Look at.** `calibration.method`, original depth columns, residuals and the
conversion of a 20-row contact tolerance into feet. Any slope error >2%, or
systematic contact error larger than the family's allowed feet tolerance,
invalidates apparent depth success. Synthetic depth MAE is a plumbing check.

## 8. Overfitting to synthetic nuisances

**Risk.** The probabilities and parameter ranges are engineering choices, not
fully measured nuisance distributions. A solver can learn thresholds for these
specific JPEG qualities, fades, gaps or dithers. Independent nuisance draws do
not reproduce correlations in a scanner or contractor. LZW is lossless: its
presence tests decoding and cannot honestly be called a visual artefact.

**Check.** Freeze probabilities before tuning. Stress-test unseen seeds and
parameter ranges, compare single-nuisance ablations, then check real scans.
Compare grayscale and 1-bit effects: final thresholding can erase injected noise.

**Look at.** Selected versus applied flags and unavailable real overlays.
Mixed-nuisance paired drops measure the combined image changes. Structural
interbeds/gaps stay in the paired clean bed plan, so their paired causal drop is
NA; the report's clean-log comparison is descriptive and confounded. If rankings
reverse after a modest probability change, or synthetic gains fail on real logs,
do not choose a pipeline from the synthetic ranking.

## 9. Leakage

**Risk.** Source-disjoint crops can still come from duplicate scans, related wells
or the same printed motif. Repeated quilting reuses pixels. Catalogue construction
and prior tuning may already have seen held-out source logs. A truth-assisted
patch classifier would get unrealistically easy boundary information.

**Check.** `split.json` assigns every source log globally to one pool, including
across pids. Test pixels, donor legend and real overlay assets come only from
test sources; fitted statistics exclude those sources. Singleton patterns stay
in training and explicitly have no test coverage. A default mixed run refuses
a legend type with no usable test pool. Check duplicate-image hashes and well
identities manually; group by well/scan lineage for a stronger future split.
Fixed classifier patch grids never use truth contacts.

**Look at.** Source lists per bed, `unavailable_test_patterns`, source-log counts,
and all earlier training sets/caches. Any shared source log across pools is an
immediate failure. Five beds of a pattern are not five independent observations.
Fewer than three independent test sources per family is exploratory evidence.

## 10. Numbers that should make us distrust this benchmark

**Risk.** A polished report and narrow intervals can turn conditional synthetic
measurements into an unjustified claim about real logs. Overall means are equal
fake-log means, not dataset-volume-weighted estimates. The square-root allocation
and rare/variety weighting deliberately alter the target distribution.

**Check.** Compare strategy ranks and absolute scores to a blinded real holdout,
report source counts and unavailable-method coverage, and repeat on fresh source
libraries. The 95% intervals bootstrap fake logs conditional on this library;
they are not confidence intervals over real wells. One-log groups have no CI.
Paid methods replay capped offline responses only; an unavailable method is NA,
not a paid-model accuracy estimate.

**Look at.** Synthetic versus real accuracy differences above 10 points, rank
correlation below 0.5 across several viable strategies, >5-point changes between
reasonable source splits, >5% method failures, implausibly perfect boundary F1,
or very narrow CIs from one reused source. Treat these as investigation triggers,
not universal statistical cutoffs. The supplied six-log D-582 mock has only one
held-out source and three rock labels; poor or excellent scores cannot establish
which full pipeline is best.

Before approving use for tuning, review source provenance, the native previews,
the alias collisions, pattern/source coverage and a truly independent real-log
comparison. Preserve that real holdout until after the synthetic design is frozen.
