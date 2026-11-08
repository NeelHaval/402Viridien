# 1. Verdict

Useful for pattern experiments and controlled column tests; not yet fit to claim exact-ground-truth, end-to-end training or evaluation on real pages.
The biggest problem is the conversion of incomplete observations into authoritative synthetic labels and bed boundaries.
Fix label validity and legend correspondence first, then sampling and independent evaluation, before spending more time polishing tiles or scan noise.

# 2. Findings, ranked by impact

Audit date: 2026-11-08. Commands: `python -B -X utf8 data/genlab/critique/a2-e2e/audit.py` and `python -B -X utf8 data/genlab/critique/a2-e2e/visuals.py`. Both completed. The numerical audit peaked at approximately 66 MiB working set. Images were rendered and inspected at native 1:1 scale; real TIFF access used bounded windows. Paths below are workspace-relative.

## 1. Unknown annotation coverage becomes a supervised lithology-gap class

**Severity: blocker.**

**Evidence:** `stats.py` collects every unknown run as a gap (`genlab/assemble/stats.py:121`). `Sampler.gap()` samples those lengths (`genlab/assemble/compose.py:298`), and composition pastes unlabelled real pixels while leaving the mask zero (`genlab/assemble/compose.py:504`). Zero explicitly means `__gap__` (`genlab/assemble/compose.py:526`). The source selector uses the longest uncovered annotation interval, without establishing that it is physically blank (`genlab/assemble/noise.py:580`).

In `out/starter/page-texas-rock-types-box-d368760651ded938-01`, **12,528/38,251 rows, 32.75%, are labelled gap**. My 512-row crop at column rows 2664–3176 has a **100% zero mask**, yet visibly contains orange/yellow lithology motifs. Across the 20 starter images, 80,271/446,172 rows carry this gap class. Inspected image: `data/genlab/critique/a2-e2e/gap-truth.png`.

**Fix:** Separate `unknown/ignore`, confirmed blank background, and known lithology in the stats and truth schemas. Stop sampling annotation coverage holes as geological gaps. If unknown source pixels are retained, emit a validity mask and exclude their rows and adjoining uncertain boundaries from losses and scores. Verify that this exact crop contributes zero supervised gap pixels; reserve supervised blank examples for independently confirmed blanks.

**Effort: M. Expected gain:** Removes direct contradictory supervision and restores the meaning of “exact ground truth.”

## 2. The assembler borrows appearances by name, breaking the page’s own legend

**Severity: blocker.**

**Evidence:** When a target lacks a card, `Composer.available()` chooses the alphabetically first same-name pattern in its family, without comparing the target swatch (`genlab/assemble/compose.py:374`). My audit found **42 eligible name-only substitutions**; this is eligibility, not a claim that all 42 were rendered in starter outputs.

For target sandstone `16c9064b18b77b61:3`, the selected donor is `5709fe9e86ce942b:7`. The target legend and accepted real crop are yellow with fine sparse marks; the rendered donor is cream with larger black marks. Measured background RGB is **[255,255,0] versus [255,255,204], distance 204**. A second example, anhydrite `7966a10fba6fa984:11`, has the same red background as its donor but a visibly different hatch. Inspected images: `data/genlab/critique/a2-e2e/substitute-16c9064b18b77b61.png` and `substitute-7966a10fba6fa984.png` in the same directory.

**Fix:** Reuse the lab’s validated visual borrowing rules, including motif, colour and native scale agreement. Withhold incompatible targets or reconstruct them from their own legend. Preserve target and donor IDs plus match evidence. Verify both examples are rejected as substitutions while an actually matching donor remains usable. For deliberate new symbol systems, change legend and column together.

**Effort: S to disable; M to integrate verification. Expected gain:** Prevents teaching the classifier that a document’s printed legend is wrong; removes a failure that extra noise cannot repair.

## 3. “Page-e2e” evaluates classification with the true column location supplied

**Severity: major.**

**Evidence:** The page detector reads `interpreted_lithology_column` from truth annotation and crops that exact box before predicting (`genlab/assemble/bench.py:801`, `genlab/assemble/bench.py:809`). The public scorer compares row intervals, not page coordinates (`genlab/assemble/bench.py:158`). In an executable counterexample, perfect row labels with a deliberately wrong bbox `[999,999,1000,1000]` still received **row accuracy 1.0 and boundary F1 1.0**; the bbox is ignored.

The current page-e2e suite contains **one page**. Starter has two page templates, with column widths **57 and 76 pixels**. Composition preserves original positions and annotations (`genlab/assemble/compose.py:540`, `genlab/assemble/compose.py:604`). In the inspected 436×512 context window, **100% of pixels outside the column were unchanged**, including the old “80% SS” narrative beside replacement lithology. Inspected image: `data/genlab/critique/a2-e2e/page-context.png`. Such context is useful for appearance, but its geological meaning is no longer guaranteed.

**Fix:** Keep this as an explicitly named “known-column classification” arm. Add an inference path receiving only page pixels, with predicted column boxes and legend-entry correspondence; score localization IoU/recall, interval overlap and class accuracy jointly. Split templates by well. Add bounded page transforms that update all boxes, masks and legend geometry together, and test widths/scales drawn from training scans. Export a pixel-to-depth mapping where known: current `real_depth_span` is pixel coordinates (`genlab/assemble/compose.py:530`). Mask or regenerate incompatible geological prose when training models that use surrounding text.

**Verification:** Wrong-position predictions must fail; a translated/rescaled page must retain correct transformed truth. Compare real-only, synthetic-only and mixed training on the same frozen real validation wells.

**Effort: L. Expected gain:** Measures the actual detection task and reveals layout, scale, legend-localization and context failures currently hidden by supplied boxes.

## 4. Example-box heights become bed lengths, and their edges become “true” boundaries

**Severity: major.**

**Evidence:** In the current statistics, **97/97 Texas “Rock Types” layers and 4/4 MC765 layers are censored**. Texas sandstone’s fitted median is **70.5 pixels from 12/12 censored observations**. Censoring is recorded but does not alter the ordinary thickness fit (`genlab/assemble/stats.py:166`); the sampler uses the resulting distribution (`genlab/assemble/compose.py:271`). This estimates annotation-window lengths when boxes are clean interior examples.

Bench also treats every accepted box endpoint as a bed boundary (`genlab/assemble/bench.py:220`, `genlab/assemble/bench.py:153`). My counterexample used a correct A bed spanning rows 0–100, with only its interior 40–60 labelled: **row accuracy was 100%, boundary F1 was 0**, solely because the crop edges were scored as boundaries. Inspected visual context: `data/genlab/assemble/out/starter/sheet-5fbeaf2a0650e013.png`; real and synthetic columns share motif scale but have markedly different sequences. The sheet alone cannot establish true bed lengths.

**Fix:** Add independent top/bottom boundary-confirmation flags and distinguish example boxes from complete intervals. Fit exact lengths from boundary-confirmed beds; use censored observations as lower bounds only through an appropriate censored fit. Withhold calibrated thickness claims where none are complete. Score real boundary F1 only in regions with confirmed boundaries, including confirmed absence of boundaries.

**Verification:** The interior-box counterexample must have an unscored boundary result, not F1=0; an all-censored family must not pass as a measured complete-bed prior.

**Effort: M. Expected gain:** Avoids synthetic over-segmentation and misleading boundary metrics; concentrates annotation effort on complete adjacent intervals.

## 5. Composition fallback confuses row prevalence with bed-selection probability

**Severity: major.**

**Evidence:** Stats define composition as weighted row shares (`genlab/assemble/stats.py:657`). The fallback uses those shares directly to choose the next bed (`genlab/assemble/compose.py:251`) and then independently draws its thickness (`genlab/assemble/compose.py:321`).

Using the actual `Sampler`, I supplied two classes with **50/50 target row shares**, deterministic lengths **10 and 100 pixels**, no transitions and no gaps. Three 100,000-row runs produced **90.53%, 90.80% and 91.10%** coverage for the long class. The mechanism therefore misses its intended coverage by about **41 percentage points**, even with perfect inputs. The native comparison inspected for this issue was `data/genlab/assemble/out/starter/sheet-5fbeaf2a0650e013.png`; its mixture difference is consistent with several causes, so the controlled experiment, not that appearance, establishes this finding.

**Fix:** For iid fallback, select beds proportional to `target_row_share / expected_bed_length`, using the actual truncated/sampled length distribution. For transition-driven sampling, check the resulting stationary row occupancy and adjust duration/transition models together when matching composition is required. Add the two-class diagnostic and a per-zone generated-versus-target occupancy check.

**Effort: S for fallback; M for transition calibration. Expected gain:** Corrects a large, deterministic composition error before any texture work; the controlled example should approach 50/50.

## 6. Assembler validation is held out after its evidence has already entered generation

**Severity: major.**

**Evidence:** All **seven dataset-validation documents** occur in assembler generation statistics; **131/354 noise measurements** come from those validation documents. All **five reference documents** in starter’s fidelity validation also occur in the generation statistics. Bench excludes wells found in document-level usage, but not contributors to pooled family/global priors or noise (`genlab/assemble/bench.py:249`, `genlab/assemble/bench.py:261`). Its saved report explicitly discloses this limitation; the remaining issue is experimental independence, not hidden reporting.

Inspected image: `data/genlab/assemble/out/starter/sheet-5fbeaf2a0650e013.png`, a validation-well template reused for composition. The v5 crop evaluator has already improved its own validation handling; that does not automatically isolate assembler artifacts.

**Test-well check:** I found **zero test documents** in stats document blocks, zero test source documents in the audited starter/iteration-2 truth, and zero test sources in current noise/gap measurements. However, imported `--stats` files are checked only for schema (`genlab/assemble/compose.py:143`): a compatible aggregate containing test-derived priors has no enforced contributor-split check. This is an ingestion risk, not observed current contamination.

**Fix:** Freeze train/validation/test well assignments before fitting patterns, durations, transitions, residuals and page templates. Record every contributing well and the split-policy hash in each artifact; reject prohibited contributors when loading external stats. Rebuild assembler validation artifacts from permitted training evidence. Keep test results out of promotion/tuning decisions.

**Verification:** Every validation-generation artifact must have an empty intersection between contributor wells and validation/test wells. A deliberately test-tagged stats fixture must be rejected before sampling.

**Effort: M. Expected gain:** Makes improvements in synthetic realism and real detection utility interpretable and closes a route for future test leakage.

# 3. Quick wins

Each item is a bounded change estimated under one hour; full downstream rebuilds are separate.

1. Disable unverified `family-substitute` selection and report withheld class coverage.
2. Export unknown pixels as ignore/invalid; remove them and their uncertain edges from supervised gap scoring.
3. Correct iid composition weights for expected duration; retain the 50/50 diagnostic.
4. Rename the present page metric to “known-column classification” and expose page/template counts.
5. Mark all-censored thickness fits unusable as exact bed priors and display complete-bed counts.

Then implement generation-wide validation isolation and the page localization arm. This order fixes label and distribution errors before paying for more pattern remakes or noise tuning.

# 4. What is already good

Native motif scale, bounded page reads, explicit pattern provenance and current test-well exclusion are valuable; preserve them.
V5’s within-document task is much better aligned with the intended classifier: augmented matcher **60/79 correct, 75.95%**, logistic **58/79, 73.42%** (`data/genlab/datasets/v5/eval.md:6`).
The inspected Hadrian sheet already reproduces convincing native motifs; better-looking individual tiles are not the only remaining bottleneck.

# 5. Doubts

- Code and artifacts were changing during this audit; line references describe the versions read. This is not an atomic snapshot or an audit of every historical artifact.
- I inspected two existing native comparison sheets and rendered gap, page-context and two substitution comparisons. I did not inspect all 51 pages or all 42 eligible substitutions, and never decoded a whole TIFF page.
- Starter had fidelity and realism results but no saved detector result when checked. The missing utility result is unfinished work, not evidence that detector execution fails. I did not run a model-training ablation or the full detector benchmark.
- Physical depth calibration, mixed lithology proportions and complete real boundary truth were not independently recovered. Existing depth tracks can supply page context without establishing a correct physical-depth training target.
- No measured downstream gain is claimed for the proposed fixes. V5’s augmented discriminator is **56.96%** (`data/genlab/datasets/v5/eval.md:53`), but that crop result cannot validate full-page detection or the correctness of synthetic interval truth.
