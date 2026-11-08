"""Render the report from the saved analysis tables."""
from pathlib import Path
import json
import pandas as pd
HERE=Path(__file__).resolve().parent
c=pd.read_csv(HERE/'correlations.csv')
f=pd.read_csv(HERE/'family_correlations.csv')
s=pd.read_csv(HERE/'domain_summary.csv').set_index('source')
means=pd.read_csv(HERE/'strategy_means.csv')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def ci(row,kind):
    return f"{row[kind]:.2f} [{row[kind+'_lo']:.2f}, {row[kind+'_hi']:.2f}]"

def correlations(comparisons):
    rows=[]
    for comparison,label in comparisons:
        part=c[(c.comparison==comparison)&(c.source=='all')&(c.bootstrap_unit=='log')]
        for _,r in part.iterrows():rows.append([label,r.metric,ci(r,'spearman'),ci(r,'kendall')])
    return table(['Proxy','Metric','Spearman ρ [95% CI]','Kendall τ-b [95% CI]'],rows)

parts=['''# Does outside performance predict Viridien performance?

Analysis date: 2026-11-08 UTC. Only the fixed outside **dev** split was evaluated. No external test images, test truth, test predictions, or removed split were used. No model, threshold, or strategy was selected using Viridien eval truth.

**Verdict: the outside set is a useful partial proxy for clustering, a tentative proxy for row accuracy, and a poor proxy for boundary F1 or the combined score.** It provides more evidence of transfer than synthetic swatch columns. The strongest visual match is to the higher-resolution colour-and-pattern logs, especially Critter Creek and Rio Nueces. Evidence for the common narrow, low-resolution offshore family is weak. Outside results should not replace the Viridien evaluation.

## Correlations

These are correlations **across strategies**, between their mean window metrics. Each metric retains the harness definition; `score` is the mean of per-window `row_acc × boundary_f1`, rather than the product of the two overall means.

The complete outside roster is **8 strategies**: b01, s01, s02, s03, s04, s07, s11, s12. The complete synthetic roster is **8 strategies**: b01, s01, s02, s03, s04, s05, s07, s11. These two rosters differ, so use the matched comparison below when comparing the proxies directly.
''',correlations([('ext_dev','Outside dev (8)'),('synthetic','Synthetic (8)')]),'''
The matched roster is **7 strategies**: b01, s01, s02, s03, s04, s07, s11. This removes s12, which has no archived synthetic run, and s05, whose DINOv2 weights are unavailable on this VM.
''',correlations([('ext_dev_common','Outside dev (7)'),('synthetic_common','Synthetic (7)')]),'''
![Rank correlations with log-bootstrap confidence intervals](rank_correlations.png)

Clustering is the most consistent positive result: on the matched roster, outside ρ = 0.64 [0.11, 0.93], versus synthetic ρ = 0.14 [−0.14, 0.50]. Row accuracy is less secure: outside ρ drops from 0.69 to 0.57 when s12 is removed, and its matched-roster interval crosses zero. The intervals do not establish that the difference between the two proxies is statistically significant; no direct difference test was performed.

Boundary F1 and combined score do not transfer reliably. For example, s01 scores 0.361 on Viridien versus 0.130 outside, while s04 scores 0.148 on Viridien versus 0.196 outside. High synthetic scores also fail to preserve the target ordering: s04 scores 0.783 on synthetic columns.

![Strategy means on each proxy and on Viridien](strategy_scatter.png)

## Data, scoring repairs, and uncertainty

Outside dev contains **59 windows from 54 page images and 6 wells**. Viridien contains the fixed **13 eval logs**, scored consistently against `gt/silver/B`. Synthetic contains **39 windows: 3 per parent eval log**. Existing Viridien predictions were rescored, with no prediction changes or fitting.

Two archived files were not directly comparable: b01 and s12 were marked `gt=gold` and had only **1 scored log**, despite listing 13 windows. Their saved CSV predictions were rescored against the same 13 silver/B files as the other strategies. All nine requested strategies now have 13 comparable in-domain results in `per_window_metrics.csv`.

The archived s07 synthetic run contained only **9 windows from the three monochrome logs**. A complete 39-window CPU run was generated in this lane, using the unchanged swatch-training strategy and synthetic generator. No silver labels were used for its training. The other synthetic results are archived runs. Synthetic patterns and target images share the same parent legends, so synthetic is not an independent domain.

Each correlation uses **5,000 percentile bootstrap replicates**, seed **20261108**, with average-rank Spearman and tie-corrected Kendall τ-b. For outside comparisons, source page logs and target logs are resampled independently; all windows from the same page stay together. For synthetic comparisons, parent logs are resampled **jointly with their target logs**, keeping all three synthetic columns together. Strategies remain fixed. Undefined bootstrap correlations are excluded and their valid counts are saved. Confidence intervals describe this fixed strategy roster and these sampled logs, rather than arbitrary future algorithms or sources.

No cross-validation or fitting was performed: this is an evaluation of fixed strategies. Well grouping is used in the sensitivity analyses, rather than treating the dev windows as 59 independent wells.

### Well and blank-window sensitivity

The well bootstrap resamples all windows of each outside well together while retaining the original macro-window weighting. The target is still resampled by log.
''']
well=c[(c.comparison=='ext_dev')&(c.bootstrap_unit=='well')]
parts += [table(['Metric','Spearman ρ [95% well CI]','Kendall τ-b [95% well CI]'],[[r.metric,ci(r,'spearman'),ci(r,'kendall')] for _,r in well.iterrows()])]
parts += ['''
Giving each of the six wells equal weight changes outside Spearman correlations to **0.69 row accuracy, 0.12 boundary F1, 0.33 clustering, and −0.05 score**. Thus, clustering strength depends on the source/well mixture. Omitting one outside well at a time gives ρ ranges of **0.62–0.71 for row accuracy**, **0.05–0.31 for boundary F1**, **0.26–0.81 for clustering**, and **−0.05 to −0.02 for score**. The overall score conclusion is stable under this check.

The Neptune Viridien window has no filled truth. Removing it leaves all four overall outside point correlations unchanged. Its family-specific correlation is undefined because the strategy scores tie. Three monochrome target logs are related D-582 scans; their log-bootstrap intervals do not capture uncertainty across independent monochrome wells.

## Outside source breakdown

Window counts: KGS 10 (9 pages, 2 wells), Idaho 6 (2 pages, 1 well), NL 22 (22 pages, 2 wells), Utah 21 (21 pages, 1 well). Each source below is compared with the complete 13-log target, using the eight available outside strategies. CIs resample page logs and target logs. In particular, Idaho and Utah intervals do not measure generalisation to new wells within those sources.
''']
for source in ['kgs','idaho','nl','utah']:
    part=c[(c.comparison=='ext_dev')&(c.source==source)&(c.bootstrap_unit=='log')]
    parts += ['\n### '+source.upper()+'\n',table(['Metric','Spearman ρ [95% CI]','Kendall τ-b [95% CI]'],[[r.metric,ci(r,'spearman'),ci(r,'kendall')] for _,r in part.iterrows()])]
parts += ['''
Utah carries the clearest clustering signal (ρ = 0.76), but it is one well. Idaho boundary ordering is reversed (ρ = −0.31); even its Spearman interval stays below zero, although Kendall's reaches zero. Source differences argue against a single pooled number as a complete description.

![Per-source mean metric values for every available strategy](source_scores.png)

The machine-readable per-source means are in [source_means.csv](source_means.csv).

## Measured image-domain gap

Colour/pattern measurements use native-pixel strips from the annotated lithology column: up to eight 256-row strips per window. Ink is any pixel with a channel below 245; chromatic ink has channel range greater than 25. **Colour share** is chromatic pixels divided by ink pixels. **Pattern share** is the complementary achromatic ink share; it is a reproducible ink statistic, not semantic detection of every geological pattern. White space is excluded from this denominator. These fixed descriptive definitions were not fitted against accuracy.

Legend entries are parsed entries flagged as lithology by the shared legend parser. They describe the usable legend API, rather than the number of rock types actually present in the window. Complete bed thicknesses come from exact outside truth and imperfect silver/B target truth. Beds crossing a window edge are omitted. Table values for width, resolution, colour share, legend size, and bed thickness are **medians across windows**; the bed columns are medians of per-window median complete-bed thicknesses.
''']
rows=[]
for source in ['viridien_eval','kgs','idaho','nl','utah']:
    r=s.loc[source]
    rows.append([source.replace('viridien_eval','Viridien eval'),f'{r.px_per_ft:.2f}',f'{r.width_px:.0f}',f'{100*r.colour_ink_share:.1f}%',f'{100*r.pattern_ink_share:.1f}%',f'{r.legend_entries:.0f}',f'{100*r.one_bit_share:.1f}%'])
parts += [table(['Set/source','px/ft','Width px','Colour ink','Pattern ink','Legend entries','1-bit pages'],rows)]
rows=[]
for source in ['viridien_eval','kgs','idaho','nl','utah']:
    r=s.loc[source];rows.append([source.replace('viridien_eval','Viridien eval'),f'{r.median_bed_px:.1f}',f'{r.median_bed_ft:.2f}',f'{100*r.legend_truth_coverage:.1f}%'])
parts += ['\n',table(['Set/source','Bed px','Bed ft','Truth rows nameable by parsed legend'],rows),'''
Outside resolution is 7.5001 px/ft throughout, taken from the exact conversion manifest. Target resolution uses the existing OCR depth calibration and has not been independently hand checked. The target is multimodal: the seven narrow coloured logs are about **2 px/ft**, D-582 monochrome logs about **20 px/ft**, Critter Creek **7.5 px/ft**, and Rio Nueces **10 px/ft**. The outside scale is therefore much closer to the last two than to the dominant narrow offshore family. The 13-log median alone hides this difference.

The scorer permits up to 20 pixels of boundary mismatch. At 7.5 px/ft, that is about 2.7 ft; at 2 px/ft it is 10 ft. Short outside windows use a smaller tolerance. Native-pixel tile sizes, rule detectors, minimum bed lengths, and boundary matching therefore operate at different physical scales across the domains.

The nameable-row statistic checks whether each truth lithology occurs among parsed legend labels under the unchanged harness normalisation. It is a label-vocabulary support measure, not actual model accuracy or a strict ceiling for strategies allowed to emit free text. Idaho's low coverage reflects its large, abbreviated, colour-subtyped legend. On Viridien, silver `Unknown` spans and OCR spelling differences also affect row accuracy. This is one reason ARI, which ignores rock names, can transfer differently from row accuracy and score.

Across **all 51 Viridien logs**, bounded image sampling gives median column width **63 px**, median colour share **73.4% of ink**, and **9/51 (17.6%) 1-bit pages**. Eval alone has 3/13 (23.1%) 1-bit pages. Only the 13 eval logs had cached legend/depth measurements; the all-51 depth and legend entries in the CSV are based on those 13 and must not be read as full-51 coverage. Full-51 beds were not inferred from predictions.

No outside dev page is stored as 1-bit. Some have gray or monochrome-looking patterns, but RGB storage does not exercise the same image-loading and mode-dependent augmentation paths as native 1-bit target scans.

![Image statistics and bed thickness distributions](domain_gap.png)

## Layout families and source resemblance

Families below are descriptive image-layout strata, not fitted classes: native 1-bit hatch scans; the blank Neptune column; coloured logs at 5 or more px/ft; and the remaining narrow coloured logs at about 2 px/ft. Their definitions use image measurements only. Family comparisons reuse the same eight outside strategies and are exploratory; they do not establish performance on unseen wells.
''']
family=f[(f.source=='all')&(f.target_family!='blank column')]
rows=[]
for name,group in family.groupby('target_family',sort=False):
    for _,r in group.iterrows():rows.append([name if r.metric=='row_acc' else '',str(r.n_target_logs) if r.metric=='row_acc' else '',r.metric,ci(r,'spearman'),ci(r,'kendall')])
parts += [table(['Target family','Logs','Metric','Spearman ρ [95% CI]','Kendall τ-b [95% CI]'],rows),'''
The high-resolution colour family has the strongest row-accuracy ordering (ρ = 0.93), but consists of only Critter Creek and Rio Nueces. Its score correlation is small despite a positive conditional interval. The narrow offshore family has uncertain row/ARI correlations and effectively no score or boundary ordering. Monochrome ordering looks positive, including score, but the three related D-582 logs provide little evidence of generalisation across monochrome layouts.

![Outside rank transfer by target image-layout family](family_correlations.png)

| Outside source | Closest in-domain layout families | Remaining mismatch |
| --- | --- | --- |
| KGS | Conventional colour-plus-glyph columns; blue brick and coloured mineral fills resemble Rio Nueces and parts of Critter Creek. | Column widths vary; some sections use multiple side-by-side glyphs/overprints. Its 20-entry flat-swatch legend differs from several target grids. |
| Idaho | Predominantly gray dash/hatch fills with thin coloured beds resemble the gray 1-inch TVD and MC761 columns. D-582 shares some line motifs. | About 7.5 px/ft rather than 2 or 20; RGB storage; 59 parsed legend entries and many colour-qualified abbreviations. It is not a native 1-bit surrogate. |
| NL | Yellow dots, gray dashes and other repeated glyphs resemble Critter Creek and Rio Nueces; Critter Creek has the same physical raster scale. | Flat Bay and Storm are two wells, not a broad source sample; grayscale and coloured sections coexist. |
| Utah | Saturated colour-plus-pattern columns, grid/brick marks and narrow subtype bands resemble the higher-resolution colour family; motifs overlap the offshore coloured family. | One well, a 43-entry legend, mixed-width beds, and much greater resolution than the narrow offshore columns. |

The numerical neighbours use log width, log pixels/ft, chromatic ink share, and total ink share, standardised by the target eval distribution. They are descriptive neighbours, not an accuracy-trained model. Legend entry count was omitted from neighbour distance because legend complexity is shown separately. Source examples are selected by closeness to their median colour/ink shares; the displayed 900-row crop comes from the middle of that window. This can only illustrate part of a heterogeneous page.

![Outside columns beside numerical Viridien neighbours](column_examples.png)

The wide, flat outside legend swatches are a material layout difference. S12 returns empty predictions on all outside dev windows, with no runtime exception. Its swatch detector accepts aspect ratios only between 0.7 and 3.5, while many outside swatches are long, thin rectangles. That is a real layout failure of the fixed strategy, retained in the full outside roster. The matched seven-strategy analysis tests whether that failure drives the overall conclusion.

![Representative outside legends](legend_examples.png)

The outside set provides no direct coverage of the empty Neptune layout, and no native 1-bit pages. It is most defensible as a **cluster-consistency and colour/glyph robustness check**, particularly for the higher-resolution colour family. Its row-accuracy rankings are encouraging but vocabulary- and roster-dependent. Boundary and final-score improvements outside should not be assumed to improve Viridien, especially its narrow offshore columns. Synthetic columns are useful controlled fixtures but showed no convincing rank-proxy value for these metrics.

## Execution and reproducibility

The nine requested offline strategies were attempted. DINOv2 s05 could not load locally cached weights (`local_files_only=True`); its failed outputs remain under `runtime/out/s05_dinov2/ext_dev`, but they are excluded from external ranks. This is an environment failure, rather than a measured zero-performing algorithm. S12 completed successfully and is retained. B01's interval `run()` made zero API calls despite the module's box-locator API tag.

The initial shared depth harness was version 1. Another lane updated it to version 2 during execution. A copy of the v2 harness was saved under `frozen/harness`, and all three earlier preprocessing failures were retried for b01, s01–s04, and s07. All eight available external strategies ended with 59 scored windows and no errors. These strategies do not use `ctx['calib']` to choose intervals, so successful earlier row/boundary/cluster predictions remain comparable; the changed depth calibration does not enter the four reported metrics. The code snapshot and result-input SHA-256 hashes are recorded for audit.

All writable artifacts are under this transfer lane. The runner redirects its output/cache root, caps PyTorch at three CPU threads (including s12's internal request for six), and disables model downloads. External runs peaked below 0.8 GiB sampled RSS; the fresh synthetic run remained below 1.6 GiB. The separate all-51 image-statistics process measured **3.64 GiB maximum RSS** with `/usr/bin/time -v`; see `domain_stats.log`. The commands below run sequentially to keep memory use bounded. The analysis does not compare laptop GPU timings with VM CPU timings.

Run from the repository root with the environment's Python. The dataset defaults to the repository's sibling `dataset/`; override `LITHOLOGY_VIRIDIEN_DATASET` if needed. All paths in code are derived from the repository or environment.

```bash
python bakeoff/analysis/transfer/run_external.py
python bakeoff/analysis/transfer/run_external.py --synthetic
python bakeoff/analysis/transfer/domain_stats.py
python bakeoff/analysis/transfer/analyse.py
python bakeoff/analysis/transfer/write_report.py
python -m pytest bakeoff/analysis/transfer/test_analysis.py -q
```

Completed external runs are skipped on restart; `--retry-errors` retries failed windows. The synthetic mode runs the complete s07 set. Local DINO weights would be needed to add a successful s05 outside run. No external test evaluation is performed by these scripts.

Validation: four tests pass, covering agreement with SciPy rank correlations including ties, undefined constant rankings, correct clustered bootstrap weighting, and undefined confidence intervals. Result audits check all 59 dev windows and the complete 13-log target/39-window synthetic cohorts. Charts were visually inspected.

Raw tables: [per-window metrics](per_window_metrics.csv), [strategy means](strategy_means.csv), [all correlations](correlations.csv), [family correlations](family_correlations.csv), [image statistics](image_statistics.csv), [individual bed statistics](bed_statistics.csv), [well omission analysis](leave_one_well_out.csv), [run audit](audit.json), and [input provenance](provenance.json).
''']
(HERE/'REPORT.md').write_text('\n\n'.join(parts),encoding='utf-8')
