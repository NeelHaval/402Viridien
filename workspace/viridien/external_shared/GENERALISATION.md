# How useful are the outside logs for judging our Viridien model?

Short answer: **useful, but for specific things.** They are the only data with exact truth, so they are the best check we have of **boundary placement**, **matching unseen legends**, and **crash robustness**. They closely resemble one of our seven layout families: the Texas "Rock Types" logs, which are about 19% of the challenge's column rows. They say little about the Gulf of Mexico 3-panel logs, which are 59% of rows. Treat them as a robustness and regression test, not as a stand-in for the Viridien score.

Measured 2026-11-08 on the outside dev+test splits (9 wells, 79 windows). The verification split stays sealed.

## 1. What carries over

| Same as the challenge | Why it matters |
|---|---|
| Files and boxes (TIFF + `depth_column` / `interpreted_lithology_column` / `lithology_legend`) | Solutions run unchanged; no format adapter to hide bugs in |
| Task: segment the column, match each bed to a printed legend swatch | Tests the same two skills we are graded on |
| Same scorer (row accuracy, boundary F1 with the same tolerance, cluster ARI) | Numbers are on the same scale |
| Legends we never trained on | A real test of generalising to new legends, which the challenge's hidden logs will also need |
| **Exact** truth from the PDF vector drawing | Our in-domain truth is hand-labelled (silver). Outside, a missed boundary is truly missed |

## 2. What does not carry over (the domain gap)

**Layout families.** Every outside page is a born-digital strip log with a "Rock Types"-style legend box.

| Our family (51 challenge logs) | Logs | Share of rows | Outside logs that resemble it |
|---|---|---|---|
| Gulf of Mexico 3-panel | 18 | 59% | none |
| Texas "Rock Types" box | 7 | 19% | **all four outside sources** |
| D-582 1-bit grid | 9 | | none |
| Geoservices grid | 5 | | none |
| One-off layouts | 5 | | partly |
| MC765 colour grid | 4 | | none |
| MC761 mineral/lithology | 3 | | none |

**Bed size.** The outside pages are rendered at 7.5 px/ft. Their beds are much thinner in pixels than ours, so boundary F1 is harder outside than inside.

| Set | Median bed height (px) | 10th percentile (px) | Beds under 5 px |
|---|---|---|---|
| Outside, Idaho | 12 | 6 | 7% |
| Outside, Kansas | 30 | 15 | 0% |
| Outside, Newfoundland | 48 | 12 | 0% |
| Outside, Utah | 60 | 15 | 0% |
| Challenge, Gulf 3-panel | 149 | 21 | |
| Challenge, Texas Rock Types | 65 | 24 | |

The converter controls the render resolution. Re-rendering at a Gulf-like pixels-per-foot would close this gap; that's an option, not done yet.

**Image quality.** Born-digital pages have no scan noise, skew, fading or 1-bit dithering. The outside set therefore **overestimates** robustness to scanning problems. Many challenge logs are scans.

**Legend text.** Outside labels are abbreviations (`Lmst`, `Sltst`, `Clyst dkgy`). Our legend parser read 116 of 246 label names (47%), so outside row accuracy partly measures abbreviation reading. Challenge legends mostly use full names.

## 3. Results so far (current best: s11_fft_legend)

| Data | Windows | Row accuracy | Boundary F1 | Cluster ARI |
|---|---|---|---|---|
| Challenge eval, 13 logs (silver truth) | 13 | 0.37 | 0.47 | 0.63 |
| Synthetic | 39 | 0.86 | 0.85 | 0.91 |
| Outside dev (6 wells) | 59 | 0.41 | 0.35 | 0.64 |
| Outside test (3 wells) | 20 | 0.07 | 0.15 | 0.44 |
| This shared subset | 17 | 0.32 | 0.16 | 0.56 |
| Outside Utah only (closest to Texas) | 21 | 0.49 | 0.53 | 0.80 |

What this shows:

1. **Synthetic data flatters us** (0.86 row accuracy vs 0.37 on real logs). The outside set is far closer to the real difficulty.
2. **Differences between wells dominate.** Dev and test differ by 0.34 in row accuracy only because different wells landed in each split. With 3 to 6 wells per split, one well moves the mean a lot. Compare models with **paired per-window differences** and **well-level bootstrap intervals**, not single averages.
3. **Both sets give the same verdict on legend detection.** Emil's model with its own legend detection scores about 0 row accuracy in both (it reads few or no swatches). Given the true legend, it reaches 0.83 on an Idaho page. Both sets agree that legend reading is the bottleneck.
4. **Crashes show up only outside.** 11 of 89 outside windows failed on depth reading (metric units, sparse labels), against 0 of 13 in-domain. That's a real robustness bug the challenge logs would not reveal.

## 4. Verdict and how to use it

- **Use it for:**
  - boundary-detector development (exact truth);
  - legend-matching generalisation to new legends;
  - crash and robustness regression tests;
  - any decision about the Texas-like family.
- **Do not use it for:**
  - predicting the Gulf 3-panel score;
  - tuning to scanned-image noise;
  - claiming an absolute Viridien score.
- **Weigh decisions.** When the outside set and the challenge eval disagree, trust the challenge eval for anything family-specific. Trust the outside set for boundary precision, because the in-domain truth is hand-made.
- **Still to measure.** A rank-correlation test is running: 9 strategies on outside dev vs the challenge eval. It will say whether "better outside" means "better on Viridien". Until it reports, transfer is plausible, not proven.
- **To make it more representative:**
  - re-render at challenge-like pixels per foot;
  - add Gulf Coast logs (a Mississippi Gulf Coast download lane is running);
  - add scan-style noise to copies of the pages.
