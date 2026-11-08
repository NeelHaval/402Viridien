# Round 6 exact lithology test windows

This set has 25 windows, 25 page images and 314 truth intervals from one new well: **DJS Properties 2-14**, API 11-075-20023, Idaho. The source has 28 pages. Pages 4–28 supply truth. Header and legend pages do not supply test windows.

Family: **Texas Rock Types box / WellSight**, similarity 3 for the printed ROCK TYPES legend and strip-log track style. This is a layout assessment, not a model score. No Gulf 3-panel or MC765 windows were admitted in this build.

## Files

- `images/`: RGB TIFF page images at 10 pixels per foot.
- `bbox_annotation/`: exclusive pixel boxes for the depth track, lithology track and original legend.
- `truth/`: CSV fields `top_y`, `bottom_y`, `lithology`, `cluster_id`, `top_depth`, `bottom_depth`. Field order follows the existing harness writer.
- `eval_set.json`: one window per admitted page, at most 6000 rows.
- `manifest.csv`: source, well and page for every image.
- `window_manifest.csv`: well, family and test split for every window.
- `split.json`: whole-well assignment. All pages of this well belong to test. Do not train or tune on this well.
- `pixel_audit.json`: pixel checks, interval checks and limits.
- `audit.json`, `extraction_audit.json`, `extraction/`: source hashes, exact tile identities, legend entries, geometry, excluded intervals and reasons for skipped pages.
- `gallery.png`, `gallery/`: source legend and track examples beside truth colours. Gallery captions retain the converter's internal r3 prefix; dataset filenames use r6.

## Source and truth rules

Source: [Idaho Oil and Gas Conservation Commission, DJS Properties 2-14 mud log](https://ogcc.idaho.gov/wp-content/uploads/2017/12/1107520023_DJS2-14_MUD_20140916_PTS.pdf).

Access is public and needs no login. No explicit reuse licence was verified. Keep source attribution. The PDF's historical CONFIDENTIAL stamp is preserved in source pixels.

The established KGS converter extracts placed PDF rock tiles, exact decoded tile identity, clip geometry and the printed legend. It does not infer labels from colour similarity. Original legend pixels from page 2 are added above continuation pages. Boxes and truth rows include that shift. Mixed rows use the largest visible width; equal-width ties use alphabetical order. Ambiguous legend aliases and blank intervals are excluded. Fractional boundaries are rounded to integer raster rows. Depths are fitted to printed PDF depth marks and remain in feet. This truth describes the displayed rock track; it is not an independent geological survey.

## Audit

The audit passed for all 25 windows, 314 intervals and **3,640,809 track pixels**. It compared every exported track pixel with a new raster of the source PDF. It checked each interval's bounds, dominant tile label, legend identity and increasing depth. It also checked non-overlap and the 6000-row window limit. Legend OCR and strategy evaluation were **not run**.

Rebuild and check from the workspace root:

```powershell
python -I -X utf8 viridien_work/bakeoff/notes/round6_local/r6_labelled/build.py C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work
python -I -X utf8 viridien_work/bakeoff/notes/round6_local/r6_labelled/finish.py C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work
python -I -X utf8 viridien_work/bakeoff/notes/round6_local/r6_labelled/audit.py C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work
```

The build uses existing trusted converter modules and writes only within this lane. It does not change `external_tests/` or `external_vault/`.

## Search limits

Twin Peaks 1W lists a mud log but has no download link. The saved round-six candidate scan records the files available during conversion. Oregon samples had no extractable text and one raster image per first page. The New Zealand sample was a human-verification page. Norway completion-log vectors and Canadian final reports need a separate exact-pattern converter; they were not assigned truth by this tile converter. No inference-based intervals were admitted.
