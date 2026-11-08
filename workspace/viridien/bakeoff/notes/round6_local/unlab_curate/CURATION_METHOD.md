# Geometry and source checks

This is a TEST set. Do not train or tune on the pages or their results.
There are no rock labels. The boxes mark image regions only.

The curation script scores every available page. It compares cheap image
features with 38 challenge log crops. All 38 logs are outside the 13-log
evaluation list. It reads no challenge rock truth. The features include
column width, ink, colour, near-binary pixels, edges, rules and palette size.
The printed source key supplies a second image comparison.

The ordinal source score has the shared 0–3 meaning. No source has a confirmed
score of 3. The continuous score combines a fixed source-style prior with
image distance. It is a ranking aid. It is not a probability or an accuracy
estimate. If semantic regions are unknown, a whole-page feature score is
reported with zero verified usability. Such a page cannot enter the set.

Selection uses fixed family budgets. It favours Gulf, Texas and MC765 styles.
It caps repeated wells. The page hash flags a possible duplicate. A small
whole-page pixel difference must also confirm duplication before rejection.
The selection uses no model output. Blank lithology strips are rejected.

Box detection starts with a source-specific search region. Persistent rules
refine its edges. A sparse depth channel helps locate an adjacent hatch
track. New Oregon headers use printed symbol frames to find the key.
Automatic detections are not accepted without a visual check. Every accepted
page has a contact-sheet check of the full page, lithology, depth and key.
The review record stores page identifiers and sheet hashes. The freeze check
rejects an altered sheet, bad box, missing page, exact duplicate or banned well.

Some body pages lack a key. Their actual source key is placed below the page.
The source crop, rotation and output offset are stored in the annotation.
The body pixels and their coordinates are retained. No rock name is invented.
Each page supplies one fixed window of at most 4,500 rows.

The layout names need care. Historic onshore logs and the Canadian offshore
log are Gulf column-style analogues. The colour geothermal logs are MC765
style analogues. DHI is a Geoservices-style analogue, while the Norwegian
pages carry a printed Geoservices key. The MC761 group contains LWD mineral
volume logs. It is a separate stress subset. These outside pages cannot
establish accuracy on the seven challenge families.

Public access does not establish a reuse licence. The manifest retains
source URLs and the source licence note. Check these notes before sharing
the image files.

## Frozen release and VM handoff

The frozen release has 130 pages and 130 windows. It covers 36 well-name
groups. The curation run scored 1,007 pages. All 130 selected annotations
were checked on 33 contact sheets. Enlarged crops also checked the depth
digits on the 12 new monochrome pages. The freeze check found no exact
duplicate image and no banned well.

The family counts are Gulf 60, Texas 13, MC765 25, One-off 5,
Geoservices 8, D-582 12 and MC761 7. These are layout guesses.

The eval-set SHA256 is:
`55a2961965557fe334524d40b8e99459933e3bcef6852bbc840ebbe15d909365`.

This release already contained 43 round-6 pages when it was frozen:
27 Mizzen pages, four Norway pages and 12 partial-bulk Oregon pages.
The owner then cancelled the late fold. Do not add, remove or replace pages
in this pinned release. The parent will build a separate supplementary set
on the boat VM. Use `round6_already_in_frozen.csv` to check overlap.

The owner moved the heavy run to the boat VM. Do not restart local image
processing or inference. Do not change the pinned dataset, runner, scan
script, metric module, bench file or model files. The parent will return
the clean and medium-scan outputs to this lane's `runs/` directory.
