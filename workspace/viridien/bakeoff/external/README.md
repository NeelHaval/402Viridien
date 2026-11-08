# KGS WellSight vector strip logs

`kgs_extract.py` builds extra test logs from reusable embedded rock images,
their **visible PDF clipping rectangles**, and vector depth labels. No OCR or
image classifier supplies truth. The sample is treated as untrusted data:
PyMuPDF opens and renders it; pypdf only parses content/resource bytes supplied by
PyMuPDF. Nothing in the PDF is executed. There are no network calls.

From `bakeoff`, with installed Python packages PyMuPDF (`fitz`), pypdf, NumPy
and Pillow:

```powershell
python -I -X utf8 external/kgs_extract.py --pdf-dir ../external_untrusted/kgs --out ../external_data/kgs
python -I -X utf8 external/kgs_extract.py --pdf-dir ../external_untrusted/wellsight --out ../external_data/kgs/round2
python -I -X utf8 external/kgs_extract.py --pdf-dir ../external_untrusted/kgs --out ../external_data/kgs/at_005 --ft-per-px 0.05 --limit 1
python -I -X utf8 external/test_kgs_extract.py
```

All output paths are checked against the resolved `external_data/kgs/` root;
paths outside it are rejected. Code belongs only in `bakeoff/external/`.
The source PDFs, dataset and other project files are read only. Tests generate
fixtures and results under `external_data/kgs/tests/`, and disable bytecode
writes. `-I` isolates Python from the working directory and user environment.

The automatic resolution is the median positive `ft_per_px` of rows whose
`family` contains `Texas` in `analysis/volume_logs.csv`. The seven current
measurements give **0.1333316415486851 ft/px**, rather than the approximate
0.05 mentioned in the request. When those measurements are unavailable,
0.05 is the fallback. `--ft-per-px` overrides this. Render scale is fitted
ft/pt divided by target ft/px; `meta.json` stores the exact scale and DPI.
PNG DPI metadata is rounded to an integer, without changing pixel geometry.

## Outputs and conventions

A one-page source produces `<out>/<pdf-stem>/` with:

- `image.png`: the entire page, at the target depth resolution.
- `boxes.json`: real dataset schema: `image_name`, `width`, `height`, and
  `bounding_boxes`. Its box keys are `depth_column`,
  `interpreted_lithology_column`, and `lithology_legend`. Coordinates are
  absolute image pixels, in `[x0, y0, x1, y1]` order.
- `truth.csv`: `top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id`.
  Image intervals are absolute, half-open `[top_y, bottom_y)`. Each boundary
  is rounded once to the nearest image-row boundary. The rows tile the whole
  plotted track, including blank margins. Blank has cluster ID `-1`; rock
  cluster IDs index alphabetically sorted legend labels starting at zero.
  If rounding would erase any interval, **all** row coordinates for that page
  instead retain fractional pixel boundaries. `meta.json.truth_conventions.y`
  records this choice. Consumers must preserve these floats: subpixel gaps
  are real PDF geometry, even when invisible at the chosen PNG resolution.
- `meta.json`: SHA-256, all legend entries and swatch paths, sections, point
  and pixel track bounds, depth regression and residual RMSE, exact point
  intervals, width mixtures, accessory placements and interpretation limits.
  `legend_label_aliases` records names sharing an identical image. Such names
  are joined with ` / ` in truth; a unique geological name cannot be recovered
  from those pixels. Original full legend names remain in `legend_entries`.
- `legend_swatches/<label>.png`: rock swatches from embedded image pixels,
  rendered at their original placement size and the page render scale.
  Accessory filenames include their category to avoid label collisions.
- `accessories/xref_<N>.png`: placed accessory images with their transparency
  masks restored. Positions, kind and `over_column` are recorded in metadata.
- `contact.png`: column beside the truth colour strip, split into readable
  panels. Large blank margins are trimmed for this review image only.
- `extraction.json`: per-page success or skip reason, including partial PDFs.

Multiple-page PDFs use `page_001/`, `page_002/`, etc. inside their PDF folder.
The PDF-level `extraction.json` stays in the parent. A page without its own
legend can inherit document legend xrefs; its metadata says so, and a legend
box is omitted because no legend is printed on that page. Several local rock
legends are retained separately; the bbox is their union.

The batch root contains `summary.csv` with one row per PDF (including corrupt
or unsupported PDFs), used rock labels, legend type count, total and logged
interval counts, full and patterned depth extents, and the worst successful
page RMSE. A PDF with some successful pages has status `ok` and explicit
skipped-page counts/reasons. `contact_sheet.png` stacks the successful page
panels, each reduced to at most 1560 pixels wide for a bounded overview.
Individual `contact.png` files retain the detailed panels. Reasons list the
first three distinct failures with counts, plus the count of other failures;
`extraction.json` retains every page's reason. An empty batch still writes a
CSV header and an explanatory sheet.
`--limit` limits the alphabetically sorted, nonrecursive PDF list.

## Extraction details

Legend sections come from geometric positions of `ROCK TYPES`, `ACCESSORIES`
and the track headers. Each swatch is matched to the nearest word on its
right at the same vertical location, independent of extracted text order.
Accessory subcategories follow geometric column order and subsection headings.
Only ROCK TYPES xrefs enter the rock map; `Anhy` in MINERAL or STRINGER cannot
become the rock swatch called `Anhy`.

Inline BI/ID/EI images get stable negative integer IDs: SHA-256 of decoded
pixels, dimensions and decoded color space, with the sign inverted. MuPDF
decodes each distinct inline encoding in an in-memory image object; repeated
encodings are cached. Compression differences do not change identity. The
existing `xref` fields carry these IDs throughout swatches, tiles, accessory
exports, mixtures and metadata. Matching always uses exact identity, never
color similarity. Full multiword labels are preserved. Invisible PDF text
(including Utah's duplicate OCR layer) is excluded from labels and depth fits.

Some Utah Form images cache many swatches in one tall XObject, including
several different rocks. These are split only when **every decoded byte**
belongs to a sequence of complete legend images with the same pixel width
and color space. Source row boundaries are mapped through the placement and
clip, retaining painting order and vertical orientation. `exact_tile_expansions`
records source xrefs, decoded hashes and the exact legend row segments.
An unmatched opaque cache covering the column is rejected, rather than being
reported as blank or accepted from a partial/similar pattern match.

The leftmost printed `Lithology` header chooses the interpreted column when
there is also a `% Lithology` column. Its horizontal position can be inherited
by same-size continuation pages; metadata records it. Without a header,
image placements establish the vertical band with greatest union coverage.
Long drawn borders establish its width and plotted vertical extent. Graphics
state, transformations and rectangular clips constrain each visible tile.
Nested Form XObjects apply their own Matrix, BBox and Resources, composed with
the caller's CTM and clip; painting order remains global. Rectangular paths
encoded with four axis-aligned line segments are also recognized. Curves,
multiple clipping subpaths, rotated/sheared image placements and rotated or
nonrectangular Form clips remain rejected. Page-edge continuation borders can
trim the first fraction of a point; bounds always use the actual drawn lines.
MuPDF's independent placement-count check excludes annotation appearances,
which are outside the page content stream (e.g. Barlow's PTS stamp).
This matters: the sample's images are 43.2 pt wide, but its track is only
36 pt wide, and terminal tiles are clipped at bed contacts.

A sweep over visible rectangles resolves overlaps by PDF painting order and
merges adjacent identical mixtures. Components retain xref and width share,
including uncovered width as `(blank)`. The largest share supplies the CSV
label. Equal shares use alphabetical label, then xref, deterministically.
Accessories are kept separately and do not change underlying rock labels.
Geometry within 0.002 pt is coalesced to remove MuPDF float32 seams; this is
about 0.00056 ft in the sample. If rounding would erase an entire interval,
the page retains fractional pixel coordinates, including tiny blank gaps.

Depth labels are grouped by horizontal position. A monotonic sequence spanning
at least 40% of the plotted height and containing at least three labels is fit
by ordinary linear regression using word-box vertical midpoints. The selected
fit must have RMSE below 2 ft. Full words are used; clipped `42` and `00` at
the sample's ends are excluded. Metadata preserves the observations and fit.
Truth depths are evaluated at original point boundaries, without snapping or
rounding. They therefore differ slightly from nominal round-number depths.

## Sample result and verification

`386825.pdf` has 20 rock legend types. Its column uses **Lmst (39), Shale (43),
Ss (47), Sltst (46)** and yields **122 rock intervals + 2 blank margins = 124
CSV intervals**. There are no side-by-side vector rock mixtures in this sample.
The nominal patterned interval is 4600–5150 ft; the plotted track spans about
4200–5199.4 ft. Bounds are x=212.4–248.4 pt, y=1298.16–4896 pt.

The 19-label depth fit has slope 0.2777778551 ft/pt and **RMSE 0.00014304 ft**.
This measures linear-fit consistency, not absolute geological depth accuracy.
Exact truth here means PDF rock identities and clipped geometric contacts;
depths are calibrated from printed labels and image boundaries are quantized.

The 31 offline tests cover the requested sample assertions, bed-contact and
track clipping, all accessory categories, restored overlay alpha, schema and
image dimensions, mixed width shares, painting order, tie handling, seams,
multiple legends, inherited legends across pages, raster-only skipping, output
path restrictions, batch results and resolution loss. Re-cropped page tile
means match their embedded swatches within 15/255 per RGB channel and identify
the correct nearest swatch; the sample's largest error is 11.59/255. Differences
include antialias phase, border strokes and accessories. Metrics are written
to `tests/sample_colour_errors.json`.
New fixtures cover inline compression-independent identity, byte-distinct
images, swatch and accessory PNGs, inline legend inheritance, nested Forms,
Matrix/Resources/BBox composition, graphics-state isolation, rejected unsafe
clips, line-path rectangles, annotations, aliases, fractional boundaries,
early page filtering, short reasons, hidden OCR, track headers and three-label
depth fits.
Composite-cache tests require full byte equality and reject a one-byte change.
On a short final page, labels farther down the same drawn depth track may
calibrate the rock column above them; no depth values are invented or snapped.

The contact sheet was visually inspected against the truth strip. Its beds
align; blank margins are retained in the actual truth. One separate raster
overlay (xref 132, alpha mask 129) holds stringers and `P` marks in the lower
column. It is preserved as an **unclassified overlay**, rather than assigning
those symbols unsupported rock or accessory names.

## Limits

Raster-only logs and logs without reusable ROCK TYPES tiles cannot provide
this kind of exact truth and are skipped. Text/legend checks run before content
parsing; continuation pages must reuse document legend images. Rotated pages,
rotated/sheared images and nonrectangular image clips remain unsupported.
Inline stencil masks and inline color spaces requiring external resource
streams are rejected; ordinary DeviceRGB/Gray/CMYK and direct Indexed spaces
are decoded by MuPDF. Form recursion is bounded to 32 levels and cycles fail.
Identical images with different printed labels retain explicit alternatives;
their individual geological names are not distinguishable by this method.
Raster overlay collages are not decomposed into named individual symbols.
Truth describes vector rocks under these overlays, not every visible pixel.
No OCR fallback or guessed rock identities are used. A page render or combined
contact sheet over 160 million pixels is rejected; reduce resolution or batch
size. Other WellSight layout variants still need real-PDF validation.
