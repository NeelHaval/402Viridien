# Dataset review page

Open `index.html` in a browser. Keep `catalog.js`, `app.js`, `app.css` and the `assets` directory beside it. The page needs no server or network service.

1. Use the family and source filters. Start with “Needs review first”.
2. Review model crops from low confidence and disagreement first. Confidence is not calibrated. Use “Show page”, select the legend region, and check the source before you confirm a crop label.
3. Use “Compare” to pin current, synthetic and outside records. Each pane can show a lithology, depth or legend region. The y slider uses original page pixels. Different pages do not share the same depth at the same y value.
4. Review the technical readiness and recorded issues. A page-level “Accept” is a triage decision. It does not remove a technical blocker or confirm truth labels.
5. Draw an issue box on a comparison image, or enter its original coordinates. Add a note. Use “Flag” or “Exclude” when needed.
6. Use “Save review” to download all decisions and crop confirmations. Give this file to the data owner for a validated import. The page does not change source truth.

Page keys: `J` next, `K` previous, `F` flag, `A` accept, `X` exclude. Keys do not apply while you type in a field.

“Save filtered record list” exports a selection list. It is **not** a runnable evaluation manifest. The dataset owner must validate paths, labels, splits and hashes before converting it to the harness format. Sealed records and exclusions are omitted from this selection.

Review decisions stay in this browser's local storage. Some `file://` browser modes do not keep local storage. Save the review file before you close the browser. “Load review” restores page and crop decisions. Refreshing the catalogue preserves local decisions by ID.

The source-count table keeps pages, windows, patterns and intervals as separate units. Family counts cover listed records only. Repeated synthetic pages from one base well are not independent samples. The page does not calculate a pooled quality score.

## Data interface

`catalog.js` sets `window.DATASET_CATALOG`. You can also load a JSON catalogue through “Load catalogue”. The catalogue has `records`, optional `sources`, and optional top-level `suggestions` or `reviewSuggestions`.

Images need to be browser-readable PNG, JPEG, GIF or WebP. A TIFF path can be linked as a source file, but most browsers cannot display it. `imageWidth` and `imageHeight` always describe the original page. For cropped images, `imageBounds` or `thumbnailBounds` gives `[x0,y0,x1,y1]` in original page pixels. This mapping keeps source and review boxes in the correct coordinate system.

Metric readiness is optional: `metricReadiness` maps metric names to booleans, strings, or `{ready,status,reason}` objects. `ready:false`, `blocked:true`, and explicit blocked evidence remain blocked after visual acceptance.

Each crop suggestion needs an `id`, a parent `recordId`, original `bbox`, source hashes, and the model suggestion. `modelConfidence.value` is displayed as uncalibrated. The export's `suggestionReviews` contains explicit confirmed or uncertain human decisions, source/image/model-input hashes, original coordinates, notes and timestamps.

The crop queue also shows `contextCropPath`, `contextBox` and `legendContextPaths` when supplied. It marks mixed-boundary issues and keeps raw legend IDs visible. Use “Cannot tell” when the crop spans a contact or the legend does not support one label.

Crop confirmations from synthetic or outside records are marked `diagnostic-only`. They do not enter the human training pack. Only a current training record can produce a `human-train-candidate`, and it still needs the import bridge's hash, box and legend checks.

The data owner's import bridge is `packs/apply_review.py`. It validates the source image, annotation and model-input hashes, the original box and the printed legend label. It creates a new derived pack; it does not change the frozen pack. Page-level acceptance never creates truth. Use the import bridge's help and the dataset owner's handoff for the exact evaluation command.

Sealed records are identified by `sealed:true`, an exact sealed/locked access value, or an exact sealed/verify split. Their images, paths, truth, suggestions and review controls are hidden. A public/shared verification set is not treated as sealed merely because its name contains “verify”.
