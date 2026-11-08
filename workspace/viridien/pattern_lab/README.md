# Strata pattern lab

Shared review of lithology patterns in the HackTheBurgh source documents.

Open http://127.0.0.1:8776 while the server is running.

To start the server again, run `start.ps1` in this folder. Or run:

```powershell
python -X utf8 server.py --port 8776
```

The tool uses the existing Python packages Pillow, NumPy, and OpenCV. It also uses the TIFF reader in `../bakeoff/harness/io.py`. No model service or API key is needed at run time.

## Shared pattern queue

The main page shows only examples that need review. It uses 82 saved visual clusters from all 51 documents. Cluster assignments and legend guesses need human checks. Accepted source areas and rejected examples do not enter new assignments.

1. Check the selected box. Drag its top, bottom, or side handles. Use **Draw a new selection** to replace the box with a clean crop. Coordinate fields use original source pixels.
2. Choose one or more labels from that document's legend. Add a comment if needed.
3. Select **Accept and continue**. A set needs at most five accepted examples. Existing clean reference examples count towards this limit. Rejected examples do not count.

When all loaded boxes are rejected or skipped, the next available set opens automatically.
4. The next set opens after the fifth accepted example. Each set mixes source documents where possible. All strips are 220 display pixels wide. Both axes use the same scale; source coordinates stay unchanged.

The app saves each edit in browser storage at once. It sends changes to the shared server after a 250 ms pause. The server saves document JSON with an atomic replacement and an edit log. A failed request stays in the browser's outbox and is retried. An in-flight operation is kept until its receipt arrives. A newer edit waits behind that operation, so a lost response can be retried with the same operation ID. A stale state response cannot replace an answer saved after that state request began. Separate boxes can save without a document revision conflict. If two clients change the same box, the app keeps the local draft and shows a conflict. **Keep my draft** reapplies the latest local draft against the saved version. **Use saved version** loads the server choice. Both local versions remain in export history. **Export work** includes drafts, in-flight requests, and conflicts.

## Suggestions and larger legends

**Suggested for this box** is a separate section above the full legend. It reads the pixels inside the selected box, shows the best pattern guess first, and adds matching colour choices. The colour shortlist updates when a box edge moves or a new box is drawn. A colour mismatch on a pattern guess is marked for checking. The list is limited to eight entries. These are suggestions: no label is chosen or accepted automatically. Selection stays in sync with the full legend, and the manual colour and text filters do not hide the automatic section. Colour reading works in the browser from the included image, including offline reviewer files.

Use the colour buttons to narrow the legend by its source swatch colours. Use **Find a label** to search names. **All colours** and **Clear filters** restore the list. Mixed swatches can match more than one colour. Colour groups are visual hints, not rock labels. Filtering never changes the selected IDs. Selected labels stay visible above the list, including labels hidden by a filter; select a selected label button to remove it. Each choice has a larger swatch and a separate **Enlarge** button.

The list uses the page scroll. Box coordinate fields are under **Box coordinates**, and the accept controls stay available while the list is used.

Select **Expand source legend** to see the complete source panel. Select **Enlarge** beside a choice to inspect its swatch. The preview has a zoom control and closes with Escape. Opening a preview does not select or save a label.

Suggestions compare colour and symbol shapes with accepted examples and the current document's own legend. A background worker checks new accepted examples, updates the matching references, and prepares upcoming strips. This uses a separate image reader, so a large source page does not block the active reader. New predictions use the last completed reference model until the next one is ready. **Update guesses** repeats this check. A matching reference crop is shown for comparison. Label selection stays manual. New boxes grow from short seeds, with limits at strong fill changes and horizontal rules. Already saved boxes, labels, and comments are not changed by this process.

An accepted example with a strong fill-colour mismatch against its chosen legend stays accepted, but is withheld from learning. This catches some wrong choices; it cannot prove that the remaining choices are correct. Five examples finish a visual cluster, which can contain several rock types. The clusters are not verified rock classes.

Callaway's machine legend had five choices that combined several source rows. The repair adds 32 separate source-checked choices and retires the five merged choices. Old IDs remain valid for existing saved work. Retired choices cannot label a new accepted example. Other machine legends can still need correction; the full source panel is available for checking.

For an older portable HTML file, export its work, make a fresh reviewer file to get the new interface, then import the work export. The server still keeps the saved work. Do not discard the old file or browser data before checking the new file.

## One file for another computer

A reviewer can also open a private `/review#review=...` link directly in a browser on the same Wi-Fi. No HTML file is required. The browser keeps the reviewer session after sign-in and removes the key from the visible address. Edits upload automatically. Create a separate reviewer session for another person so they receive separate work.

To continue your current assignment, select **Continue on another device** and open the private link on the other device. Or select **Download this reviewer file** and copy that HTML file. Both keep the same reviewer and assignments. Flush failures must be resolved before a continuation link or file is made.

For a separate reviewer, select **New reviewer file** on the host. Send the downloaded HTML file to one reviewer. It has its own assignment, five pattern sets, source strips, legend crops, and the complete app. No installation is needed on the reviewer's computer. Make a new file for each reviewer so they receive different sets.

Open the HTML file in Chrome or Edge. Keep the file and use the same browser to resume. Allow local network access if the browser asks. The file saves labels, boxes, and comments to this host when it can reach the displayed server address. Edits in the HTML send to the host automatically after a short pause. Export is an optional backup. New sets load from the host as work continues. The browser keeps about three unfinished sets ready; a portable file starts with five. **Load more work** asks for a fresh set, up to six loaded sets. Source images already loaded are omitted from later bundle responses when their source box and legend are unchanged. Completed image packs are limited to the two most recent sets. The host's full document editor checks for new saved work every five seconds. **Connection and device link** shows the last confirmed server upload for each reviewer on the host. Offline drafts reach the host only after the device reconnects. If image storage fails, the page reports that offline images are not ready; answer saving is separate.

The host must stay on, with `start.ps1` running. The other computer must reach the host's network address, usually on the same Wi-Fi. The file also works without the network for the sets already included or cached. Edits stay in browser storage and sync when the connection returns. The **Connection and device** panel shows the server address and lets you change it if the Wi-Fi address changes. Each tab keeps a separate local draft. Opening the same reviewer on two devices is supported; a simultaneous change to one box requires a conflict decision. A copied HTML file does not include later unsent browser edits: use **Export work** to transfer those edits, then **Import work** to restore them. Export work also provides a separate backup.

If an older or local-only service is already running, run `./start.ps1 -Restart` once. It checks the service on port 8776 before it restarts that Python process. The shared service listens on loopback and the selected physical network address.

`start.ps1` selects the local network address. For a fixed address, run:

```powershell
python -X utf8 server.py --port 8776 --share-url http://YOUR-LAN-IP:8776
```

Reviewer files include a private key for their assignment. Share each file with its intended reviewer. Host-only document and generator routes do not accept remote reviewer requests.

## Review a document

Open **Full document editor** from the queue, or visit http://127.0.0.1:8776/editor. This view includes accepted boxes for inspection and correction.

1. Choose a document from the list.
2. Choose one of its legend labels. The list shows the boxes for that pattern.
3. Select a box. Drag the top, bottom, or side handles. Drag inside the box to move it. Use **Top edge** or **Bottom edge** to reach an edge outside the current window.
4. Use **Draw box** to make a new selection. Use the coordinate fields for exact pixel limits. **Split box at its centre** makes two boxes that need review.
5. Select the correct legend labels. A box can have more than one label. Then select **Accept segment**.
6. Add a comment to record an issue with that box.
7. Select **Use as pattern source** for a clean example. Open **Pattern lab** to compare the real crop with a generated pattern.
8. Export the pattern dataset. The ZIP has PNG samples, a manifest with source boxes and labels, and the document legend.

Use the wheel, the native scroll bar, or the vertical slider to move through the document. The viewer caches 1,024-row source strips, so scrolling within cached strips does not need a new image request. The normal view uses the supplied lithology-column bounds exactly. More context includes the supplied depth-column bounds. Both axes use the same scale, which is reduced when needed to fit the viewer width. Use `[` and `]` for the previous and next boxes. Changes to a box or its labels remove its acceptance and pattern-source status until it is accepted again. Comments do not remove acceptance.

## Initial data

The catalog contains all 51 source documents. Saved machine guesses come from `../bakeoff/labeltool/data/prefill.jsonl`. If that file is absent, preparation uses `queue.jsonl`. Machine guesses need review. The legend names for those documents also come from the existing machine cache and can be corrected in **Edit this legend**.

The supplied Hadrian JSON is copied to `data/imported/hadrian.json`. Its 680 layers and 18 legend IDs are kept. Each composition box uses its own horizontal range. The source contains five used labels: Halite, Shale grey, Siltstone, Sandstone, and Marl. One source per used label is selected against its legend background colour and crop length for the initial generator. Legend entries with no labelled example stay empty.

The generator copies pixels at their native scale. It estimates pattern repeats and uses source patches when a repeat is unclear. Generated patterns can have seams. Use the side-by-side view to check them. The output is a dataset of pattern patches. A mixed source keeps its label set; the tool does not infer new component positions within the output.

**Approve sample** and **Reject sample** save sample-quality reviews. Dataset export uses accepted boxes marked **Use as pattern source**. It generates the requested number of samples again from those sources and the export seed. Sample-quality reviews do not change the export list.

## Real test data and pattern sources

Select **Normalized dataset** on the host. The background worker makes an immutable ZIP after accepted annotations or legend metadata change. Draft typing does not make new dataset versions. The page shows background status and the counts from the last complete snapshot. New work can continue while the next snapshot is made.

The ZIP contains `manifest.json`, `train.jsonl`, `test.jsonl`, `needs-review.jsonl`, `generator-sources.json`, and image assets. Each real record includes:

- A stable sample ID, annotation version, source document and well group, and printed and common label IDs.
- Original pixel bounds, document-relative bounds, source dimensions, comments, and review provenance.
- A native RGB crop and a 224 × 224 RGB copy. The copy keeps the aspect ratio and uses white padding. Its separate mask is 255 for content and 0 for padding. Scale and offset values describe the transform. Native pixels remain authoritative.
- A fixed `train` or `test` assignment and quality flags. Divide RGB values by 255 for float inputs and ignore padding in the mask. Apply the same transform to real and generated samples.

The fixed split groups known pages, scales, and sidetracks from the same well. A well group stays in one split. Exact pixel duplicates of a test crop are excluded from training. Strong legend-fill mismatches and flagged source annotations stay saved and accepted, but enter `needs-review.jsonl` instead of reusable training or test data. These checks find some errors; they do not establish label accuracy.

The matcher and both generator paths use only current, quality-accepted training sources. Real test records are withheld. Existing reviewed groups can have been used by earlier matching or generation code. Their evaluation status is `retrospective`; only new reserved groups are `prospective`. The small current test sample is not a validated accuracy estimate.

The generator lab on port 8781 watches these snapshots. Its synthetic records use the same common label IDs and record source sample IDs and annotation versions. Generated samples are marked `kind=synthetic`, `split=train`, and `generation_review=unreviewed`. Their normalized RGB copies and padding masks use the same transform. Synthetic outputs never enter the real test set. Machine-verified generator entries remain identified as machine sources; they are not human test truth.

## Saved files

- `data/seeds/`: prepared copies of source guesses and supplied labels.
- `data/reviews/`: current document reviews. The original source files are not changed.
- `data/events.jsonl`: record of saved edits.
- `data/generation-reviews.jsonl`: generated-sample reviews.
- `data/shared-reviews.sqlite3`: reviewer sessions, assignments, and save receipts.
- `data/pattern-queue.json`: prepared cluster samples and reference mappings.
- `data/review-crops/`: cached source crops for quick loading.
- `data/normalized/`: fixed split policy, latest real manifest, and content-addressed native and normalized images.
- `data/background/`: future-strip proposals and background status.
- `data/feedback-features.json`: cached visual descriptors; no automatic approvals.
- `data/quality-audit/`: the 25-example audit, source comparisons, replay checks, and a backup from before the legend repair.
- `data/exports/`: dataset folders and ZIP files.

Back up the whole `data/` folder while the server is stopped. It keeps the review JSON, edit history, shared assignments, and exports together. The original image dataset is also needed to run the host on another computer.

**Export review JSON** downloads the current document review, including local edits if a save failed. Save failures stay visible and block document switches. Use **Retry save** for a temporary failure. If another page has saved a newer revision, export the local draft, then reload. There is no automatic merge for conflicting edits.

## Data paths

By default the source dataset is found at:

`../../viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset`

Set `LITHOLOGY_DATASET` before preparation and server start to use a different source directory. Regenerate the seeds after a relevant source change:

```powershell
python -X utf8 server.py --prepare
```

Preparation does not replace files in `data/reviews/`.

## Checks

The second review round uses saved v4 bed predictions from
`../bakeoff/analysis/categories/`. Run `python -X utf8 prepare_verification.py`
from this app folder to prepare the prediction pack and source strips.
`static/verification-predictions.txt` contains JSON and is loaded once per page.
No model runs between clicks. A source version check protects saved edits.

The reviewer can select **Confirm [label] and continue** after checking the box
against its own legend. This action saves one human confirmation. A box edit
disables that action until the reviewer selects a label manually. Model scores
are suggestions, not measured accuracy. Matching symbols and unmapped first
guesses require a manual choice. The save records the model version, proposed
box, and review method. Existing local drafts, save receipts, and conflict
checks stay active. No label is accepted without a reviewer action.

The prepared second round gives the existing Wi-Fi reviewer ten open sets,
with five accepted examples needed per set. Sets 44 and 52 are held for a source
legend check. Their rejected examples remain saved. The private continuation
link and PowerShell command are in `data/verification-round-2/teammate-link.txt`.
Open that link on the same Wi-Fi. Keep this host running. Each save goes to the
shared server; no HTML file or manual export is needed.

From the project workspace:

```powershell
python -X utf8 -m pytest viridien_work/pattern_lab/tests viridien_work/pattern_lab/generator/test_core.py viridien_work/pattern_lab/genlab/tests/test_genlab.py -q
node --test viridien_work/pattern_lab/tests/test_viewport.cjs viridien_work/pattern_lab/generator/test_lab.cjs
node --check viridien_work/pattern_lab/static/review.js
node --check viridien_work/pattern_lab/static/app.js
```

`tests/qa_server.py` starts an isolated review server on port 8777. `tests/browser-check.js` checks box creation, top and bottom resizing, acceptance, source selection, comments, save and reload, rejection, deletion, document switching, and phone width through the Playwright CLI. The isolated server writes under `data/qa/`.

`tests/qa_shared_server.py` also uses port 8777 and writes under `data/qa-shared/`. Use one test server at a time. `tests/browser-shared-check.js` checks source mixing, equal widths, dragging, immediate local saves, shared saves, host updates, the five-example limit, next-set changes, offline recovery, phone width, and file-to-LAN sync in independent browser contexts. Test labels stay outside the real `data/reviews/` folder.

`tests/qa_feedback_server.py` copies current saved work into `data/qa-feedback/` for checks on port 8777. `tests/browser-continuation-check.js` checks same-assignment device links, concurrent changes, tab storage, offline retries, phone width, authentication, and the normalized export. `tests/run-continuation-browser.ps1` runs the continuation, shared, and legend checks against that isolated server through the Playwright CLI.

The colour and remote checks are in `tests/browser-legend-remote-check.js`. `tests/run-legend-browser.ps1` also runs the earlier continuation, queue and legend checks. They use the isolated server started with `python -X utf8 tests/qa_feedback_server.py --lan YOUR-LAN-IP` from this app folder. The optional LAN address gives the test reviewer file a real Wi-Fi destination while all test edits stay in `data/qa-feedback/`.
