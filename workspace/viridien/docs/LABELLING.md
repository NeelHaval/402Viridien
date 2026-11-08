# Review and save pattern labels

Use `pattern_lab/` for current manual pattern examples. The earlier tool in
`bakeoff/labeltool/` contains category, legend, and confidence review experiments.
Their saved data is separate.

Set `LITHOLOGY_DATASET` as shown in [setup](../START_HERE.md). From the repository
root, install the seed once in a fresh checkout, then start the labeller:

```bash
python viridien/tools/workspace.py download lab
cd viridien/pattern_lab
python -X utf8 server.py --host 127.0.0.1 --port 8776
```

Open <http://127.0.0.1:8776>. The seed includes accepted document labels and
clearly marked machine suggestions. It has no private reviewer sessions. Start
a new reviewer from the host page. Old private links belong to the original host.

## Review an example

1. Compare the crop with that document's legend. Expand the complete legend if
   the parsed choices look wrong.
2. Drag the crop edges to contain the intended pattern. Coordinates stay in the
   source image's pixels.
3. Select the correct printed label. Use several labels for a mixed crop.
4. Add a comment for ambiguity, legend damage, or a boundary issue.
5. Accept the crop. Wait for the saved state before closing the page.

The queue aims for five accepted examples per visual set. A visual set can
contain several rock types. Finishing a set does not certify a geological class.
Suggestions are guesses until accepted. A changed box disables the old quick
confirmation. The full document editor is at <http://127.0.0.1:8776/editor>.

## Save and share

The browser keeps drafts and sends changes to the host. Failed sends remain in
its outbox. A conflict keeps the local draft for a decision. Use **Export work**
as a backup before changing a browser or machine.

Use **New reviewer file** for another reviewer. Use **Continue on another device**
for the same reviewer. Keep the host on and use a reachable LAN address. Windows
`start.ps1` selects the physical network address. See the
[full tool guide](../pattern_lab/README.md) for shared access.

Each checkout has its own save database. A release download does not sync new
labels back to the original host. Live reviews, session databases, and exports
stay under `pattern_lab/data/` and are ignored by Git. Publish a sanitised snapshot;
do not commit the live database or reviewer files.

Test wells, exact test duplicates, and quality holds must not supply training
pixels. The normalized export retains accepted work even when it is withheld.
A colour check catches some errors; it cannot prove correctness. Use complete
interval stretches for boundary and thickness evaluation. Keep ambiguity notes
for near-identical patterns such as Clay and Shale.

## Earlier review pages

From `viridien/bakeoff/`, run `python labeltool/serve.py --help`. Its queues and
prediction caches can need rebuilding. Follow its
[README](../bakeoff/labeltool/README.md). Historical `/legend`, `/mix`, `/cat`, and
`/conf` URLs are local services. They do not run on GitHub.
