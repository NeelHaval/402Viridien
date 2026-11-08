# LLM labelling harness

Run from the bakeoff root:

```powershell
python -X utf8 labelling/llm_label.py
python -X utf8 labelling/render_review.py
```

`--validation-only` limits inference to the supplied log06 window. Successful
responses are cached by image bytes, prompt, model and effort. All CLI starts,
including failed startup attempts, count against the persistent 220-call cap in
`out/calls.json`. A failed response is retained before retrying. The vocabulary
is inferred once per log; model S and model T then independently label each tile.

The first iteration uses approximately 580-row tiles. If accuracy on agreed
gold rows is below 0.85, the next iterations use approximately 500-row tiles,
larger images and additional generic texture instructions. At most three
validation iterations run. No gold content enters images, prompts, vocabulary,
boundary snapping or consensus selection. Gold is read only after both models
and consensus have been saved for a completed validation iteration.

Disputed consensus rows select the model with greater reported confidence;
ties select S. They remain explicitly disputed. Adjacent equal rock labels are
merged unless agreement status differs. CSV depths come from harness calibration;
existing calibration caches are read without modifying them and new calibration
caches are redirected below `labelling/`.

## Current execution result

The four attempted vocabulary calls all failed before inference. Codex CLI
0.166.0 and the previously installed 0.163.0 both reported:

```
failed to initialize in-process app-server client: Access is denied. (os error 5)
```

The required account home was preserved. Redirecting writable runtime SQLite,
logs and temporary files below `labelling/` did not resolve initialization.
No model labels were returned and no gold CSV was opened. The blocked status
is recorded in `out/validation.json` and `out/summary.json`; unavailable metrics
and agreement values are null. Prediction and vocabulary JSON/CSV files have
not been fabricated.

`review.html` currently contains the ten requested source windows and their
original legends, with clearly unlabelled model tracks. It is self-contained
and below 25 MB. The headless Edge command was attempted once, but Edge's
Crashpad registration failed with `CreateFile: Access is denied (0x5)` and no
screenshot appeared. See `out/render_check.json`.

```powershell
python -X utf8 labelling/check_local.py
```

The local deterministic checks pass for native-row coverage, exact 60-row
overlaps, allowed image dimensions, image-change boundary snapping, agreement
interval splitting, CSV booleans and JSON parsing. They do not measure model
quality. To regenerate only the unlabelled source review without model calls:

```powershell
python -X utf8 labelling/llm_label.py --preview-only
```
