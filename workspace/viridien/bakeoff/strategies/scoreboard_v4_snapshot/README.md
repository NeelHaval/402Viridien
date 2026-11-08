# Frozen v4 scoring inputs

These files were copied from the local project before the score run. No rule or
threshold was changed. scoreboard_v4.source_manifest() returns their SHA-256
hashes. Source paths, relative to the bakeoff root:

- labeltool/legend_classify.py
- labeltool/beds.py
- analysis/categories/native_swatches.pkl
- analysis/categories/log_scale.json

The adapter supplies the harness Log and the evaluation window to the original
bed code. Frame statistics therefore use the evaluation window, rather than the
full lithology column. This keeps the image extent equal for all scored methods.
The classifier receives the frozen native-swatch cache. A log that is absent in
that cache still has an empty legend. There is no automatic parser repair.
The frozen configuration has scales=(), so the old scale cache is not read or
written during inference. Twin labels and merge rules are retained exactly.
Blank beds have no output rows. Nonblank unresolved beds use Unknown. Depths
come from the shared harness calibration supplied in ctx.

The frozen files import the local labeltool.jev2 and common modules. No Context
or human-example function is called. beds.C is replaced by an isolated frozen
classifier namespace before segmentation. Each call owns that namespace, so
DINO fusion can replace classify without changing s20.
