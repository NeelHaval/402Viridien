# Model experiments

Implement an importable function:

```python
def predict(request: dict, config: dict) -> list[dict]:
    # request['image']: original image path
    # request['annotation']: public column/legend bounding boxes
    # request['image_size']: [width, height]
    # request['window']: [top_y, bottom_y], full-image pixels
    return [{"top_y": 1000, "bottom_y": 1200,
             "lithology": "Unknown", "cluster_id": 0}]
```

The rows must fit the requested window, have positive height, be sorted, and not overlap. Blank stretches may be omitted. Return a list, including an empty list when no lithology is detected. Use the page's own legend label or `Unknown`. Cluster IDs are integers local to the page, with repeated IDs for repeated visual patterns. Depth endpoint fields are optional together. Provide `config.depth_unit='ft'` and the matching `config.depth_datum` only when those predictions genuinely use that unit/datum.

Evaluation passes public inputs only; it never supplies the truth CSV/path, source labels, split, or expected scores to the predictor. `full_image` means image plus supplied challenge bounding boxes, as allowed in the brief. `supplied_legend` additionally supplies explicitly declared record-level legend/depth assistance. These modes are different experiments and comparisons reject mode mismatches.

Python plugins are **trusted team code**, not a filesystem/network sandbox. A plugin could read labels on disk if written to do so. Independent sealed evaluation needs a separate environment with no accessible truth files. Worker processes inherit a minimal environment without API/repository tokens. Paid APIs need a separately reviewed credential mechanism; the default workflow makes no paid calls.

## Change one thing and compare

```sh
python -m viridian402 evaluate --manifest data/local/human.json --model your_package.model:predict --config configs/experiment.json --output runs
python -m viridian402 compare runs/BASELINE_RUN runs/CANDIDATE_RUN --output runs/comparison.json
```

Place importable code under `src/` and add needed dependencies to `pyproject.toml` with an appropriate extra. Keep configuration in tracked JSON. Declare all training/tuning well groups as `training_groups`; evaluation rejects declared overlaps with dev/test/verify groups. An undeclared training history is unknown, not proof of independence. Use train-only manifests for training and record augmentation parent lineage. Do not train or tune during the evaluation function.

Checkpoint paths in config are hashed when they identify existing files. Keep weight bytes ignored or in a versioned private release, and put the expected checksum and training-data/model revision in the config. Re-run after any change to model code, checkpoint, preprocessing, legend parsing, depth mapping, or dependencies. Package versions, code/input hashes, config, environment, seed and selected cases are retained per run.

## Included models

- `builtin:baseline`: small NumPy colour/ink boundary detector and texture/colour clustering. It reports `Unknown` until a diagnostic supplied legend is available. Useful for testing the evaluator and iterating cheap image features.
- `builtin:legacy-s11`: the existing team FFT hatch/Lab matcher and PELT boundaries, frozen from source commit `663721e72e3fa2ba496987c13b66610e52d1c8f3`. `src/viridian402/_legacy` contains only model/loader/legend/depth code. It contains no truth or old answer caches. The adapter uses an empty disposable OCR cache and a bounded image reader. Changes to the frozen copy should get a new model ID/config for fair baselines.
- `builtin:empty`: failure/coverage diagnostic, never a competitive model.

S11 requires `.[legacy]`. `configs/legacy-s11.json` skips depth OCR so legend/boundary work can run independently; missing depths stay visible. `configs/legacy-s11-depth.json` enables OCR and explicitly declares printed feet with unknown datum. For metric pages use a separate config with `image_depth_unit='m'`. For MD/TVD evaluation use separate configs declaring the correct datum; do not mix them under a false common unit.

Cold worker startup, loading and preprocessing are included in recorded inference latency. Parent manifest validation, hashing, scoring and previews are outside this timing/RSS scope. The worker enforces a per-case timeout and samples process-tree RSS; the reader also applies image decode/crop budgets. RSS sampling can miss brief peaks. GPU VRAM and paid API costs are not measured automatically in this CPU harness. If introducing such models, add explicit instrumentation and a fixed hardware profile before interpreting speed/cost comparisons.
