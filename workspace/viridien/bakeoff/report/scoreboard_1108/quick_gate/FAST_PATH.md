# Five-minute workflow check

This is a new advisory. It does not replace the frozen Availability V2 release, change a model, or lift an owner hold.

## What is verified now

The four strategies share an outer interface: `run(log, window, ctx)` returns interval rows. Inputs bind the same image, annotation, evaluation window and truth hashes. The saved predictions keep full-image pixel coordinates. Truth is an evaluation input; it is not supplied to inference.

The cached gate checks all 272 records. It exports each successful prediction to both published CSV schemas and reads it back. It preserves raw labels, contacts, cluster IDs and depths. One failed/empty S19 case stays a failure. The gate does not read source images or truth, run a model, call an API, or recompute accuracy.

Run from `viridien_work/bakeoff`:

```text
python -X utf8 -B report/scoreboard_1108/quick_gate/background.py
```

The parent gives the gate a 180-second hard timeout and saves a command receipt. The observed command, real exit code and measured time are in `command_receipt.json`. Test details are in `tests.log`. Exports are under `out/quick_contract_20261108/<strategy>/<case>/`.

## One complete target workflow

1. Read one source page and its supplied column, legend and depth boxes.
2. Parse its legend and calibrate depth once for the exact context.
3. Process the entire lithology column in strips. Reuse only page-level work whose identity matches.
4. Find beds, assign lithology and visual cluster IDs, and assign boundary depths.
5. Write one interval CSV and one annotated image for that document. Validate full-image coordinates and record failures.
6. Evaluate cached predictions separately against permitted truth. Report strict and approved Clay/Shale-equivalent values, explicit denominators, depth, runtime and resources.

The existing evaluation uses selected windows. For each strategy, 44 of 61 physical documents have a saved record spanning the full supplied lithology box. Seventeen have only partial/unknown coverage in this matrix. Full challenge readiness is not verified. Per-document final annotated images and correct depths still need validation.

The slide calls the lower depth `base_depth`; the notebook uses `bottom_depth`. Both exports are kept explicit. The supplied challenge material says correctness and efficiency. It provides no verified numeric threshold or scoring weights. Our `row accuracy * boundary F1` and Clay/Shale rule are local evaluation policies, not confirmed official grades.

## Strategies and the next useful speed path

- S19: clean synthetic models, v3 legend OCR, and a joint interval decoder. It has the strongest broader development result. It is the reference candidate for the next permitted full-workflow test; no champion is promoted here.
- V4: bed segmentation and nearest-swatch matching. Its frozen outside-data path lacks native legend entries. Working legend coverage is required before its matcher can be assessed there.
- V4 plus DINO: frozen DINOv2-small at native tile scale, fused with hand features. It leads on the three QA/gold logs. This is a small subset, not full-dataset proof.
- S23 keeps legend captions distinct. Its result does not establish a gain over DINO.
- Optimized JEV is partial: 29 verified pairs, 30 saved, one unpromoted. New calls remain held.

The reviewed cache pilot already reuses exact-matching legend context across windows. Four windows returned 612 identical intervals. Prediction time was 6.767 to 3.745 seconds in one ordered pilot. This change is in a separate opt-in source packet. It is not merged or used to revise the frozen scores.

For the next permitted run, keep page context and the model warm, parse legend/depth once, decode strips, hash unchanged page content once per page, and checkpoint every case. Use the existing two-core worker lock. Keep queue wait, preprocessing, inference and rendering time separate. Enforce a whole-job deadline and preserve completed cases. A new five-minute full-run promise needs a measured, same-coverage test.

Current owner holds prevent dataset inference, tuning, new labels and JEV dispatch. Depth repair is UNASSIGNED with no active claim. This advisory assigns no worker and changes no source ownership.

## Runtime interpretation

The new cached check is measured directly. Historical model timings are window-level values from different hardware and cache states. They are not a forecast for a full-page or full-dataset run. S19 also has one failed attempt with unknown inference time; its 600-second attempt cap is not a measured inference runtime.

The five contract tests are run by this gate. The older full harness still has its known PNG-fixture/TIFF-loader failure. It was not rerun or fixed by this work.
