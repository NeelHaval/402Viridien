# S13 real-boundary CNN

Selection uses the 59 external dev windows with leave-one-well-out prediction across six wells. The same pooled dev predictions select architecture, threshold, spacing and merge policy; these are selection scores, not an additional unbiased test. Final test windows and the 13 in-domain eval logs are scored only after the checkpoint is frozen.

| Split | Strategy | Row accuracy | Boundary F1 | ARI | Score | Mean seconds | Max seconds | Crashes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev CV | s11 | 0.4388 | 0.3519 | 0.6563 | 0.1710 | 0.1377 | 0.6342 | 0 |
| dev CV | s13 | 0.4617 | 0.5551 | 0.7676 | 0.2692 | — | — | 0 |
| ext_dev | s12 | 0.0000 | 0.0169 | 0.1186 | 0.0000 | 0.1137 | 0.5845 | 0 |
| ext_test | s12 | 0.0000 | 0.0323 | 0.2860 | 0.0000 | 0.1040 | 0.2636 | 0 |
| viridien_eval | s12 | 0.0389 | 0.0530 | 0.0223 | 0.0268 | 0.2542 | 1.1203 | 12 |
| ext_test | s11 | 0.1021 | 0.2164 | 0.5388 | 0.0587 | 0.2193 | 0.7156 | 0 |
| viridien_eval | s11 | 0.3647 | 0.5462 | 0.6322 | 0.2029 | 0.5791 | 2.2911 | 0 |
| ext_test | s13 | 0.1557 | 0.3242 | 0.4303 | 0.0712 | 0.1719 | 0.4476 | 0 |
| viridien_eval | s13 | 0.3967 | 0.5694 | 0.6773 | 0.2450 | 0.2854 | 0.9012 | 0 |

In-domain truth is gold for log06 and consensus of available silver labels for the other 12 logs. External truth is exact. Means weight windows equally. Shared legend/depth preprocessing is excluded from strategy inference timing.

Failures count as zero in displayed accuracy, boundary, ARI and composite-score means; the raw harness JSON may omit a failed strategy’s unavailable metric fields. S12 requires EasyOCR weights that are absent on this VM; in-domain OCR cache misses fail when downloading them. Those failures are retained, with no replacement or rerun.

Real boundary targets, panel features and changes across row scales improved the selected pipeline. Synthetic augmentation improved raw proposals but did not beat the real-only model on the full dev score. Keeping every learned boundary between identical matched labels also reduced the dev score. S13 improved final test and in-domain row accuracy and boundary F1. External test ARI fell from 0.5388 to 0.4303, while in-domain ARI rose from 0.6322 to 0.6773; the full metrics retain these tradeoffs.

## Dev ablations

| Features / training | Threshold | Gap | Preserve boundaries with same matched label | Row accuracy | Boundary F1 | Score | Proposal F1 |
|---|---:|---:|---|---:|---:|---:|---:|
| rich_real | 0.7 | 24 | False | 0.4617 | 0.5551 | 0.2692 | 0.4459 |
| rich_real | 0.55 | 24 | False | 0.4648 | 0.5618 | 0.2682 | 0.4369 |
| rich | 0.7 | 12 | False | 0.4713 | 0.5554 | 0.2634 | 0.5002 |
| rich_real | 0.8 | 12 | False | 0.4735 | 0.5446 | 0.2605 | 0.4629 |
| rich | 0.7 | 24 | False | 0.4657 | 0.5352 | 0.2530 | 0.4854 |
| six | 0.55 | 12 | False | 0.4531 | 0.5577 | 0.2475 | 0.4784 |
| rich | 0.7 | 12 | True | 0.4713 | 0.5002 | 0.2437 | 0.5002 |
| rich_real | 0.9 | 12 | False | 0.4600 | 0.5028 | 0.2387 | 0.4705 |
| six | 0.7 | 12 | False | 0.4508 | 0.5087 | 0.2348 | 0.4763 |
| rich | 0.7 | 24 | True | 0.4657 | 0.4854 | 0.2336 | 0.4854 |
| six | 0.7 | 12 | True | 0.4508 | 0.4763 | 0.2300 | 0.4763 |
| rich_real | 0.9 | 12 | True | 0.4600 | 0.4705 | 0.2275 | 0.4705 |
| rich_real | 0.8 | 12 | True | 0.4735 | 0.4629 | 0.2254 | 0.4629 |
| six | 0.55 | 12 | True | 0.4531 | 0.4784 | 0.2206 | 0.4784 |
| rich_real | 0.7 | 24 | True | 0.4617 | 0.4459 | 0.2205 | 0.4459 |
| rich_real | 0.55 | 24 | True | 0.4648 | 0.4369 | 0.2201 | 0.4369 |

## Dev by held-out well

| Well | Windows | S11 row accuracy | S11 boundary F1 | S13 row accuracy | S13 boundary F1 |
|---|---:|---:|---:|---:|---:|
| 386825 | 2 | 0.1333 | 0.2106 | 0.0827 | 0.5357 |
| 4304150011 | 21 | 0.5467 | 0.5247 | 0.5592 | 0.7552 |
| ForsythB2-17 | 8 | 0.4405 | 0.4717 | 0.4962 | 0.6849 |
| barlow3-14 | 6 | 0.0899 | 0.1041 | 0.1107 | 0.2411 |
| flat-bay-2 | 9 | 0.5169 | 0.1938 | 0.5282 | 0.3002 |
| storm | 13 | 0.4176 | 0.2444 | 0.4571 | 0.4763 |

## Frozen model

```json
{
  "steps": 450,
  "seed": 100,
  "training_wells": [
    "386825",
    "4304150011",
    "ForsythB2-17",
    "barlow3-14",
    "flat-bay-2",
    "storm"
  ],
  "real_windows": 59,
  "synthetic_columns": 0,
  "dev_items_sha256": "567345671c9376298432e92e621538a6a1cee9db6fcba9e05e30c8254125e9bf",
  "config": {
    "gap": 24,
    "preserve": false,
    "proposal_f1": 0.4458937882049002,
    "threshold": 0.7,
    "variant": "rich_real",
    "kind": "rich"
  },
  "seconds": 25.520450074996916,
  "peak_rss_gb": 0.5023765563964844,
  "weights_sha256": "5f547847661a92ed7d2fae161bdd6374a738d12fa42bc06cb749e1cd08178cb1"
}
```

Checkpoint size: 49,885 bytes.

Six dilated Conv1d/ReLU layers (dilations 1, 2, 4, 8, 16, 32), 24 hidden channels, one sigmoid output. Weighted binary cross entropy uses positive weight 6 and Gaussian boundary targets with sigma 4 pixels. Training samples wells uniformly, then windows within wells; batches contain eight 768-row crops, with reversal, mild feature scaling and noise. AdamW uses learning rate 0.0015 and weight decay 0.001. Inference runs on CPU with at most three torch threads and chunks of 8192 rows plus 128 rows of context.

Rich features contain whole-column and three-panel RGB/ink/transition means, dark-rule extent, and changes measured at 4, 12 and 32 rows. The six-feature ablation uses Emil’s normalized RGB, ink, horizontal transitions and vertical transitions. Legend matching reuses S11 Fourier/Lab descriptors, distance weights, rejection threshold, sampling and unknown clusters.

Synthetic data: 72 procedural columns seeded 3000–3071, each 2048 rows, balanced across three-panel colour fills, sparse Rock Types-style patterns and binary hatch. They randomize bed thickness, pattern frequency/phase, rules, gaps, panel mixtures and noise. Synthetic-augmented runs sample synthetic crops with probability 0.25. No additional human-labelled non-eval logs were available in the standard truth directories; none of the 13 in-domain eval logs or their labels were used for training or selection.

Per-window log, well, source, window coordinates and exact training truth are recorded in `data/index.json`. Cached features/targets are in `data/*.npz`, fold checkpoints and held-well probabilities in `runs/`, and final harness CSVs, figures and raw metrics in `harness_workspace/out/`.

The Emil baseline loads the existing bakeoff/emil/boundary_net.pt checkpoint. Its archived pseudo-label training provenance includes some in-domain eval images (without eval truth), so it is reported as an existing baseline rather than a cleanly retrained comparison. S12’s explicit six-thread request is capped at three in the evaluation adapter, and its missing archived default checkpoint path is redirected to that existing portable checkpoint. OCR cache reuse is by exact legend pixels.

Raw proposal comparisons (before legend matching and merges):
- s11_proposals: 0.3324.
- s11_full_proposals: 0.3422.
- emil_proposals: 0.3613.

The recorded S11 proposal ablation is its PELT pass alone; the full S11 rows above include dark-rule proposals and label merging. The Emil proposal comparison uses its existing checkpoint and fixed 0.5 threshold. The final comparison is recorded without post-test tuning.

## Reproduction

From `bakeoff/`, set `LITHOLOGY_DATASET` to the local dataset and cap OMP/MKL/OpenBLAS threads at three. Use the repository’s Python environment.

```text
python -m ml.boundary.experiment prepare
python -m ml.boundary.experiment cv --steps 450
python -m ml.boundary.experiment select
python -m ml.boundary.diagnostics
python -m ml.boundary.experiment fit --steps 450
python -m pytest ml/boundary/test_boundary.py -o cache_dir=ml/boundary/.pytest_cache
python -m ml.boundary.evaluate --split dev --strategies s12
python -m ml.boundary.evaluate --split final --strategies s11,s12,s13
python -m ml.boundary.report
```

The evaluation adapter invokes `harness.run.main` unchanged for scoring, depth filling, CSVs, PNGs, timing and resource measurement. It redirects all outputs and caches into this lane. The final run marker prevents accidentally rerunning the holdout for selection.

## Naming diagnostic

Giving the unchanged S11 matcher exact dev boundaries yields 0.4865 row accuracy. This is a diagnostic for the existing matcher, not a universal upper bound: mixed intervals and omitted legend names still cause errors. It explains why substantial boundary gains can produce smaller row-accuracy gains.
