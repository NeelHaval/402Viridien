"""Produce a compact report and model card from immutable experiment artifacts."""
import hashlib
import json
from pathlib import Path
import statistics

from ml.boundary.model import HERE


def mean(records, field):
    penalized=field in ('row_acc','boundary_f1','cluster_ari','score')
    values=[r[field] if r.get(field) is not None else 0.
            for r in records if r.get(field) is not None or (penalized and 'error' in r)]
    return statistics.mean(values) if values else None


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def main():
    selection=json.loads((HERE/'runs'/'selection.json').read_text())
    training=json.loads((HERE/'training.json').read_text())
    candidates=selection['candidates']
    selected=candidates[0]
    lines=['# S13 real-boundary CNN', '',
           'Selection uses the 59 external dev windows with leave-one-well-out prediction across six wells. '
           'The same pooled dev predictions select architecture, threshold, spacing and merge policy; these are selection scores, '
           'not an additional unbiased test. Final test windows and the 13 in-domain eval logs are scored only after the checkpoint is frozen.', '',
           '| Split | Strategy | Row accuracy | Boundary F1 | ARI | Score | Mean seconds | Max seconds | Crashes |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    summaries=[]
    base=selection['baselines']['s11']
    summaries.append(dict(split='dev CV',strategy='s11',records=base))
    summaries.append(dict(split='dev CV',strategy='s13',records=selected['records']))
    root=HERE/'harness_workspace'/'out'
    for strategy in ('s12_emil','s11_fft_legend','s13_boundary_net'):
        for split in ('ext_dev','ext_test','viridien_eval'):
            path=root/strategy/split/'metrics.json'
            if path.exists():
                data=json.loads(path.read_text())
                summaries.append(dict(split=split,strategy=strategy.split('_')[0],records=data['logs']))
    output=[]
    for item in summaries:
        rows=item['records']
        values={field:mean(rows,field) for field in ('row_acc','boundary_f1','cluster_ari','score','seconds','peak_rss_gb','depth_mae_ft')}
        times=[r['seconds'] for r in rows if 'seconds' in r]
        values.update(split=item['split'],strategy=item['strategy'],windows=len(rows),
                      max_seconds=max(times) if times else None,crashes=sum('error' in r for r in rows),
                      peak_rss_gb=max([r['peak_rss_gb'] for r in rows if r.get('peak_rss_gb') is not None],default=None))
        output.append(values)
        lines.append(f"| {item['split']} | {item['strategy']} | {fmt(values['row_acc'])} | {fmt(values['boundary_f1'])} | {fmt(values['cluster_ari'])} | {fmt(values['score'])} | {fmt(values['seconds'])} | {fmt(values['max_seconds'])} | {values['crashes']} |")
    lines.extend(['','In-domain truth is gold for log06 and consensus of available silver labels for the other 12 logs. '
                  'External truth is exact. Means weight windows equally. Shared legend/depth preprocessing is excluded from strategy inference timing.',
                  '', 'Failures count as zero in displayed accuracy, boundary, ARI and composite-score means; '
                  'the raw harness JSON may omit a failed strategy’s unavailable metric fields. '
                  'S12 requires EasyOCR weights that are absent on this VM; in-domain OCR cache misses fail when downloading them. '
                  'Those failures are retained, with no replacement or rerun.',
                  '', 'Real boundary targets, panel features and changes across row scales improved the selected pipeline. '
                  'Synthetic augmentation improved raw proposals but did not beat the real-only model on the full dev score. '
                  'Keeping every learned boundary between identical matched labels also reduced the dev score. '
                  'S13 improved final test and in-domain row accuracy and boundary F1. External test ARI fell from '
                  '0.5388 to 0.4303, while in-domain ARI rose from 0.6322 to 0.6773; the full metrics retain these tradeoffs.',
                  '', '## Dev ablations', '',
                  '| Features / training | Threshold | Gap | Preserve boundaries with same matched label | Row accuracy | Boundary F1 | Score | Proposal F1 |',
                  '|---|---:|---:|---|---:|---:|---:|---:|'])
    for candidate in candidates:
        c,m=candidate['config'],candidate['mean']
        lines.append(f"| {c['variant']} | {c['threshold']} | {c['gap']} | {c['preserve']} | {m['row_acc']:.4f} | {m['boundary_f1']:.4f} | {m['score']:.4f} | {m['proposal_f1']:.4f} |")
    lines.extend(['','## Dev by held-out well','','| Well | Windows | S11 row accuracy | S11 boundary F1 | S13 row accuracy | S13 boundary F1 |','|---|---:|---:|---:|---:|---:|'])
    for well in training['training_wells']:
        a=[r for r in base if r['well']==well]
        b=[r for r in selected['records'] if r['well']==well]
        lines.append(f"| {well} | {len(b)} | {mean(a,'row_acc'):.4f} | {mean(a,'boundary_f1'):.4f} | {mean(b,'row_acc'):.4f} | {mean(b,'boundary_f1'):.4f} |")
    lines.extend(['','## Frozen model','','```json',json.dumps(training,indent=2),'```','',
                  f"Checkpoint size: {(HERE/'boundary_net.pt').stat().st_size:,} bytes.", '',
                  'Six dilated Conv1d/ReLU layers (dilations 1, 2, 4, 8, 16, 32), 24 hidden channels, one sigmoid output. '
                  'Weighted binary cross entropy uses positive weight 6 and Gaussian boundary targets with sigma 4 pixels. '
                  'Training samples wells uniformly, then windows within wells; batches contain eight 768-row crops, '
                  'with reversal, mild feature scaling and noise. AdamW uses learning rate 0.0015 and weight decay 0.001. '
                  'Inference runs on CPU with at most three torch threads and chunks of 8192 rows plus 128 rows of context.', '',
                  'Rich features contain whole-column and three-panel RGB/ink/transition means, dark-rule extent, '
                  'and changes measured at 4, 12 and 32 rows. The six-feature ablation uses Emil’s normalized RGB, ink, '
                  'horizontal transitions and vertical transitions. Legend matching reuses S11 Fourier/Lab descriptors, '
                  'distance weights, rejection threshold, sampling and unknown clusters.', '',
                  'Synthetic data: 72 procedural columns seeded 3000–3071, each 2048 rows, balanced across '
                  'three-panel colour fills, sparse Rock Types-style patterns and binary hatch. They randomize '
                  'bed thickness, pattern frequency/phase, rules, gaps, panel mixtures and noise. '
                  'Synthetic-augmented runs sample synthetic crops with probability 0.25. '
                  'No additional human-labelled non-eval logs were available in the standard truth directories; '
                  'none of the 13 in-domain eval logs or their labels were used for training or selection.', '',
                  'Per-window log, well, source, window coordinates and exact training truth are recorded in '
                  '`data/index.json`. Cached features/targets are in `data/*.npz`, fold checkpoints and held-well '
                  'probabilities in `runs/`, and final harness CSVs, figures and raw metrics in `harness_workspace/out/`.', '',
                  'The Emil baseline loads the existing bakeoff/emil/boundary_net.pt checkpoint. '
                  'Its archived pseudo-label training provenance includes some in-domain eval images (without eval truth), '
                  'so it is reported as an existing baseline rather than a cleanly retrained comparison. '
                  'S12’s explicit six-thread request is capped at three in the evaluation adapter, and its missing archived '
                  'default checkpoint path is redirected to that existing portable checkpoint. OCR cache reuse is by exact legend pixels.', '',
                  'Raw proposal comparisons (before legend matching and merges):'])
    for name in ('s11_proposals','s11_full_proposals','emil_proposals'):
        vals=selection['baselines'][name]
        lines.append(f"- {name}: {mean(vals,'boundary_f1'):.4f}.")
    lines.extend(['','The recorded S11 proposal ablation is its PELT pass alone; the full S11 rows above include dark-rule proposals and label merging. '
                  'The Emil proposal comparison uses its existing checkpoint and fixed 0.5 threshold. '
                  'The final comparison is recorded without post-test tuning.', '',
                  '## Reproduction', '', 'From `bakeoff/`, set `LITHOLOGY_DATASET` to the local dataset and cap OMP/MKL/OpenBLAS threads at three. '
                  'Use the repository’s Python environment.', '',
                  '```text','python -m ml.boundary.experiment prepare',
                  'python -m ml.boundary.experiment cv --steps 450',
                  'python -m ml.boundary.experiment select',
                  'python -m ml.boundary.diagnostics',
                  'python -m ml.boundary.experiment fit --steps 450',
                  'python -m pytest ml/boundary/test_boundary.py -o cache_dir=ml/boundary/.pytest_cache',
                  'python -m ml.boundary.evaluate --split dev --strategies s12',
                  'python -m ml.boundary.evaluate --split final --strategies s11,s12,s13',
                  'python -m ml.boundary.report','```','',
                  'The evaluation adapter invokes `harness.run.main` unchanged for scoring, depth filling, CSVs, PNGs, timing and resource measurement. '
                  'It redirects all outputs and caches into this lane. The final run marker prevents accidentally rerunning the holdout for selection.'])
    oracle_path=HERE/'runs'/'dev_oracle_naming.json'
    if oracle_path.exists():
        oracle=json.loads(oracle_path.read_text())
        lines.extend(['','## Naming diagnostic','',
                      f"Giving the unchanged S11 matcher exact dev boundaries yields {mean(oracle,'row_acc'):.4f} row accuracy. "
                      'This is a diagnostic for the existing matcher, not a universal upper bound: mixed intervals and omitted legend names still cause errors. '
                      'It explains why substantial boundary gains can produce smaller row-accuracy gains.'])
    (HERE/'MODEL_CARD.md').write_text('\n'.join(lines)+'\n')
    (HERE/'results.json').write_text(json.dumps(dict(summary=output,selected=selected['config'],dev_cv=selected['mean']),indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
