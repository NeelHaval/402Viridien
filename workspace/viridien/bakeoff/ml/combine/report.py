"""Write the S19 model card from allowed-data results only."""
import json
from pathlib import Path
import re
from ml.combine.runtime import HERE, BAKEOFF
from ml.combine.experiment import digest


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def row(label,item):
    d=item['dev']['well_macro'];s=item['synthetic']['well_macro']
    return f"| {label} | {d['row_acc']:.4f} | {d['boundary_f1']:.4f} | {d['cluster_ari']:.4f} | {d['score']:.4f} | {s['score']:.4f} | {item['objective']:.4f} |"


def main():
    selection=read('selection.json');ranking=selection['ranking'];best=selection['selected']
    cfg=read('selected.json');idx=read('index.json');fit=read('fit.json')
    comparisons=[]
    for label,filters in [
        ('Real models, joint decoder, v3',dict(boundary_fraction=0.,matcher_fraction=0.,mode='joint',threshold=.4)),
        ('Synthetic models, CNN cuts, v3',dict(boundary_fraction=.35,matcher_fraction=.35,mode='cnn',threshold=.55)),
        ('Synthetic models, joint decoder, v3',dict(boundary_fraction=.35,matcher_fraction=.35,mode='joint',threshold=.4,preserve='rules',min_bed=16,penalty=0.)),
        ('Real boundary, synthetic matcher, joint',dict(boundary_fraction=0.,matcher_fraction=.35,mode='joint',threshold=.4)),
        ('Synthetic boundary, real matcher, joint',dict(boundary_fraction=.35,matcher_fraction=0.,mode='joint',threshold=.4))]:
        found=next(r for r in ranking if r['config'].get('model_mix',1.)==1.
                   and all(r['config'][k]==v for k,v in filters.items()))
        comparisons.append(row(label,found))
    comparisons.append(row('Selected S19',best))
    table='\n'.join(comparisons)
    wells='\n'.join(f"| {w} | {sum(c['well']==w for c in idx['dev'])} | {v['row_acc']:.4f} | {v['boundary_f1']:.4f} | {v['cluster_ari']:.4f} | {v['score']:.4f} |" for w,v in best['dev']['by_well'].items())
    parser=read('parser_ablation.json')
    p1=parser['v1']['well_macro'];p3=best['dev']['well_macro']
    ceilings=parser['ceilings'];groups={r['well'] for r in ceilings}
    ceiling=lambda key:sum(sum(r[key] for r in ceilings if r['well']==w)/sum(r['well']==w for r in ceilings) for w in groups)/len(groups)
    depth=read('depth_check.json');audited=[r for r in depth if 'changed' in r]
    depth_changed=[r for r in audited if r['changed']]
    oldmae=sum(r['old_endpoint_mae_ft'] for r in audited)/len(audited) if audited else 0.
    newmae=sum(r['new_endpoint_mae_ft'] for r in audited)/len(audited) if audited else 0.
    resources=[]
    for path in sorted(HERE.glob('*_resources.json')):
        r=json.loads(path.read_text())
        resources.append(f"| {path.stem.removesuffix('_resources')} | {r['seconds']:.2f} | {r['peak_rss_gb']:.3f} |")
    resource_table='\n'.join(resources)
    checks=read('runtime_check.json')
    test_log=(HERE/'tests.log').read_text(encoding='utf-8')
    passed=re.search(r'(\d+) passed',test_log)
    if not passed or re.search(r'\d+ (failed|errors?)',test_log):
        raise RuntimeError('The checks must pass before READY.json is written')
    from ml.combine.runtime import fit_depth_points
    # The task names this old depth failure. Read its saved OCR evidence only.
    anchor_file=BAKEOFF/'out'/'_cache'/'depth'/'ext_kgs_EmmaWard1_p002.rapidocr.json'
    if anchor_file.exists():
        archived=json.loads(anchor_file.read_text(encoding='utf-8'))
        a,b,n,rmse=fit_depth_points(archived['points'])
        audit=dict(source='saved OCR anchors; no image or truth opened',
                   before=archived['calibration'],
                   after=dict(a=a,b=b,n_points=n,rmse_px=rmse))
        (HERE/'depth_anchor_check.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    card=f'''# S19 model card

S19 uses the S18 joint decoder, the v3 legend parser, and the S17 model design.
All S19 model fits use the new data split. The final benchmark is pending.
Use `--strategy s19` in the shared harness. Only the team lead can run the final benchmark.

## Data limits

The dev set has 51 windows from five wells. ForsythB2-17 is outside this set.
This lane did not open its images or truth, or use its queries for a fit.
This lane did not search for the three sealed verify wells.
This lane did not run ext_test, ext_shared_verify or viridien_eval.
This lane did not open their truth files.
Earlier reports contain old final scores. This lane read those reports as requested.
Those scores did not set the new grid or select S19.

The S17 synthetic set has 1,920 train windows and 480 held windows.
The check uses 84 held windows, with 12 windows per family. The split uses source logs.
No source log occurs in both parts. No source is one of the 13 eval logs.
The check supplies the native legend swatches and names. It does not test legend OCR.
The D-582 and MC765 sources share legend art across logs. Their check does not test new art.
There is no human interval truth for non-eval in-domain logs in this mirror.
Thus, the in-domain selection check uses synthetic data from non-eval source logs.

The dev list SHA-256 is `{idx['dev_sha256']}`.
The fit files list the five wells and each excluded query well.
The code checks that the dev list still has 51 windows before each main step.
It does not use eval_set_dev_full.json.

## Old model limits

The old S13, S14, S17 and S18 fits used ForsythB2-17 before the split changed.
Their weights are contaminated under the new split rule.
S19 does not load those weights or the old confidence model.
The boundary fits start from random weights. The matcher fits start from new estimates.
This lane also did not load the old selected parser file.
The parser uses explicit defaults: `{json.dumps(idx['parser'],sort_keys=True)}`.
S19 reuses model code, synthetic images, and raw OCR evidence from earlier lanes.
The inherited code and parser vocabulary came from earlier work on the old split.
The clean refit removes weight contamination. It cannot remove the history of that code.

## Method and selection

The boundary net has six dilated convolution layers and 24 hidden channels.
It uses 32 row features. The pair matcher uses 27 visual features and logistic regression.
There are two training choices: real data only, or 35% synthetic data.
Each choice has five fits that exclude one well, and one fit for all five wells.
The matcher also excludes the query well and all its swatch copies.
The synthetic train sources stay separate from the held source logs.
The selected net and matcher use only allowed train data. They do not use held synthetic data.

The grid has {len(ranking)} settings. {sum(r['eligible'] for r in ranking)} settings keep the requested joint decoder
and a nonzero contribution from both synthetic-mixed models.
Only these settings can be selected for S19. The other settings are comparison checks.
This limit follows the lane task.
The first grid had 22 settings. The mixed models lost dev accuracy in this check.
We then checked six blends of the real-only and synthetic-mixed model outputs.
The boundary and matcher use the same blend weight. S19 gives the mixed models
weight {cfg.get('model_mix',1.):g}. It gives the real-only models the remaining weight.
Both grids use the same CV results. This extra selection can increase optimism.
The score for each window is row accuracy times boundary F1.
The selection value is 75% dev mean by well plus 25% synthetic mean by family.
The same CV results select the settings. These values are selection estimates.
They are not scores from a new blind test. The reported well means give each well equal weight.
The small dev set can give weak estimates for a new well or scan type.

| Method | Dev row | Dev boundary | Dev ARI | Dev score | Synthetic score | Selection value |
|---|---:|---:|---:|---:|---:|---:|
{table}

These comparisons use new fits and the same allowed data. They are not reruns of old final scores.
The first row keeps the S18 joint rule with new real-only models and v3 names.
The second row uses CNN cuts with new S17-style synthetic-mixed models and v3 names.
It keeps the same drawn-rule policy as the other grid settings.

| Dev well | Windows | Row accuracy | Boundary F1 | ARI | Score |
|---|---:|---:|---:|---:|---:|
{wells}

The selected settings are:

```json
{json.dumps(cfg,indent=2,sort_keys=True)}
```

## Parser check

A fixed check replaces v3 with v1 on the 51 dev windows after selection.
It keeps the selected model fits and all decoder settings. It does not refit the matcher.
Thus, it measures the parser change with a matcher trained with v3 entries.
The dev mean by well changes from row accuracy {p1['row_acc']:.4f} and score {p1['score']:.4f}
with v1 to row accuracy {p3['row_acc']:.4f} and score {p3['score']:.4f} with v3.
The fraction of filled dev rows with an exact name in the parsed legend changes
from {ceiling('v1'):.4f} to {ceiling('v3'):.4f}, with equal weight per well.
Scores use the shared exact-name rule. They do not use expanded rock-name aliases.

## Depth repair

The old report lists a 384.4 ft mean boundary error for ext_kgs_EmmaWard1_p002.
This lane inspected its saved OCR anchors and calibration. It did not open its image or truth.
The old fit chose 3.66429 ft per pixel. It ranked anchor span before residual error.
Three printed anchors, 4000, 4050 and 4100, support about 0.13323 ft per pixel.
The new fit ranks equal support by pixel residual before span.
The regression check uses the saved OCR anchors only. No truth sets the correction.
S19 reads small depth-track crops when the input calibration has few anchors,
a large residual, or a negative top depth. It accepts a better fit with at least three anchors.
It uses full-image rows and keeps the unit conversion. It returns explicit depths,
so the harness keeps the repaired scale in the six-column output.
There are no log-name rules or fixed depth offsets.
The final benchmark must confirm the depth error on that held window.

The fixed dev audit checked {len(audited)} unique logs and changed {len(depth_changed)} calibrations.
Mean absolute error at all exact dev interval endpoints was {oldmae:.4f} ft before
and {newmae:.4f} ft after the check. Each log has equal weight in these two means.
This is a depth check, not the harness metric over matched boundaries.

## Checks and limits

`tests.log` records checks for split limits, source separation, the depth failure,
several depth scales, duplicate OCR, metre conversion, blank gaps, drawn rules,
absolute pixel rows, and excluded-well fit records.
All {passed.group(1)} checks passed.
The deployment check uses two allowed dev windows. Its predictions match the saved-data path.
Both return ordered, non-overlapping intervals inside the requested window.
The larger runtime in this check was {max(r['seconds'] for r in checks):.3f} seconds.
This is a smoke check on dev. It is not a runtime bound for all logs.
The model exports use numeric arrays and CPU torch weights. All checks use CPU.
The lane limits torch, BLAS, OMP and MKL to three compute threads. OpenCV uses one.
The code derives paths from the repo or the dataset environment variable.
There are no fixed laptop or VM paths in the code.
All lane files are under ml/combine or the s19 strategy file. There were no commits or pushes.

| Step | Seconds | Peak RSS GiB |
|---|---:|---:|
{resource_table}

S19 returns visual match costs and margins as extra fields. They are not calibrated probabilities.
The harness writes its normal six CSV columns. S19 does not use the old confidence weights.
Short beds, incorrect OCR names, mixtures, and missing swatches remain possible failure modes.
The held synthetic check uses artificial scan effects. Real scan results can differ.
No final-set score is claimed here. The team lead will run the one final benchmark.

## Reproduce

Set LITHOLOGY_DATASET to the local dataset. Use the project Python environment.
Run these commands from bakeoff. Set OMP_NUM_THREADS, MKL_NUM_THREADS and
OPENBLAS_NUM_THREADS to 3. Set PYTHONDONTWRITEBYTECODE to 1 to keep shared code read-only.

```sh
python -m ml.combine.experiment prepare
python -m ml.combine.experiment train --steps 450
python -m ml.combine.experiment select
python -m ml.combine.experiment select_blends
python -m ml.combine.experiment fit
python -m ml.combine.experiment check
python -m ml.combine.experiment compare
python -m ml.combine.experiment depth_check
python -m pytest ml/combine/test_combine.py -q -o cache_dir=ml/combine/.pytest_cache
python -m ml.combine.report
```

The experiment has no command for final sets. Do not use the normal harness on them in this lane.
READY.json records code and model hashes for the team lead.
'''
    (HERE/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    paths=[HERE/'runtime.py',HERE/'selected.json',HERE/'boundary_net.pt',HERE/'matcher.npz',
           BAKEOFF/'strategies'/'s19_combine.py',BAKEOFF/'ml'/'integrate'/'core.py',
           BAKEOFF/'ml'/'boundary'/'model.py',BAKEOFF/'ml'/'match'/'features.py',
           BAKEOFF/'strategies'/'s14_learned_match.py',BAKEOFF/'strategies'/'s02_features_pelt.py',
           BAKEOFF/'harness'/'legend_v3.py',BAKEOFF/'harness'/'legend_v2.py',
           BAKEOFF/'harness'/'legend.py',BAKEOFF/'harness'/'depth.py']
    if cfg.get('model_mix',1.)<1.:
        paths += [HERE/'boundary_real.pt',HERE/'matcher_real.npz']
    (HERE/'READY.json').write_text(json.dumps(dict(strategy='s19',final_benchmark_run=False,
          dev_windows=51,training_wells=fit['training_wells'],inherited_weights_loaded=False,
          hashes={str(p.relative_to(BAKEOFF)):digest(p) for p in paths}),indent=2),encoding='utf-8')
    print('Wrote MODEL_CARD.md and READY.json')


if __name__=='__main__':
    main()
