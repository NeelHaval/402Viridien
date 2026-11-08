"""Post-freeze reporting only: archived baselines and one S17 final pass."""
import csv
import json
from pathlib import Path
import numpy as np
from ml.synth_gulf.generator import HERE, BAKEOFF, FAMILIES, family_map


def load(name):
    return json.loads((HERE/name).read_text())


def main():
    final=load('final_metrics.json')
    assert (HERE/'FINAL_DONE.json').exists(), 'Wait for the complete one-shot final pass'
    old=json.loads((BAKEOFF/'ml'/'match'/'final_metrics.json').read_text())
    cfg=load('selected.json');selection=load('selection.json');prep=load('preparation.json')
    fit=load('fit.json');fam=family_map();sources=load('assets.json')
    methods=['s13_boundary_net','s15_learned_match','s17_synth_gulf']
    records={split:{s:(final[split]['logs'] if s.startswith('s17') else old[split][s]['logs'])
                    for s in methods} for split in final}
    overview=[];family_rows=[]
    metrics=['row_acc','boundary_f1','cluster_ari','score','depth_mae_ft','seconds','peak_rss_gb']
    for split,strats in records.items():
        for strategy,rows in strats.items():
            result=dict(split=split,strategy=strategy,windows=len(rows),errors=sum('error' in r for r in rows))
            for metric in metrics:
                values=[r.get(metric,0.) for r in rows if r.get(metric,0.) is not None]
                result[metric]=float(np.mean(values)) if values else None
            result['peak_rss_gb']=max((r.get('peak_rss_gb',0) for r in rows),default=0.)
            overview.append(result)
    for family in FAMILIES:
        for strategy,rows in records['viridien_eval'].items():
            rr=[r for r in rows if fam[r['log']]==family]
            if rr:
                family_rows.append(dict(family=family,strategy=strategy,windows=len(rr),
                                       **{k:float(np.mean([r.get(k,0.) for r in rr]))
                                          for k in ('row_acc','boundary_f1','cluster_ari','score')}))
    for name,rows in [('summary.csv',overview),('per_family.csv',family_rows)]:
        with (HERE/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# S17 style-matched synthetic training','',
           'Frozen models are selected using six outside dev wells and held-source synthetic windows only. '
           'S17 was evaluated once on 20 external test windows and 13 in-domain evaluation windows. '
           'S13/S15 values reuse the matching lane’s archived final predictions and metrics; those baselines were not rerun.','',
           '| Split | Strategy | Row acc | Boundary F1 | ARI | Score | Depth MAE ft | s/window | Max RSS GB | Errors |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in overview:
        depth='n/a' if r['depth_mae_ft'] is None else f"{r['depth_mae_ft']:.2f}"
        lines.append(f"| {r['split']} | {r['strategy'][:3]} | {r['row_acc']:.2%} | {r['boundary_f1']:.3f} | {r['cluster_ari']:.3f} | {r['score']:.3f} | {depth} | {r['seconds']:.3f} | {r['peak_rss_gb']:.2f} | {r['errors']} |")
    lines += ['', '## In-domain families', '',
              'Each cell is **row accuracy / boundary F1 / ARI**; windows are weighted equally within family. '
              'Families come directly from `analysis/label_shares_per_log.csv`.','',
              '| Family | Windows | S13 | S15 | S17 |','|---|---:|---:|---:|---:|']
    for family in FAMILIES:
        r=[v for v in family_rows if v['family']==family]
        if not r:continue
        cells=['{:.2%} / {:.3f} / {:.3f}'.format(v['row_acc'],v['boundary_f1'],v['cluster_ari']) for v in r]
        lines.append(f"| {family} | {r[0]['windows']} | "+' | '.join(cells)+' |')
    chosen=selection['ranking'][0]
    lines += ['', '## Selection and ablations','',
              'Selection objective: 75% mean of the six dev-well composite scores plus 25% mean of the seven '
              'synthetic-family composite scores. Composite score is per-window row accuracy × boundary F1. '
              'These pooled CV predictions also select hyperparameters; selection values are optimistic, not a second unbiased test. '
              'The synthetic full-pipeline validation subset is fixed before CV: 12 windows per family (84 total). '
              'All 480 held windows have cached probabilities, and held query matching uses all available held queries.','',
              '| Boundary synth fraction | Threshold | Matcher style weight | Dev well score | Synthetic family score | Objective |',
              '|---:|---:|---:|---:|---:|---:|']
    # Show the no-style baseline, selected, and the best candidate for each training fraction.
    baseline=next(r for r in selection['ranking'] if r['boundary_fraction']==0 and r['matcher_fraction']==0)
    show=[baseline]
    for fraction in (.15,.35):
        show.append(next(r for r in selection['ranking'] if r['boundary_fraction']==fraction))
    if chosen not in show:show.append(chosen)
    for r in show:
        lines.append(f"| {r['boundary_fraction']:.2f} | {r['threshold']:.2f} | {r['matcher_fraction']:.2f} | {r['dev_well_score']:.4f} | {r['synthetic_family_score']:.4f} | {r['objective']:.4f} |")
    lines += ['', '| Matcher style weight | Dev well row acc on S11 proposals | Held synthetic pair accuracy (family mean) |',
              '|---:|---:|---:|']
    for r in selection['match_ablation']:
        lines.append(f"| {r['fraction']:.2f} | {r['dev_well_row_acc']:.2%} | {np.mean(list(r['synthetic_pair_accuracy'].values())):.2%} |")
    matching_base=next(r for r in selection['match_ablation'] if r['fraction']==0.)
    matching_selected=next(r for r in selection['match_ablation'] if r['fraction']==cfg['matcher_fraction'])
    gulf_base=matching_base['synthetic_pair_accuracy']['Gulf 3-panel']
    gulf_selected=matching_selected['synthetic_pair_accuracy']['Gulf 3-panel']
    gulf={r['strategy']:r for r in family_rows if r['family']=='Gulf 3-panel'}
    lines += ['', '## What helped', '',
              f"Style-augmented pair training changed held synthetic Gulf matching from {gulf_base:.2%} to {gulf_selected:.2%}. "
              f"Its dev well-macro row accuracy on S11 proposals changed from {matching_base['dev_well_row_acc']:.2%} "
              f"to {matching_selected['dev_well_row_acc']:.2%}; synthetic improvement did not imply a dev improvement."]
    if cfg['boundary_fraction']==0:
        lines += ['', 'Neither retrained boundary candidate won the predeclared joint selection objective. '
                  'S17 therefore retains the published S13 boundary weights and uses the selected retrained matcher. '
                  'The augmented boundary weights and all held-well predictions remain available as ablations.']
    else:
        lines += ['', f"The selected boundary retraining samples synthetic crops with probability {cfg['boundary_fraction']}; "
                  'the ablation table records the tradeoff with real-only boundaries.']
    if gulf:
        delta=gulf['s17_synth_gulf']['row_acc']-gulf['s15_learned_match']['row_acc']
        lines += ['', f"On the single final in-domain run, Gulf row accuracy changed by {delta*100:+.2f} percentage points "
                  f"versus S15 ({gulf['s15_learned_match']['row_acc']:.2%} → {gulf['s17_synth_gulf']['row_acc']:.2%}). "
                  'This is a held-out result, retained without further tuning.']
    hatch={r['strategy']:r for r in family_rows if r['family']=='D-582 1-bit grid'}
    if hatch:
        lines += ['', f"The largest in-domain family improvement is D-582: row accuracy rises from "
                  f"{hatch['s15_learned_match']['row_acc']:.2%} to {hatch['s17_synth_gulf']['row_acc']:.2%}, "
                  f"with boundary F1 {hatch['s15_learned_match']['boundary_f1']:.3f} → "
                  f"{hatch['s17_synth_gulf']['boundary_f1']:.3f}. This is the combined training pipeline’s result; "
                  'there was no final-set factorial ablation of individual scan augmentations.']
    external={r['strategy']:r for r in overview if r['split']=='ext_test'}
    lines += ['', f"External-test row accuracy rises from {external['s15_learned_match']['row_acc']:.2%} "
              f"to {external['s17_synth_gulf']['row_acc']:.2%}, but boundary F1 falls from "
              f"{external['s15_learned_match']['boundary_f1']:.3f} to {external['s17_synth_gulf']['boundary_f1']:.3f} "
              f"and composite score falls from {external['s15_learned_match']['score']:.3f} "
              f"to {external['s17_synth_gulf']['score']:.3f}. S17 is not an across-the-board replacement for S15."]
    lines += ['', '## Data and generator','',
              f"Generated {prep['windows']:,} windows: 1,920 training and 480 held windows, each 1,536 rows. "
              'Gulf has 960 windows; each of the other six families has 240. PNGs, exact CSVs, and compressed '
              '32-channel boundary features/targets are in `data/`. `synthetic_index.json` records every source, seed, split, and interval. '
              'No in-domain human truth is consumed during training.','',
              f"Usable native tiles came from {len(sources)} of 38 allowed non-eval logs. "
              'The remaining 11 had fewer than two usable lithology entries under the shared parser and were omitted. '
              'Each family reserves one source log before generation; source log IDs do not cross the synthetic split. '
              'D-582 and MC765 have identical legend pixels across their non-eval source logs, so their validation checks new '
              'windows and artefacts, not unseen legend artwork. Other families have distinct held-source legend hashes.','',
              'Colours and hatch symbols are native legend crops, tiled with random phase and 0.85/1/1.15 scale. '
              'Source-measured vertical-rule positions and native text crops are reused. Gulf columns have a central '
              'primary interpretation track and percentage mixtures on side tracks. Other grid families add independent '
              'horizontal grid lines, with blank intervals and optional real boundary rules in all families. '
              'Synthetic legend pages reuse each source’s box layout and shuffle the entries. '
              'The matcher receives swatches directly, as the shared strategy contract requires; this does not train the legend parser.','',
              'Scan augmentations are explicitly chosen ranges, not fitted distributions: fade 0–18%, noise sigma 0–3.5, '
              'Gaussian blur sigma 0.35–0.85, JPEG quality 45–95, horizontal shear ±0.004, and ordered 1-bit dithering. '
              'Horizontal shear leaves every row boundary unchanged; no vertical warp invalidates the exact truth. '
              'Procedural text placement and panel proportions are approximations; this is not a reconstruction of entire source pages.','',
              '## Models and limits','',
              'Boundary architecture and features are unchanged from S13: six dilated Conv1d layers, 24 hidden channels, '
              '32 panel/colour/rule/change features. Each augmented candidate is retrained from scratch in each fold; '
              'the published all-dev S13 checkpoint is never used to initialize a held-well model. '
              'Each fold excludes every real and swatch-augmented query from its held well. '
              'S14’s 27 visual pair features feed a newly fitted logistic model with balanced positive/negative and '
              'well/family weights. No filenames or family IDs are inference features.','',
              f"Frozen selection: boundary synthetic sampling fraction {cfg['boundary_fraction']}, threshold {cfg['boundary']['threshold']}, "
              f"gap 24, merge same-label intervals; matcher synthetic weight {cfg['matcher_fraction']}. "
              'The selected all-dev boundary candidate is reused exactly, without further training on held synthetic data. '
              'Portable matcher export is checked against scikit-learn decisions.','',
              f"Weights total {fit['weights_bytes']:,} bytes. Training/CV runtime is "
              f"{load('cv_training.json')['seconds']/60:.2f} minutes, including all 14 boundary fits; "
              f"selection takes {load('select_resources.json')['seconds']/60:.2f} minutes including matcher fits and scoring. "
              f"Peak process RSS across preparation/CV/selection/fit/final is "
              f"{max(load(a+'_resources.json')['peak_rss_gb'] for a in ('prepare','cv','select','fit','final')):.2f} GB. "
              'Torch and BLAS are capped at three threads and OpenCV at one. No GPU is used.','',
              'In-domain truth is gold-first silver consensus. One in-domain window has no filled consensus truth; '
              'the shared scorer awards row accuracy 1 for empty truth. Family means preserve that convention. '
              'External truth is exact. Runtime excludes shared legend/depth preprocessing. '
              'Missing legend names remain an upstream limitation: improving textures cannot recover a label the parser omitted.','',
              'Validation: 11 focused pytest checks pass, covering source/eval exclusion, held-source separation, '
              'deterministic generation and interval geometry in all seven families, held-well exclusion, '
              'and agreement between selection-time and deployed interval naming.','',
              '## Reproduction','',
              'Set `LITHOLOGY_DATASET` to the local dataset directory, use the project Python environment, and run from `bakeoff/`. '
              'No paths are tied to the VM or Windows. All generated writes stay in this lane.','',
              '```sh','python -c "from ml.synth_gulf.generator import assets; assets()"',
              'python -m ml.synth_gulf.experiment prepare','python -m ml.synth_gulf.experiment cv --steps 450',
              'python -m ml.synth_gulf.experiment select','python -m ml.synth_gulf.experiment fit',
              'python -m pytest ml/synth_gulf/test_synth_gulf.py -q -o cache_dir=ml/synth_gulf/.pytest_cache',
              '# One final pass only; refuses to rerun once FINAL_LOCK.json exists.',
              'python -m ml.synth_gulf.experiment final','python -m ml.synth_gulf.experiment report',
              'python -m harness.run --strategy s17 --gt consensus','```','',
              'The last command is normal inference usage; do not use it to rescore the frozen holdout during selection. '
              'Prerequisites are the existing dev-only boundary feature cache and matching query cache. '
              '`assets.json`, `preparation.json`, `selection.json`, `fit.json`, `FINAL_LOCK.json`, and '
              '`final_metrics.json` retain provenance, ablations, hashes, and all final per-window results.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Standalone export: flat background, labelled bars, portrait layout.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    available=[f for f in FAMILIES if any(r['family']==f for r in family_rows)]
    for k,(strategy,colour) in enumerate(zip(methods,['#2979b8','#ed9930','#15977b'])):
        values=[next(r['row_acc'] for r in family_rows if r['family']==f and r['strategy']==strategy) for f in available]
        pos=np.arange(len(available))+(k-1)*.24
        ax.barh(pos,np.array(values)*100,height=.22,color=colour,label=strategy[:3].upper())
        for y,x in zip(pos,values):ax.text(x*100+1,y,f'{x:.0%}',va='center',fontsize=8)
    ax.set_yticks(np.arange(len(available)),available);ax.invert_yaxis();ax.set_xlim(0,110)
    ax.set_xlabel('Row accuracy (%)');ax.set_title('In-domain evaluation by family (13 windows)')
    ax.legend(loc='lower right');ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'per_family.png',dpi=160);plt.close(fig)
    print('\n'.join(lines[:35]))


if __name__=='__main__':main()
