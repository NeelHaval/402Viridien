"""Render a compact, reproducible lane report from recorded results."""
from pathlib import Path
import json
import platform
import numpy as np
import sklearn

HERE=Path(__file__).resolve().parent


def main():
    dev=json.loads((HERE/'dev_selection.json').read_text())
    base=json.loads((HERE/'dev_baseline.json').read_text())
    final=json.loads((HERE/'final_metrics.json').read_text())
    selected=json.loads((HERE/'selected.json').read_text())
    lines=['Learned legend matching lane', '',
        'Selected on six external dev wells only, using leave-one-well-out validation. '
        'Every real and synthetic query from a held-out well is excluded from that fold\'s training. '
        'The 20 external test windows and 13 in-domain evaluation windows were opened only in the locked final run.', '',
        f"The selected scorer is {selected['learner']} on 27 visual pair distances, trained with balanced well weights. "
        'Features include multiscale FFT, Lab colour, ink density, stroke run lengths, gradient orientation, '
        'and local pixel comparisons. No filenames or label vocabulary enter the model. '
        'The numeric matcher.npz checkpoint runs without scikit-learn pickle compatibility requirements.', '',
        'S14 uses the original S11 PELT and drawn-rule boundary proposals; S15 uses the published S13 checkpoint. '
        'Matching training uses exact dev legend swatches and merged truth intervals, plus phase/scale-augmented dev swatch tiles. '
        'A linear model, boosted trees, FFT, multiscale FFT, and stroke-run matching were compared. '
        'No CNN embedding was trained. Rejection is disabled because an Unknown prediction cannot improve exact-name row accuracy on this truth.', '',
        '| Dev row accuracy | S11 | Selected matcher (out of fold) |',
        '|---|---:|---:|',
        f"| Macro over six wells | {100*base['well_macro']:.2f}% | {100*dev['ranking'][0]['well_macro']:.2f}% |",
        f"| Macro over 59 windows | {100*base['window_macro']:.2f}% | {100*dev['ranking'][0]['window_macro']:.2f}% |", '',
        '| Held-out dev well | S11 | Selected matcher |', '|---|---:|---:|']
    key=json.dumps({k:selected[k] for k in ('method','learner','blend')},sort_keys=True)
    for fold in dev['folds']:
        lines.append(f"| {fold['well']} | {100*base['wells'][fold['well']]:.2f}% | {100*fold['candidates'][key]:.2f}% |")
    diag=dev['diagnostic_mean']
    lines += ['', 'Parser versus matcher (59 dev-window means; scorer remains out of fold):', '',
        '| Legend entries | True intervals | S11 boundary proposals |', '|---|---:|---:|',
        f"| Shared parser boxes/names | {100*diag['parsed_true_intervals']:.2f}% | {100*diag['parsed_s11_proposals']:.2f}% |",
        f"| Oracle boxes/names | {100*diag['oracle_true_intervals']:.2f}% | {100*diag['oracle_s11_proposals']:.2f}% |", '',
        f"Parsed-name availability limits perfect matching to {100*diag['parser_ceiling']:.2f}% row accuracy. "
        f"Missing, excluded, or misread names therefore account for at least {100*(1-diag['parser_ceiling']):.2f} percentage points of error. "
        f"With true intervals, {100*(diag['parser_ceiling']-diag['parsed_true_intervals']):.2f} further points are lost in choosing among available names. "
        f"Oracle boxes/names improve true-interval accuracy by {100*(diag['oracle_true_intervals']-diag['parsed_true_intervals']):.2f} points; "
        f"the remaining {100*(1-diag['oracle_true_intervals']):.2f} points reflect matching/visual ambiguity even with oracle legends. "
        'These are complementary diagnostics, not disjoint additive attributions: correcting names also changes the set and quality of swatches.', '',
        'Single final evaluation (window-macro means; shared parser and depth calibration; in-domain truth is gold-first silver consensus):', '',
        '| Split | Strategy | Row acc | Boundary F1 | Cluster ARI | Depth MAE (ft) | Seconds/window | Peak RSS (GB) | Errors |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split,strategies in final.items():
        for strategy in ('s11_fft_legend','s12_emil','s13_boundary_net','s14_learned_match','s15_learned_match'):
            if strategy not in strategies:continue
            records=strategies[strategy];m=records['mean']
            peak=max((r.get('peak_rss_gb',0) for r in records['logs']),default=0)
            num=lambda key,digits=3:'n/a' if m.get(key) is None else f'{m[key]:.{digits}f}'
            row='n/a' if m.get('row_acc') is None else f"{100*m['row_acc']:.2f}%"
            lines.append(f"| {split} | {strategy.split('_')[0].upper()} | {row} | {num('boundary_f1')} | {num('cluster_ari')} | {num('depth_mae_ft',1)} | {num('seconds',2)} | {peak:.2f} | {m['crashes']} |")
    errors={split:{s:sorted({r['error'] for r in v['logs'] if 'error' in r}) for s,v in strategies.items() if any('error' in r for r in v['logs'])} for split,strategies in final.items()}
    if any(errors.values()):
        lines += ['', 'Recorded runtime failures:', '']
        for split,strategies in errors.items():
            for strategy,messages in strategies.items():
                lines.append(f"- {split}, {strategy}: "+'; '.join(messages))
        lines += ['', 'S12\'s missing EasyOCR weights and failed network download make its affected in-domain baseline incomplete. '
            'Those windows retain zero predictions and were not rerun.']
    lines += ['', 'Error details and per-window timings are retained in final_metrics.json. '
        'Preprocessing failures produce empty predictions for every strategy on that window, consistent with the harness. '
        'One in-domain consensus window has no filled truth rows: the shared scorer returns row_acc=1 for empty truth, even for an empty prediction. '
        'Reported window means retain that convention; S12 therefore receives this credit despite its OCR error on that window. '
        'Runtime measurements exclude shared OCR/calibration. S12 uses its strategy-default Drive checkpoint; the exact hash is recorded. '
        'CPU thread requests are capped at three; no GPU or paid APIs are used.', '',
        'Reproduce from bakeoff with the repository Python environment:', '', '```sh',
        'python -m ml.match.experiment prepare', 'python -m ml.match.experiment select',
        'python -m ml.match.experiment baseline', 'python -m pytest ml/match/test_match.py -q',
        '# final is guarded: it refuses a second run when FINAL_LOCK.json exists',
        'python -m ml.match.experiment final', 'python -m ml.match.report', '```', '',
        'Use LITHOLOGY_DATASET for the in-domain dataset location. External data paths are derived from the repository. '
        'For normal harness inference use --strategy s14 or --strategy s15. '
        'All dev data, selection results, final CSVs and caches for this lane are under bakeoff/ml/match/. '
        'The runtime strategies never import this experiment module or access truth.', '',
        f'Environment: Python {platform.python_version()}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}. '
        'Validation: six focused matching tests pass; exporting the selected numeric checkpoint changed zero dev decisions.', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
