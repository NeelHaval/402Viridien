"""Write the S21 report from this lane's records only."""
import json
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ml.synth_v2.generator import HERE, FAMILIES


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def main():
    prep = read('preparation.json')
    selection = read('selection.json')
    fit = read('fit.json')
    smoke = read('smoke.json')
    best = selection['ranking'][0]
    records = selection['records'][best['key']]
    resources = {name:read(name+'_resources.json') for name in ('prepare','train','select')}
    sources = read('assets.json')
    # A narrow two-column view keeps the labels legible on a phone.
    preview = Image.new('RGB',(480,2640),'white')
    pen = ImageDraw.Draw(preview)
    font = ImageFont.load_default(size=15)
    corpus = read('synthetic_index.json')
    for i,family in enumerate(FAMILIES):
        r = next(r for r in corpus if r['family']==family and r['selection'])
        x0,y0 = (i%2)*240,(i//2)*660
        pen.multiline_text((x0+10,y0+10),textwrap.fill(family,25),fill='black',font=font,spacing=4)
        with Image.open(HERE/'data'/(r['key']+'.png')) as image:
            image.thumbnail((220,590))
            preview.paste(image,(x0+10,y0+60))
    preview.save(HERE/'synthetic_preview.png')
    lines = ['# S21 synthetic model report', '',
        'S21 is ready for the team benchmark. The team must run the final benchmark.',
        'This lane did not run the final benchmark.', '',
        '## Data limits', '',
        'The current dev set has 51 windows from five wells. ForsythB2-17 is not in this set.',
        'No sealed verify well was opened or rebuilt. No external test truth was opened.',
        'No truth from the 13 in-domain eval logs was opened.',
        'The eval list was used only to exclude source logs.',
        'The old full dev list was not used.', '',
        'All real data were read again from the current dev list. Old real training caches were not used.',
        'Each CV fold leaves out one whole well. Boundary and mixed matcher fits use the same well split.',
        'The final boundary fit uses the five allowed dev wells and training synthetic sources.',
        ('The selected matcher uses synthetic pairs only.' if best['matcher_fraction']==1.
         else 'The selected matcher uses real and synthetic pairs.'),
        'The mixed matcher candidates use the same well folds.', '',
        '## Old model data', '',
        'S17 used the old six-well dev set. That set included ForsythB2-17.',
        'Thus, the old S17 boundary and matcher weights are not clean for this split.',
        'S21 does not load these weights. All boundary fits start with random weights.',
        'All matcher fits start again from visual pair features.',
        'S21 uses S17 code and visual tiles from allowed non-eval source logs.',
        'The old S17 report was read as a code reference. It contains old final scores.',
        'Those scores are not used in the S21 selection rule. No old final prediction file was read.',
        'The inherited design has prior benchmark exposure. A new fit does not remove that design history.', '',
        '## Synthetic data', '',
        f"There are {prep['windows']:,} exact synthetic windows. Each window has {prep['height']} rows.",
        'There are 8,000 training windows and 2,000 held windows.',
        'The source log sets do not cross the training and held split.',
        'The held set supplies 20 fixed selection windows per family, for 140 windows in total.',
        'The other held windows supply pair checks or remain unused.', '',
        '| Family | Total windows | Training source logs | Held source logs |',
        '|---|---:|---:|---:|']
    from ml.synth_v2.experiment import COUNTS
    for family, count in zip(FAMILIES, COUNTS):
        lines.append(f"| {family} | {count} | {sum(s['family']==family and s['split']=='train' for s in sources)} | {sum(s['family']==family and s['split']=='held' for s in sources)} |")
    lines += ['',
        'The generator adds grid rules to Geoservices, MC765, and D-582 columns.',
        'Grid rows do not have to match a class change. D-582 output has only black and white pixels.',
        'MC761 columns have side panels with mineral textures. The centre panel keeps the true class.',
        'The generator keeps Gulf panels, blank spans, text, fade, blur, JPEG loss, and horizontal shear.',
        'Horizontal shear does not change the row truth. No vertical warp is used.',
        'The source tiles and exact interval rows remain available in this lane.',
        'The images can be made again from the source record and seed.',
        'D-582 and MC765 source logs can share the same legend art.',
        'A held source log does not always mean new legend art.',
        'The layouts are approximations. They are not full source pages.', '',
        '## Selection checks', '',
        'The selection score is 0.75 times the mean dev well score plus 0.25 times the mean synthetic family score.',
        'The window score is row accuracy times boundary F1.',
        'CV results choose the settings. Thus, these results are selection results, not an unbiased final test.',
        'Two boundary mixes, four probability thresholds, and three matcher mixes were checked.',
        'The boundary gap is 24 rows. Adjacent equal labels are merged.', '',
        f"Selected boundary synthetic fraction: {best['boundary_fraction']:.2f}.",
        f"Selected boundary threshold: {best['threshold']:.2f}.",
        f"Selected matcher synthetic fraction: {best['matcher_fraction']:.2f}.", '',
        '| Selection set | Row accuracy | Boundary F1 | ARI | Score |',
        '|---|---:|---:|---:|---:|']
    for split in ('dev','synthetic'):
        m = best['macro'][split]
        lines.append(f"| {split} group mean | {m['row_acc']:.4f} | {m['boundary_f1']:.4f} | {m['cluster_ari']:.4f} | {m['score']:.4f} |")
    lines += ['', '| Dev well | Windows | Row accuracy | Boundary F1 | Score |',
              '|---|---:|---:|---:|---:|']
    for group in sorted({r['group'] for r in records['dev']}):
        vals = [r for r in records['dev'] if r['group']==group]
        means = {k:np.mean([r[k] for r in vals]) for k in ('row_acc','boundary_f1','score')}
        lines.append(f"| {group} | {len(vals)} | {means['row_acc']:.4f} | {means['boundary_f1']:.4f} | {means['score']:.4f} |")
    lines += ['', '| Held synthetic family | Row accuracy | Boundary F1 | Score |',
              '|---|---:|---:|---:|']
    for group in FAMILIES:
        vals = [r for r in records['synthetic'] if r['group']==group]
        means = {k:np.mean([r[k] for r in vals]) for k in ('row_acc','boundary_f1','score')}
        lines.append(f"| {group} | {means['row_acc']:.4f} | {means['boundary_f1']:.4f} | {means['score']:.4f} |")
    weakest = min(FAMILIES,key=lambda g:np.mean([r['boundary_f1'] for r in records['synthetic'] if r['group']==g]))
    weakest_f1 = np.mean([r['boundary_f1'] for r in records['synthetic'] if r['group']==weakest])
    lines += ['',f"The weakest synthetic boundary result is {weakest}, with F1 {weakest_f1:.4f}."]
    lines += ['', '## Model size and time', '',
        'The boundary model has six dilated Conv1d layers with 24 hidden channels.',
        'It uses the same 32 row features as S13. The matcher uses 27 visual pair features.',
        'The matcher is a scaled linear logistic model. The stored arrays do not need a pickle loader.',
        'These fits train boundaries and class matching. They do not train the legend parser.',
        f"The two deployed weight files total {fit['weights_bytes']:,} bytes. The limit is 10 MB.",
        f"Boundary CV and training took {resources['train']['seconds']/60:.2f} minutes. The limit is 60 minutes.",
        f"Preparation took {resources['prepare']['seconds']/60:.2f} minutes. Selection took {resources['select']['seconds']/60:.2f} minutes.",
        f"Peak measured RAM was {max(r['peak_rss_gb'] for r in resources.values()):.2f} GB. The limit is 4 GB.",
        'Torch and BLAS use at most three threads. OpenCV uses one thread. No GPU is used.',
        'The corpus uses memory maps. Training reads only the requested feature windows.', '',
        f"A speed check used seven new synthetic columns with 6,000 rows each. The slowest run took {max(r['seconds'] for r in smoke['records']):.3f} seconds.",
        f"The speed check peaked at {smoke['peak_rss_gb']:.2f} GB RAM. It did not use a benchmark log or choose settings.", '',
        '## Checks and files', '',
        'The test command checks source exclusion, source splits, seeded output, row geometry,',
        'the 51-window list, well folds, model size, and strategy use without truth access.',
        'It also checks that selection and deployed naming give the same rows.',
        'The fit step checks exported matcher probabilities against the training model.',
        'See test_results.txt for the test result.', '',
        'The strategy entry point is strategies/s21_synth_v2.py.',
        'The source records are in assets.json. The window records are in synthetic_index.json.',
        'Each window record gives its source, seed, split, and exact truth.',
        'The best settings are in selected.json. The full selection results are in selection.json.',
        'The deployed models are boundary_net.pt and matcher.npz. Their hashes are in fit.json.',
        'READY.json gives the final code, report, data record, and model hashes.',
        'The preview image is synthetic_preview.png.', '',
        'The shared legend parser can omit or misname a class. This model cannot restore a missing label.',
        'No gain on the final benchmark is claimed.', '',
        '## Run again', '',
        'Set LITHOLOGY_DATASET to the local dataset folder. Run these commands from bakeoff.',
        'Use the project Python environment. The code has no fixed host paths.', '',
        '```sh',
        'python -m ml.synth_v2.experiment prepare',
        'python -m ml.synth_v2.experiment train --steps 1500',
        'python -m ml.synth_v2.experiment select',
        'python -m pytest ml/synth_v2/test_synth_v2.py -q -o cache_dir=ml/synth_v2/.pytest_cache',
        'python -m ml.synth_v2.smoke',
        'python -m ml.synth_v2.report',
        '```', '',
        'Set PYTHONDONTWRITEBYTECODE=1 to keep imports from writing shared bytecode caches.',
        'No final benchmark command is part of this lane.', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()
