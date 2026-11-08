"""Separate smoke-test inputs; NEVER substitutes for human approved.json."""
from pathlib import Path

from PIL import Image
from harness import io
from harness.score import normalize
from .statistics import ROOT, csv_rows, dump, read_json
from .textures import donor_assets

HERE = Path(__file__).resolve().parent


def build_mock(destination=HERE / 'data/mock'):
    destination = Path(destination).resolve()
    if not destination.is_relative_to(HERE):
        raise ValueError('Mock outputs must stay in synth/')
    destination.mkdir(parents=True, exist_ok=True)
    gold = sorted((ROOT / 'gt/gold').glob('*.csv'))[0]
    donors = [(gold.stem, csv_rows(gold), 'gold'),
              ('0000000009_d-582_0000020435-0001_log02_fe_5in_md',
               [r for r in csv_rows(ROOT / 'labelling/out/d582_log02/consensus.csv')
                if r.get('agreed') == 'true'], 'agreed_consensus_mock')]
    labels = ['Shale', 'Siltstone', 'Sand and Sandstone']
    approved = {f'mock_{i}': [] for i in range(len(labels))}
    patterns = []
    for i, label in enumerate(labels):
        patterns.append(dict(pid=f'mock_{i}', family='D-582 1-bit grid', legend_type='mock_d582',
            label=label, logs=[name for name, _, _ in donors], share=[.6, .25, .15][i],
            n_target=5, swatch_files=[]))
    for name, rows, kind in donors:
        assets = donor_assets(name, destination / '_cache')
        log = io.open_log(name)
        try:
            for i, label in enumerate(labels):
                p = patterns[i]
                entry = next(e for e in assets['entries'] if normalize(e.label) == normalize(label))
                swatch = destination / f'{name}_swatch_{i}.png'
                Image.fromarray(entry.swatch).save(swatch)
                p['swatch_files'].append(swatch.relative_to(ROOT).as_posix())
                chosen = [r for r in rows if normalize(r['lithology']) == normalize(label)
                          and int(float(r['bottom_y'])) - int(float(r['top_y'])) >= 12][:3]
                if not chosen:
                    raise ValueError('Mock needs genuine full-width crops from both source logs')
                for j, row in enumerate(chosen):
                    a, b = int(float(row['top_y'])), int(float(row['bottom_y']))
                    # Interior cuts avoid carrying every real boundary into a bed.
                    h = min(320, b - a - 6)
                    a = (a + b - h) // 2
                    rgb = io.read_rgb(log, 'interpreted_lithology_column', a, a + h)
                    path = destination / f'{name}_{i}_{j}.png'
                    Image.fromarray(rgb).save(path)
                    approved[p['pid']].append(dict(file=path.relative_to(ROOT).as_posix(), log=name,
                        top_y=a, bottom_y=a + h, col_w=rgb.shape[1], h=h, source_kind=kind))
        finally:
            log.close()
    catalog = destination / 'catalog.json'
    approvals = destination / 'approved.json'
    dump(catalog, dict(mock=True, patterns=patterns,
        warning='Smoke-test only: gold and agreed-consensus crops, donor legend swatches. Not human pattern approvals.'))
    dump(approvals, approved)
    return catalog, approvals


if __name__ == '__main__':
    paths = build_mock()
    print('\n'.join(str(p) for p in paths))
