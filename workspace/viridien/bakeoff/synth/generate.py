"""Generate native-scale synthetic columns from approved real crops.

Run from the bake-off root: python -B -m synth.generate --n-logs 120
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from harness import depth, io, legend, legend_v2
from harness.score import normalize
from .statistics import ROOT, dump, fit_models, read_json, resolve_crop, split_sources
from .textures import donor_assets, quilt
from .nuisances import PROBABILITIES, apply_nuisances, draw_nuisances

HERE = Path(__file__).resolve().parent
TRUTH_FIELDS = ['top_depth', 'bottom_depth', 'lithology', 'top_y', 'bottom_y', 'cluster_id']


def under_synth(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError('All derived outputs must be under synth/')
    return path


def allocate(models, n_logs):
    keys = sorted(models)
    if n_logs < 2 * len(keys):
        raise ValueError(f'--n-logs needs at least {2 * len(keys)} for {len(keys)} legend types')
    weights = np.sqrt([max(1e-12, models[k]['filled_share']) for k in keys])
    exact = np.zeros(len(keys), float)
    active = np.ones(len(keys), bool)
    while active.any():
        free = n_logs - exact[~active].sum()
        proposed = free * weights[active] / weights[active].sum()
        small = proposed < 2
        if not small.any():
            exact[active] = proposed
            break
        fixed = np.flatnonzero(active)[small]
        exact[fixed] = 2
        active[fixed] = False
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - np.floor(exact)), kind='stable')[:n_logs - int(counts.sum())]:
        counts[i] += 1
    return dict(zip(keys, map(int, counts)))


def sample_height(model, rng, thin=False):
    if thin:
        return int(rng.integers(6, 31))
    return int(round(np.clip(rng.lognormal(model['mu'], model['sigma']), model['p1'], model['p99'])))


def build_beds(patterns, pools, model, height, rng, counts, quota, flags):
    available = [p for p in patterns if pools[str(p['pid'])]]
    if not available:
        raise ValueError('No usable crops for this legend type and pool')
    labels = model['labels']
    li = {v: i for i, v in enumerate(labels)}
    rows, y, previous = [], 0, None
    alternating = None
    alt_left = 0
    if flags['interbedded']['selected'] and len({q['label'] for q in available}) > 1:
        first = available[int(rng.integers(len(available)))]
        other = [q for q in available if q['label'] != first['label']]
        alternating = [first, other[int(rng.integers(len(other)))]]
        alt_left = int(rng.integers(4, 9))
    gap_done = False
    while y < height:
        remaining = height - y
        deficits = [p for p in available if counts[str(p['pid'])] < quota.get(str(p['pid']), 0)]
        candidates = deficits or available
        different = [p for p in candidates if p['label'] != previous]
        if different:
            candidates = different
        weights = []
        for p in candidates:
            i = li[p['label']]
            t = model['thickness'][p['label']]
            expected_h = np.clip(np.exp(t['mu'] + t['sigma'] ** 2 / 2), t['p1'], t['p99'])
            # Convert filled-row shares to bed probabilities, then retain the
            # learned first-order sequence preference. Coverage quotas are exact.
            probability = model['shares'][i] / max(6, expected_h)
            if previous in li and len(labels) > 1:
                probability *= model['transitions'][li[previous]][i]
            siblings = sum(q['label'] == p['label'] for q in available)
            rarity = 1 / np.sqrt(max(.005, float(p.get('share', .01))))
            variety = max(0., float(p.get('variety', (float(p.get('n_target', 10)) - 10) / 30)))
            weights.append(max(1e-12, probability * min(6, rarity) * (1 + 2 * variety) * np.sqrt(siblings) / siblings))
        weights = np.asarray(weights, float)
        weights /= weights.sum()
        p = candidates[int(rng.choice(len(candidates), p=weights))]
        thin = rng.random() < .08
        interbed = False
        if flags['interbedded']['selected'] and len({q['label'] for q in available}) > 1:
            if not alt_left and rng.random() < .07:
                other = [q for q in available if q['label'] != p['label']]
                alternating = [p, other[int(rng.integers(len(other)))]]
                alt_left = int(rng.integers(4, 9))
            if alt_left and not (remaining <= 6 * sum(max(0, quota.get(str(q['pid']), 0) - counts[str(q['pid'])])
                                                    for q in available) and deficits):
                p = alternating[alt_left % 2]
                alt_left -= 1
                thin, interbed = True, True
        h = sample_height(model['thickness'][p['label']], rng, thin=thin)
        # Leave room to fulfill quotas in this log; never create a 1..5 row tail.
        owed = sum(max(0, quota.get(str(q['pid']), 0) - counts[str(q['pid'])]) for q in available)
        if owed > 1:
            h = min(h, max(6, remaining - (owed - 1) * 6))
        h = min(h, remaining)
        if 0 < remaining - h < 6:
            h = remaining
        gap = flags['blank_gaps']['selected'] and not interbed and not deficits and (
            not gap_done or rng.random() < .06)
        if gap:
            h = min(h, int(rng.integers(6, 61)))
            if 0 < remaining - h < 6:
                h = remaining
            gap_done = True
        bed = dict(top_y=y, bottom_y=y + h, lithology='' if gap else p['label'],
                   cluster_id=-1 if gap else li[p['label']], pid=None if gap else str(p['pid']),
                   thin_component=bool(thin), interbed=interbed)
        rows.append(bed)
        if not gap:
            counts[str(p['pid'])] += 1
        previous = p['label'] if not gap else None
        y += h
    flags['interbedded'].update(applied=any(r['interbed'] for r in rows),
                               ranges=[[r['top_y'], r['bottom_y']] for r in rows if r['interbed']])
    return rows


def save_legend_cache(folder, name, assets, new_box):
    """Reuse donor OCR, shifted to the copied region; pixels remain unmodified."""
    serial = []
    dx = new_box[0] - assets['legend_box'][0]
    dy = new_box[1] - assets['legend_box'][1]
    for entry in assets['entries']:
        sw = name + f'.swatch_{entry.id:02d}.png'
        b = entry.swatch_box
        item = dict(id=entry.id, label=entry.label, swatch_box=[b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy],
                    mean_rgb=list(entry.mean_rgb), is_lithology=entry.is_lithology, swatch_file=sw)
        serial.append(item)
        for cache_name, version in [('legend', legend.VERSION), ('legend_v2', legend_v2.VERSION)]:
            cache = folder / 'parser_cache' / cache_name
            cache.mkdir(parents=True, exist_ok=True)
            Image.fromarray(entry.swatch).save(cache / sw)
    for cache_name, version in [('legend', legend.VERSION), ('legend_v2', legend_v2.VERSION)]:
        dump(folder / 'parser_cache' / cache_name / (name + '.json'),
             dict(version=version, entries=serial, donor=assets['name'], ocr=[]))
    dump(folder / 'legend_entries.json', serial)


def render_log(folder, name, beds, column, clean, assets, calibration, flags, mono):
    width, height = column.shape[1], column.shape[0]
    legend_rgb = assets['legend_rgb']
    lx, ly = 100 + width, 12
    boxes = dict(depth_column=[4, 0, 78, height],
                 interpreted_lithology_column=[84, 0, 84 + width, height],
                 lithology_legend=[lx, ly, lx + legend_rgb.shape[1], ly + legend_rgb.shape[0]])
    page_w = boxes['lithology_legend'][2] + 8
    page_h = max(height, boxes['lithology_legend'][3] + 8)
    page = Image.new('RGB', (page_w, page_h), 'white')
    page.paste(Image.fromarray(legend_rgb), (lx, ly))
    draw = ImageDraw.Draw(page)
    for y in range(20, height, 200):
        draw.text((6, y), f'{calibration.to_depth(y):.1f}', fill='black')
        draw.line((73, y, 80, y), fill='black')
    page.paste(Image.fromarray(clean), (84, 0))
    if mono:
        clean_page = page.convert('1', dither=Image.Dither.NONE)
    else:
        clean_page = page
    clean_page.save(folder / 'image_clean.png')
    page.paste(Image.fromarray(column), (84, 0))
    if mono:
        page = page.convert('1', dither=Image.Dither.NONE)
    filename = 'image.tif' if flags['lzw']['selected'] else 'image.png'
    if filename.endswith('.tif'):
        page.save(folder / filename, compression='tiff_lzw')
        flags['lzw'].update(applied=True, note='LZW is lossless; no visual artefact is claimed')
    else:
        page.save(folder / filename)
    Image.fromarray(legend_rgb).save(folder / 'legend.png')
    dump(folder / 'boxes.json', dict(bounding_boxes=boxes))
    save_legend_cache(folder, name, assets, boxes['lithology_legend'])
    import csv
    # Adjacent same-label texture placements are not observable contacts unless
    # a rule was actually drawn. Keep provenance per placement in meta.json,
    # but honor the challenge convention in scored truth.
    truth = []
    ruled = set(flags['boundary_lines'].get('rows', []))
    for bed in beds:
        if truth and truth[-1]['lithology'] == bed['lithology'] and bed['top_y'] not in ruled:
            truth[-1]['bottom_y'] = bed['bottom_y']
        else:
            truth.append(dict(bed))
    with (folder / 'truth.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=TRUTH_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for bed in truth:
            writer.writerow(dict(bed, top_depth=round(calibration.to_depth(bed['top_y']), 1),
                                 bottom_depth=round(calibration.to_depth(bed['bottom_y']), 1)))
    return filename, boxes


def generate(catalog=HERE / 'data/catalog.json', approved=HERE / 'data/approved.json',
             output=HERE / 'data/fake', n_logs=120, seed=0, pool='both',
             min_height=2000, max_height=8000, clean_fraction=.2, probabilities=None, min_beds=5,
             texture='mix'):
    output = under_synth(output)
    if texture not in ('procedural', 'quilt', 'mix'):
        raise ValueError('texture must be procedural, quilt or mix')
    from .procgen import render_pattern, finish_column
    from .texstats import load_stats, aggregate, texture_stats, boundary_summary, structure_summary
    measured = load_stats()['patterns'] if texture != 'quilt' else {}
    if min_height < 2000 or max_height > 8000 or min_height > max_height:
        raise ValueError('Column height range must be inside 2000..8000')
    if pool not in ('both', 'train', 'test') or not 0 <= clean_fraction <= 1:
        raise ValueError('Invalid pool or clean fraction')
    approved = Path(approved).resolve()
    raw = read_json(approved)
    catalog_data = read_json(catalog)
    if not isinstance(raw, dict) or not catalog_data:
        raise ValueError('catalog.json and approved.json are required; use --mock to build separate mock inputs')
    patterns = [dict(p, pid=str(p['pid']), legend_type=str(p['legend_type']))
                for p in catalog_data['patterns'] if raw.get(str(p['pid']))]
    if not patterns:
        raise ValueError('No approved patterns')
    raw = {p['pid']: raw[p['pid']] for p in patterns}
    for crops in raw.values():
        for crop in crops:
            if not resolve_crop(crop['file'], approved).is_file():
                raise ValueError('Approved crop file is missing')
    split = split_sources(raw, seed)
    models = fit_models(patterns, split)
    allocations = allocate(models['types'], n_logs)
    requested_pools = ('train', 'test') if pool == 'both' else (pool,)
    for legend_type in allocations:
        ps = [p for p in patterns if p['legend_type'] == legend_type]
        for requested in requested_pools:
            if not any(split['patterns'][p['pid']][requested] for p in ps):
                raise ValueError(f'Legend type {legend_type} has no {requested} source pool. '
                                 'Approve crops from more source logs or generate --pool train only.')
    if pool == 'test' and split['unavailable_test_patterns']:
        raise ValueError('Some patterns have no held-out source. Use --pool both for training-only patterns '
                         'or approve more source logs; see source-split limitations in synth/README.md')
    # Refuse stale mixtures of runs, but permit byte-identical seeded reruns.
    output.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(Path(catalog).read_bytes() + approved.read_bytes()).hexdigest()
    config = dict(seed=seed, n_logs=n_logs, pool=pool, min_height=min_height, max_height=max_height,
                  clean_fraction=clean_fraction, probabilities=probabilities, min_beds=min_beds, signature=signature,
                  texture=texture)
    if texture != 'quilt':
        config['texture_stats_sha256'] = hashlib.sha256((HERE/'data/texstats.json').read_bytes()).hexdigest()
    old = read_json(output / 'run.json')
    if any(output.glob('*/meta.json')) and (not old or old.get('config') != config):
        raise ValueError('Output already contains a different run; choose a new --output directory')
    dump(output / 'run.json', dict(config=config, state='generating'))
    # Default contract path; custom runs keep all split/model state alongside inputs.
    split_path = under_synth(approved.parent / 'split.json')
    dump(split_path, split)
    dump(output / 'split.json', split)
    dump(output / 'fitted_models.json', models)
    rng = np.random.default_rng(seed)
    manifest, total_counts, counters = [], Counter(), defaultdict(Counter)
    asset_cache = {}
    procedural_models = {}
    for type_index, (legend_type, count) in enumerate(allocations.items()):
        ps = [p for p in patterns if p['legend_type'] == legend_type]
        model = models['types'][legend_type]
        pools = ['train'] * max(1, min(count - 1, round(.7 * count)))
        pools += ['test'] * (count - len(pools))
        if pool != 'both':
            pools = [pool] * count
        pattern_targets = {value: {} for value in set(pools)}
        for p in ps:
            supported = [value for value in pattern_targets if split['patterns'][p['pid']][value]]
            if len(supported) == 2:
                train_target = min(min_beds - 1, max(1, round(min_beds * pools.count('train') / count)))
                pattern_targets['train'][p['pid']] = train_target
                pattern_targets['test'][p['pid']] = min_beds - train_target
            elif supported:
                pattern_targets[supported[0]][p['pid']] = min_beds
        clean_indices = set()
        clean_rng = np.random.default_rng(seed ^ int(hashlib.sha256(legend_type.encode()).hexdigest()[:8], 16))
        for selected_pool in sorted(set(pools)):
            indices = [i for i, value in enumerate(pools) if value == selected_pool]
            n_clean = round(len(indices) * clean_fraction)
            if clean_fraction > 0 and len(indices) >= 3:
                n_clean = max(1, n_clean)
            if n_clean:
                clean_indices.update(map(int, clean_rng.choice(indices, size=n_clean, replace=False)))
        for index, selected_pool in enumerate(pools):
            crops = {p['pid']: split['patterns'][p['pid']][selected_pool] for p in ps}
            active = [p for p in ps if crops[p['pid']]]
            if not active:
                raise ValueError(f'No {selected_pool} crops for legend type {legend_type}; see split.json')
            # Distribute the five-bed minimum across supported fake-log pools.
            # Requiring five separately in both would distort thin-bed rates.
            counter = counters[(legend_type, selected_pool)]
            remaining_logs = pools[index:].count(selected_pool)
            quota = {p['pid']: counter[p['pid']] + int(np.ceil(
                max(0, pattern_targets[selected_pool][p['pid']] - counter[p['pid']]) / remaining_logs)) for p in active}
            log_seed = int(rng.integers(0, 2**32))
            local = np.random.default_rng(log_seed)
            height = int(local.integers(min_height, max_height + 1))
            owed = sum(quota[p['pid']] - counter[p['pid']] for p in active)
            if owed * 6 > height:
                raise ValueError('Pattern coverage cannot fit; increase --n-logs or column heights')
            donors = sorted({c['log'] for p in active for c in crops[p['pid']]})
            donor = donors[int(local.integers(len(donors)))]
            if donor not in asset_cache:
                asset_cache[donor] = donor_assets(donor, output / '_cache')
            assets = asset_cache[donor]
            mono = '1-bit' in model['family'].lower() or 'd-582' in model['family'].lower() or assets['mode'] == '1'
            local.random()  # Keep the per-log random stream independent of clean allocation.
            is_clean = index in clean_indices
            flags = draw_nuisances(local, is_clean, mono, probabilities)
            # Procedural contacts use the observed hard cut/rule model. Explicit
            # user nuisance overrides still permit an intentional ablation.
            if texture != 'quilt':
                for key in ('gradational_boundaries', 'boundary_lines', 'depth_grid', 'column_borders'):
                    if key not in (probabilities or {}):
                        flags[key].update(selected=False, reason='Measured procedural contact/grid model')
            beds = build_beds(ps, crops, model, height, local, counter, quota, flags)
            rgb = np.full((height, assets['width'], 3), 255, np.uint8)
            for bed in beds:
                if bed['pid'] is None:
                    bed['source_crops'] = []
                    continue
                # Mock catalogues retain their distinct pids; match their label
                # and family to measured production textures when available.
                proc_pid = bed['pid'] if bed['pid'] in measured else next((
                    k for k, v in measured.items() if v['family'] == model['family'] and
                    normalize(v['label']) == normalize(bed['lithology'])), None)
                use_proc = texture == 'procedural' or (texture == 'mix' and proc_pid is not None and local.random() < .5)
                if use_proc:
                    if proc_pid is None:
                        raise ValueError('No measured procedural pattern: ' + bed['pid'])
                    model_key = (proc_pid, selected_pool)
                    if model_key not in procedural_models:
                        base = measured[proc_pid]
                        allowed = set(split[selected_pool+'_logs'])
                        matched = [(v, s) for v, s in zip(base['variants'], base['variant_sources']) if s['log'] in allowed]
                        if not matched:
                            from .textures import pixels
                            matched = [(texture_stats(pixels(str(resolve_crop(c['file'], approved)))), c)
                                       for c in crops[bed['pid']]]
                        fitted = dict(base, **aggregate([v for v, _ in matched]))
                        fitted['variants'] = [v for v, _ in matched]
                        fitted['variant_sources'] = [s for _, s in matched]
                        contacts = [v for v in base['boundaries'] if v['log'] in allowed]
                        fitted['boundary'] = boundary_summary(contacts)
                        fitted['structure'] = structure_summary([s for s in load_stats()['families'][base['family']].get('structure_sources', []) if s['log'] in allowed])
                        procedural_models[model_key] = fitted
                    fitted = procedural_models[model_key]
                    tile = render_pattern(proc_pid, bed['bottom_y']-bed['top_y'], assets['width'], int(local.integers(2**32)), stats=fitted)
                    sources = []
                    bed['procedural_pid'] = proc_pid
                    bed['parameter_source_logs'] = sorted({s['log'] for s in fitted['variant_sources']})
                else:
                    tile, sources = quilt(crops[bed['pid']], approved, bed['bottom_y'] - bed['top_y'], assets['width'], local)
                bed['texture'] = 'procedural' if use_proc else 'quilt'
                rgb[bed['top_y']:bed['bottom_y']] = tile
                bed['source_crops'] = sources
                total_counts[bed['pid']] += 1
            if texture != 'quilt':
                proc_beds = [dict(b, pid=b.get('procedural_pid', b['pid'])) for b in beds]
                rgb, drawn = finish_column(rgb, proc_beds, int(local.integers(2**32)),
                    grid='depth_grid' not in (probabilities or {}), borders='column_borders' not in (probabilities or {}),
                    models={pid: value for (pid, selected), value in procedural_models.items() if selected == selected_pool})
                flags['boundary_lines'].update(applied=bool(drawn), rows=[v['row'] for v in drawn],
                    model='measured', measured_rules=drawn)
            # The control uses the same bed plan and textures, including blank
            # intervals, and differs only in the applied image nuisances.
            clean = rgb.copy()
            if mono:
                clean = np.where(clean.mean(axis=2, keepdims=True) < 180, 0, 255).astype(np.uint8)
                clean = np.repeat(clean, 3, axis=2)
            dirty = apply_nuisances(rgb, beds, assets, flags, local)
            scale = float(local.choice(model['ft_per_px']))
            calib = depth.Calib(scale, float(local.uniform(1000, 26000)), 0, 0., 'synthetic-linear-known')
            name = f'fake_{type_index:02d}_{selected_pool}_{index:03d}'
            folder = output / name
            folder.mkdir(parents=True, exist_ok=True)
            image_file, boxes = render_log(folder, name, beds, dirty, clean, assets, calib, flags, mono)
            meta = dict(name=name, seed=log_seed, run_seed=seed, pool=selected_pool,
                family=model['family'], legend_type=legend_type, height=height, width=assets['width'],
                image_file=image_file, mode='1' if mono else 'RGB', donor_log=donor,
                calibration=asdict(calib), nuisances=flags, beds=beds, controlled_clean=is_clean,
                mock=bool(catalog_data.get('mock')), approved_manifest=str(approved),
                blank_truth='Empty-label intervals explicitly tile blank rows; harness.score ignores them',
                legend='Copied native donor legend with reused donor parser cache; no oracle pattern entries',
                clean_control='Same bed plan including interbeds/gaps; image nuisances absent',
                allocation='Smoothed row shares / expected height, transitions, rarity/variety weights, coverage quotas')
            meta['truth_merging'] = 'Adjacent equal labels merge unless a contact line is drawn; beds retain placement provenance'
            dump(folder / 'meta.json', meta)
            manifest.append(dict(name=name, family=model['family'], legend_type=legend_type,
                                 pool=selected_pool, height=height, seed=log_seed))
    short = {p['pid']: total_counts[p['pid']] for p in patterns if total_counts[p['pid']] < min_beds}
    if short:
        raise ValueError('Requested pool cannot cover all approved patterns at least five times: ' + str(short))
    dump(output / 'manifest.json', manifest)
    if procedural_models:
        dump(output/'procedural_models.json', {pool_name+':'+pid: value for (pid, pool_name), value in procedural_models.items()})
    dump(output / 'run.json', dict(config=config, state='complete', allocation=allocations, pattern_beds=dict(total_counts),
                                  n_logs=len(manifest), mock=bool(catalog_data.get('mock'))))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=HERE / 'data/catalog.json')
    parser.add_argument('--approved', type=Path, default=HERE / 'data/approved.json')
    parser.add_argument('--output', type=Path, default=HERE / 'data/fake')
    parser.add_argument('--n-logs', type=int, default=120)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--texture', choices=['procedural', 'quilt', 'mix'], default='mix')
    parser.add_argument('--pool', choices=['both', 'train', 'test'], default='both')
    parser.add_argument('--min-height', type=int, default=2000)
    parser.add_argument('--max-height', type=int, default=8000)
    parser.add_argument('--clean-fraction', type=float, default=.2)
    parser.add_argument('--probabilities', type=Path, help='JSON mapping nuisance names to probabilities')
    parser.add_argument('--mock', action='store_true', help='Use separate legend/gold/consensus mock inputs')
    args = parser.parse_args()
    if args.mock:
        from .mock import build_mock
        args.catalog, args.approved = build_mock()
        if args.output == HERE / 'data/fake':
            args.output = HERE / 'data/mock_fake'
    probabilities = read_json(args.probabilities) if args.probabilities else None
    if probabilities and set(probabilities) - set(PROBABILITIES):
        parser.error('Unknown nuisance names')
    result = generate(args.catalog, args.approved, args.output, args.n_logs, args.seed, args.pool,
                      args.min_height, args.max_height, args.clean_fraction, probabilities, texture=args.texture)
    print(f'Generated {len(result)} logs in {args.output}')


if __name__ == '__main__':
    main()
