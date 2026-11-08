"""Copy the read-only source evidence into portable, provenance-rich suites."""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

from PIL import Image

from .data import load_boxes, load_intervals, load_manifest, sha256


def _group(log: str) -> str:
    if log.startswith('0000000009_d-582_'):
        return 'challenge:d-582'
    match = re.match(r'(\d{14})', log)
    if match:
        return 'challenge:'+match.group(1)
    # Conservative page-family grouping: many MD/TVD and scale variants are the same well.
    if log.startswith('MC765_'):
        return 'challenge:MC765-P011'
    if log.startswith('Shell_08852_'):
        return 'challenge:Shell-08852-KE001'
    return 'challenge:'+re.sub(r'(?i)(?:_mud.*|_\d*(?:inch|in|md|tvd).*|_log\d+.*)$', '', log)


def _find_image(directory: Path, stem: str) -> Path:
    for suffix in ('.tif', '.tiff', '.png', '.jpg', '.jpeg'):
        file = directory/(stem+suffix)
        if file.is_file():
            return file
    raise FileNotFoundError(f'image not found for {stem} in {directory}')


def import_source(source_root, dataset_root=None, output_dir=None, *, challenge_dataset=None, out=None,
                  include_unlabelled=True) -> Path:
    """Import human(10), supplied organizer(1), dev(17), diagnostic verify(8).

    Returns all.json. Copies every referenced asset, leaves source bytes intact,
    and emits separate suite manifests for independently downloadable bundles.
    Existing images are reused only when their checksum agrees with the source.
    """
    source_root = Path(source_root).resolve()
    base = source_root/'viridien' if (source_root/'viridien').is_dir() else source_root
    dataset_root = Path(dataset_root or challenge_dataset).resolve()
    output = Path(output_dir or out).resolve()
    if output == source_root or source_root in output.parents or output == dataset_root or dataset_root in output.parents:
        raise ValueError('output must be outside both read-only source roots')
    output.mkdir(parents=True, exist_ok=True)
    snapshot, copies = [], {}

    def copy_asset(original: Path, relative: Path) -> str:
        if not original.is_file():
            raise FileNotFoundError(f'required source asset unavailable: {original}')
        key = str(original.resolve())
        if key in copies:
            return copies[key]
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(original)
        if not destination.exists() or sha256(destination) != digest:
            shutil.copy2(original, destination)
        portable = relative.as_posix()
        copies[key] = portable
        snapshot.append({'source': str(original), 'destination': portable, 'sha256': digest, 'bytes': original.stat().st_size})
        return portable

    def archive_metadata(folder: Path, suite: str):
        for name in ('README.md', 'manifest.csv', 'eval_set.json', 'HUMAN_TAG.json', 'SHARED_TAG.json'):
            if (folder/name).is_file():
                copy_asset(folder/name, Path('provenance')/suite/name)

    def make_record(log, window, suite, image_dir, annotation_dir, truth_dir, *, source, split, group,
                    provenance, tags=None, truth_window=None, metadata=None):
        image = _find_image(image_dir, log)
        annotation = annotation_dir/(log+'.json')
        with Image.open(image) as im:
            size = im.size
        load_boxes(annotation, size)
        image_rel = copy_asset(image, Path('assets')/suite/'images'/image.name)
        annotation_rel = copy_asset(annotation, Path('assets')/suite/'bbox_annotation'/annotation.name)
        truth = truth_dir/(log+'.csv') if truth_dir is not None else None
        record = {'id': f'{suite}:{log}__y{window[0]}_{window[1]}', 'image': image_rel,
                  'annotation': annotation_rel, 'truth': None, 'source': source, 'split': split,
                  'group': group, 'window': list(window), 'tags': tags or [], 'fresh_holdout': False,
                  'label_provenance': provenance, 'log': log}
        if truth:
            record['truth'] = copy_asset(truth, Path('assets')/suite/'truth'/truth.name)
            record['truth_window'] = truth_window or list(window)
        if metadata:
            record.update(metadata)
        return record

    suites = {}
    human = base/'human_labels'
    archive_metadata(human, 'human')
    with (human/'manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        human_meta = {r['id']: r for r in csv.DictReader(handle)}
    suites['human'] = []
    labelled_logs = set()
    for item in json.loads((human/'eval_set.json').read_text(encoding='utf-8')):
        log = item['log']; labelled_logs.add(log)
        coverage = [int(v) for v in human_meta[log]['labelled_y'].split('-')]
        suites['human'].append(make_record(log, (item['y0'], item['y1']), 'human', dataset_root/'images',
            human/'bbox_annotation', human/'truth', source='human', split='dev', group=_group(log),
            provenance={'kind': 'human', 'labeller': human_meta[log]['labeller'],
                        'snapshot': human_meta[log]['snapshot'], 'review': 'two AI reviews plus agreed adjudicated fixes; AI review is not ground truth',
                        'source_metadata': 'provenance/human/manifest.csv'},
            tags=['visible_development', 'partial_truth' if 'shenzi' in log else 'human_labelled'],
            truth_window=coverage, metadata={'depth_unit': 'ft', 'depth_datum': 'TVD' if '1inch tvd' in log else 'unknown',
                                           'image_depth_unit': 'ft'}))
    suites['organizer'] = []
    gold = base/'bakeoff'/'gt'/'gold'
    for item in json.loads((base/'bakeoff'/'eval_set.json').read_text(encoding='utf-8')):
        log = item['log']
        if item.get('tier') != 'gold':
            continue
        labelled_logs.add(log)
        suites['organizer'].append(make_record(log, (item['y0'], item['y1']), 'organizer', dataset_root/'images',
            dataset_root/'bbox_annotation', gold, source='organizer', split='dev', group=_group(log),
            provenance={'kind': 'organizer_supplied_example', 'source': 'bakeoff/gt/gold',
                        'known_use': 'supplied example and repeatedly evaluated development page'},
            tags=['visible_development','organizer_example'], metadata={'depth_unit':'ft','depth_datum':'MD','image_depth_unit':'ft'}))
    for folder_name, suite, split, tags in (
        ('external_shared', 'external-dev', 'dev', ['visible_development']),
        ('external_shared_verify', 'external-diagnostic', 'verify', ['previously_tuned','diagnostic_only'])):
        folder = base/folder_name
        archive_metadata(folder, suite)
        with (folder/'manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
            metadata = {row['id']: row for row in csv.DictReader(handle)}
        suites[suite] = []
        for item in json.loads((folder/'eval_set.json').read_text(encoding='utf-8')):
            log = item['log']; info = metadata[log]
            metric = info['source'] == 'nl'
            suites[suite].append(make_record(log, (item['y0'], item['y1']), suite, folder/'images',
                folder/'bbox_annotation', folder/'truth', source='external_labelled', split=split,
                group='external:'+info['source']+':'+info['well'],
                provenance={'kind':'pdf_vector_extraction', 'source_agency':info['source'], 'source_url':info['source_url'],
                            'notes':info['notes'], 'source_metadata':f'provenance/{suite}/manifest.csv',
                            'prior_use':'s13-s18 grouped CV and tuning' if split == 'verify' else 'shared development'},
                tags=tags+(['unit_metadata_conflict'] if metric else []),
                metadata={'depth_unit':'ft', 'depth_datum':'unknown', 'image_depth_unit':'m' if metric else 'ft',
                          'depth_unit_note':'Per-page manifest explicitly says vector truth converted m/0.3048 to ft; README says page units. Truth retained unchanged.' if metric else 'Per-page vector manifest',
                          'source_url':info['source_url']}))
    suites['challenge-unlabelled'] = []
    if include_unlabelled:
        for annotation in sorted((dataset_root/'bbox_annotation').glob('*.json')):
            log = annotation.stem
            if log in labelled_logs or '(1)' in log:
                continue
            image = _find_image(dataset_root/'images', log)
            with Image.open(image) as im:
                box = load_boxes(annotation, im.size)['interpreted_lithology_column']
            suites['challenge-unlabelled'].append(make_record(log, box[1::2], 'challenge-unlabelled',
                dataset_root/'images', dataset_root/'bbox_annotation', None, source='challenge_unlabelled',
                split='dev', group=_group(log), provenance=None, tags=['unlabelled','visible_development'],
                metadata={'depth_unit':'unknown','depth_datum':'unknown'}))
    all_records = []
    for name, records in suites.items():
        manifest = {'schema_version':1, 'name':name, 'records':records,
                    'evidence_note':'Visible development evidence. Shared verification was previously tuned and is diagnostic only; no fresh holdout claim.'}
        (output/(name+'.json')).write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        all_records.extend(records)
    (output/'source-snapshot.json').write_text(json.dumps({'schema_version':1, 'source_root':str(source_root),
        'challenge_dataset':str(dataset_root), 'files':snapshot}, indent=2)+'\n', encoding='utf-8')
    combined = output/'all.json'
    combined.write_text(json.dumps({'schema_version':1, 'name':'all-imported-evidence', 'records':all_records,
        'source_snapshot':'source-snapshot.json', 'fresh_holdout':False}, indent=2)+'\n', encoding='utf-8')
    load_manifest(combined)
    return combined
