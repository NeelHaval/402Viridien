"""Versioned synthetic datasets from the lab's saved models (CPU only).

    python -X utf8 -m genlab.dataset build
    python -X utf8 -m genlab.dataset eval v1
    python -X utf8 -m genlab.dataset loop
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

# Limit optional BLAS users before numpy/sklearn imports.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from PIL import Image, ImageDraw

from . import candidates, learn, paths, realism, render, score
from .build import _write, labstore
from feedback import canonical
from normalized import group_key, letterbox, sample_id
from review_queue import version as annotation_version
from .reading import ReadOnlyStore

ROOT = paths.OUT / 'datasets'
SCHEMA = 4
MIN_N = 3


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pixel_hash(rgb):
    h, w = rgb.shape[:2]
    return hashlib.sha256(f'{w}x{h}:'.encode() + rgb.tobytes()).hexdigest()


def eligible_record(row, doc, segment):
    """Fail closed on quarantine, stale annotations, labels, or policy."""
    legends = {e['id']: e for e in doc['legend']}
    ids = segment.get('legend_ids', [])
    stamp = hashlib.sha256(json.dumps([legends[i] for i in ids], sort_keys=True).encode()).hexdigest()[:16]
    return bool(row and row.get('accepted') and row.get('usable_for_generation')
        and row.get('split') == 'train' and row.get('quality') == 'accepted'
        and not row.get('quality_issues') and not row.get('source_flags')
        and row.get('annotation_version') == annotation_version(segment)
        and row.get('legend_version') == stamp and row.get('box_xyxy') == segment['box']
        and [t['legend_id'] for t in row.get('targets', [])] == ids)


def model_lineage(card, manifest, cards, seen=None, manifests=None):
    """Union every declaration, including parent manifests, before pixel access."""
    key = card['key']
    seen = set() if seen is None else seen
    if key in seen:
        raise ValueError('Cyclic model ancestry: ' + key)
    seen = seen | {key}
    ancestors, refs = {card['doc_id']}, []
    manifests = manifests or {}
    donors = [v['donor'] for v in (card, manifest) if v.get('donor')]
    identities = []
    for donor in donors:
        if not isinstance(donor, dict) or not donor.get('doc_id'):
            raise ValueError('Unknown donor provenance')
        identities.append((donor['doc_id'], donor.get('legend_id')))
    if len(set(identities)) > 1:
        raise ValueError('Conflicting donor declarations: ' + key)

    def visit(value, stack=frozenset()):
        if not isinstance(value, dict) or id(value) in stack:
            raise ValueError('Malformed or cyclic source lineage')
        stack = stack | {id(value)}
        for field in ('doc_id','document_id','real_doc_id','source_doc_id','source_doc','doc'):
            if value.get(field):
                ancestors.add(value[field])
        declared = value.get('ancestor_docs', [])
        if not isinstance(declared, list) or any(not isinstance(d,str) or not d for d in declared):
            raise ValueError('Malformed explicit ancestors')
        ancestors.update(declared)
        refs.extend(value.get('normalized_sources', []))
        if value.get('sample_id'):
            refs.append(value)
        for field in ('lineage','sources'):
            entries = value.get(field, [])
            if not isinstance(entries, list):
                raise ValueError('Malformed source list')
            for entry in entries:
                visit(entry, stack)
        if value.get('recovery'):
            visit(value['recovery'], stack)
        if value.get('render_model'):
            payload = value['render_model']
            if not isinstance(payload,dict) or not isinstance(payload.get('provenance',{}),dict):
                raise ValueError('Malformed embedded rendering provenance')
            visit(payload.get('provenance',{}), stack)
        donor = value.get('donor')
        if donor:
            if not isinstance(donor,dict) or not donor.get('doc_id'):
                raise ValueError('Unknown donor provenance')
            visit(donor, stack)
            if donor.get('legend_id') is not None:
                parent_key = class_id(donor['doc_id'],donor['legend_id'])
                parent = cards.get(parent_key)
                if parent is None:
                    raise ValueError('Unresolved donor model: ' + parent_key)
                docs, sources = model_lineage(parent,manifests.get(parent_key,{}),cards,seen,manifests)
                ancestors.update(docs)
                refs.extend(sources)
    visit(card)
    visit(manifest)
    return ancestors, list({digest(r):r for r in refs}.values())


def model_ancestors(card, manifest, cards, seen=None, manifests=None):
    return model_lineage(card,manifest,cards,seen,manifests)[0]


def rendering_payload(card, manifest=None):
    """A tile without explicit placement metadata is not a repeating model."""
    manifest = manifest or {}
    serialized = manifest.get('render_model') or card.get('render_model')
    if serialized:
        if card.get('render_model') and manifest.get('render_model') and digest(card['render_model'])!=digest(manifest['render_model']):
            raise ValueError('Conflicting serialized rendering models')
        if serialized.get('schema')!='genlab-render-2' or serialized.get('signature')!=card.get('signature'):
            raise ValueError('Unbound rendering model')
        payload = dict(serialized['placement'],kind=serialized['kind'])
    else:
        payload = manifest.get('rendering') or card.get('rendering') or card
    if payload.get('kind') in ('quilt','texture','jitter'):
        return dict(kind=payload['kind'])
    required = ('kind','phase','anchored_x','anchored_y','x_repeat','x_origin')
    if any(k not in payload for k in required):
        raise ValueError('Missing serialized rendering placement')
    return {k:payload[k] for k in required}


def snapshot_render_model(card, manifest, folder, source_root):
    """Copy the full rendering bank; display previews are never model assets."""
    payload = manifest.get('render_model') or card.get('render_model')
    if not payload:
        return None,None
    rendering_payload(card,manifest)
    model = learn.load_model(payload,root=source_root,signature=card['signature'])
    rel = 'models/'+payload['asset']['sha256']+'.npz'
    source = Path(source_root)/payload['asset']['path']
    target = Path(folder)/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    copied = dict(payload,asset=dict(payload['asset'],path=rel))
    if copied.get('binding_sha256'):
        copied['binding_sha256'] = learn.model_binding(copied)
    return model,copied


def class_id(doc, lid):
    return f'{doc}:{lid}'


def well_split(doc, policy):
    group = policy['document_groups'].get(doc['id'], group_key(doc['name']))
    h = hashlib.sha256((policy['seed'] + ':' + group).encode()).hexdigest()
    if int(h[:8], 16) % 5 == 0:
        return group, 'test'
    h = hashlib.sha256(('genlab-validation-v1:' + group).encode()).hexdigest()
    return group, ('validation' if int(h[:8], 16) % 5 == 0 else 'train')


def assert_truth(row):
    if (row.get('kind') != 'real' or row.get('evidence') != 'human-accepted'
            or row.get('status') != 'approved' or row.get('split') != 'validation'):
        raise ValueError('Evaluation requires human-accepted validation boxes; machine evidence is not truth')


def save_rgb(root, rel, rgb, normalize=True):
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    image.save(target)
    out = dict(file=rel, pixel_sha256=hashlib.sha256(
        f'{image.width}x{image.height}:'.encode() + image.tobytes()).hexdigest())
    if normalize:
        fixed, valid, transform = letterbox(image)
        out.update(normalized_png=str(Path(rel).with_name(target.stem + '-224.png')).replace('\\', '/'),
                   valid_mask_png=str(Path(rel).with_name(target.stem + '-valid.png')).replace('\\', '/'),
                   normalization=transform)
        fixed.save(root / out['normalized_png'])
        valid.save(root / out['valid_mask_png'])
    return out


def family_name(doc, prefill):
    rows = prefill.get(doc['name'], [])
    if rows and rows[0].get('family'):
        return rows[0]['family']
    name = doc['name'].lower()
    if 'mc765' in name:
        return 'MC765 colour grid'
    if '608' in name[:14]:
        return 'Gulf 3-panel'
    if 'rio' in name:
        return "Texas 'Rock Types' box"
    return 'One-off layout'


def dataset_augment(clean, measurement, overlay, rng):
    """Training-only style; preserve entire connected nuisance components."""
    fitted = dict(measurement) if measurement else None
    if fitted:
        fitted['blur'] = min(.7, fitted['blur'])
        fitted['noise'] = min(6., fitted['noise'])
        fitted['drift'] = np.clip(fitted['drift'], -12, 12).tolist()
    if overlay is not None:
        mask, colours = overlay
        mask = realism.component_budget(mask, colours, clean, .05)
        overlay = mask, colours
    rgb, params = realism.augment(clean, fitted, overlay, rng)
    params['policy'] = 'whole dark components placed once <=5%; train-only family style; native scale'
    params['calibration'] = 'measured' if measurement else 'unmeasured'
    return rgb, params


def legend_model(raw):
    """Discard white outer scanner margins on a coloured swatch, source only.

    Keep white-background patterns and all interior white rows. In particular,
    do not invent missing strokes or infer a repeat from a held-out column.
    """
    from .swatch import remove_frame
    rgb, frame = remove_frame(raw)
    top = bottom = 0
    bg = learn.background(rgb)
    if float(bg.min()) < 225 and np.linalg.norm(bg.astype(float)-255) > 60:
        white = (rgb.min(2)>=240).mean(1)
        limit = min(4,max(0,len(rgb)//4))
        while top < limit and white[top]>=.95:
            top += 1
        while bottom < limit and white[len(rgb)-1-bottom]>=.95:
            bottom += 1
    rgb = rgb[top:len(rgb)-bottom if bottom else len(rgb)]
    cleaned, model, detail = candidates.swatch(rgb if top or bottom else raw)
    detail['source_frame_trim'] = frame
    detail['white_edge_trim'] = [top,bottom]
    return cleaned, model, detail


def weighted_column(models, rails, thickness, transitions, prior, height, rng):
    """Use observed class odds whenever no observed outgoing transition exists."""
    ids = list(models)
    rgb = np.empty((height,rails['x1']-rails['x0'],3),np.uint8)
    truth, y, previous = [], 0, None
    while y < height:
        odds = {k:v for k,v in transitions.get(previous,{}).items() if k in models and k!=previous}
        if not odds:
            # Row prevalence divided by mean bed height gives layer odds.
            # Excluding the previous class would force two-class columns to
            # alternate 50/50 even when observed prevalence is 90/10.
            odds = {k:prior.get(k,0)/max(1,float(np.mean(thickness[k]))) for k in ids}
        if not sum(odds.values()):
            raise ValueError('No observed class prior; refusing uniform geology')
        keys = list(odds)
        weights = np.array(list(odds.values()),float)
        lid = keys[int(rng.choice(len(keys),p=weights/weights.sum()))]
        distribution = thickness[lid]
        h = min(height-y,max(1,int(distribution[int(rng.integers(len(distribution)))])))
        rgb[y:y+h] = render.layer(local_model(models[lid]),rails['x0'],rails['x1'],y,h,rng)
        if truth and truth[-1]['legend_id']==lid:
            truth[-1]['bottom'] = y+h
        else:
            truth.append(dict(top=y,bottom=y+h,legend_id=lid))
        previous, y = lid, y+h
    return render.frame(rgb,rails),truth


def local_model(model):
    """Dataset images use local x=0; archived model placement stays in page space."""
    if model.get('kind')=='tile' and not model.get('x_repeat',True):
        return dict(model,x_origin=0)
    if model.get('kind')=='quilt':
        origin = model.get('x_origin',0)
        rebased = dict(model,x_origin=0,
            strip_origins=[x-origin for x in model.get('strip_origins',[origin]*len(model['strips']))])
        if model.get('native_extent'):
            rebased['native_extent'] = [x-origin for x in model['native_extent']]
        return rebased
    return model


class DatasetBuilder:
    def __init__(self, root=ROOT, baseline=None):
        from .dataset_loop import snapshot as watch_snapshot
        self.initial_fingerprint = watch_snapshot()
        self.root = Path(root)
        from .dataset_eval import revoke_unsafe_versions
        revoke_unsafe_versions(self.root)
        self.store = ReadOnlyStore(labstore.Store())
        self.policy = read(paths.DATA / 'normalized/split-policy.json')
        if not self.policy:
            raise RuntimeError('Missing labeller split policy')
        self.images = labstore.Images()
        self.crops = learn.Crops(self.images)
        self.lock = self.images.lock
        self.prefill = score.load_prefill()
        self.docs = {e['id']: self.store.get(e['id']) for e in self.store.catalog}
        self.doc_info = {did: dict(zip(('well', 'split'), well_split(d, self.policy)),
                                  family=family_name(d, self.prefill)) for did, d in self.docs.items()}
        self.baseline = Path(baseline) if baseline else None
        if self.baseline:
            # Controlled rebuild: freeze wells, annotations and training models
            # while other lab jobs continue changing the live source state.
            self.doc_info = read(self.baseline/'wells.json')
            self.docs = {did:self.docs[did] for did in self.doc_info}
            # Current revocations and test policy always override an archive.
            for did, info in self.doc_info.items():
                if well_split(self.docs[did], self.policy)[1] == 'test':
                    info['split'] = 'test'
        self.load_eligibility()
        self.code_dependencies = {}
        for module in (Path(__file__),Path(realism.__file__),Path(render.__file__),Path(candidates.__file__)):
            self.code_dependencies[module.name] = hashlib.sha256(module.read_bytes()).hexdigest()

    def load_eligibility(self):
        index = read(paths.DATA/'normalized/current.json', {})
        if not index.get('records'):
            raise RuntimeError('Missing normalized eligibility index; refusing unverified sources')
        self.normalized_snapshot = index['snapshot']
        self.normalized = {r['sample_id']: r for r in index['records']}
        self.test_hashes = {r['pixel_sha256'] for r in index['records'] if r['split']=='test'}
        self.eligible = {}
        for did, doc in self.docs.items():
            if self.doc_info[did]['split'] == 'test':
                continue
            for seg in doc['segments']:
                if (seg.get('status') != 'approved' or seg.get('flags')
                        or len(seg.get('legend_ids', [])) != 1):
                    continue
                row = self.normalized.get(sample_id(doc, seg))
                if eligible_record(row, doc, seg) and row['pixel_sha256'] not in self.test_hashes:
                    self.eligible[(did, seg['id'])] = row

    def crop(self, did, box):
        # Never request an entire TIFF page. The reader keeps one page open.
        box = [int(x) for x in box]
        box[3] = min(box[3], box[1] + 480)
        if digest(read(paths.DATA/'normalized/split-policy.json',{}))!=digest(self.policy):
            raise ValueError('Current split policy changed before crop access')
        if self.doc_info.get(did,{}).get('split')=='test':
            raise ValueError('Held-out document crop refused')
        rgb = self.normalized_window(did,box)
        if rgb is not None:
            return rgb
        with self.lock:
            rgb = self.crops.get(self.docs[did], box).copy()
        if pixel_hash(rgb) in self.test_hashes:
            raise ValueError('Quarantined test-identical pixels: ' + did)
        return rgb

    def normalized_window(self, did, box):
        """Read a bounded prefix from an eligible native PNG, never from TIFF."""
        candidates = [r for (doc,_),r in self.eligible.items() if doc==did and r.get('native_png')
            and r['box_xyxy'][0]<=box[0]<box[2]<=r['box_xyxy'][2]
            and r['box_xyxy'][1]<=box[1]<box[3]<=r['box_xyxy'][3]]
        if not candidates:
            return None
        path = paths.DATA/'normalized/current.json'
        stamp = (path.stat().st_mtime_ns,path.stat().st_size)
        if getattr(self,'_native_index_stamp',None)!=stamp:
            records = read(path,{}).get('records',[])
            if stamp!=(path.stat().st_mtime_ns,path.stat().st_size):
                raise ValueError('Normalized eligibility changed during crop access')
            self._native_index = {r['sample_id']:r for r in records}
            self._native_forbidden = {r['pixel_sha256'] for r in records if r['split']=='test'
                or not r.get('usable_for_generation') or r.get('quality_issues')}
            self._native_index_stamp = stamp
        fields = ('document_id','annotation_id','annotation_version','legend_version','box_xyxy','pixel_sha256')
        for source in sorted(candidates,key=lambda r:(r['box_xyxy'][2]-r['box_xyxy'][0])*(r['box_xyxy'][3]-r['box_xyxy'][1])):
            current = self._native_index.get(source['sample_id'])
            if (not current or not current.get('usable_for_generation') or current.get('quality_issues')
                    or current.get('split')=='test' or any(current.get(k)!=source.get(k) for k in fields)):
                raise ValueError('Revoked or changed native crop source')
            root = path.parent.resolve()
            native = (root/current['native_png']).resolve()
            if not native.is_relative_to(root) or not native.is_file():
                raise ValueError('Missing or unsafe native crop path')
            x0,y0,x1,y1 = current['box_xyxy']
            if (x1-x0)*(y1-y0)*3>64*1024*1024:
                continue
            with Image.open(native) as image:
                if image.format!='PNG' or image.size!=(x1-x0,y1-y0):
                    raise ValueError('Native crop format or geometry mismatch')
                rgb = np.asarray(image.convert('RGB')).copy()
            ph = pixel_hash(rgb)
            if ph!=current['pixel_sha256'] or ph in self._native_forbidden:
                raise ValueError('Native crop pixels changed or quarantined')
            window = rgb[box[1]-y0:box[3]-y0,box[0]-x0:box[2]-x0].copy()
            if pixel_hash(window) in self._native_forbidden:
                raise ValueError('Quarantined native crop window')
            return window
        return None

    def legend_pattern(self, did, entry, folder, original=None):
        """Held well patterns have exactly one pixel ancestor: their legend box."""
        info, doc = self.doc_info[did], self.docs[did]
        raw = self.crop(did, entry['box'])
        _, model, detail = legend_model(raw)
        withheld = candidates.not_lithology(entry['label_printed'], raw)
        recognition_eligible = not bool(withheld)
        if not withheld and detail.get('seam_risk', True):
            withheld = 'Unsafe legend seam or text/mixed swatch; no cyclic rendering'
        cid = class_id(did, entry['id'])
        rel = 'models/' + digest(cid)[:20] + '.png'
        model_asset = save_rgb(folder, rel, model['tile'], normalize=False)
        raw_file = 'models/' + digest(cid)[:20] + '-legend.png'
        save_rgb(folder, raw_file, raw, normalize=False)
        provenance = dict(kind='legend-swatch-only', doc=did, box=list(entry['box']),
            raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), real_crop_inputs=[],
            raw_file=raw_file, extractor='native legend extraction; white edge margins removed v2',
            white_edge_trim=detail['white_edge_trim'], seam_risk=detail.get('seam_risk'))
        provenance.update(raw_pixel_sha256=pixel_hash(raw),seam_x=detail.get('seam_x'),seam_y=detail.get('seam_y'))
        meta = dict(class_id=cid, legend_id=entry['id'], name=canonical(entry['label_printed']),
            label=entry['label_printed'], doc=did, document=doc['name'], family=info['family'],
            well=info['well'], split=info['split'], source='swatch', method='legend swatch',
            tile_source='legend swatch', donor=None, quilt_file=None, model_file=rel,
            model_pixel_sha256=model_asset['pixel_sha256'],
            width=doc['boxes']['interpreted_lithology_column'][2]-doc['boxes']['interpreted_lithology_column'][0],
            signature=digest([provenance, SCHEMA]), has_real_example=False, tile_provenance=provenance,
            renderable=not bool(withheld), withheld_reason=withheld, ancestor_docs=[did],
            synthesis_eligible=not bool(withheld), recognition_eligible=recognition_eligible,
            recognition_support=('generated-legend' if not withheld else 'finite-legend' if recognition_eligible else 'excluded'),
            rendering=dict(kind='tile',phase=list(model.get('phase',[0,0])),
                anchored_x=bool(model.get('anchored_x',False)),anchored_y=bool(model.get('anchored_y',False)),
                x_repeat=bool(model.get('x_repeat',True)),x_origin=int(model.get('x_origin',0))),
            normalized_snapshot=self.normalized_snapshot)
        return dict(meta=meta, card=(original or {}).get('card', {}), manifest={},
                    tile=model['tile'], model=model)

    def snapshot(self, folder):
        """Copy active pixels immediately; retry if the lab prunes during a rebuild."""
        if self.baseline:
            snap = read(self.baseline/'lab-snapshot.json')
            cards = {c['key']:c for c in snap['cards']}
            manifests = {m['pattern_id']:m for m in snap['patterns']}
            self.model_manifests = manifests
            result = []
            for meta in read(self.baseline/'stats.json')['classes']:
                if meta['split'] != 'train':
                    continue
                meta = dict(meta)
                card = cards.get(meta['class_id'], {})
                manifest = manifests.get(meta['class_id'], {})
                if meta.get('render_model'):
                    card = dict(card,render_model=meta['render_model'])
                    manifest = dict(manifest,render_model=meta['render_model'])
                if not self.safe_model(card, manifest, cards):
                    entry = next((e for e in self.docs[meta['doc']]['legend'] if e['id']==meta['legend_id']), None)
                    if entry:
                        result.append(self.legend_pattern(meta['doc'], entry, folder))
                    continue
                with Image.open(self.baseline/meta['model_file']) as image:
                    rgb = np.asarray(image.convert('RGB')).copy()
                model_asset = save_rgb(folder,meta['model_file'],rgb,False)
                meta['model_pixel_sha256'] = model_asset['pixel_sha256']
                model = dict(rendering_payload(card,manifest),tile=rgb)
                full_model,payload = snapshot_render_model(card,manifest,folder,self.baseline)
                if full_model is not None:
                    model = full_model
                    meta['render_model'] = payload
                elif meta.get('quilt_file'):
                    shutil.copyfile(self.baseline/meta['quilt_file'],folder/meta['quilt_file'])
                    model = candidates.load_data(folder/meta['quilt_file'])
                meta.update(ancestor_docs=sorted(model_ancestors(card,manifest,cards,manifests=manifests)),
                    normalized_snapshot=self.normalized_snapshot, renderable=True,
                    synthesis_eligible=True,recognition_eligible=True,recognition_support='generated-model',
                    rendering=rendering_payload(card,manifest),
                    normalized_sources=self.resolved_sources(card,manifest,cards))
                result.append(dict(meta=meta,card=cards[meta['class_id']],manifest=manifests[meta['class_id']],tile=rgb,model=model))
            for did,info in sorted(self.doc_info.items()):
                if info['split']=='validation':
                    for entry in self.docs[did]['legend']:
                        result.append(self.legend_pattern(did,entry,folder))
            _write(folder/'lab-snapshot.json',dict(snap,controlled_baseline=self.baseline.name))
            return result, dict(built_at=read(self.baseline/'stats.json')['created_at'])
        for attempt in range(5):
            state = read(paths.STATE, {})
            manifest = read(paths.GEN / 'manifest.json', {})
            rows = {r['pattern_id']: r for r in manifest.get('patterns', [])}
            self.model_manifests = rows
            cards = sorted(state.get('cards', []), key=lambda c: c['key'])
            if not cards or any(c['key'] not in rows for c in cards):
                time.sleep(2)
                continue
            result = []
            try:
                # Borrowing connects wells. If a donor is held out, every class
                # in the dependent well is held out too, including other pages.
                while True:
                    promote = set()
                    for c in cards:
                        donor = (rows[c['key']].get('donor') or {}).get('doc_id')
                        if donor and self.doc_info[donor]['split'] == 'validation':
                            promote.add(self.doc_info[c['doc_id']]['well'])
                    changed = False
                    for info in self.doc_info.values():
                        if info['well'] in promote and info['split'] == 'train':
                            info['base_split'] = 'train'
                            info['split'] = 'validation'
                            info['reason'] = 'held-out borrowing donor closure'
                            changed = True
                    if not changed:
                        break
                for c in cards:
                    did = c['doc_id']
                    info = self.doc_info[did]
                    if info['split'] == 'test':
                        raise ValueError('Lab published a test well: ' + did)
                    if info['split'] == 'validation':
                        # Do not even open the lab tile/quilt: it may have used
                        # accepted validation crops, including the evaluation box.
                        continue
                    row = rows[c['key']]
                    if not self.safe_model(c, row, {a['key']:a for a in cards}):
                        entry = next((e for e in self.docs[did]['legend'] if e['id']==c['legend_id']), None)
                        if entry:
                            result.append(self.legend_pattern(did,entry,folder))
                        continue
                    donor = (row.get('donor') or {}).get('doc_id')
                    if donor and self.doc_info[donor]['split'] == 'test':
                        raise ValueError('Test well donor: ' + donor)
                    # Validation-derived donors cannot enter training either.
                    split = info['split']
                    if donor and self.doc_info[donor]['split'] == 'validation':
                        split = 'validation'
                    rel = 'models/' + digest(c['key'])[:20] + '.png'
                    src = c.get('tile') or c['samples'][0]
                    with Image.open(paths.OUT / src) as im:
                        rgb = np.asarray(im.convert('RGB')).copy()
                    model_asset = save_rgb(folder, rel, rgb, normalize=False)
                    model = dict(rendering_payload(c,row), tile=rgb)
                    quilt_file = None
                    full_model,payload = snapshot_render_model(c,row,folder,paths.OUT)
                    if full_model is not None:
                        model = full_model
                    elif c.get('kind') == 'quilt':
                        report = read(paths.CANDIDATES / (did + '.json'), {})
                        entry = next((e for e in report.get('entries', []) if e['legend_id'] == c['legend_id']), {})
                        if not entry.get('data_file'):
                            raise RuntimeError('Missing saved quilt model: ' + c['key'])
                        quilt_file = 'models/' + digest(c['key'])[:20] + '.npz'
                        shutil.copyfile(paths.CANDIDATES / entry['data_file'], folder / quilt_file)
                        model = candidates.load_data(folder / quilt_file)
                    meta = dict(class_id=class_id(did, c['legend_id']), legend_id=c['legend_id'],
                                name=canonical(c['label']), label=c['label'], doc=did,
                                document=self.docs[did]['name'], family=info['family'], well=info['well'],
                                split=split, source=c['source'], method=c.get('method') or c.get('tile_source', 'unknown'),
                                reviewer_verdict=(c.get('verdict') or {}).get('verdict'),
                                reviewer_note=(c.get('verdict') or {}).get('note', ''),
                                signature=c.get('signature'), model_file=rel,
                                model_pixel_sha256=model_asset['pixel_sha256'],
                                tile_source=c.get('tile_source'), donor=donor,
                                width=c['width'], has_real_example=False, quilt_file=quilt_file)
                    if payload:
                        meta['render_model'] = payload
                    meta.update(ancestor_docs=sorted(model_ancestors(c,row,{a['key']:a for a in cards},manifests=rows)),
                        normalized_snapshot=self.normalized_snapshot, renderable=True,
                        synthesis_eligible=True,recognition_eligible=True,recognition_support='generated-model',
                        rendering=rendering_payload(c,row),
                        normalized_sources=self.resolved_sources(c,row,{a['key']:a for a in cards}))
                    result.append(dict(meta=meta, card=c, manifest=row, tile=rgb, model=model))
                for did, info in sorted(self.doc_info.items()):
                    if info['split'] == 'validation':
                        for entry in self.docs[did]['legend']:
                            result.append(self.legend_pattern(did, entry, folder))
                if read(paths.STATE, {}).get('built_at') != state.get('built_at'):
                    continue
                _write(folder / 'lab-snapshot.json', dict(state_built_at=state.get('built_at'),
                    cards=cards, patterns=[rows[c['key']] for c in cards]))
                return result, state
            except FileNotFoundError:
                time.sleep(2)
        raise RuntimeError('Lab kept changing during snapshot; retry after its rebuild')

    def safe_model(self, card, manifest, cards):
        """Archives cannot bypass current quality, version, or ancestor guards."""
        try:
            ancestors, refs = model_lineage(card, manifest, cards,manifests=getattr(self,'model_manifests',{}))
            if any(self.doc_info.get(d, {}).get('split') != 'train' for d in ancestors):
                return False
            if card.get('source') in ('agent-made','unlisted'):
                merged = dict(card,lineage=[card,manifest])
                if not learn.lineage_allowed(merged,self.docs,self.policy):
                    return False
            # Saved proposals lack a complete normalized source lineage. Rebuild
            # their legend instead of trusting unknown column references.
            if not refs or card.get('tile_source') == 'legend swatch':
                return False
            for ref in refs:
                row = self.normalized.get(ref['sample_id'])
                if not row or (row['document_id'],row['annotation_id']) not in self.eligible:
                    return False
                if row['annotation_version'] != ref['annotation_version']:
                    return False
                if ref.get('pixel_sha256',ref.get('source_pixel_sha256',row['pixel_sha256']))!=row['pixel_sha256']:
                    return False
                if self.doc_info.get(row['document_id'],{}).get('split')!='train' or row['document_id'] not in ancestors:
                    return False
            if not ancestors <= {self.normalized[r['sample_id']]['document_id'] for r in refs}:
                return False
            # Legacy axis metadata may be absent: retain only finite legend evidence.
            rendering_payload(card,manifest)
            if card.get('tile_is_preview_only') or manifest.get('tile_is_preview_only'):
                return False
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def resolved_sources(self, card, manifest, cards):
        _, refs = model_lineage(card,manifest,cards,manifests=getattr(self,'model_manifests',{}))
        return [dict(r,source_pixel_sha256=self.normalized[r['sample_id']]['pixel_sha256']) for r in refs]

    @staticmethod
    def frame_for(meta, measurements, rng):
        pool = [m for m in measurements if m['split']=='train' and m['doc']==meta['doc']]
        pool = pool or [m for m in measurements if m['split']=='train' and m['family']==meta['family']]
        pool = pool or [m for m in measurements if m['split']=='train']
        source = pool[int(rng.integers(len(pool)))] if pool else None
        width = meta['width']
        mask = np.zeros(width, bool)
        colours = np.zeros((width,3), np.uint8)
        for rail in (source or {}).get('rails', []):
            x = rail['offset'] if rail['side']=='left' else width-1-rail['offset']
            if 0 <= x < width:
                mask[x] = True
                colours[x] = rail['colour']
        return dict(x0=0,x1=width,mask=mask,ink=colours[mask]), source

    @staticmethod
    def nuisance_for(meta, measurements, rng):
        pool = [m for m in measurements if m['split']=='train' and m['family']==meta['family']]
        pool = pool or [m for m in measurements if m['split']=='train']
        if not pool:
            return None
        source = dict(pool[int(rng.integers(len(pool)))])
        source['rails'] = []  # The frame is applied independently to both arms.
        source['stroke_delta'] = (source.get('legend_stroke_delta',0)
            if meta.get('tile_provenance',{}).get('kind')=='legend-swatch-only' else 0)
        return source

    def context_library(self, folder, humans):
        """Bounded training margin patches; component labels are heuristic."""
        library, unavailable = [], []
        seen = set()
        for row in humans:
            did = row['doc']
            if row['split']!='train' or did in seen:
                continue
            seen.add(did)
            doc = self.docs[did]
            column = doc['boxes']['interpreted_lithology_column']
            x0 = column[2]+4
            x1 = min(doc['width'],x0+96)
            y0 = row['box'][1]
            y1 = min(doc['height'],y0+96)
            if x1-x0 < 8 or y1-y0 < 8:
                continue
            box = [x0,y0,x1,y1]
            try:
                rgb = self.crop(did,box)
            except ValueError as error:
                if not str(error).startswith('Unsafe TIFF crop-cache miss:'):
                    raise
                unavailable.append(dict(doc=did,box=box,reason='Bounded native margin cache unavailable'))
                continue
            mask = (rgb.max(2)<80).astype(np.uint8)*255
            mask = realism.component_budget(mask,rgb,np.full_like(rgb,245),.15)
            if not mask.any():
                continue
            key = digest([did,box,pixel_hash(rgb),SCHEMA])[:24]
            rel = f'nuisance/{key}.png'
            mr = f'nuisance/{key}-mask.png'
            save_rgb(folder,rel,rgb,False)
            Image.fromarray(mask).save(folder/mr)
            library.append(dict(id=key,doc=did,ancestor_docs=[did],family=row['family'],split='train',overlay_usable=True,
                box=box,colours=rel,mask=mr,normalized_snapshot=self.normalized_snapshot,
                mask_pixel_sha256=pixel_hash(mask),colours_pixel_sha256=pixel_hash(rgb),
                semantic_status='unverified dark margin components; text/grid proxy'))
        _write(folder/'nuisance.json', dict(components=library,unavailable=unavailable,
            policy='Train-only page margins; whole dark components. No held-column pixels.'))
        return library

    def human_rows(self):
        """Read only accepted, single-label, unflagged human boxes."""
        if not hasattr(self, 'eligible'):
            self.load_eligibility()
        frozen = ({(r['doc'],r['annotation_id']): r for r in read(self.baseline/'real.json')}
                  if self.baseline else None)
        rows = []
        for did, doc in sorted(self.docs.items()):
            info = self.doc_info[did]
            if info['split'] == 'test':
                continue
            names = {e['id']: e['label_printed'] for e in doc['legend']}
            for seg in doc['segments']:
                if (seg.get('status') != 'approved' or seg.get('flags')
                        or len(seg.get('legend_ids', [])) != 1):
                    continue
                lid = seg['legend_ids'][0]
                eligibility = self.eligible.get((did, seg['id']))
                if eligibility is None:
                    continue
                if frozen is not None:
                    prior = frozen.get((did, seg['id']))
                    if not prior or prior['box'] != seg['box'] or prior['name'] != canonical(names.get(lid,'')):
                        continue
                    if prior.get('annotation_version', eligibility['annotation_version']) != eligibility['annotation_version']:
                        continue
                box = seg['box']
                if box[2] - box[0] < 8 or box[3] - box[1] < 6:
                    continue
                rows.append(dict(class_id=class_id(did, lid), name=canonical(names.get(lid, str(lid))),
                    label=names.get(lid, str(lid)), doc=did, family=info['family'], well=info['well'],
                    split=info['split'], evidence='human-accepted', kind='real', status='approved',
                    box=box, annotation_id=seg['id'], sample_id=eligibility['sample_id'],
                    annotation_version=eligibility['annotation_version'], legend_version=eligibility['legend_version'],
                    quality=eligibility['quality'], quality_issues=eligibility['quality_issues'],
                    normalized_snapshot=self.normalized_snapshot, source_pixel_sha256=eligibility['pixel_sha256']))
        return rows

    def build(self):
        from .dataset_loop import snapshot as watch_snapshot
        watch_fingerprint = self.initial_fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        nums = [int(p.name[1:]) for p in self.root.glob('v[0-9]*') if p.is_dir() and p.name[1:].isdigit()]
        number = max(nums, default=0) + 1
        folder = self.root / f'.building-v{number}-{os.getpid()}-{time.time_ns()}'
        folder.mkdir()
        print('BUILD snapshot', flush=True)
        patterns, state = self.snapshot(folder)
        human = self.human_rows()
        by_class = defaultdict(list)
        for row in human:
            by_class[row['class_id']].append(row)
        _write(folder / 'split-policy.json', self.policy)
        _write(folder / 'wells.json', self.doc_info)
        _write(folder/'eligibility.json', dict(snapshot=self.normalized_snapshot,
            policy_sha256=digest(self.policy), records=list(self.eligible.values()),
            quarantined_test_hashes=sorted(self.test_hashes)))
        _write(folder/'build-dependencies.json',dict(code=self.code_dependencies,
            normalized_snapshot=self.normalized_snapshot,policy_sha256=digest(self.policy)))
        measurements, pools = [], defaultdict(list)
        real_index, previews = [], {}
        # Real evidence is saved separately, never in the synthetic training manifest.
        for row in human:
            rgb = self.crop(row['doc'], row['box'])
            if row['box'][3]-row['box'][1] <= 480 and pixel_hash(rgb) != row['source_pixel_sha256']:
                raise ValueError('Normalized pixels changed: ' + row['sample_id'])
            rel = 'real/' + digest([row['doc'], row['annotation_id'], row['box']])[:24] + '.png'
            row.update(save_rgb(folder, rel, rgb))
            row['source_box'] = list(row['box'])
            row['crop_box'] = [row['box'][0],row['box'][1],
                               row['box'][0]+rgb.shape[1],row['box'][1]+rgb.shape[0]]
            row['image_size'] = [rgb.shape[1],rgb.shape[0]]
            real_index.append(row)
        print('BUILD measured real boxes', len(real_index), flush=True)
        style_cache = {}
        for pattern in patterns:
            meta, tile = pattern['meta'], pattern['tile']
            actual = by_class[meta['class_id']]
            # Only current quality-clean normalized human boxes fit appearance.
            evidence = [(r['box'], 'human-accepted') for r in actual]
            meta['has_real_example'] = bool(evidence)
            meta['has_human_example'] = bool(actual)
            if meta['split'] == 'validation' or not meta.get('renderable',True):
                # Real validation images are truth/preview only, never fitted
                # corruption fields or augmentation donors.
                if actual:
                    previews[meta['class_id']] = self.crop(meta['doc'], actual[0]['box'])
                continue
            for i, (box, origin) in enumerate(evidence):
                rgb = self.crop(meta['doc'], box)
                params, mask, colours = realism.measure(tile, rgb)
                legend = next((e for e in self.docs[meta['doc']]['legend'] if e['id']==meta['legend_id']),None)
                if legend:
                    if meta['class_id'] not in style_cache:
                        style_cache[meta['class_id']] = legend_model(self.crop(meta['doc'],legend['box']))[1]
                    swatch = style_cache[meta['class_id']]
                    # Fit the family style from TRAINING legends/columns only.
                    losses = []
                    for delta in (0,1,-1):
                        predicted, _ = realism.aligned(realism.stroke(swatch['tile'],delta),rgb)
                        diff = np.linalg.norm(predicted.astype(float)-rgb.astype(float),axis=2)
                        interior = diff[:,3:-3] if diff.shape[1]>6 else diff
                        losses.append((float(np.minimum(interior,45).mean()),delta))
                    params['legend_stroke_delta'] = min(losses,key=lambda x:x[0])[1]
                source = next(r for r in actual if r['box']==box)
                mid = digest([meta['class_id'], box, origin, source['annotation_version'],
                    self.normalized_snapshot,pixel_hash(tile),SCHEMA])[:24]
                mp, cp = f'overlays/{mid}-mask.png', f'overlays/{mid}-rgb.png'
                (folder / 'overlays').mkdir(exist_ok=True)
                Image.fromarray(mask).save(folder / mp)
                Image.fromarray(colours).save(folder / cp)
                record = dict(params, id=mid, class_id=meta['class_id'], doc=meta['doc'],
                    family=meta['family'], split=meta['split'], evidence=origin, box=box, mask=mp, colours=cp)
                record.update({k:source[k] for k in ('sample_id','annotation_version','quality','quality_issues',
                    'normalized_snapshot','source_pixel_sha256')})
                record.update(ancestor_docs=meta['ancestor_docs'],model_sha256=pixel_hash(tile),
                    mask_pixel_sha256=pixel_hash(mask),colours_pixel_sha256=pixel_hash(colours),
                    model_signature=meta['signature'],policy_sha256=digest(self.policy),fitter_version='dataset-realism-3',
                    fitter_sha256=self.code_dependencies['realism.py'])
                measurements.append(record)
                # Pool exclusively within a split. No validation pixels augment train.
                pools[(meta['split'], meta['doc'])].append(record)
                pools[(meta['split'], meta['family'])].append(record)
                pools[(meta['split'], '*')].append(record)
                if i == 0:
                    previews[meta['class_id']] = rgb
        _write(folder / 'real.json', real_index)
        distributions = {}
        for key in ('doc', 'family'):
            grouped = defaultdict(list)
            for r in measurements:
                grouped[(r['split'], r[key])].append(r)
            distributions[key] = {f'{split}/{value}': dict(n=len(rr),
                blur=[r['blur'] for r in rr], noise=[r['noise'] for r in rr],
                drift=[r['drift'] for r in rr], overlay_fraction=[r['overlay_fraction'] for r in rr],
                usable_overlays=sum(r['overlay_usable'] for r in rr)) for (split, value), rr in grouped.items()}
        _write(folder / 'realism.json', dict(measurements=measurements, distributions=distributions,
            policy='Native cyclic least squares; blur grid fit, robust RGB drift/noise, strong residual overlays. Split-isolated pools.',
            limits='Residuals can contain motif/phase errors; fields with >35% disagreement are not pasted. Crop windows capped at 480 rows.'))
        self._measurements = measurements
        nuisance = self.context_library(folder,human)
        rows, examples = [], {}
        for index, pattern in enumerate(patterns):
            meta, tile = pattern['meta'], pattern['tile']
            if not meta.get('renderable',True):
                continue
            rng = np.random.default_rng(int(digest([meta['class_id'], meta['signature'], SCHEMA])[:16], 16))
            model = local_model(pattern['model'])
            for k in range(4):
                # Native motif scale; vary window height rather than squashing
                # every real box and 240-row strip into the same tiny image.
                height = (48, 96, 144, 240)[k]
                rails, frame_source = self.frame_for(meta,measurements,rng)
                clean = render.sample(model, rails, height, rng, y0=0)
                measurement = self.nuisance_for(meta,measurements,rng)
                context = [n for n in nuisance if n['family']==meta['family']] or nuisance
                context = context[int(rng.integers(len(context)))] if context and k%2 else None
                overlay = self.overlay(folder, context) if context else None
                augmented, params = dataset_augment(clean, measurement, overlay, rng)
                # Partial neighbouring beds are a class-agnostic contamination
                # arm. Their row extent is explicit, and columns never use it.
                neighbours = [p for p in patterns if p['meta']['doc']==meta['doc']
                    and p['meta']['class_id']!=meta['class_id'] and p['meta'].get('renderable',True)]
                if neighbours and k==3:
                    donor = neighbours[int(rng.integers(len(neighbours)))]
                    n = int(rng.integers(1,11))
                    augmented[:n] = render.sample(local_model(donor['model']),rails,n,rng)
                    params['neighbour'] = dict(class_id=donor['meta']['class_id'],rows=[0,n],
                        ancestor_docs=donor['meta']['ancestor_docs'])
                render.frame(augmented,rails)
                params.update(frame_source_id=(frame_source or {}).get('id'),
                    nuisance_component_id=(context or {}).get('id'),
                    identity=bool(np.array_equal(clean,augmented)))
                for variant, rgb, aug in [('clean', clean, {}), ('augmented', augmented, params)]:
                    sid = digest([meta['class_id'], meta['signature'], k, variant, SCHEMA])[:24]
                    row = dict(meta, kind='synthetic', format='strip', variant=variant, sample_id=sid,
                               augmentation=aug, measurement_id=(measurement or {}).get('id') if variant=='augmented' else None,
                               frame_source_id=(frame_source or {}).get('id'),frame_policy='training-only family prior',
                               augmentation_basis=('document' if measurement and measurement['doc'] == meta['doc'] else
                                    'family' if measurement and measurement['family'] == meta['family'] else
                                    'global same split' if measurement else 'unmeasured identity'))
                    row.update(save_rgb(folder, f'strips/{sid}.png', rgb))
                    rows.append(row)
                if k == 0:
                    examples[meta['class_id']] = (clean, augmented)
            self.ablation(folder, pattern, rng, rows)
            if index % 50 == 0:
                print('BUILD patterns', index + 1, '/', len(patterns), flush=True)
        self.columns(folder, patterns, human, pools, rows)
        sheets = self.sheets(folder, [p for p in patterns if p['meta']['class_id'] in examples], previews, examples)
        with (folder / 'manifest.jsonl').open('w', encoding='utf-8') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
        counts = lambda key: dict(Counter(r[key] for r in rows))
        stats = dict(version=f'v{number}', schema=SCHEMA, created_at=time.time(), samples=len(rows),
            patterns=len(patterns), columns=sum(r['format'] == 'column' for r in rows),
            controlled_baseline=self.baseline.name if self.baseline else None,
            counts_by_class=counts('class_id'), counts_by_source=counts('source'), counts_by_family=counts('family'),
            counts_by_split=counts('split'), counts_by_variant=counts('variant'),
            real_example_coverage=dict(patterns=len(patterns), with_real=sum(p['meta']['has_real_example'] for p in patterns),
                with_human=sum(p['meta']['has_human_example'] for p in patterns), measured_crops=len(measurements)),
            real_boxes=dict(Counter(r['split'] for r in real_index)), sheets=sheets,
            classes=[p['meta'] for p in patterns], input_fingerprint=digest([p['meta'] for p in patterns]),
            watch_fingerprint=watch_fingerprint)
        _write(folder / 'stats.json', stats)
        from .dataset_eval import leak_audit
        audit = leak_audit(folder)
        _write(folder/'leak-audit.json',audit)
        if not audit['passed']:
            raise ValueError('Dataset leak audit failed: ' + json.dumps(audit))
        (folder / 'CARD.md').write_text(self.card(stats), encoding='utf-8')
        final = self.root / f'v{number}'
        # Antivirus/indexers can briefly keep a newly written PNG directory open.
        for attempt in range(10):
            try:
                folder.rename(final)
                break
            except PermissionError:
                if attempt==9:
                    raise
                time.sleep(.1*(attempt+1))
        print('PASS build', final.name, 'patterns', len(patterns), 'samples', len(rows), flush=True)
        return final

    @staticmethod
    def overlay(folder, measurement):
        if not measurement or not measurement.get('overlay_usable'):
            return None
        with Image.open(folder / measurement['mask']) as im:
            mask = np.asarray(im).copy()
        with Image.open(folder / measurement['colours']) as im:
            colours = np.asarray(im.convert('RGB')).copy()
        if measurement.get('mask_pixel_sha256') and pixel_hash(mask)!=measurement['mask_pixel_sha256']:
            raise ValueError('Overlay mask pixel hash mismatch')
        if measurement.get('colours_pixel_sha256') and pixel_hash(colours)!=measurement['colours_pixel_sha256']:
            raise ValueError('Overlay colour pixel hash mismatch')
        return mask, colours

    def ablation(self, folder, pattern, rng, rows):
        """Use the scanner's saved column model and its native legend extractor."""
        meta = pattern['meta']
        if meta['split'] != 'train' or not meta['has_real_example'] or str(meta['legend_id']).startswith('u'):
            return
        if not self.safe_model(pattern['card'],pattern['manifest'],{pattern['card'].get('key'):pattern['card']}):
            return
        report = read(paths.CANDIDATES / (meta['doc'] + '.json'), {})
        entry = next((e for e in report.get('entries', []) if e['legend_id'] == meta['legend_id']), None)
        if not entry or entry.get('tier') not in ('human', 'verified') or not entry.get('data_file'):
            return
        # No column learning here: use the existing cached pre-selection model.
        column = candidates.load_data(paths.CANDIDATES / entry['data_file'])
        if not entry.get('swatch_box'):
            return
        _, swatch, repeat = candidates.swatch(self.crop(meta['doc'], entry['swatch_box']))
        if swatch is None or repeat.get('seam_risk',True):
            return
        for variant, model in [('legend-only', swatch), ('real-learned', column)]:
            for k in range(4):
                paired_rng = np.random.default_rng(int(digest([meta['class_id'],'ablation',k,SCHEMA])[:16],16))
                fractional_phase = paired_rng.random(2)
                paired = dict(model)
                if paired['kind']=='tile':
                    ph,pw = paired['tile'].shape[:2]
                    paired.update(anchored=True,phase=[int(fractional_phase[0]*ph),int(fractional_phase[1]*pw)])
                rails,_ = self.frame_for(meta,self._measurements,paired_rng)
                rgb = render.sample(local_model(paired),rails,(48,96,144,240)[k],paired_rng,y0=0)
                sid = digest([meta['class_id'], meta['signature'], variant, k, SCHEMA])[:24]
                row = dict(meta, kind='synthetic', format='strip', variant=variant, sample_id=sid,
                           augmentation={}, ablation_seam_risk=repeat.get('seam_risk'),
                           paired_phase_fraction=fractional_phase.tolist(),paired_seed=int(digest([meta['class_id'],'ablation',k,SCHEMA])[:16],16),
                           ablation_model='cached column' if variant == 'real-learned' else 'native legend extraction')
                row.update(save_rgb(folder, f'strips/{sid}.png', rgb))
                rows.append(row)

    def columns(self, folder, patterns, humans, pools, rows):
        by_doc = defaultdict(list)
        by_family = defaultdict(list)
        human_doc = defaultdict(list)
        family_transition = defaultdict(Counter)
        for p in patterns:
            if p['meta'].get('renderable',True):
                by_doc[p['meta']['doc']].append(p)
        for r in humans:
            if r['split']!='train':
                continue
            h = r['box'][3] - r['box'][1]
            by_family[(r['split'], r['family'])].append(h)
            human_doc[r['doc']].append(r)
        for did, records in human_doc.items():
            ordered = sorted(records, key=lambda r: r['box'][1])
            info = self.doc_info[did]
            for a, b in zip(ordered, ordered[1:]):
                if 0 <= b['box'][1] - a['box'][3] <= 4:
                    family_transition[(info['split'], info['family'])][(a['name'], b['name'])] += 1
        # Machine intervals are measurements of thickness/order, not truth.
        for did, d in self.docs.items():
            info = self.doc_info[did]
            if info['split'] != 'train':
                continue
            for g in self.prefill.get(d['name'], []):
                h = int(g['bottom_y']) - int(g['top_y'])
                if h > 0:
                    by_family[(info['split'], info['family'])].append(h)
            guesses = sorted(self.prefill.get(d['name'], []), key=lambda g: g['top_y'])
            for a, b in zip(guesses, guesses[1:]):
                if 0 <= b['top_y'] - a['bottom_y'] <= 4:
                    family_transition[(info['split'], info['family'])][
                        (canonical(a.get('suggestion') or ''), canonical(b.get('suggestion') or ''))] += 1
        _write(folder / 'thickness.json', {f'{s}/{f}': dict(n=len(v), values=v)
                                         for (s, f), v in by_family.items()})
        _write(folder / 'transitions.json', {f'{s}/{f}': [dict(previous=a, next=b, n=n) for (a, b), n in v.items()]
                                           for (s, f), v in family_transition.items()})
        for did, pp in by_doc.items():
            # A donor can force one class into validation; never mix splits in a column.
            for split in sorted({p['meta']['split'] for p in pp}):
                group = [p for p in pp if p['meta']['split'] == split]
                if not group:
                    continue
                meta = group[0]['meta']
                doc = self.docs[did]
                models = {p['meta']['legend_id']: p['model'] for p in group}
                names = {p['meta']['name']: p['meta']['legend_id'] for p in group}
                thick = defaultdict(list)
                transition = defaultdict(Counter)
                ordered = sorted(human_doc[did], key=lambda r: r['box'][1])
                for r in ordered:
                    if r['name'] in names:
                        thick[names[r['name']]].append(r['box'][3] - r['box'][1])
                for a, b in zip(ordered, ordered[1:]):
                    if 0 <= b['box'][1] - a['box'][3] <= 4 and a['name'] in names and b['name'] in names:
                        transition[names[a['name']]][names[b['name']]] += 1
                guesses = sorted(self.prefill.get(doc['name'], []), key=lambda g: g['top_y'])
                if split!='train':
                    guesses = []
                for g in guesses:
                    n = canonical(g.get('suggestion') or '')
                    h = int(g['bottom_y']) - int(g['top_y'])
                    if n in names and h > 0:
                        thick[names[n]].append(h)
                for a, b in zip(guesses, guesses[1:]):
                    na, nb = canonical(a.get('suggestion') or ''), canonical(b.get('suggestion') or '')
                    if 0 <= b['top_y'] - a['bottom_y'] <= 4 and na in names and nb in names:
                        transition[names[na]][names[nb]] += 1
                family_counts = family_transition.get(('train', meta['family']), {})
                for (na, nb), count in family_counts.items():
                    if na in names and nb in names and not transition.get(names[na]):
                        # Add the complete outgoing family distribution below.
                        transition[names[na]] = Counter({names[n]: value for (a, n), value in family_counts.items()
                                                       if a == na and n in names})
                family_thickness = by_family.get(('train', meta['family']), [])
                if not family_thickness:
                    family_thickness = [h for (s, _), v in by_family.items() if s == 'train' for h in v]
                if not family_thickness:
                    # Do not invent a geological thickness distribution.
                    continue
                for lid in models:
                    if not thick[lid]:
                        thick[lid] = family_thickness
                prior = Counter()
                for r in ordered:
                    if r['name'] in names:
                        prior[names[r['name']]] += r['box'][3]-r['box'][1]
                prior_basis = 'training document human row frequency'
                if not prior:
                    for g in guesses:
                        n = canonical(g.get('suggestion') or '')
                        if n in names:
                            prior[names[n]] += max(0,int(g['bottom_y'])-int(g['top_y']))
                    prior_basis = 'training document machine row frequency'
                if not prior:
                    eligible_humans = [r for r in humans if r['split']=='train' and r['family']==meta['family']]
                    if not eligible_humans:
                        eligible_humans = [r for r in humans if r['split']=='train']
                    for r in eligible_humans:
                        if r['name'] in names:
                            prior[names[r['name']]] += r['box'][3]-r['box'][1]
                    prior_basis = 'training family/global human row frequency'
                if not prior:
                    continue
                width = meta['width']
                mask = np.zeros(width, bool)
                rail_rgb = None
                if ordered and split == 'train':
                    rail_rgb = self.crop(did, ordered[0]['box'])
                    if rail_rgb.shape[1] == width:
                        dark = (rail_rgb.max(2) < 80).mean(0) > .75
                        edges = np.zeros(width, bool)
                        edges[:3] = True
                        edges[-3:] = True
                        mask = dark & edges
                rails = dict(x0=0, x1=width, mask=mask, ink=(np.median(rail_rgb[:, mask], axis=0).astype(np.uint8)
                             if mask.any() else np.empty((0, 3), np.uint8)))
                rng = np.random.default_rng(int(digest([did, split, 'column', SCHEMA])[:16], 16))
                rails, _ = self.frame_for(meta,self._measurements,rng)
                pool = pools.get(('train', did)) or pools.get(('train', meta['family'])) or pools.get(('train', '*')) or []
                by_lid = {p['meta']['legend_id']: p['meta'] for p in group}
                for k in range(2):
                    clean, truth = weighted_column(models, rails, thick, transition,prior,1200,rng)
                    mid = pool[int(rng.integers(len(pool)))] if pool else None
                    # A class-specific residual field is inappropriate across
                    # a column's different layers. Apply fitted colour/noise only.
                    augmented, aug = dataset_augment(clean, mid, None, rng)
                    render.frame(augmented,rails)
                    sid = digest([did, split, 'column', k, [(x['class_id'], x['signature']) for x in by_lid.values()]])[:24]
                    mask_rgb = np.zeros(clean.shape[:2], np.uint16)
                    palette = {lid: i + 1 for i, lid in enumerate(sorted(models, key=str))}
                    for t in truth:
                        m = by_lid[t['legend_id']]
                        t.update(class_id=m['class_id'], name=m['name'], mask_value=palette[t['legend_id']],
                                 source=m['source'], partial=(t['bottom'] == 1200),
                                 thickness_basis='human accepted boxes and machine guesses; family fallback')
                        mask_rgb[t['top']:t['bottom']] = palette[t['legend_id']]
                    mask_rel, truth_rel = f'columns/{sid}-labels.png', f'columns/{sid}.json'
                    (folder / 'columns').mkdir(exist_ok=True)
                    Image.fromarray(mask_rgb).save(folder / mask_rel)
                    _write(folder / truth_rel, dict(layers=truth, height=1200, width=width,
                                                   rails=np.asarray(rails['mask']).tolist(),transitions_basis='training human + machine intervals',
                                                   class_prior=dict(prior),class_prior_basis=prior_basis))
                    row = dict(meta, sample_id=sid, kind='synthetic', format='column', variant='augmented',
                        class_id='column:' + did, source='stacked', augmentation=aug,
                        name='stacked column', reviewer_verdict=None, reviewer_note='See layer_metadata',
                        label_mask_png=mask_rel, truth_json=truth_rel, classes=[m['class_id'] for m in by_lid.values()],
                        layer_metadata=list(by_lid.values()))
                    row.update(save_rgb(folder, f'columns/{sid}.png', augmented, normalize=False))
                    row['clean_png'] = f'columns/{sid}-clean.png'
                    save_rgb(folder, row['clean_png'], clean, normalize=False)
                    rows.append(row)

    def sheets(self, folder, patterns, previews, examples):
        by_source = defaultdict(list)
        for p in patterns:
            by_source[p['meta']['source']].append(p)
        sheets = []
        (folder / 'sheets').mkdir()
        for source, pp in sorted(by_source.items()):
            # Show up to twelve: real evidence first, then source-only classes.
            pp = sorted(pp, key=lambda p: (p['meta']['class_id'] not in previews, p['meta']['class_id']))[:12]
            sheet = Image.new('RGB', (850, 38 + 270 * len(pp)), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f'{source}: real / clean / augmented (same native pixel scale)', fill='black')
            for i, p in enumerate(pp):
                meta = p['meta']
                real = previews.get(meta['class_id'])
                clean, aug = examples[meta['class_id']]
                y = 38 + i * 270
                draw.text((8, y), f"{meta['class_id']}  {meta['label']}  {meta['split']}", fill='black')
                for j, rgb in enumerate((real, clean, aug)):
                    x = 10 + j * 280
                    if rgb is None:
                        draw.text((x, y + 36), 'No real column example', fill='black')
                    else:
                        sheet.paste(Image.fromarray(rgb[:240, :270]), (x, y + 22))
            rel = f'sheets/{source}.png'
            sheet.save(folder / rel)
            sheets.append(rel)
        return sheets

    @staticmethod
    def card(stats):
        return f"""# {stats['version']} synthetic pattern dataset (schema {SCHEMA})

{stats['patterns']} document classes; {stats['samples']} samples. Native strips,
letterboxes with valid masks, and stacked columns with row truth and masks.
Unrenderable legend classes remain in the evaluation vocabulary and chance
denominator. Seam-only exclusions can supply a finite original native legend
reference for recognition; they never enter repeating generation or augmentation.
Non-lithology symbols remain excluded. Generation-only scores and coverage are
reported separately from legend-conditioned recognition with finite support.

Current normalized eligibility, annotation/legend versions and test-pixel quarantine
are enforced even for controlled baselines. Only quality-clean training human
annotations fit style. Every measurement binds source snapshot, pixel hash,
annotation version, tile signature/hash, policy and ancestor documents. Models
with unknown/revoked ancestry fall back to their legend; test sources are forbidden.
Validation legends are permitted inference support, explicitly tagged. Validation
column pixels never enter generation, statistics, style, nuisance or priors.

Frames are rendered with render.sample using training document/family rail priors.
Class-agnostic training family style includes measured stroke correction, bounded
blur/drift and winsorized residual noise. Dark margin components are text/grid
proxies, not verified semantic labels; whole components are placed once. Strip
neighbour contamination is explicitly localized. Identities and unmeasured styles
are reported separately; no blanket scanner noise is assumed for vector logs.

Column odds use observed training human row frequencies first, training machine
frequencies second and training family/global human fallback. No uniform fallback.
Source labels in layer_metadata describe the actual model. Sparse accepted crop
heights are censored diagnostics rather than certified geological bed thicknesses.

Evaluation pairs document, class and native size without another-log substitution.
Both legend-conditioned oracle-column recognition and the supported-name cross-well
generator ablation use native colour/repeat features. Ablation seeds, fractional
phase and dimensions are paired. Whole-well bootstrap CIs, fixed linear/nonlinear
discriminators, rail/edge shortcut probes and controls are reported. Promotion
requires clean leak audits, near-chance shortcuts and paired within-document
noninferiority to the promoted baseline (95% lower bound >= -2 points).
Few wells, unrenderable classes and unverified nuisance semantics limit the claims.
Detector transfer utility and continuous-window geological fidelity are unmeasured.
"""
def closed_summary(truth, predictions, candidate_counts):
    """Macro over observed document classes, including n=1; no silent filtering."""
    if len(truth)!=len(predictions) or len(truth)!=len(candidate_counts):
        raise ValueError('Truth, predictions and candidate counts must have identical lengths')
    counts, good, chances = Counter(), Counter(), defaultdict(list)
    for row, predicted, n in zip(truth, predictions, candidate_counts):
        cid = row['class_id']
        counts[cid] += 1
        good[cid] += cid == predicted
        chances[cid].append(100 / n if n else 0.)
    per_class = [dict(class_id=cid, name=next(r['name'] for r in truth if r['class_id']==cid),
        n=n, correct=good[cid], accuracy=round(100*good[cid]/n, 3),
        chance=round(float(np.mean(chances[cid])), 3), flag='too few' if n < MIN_N else None)
        for cid, n in sorted(counts.items())]
    n = len(truth)
    return dict(n=n, correct=sum(good.values()), accuracy=round(100*sum(good.values())/n,3) if n else None,
        macro_accuracy=round(float(np.mean([r['accuracy'] for r in per_class])),3) if per_class else None,
        macro_classes=len(per_class), macro_n=n, per_class=per_class,
        chance=round(float(np.mean([100/k if k else 0 for k in candidate_counts])),3) if n else None,
        macro_chance=round(float(np.mean([np.mean(v) for v in chances.values()])),3) if n else None,
        macro_policy='All observed document legend IDs, including classes with fewer than three boxes')


def evaluation_candidates(folder, truth):
    """Build at evaluation time, without any column-trained model as input.

    The source snapshot is saved as pixels, so later evaluations reuse exactly
    the same swatches even while the human lab and other jobs keep changing.
    """
    folder = Path(folder)
    from .dataset_eval import checked_rgb,eligibility_context,source_errors,pair_dependency,require_current_policy
    context = eligibility_context(folder)
    require_current_policy(folder,context)
    for row in truth:
        if source_errors(row,context,real=True):
            raise ValueError('Evaluation truth source no longer eligible')
        checked_rgb(folder,row,context)
    cache = folder/'eval-swatches.json'
    saved = read(cache)
    stats = read(folder/'stats.json')
    dependency = digest([stats['classes'],read(folder/'realism.json'), 'finite-legend-native-v1'])
    if saved is not None:
        if stats.get('schema',1)>=4:
            support = read(folder/'evaluation-support.json',{})
            if support.get('schema')==2:
                dependency = pair_dependency(folder,truth,context)
            if support.get('dependency')!=dependency or support.get('inventory_sha256')!=digest(saved):
                raise ValueError('Evaluation support cache changed or has stale dependencies')
            for row in saved:
                if source_errors(row,context):
                    raise ValueError('Evaluation support source no longer eligible')
                if row.get('file'):
                    raw = checked_rgb(folder,row,context)
                if row.get('format')=='finite-legend':
                    if pixel_hash(raw)!=row['tile_provenance']['raw_pixel_sha256']:
                        raise ValueError('Finite legend source pixels changed')
        return saved
    classes = [r for r in stats['classes'] if r['split']=='validation']
    measured = [m for m in read(folder/'realism.json')['measurements'] if m['split']=='train']
    rows = []
    for did in sorted({r['doc'] for r in truth}):
        for original in [r for r in classes if r['doc']==did]:
            meta = dict(original)
            if source_errors(meta,context):
                raise ValueError('Evaluation model source no longer eligible')
            if not meta.get('renderable',True):
                rows.append(dict(meta,kind='synthetic',format='unrenderable',variant='unrenderable'))
                if meta.get('recognition_eligible',False) and meta.get('recognition_support')=='finite-legend':
                    provenance = meta['tile_provenance']
                    raw_file = provenance['raw_file']
                    with Image.open(folder/raw_file) as image:
                        raw = np.asarray(image.convert('RGB')).copy()
                    if pixel_hash(raw)!=provenance['raw_pixel_sha256']:
                        raise ValueError('Finite legend source pixels changed: '+meta['class_id'])
                    if candidates.not_lithology(meta['label'],raw):
                        raise ValueError('Non-lithology cannot supply finite recognition support')
                    from .dataset_eval import current_test_hashes
                    if pixel_hash(raw) in current_test_hashes():
                        raise ValueError('Test-identical finite legend')
                    for variant in ('clean','augmented'):
                        rows.append(dict(meta,kind='reference',format='finite-legend',variant=variant,
                            file=raw_file,pixel_sha256=pixel_hash(raw),
                            recognition_policy='Original finite native legend; no tiling, resizing, or augmentation'))
                continue
            with Image.open(folder/meta['model_file']) as image:
                tile = np.asarray(image.convert('RGB')).copy()
            model = dict(meta.get('rendering',dict(kind='tile',anchored=False,phase=[0,0])),tile=tile)
            if meta.get('render_model'):
                model = learn.load_model(meta['render_model'],root=folder,signature=meta['signature'])
            model = local_model(model)
            lid = meta['legend_id']
            rng = np.random.default_rng(int(digest([did, lid, 'eval'])[:16],16))
            for k in range(4):
                height = (48,96,144,240)[k] if read(folder/'stats.json').get('schema',1)>=2 else 240
                rails, frame_source = DatasetBuilder.frame_for(meta,measured,rng)
                clean = render.sample(model,rails,height,rng,y0=0)
                # The augmented held-well arm uses training-only nuisance
                # measurements; never held-well real pixels or residual colours.
                measurement = DatasetBuilder.nuisance_for(meta,measured,rng)
                aug, params = dataset_augment(clean, measurement, None, rng)
                render.frame(aug,rails)
                params.update(identity=bool(np.array_equal(clean,aug)),frame_source_id=(frame_source or {}).get('id'))
                for variant, rgb in [('clean',clean),('augmented',aug)]:
                    row = dict(meta, kind='synthetic', format='strip', variant=variant,
                        augmentation=params if variant=='augmented' else {},
                        augmentation_donor_split='train' if variant=='augmented' and measurement else None)
                    row.update(save_rgb(folder, f'evaluation/strips/{digest([did,lid,k,variant])[:24]}.png',rgb,False))
                    rows.append(row)
    _write(cache,rows)
    _write(folder/'evaluation-support.json',dict(schema=2,dependency=pair_dependency(folder,truth,context),inventory_sha256=digest(rows),
        policy='finite-legend-native-v1; separate generation and recognition eligibility'))
    return rows


def within_document(features, candidates_, truth, algorithm, variant):
    from .dataset_eval import fit_predict
    predictions, sizes, documents = [], [], []
    ordered = []
    for did in sorted({r['doc'] for r in truth}):
        real = [r for r in truth if r['doc']==did]
        for row in real:
            assert_truth(row)
        train = [r for r in candidates_ if r['doc']==did and r['variant']==variant]
        ids = {r['class_id'] for r in candidates_ if r['doc']==did}
        if any(r['tile_provenance']['real_crop_inputs'] or r['tile_provenance']['kind']!='legend-swatch-only' for r in train):
            raise ValueError('Validation crop leakage into legend candidates')
        if any(r['split']!='validation' for r in train):
            raise ValueError('Within-document candidates must belong to validation documents')
        # Reuse the identical baseline algorithms but predict legend IDs. Two
        # entries bearing the same name remain separate candidate classes.
        pred = fit_predict(features, [dict(r,name=r['class_id']) for r in train],real,algorithm)
        local = closed_summary(real,pred,[len(ids)]*len(real))
        local.update(doc=did,candidates=len(ids),train_n=len(train),
                     missing_truth_ids=sorted({r['class_id'] for r in real}-ids))
        documents.append(local)
        predictions.extend(pred); sizes.extend([len(ids)]*len(real)); ordered.extend(real)
    result = closed_summary(ordered,predictions,sizes)
    result.update(train_n=sum(r['train_n'] for r in documents), documents=documents,
        predictions=[dict(doc=r['doc'],annotation_id=r['annotation_id'],actual=r['class_id'],
                          predicted=p,candidate_count=k,chance=100/k if k else 0.)
                     for r,p,k in zip(ordered,predictions,sizes)],
        task='legend-conditioned recognition with finite support',
        mean_document_accuracy=round(float(np.mean([r['accuracy'] for r in documents])),3) if documents else None)
    generated = {r['class_id'] for r in candidates_ if r.get('format')=='strip' and r['variant']==variant}
    supported = {r['class_id'] for r in candidates_ if r.get('file') and r['variant']==variant}
    result['support_coverage'] = dict(truth_n=len(ordered),observed_classes=len({r['class_id'] for r in ordered}),
        generation_supported_boxes=sum(r['class_id'] in generated for r in ordered),
        generation_supported_classes=len(generated & {r['class_id'] for r in ordered}),
        recognition_supported_boxes=sum(r['class_id'] in supported for r in ordered),
        recognition_supported_classes=len(supported & {r['class_id'] for r in ordered}),
        finite_reference_classes=len({r['class_id'] for r in candidates_ if r.get('format')=='finite-legend'}))
    result['support_coverage']['generation_well_ceiling'] = float(np.mean([
        100*np.mean([r['class_id'] in generated for r in ordered if r['doc']==doc['doc']]) for doc in documents])) if documents else None
    return result


def add_document_evaluation(folder, result):
    """Extend the original evaluation without changing its promotion metrics."""
    from .dataset_eval import Features, classifier_run, markdown, prediction_uncertainty, finalize_evaluation
    folder = Path(folder)
    started = time.monotonic()
    truth = [r for r in read(folder/'real.json') if r['split']=='validation']
    samples = evaluation_candidates(folder,truth)
    class NativeFeatures(Features):
        def get(self, row):
            native, _ = super().get(row)
            return native, native
    features = NativeFeatures(folder)
    manifest = [json.loads(s) for s in (folder/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if 'logistic' in result['fake_to_real']:
        runs = {variant:classifier_run(features,[r for r in manifest if r['split']=='train' and r['format']=='strip' and r['variant']==variant],truth,'logistic')
                for variant in ('clean','augmented')}
        runs['macro_delta_points'] = round(runs['augmented']['macro_accuracy']-runs['clean']['macro_accuracy'],3) if runs['clean']['macro_accuracy'] is not None and runs['augmented']['macro_accuracy'] is not None else None
        result['clean_vs_augmented']['logistic'] = runs
    result['target'] = 'cross-log name generalisation'
    result['closed_set_within_document'] = {}
    result['generation_only_within_document'] = {}
    result['recognition_controls'] = {}
    for algorithm in result['fake_to_real']:
        result['closed_set_within_document'][algorithm] = {
            variant:within_document(features,samples,truth,algorithm,variant)
            for variant in ('clean','augmented')}
        for run in result['closed_set_within_document'][algorithm].values():
            prediction_uncertainty(run,truth)
        result['generation_only_within_document'][algorithm] = {
            v:prediction_uncertainty(within_document(features,[r for r in samples if r.get('format')!='finite-legend'],truth,algorithm,v),truth)
            for v in ('clean','augmented')}
        experiments = list(result['clean_vs_augmented'][algorithm].values()) + list(result['ablation'][algorithm].values())
        for experiment in experiments:
            if isinstance(experiment,dict):
                # Uniform choice from the training name vocabulary. Unknown
                # names cannot be guessed, so their individual chance is zero.
                count = experiment['training_names']
                experiment['chance'] = round(100*(1-experiment['unseen_real_boxes']/max(1,experiment['n']))/count,3) if count else 0.
                experiment['candidate_count'] = count
                experiment['task'] = 'cross-log name generalisation'
        result['fake_to_real'][algorithm] = result['clean_vs_augmented'][algorithm]['augmented']
    class HistogramFeatures(NativeFeatures):
        def get(self,row):
            a,b = super().get(row)
            return a[:64],b[:64]
    result['recognition_controls']['histogram_only'] = prediction_uncertainty(
        within_document(HistogramFeatures(folder),samples,truth,'matcher','augmented'),truth)
    class RailMaskedFeatures(NativeFeatures):
        def get(self,row):
            with Image.open(folder/row['file']) as image:
                rgb = np.asarray(image.convert('RGB'))
            rgb = rgb[:,3:-3] if rgb.shape[1]>6 else rgb
            f = score.features(rgb)
            return f,f
    result['recognition_controls']['rail_masked'] = prediction_uncertainty(
        within_document(RailMaskedFeatures(folder),samples,truth,'matcher','augmented'),truth)
    result['legacy_seconds'] = result['seconds']
    result['seconds'] = round(result['seconds'] + time.monotonic()-started,2)
    result['validation_tile_policy'] = 'legend-swatch-only; zero validation column/residual inputs'
    result['limits'] = [s for s in result['limits'] if 'validation source crops' not in s and 'Existing tile selection' not in s]
    result['limits'] += ['Within-document candidates use only document legend swatches. Held-well real crops never shape those tiles or their augmentation. Closed-set macro includes every observed document class; cross-log macro retains the legacy n>=3 name rule.',
        'Recognition may use an original finite native legend reference for seam-only synthesis exclusions. These references never enter generation, augmentation, or fake-only evaluation. Generation-only recognition and coverage are reported separately; all legend IDs remain in the chance vocabulary.',
        'Finite support was selected after inspecting these validation wells; the comparison is exploratory. Recognition accuracy cannot certify motif geometry, repaired repeats, or detector benefit. Histogram-only and rail-masked controls expose sensitivity.']
    lines = [f"# {folder.name}: legend-conditioned recognition with finite support",'',
        'Finite native legend references support recognition only. Long repeats remain withheld; no finite reference enters the synthetic training manifest.', '',
        '| Model / variant | Support n | Real n | Correct | Top-1 % | Macro % (all document classes) | Chance % | Macro chance % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for alg, variants in result['closed_set_within_document'].items():
        for variant,r in variants.items():
            lines.append(f"| {alg} / {variant} | {r['train_n']} | {r['n']} | {r['correct']} | {r['accuracy']} | {r['macro_accuracy']} | {r['chance']} | {r['macro_chance']} |")
    lines += ['', '| Recognition input / control | Correct / boxes | Well mean % | 95% well CI |',
        '|---|---:|---:|---|']
    controls = dict(generation_only=result['generation_only_within_document']['matcher']['augmented'],
        **result['recognition_controls'])
    for name,r in controls.items():
        lines.append(f"| {name} | {r['correct']} / {r['n']} | {r['well_macro']['estimate']} | {r['well_macro']['ci95']} |")
    coverage = result['closed_set_within_document']['matcher']['augmented']['support_coverage']
    lines += ['', 'Support coverage: '+json.dumps(coverage),
        'Generation coverage and its ceiling remain unchanged by finite recognition support.']
    lines += ['', 'Chance is the average of 1 / candidate count for each box. Macro chance weights observed document classes equally.',
        '', '| Document | Candidates | Real n | Clean matcher top-1 % | Macro % | Chance % |', '|---|---:|---:|---:|---:|---:|']
    for r in result['closed_set_within_document']['matcher']['clean']['documents']:
        lines.append(f"| {r['doc']} | {r['candidates']} | {r['n']} | {r['accuracy']} | {r['macro_accuracy']} | {r['chance']} |")
    lines += ['', '## Cross-log name generalisation', '',
        '| Model / variant | Candidates | Chance % (unknown names = 0) |', '|---|---:|---:|']
    for alg in result['fake_to_real']:
        for variant in ('clean','augmented'):
            r=result['clean_vs_augmented'][alg][variant]
            lines.append(f"| {alg} / {variant} | {r['candidate_count']} | {r['chance']} |")
    result = finalize_evaluation(folder,result)
    _write(folder/'eval.json',result)
    (folder/'eval.md').write_text('\n'.join(lines)+'\n\n'+markdown(result),encoding='utf-8')
    print('PASS closed-set',folder.name,json.dumps({alg:{v:(r['correct'],r['n'],r['accuracy'],r['macro_accuracy'],r['chance']) for v,r in variants.items()} for alg,variants in result['closed_set_within_document'].items()}),flush=True)
    return result


def main():
    from .dataset_eval import configure_loop, revoke_unsafe_versions
    configure_loop()
    revoke_unsafe_versions(ROOT)
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    bp = sub.add_parser('build')
    bp.add_argument('--baseline', help='Freeze wells, accepted boxes and training tiles from this version')
    ev = sub.add_parser('eval')
    ev.add_argument('version')
    lp = sub.add_parser('loop')
    lp.add_argument('--once', action='store_true')
    args = ap.parse_args()
    if args.command == 'build':
        from .dataset_loop import writer_lock
        with writer_lock(ROOT):
            DatasetBuilder(baseline=ROOT/args.baseline if args.baseline else None).build()
    elif args.command == 'eval':
        from .dataset_eval import evaluate
        from .dataset_loop import publish, writer_lock
        folder = ROOT / args.version
        if not folder.resolve().is_relative_to(ROOT.resolve()) or not folder.is_dir():
            raise SystemExit('Unknown version')
        with writer_lock(ROOT):
            evaluate(folder)
            publish(ROOT, folder)
    else:
        from .dataset_loop import loop
        loop(ROOT, once=args.once)


if __name__ == '__main__':
    main()
