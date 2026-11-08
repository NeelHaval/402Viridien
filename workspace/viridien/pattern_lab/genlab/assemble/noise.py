"""Measured scan appearance and real residual library; training wells only.

python -X utf8 -m genlab.assemble.noise
No dataset-builder dependency. TIFF decoding is restricted to source strips.
"""
from __future__ import annotations

from genlab.assemble import training_v2
import argparse
import ctypes
import ctypes.util
import hashlib
import io
import json
import struct
import sys
import time
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

import model
from genlab import paths, render

ROOT = paths.OUT / 'assemble'
NOISE = ROOT / 'noise'


def read(path):
    from genlab.assemble.training_v2 import read_json
    return read_json(path)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def training(doc, policy=None):
    from .stats import training_document
    return training_document(doc,policy or read(paths.DATA / 'normalized/split-policy.json'))


def require_training(doc):
    if not training(doc):
        raise ValueError('Test-well or validation-well input forbidden: ' + doc['id'])


def crop_allowed(did, box, records=None):
    """Never reuse explicitly quarantined normalized pixels, including duplicates."""
    a,b,c,d=box
    training_v2.require_training_id(did)
    for denied in training_v2.load_exclusions()['denied_boxes']:
        if denied['document_id']!=did:continue
        x,y,z,w=denied['box_xyxy']
        if min(c,z)>max(a,x) and min(d,w)>max(b,y):return False
    records=read(paths.DATA/'genlab/fasttrack/training-v2.json')['records'] if records is None else records
    for r in records:
        if r['document_id']!=did or (r.get('usable_for_generation') and not r.get('quality_issues')):
            continue
        x,y,z,w=r['box_xyxy']
        if min(c,z)>max(a,x) and min(d,w)>max(b,y):
            return False
    return True


def ancestors(card,cards,seen=None):
    seen=set() if seen is None else seen
    key=card['key']
    if key in seen:
        raise ValueError('Cyclic pattern ancestry: '+key)
    seen=seen|{key}
    if card.get('source') in ('agent-made','unlisted') and not (
        card.get('lineage') or card.get('recovery',{}).get('lineage')):
        raise ValueError('Unknown proposal ancestry: '+key)
    if card.get('tile_is_preview_only'):
        raise ValueError('Preview-only pattern is not a rendering model: '+key)
    result={card['doc_id'],card.get('real_doc_id',card['doc_id'])}
    result.update(card.get('ancestor_docs',[]))
    def lineage(node, stack):
        if not isinstance(node,dict):raise ValueError('Invalid pattern lineage: '+key)
        ident=id(node)
        if ident in stack:raise ValueError('Cyclic source lineage: '+key)
        stack=stack|{ident}
        for source in node.get('lineage',[])+node.get('recovery',{}).get('lineage',[]):
            did=source.get('doc_id') or source.get('document_id')
            if not did:raise ValueError('Unknown source lineage: '+key)
            result.add(did);result.update(source.get('ancestor_docs',[]))
            lineage(source,stack)
        nested=node.get('donor')
        if nested:
            did=nested.get('doc_id') or nested.get('document_id')
            if not did:raise ValueError('Unknown donor ancestry: '+key)
            result.add(did);lineage(nested,stack)
    lineage(card,set())
    donor=card.get('donor')
    if donor:
        if not isinstance(donor,dict) or not donor.get('doc_id'):
            raise ValueError('Unknown donor ancestry: '+key)
        result.add(donor['doc_id'])
        if donor.get('legend_id') is not None:
            parent=f"{donor['doc_id']}:{donor['legend_id']}"
            if parent not in cards:
                raise ValueError('Unresolved donor ancestry: '+parent)
            result |= ancestors(cards[parent],cards,seen)
    return result


def cache_provenance():
    files=[paths.DATA/'genlab/fasttrack/training-v2.json',paths.DATA/'normalized/split-policy.json',Path(__file__)]
    files.append(Path(__file__).with_name('stats.py'))
    calibration=ROOT/'stats/depth-calibration.json'
    if calibration.exists():files.append(calibration)
    hashes={str(p.relative_to(paths.LAB)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if paths.PREFILL.exists():hashes['prefill']=hashlib.sha256(paths.PREFILL.read_bytes()).hexdigest()
    metadata=ROOT/'stats/mix-metadata.json'
    if metadata.exists():
        semantic=read(metadata);semantic.pop('fetched_at',None)
        hashes['family-metadata']=hashlib.sha256(json.dumps(semantic,sort_keys=True).encode()).hexdigest()
    return dict(normalized_snapshot=read(files[0])['snapshot'],hashes=hashes,
                signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())


def source_binding(did, box, class_id, records=None):
    """Join source evidence to current targets; machine-only is explicitly marked."""
    records=read(paths.DATA/'genlab/fasttrack/training-v2.json')['records'] if records is None else records
    a,b,c,d=box; bindings=[]
    lid=class_id.split(':',1)[1]
    for row in records:
        if row['document_id']!=did:continue
        x,y,z,w=row['box_xyxy']
        if min(c,z)<=max(a,x) or min(d,w)<=max(b,y):continue
        if (not row.get('accepted') or not row.get('usable_for_generation') or row.get('quality_issues') or
            len(row.get('targets',[]))!=1 or str(row['targets'][0]['legend_id'])!=lid):
            raise ValueError('Revoked or relabelled appearance source: '+class_id)
        bindings.append({k:row.get(k) for k in ('sample_id','annotation_version','legend_version',
            'document_id','source_revision','pixel_sha256','targets','box_xyxy')})
    return dict(kind='normalized-target' if bindings else 'machine-only-unconfirmed',normalized_sources=bindings)


def asset_path(root, record, field):
    file=Path(root)/record[field]
    expected=record.get('asset_sha256',{}).get(field)
    if not expected or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
        raise ValueError('Missing or changed appearance asset: '+str(file))
    return file


def evidence_sources(did, boxes, doc=None, records=None):
    """Bind every nuisance crop to its current eligibility and label revisions."""
    records=read(paths.DATA/'genlab/fasttrack/training-v2.json')['records'] if records is None else records
    sources=[]
    for box in boxes:
        if not crop_allowed(did,box,records):raise ValueError('Revoked nuisance source crop')
        a,b,c,d=box
        rows=[r for r in records if r['document_id']==did and
              min(c,r['box_xyxy'][2])>max(a,r['box_xyxy'][0]) and
              min(d,r['box_xyxy'][3])>max(b,r['box_xyxy'][1])]
        sources.append(dict(box=list(box),binding=hashlib.sha256(
            json.dumps(rows,sort_keys=True).encode()).hexdigest()))
    training_v2.require_training_id(did)
    doc=doc or model.Store().get(did)
    return dict(source_doc=did,sources=sources,
        source_document_sha256=hashlib.sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest())


def nuisance_allowed(record, docs=None, records=None):
    """Legacy unbound rails/contacts cannot be certified or consumed."""
    did=record.get('source_doc') or record.get('doc')
    if not did or not record.get('sources'):return False
    try:
        training_v2.require_training_id(did)
        doc=docs[did] if docs is not None else model.Store().get(did)
        if not training(doc):return False
        current=evidence_sources(did,[s['box'] for s in record['sources']],doc,records)
        return all(record.get(k)==current[k] for k in ('sources','source_document_sha256'))
    except (ValueError,KeyError,FileNotFoundError):return False


def immutable_image(file, array):
    if file.exists():
        with Image.open(file) as image:stored=np.asarray(image)
        if not np.array_equal(stored,array):
            raise ValueError('Existing immutable appearance content differs: '+str(file))
    else:Image.fromarray(array).save(file)


def save_measurement(output, card, did, box, family, ancestry, rgb, tile):
    binding=source_binding(did,box,card['key'])
    from genlab.assemble import learn_v2 as learn
    if not training_v2.pixels_allowed(rgb):raise ValueError('Quarantined source pixels')
    h,w=rgb.shape[:2];native_digest=hashlib.sha256(f'{w}x{h}:'.encode()+rgb.tobytes()).hexdigest()
    for source in binding['normalized_sources']:
        if source['box_xyxy']==box and source['pixel_sha256']!=native_digest:
            raise ValueError('Source pixels no longer match normalized sample')
    params,mask,colours,residual=measure(tile,rgb)
    valid=(mask==0)&flat_pixels(rgb)&(np.linalg.norm(residual,axis=2)<max(6.,6*params['noise_sigma']))
    field=np.clip(residual,-32,32).astype(np.int8)
    dependencies=dict(class_id=card['key'],box=box,tile_signature=card.get('signature'),source_doc=did,
        source_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),binding=binding,
        fitter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),params=params)
    ident=hashlib.sha256(json.dumps(dependencies,sort_keys=True).encode()+mask.tobytes()+colours.tobytes()+
        field.tobytes()+valid.tobytes()).hexdigest()[:24]
    names=dict(mask=f'{ident}-mask.png',colours=f'{ident}-rgb.png',residual=f'{ident}-residual.npz')
    # An identity includes outputs and fitter version. Old artifact bytes are immutable.
    # Validate *all* existing files before writing any new member of this identity.
    for key,array in [('mask',mask),('colours',colours)]:
        if (output/names[key]).exists():immutable_image(output/names[key],array)
    residual_file=output/names['residual']
    if residual_file.exists():
        try:
            with np.load(residual_file,allow_pickle=False) as saved:
                stored_field=saved['residual'];stored_valid=saved['valid']
                matches=(stored_field.dtype==field.dtype and stored_valid.dtype==valid.dtype and
                         np.array_equal(stored_field,field) and np.array_equal(stored_valid,valid))
        except Exception as error:
            raise ValueError('Existing immutable residual is unreadable') from error
        if not matches:raise ValueError('Existing immutable residual content differs: '+ident)
    immutable_image(output/names['mask'],mask)
    immutable_image(output/names['colours'],colours)
    if not residual_file.exists():np.savez_compressed(residual_file,residual=field,valid=valid)
    hashes={k:hashlib.sha256((output/v).read_bytes()).hexdigest() for k,v in names.items()}
    return dict(params,**names,id=ident,class_id=card['key'],doc_id=card['doc_id'],source_doc=did,
        family=family,box=box,ancestor_docs=sorted(ancestry),tile_signature=card.get('signature'),
        binding=binding,dependencies=dependencies,asset_sha256=hashes,overlay_usable=False,
        overlay_policy='disabled: residual motif fragments are not validated nuisance ink')


def families():
    """Use the category tool's family labels saved in the prefill, not heuristics."""
    result = {}
    with paths.PREFILL.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('family'):
                result[row['log']] = row['family']
    mix_path = NOISE / 'mix.json'
    if mix_path.exists():
        mix = read(mix_path)
        for pair in mix['pairs']:
            for name in pair.get('logs', [pair['log']]):
                result[name] = pair['family']
        import csv
        source = paths.LAB.parent / 'analysis/label_shares_per_log.csv'
        if source.exists():
            with source.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.reader(stream):
                    if len(row)>1 and row[1] in mix['families']:
                        result.setdefault(row[0], row[1])
    # Published asm-1 family membership is authoritative and metadata-only.
    stats_path = ROOT / 'stats.json'
    if stats_path.exists():
        store = model.Store()
        names = {d['id']:d['name'] for d in store.catalog}
        for key, block in read(stats_path)['levels'].items():
            if key.startswith('family:'):
                for did in block.get('document_ids', []):
                    result[names[did]] = key[7:]
    return result


class Windows:
    """One open TIFF; decode intersecting strips, including CCITT via mini-TIFFs."""
    def __init__(self):
        self.file = None
        self.ident = None
        self.scan = None
        self.scan_next = 0
        self.lib = None

    def close(self):
        if self.scan:
            self.lib.TIFFClose(self.scan)
        self.scan = None
        if self.file is not None:
            self.file.close()
        self.file = None
        self.ident = None

    def _scanlines(self, doc, box, page):
        """LZW scans often encode 50k+ rows in ONE strip. Decode incrementally."""
        import os
        if self.lib is None:
            if sys.platform == 'win32':
                dlls = Path(sys.prefix) / 'Library/bin'
                self._dll_directory = os.add_dll_directory(str(dlls))
                self.lib = ctypes.CDLL(str(dlls / 'tiff.dll'))
            else:
                self.lib = ctypes.CDLL(ctypes.util.find_library('tiff'))
            self.lib.TIFFOpen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            self.lib.TIFFOpen.restype = ctypes.c_void_p
            self.lib.TIFFReadScanline.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16]
            self.lib.TIFFReadScanline.restype = ctypes.c_int
            self.lib.TIFFClose.argtypes = [ctypes.c_void_p]
            self.lib.TIFFSetWarningHandler.argtypes = [ctypes.c_void_p]
            self.lib.TIFFSetWarningHandler(None)
            self.lib.TIFFScanlineSize.argtypes = [ctypes.c_void_p]
            self.lib.TIFFScanlineSize.restype = ctypes.c_ssize_t
        x0,y0,x1,y1 = box
        if self.scan and y0 < self.scan_next:
            self.lib.TIFFClose(self.scan)
            self.scan = None
        if not self.scan:
            path = model.DATASET / 'images' / doc['image_name']
            self.scan = self.lib.TIFFOpen(str(path).encode('utf-8'), b'r')
            if not self.scan:
                raise OSError('libtiff could not open '+str(path))
            self.scan_next = 0
        size = self.lib.TIFFScanlineSize(self.scan)
        buffer = np.empty(size, np.uint8)
        out = np.empty((y1-y0,x1-x0,3),np.uint8)
        for y in range(self.scan_next,y1):
            if self.lib.TIFFReadScanline(self.scan,buffer.ctypes.data,y,0) < 0:
                raise OSError('TIFF scanline decode failed')
            if y >= y0:
                if page.photometric == 3 and page.bitspersample == 8:
                    out[y-y0] = (page.colormap >> 8).astype(np.uint8).T[buffer[x0:x1]]
                elif page.samplesperpixel >= 3 and page.bitspersample == 8:
                    out[y-y0] = buffer[:page.imagewidth*page.samplesperpixel].reshape(-1,page.samplesperpixel)[x0:x1,:3]
                else:
                    raise ValueError('Unsupported giant-strip pixel format')
        self.scan_next = y1
        return out

    @staticmethod
    def _ccitt(page, encoded, rows):
        # A valid TIFF containing just this independently compressed source strip.
        tags = [(256, 4, 1, page.imagewidth), (257, 4, 1, rows),
                (258, 3, 1, 1), (259, 3, 1, int(page.compression)),
                (262, 3, 1, int(page.photometric)), (266, 3, 1, int(page.fillorder)),
                (273, 4, 1, 0), (277, 3, 1, 1), (278, 4, 1, rows),
                (279, 4, 1, len(encoded))]
        for tag in (292, 293):
            if tag in page.tags:
                tags.append((tag, 4, 1, int(page.tags[tag].value)))
        offset = 8 + 2 + len(tags) * 12 + 4
        tags = [(k, t, c, offset if k == 273 else v) for k, t, c, v in sorted(tags)]
        header = b'II' + struct.pack('<HIH', 42, 8, len(tags))
        header += b''.join(struct.pack('<HHII', *tag) for tag in tags) + b'\0' * 4
        with Image.open(io.BytesIO(header + encoded)) as image:
            return np.asarray(image.convert('RGB')).copy()

    def get(self, doc, box):
        require_training(doc)
        if not crop_allowed(doc['id'],box):
            raise ValueError('Quarantined normalized pixels in source window: '+doc['id'])
        import tifffile
        if self.ident != doc['id']:
            self.close()
            self.file = tifffile.TiffFile(model.DATASET / 'images' / doc['image_name'])
            self.ident = doc['id']
        p = self.file.pages[0]
        if p.is_tiled or p.planarconfig != 1:
            raise ValueError('Unsupported source layout; refusing whole-page decode')
        x0, y0, x1, y1 = map(int, box)
        model.Store.validate_box(doc, [x0, y0, x1, y1])
        if p.rowsperstrip > 4000:
            return self._scanlines(doc, [x0,y0,x1,y1], p)
        out = np.empty((y1-y0, x1-x0, 3), np.uint8)
        for i in range(y0 // p.rowsperstrip, (y1-1) // p.rowsperstrip + 1):
            self.file.filehandle.seek(p.dataoffsets[i])
            encoded = self.file.filehandle.read(p.databytecounts[i])
            start = i * p.rowsperstrip
            if int(p.compression) in (2, 3, 4):
                rgb = self._ccitt(p, encoded, min(p.rowsperstrip, p.imagelength-start))
            else:
                segment, _, _ = p.decode(encoded, i)
                rgb = segment[0]
                if p.photometric == 3:
                    rgb = (p.colormap >> 8).astype(np.uint8).T[rgb[..., 0]]
                elif rgb.shape[-1] == 1:
                    rgb = rgb.astype(np.uint8) * (255 if rgb.dtype == bool else 1)
                    if p.photometric == 0:
                        rgb = 255-rgb
                    rgb = np.repeat(rgb, 3, axis=2)
            a, b = max(y0, start), min(y1, start+len(rgb))
            out[a-y0:b-y0] = rgb[a-start:b-start, x0:x1, :3]
        return out


def aligned(tile, real):
    """Exact cyclic least squares with a folded FFT, independently implemented."""
    ph, pw = tile.shape[:2]
    yy, xx = np.indices(real.shape[:2])
    pos = ((yy % ph) * pw + xx % pw).ravel()
    counts = np.bincount(pos, minlength=ph*pw).reshape(ph, pw)
    loss = np.zeros((ph, pw))
    for c in range(3):
        t = tile[..., c].astype(float)
        sums = np.bincount(pos, weights=real[..., c].ravel(), minlength=ph*pw).reshape(ph, pw)
        loss += np.fft.irfft2(np.conj(np.fft.rfft2(counts))*np.fft.rfft2(t*t)
                             - 2*np.conj(np.fft.rfft2(sums))*np.fft.rfft2(t), s=(ph, pw))
    dy, dx = np.unravel_index(loss.argmin(), loss.shape)
    return tile[(yy+dy) % ph, (xx+dx) % pw].copy(), [int(dy), int(dx)]


def clean_overlay(mask, colours):
    """Frame fragments stay in the rail library, never inside pasted overlays."""
    mask = np.asarray(mask, bool).copy()
    rails = ((colours.max(2)<80).mean(0)>.85) & (mask.mean(0)>.65)
    mask[:,rails] = False
    return mask, np.where(rails)[0].tolist()


def flat_pixels(colours):
    hi = ndimage.maximum_filter(colours,size=(3,3,1)).astype(float)
    lo = ndimage.minimum_filter(colours,size=(3,3,1)).astype(float)
    return (hi-lo).max(2)<12


def block_energy(residual, valid):
    """8-pixel phase diagnostic on flat, matched pixels; never motif edges."""
    gray = residual.astype(float).mean(2)
    result = []
    for axis in (0,1):
        delta = np.abs(np.diff(gray,axis=axis))
        a,b = [slice(None)]*2,[slice(None)]*2
        a[axis],b[axis] = slice(None,-1),slice(1,None)
        take = valid[tuple(a)]&valid[tuple(b)]
        phases = []
        for k in range(8):
            sl = [slice(None)]*2
            sl[axis] = slice(k,None,8)
            values,keep = delta[tuple(sl)],take[tuple(sl)]
            phases.append(float(values[keep].mean()) if keep.any() else 0.)
        result.append(phases)
    return result


def measure(tile, real):
    fake, phase = aligned(tile, real)
    actual = real.astype(float)
    best = None
    for blur in (0., .25, .5, .75, 1., 1.5, 2.):
        base = np.asarray(Image.fromarray(fake).filter(ImageFilter.GaussianBlur(blur)), float)
        drift = np.median(actual-base, axis=(0, 1))
        residual = actual-base-drift
        err = np.linalg.norm(residual, axis=2)
        score = np.mean(np.minimum(err, 45))
        if best is None or score < best[0]:
            best = score, blur, drift, residual, err
    _, blur, drift, residual, err = best
    med = np.median(err)
    threshold = max(45., med+4.4478*np.median(abs(err-med)))
    mask = err > threshold
    # Only unexplained darker ink is reusable: colour disagreements are not text.
    ink = mask & (actual.mean(2) < (actual-residual).mean(2)-25)
    ink[:, :2] = False
    ink[:, -2:] = False
    ink, residual_rails = clean_overlay(ink, real)
    good = residual[~mask]
    sigma = 1.4826*np.median(abs(good-np.median(good, axis=0))) if len(good) else 0.
    labels, n = ndimage.label(ink)
    components = []
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        sy, sx = sl
        h, w = sy.stop-sy.start, sx.stop-sx.start
        area = int((labels[sl] == i).sum())
        kind = ('speck' if area <= 3 else 'depth-tick' if w > 3*h else
                'curve' if h > 4*w else 'text-or-symbol')
        components.append(dict(box=[sx.start, sy.start, sx.stop, sy.stop], area=area, kind=kind))
    scan_valid = (~mask)&flat_pixels(real)&(np.linalg.norm(residual,axis=2)<max(6.,6*sigma))
    block = block_energy(residual,scan_valid)
    params = dict(blur=blur, noise_sigma=float(sigma), drift=drift.tolist(), phase=phase,
                  residual_fraction=float(mask.mean()), overlay_fraction=float(ink.mean()),
                  overlay_usable=bool(mask.mean() < .25), threshold=float(threshold),
                  jpeg_block_phase_energy=block, components=components,
                  fit_error=float(best[0]), shape=list(real.shape[:2]))
    params['residual_rail_columns'] = residual_rails
    return params, ink.astype(np.uint8)*255, real.copy(), residual


def load_model(card,root=None):
    from genlab.assemble import learn_v2 as learn
    if not card.get('render_model'):
        raise ValueError('Missing complete genlab-render-2 model export: '+card.get('key','fixture'))
    return learn.load_model(card['render_model'], root or paths.OUT, card.get('signature'))


def fit(output=NOISE):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_provenance=cache_provenance()
    store, windows = model.Store(), Windows()
    fmap = families()
    cards = sorted(read(paths.STATE)['cards'], key=lambda c:(c.get('real_doc_id',c['doc_id']),
                   min([b[1] for b in c.get('real',[])] or [0]),c['key']))
    cardmap={c['key']:c for c in cards}
    records = read(paths.DATA / 'genlab/fasttrack/training-v2.json')['records']
    docs = {d['id']: store.get(d['id']) for d in store.catalog if training_v2.training_id(d['id'])}
    measurements, rails, boundaries = [], {}, defaultdict(list)
    comparison = []
    for did, doc in docs.items():
        if not training(doc):
            continue
        family = fmap.get(doc['name'], 'One-off layout')
        x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
        boxes=[[x0,int(y),x1,min(y1,int(y)+160)] for y in np.linspace(y0,max(y0,y1-160),7).astype(int)]
        boxes=[box for box in boxes if crop_allowed(did,box)]
        probes = [windows.get(doc,box) for box in boxes]
        if not probes:continue
        dark = np.concatenate([(p.max(2) < 80) for p in probes])
        edge = np.zeros(x1-x0, bool)
        edge[:min(5, len(edge)//4)] = True
        edge[-min(5, len(edge)//4):] = True
        keep = (dark.mean(0) > .85) & edge
        sample = np.concatenate(probes)
        railpixels=sample[:, keep] if keep.any() else sample[:, :1]
        railfile = f"rail-{did}-{hashlib.sha256(railpixels.tobytes()).hexdigest()[:20]}.png"
        immutable_image(output/railfile,railpixels)
        rails[did] = dict(columns=np.where(keep)[0].tolist(), file=railfile,
                          asset_sha256={'file':hashlib.sha256((output/railfile).read_bytes()).hexdigest()},
                          family=family, probe_rows=int(len(sample)),**evidence_sources(did,boxes,doc))
        rr = sorted([r for r in records if r['document_id'] == did and
                     r.get('usable_for_generation') and len(r['targets']) == 1],
                    key=lambda r: r['box_xyxy'][1])
        for a, b in zip(rr, rr[1:]):
            from .stats import infer_box_kind
            if (a['targets']==b['targets'] or infer_box_kind(a,rr)!='bed' or infer_box_kind(b,rr)!='bed'):
                continue
            if abs(b['box_xyxy'][1]-a['box_xyxy'][3]) > 2:
                continue
            y = b['box_xyxy'][1]
            if y < y0+4 or y > y1-5:
                continue
            if not crop_allowed(did,[x0+2,y-4,x1-2,y+5]):continue
            p = windows.get(doc, [x0+2, y-4, x1-2, y+5])
            frac = (p.max(2) < 80).mean(1)
            line = bool(frac[3:6].max() > .75 and frac[3:6].max() > np.median(frac[[0, 1, 7, 8]])+.25)
            # Fit intermediate edge colours along the RGB segment joining flanks;
            # unrelated motif ink is off this segment and does not count as AA.
            top, bottom = np.median(p[:2], (0, 1)), np.median(p[-2:], (0, 1))
            mid = p[3:6].astype(float)
            delta = bottom-top
            alpha = ((mid-top)*delta).sum(2)/max(float((delta*delta).sum()),1.)
            distance = np.linalg.norm(mid-(top+alpha[...,None]*delta),axis=2)
            intermediate = (alpha>.1) & (alpha<.9) & (distance<20) & (np.linalg.norm(delta)>35)
            row = p[3+int(frac[3:6].argmax())]
            black = row[row.max(1)<80]
            boundaries[family].append(dict(line=line, antialias=float(intermediate.mean()),
                antialias_alpha=float(np.median(alpha[intermediate])) if intermediate.any() else 0.,
                line_width=int((frac[2:7] > .75).sum()) if line else 0,
                ink=np.median(black,axis=0).astype(int).tolist() if len(black) else [0,0,0], doc=did,page_y=y,
                **evidence_sources(did,[[x0+2,y-4,x1-2,y+5]],doc)))
        print('FIT document', did, 'boundaries', len(boundaries[family]), flush=True)
    for i, card in enumerate(cards):
        doc = docs[card['doc_id']]
        realdoc = docs.get(card.get('real_doc_id', doc['id']), doc)
        if not training(doc) or not training(realdoc):
            continue
        try:
            ancestry=ancestors(card,cardmap)
            from genlab.assemble import learn_v2 as learn
            if not learn.lineage_allowed(card,docs):continue
        except ValueError:
            continue
        if any(d not in docs or not training(docs[d]) for d in ancestry):
            continue
        family = fmap.get(doc['name'], 'One-off layout')
        try:
            data = load_model(card)
        except (FileNotFoundError,ValueError):
            continue  # active lab rebuild may prune a superseded card
        tile = data['tile'] if data['kind'] == 'tile' else data['strips'][0]
        col = realdoc['boxes']['interpreted_lithology_column']
        for j, box in enumerate(card.get('real', [])):
            if not crop_allowed(realdoc['id'],box):
                continue
            # Ignore swatch-only cards masquerading as a real evidence box.
            a, b, c, d = map(int, box)
            a, b, c, d = max(a, col[0]), max(b, col[1]), min(c, col[2]), min(d, col[3])
            if c-a < 6 or d-b < 4:
                continue
            box = [a, b, c, min(d, b+480)]
            cached=paths.CROPS/realdoc['id']/('%d_%d_%d_%d.png'%tuple(box))
            if cached.exists():
                with Image.open(cached) as im:rgb=np.asarray(im.convert('RGB'))
            else:rgb = windows.get(realdoc, box)
            try:
                fitted=save_measurement(output,card,realdoc['id'],box,family,ancestry,rgb,tile)
            except ValueError:
                continue
            measurements.append(fitted);params=fitted;ident=fitted['id']
            if len(comparison) < 12:
                from genlab import realism
                other, _, _ = realism.measure(tile, rgb)
                comparison.append(dict(id=ident, own_blur=params['blur'], dataset_blur=other['blur'],
                                       own_noise=params['noise_sigma'], dataset_noise=other['noise']))
        if i % 50 == 0:
            print('FIT', i, '/', len(cards), 'crops', len(measurements), flush=True)
    windows.close()
    summary = {}
    for family in sorted(set(fmap.values())):
        mm = [m for m in measurements if m['family'] == family]
        bb = boundaries[family]
        summary[family] = dict(crops=len(mm), blur_median=float(np.median([m['blur'] for m in mm])) if mm else None,
            noise_sigma_median=float(np.median([m['noise_sigma'] for m in mm])) if mm else None,
            drift_median=np.median([m['drift'] for m in mm], axis=0).tolist() if mm else None,
            overlay_fraction=float(np.mean([m['overlay_fraction'] for m in mm])) if mm else None,
            usable_overlays=sum(m['overlay_usable'] for m in mm), boundaries=len(bb),
            boundary_line_rate=float(np.mean([b['line'] for b in bb])) if bb else 0.,
            boundary_widths=[b['line_width'] for b in bb if b['line']],
            boundary_antialias=float(np.median([b['antialias'] for b in bb])) if bb else None)
    if cache_provenance()!=source_provenance:raise ValueError('Sources changed during appearance fit; retry from stable inputs')
    result = dict(schema='asm-noise-2', provenance=source_provenance,built_at=time.time(), measurements=measurements,
                  rails=rails, boundaries=dict(boundaries), families=summary,
                  dataset_comparison=comparison,
                  limits='Residuals can include model mismatch; dark unexplained ink only is pasted. '
                  'Blur/noise are incremental to native saved scan tiles. Boundary detection uses accepted touching boxes.')
    publish_library(output,result)
    print('PASS noise', len(measurements), 'crops', json.dumps(summary), flush=True)
    fit_boundary_candidates(output)
    fit_gap_sources(output)
    if cache_provenance()!=source_provenance:raise ValueError('Sources changed during nuisance fit; library is stale')
    return read(output/'measurements.json')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=NOISE)
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--boundaries', action='store_true')
    parser.add_argument('--gaps', action='store_true')
    parser.add_argument('--refresh',action='store_true',help='Refit current tiles against cached bounded native crops')
    args = parser.parse_args()
    if args.refresh:
        refresh_library(args.output)
        return
    if args.refine:
        refine_library(args.output)
        return
    if args.boundaries:
        fit_boundary_candidates(args.output)
        return
    if args.gaps:
        fit_gap_sources(args.output)
        return
    fit(args.output)


def refine_library(output=NOISE):
    # Refinement is a new fit, never an in-place edit of previously identified assets.
    return refresh_library(output)


def publish_library(output,result):
    output=Path(output)
    result['model_signatures']={m['class_id']:m['tile_signature'] for m in result['measurements']}
    for family,summary in result.get('families',{}).items():
        rows=[m for m in result['measurements'] if m['family']==family]
        summary.update(crops=len(rows),usable_overlays=sum(m.get('overlay_usable',False) for m in rows),
            blur_median=float(np.median([m['blur'] for m in rows])) if rows else None,
            noise_sigma_median=float(np.median([m['noise_sigma'] for m in rows])) if rows else None,
            drift_median=np.median([m['drift'] for m in rows],axis=0).tolist() if rows else None,
            overlay_fraction=float(np.mean([m['overlay_fraction'] for m in rows])) if rows else None)
    for rail in result.get('rails',{}).values():
        asset_path(output,rail,'file')  # Never restamp changed bytes as trusted evidence.
    target=output/'measurements.json';temporary=output/'measurements.json.tmp'
    write(temporary,result);temporary.replace(target)


def fit_boundary_candidates(output=NOISE):
    """Supplement sparse touching human boxes with directly observed colour edges.

    These are appearance observations, never class/transition truth. A persistent
    change of flank background is required, so grid lines inside a fill are excluded.
    """
    output = Path(output)
    result = read(output/'measurements.json')
    store, windows = model.Store(), Windows()
    fmap = families()
    count = 0
    for entry in store.catalog:
        if not training_v2.training_id(entry['id']):continue
        doc = store.get(entry['id'])
        if not training(doc):
            continue
        family = fmap[doc['name']]
        if any(b.get('doc')==doc['id'] for b in result['boundaries'].get(family,[])):
            continue
        x0,y0,x1,y1 = doc['boxes']['interpreted_lithology_column']
        mm = [m for m in result['measurements'] if m['source_doc']==doc['id']]
        centers = sorted({int((m['box'][1]+m['box'][3])/2) for m in mm})
        if len(centers)>20:
            centers = [centers[i] for i in np.linspace(0,len(centers)-1,20).astype(int)]
        observations = []
        for center in centers:
            top,bottom = max(y0,center-400),min(y1,center+400)
            if not crop_allowed(doc['id'],[x0,top,x1,bottom]):continue
            rgb = windows.get(doc,[x0,top,x1,bottom])
            profile=[]
            for row in rgb[:,3:-3]:
                bright=row[row.max(1)>=100]
                profile.append(np.median(bright,axis=0) if len(bright) else [np.nan]*3)
            profile=np.asarray(profile)
            known=np.flatnonzero(np.isfinite(profile).all(1))
            if not len(known):continue
            for channel in range(3):
                profile[:,channel]=np.interp(np.arange(len(profile)),known,profile[known,channel])
            profile = ndimage.median_filter(profile,size=(15,1))
            distance = np.linalg.norm(profile[12:]-profile[:-12],axis=1)
            good = distance>40
            labels,n = ndimage.label(good)
            for i in range(1,n+1):
                indices = np.where(labels==i)[0]
                y = int(indices[len(indices)//2]+6)
                if y<8 or y>=len(rgb)-8:
                    continue
                p = rgb[y-5:y+6,3:-3]
                a,b = profile[y-6],profile[y+6]
                if np.linalg.norm(b-a)<=40:continue
                if any(o['doc']==doc['id'] and abs(o['page_y']-(top+y))<=8
                       for o in result['boundaries'].get(family,[])+observations):continue
                delta = b-a
                if np.linalg.norm(delta)<35:
                    continue
                mid = p[4:7].astype(float)
                alpha = ((mid-a)*delta).sum(2)/max(float((delta*delta).sum()),1.)
                error = np.linalg.norm(mid-(a+alpha[...,None]*delta),axis=2)
                aa = (alpha>.1)&(alpha<.9)&(error<20)
                dark = (p.max(2)<80).mean(1)
                line = bool(dark[4:7].max()>.75 and dark[4:7].max()>np.median(dark[[0,1,9,10]])+.25)
                ink = p[4+int(dark[4:7].argmax())]
                ink = ink[ink.max(1)<80]
                observations.append(dict(doc=doc['id'],line=line,
                    line_width=int((dark[3:8]>.75).sum()) if line else 0,
                    antialias=float(aa.mean()),antialias_alpha=float(np.median(alpha[aa])) if aa.any() else 0.,
                    ink=np.median(ink,axis=0).astype(int).tolist() if len(ink) else [0,0,0],
                    evidence='observed-background-edge; class unknown',page_y=top+y,
                    **evidence_sources(doc['id'],[[x0,top,x1,bottom]],doc)))
        result['boundaries'].setdefault(family,[]).extend(observations)
        count += len(observations)
        print('BOUNDARY',doc['id'],len(observations),flush=True)
    windows.close()
    for family,summary in result['families'].items():
        bb = result['boundaries'].get(family,[])
        summary.update(boundaries=len(bb),boundary_line_rate=float(np.mean([b['line'] for b in bb])) if bb else 0.,
                       boundary_widths=[b['line_width'] for b in bb if b['line']],
                       boundary_antialias=float(np.median([b['antialias'] for b in bb])) if bb else None)
    write(output/'measurements.json',result)
    print('PASS boundaries',count,'observed colour edges',flush=True)


def refresh_library(output=NOISE):
    """Recompute measurements using cached original crop pixels, never old fits."""
    output=Path(output)
    result=read(output/'measurements.json')
    source_provenance=cache_provenance()
    records=read(paths.DATA/'genlab/fasttrack/training-v2.json')['records']
    cards={c['key']:c for c in read(paths.STATE)['cards']}
    store=model.Store(); kept=[]
    docs={d['id']:store.get(d['id']) for d in store.catalog if training_v2.training_id(d['id'])}
    for old in result['measurements']:
        key=old['class_id'];card=cards.get(key)
        if card is None:
            continue
        try:
            parents=ancestors(card,cards)|{old['source_doc']}
            from genlab.assemble import learn_v2 as learn
            if not learn.lineage_allowed(card,docs):continue
            if any(not training(store.get(d)) for d in parents) or not crop_allowed(old['source_doc'],old['box']):
                continue
            data=load_model(card)
        except (ValueError,FileNotFoundError):
            continue
        tile=data['tile'] if data['kind']=='tile' else data['strips'][0]
        try:
            binding=source_binding(old['source_doc'],old['box'],key)
            if old.get('binding') and old['binding']!=binding:continue
        except ValueError:
            continue
        # A refit must not launder tampered fields, including fields it replaces.
        for field in ('mask','colours','residual'):asset_path(output,old,field)
        with Image.open(output/old['colours']) as im:
            rgb=np.asarray(im.convert('RGB'))
        if old.get('asset_sha256'):
            asset_path(output,old,'colours')
        kept.append(save_measurement(output,card,old['source_doc'],old['box'],old['family'],parents,rgb,tile))
    if cache_provenance()!=source_provenance:raise ValueError('Sources changed during appearance refresh; retry from stable inputs')
    result.update(schema='asm-noise-2',built_at=time.time(),measurements=kept,provenance=source_provenance,gap_sources={})
    result['rails']={d:r for d,r in result['rails'].items() if nuisance_allowed(r,docs,records)}
    result['boundaries']={f:[b for b in rows if nuisance_allowed(b,docs,records)] for f,rows in result['boundaries'].items()}
    if cache_provenance()!=source_provenance:raise ValueError('Sources changed during nuisance refresh; retry from stable inputs')
    publish_library(output,result)
    print('PASS noise refresh',len(kept),'remeasured permitted native crops',flush=True)


def fit_gap_sources(output=NOISE):
    """Unknown annotations never supply a blank or lithology appearance pool."""
    output = Path(output)
    result = read(output/'measurements.json')
    result['gap_sources'] = {}
    write(output/'measurements.json',result)
    print('PASS gap appearance disabled: unknown is not blank',flush=True)


if __name__ == '__main__':
    main()
