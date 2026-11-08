"""Learn one clean repeat tile per labelled pattern from human-approved boxes.

Every output pixel comes from real scan pixels: the tile uses observed RGB
vectors nearest the median of period-aligned crops, so a single dirty crop (a stray
boundary row, a neighbouring layer) is out-voted by the clean ones.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import paths
from generator._texture import estimate_period
from generator.core import generate

DARK = 80          # max channel below this = ink
MODEL_VERSION = 'tiles-fix-r3-1'
_QUALITY_CACHE = (None,set())
_NATIVE_CROP_CACHE = (None,{})
_REPROV_CACHE = (None,{})
REPROV_SHA256 = 'cdda3f070e7e38d2340621582d7996e28f3113750e002538093136223698873d'
MAX_SOURCES = 12   # crops used to vote a tile
QUILT_ROWS = 2400  # length of the pre-quilted strip for patterns with no clean repeat


def norm(name):
    return ''.join(str(name or '').casefold().split())


def generation_split(doc, policy=None):
    """Read-only split including the dataset's held validation wells."""
    if policy is None:
        policy = json.loads((paths.DATA/'normalized'/'split-policy.json').read_text(encoding='utf-8'))
    from normalized import group_key
    group = policy['document_groups'].get(doc['id'],group_key(doc['name']))
    digest = hashlib.sha256((policy['seed']+':'+group).encode()).hexdigest()
    if int(digest[:8],16)%5==0:
        return group,'test'
    digest = hashlib.sha256(('genlab-validation-v1:'+group).encode()).hexdigest()
    return group, 'validation' if int(digest[:8],16)%5==0 else 'train'


def pixels_allowed(rgb):
    """Honor normalized test-identical/quarantined pixels before any fitting."""
    global _QUALITY_CACHE
    path=paths.DATA/'normalized'/'current.json'
    stamp=(str(path),path.stat().st_mtime_ns) if path.exists() else None
    if stamp!=_QUALITY_CACHE[0]:
        rows=json.loads(path.read_text(encoding='utf-8')).get('records',[]) if path.exists() else []
        denied={r['pixel_sha256'] for r in rows if r.get('split')=='test' or not r.get('usable_for_generation',False)}
        _QUALITY_CACHE=(stamp,denied)
    h,w=rgb.shape[:2]
    digest=hashlib.sha256(f'{w}x{h}:'.encode()+rgb.tobytes()).hexdigest()
    return digest not in _QUALITY_CACHE[1]


def lineage_allowed(proposal, docs, policy=None):
    """Fail closed on incomplete, revoked, cyclic or held-out source ancestry.

    A document ID alone is not pixel provenance. Every terminal source needs
    its revision, native boxes and dimension-prefixed pixel hashes. Declared
    ancestors and donors must all resolve to these sources, including aliases.
    """
    path=paths.DATA/'normalized'/'current.json'
    rows=json.loads(path.read_text(encoding='utf-8')).get('records',[]) if path.exists() else []
    denied={r['pixel_sha256'] for r in rows if r.get('split')=='test' or not r.get('usable_for_generation',False)}
    sources=[];declared=set();active=set()
    by_id={r['sample_id']:r for r in rows}
    def visit(node):
        if not isinstance(node,dict) or id(node) in active:
            raise ValueError('Invalid or cyclic lineage')
        active.add(id(node))
        if 'render_model' in node:
            payload=node['render_model']
            if not isinstance(payload,dict) or not isinstance(payload.get('provenance'),dict) or not payload['provenance']:
                raise ValueError('Missing embedded ancestry')
            outer={k:v for k,v in node.items() if k!='render_model'}
            if source_identity(outer)!=source_identity(payload['provenance']):
                raise ValueError('Conflicting embedded ancestry')
            if node.get('signature') and node['signature']!=payload.get('signature'):
                raise ValueError('Conflicting model identity')
        declared.update(node.get('ancestor_docs') or [])
        for ref in node.get('normalized_sources') or []:
            row=by_id.get(ref.get('sample_id'))
            if not row or not row.get('usable_for_generation') or ref.get('annotation_version')!=row.get('annotation_version'):
                raise ValueError('Revoked normalized source')
            declared.add(row['document_id'])
        for field in ('lineage','donor','recovery','provenance','render_model'):
            value=node.get(field)
            if field=='provenance' and not isinstance(value,(dict,list)):continue
            children=value if isinstance(value,list) else [value] if value else []
            for child in children:visit(child)
        if node.get('doc_id') or node.get('document_id'):
            sources.append(node)
        active.remove(id(node))
    try:
        visit(proposal)
    except (ValueError,TypeError):
        return False
    complete={}
    for source in sources:
        did = source.get('doc_id') or source.get('document_id')
        if did not in docs:
            return False
        doc = docs[did].get('doc',docs[did])
        group,split = generation_split(doc,policy)
        if split!='train' or source.get('source_group',group)!=group:
            return False
        if 'source_revision' in source and source['source_revision']!=doc.get('revision'):
            return False
        boxes=source.get('boxes') or source.get('column_boxes') or []
        hashes=source.get('pixel_sha256') or []
        hashes=[hashes] if isinstance(hashes,str) else hashes
        if any(h in denied for h in hashes):return False
        if boxes and hashes and len(boxes)==len(hashes) and 'source_revision' in source:
            for box,digest in zip(boxes,hashes):
                if len(box)!=4 or not isinstance(digest,str) or len(digest)!=64:return False
                matching=[r for r in rows if r['document_id']==did and r.get('box_xyxy')==box]
                if matching and not any(r.get('usable_for_generation') and r['pixel_sha256']==digest and
                        (source.get('legend_id') is None or source['legend_id'] in [t.get('legend_id') for t in r.get('targets',[])]) and
                        (not source.get('sample_id') or source['sample_id']==r['sample_id']) and
                        (not source.get('annotation_version') or source['annotation_version']==r['annotation_version'])
                        for r in matching):return False
                if any(r['document_id']==did and not r.get('usable_for_generation',False) and
                       boxes_overlap(box,r['box_xyxy']) for r in rows):return False
            complete.setdefault(did,[]).append(source)
        if source.get('tile_path') and source.get('tile_sha256'):
            tile_path=(paths.OUT/source['tile_path']).resolve()
            if not tile_path.is_relative_to(paths.OUT.resolve()) or not tile_path.exists():
                return False
            if hashlib.sha256(tile_path.read_bytes()).hexdigest()!=source['tile_sha256']:
                return False
    # Root recipient may have no pixels of its own only if separately declared
    # target and donor pixel sources exist. A donor stub never supplies evidence.
    return bool(complete) and all((s.get('doc_id') or s.get('document_id')) in complete for s in sources) and declared<=complete.keys()


def ancestry_metadata(proposal):
    """Copy every ancestry declaration without reducing recursive evidence."""
    import copy
    fields=('lineage','recovery','donor','ancestor_docs','normalized_sources')
    result={k:copy.deepcopy(proposal[k]) for k in fields if k in proposal}
    ancestors=set(result.get('ancestor_docs') or [])
    active=set()
    def visit(node):
        if isinstance(node,list):
            for item in node:visit(item)
        elif isinstance(node,dict):
            if id(node) in active:raise ValueError('Cyclic ancestry')
            active.add(id(node))
            did=node.get('doc_id') or node.get('document_id')
            if did:ancestors.add(did)
            ancestors.update(node.get('ancestor_docs') or [])
            for k in ('lineage','donor','recovery','provenance','render_model'):visit(node.get(k))
            active.remove(id(node))
    visit(proposal)
    embedded=(proposal.get('render_model') or {}).get('provenance')
    if isinstance(embedded,dict):
        # Preserve embedded evidence as well as the ancestor names. Never turn
        # a loaded donor into a recipient-only declaration on re-export.
        for field in ('lineage','normalized_sources'):
            values=result.setdefault(field,[]) if embedded.get(field) else result.get(field,[])
            for value in embedded.get(field,[]):
                if value not in values:values.append(copy.deepcopy(value))
    result['ancestor_docs']=sorted(ancestors)
    return result


def source_identity(node):
    """Canonical declared source evidence, independent of wrapper nesting."""
    sources=set();ancestors=set();normalized=set();active=set()
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict):
            if id(value) in active:raise ValueError('Cyclic ancestry')
            active.add(id(value))
            ancestors.update(value.get('ancestor_docs') or [])
            did=value.get('doc_id') or value.get('document_id')
            if did:ancestors.add(did)
            boxes=value.get('boxes') or value.get('column_boxes')
            hashes=value.get('pixel_sha256')
            if did and boxes and hashes:
                sources.add(json.dumps([did,value.get('source_revision'),value.get('legend_id'),boxes,hashes],sort_keys=True))
            for ref in value.get('normalized_sources') or []:
                normalized.add(json.dumps(ref,sort_keys=True))
            for field in ('lineage','recovery','donor','provenance','render_model'):visit(value.get(field))
            active.remove(id(value))
    visit(node)
    return sources,ancestors,normalized


def model_binding(payload):
    return hashlib.sha256(json.dumps({k:v for k,v in payload.items() if k!='binding_sha256'},
                                    sort_keys=True,separators=(',',':')).encode()).hexdigest()


def reproduced_proposal(proposal, docs):
    """Overlay the reviewed ancestry ledger without changing proposal files.

    The inventory owns the live review and tile path. The pinned proof supplies
    only lineage, and remains revocable by bytes, revisions and current policy.
    """
    import copy
    global _REPROV_CACHE
    path=paths.OUT/'reprov'/'ancestry.jsonl'
    if not path.exists():return proposal
    stamp=(str(path),path.stat().st_mtime_ns,path.stat().st_size)
    if stamp!=_REPROV_CACHE[0]:
        raw=path.read_bytes();records={}
        if hashlib.sha256(raw).hexdigest()==REPROV_SHA256:
            for line in raw.decode('utf-8').splitlines():
                record=json.loads(line)
                if record.get('admitted') and record.get('status')=='admitted':records[record['key']]=record
        _REPROV_CACHE=(stamp,records)
    record=_REPROV_CACHE[1].get(f"{proposal['doc_id']}:{proposal['legend_id']}")
    if not record:return proposal
    try:
        source=(paths.LAB/record['proposal_path']).resolve()
        tile=Path(proposal['tile_path']).resolve()
        if not source.is_relative_to(paths.PROPOSALS.resolve()) or not tile.is_relative_to(paths.PROPOSALS.resolve()):return proposal
        if tile!=(paths.OUT/record['tile_path']).resolve():return proposal
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['proposal_sha256']:return proposal
        if hashlib.sha256(tile.read_bytes()).hexdigest()!=record['tile_sha256']:return proposal
        if proposal.get('round')!=record['round'] or proposal['doc_id']!=record['doc_id'] or proposal['legend_id']!=record['legend_id']:return proposal
        enriched=copy.deepcopy(proposal)
        for field in ('lineage','ancestor_docs','normalized_sources'):enriched[field]=copy.deepcopy(record[field])
        if any(not old<=new for old,new in zip(source_identity(proposal),source_identity(enriched))):return proposal
        if not lineage_allowed(enriched,docs):return proposal
        return enriched
    except (KeyError,OSError,TypeError,ValueError):
        return proposal


def export_model(data, folder, signature, provenance=None):
    """Save rendering state separately from display samples, with bound bytes."""
    import io
    prior=source_identity(data)
    supplied=source_identity(provenance) if provenance is not None else prior
    if any(not a<=b for a,b in zip(prior,supplied)):
        raise ValueError('Conflicting model ancestry; re-export cannot discard sources')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    kind=data['kind'];arrays={'bg':np.asarray(data['bg'])}
    if kind=='tile':arrays['tile']=data['tile']
    elif kind=='quilt':arrays.update({f'strip_{i}':a for i,a in enumerate(data['strips'])})
    elif kind in ('texture','jitter'):arrays.update({f'glyph_{i}':a for i,a in enumerate(data['glyphs'])})
    else:raise ValueError('Unsupported render model: '+kind)
    fields=dict(phase=list(map(int,data.get('phase',(0,0)))),anchored=bool(data.get('anchored',False)),
                anchored_x=bool(data.get('anchored_x',data.get('anchored',False))),
                anchored_y=bool(data.get('anchored_y',data.get('anchored',False))),
                x_repeat=bool(data.get('x_repeat',True)),x_origin=int(data.get('x_origin',0)))
    for k in ('density','large_density','small_density','spacing','jitter','glyph_evidence','strip_origins','max_layer_height','native_extent','quilt_local'):
        if k in data:fields[k]=data[k]
    if kind=='quilt':
        fields['max_layer_height']=min(data.get('max_layer_height',min(len(s) for s in data['strips'])),min(len(s) for s in data['strips']))
        fields['quilt_local']=data.get('quilt_local','strip_origins' not in data and 'x_origin' not in data)
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);raw=stream.getvalue()
    digest=hashlib.sha256(raw).hexdigest();asset=folder/('model-'+signature+'-'+digest[:16]+'.npz')
    if not asset.exists():asset.write_bytes(raw)
    payload=dict(schema='genlab-render-2',kind=kind,signature=signature,placement=fields,
                provenance=ancestry_metadata(provenance or data),
                asset=dict(path=asset.relative_to(paths.OUT).as_posix(),sha256=digest,
                           arrays={k:list(a.shape) for k,a in arrays.items()}))
    payload['binding_sha256']=model_binding(payload)
    return payload


def load_model(payload, root=None, signature=None):
    """Reload a complete model; missing state and changed bytes are errors."""
    if payload.get('schema')!='genlab-render-2':raise ValueError('Unsupported rendering schema')
    if payload.get('binding_sha256') and payload['binding_sha256']!=model_binding(payload):
        raise ValueError('Model metadata binding hash mismatch')
    if signature is not None and payload['signature']!=signature:raise ValueError('Model signature mismatch')
    root=Path(root or paths.OUT).resolve();asset=(root/payload['asset']['path']).resolve()
    if not asset.is_relative_to(root) or not asset.is_file() or asset.stat().st_size>64*1024*1024:
        raise ValueError('Missing, unsafe or oversized model asset')
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=payload['asset']['sha256']:raise ValueError('Model asset hash mismatch')
    fields=payload['placement']
    if not all(k in fields for k in ('phase','anchored_x','anchored_y','x_repeat','x_origin')):
        raise ValueError('Incomplete rendering placement')
    shapes=payload['asset']['arrays']
    if sum(int(np.prod(s)) for s in shapes.values())>64*1024*1024:raise ValueError('Oversized model arrays')
    import zipfile
    with zipfile.ZipFile(asset) as bank:
        if sum(info.file_size for info in bank.infolist())>64*1024*1024:
            raise ValueError('Oversized decompressed model bank')
    with np.load(asset,allow_pickle=False) as z:
        if set(z.files)!=set(shapes):raise ValueError('Model bank member mismatch')
        arrays={k:z[k].copy() for k in z.files}
    if any(list(a.shape)!=shapes[k] for k,a in arrays.items()):raise ValueError('Model dimensions mismatch')
    result=dict(fields,kind=payload['kind'],bg=arrays['bg'],**payload.get('provenance',{}))
    if result['kind']=='tile':result['tile']=arrays['tile']
    elif result['kind']=='quilt':
        result['strips']=[arrays[f'strip_{i}'] for i in range(len(arrays)-1)]
        if 'strip_origins' not in result:result.setdefault('quilt_local',True)
    elif result['kind'] in ('texture','jitter'):result['glyphs']=[arrays[f'glyph_{i}'] for i in range(len(arrays)-1)]
    else:raise ValueError('Unsupported rendering kind')
    return result


def load_proposal(proposal, examples, docs, policy=None):
    """Guarded proposal bridge: preview images never become rendering tiles."""
    if not lineage_allowed(proposal,docs,policy):raise ValueError('Missing or revoked proposal ancestry')
    p=proposal
    if p.get('render_model'):
        outer={k:v for k,v in p.items() if k!='render_model'}
        provenance=p['render_model'].get('provenance')
        if not isinstance(provenance,dict) or not lineage_allowed(provenance,docs,policy):
            raise ValueError('Missing or revoked embedded model ancestry')
        if source_identity(outer)!=source_identity(provenance):
            raise ValueError('Conflicting proposal and embedded model ancestry')
        data=load_model(p['render_model'])
    elif p.get('recovery',{}).get('tile_is_preview_only'):
        raise ValueError('Preview-only proposal requires a hashed rendering model')
    else:
        with Image.open(p['tile_path']) as im:tile=np.asarray(im.convert('RGB')).copy()
        if not pixels_allowed(tile):raise ValueError('Quarantined proposal pixels')
        data=dict(kind='tile',tile=tile,bg=background(tile),phase=(0,0),anchored=False,
                  anchored_x=False,anchored_y=False,x_repeat=True,x_origin=0)
    arrays=([data['tile']] if data['kind']=='tile' else data['strips'] if data['kind']=='quilt' else data['glyphs'])
    if any(not pixels_allowed(a) for a in arrays):raise ValueError('Quarantined model pixels')
    m=dict(key=f"{p['doc_id']}:{p['legend_id']}",doc_id=p['doc_id'],legend_id=p['legend_id'],
        label=p['label'],kind=data['kind'],period=p.get('period',dict(x=None,y=None)),
        examples=len(p.get('column_boxes') or []),sources=[e['id'] for e in examples],
        suspects=[],suspect_count=0,voters=0,method=p.get('method','unknown'),
        looks_like=p.get('looks_like'),proposal_round=p['round'],proposal_notes=p.get('notes'),
        tile_source='agent proposal',evidence_kind='column' if p.get('column_boxes') else 'swatch',
        **ancestry_metadata(p))
    if data['kind']=='tile':m['tile_size']=[data['tile'].shape[1],len(data['tile'])]
    return m,data


def stroke_weight(rgb):
    from scipy.ndimage import distance_transform_edt
    mask = np.linalg.norm(rgb.astype(float)-background(rgb),axis=2)>35
    return float(2*np.percentile(distance_transform_edt(mask)[mask],90)-1) if mask.any() else 0.


def fit_transfer(pairs):
    """Robust family offsets, checked by leaving each training document out.

    Each pair records doc_id, family, period_delta, stroke_delta, colour_delta
    and real source lineage. A family is usable only with >=2 documents, >=6
    pairs, and non-inferior LODO error on every parameter. Sparse families keep
    explicit uncertainty and do not change tiles.
    """
    families = defaultdict(list)
    for pair in pairs:
        families[pair['family']].append(pair)
    result = dict(schema='legend-column-transfer-1',families={},pairs=pairs)
    for family, rows in families.items():
        docs = sorted({r['doc_id'] for r in rows})
        def vector(r):
            return np.array([r['period_delta'],r['stroke_delta'],*r['colour_delta']],float)
        values = np.stack([vector(r) for r in rows])
        center = np.median(values,axis=0)
        checks = []
        for doc in docs:
            fit = [vector(r) for r in rows if r['doc_id']!=doc]
            held = [vector(r) for r in rows if r['doc_id']==doc]
            if not fit:
                continue
            prediction = np.median(fit,axis=0)
            checks.append(dict(doc_id=doc,pairs=len(held),before=np.abs(held).mean(0).tolist(),
                               after=np.abs(np.stack(held)-prediction).mean(0).tolist()))
        before = np.mean([c['before'] for c in checks],axis=0) if checks else np.full(5,np.nan)
        after = np.mean([c['after'] for c in checks],axis=0) if checks else np.full(5,np.nan)
        spread=np.percentile(values,90,axis=0)-np.percentile(values,10,axis=0)
        usable = bool(len(docs)>=2 and len(rows)>=6 and bool(np.all(after<=before+1e-6)) and
                      spread[0]<=2 and spread[1]<=1 and bool(np.all(spread[2:]<=10)))
        result['families'][family] = dict(documents=docs,pairs=len(rows),usable=usable,
            period_delta=int(round(center[0])),stroke_delta=int(np.clip(round(center[1]),-1,1)),
            colour_delta=center[2:].tolist(),lodo=checks,
            lodo_before=before.tolist() if checks else None,lodo_after=after.tolist() if checks else None,
            uncertainty=dict(period_delta=np.percentile(values[:,0],[10,90]).tolist(),
                             stroke_delta=np.percentile(values[:,1],[10,90]).tolist(),
                             colour_delta=np.percentile(values[:,2:],[10,90],axis=0).tolist()),
            lineage=[r['lineage'] for r in rows],
            reason='LODO non-inferior on period, stroke and colour' if usable else 'Insufficient independent documents, broad parameter uncertainty or LODO regression; no transfer applied')
    return result


def apply_transfer(data, rule):
    if not rule or not rule.get('usable') or data['kind']!='tile':
        return data
    from scipy.ndimage import binary_dilation,binary_erosion,distance_transform_edt
    tile = data['tile'].copy()
    height = max(1,len(tile)+rule['period_delta'])
    if height!=len(tile):
        tile = np.asarray(Image.fromarray(tile).resize((tile.shape[1],height),Image.Resampling.NEAREST)).copy()
    bg = background(tile)
    mask = np.linalg.norm(tile.astype(float)-bg,axis=2)>35
    shift = np.asarray(rule['colour_delta'])
    tile[~mask] = np.clip(tile[~mask].astype(float)+shift,0,255).astype(np.uint8)
    stroke = rule['stroke_delta']
    if mask.any() and stroke:
        changed = binary_dilation(mask,iterations=stroke) if stroke>0 else binary_erosion(mask,iterations=-stroke)
        if stroke>0:
            nearest = distance_transform_edt(~mask,return_distances=False,return_indices=True)
            tile[changed & ~mask] = tile[tuple(nearest[:,changed & ~mask])]
        else:
            tile[mask & ~changed] = np.clip(bg+shift,0,255).astype(np.uint8)
    return dict(data,tile=tile,bg=background(tile),transfer=dict(period_delta=rule['period_delta'],
                stroke_delta=stroke,colour_delta=rule['colour_delta'],documents=rule['documents']))


# ---------------------------------------------------------------- examples
def human_examples(store):
    """All approved single-label boxes, grouped by (document, legend id)."""
    groups = defaultdict(list)
    docs = {}
    from normalized import SplitPolicy
    policy=SplitPolicy(store)
    policy.load_index()
    for entry in store.catalog:
        doc = store.get(entry['id'])
        names = {e['id']: e['label_printed'] for e in doc['legend']}
        docs[doc['id']] = dict(id=doc['id'], name=doc['name'], legend=names,
                               column=doc['boxes']['interpreted_lithology_column'], doc=doc, training_allowed=policy.split(doc)=='train')
        for seg in doc['segments']:
            if seg.get('status') != 'approved' or len(seg.get('legend_ids') or []) != 1:
                continue
            if not policy.allowed(doc,seg):
                continue
            x0, y0, x1, y1 = seg['box']
            if x1 - x0 < 8 or y1 - y0 < 6:
                continue
            lid = seg['legend_ids'][0]
            from normalized import sample_id
            record=policy.rows[sample_id(doc,seg)]
            groups[(doc['id'], lid)].append(dict(
                id=seg['id'], doc_id=doc['id'], legend_id=lid, label=names.get(lid, str(lid)),
                box=list(seg['box']), provenance=seg.get('provenance', ''),
                flagged=bool(seg.get('flags')),normalized_sample_id=record['sample_id'],annotation_version=record['annotation_version']))
    return docs, groups


# ---------------------------------------------------------------- crops
class Crops:
    """Disk cache; cold reads use bounded TIFF segments, never a whole page."""

    def __init__(self, images):
        self.images = images

    def get(self, doc, box):
        path = paths.CROPS / doc['id'] / ('%d_%d_%d_%d.png' % tuple(box))
        if path.exists():
            with Image.open(path) as image:return np.asarray(image.convert('RGB')).copy()
        # Verification may trim an already cached source rectangle. Reuse that
        # bounded observation rather than sending a smaller request to TIFF.
        x0,y0,x1,y1=map(int,box)
        containers=[]
        for cached in path.parent.glob('*.png'):
            try:a,b,c,d=map(int,cached.stem.split('_'))
            except ValueError:continue
            if a<=x0<x1<=c and b<=y0<y1<=d and (c-a)*(d-b)*3<=64*1024*1024:
                containers.append(((c-a)*(d-b),cached,(x0-a,y0-b,x1-a,y1-b)))
        if containers:
            _,cached,relative=min(containers,key=lambda item:item[0])
            with Image.open(cached) as image:return np.asarray(image.crop(relative).convert('RGB')).copy()
        # The normalized index also holds authoritative bounded native PNGs.
        # Only current, eligible observations of this exact box may supply a
        # cache miss. No held/test or revoked object is opened here.
        global _NATIVE_CROP_CACHE
        index=paths.DATA/'normalized'/'current.json'
        stamp=(str(index),index.stat().st_mtime_ns) if index.exists() else None
        if stamp!=_NATIVE_CROP_CACHE[0]:
            records=json.loads(index.read_text(encoding='utf-8')).get('records',[]) if index.exists() else []
            native={}
            for row in records:
                if row.get('split')!='test' and row.get('usable_for_generation') and row.get('native_png'):
                    native.setdefault((row['document_id'],tuple(row['box_xyxy'])),[]).append(row)
            _NATIVE_CROP_CACHE=(stamp,native)
        native_candidates=[]
        for (did,bounds),records in _NATIVE_CROP_CACHE[1].items():
            a,b,c,d=bounds
            if did==doc['id'] and a<=x0<x1<=c and b<=y0<y1<=d and (c-a)*(d-b)*3<=64*1024*1024:
                native_candidates.extend(records)
        native_candidates.sort(key=lambda row:(row['box_xyxy'][2]-row['box_xyxy'][0])*(row['box_xyxy'][3]-row['box_xyxy'][1]))
        for row in native_candidates:
            from normalized import version,sample_id
            segment=next((s for s in doc.get('segments',[]) if s['id']==row.get('annotation_id')),None)
            if not segment or segment.get('status')!='approved' or version(segment)!=row['annotation_version'] or sample_id(doc,segment)!=row['sample_id']:continue
            entries={e['id']:e for e in doc['legend']}
            legend_stamp=hashlib.sha256(json.dumps([entries[i] for i in segment['legend_ids']],sort_keys=True).encode()).hexdigest()[:16]
            if row.get('legend_version')!=legend_stamp:continue
            native_root=index.parent.resolve();native=(native_root/row['native_png']).resolve()
            if not native.is_relative_to(native_root) or not native.is_file():continue
            a,b,c,d=row['box_xyxy']
            with Image.open(native) as image:
                if image.size!=(c-a,d-b) or image.width*image.height*3>64*1024*1024:continue
                rgb=np.asarray(image.convert('RGB')).copy()
            digest=hashlib.sha256(f'{c-a}x{d-b}:'.encode()+rgb.tobytes()).hexdigest()
            if digest==row['pixel_sha256'] and pixels_allowed(rgb):
                view=rgb[y0-b:y1-b,x0-a:x1-a].copy()
                if not pixels_allowed(view):raise ValueError('Quarantined normalized native subcrop')
                return view
        # model.Images has a silent whole-page Pillow fallback. Preflight and
        # read supported strips here so that fallback is unreachable.
        import model
        import tifffile
        with tifffile.TiffFile(model.DATASET/'images'/doc['image_name']) as tf:
            page=tf.pages[0]
            segment_bytes=page.rowsperstrip*page.imagewidth*max(3,page.samplesperpixel)*max(1,page.bitspersample//8)
            if (page.is_tiled or page.planarconfig!=1 or page.photometric not in (0,1,2,3) or
                    page.rowsperstrip>4000 or segment_bytes>64*1024*1024):
                raise ValueError('Unsafe TIFF crop-cache miss: bounded cached PNG required for '+doc['id']+' '+str(box))
            try:page.decode(None,0)
            except (ValueError,NotImplementedError) as error:
                raise ValueError('Unsupported bounded TIFF codec; cached PNG required') from error
            x0,y0,x1,y1=map(int,box)
            if not (0<=x0<x1<=page.imagewidth and 0<=y0<y1<=page.imagelength) or (x1-x0)*(y1-y0)*3>64*1024*1024:
                raise ValueError('Invalid or oversized bounded crop')
            out=np.empty((y1-y0,x1-x0,3),np.uint8)
            for index in range(y0//page.rowsperstrip,(y1-1)//page.rowsperstrip+1):
                tf.filehandle.seek(page.dataoffsets[index])
                segment,where,_=page.decode(tf.filehandle.read(page.databytecounts[index]),index)
                start=int(where[2]);a=max(y0,start);b=min(y1,start+segment.shape[1])
                pixels=segment[0,a-start:b-start,x0:x1]
                if page.photometric==3:pixels=(page.colormap>>8).astype(np.uint8).T[pixels[:,:,0]]
                elif pixels.shape[-1]==1:
                    if pixels.dtype==bool:pixels=pixels.astype(np.uint8)*255
                    else:pixels=np.clip(pixels,0,255).astype(np.uint8)
                    if page.photometric==0:pixels=255-pixels
                    pixels=np.repeat(pixels,3,axis=2)
                out[a-y0:b-y0]=pixels[:,:,:3]
        rgb=Image.fromarray(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(path)
        return np.asarray(rgb)


def background(rgb):
    light = rgb[rgb.max(2) >= DARK]
    if not len(light):
        return np.array([0, 0, 0], float)
    q = light // 16
    vals, counts = np.unique(q, axis=0, return_counts=True)
    top = vals[counts.argmax()]
    return np.median(light[(q == top).all(1)], axis=0)


def clean(rgb):
    """Cut edge rails and rows that belong to a neighbouring layer.

    Returns (crop, (left, top)) where left/top are offsets into the input.
    """
    from .swatch import remove_frame
    rgb, (fl, ft, _, _) = remove_frame(rgb)
    h, w = rgb.shape[:2]
    dark = rgb.max(2) < DARK
    colfrac = dark.mean(0)
    left = 0
    while left < w // 3 and colfrac[left] > .7:
        left += 1
    right = w
    while right > w - w // 3 and colfrac[right - 1] > .7:
        right -= 1
    core = rgb[:, left:right]
    bg = background(core)
    ok = (np.sqrt(((core.astype(float) - bg) ** 2).sum(2)) < 45) | (core.max(2) < DARK)
    rowok = ok.mean(1)
    top, bottom = 0, h
    while top < h // 3 and rowok[top] < .85:
        top += 1
    while bottom > h - h // 3 and rowok[bottom - 1] < .85:
        bottom -= 1
    return core[top:bottom].copy(), (left + fl, top + ft)


def _block_tile(rgb, py, px):
    """Median over all whole periods of one crop: that crop's own tile."""
    ky, kx = max(1, rgb.shape[0] // py), max(1, rgb.shape[1] // px)
    blocks = rgb[:ky * py, :kx * px].reshape(ky, py, kx, px, 3).transpose(0, 2, 1, 3, 4).reshape(-1, py, px, 3)
    return supported_vote(blocks)


def supported_vote(votes, fallback=None):
    """Choose an observed RGB vector nearest the median; never mix channels."""
    votes = np.asarray(votes, dtype=float)
    valid = np.isfinite(votes).all(-1)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            center = np.nanmedian(votes, axis=0)
    distance = np.square(votes-center).sum(-1)
    distance[~valid] = np.inf
    pick = np.argmin(distance, axis=0)
    yy, xx = np.indices(pick.shape)
    result = votes[pick, yy, xx].copy()
    missing = ~valid.any(0)
    if fallback is not None:
        result[missing] = fallback[missing]
    else:
        result[missing] = 0
    return result.astype(np.uint8)


def boxes_overlap(a, b, guard=0):
    return (min(a[2],b[2]) > max(a[0],b[0]) and
            min(a[3]+guard,b[3]+guard) > max(a[1],b[1]))


def reserve_evidence(examples, guard=512):
    """Reserve a separated depth range using geometry before reading pixels.

    512 native rows exceeds the supported repeat search. Overlapping IDs belong
    to one component. All selection/learning callers use only the first list.
    """
    ordered = sorted(examples, key=lambda e:(e['box'][1],e['box'][3],str(e['id'])))
    components = []
    for e in ordered:
        if components and e['box'][1] < max(v['box'][3] for v in components[-1])+guard:
            components[-1].append(e)
        else:
            components.append([e])
    if len(ordered) >= 3 and len(components) >= 2:
        held = components[-1]
        return [e for part in components[:-1] for e in part], held
    return ordered, []


def repeat_supported(rgb, period):
    """Require multiple cycles and motif overlap as well as background match."""
    if not period or len(rgb) < 2*period:
        return False
    a, b = rgb[:-period], rgb[period:]
    bg = background(rgb)
    ma = np.linalg.norm(a.astype(float)-bg,axis=2)>35
    mb = np.linalg.norm(b.astype(float)-bg,axis=2)>35
    dice = 2*(ma & mb).sum()/max(1,ma.sum()+mb.sum())
    pixels = (np.linalg.norm(a.astype(float)-b.astype(float),axis=2)<45).mean()
    return bool(pixels >= .90 and (dice >= .78 or not (ma.any() or mb.any())))


def scanner_lines(rgb):
    """Repair isolated white scanner scratches in coloured learning crops.

    Evaluation crops remain untouched. A majority of neighbouring native
    scanlines votes each replacement pixel, retaining strokes crossing a scratch.
    """
    out = rgb.copy()
    count = 0
    # A one-pixel knockout through otherwise continuous horizontal lines is
    # another scanner scratch. Dashed fills fail this near-uniform-row test.
    ink = out.max(2) < DARK
    row = ink.mean(1)
    if (((row < .02) | (row > .85)).mean() >= .95 and (row > .85).any()):
        bg = background(out)
        blank = (np.linalg.norm(out.astype(float)-bg,axis=2)<35).mean(0) >= .98
        for i in range(1,out.shape[1]-1):
            if (blank[i] and not blank[i-1] and not blank[i+1] and
                    (np.linalg.norm(out[:,i-1].astype(float)-out[:,i+1],axis=1)<35).mean()>=.98):
                out[:,i] = out[:,i-1]
                count += 1
    if np.linalg.norm(background(rgb) - np.array([255,255,255])) < 45:
        return out, count
    for axis in (1, 0):
        v = np.moveaxis(out, axis, 0)
        bad = (v.min(2) >= 240).mean(1) >= .95
        for i in np.flatnonzero(bad):
            if i < 2 or i >= len(v)-2 or (bad[i-1] and bad[i+1]):
                continue
            near = [j for j in range(max(0,i-3),min(len(v),i+4)) if not bad[j]]
            if len(near) >= 3:
                near = near if len(near)%2 else near[:-1]
                v[i] = supported_vote(v[near,:,None,:])[:,0]
                count += 1
    return out, count


def horizontal_period(rgb):
    """Native overlap may establish a period wider than half a column lane."""
    ink = (rgb.max(2)<DARK).mean(1)
    if (((ink<.02)|(ink>.85)).mean()>=.95 and (ink>.85).any() and (ink<.02).any()):
        return 1  # median across x outvotes thin scratches through solid stripes
    from .swatch import axis_repeat
    p, measured, seam = axis_repeat(rgb, 1, edge_guard=0)
    if seam['ok'] and seam['method']=='repeat-overlap':
        return p
    # A short lane can show exactly one cycle plus its closing pixel column.
    # Require textured, identical closing columns, not two blank background edges.
    ink = rgb[:,0].max(1) < DARK
    if rgb.shape[1] > 6 and .05 < ink.mean() < .95 and np.array_equal(rgb[:,0],rgb[:,-1]):
        return rgb.shape[1]-1
    return None


def _align(tile, ref):
    """Cyclic shift of tile that best matches ref (FFT cross-correlation).

    Returns (dy, dx, mean absolute error after the shift).
    """
    a = tile.astype(np.float64) - tile.mean((0, 1))
    r = ref.astype(np.float64) - ref.mean((0, 1))
    corr = np.zeros(tile.shape[:2])
    for ch in range(3):
        corr += np.real(np.fft.ifft2(np.conj(np.fft.fft2(a[..., ch])) * np.fft.fft2(r[..., ch])))
    dy, dx = np.unravel_index(int(corr.argmax()), corr.shape)
    err = float(np.abs(np.roll(np.roll(tile.astype(np.int16), dy, 0), dx, 1) - ref.astype(np.int16)).mean())
    return int(dy), int(dx), err


def learn(crops, docs, key, examples):
    """Build the pattern model for one (document, legend id)."""
    doc_id, legend_id = key
    doc = docs[doc_id]['doc']
    # Long, unflagged crops first; they carry the most repeat cycles.
    training, held = reserve_evidence(examples)
    ranked = sorted(training, key=lambda e: (e['flagged'], -(e['box'][3] - e['box'][1]) * (e['box'][2] - e['box'][0])))
    distinct = []
    for e in ranked:
        if not any(boxes_overlap(e['box'],v['box']) for v in distinct):
            distinct.append(e)
    ranked = distinct
    everyone = []
    for ex in ranked[:400]:
        rgb = crops.get(doc, ex['box'])
        if not pixels_allowed(rgb):
            continue
        c, (dl, dt) = clean(rgb)
        c, repairs = scanner_lines(c)
        if c.shape[0] >= 4 and c.shape[1] >= 6:
            everyone.append(dict(ex=ex, rgb=c, raw=rgb, offset=(dl,dt), x0=ex['box'][0] + dl, y0=ex['box'][1] + dt,
                                 bg=tuple(int(v) // 24 for v in background(c)), repairs=repairs))
    # The pattern is what most boxes agree on; a long box with another
    # colour is a likely label slip and must not become the reference.
    majority = Counter(c['bg'] for c in everyone).most_common(1)[0][0] if everyone else None
    near = lambda bg: majority is not None and max(abs(a - b) for a, b in zip(bg, majority)) <= 1
    suspects = [dict(id=c['ex']['id'], box=c['ex']['box'], reason='background colour differs from the other boxes')
                for c in everyone if not near(c['bg'])]
    cleaned = [c for c in everyone if near(c['bg']) and c['rgb'].shape[0] >= 6][:MAX_SOURCES]
    # Use training crops only to isolate banks with extra tall/merged symbols.
    # A small clean cohort is conservative evidence, never an automatic badge.
    if len(cleaned)>1:
        from .score import component_profile,component_comparison
        reference=min(cleaned,key=lambda c:component_profile(c['rgb'])['height_q95'])
        profile=component_profile(reference['rgb'])
        compact=0<profile['height_q95']<=8 and len(profile['components'])>=3
        retained=[]
        for c in cleaned:
            detail=component_comparison(reference['rgb'],c['rgb'])
            if compact and not detail['acceptable']:
                suspects.append(dict(id=c['ex']['id'],box=c['ex']['box'],
                    reason='Component ink distribution differs from clean training evidence',component_check=detail))
                folder=paths.OUT/'recover'/'fix-round3'/'rejected-sources';folder.mkdir(parents=True,exist_ok=True)
                from PIL import ImageDraw
                panes=[reference['rgb'][:240],c['rgb'][:240]]
                sheet=Image.new('RGB',(sum(a.shape[1]*3+16 for a in panes),max(len(a) for a in panes)*3+45),'#ddd')
                draw=ImageDraw.Draw(sheet);x=0
                for title,a in zip(('Training reference @3x','Rejected source @3x'),panes):
                    draw.text((x+2,3),title,fill='black');im=Image.fromarray(a)
                    sheet.paste(im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST),(x,40));x+=im.width*3+16
                path=folder/(f'{doc_id}-{legend_id}-{len(suspects)}.png');sheet.save(path)
                suspects[-1]['diagnostic_sheet']=path.relative_to(paths.OUT).as_posix()
            else:retained.append(c)
        cleaned=retained
    model = dict(key='%s:%s' % key, doc_id=doc_id, legend_id=legend_id,
                 label=examples[0]['label'], examples=len(examples),
                 sources=[c['ex']['id'] for c in cleaned], suspects=suspects,
                 scanner_lines_removed=sum(c['repairs'] for c in cleaned),
                 suspect_count=len(suspects))
    model.update(training_boxes=[e['box'] for e in training],
                 heldout_ids=[e['id'] for e in held], heldout_boxes=[e['box'] for e in held],
                 evidence_guard_rows=512, independent_source_ranges=len(cleaned))
    if not cleaned:
        model.update(kind='none', reason='No example survived edge cleaning (too thin).')
        return model, None
    # Pool the repeat length over several long boxes; one odd crop must not decide it.
    longest = sorted(cleaned, key=lambda c: -c['rgb'].shape[0])[:6]
    ys = [estimate_period(c['rgb'], 0) for c in longest if c['rgb'].shape[0] >= 24]
    xs = [horizontal_period(c['rgb']) for c in longest if c['rgb'].shape[0] >= 24]
    ys = [p['pixels'] for p in ys if p['pixels']]
    xs = [p for p in xs if p]
    PY = Counter(ys).most_common(1)[0][0] if ys else None
    support = [repeat_supported(c['rgb'],PY) for c in longest if PY and len(c['rgb'])>=2*PY]
    model['repeat_support'] = dict(period=PY, tested=len(support), agreeing=sum(support),
        status='insufficient evidence' if not support else 'supported' if sum(support)/len(support)>=.6 else 'contradictory evidence')
    residuals = [float((np.linalg.norm(c['rgb'][:-PY].astype(float)-c['rgb'][PY:].astype(float),axis=2)>=45).mean())
                 for c in longest if PY and len(c['rgb'])>=2*PY]
    model['measured_repeat_residual'] = float(np.median(residuals)) if residuals else None
    if not support or sum(support)/len(support) < .6:
        PY = None
    width = int(np.median([c['rgb'].shape[1] for c in cleaned]))
    xrep = Counter(xs).most_common(1)[0][0] if xs else None
    model['period'] = dict(y=PY, x=xrep, y_votes=len(ys), x_votes=len(xs))
    ref = max([c for c in cleaned if PY and c['rgb'].shape[0] >= PY] or cleaned,
              key=lambda c: c['rgb'].shape[0] * c['rgb'].shape[1])
    if PY is None:
        # No measurable vertical repeat: keep the cleanest real crops for quilting.
        model.update(kind='quilt', reason='No reliable vertical repeat; samples are quilted from real crops.',
                     agreement=None, anchored=False,repeat_verified=False,
                     validation_scope='Multi-window reconstruction only; long repeat remains unverified')
        # Quilt each source once into a long strip; layers take windows of it.
        strips = [];origins=[];native_boxes=[]
        jitter = None
        if ((model['measured_repeat_residual'] or 0)>.10 and
                any(word in norm(model['label']) for word in ('sand','silt'))):
            from . import render
            candidate = render.glyph_model(ref['rgb'],'jitter')
            if (candidate and len(candidate['glyphs'])>=6 and candidate['density']<.25 and
                    all(g.shape[0]<=5 and g.shape[1]<=5 and
                        max(g.shape[:2])/min(g.shape[:2])<=2 for g in candidate['glyphs'])):
                area = np.mean([np.sum(np.linalg.norm(g.astype(float)-candidate['bg'],axis=2)>35) for g in candidate['glyphs']])
                spacing = max(3,int(round(np.sqrt(area/max(candidate['density'],.001)))))
                candidate['spacing']=(spacing,spacing)
                jitter=candidate
                model.update(renderer='jittered lattice',jitter_pixels=[2,2],
                             glyph_count=len(jitter['glyphs']),
                             reason='Irregular compact dot fill: native glyphs on a lattice with 2px jitter; no reliable exact repeat.')
        for i, c in enumerate(cleaned[:3]):
            # Restore observed margins at their native x coordinates before
            # quilting. Cleaned interiors retain repairs; removed frame pixels
            # come from the same source, never an extended edge column.
            dl,dt=c['offset'];native=c['raw'][dt:dt+len(c['rgb'])].copy()
            native_box=list(c['ex']['box']);native_x0=native_box[0]
            lane=doc.get('boxes',{}).get('interpreted_lithology_column')
            if lane and (native_box[0]>lane[0] or native_box[2]<lane[2]):
                expanded=[lane[0],native_box[1],lane[2],native_box[3]]
                try:
                    full=crops.get(doc,expanded)
                    if pixels_allowed(full):
                        native=full[dt:dt+len(c['rgb'])].copy();native_x0=lane[0];native_box=expanded
                except ValueError as error:
                    model.setdefault('unsupported_source_margins',[]).append(dict(box=expanded,reason=str(error)))
            dl=c['x0']-native_x0
            native[:,dl:dl+c['rgb'].shape[1]]=c['rgb']
            if jitter:
                strip=native[np.arange(QUILT_ROWS)%len(native)].copy()
                strip[:,dl:dl+c['rgb'].shape[1]]=render.symbol_fill(jitter,c['rgb'].shape[1],QUILT_ROWS,np.random.default_rng(i))
                strips.append(strip)
            else:
                img, _ = generate(Image.fromarray(native), [0, 0, native.shape[1], native.shape[0]],
                                  width=native.shape[1], height=QUILT_ROWS, seed=i, variation=.5)
                strips.append(np.array(img))
            origins.append(native_x0);native_boxes.append(native_box)
        extent=[min(origins),max(x+s.shape[1] for x,s in zip(origins,strips))]
        model.update(strip_origins=origins,max_layer_height=QUILT_ROWS,native_extent=extent,
                     native_source_boxes=native_boxes,source_clean_offsets=[list(c['offset']) for c in cleaned[:3]])
        return model, dict(kind='quilt', strips=strips, bg=background(ref['rgb']),
                           strip_origins=origins,x_origin=extent[0],native_extent=extent,max_layer_height=QUILT_ROWS)
    if xrep:
        PX = xrep
        usable = [c for c in cleaned if c['rgb'].shape[1] >= PX]
    else:
        # No horizontal repeat inside the column: line the crops up by page
        # x and keep the page columns most of them share.
        X0, X1 = ref['x0'], ref['x0'] + ref['rgb'].shape[1]
        keep = [c for c in cleaned
                if min(X1, c['x0'] + c['rgb'].shape[1]) - max(X0, c['x0']) >= .85 * (X1 - X0)]
        X0 = max(c['x0'] for c in keep)
        X1 = min(c['x0'] + c['rgb'].shape[1] for c in keep)
        usable = [dict(c, rgb=c['rgb'][:, X0 - c['x0']:X1 - c['x0']], x0=X0) for c in keep]
        ref = next((c for c in usable if c['ex']['id'] == ref['ex']['id']), usable[0])
        PX = X1 - X0
    ref_tile = _block_tile(ref['rgb'], PY, PX).astype(np.int16)
    if (ref_tile.max(2)<DARK).mean() < .5*(ref['rgb'].max(2)<DARK).mean():
        native = (supported_vote(ref['rgb'].transpose(1,0,2)[:,:,None,:])
                  if PX==1 else ref['rgb'][:,:PX])
        start = (native.shape[0]-PY)//2
        ref_tile = native[start:start+PY].astype(np.int16)
    aligned, phases = [], []
    for c in usable:
        rgb = (c['rgb'][:, :PX] if not xrep else c['rgb'])[:6 * PY]
        h, w = rgb.shape[:2]
        probe = min(h, 2 * PY)
        if h < min(PY, 12):
            continue
        a = rgb.astype(np.int16)
        best = (float('inf'), 0, 0)
        for dy in range(PY):
            rows = ref_tile[(np.arange(probe) + dy) % PY]
            for dx in (range(PX) if xrep else (0,)):
                err = float(np.abs(rows[:, (np.arange(w) + dx) % PX] - a[:probe]).mean())
                if err < best[0]:
                    best = (err, dy, dx)
        err, dy, dx = best
        if err > 40:          # a different pattern hiding under this label
            model['suspects'].append(dict(id=c['ex']['id'], box=c['ex']['box'],
                                          reason='symbols do not match the other boxes'))
            model['suspect_count'] += 1
            continue
        # every real row votes for the tile row it lands on
        votes = np.full((h // PY + 2, PY, PX, 3), np.nan)
        ty = (np.arange(h) + dy) % PY
        tx = (np.arange(w) + dx) % PX
        lap = (np.arange(h) + dy) // PY
        for r in range(h):
            votes[lap[r], ty[r], tx] = a[r]
        aligned.extend(votes)
        phases.append(((c['y0'] - dy) % PY, (c['x0'] - dx) % PX))
        if len(phases) >= MAX_SOURCES:
            break
    if not aligned:
        aligned, phases = [ref_tile.astype(float)], [(ref['y0'] % PY, ref['x0'] % PX)]
    tile = supported_vote(np.stack(aligned), ref_tile)
    if (tile.max(2)<DARK).mean() < .5*(ref_tile.max(2)<DARK).mean():
        tile = ref_tile.astype(np.uint8)  # do not vote away the observed symbols
    agree = []
    for v in aligned:
        m = ~np.isnan(v).any(2)
        if m.any():
            agree.append(float((np.sqrt(((v - tile) ** 2).sum(2))[m] < 45).mean()))
    agree = agree or [1.0]
    voters = len(phases)
    phase, phase_votes = Counter(phases).most_common(1)[0]
    anchored = voters >= 2 and phase_votes / voters >= .6
    model.update(kind='tile', tile_size=[PX, PY], agreement=round(float(np.mean(agree)), 3),
                 voters=voters, anchored=anchored, phase=list(map(int, phase)),
                 reason=('Tile voted from %d real crops; page-anchored phase.' if anchored else
                         'Tile voted from %d real crops; phase varies per layer.') % voters)
    model['anchored_x'] = not bool(xrep) or anchored
    model['anchored_y'] = anchored
    model.update(x_repeat=bool(xrep),x_origin=ref['x0'])
    return model, dict(kind='tile', tile=tile, phase=phase, anchored=anchored,
                       anchored_x=not bool(xrep) or anchored, anchored_y=anchored,
                       x_origin=ref['x0'], x_repeat=bool(xrep), bg=background(tile))


def rails(crops, docs, doc_id, examples):
    """Learn the doc's column frame from the commonest box x-range.

    Pixel columns dark in >=90% of every approved crop are frame rails, not
    pattern. Returns (x0, x1, rail_mask, rail_rgb) or None.
    """
    doc = docs[doc_id]['doc']
    spans = Counter((e['box'][0], e['box'][2]) for e in examples)
    (x0, x1), _ = spans.most_common(1)[0]
    same = [e for e in examples if (e['box'][0], e['box'][2]) == (x0, x1)]
    same = sorted(same, key=lambda e: -(e['box'][3] - e['box'][1]))[:20]
    fracs, colour = [], []
    for e in same:
        rgb = crops.get(doc, e['box'])
        d = rgb.max(2) < DARK
        fracs.append(d.mean(0))
        colour.append(np.median(rgb.reshape(-1, 3)[d.reshape(-1)], axis=0) if d.any() else np.zeros(3))
    frac = np.min(np.stack(fracs), axis=0) if len(fracs) >= 2 else fracs[0]
    mask = frac > .9
    return dict(x0=int(x0), x1=int(x1), mask=mask.tolist(), ink=np.median(np.stack(colour), 0).astype(int).tolist())


def signature(examples):
    h = hashlib.sha256(json.dumps(sorted((e['id'], e['box'],e.get('annotation_version'),e.get('label')) for e in examples)).encode())
    return h.hexdigest()[:16]
