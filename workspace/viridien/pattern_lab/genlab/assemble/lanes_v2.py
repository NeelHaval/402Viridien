"""GenerationRequest adapter and CLI. Generates source pages only, never a release.

Packets declare complete lineage before any supervised asset is opened. Inputs
are supplied via GENLAB_GENERATOR_INPUTS or --inputs; there is no implicit base.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image

from . import training_v2
from .safety_v2 import file_hash, cache_binding, cache_valid, seal_cache, spans
from .native_v2 import NativePage, EmpiricalSequence, BoundWindows, write

ROOT=Path(__file__).resolve().parents[2]
SCRATCH=ROOT/'data/genlab/scratch/generators'
FAMILIES=('geo','geoscale','geoseq','gulf-legacy','gulf','gulf_texas','gtrel','mc765','oneoff','mc761')


def local(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing/escaping packet input')
    return path


def release_api():
    """Use the parallel release lane's canonical contract without editing it."""
    path=ROOT/'data/genlab/release/v2'
    sys.path.insert(0,str(path))
    try:
        # Standalone generator entry points use no legacy lane module names.
        return importlib.import_module('common'),importlib.import_module('contract')
    finally:sys.path.pop(0)


def load_packet(folder,request):
    header=json.loads((folder/'generator-inputs-v2.json').read_text(encoding='utf8'))
    if header.get('schema')!='generator-inputs-v2' or header.get('base_doc')!=request.base_doc:
        raise ValueError('Missing versioned input packet or base disagreement')
    lineage=header['lineage']
    if set(lineage)!={'base','texture','sequence','calibration'} or lineage['base']!=[request.base_doc]:
        raise ValueError('Complete base/texture/sequence/calibration lineage required')
    ancestors=sorted(set(sum(lineage.values(),[])))
    if any(not lineage[k] for k in lineage):raise ValueError('Missing role ancestry')
    groups=training_v2.check_lineage(lineage)
    wells=sorted(set(groups.values()))
    # All following input opens occur after policy-only ancestry checks.
    for name,sha in header['inputs'].items():
        if file_hash(local(folder,name))!=sha:raise ValueError('Packet input changed: '+name)
    def read(name):
        if name not in header['inputs']:raise ValueError('Unbound packet input: '+name)
        return json.loads(local(folder,name).read_text(encoding='utf8'))
    doc=read('document.json');anchor=read('anchors.json');mapping=dict(anchor['calibration'])
    points=anchor['independent_anchors']
    if (len(points)<2 or mapping.get('rows_per_ft',0)<=0 or
        not np.isfinite([mapping['rows_per_ft'],mapping['page_intercept'],*mapping['valid_page_span']]).all() or
        not all(np.isfinite([p['page_y'],p['depth_ft']]).all() for p in points)):
        raise ValueError('Verified finite ruler anchors required')
    if any(abs((p['page_y']-mapping['page_intercept'])/mapping['rows_per_ft']-p['depth_ft'])>anchor['tolerance_ft'] for p in points):
        raise ValueError('Ruler anchor residual failed')
    mapping['confidence']='verified'
    if doc['id']!=request.base_doc:raise ValueError('Document identity changed')
    if set(doc['boxes'])!={'depth_column','interpreted_lithology_column','lithology_legend'}:
        raise ValueError('Three explicit native boxes required')
    exclusions=read('exclusions-v2.json')
    signature=__import__('hashlib').sha256(json.dumps(json.loads(
        (ROOT/'data/normalized/split-policy.json').read_text(encoding='utf8')),sort_keys=True).encode()).hexdigest()
    if exclusions.get('schema')!='generator-exclusions-v2' or exclusions.get('policy_sha256')!=signature:
        raise ValueError('Unbound label-free exclusions')
    global_spans=exclusions['reserved_spans']
    reserved=spans(header['reserved_spans']+global_spans.get(request.base_doc,[]))
    no_data=spans(header['native_no_data_spans'])
    doc.update(reserved_spans=reserved,native_no_data_spans=no_data)
    x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column'];width=x1-x0
    if doc['width']*512*3>64*1024**2 or (y1-y0)*width*30>256*1024**2:
        raise ValueError('Requested geometry exceeds bounded generator memory budget')
    records=training_v2.load_export(local(folder,'training-v2.json'))['records']
    active={r['sample_id']:r for r in records if r.get('usable_for_generation') and not r.get('quality_issues')}
    bank=[];byclass=defaultdict(list);class_ids={};class_indices={}
    denied=set(exclusions['denied_pixel_sha256'])
    for meta in read('source-bank.json')['sources']:
        did=meta['document_id'];training_v2.require_training_id(did)
        if did not in lineage['texture']:raise ValueError('Undeclared texture source')
        box=meta['box_xyxy'];row=active.get(meta['sample_id'])
        if not row or row['box_xyxy']!=box or row['pixel_sha256']!=meta['pixel_sha256']:
            raise ValueError('Revoked or changed training texture')
        donor_reserved=reserved if did==doc['id'] else spans(global_spans.get(did,[]))
        if any(box[1]<b and box[3]>a for a,b in donor_reserved):
            raise ValueError('Reserved donor image span')
        for denial in exclusions['denied_boxes']:
            x,y,z,w=denial['box_xyxy']
            if denial['document_id']==did and max(x,box[0])<min(z,box[2]) and max(y,box[1])<min(w,box[3]):
                raise ValueError('Quarantined donor image box')
        with Image.open(local(folder,meta['file'])) as im:
            if im.width*im.height*3>16*1024**2:raise ValueError('Donor exceeds bounded crop budget')
            rgb=np.asarray(im.convert('RGB')).copy()
        sha=__import__('hashlib').sha256(f'{rgb.shape[1]}x{len(rgb)}:'.encode()+rgb.tobytes()).hexdigest()
        if sha!=meta['pixel_sha256'] or sha in denied:raise ValueError('Texture pixels changed/quarantined')
        if rgb.shape[1]!=width or len(rgb)<16:raise ValueError('Native donor geometry unsupported; never resize')
        name=meta['label'];class_ids[name]=meta['class_id'];class_indices[name]=meta['index']
        byclass[name].append(len(bank));bank.append(dict(meta=meta,rgb=rgb))
        if sum(b['rgb'].nbytes for b in bank)>256*1024**2:raise ValueError('Appearance bank exceeds memory budget')
    beds=read('fit-beds.json')
    for bed in beds:
        did=bed['document_id'];training_v2.require_training_id(did)
        if did not in lineage['sequence']:raise ValueError('Undeclared sequence ancestor')
        if did!=doc['id']:raise ValueError('Cross-document row duration pooling requires verified coordinate conversion')
        if any(bed['top']<b and bed['bottom']>a for a,b in reserved):raise ValueError('Reserved sequence evidence')
    required=header['required_classes']
    if set(required)-set(byclass):raise ValueError('Missing required class appearance support')
    e=SimpleNamespace(doc=doc,mapping=mapping,rate=mapping['rows_per_ft'],bank=bank,byclass=byclass,
                      class_ids=class_ids,class_indices=class_indices,fit_beds=beds,
                      required_classes=required,sequence_tier=header['sequence_tier'],
                      fit_spans=header['fit_spans'],reserved_spans=reserved,rails=header.get('rails',[]),
                      ancestors=ancestors,ancestor_wells=wells,lineage=lineage,
                      family=request.family,well=groups[request.base_doc],output_dir=request.out)
    return header,e


def generate(request):
    """generate(GenerationRequest) -> list[Path], exactly n canonical folders."""
    if request.family not in FAMILIES or request.profile!='native':raise ValueError('Unsupported family/profile')
    if request.family in ('mc765','mc761'):
        raise ValueError('Protected historical family: generation/registration withdrawn (ASTRA-01)')
    if request.family=='geoscale':
        raise ValueError('Historical scale-v5 model not parameterized; use explicit geo/geoseq hierarchy')
    if request.seed<0:raise ValueError('Seed must be nonnegative')
    if not 1<=request.n<=2:raise ValueError('Generator smoke work is limited to 1–2 pages per family')
    training_v2.require_training_id(request.base_doc)
    output=Path(request.out).resolve();cache=Path(request.cache).resolve()
    if not output.is_relative_to(SCRATCH.resolve()) or not cache.is_relative_to(SCRATCH.resolve()):
        raise ValueError('This lane generates only under scratch/generators')
    source=os.environ.get('GENLAB_GENERATOR_INPUTS')
    if not source:raise ValueError('Set GENLAB_GENERATOR_INPUTS to a reviewed training-only v2 packet')
    folder=Path(source).resolve();header,e=load_packet(folder,request)
    os.environ['GENLAB_TRAINING_EXPORT']=str(local(folder,'training-v2.json'))
    os.environ['GENLAB_EXCLUSION_INDEX']=str(local(folder,'exclusions-v2.json'))
    common,contract=release_api()
    output.mkdir(parents=True,exist_ok=True);e.output_dir=output
    if request.family in ('geo','geoscale','geoseq'):
        from data.genlab.fasttrack.geoseq.hierarchy import Hierarchy
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):model=Hierarchy(e,**header.get('model_parameters',{}))
    else:model=EmpiricalSequence(e,request.family)
    source_image=local(folder,header['base_image'])
    windows=BoundWindows(e,cache,source_image)
    # The decoder resolves the actual native dataset image; bind it explicitly.
    import model as document_model
    actual=document_model.DATASET/'images'/e.doc['image_name']
    if file_hash(actual)!=file_hash(source_image):raise ValueError('Packet/native base image mismatch')
    adapter=NativePage(e,model,windows);result=[]
    renderer=[Path(__file__),Path(__file__).with_name('native_v2.py'),Path(__file__).with_name('compose.py'),Path(__file__).with_name('safety_v2.py'),Path(__file__).with_name('training_v2.py')]
    renderer += [ROOT/'data/genlab/fasttrack/geoseq'/n for n in ('hierarchy.py','render.py')]
    renderer += list((ROOT/'data/genlab/release/v2').glob('*.py'))
    binding=cache_binding([folder/'generator-inputs-v2.json',training_v2.PROTECTED_REGISTRY,
                           ROOT/'data/normalized/split-policy.json']+[local(folder,n) for n in header['inputs']],
                          model.description,dict(family=request.family,seed=request.seed,base=request.base_doc,profile=request.profile),renderer)
    try:
        for i in range(request.n):
            path=output/f'page-{i:03d}'
            page_binding=dict(binding,page_seed=request.seed+i,page_index=i)
            if not cache_valid(path,page_binding):
                adapter.page(request.base_doc,request.seed+i,path)
                finalize(path,folder,header,e,model,common,contract)
                seal_cache(path,page_binding)
            contract.validate_page_folder(path)
            result.append(path)
    finally:windows.close()
    return result


def generate_family(request, *families):
    if request.family not in families:
        raise ValueError('Family does not match this generator entry point')
    return generate(request)


def finalize(path,packet,header,e,model,common,contract):
    """Finish the parallel release lane's canonical source-folder contract."""
    truth=json.loads((path/'truth.json').read_text(encoding='utf8'))
    truth['lineage']=header['lineage']
    truth.update(training_v2.lineage_eligibility(header['lineage']))
    if not truth['trainingEligible']:
        raise ValueError('Ineligible ancestry at finalization')
    anchor=json.loads(local(packet,'anchors.json').read_text(encoding='utf8'))
    write(path/'inputs/anchors.json',anchor)
    legend=json.loads(local(packet,'legend.json').read_text(encoding='utf8'))
    write(path/'inputs/legend.json',legend)
    bank=json.loads(local(packet,'source-bank.json').read_text(encoding='utf8'))
    donor_bindings=[]
    for i,source in enumerate(bank['sources']):
        donor_bindings.append(dict(source_index=i,document_id=source['document_id'],
             file=common.reference(local(packet,source['file'])),box=source['box_xyxy']))
        target='inputs/textures/'+Path(source['file']).name
        (path/target).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(local(packet,source['file']),path/target)
        source['file']=target
    write(path/'inputs/source-bank.json',bank)
    write(path/'inputs/pixel-donors.json',donor_bindings)
    truth['depth_mapping']=dict(anchor['calibration'],confidence='verified',coordinate=anchor.get('coordinate','MD'),
                                anchors=anchor['independent_anchors'],ruler_sha256=file_hash(path/'inputs/anchors.json'))
    truth['context_policy']='Independent column only; reserved rows redacted across every track; cross-track tasks unavailable'
    write(path/'truth.json',truth)
    bands=json.loads((path/'page-bands.json').read_text(encoding='utf8'))
    for band in bands['bands']:band['sha256']=file_hash(path/band['file'])
    write(path/'page-bands.json',bands)
    # Check native donor coordinates against final RGB, including restoration.
    with Image.open(path/'column.png') as im:column=np.asarray(im.convert('RGB')).copy()
    with Image.open(path/'mask.png') as im:mask=np.asarray(im).copy()
    with np.load(path/'pixel-sources.npz') as p:ids,sy,sx=[p[k] for k in ('source_index','page_y','page_x')]
    errors=0
    for i,source in enumerate(e.bank):
        take=ids==i;box=source['meta']['box_xyxy']
        errors+=int(np.any(column[take]!=source['rgb'][sy[take]-box[1],sx[take]-box[0]],axis=1).sum())
    if errors:raise ValueError('Native RGB ancestry mismatch')
    bb=json.loads((path/'bbox_annotation.json').read_text(encoding='utf8'))
    cursor=0
    for band in bands['bands']:
        if band['top']!=cursor:raise ValueError('Noncontiguous output bands')
        with Image.open(path/band['file']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        x0,y0,x1,y1=e.doc['boxes']['interpreted_lithology_column']
        a,b=max(y0,band['top']),min(y1,band['bottom'])
        if a<b and not np.array_equal(rgb[a-band['top']:b-band['top'],x0:x1],column[a-y0:b-y0]):
            raise ValueError('Page/column embedding mismatch')
        for lo,hi in e.reserved_spans:
            lo,hi=max(lo,band['top']),min(hi,band['bottom'])
            if lo<hi and not np.all(rgb[lo-band['top']:hi-band['top']]==255):raise ValueError('Reserved image exposure')
        cursor=band['bottom']
    if cursor!=e.doc['height']:raise ValueError('Missing output bands')
    # Preserve construction evidence for the release lane's independent replay.
    shutil.copyfile(path/'truth.json',path/'inputs/original-truth.json')
    shutil.copyfile(path/'mask.png',path/'inputs/original-mask.png')
    write(path/'inputs/original-audit.json',dict(status='pass',errors=errors,
          checks=['native donor RGB reconstruction','page-column embedding','full-width reservation redaction']))
    from tiffio import header as base_header
    write(path/'inputs/base-header.json',base_header(local(packet,header['base_image'])))
    descriptor=dict(header['descriptor'],blank_no_data_spans=e.doc['native_no_data_spans'])
    external={common.reference(local(packet,name)):sha for name,sha in header['inputs'].items()}
    external.update({common.reference(p):file_hash(p) for p in
        [Path(__file__),Path(__file__).with_name('native_v2.py'),Path(__file__).with_name('compose.py'),
         ROOT/'data/normalized/split-policy.json',training_v2.PROTECTED_REGISTRY,
         Path(__file__).with_name('training_v2.py'),ROOT/'data/genlab/fasttrack/geoseq/hierarchy.py']})
    binding=dict(schema='genlab.page-binding.v1',base_doc=e.doc['id'],lineage=header['lineage'],
                 base=dict(document_id=e.doc['id'],group_id=training_v2.require_training_id(e.doc['id']),split='train',
                           revision=e.doc['revision'],image_sha256=file_hash(local(packet,header['base_image'])),
                           annotation_sha256=file_hash(local(packet,'document.json'))),
                 legend=legend['legend'],output_profile='rgb-lzw',source_bank='inputs/source-bank.json',
                 pixel_provenance='pixel-sources.npz',descriptor=descriptor,model_sha256=common.digest(model.description),
                 split_policy_sha256=file_hash(ROOT/'data/normalized/split-policy.json'),code=common.code_hashes(),
                 external_inputs=external,fold=header.get('fold'),reserved_spans=e.reserved_spans,
                 exposure=truth['exposure'],inputs={p.relative_to(path).as_posix():file_hash(p)
                    for p in sorted(path.rglob('*')) if p.is_file() and p.name not in ('cache-v2.json','page-binding.json','page-audit.json')})
    required={'truth.json','bbox_annotation.json','mask.png','row-mask.npy','page-bands.json',
              'inputs/original-truth.json','inputs/original-mask.png','inputs/original-audit.json',
              'inputs/anchors.json','inputs/legend.json','inputs/base-header.json','inputs/pixel-donors.json',
              binding['source_bank'],binding['pixel_provenance']}
    required.update(b['file'] for b in bands['bands'])
    missing=required-set(binding['inputs'])
    if missing:raise ValueError('Generator handoff missing inputs: '+str(sorted(missing)))
    binding['input_root_sha256']=contract.input_binding(path,binding)
    write(path/'page-binding.json',binding)
    write(path/'page-audit.json',dict(status='pass',errors=0,input_root_sha256=binding['input_root_sha256'],
                                     auditor_sha256=file_hash(Path(__file__))))
    contract.validate_page_folder(path)
    if not np.any(mask!=255):raise ValueError('No nonempty supervised page; generation refused')


def main(family=None):
    p=argparse.ArgumentParser(description=__doc__)
    if family is None:p.add_argument('--family',required=True,choices=FAMILIES)
    p.add_argument('--n',type=int,default=1);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--base-doc',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--profile',default='native')
    args=p.parse_args();os.environ['GENLAB_GENERATOR_INPUTS']=str(args.inputs.resolve())
    request=SimpleNamespace(family=family or args.family,n=args.n,seed=args.seed,base_doc=args.base_doc,
                            out=args.output,cache=args.cache_dir,profile=args.profile)
    for path in generate(request):print(path)


if __name__=='__main__':main()
