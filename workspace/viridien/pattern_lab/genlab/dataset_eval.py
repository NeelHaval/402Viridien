"""Fast CPU evaluation with explicit sample counts and human-only truth."""
from __future__ import annotations

import importlib.util
import json
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import score
from .build import _write, ratings
from .dataset import MIN_N, assert_truth, read, digest, ROOT, pixel_hash


def eligibility_context(folder):
    """Resolve current policy for all archived documents, even unlabelled wells."""
    from . import paths
    from .dataset import well_split
    policy = read(paths.DATA/'normalized/split-policy.json',{})
    index = read(paths.DATA/'normalized/current.json',{})
    classes = read(Path(folder)/'stats.json',{}).get('classes',[])
    names = {r['doc']:r.get('document') for r in classes}
    wells = read(Path(folder)/'wells.json',{})
    splits = {}
    store = None
    for did, archived in wells.items():
        name = names.get(did)
        if not name and did not in policy.get('document_groups',{}):
            from .dataset import labstore
            if store is None:
                store = labstore.Store()
            try:
                name = store.get(did)['name']
            except (KeyError,FileNotFoundError):
                pass
        if not policy.get('seed') or (not name and did not in policy.get('document_groups',{})):
            splits[did] = 'unknown'
            continue
        group, split = well_split(dict(id=did,name=name or did),policy)
        # Historical dependent-well promotions can only make a split stricter.
        if split!='test' and archived.get('split')=='validation':
            split = 'validation'
        splits[did] = split
    records = {r['sample_id']:r for r in index.get('records',[])}
    stamps = [(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in
              (paths.DATA/'normalized/split-policy.json',paths.DATA/'normalized/current.json')]
    return dict(policy=policy,policy_sha256=digest(policy),splits=splits,records=records,
        archived_policy=read(Path(folder)/'eligibility.json',{}).get('policy_sha256'),source_stamps=stamps,
        snapshot=index.get('snapshot'),
        forbidden={r['pixel_sha256'] for r in records.values()
                   if r['split']=='test' or not r.get('usable_for_generation') or r.get('quality_issues')})


def source_errors(row, context, real=False):
    """Validate source identity before opening any input image."""
    errors = []
    did = row.get('doc')
    split = context['splits'].get(did,'unknown')
    if split in ('test','unknown') or row.get('split')!=split:
        errors.append('current-document-split:'+str(did))
    ancestors = set(row.get('ancestor_docs',[did] if real else []))
    refs = []
    def declared(value):
        if isinstance(value,dict):
            refs.extend(value.get('normalized_sources',[]))
            if value is not row and value.get('sample_id') and value.get('annotation_version'):
                refs.append(value)
            ancestors.update(value.get('ancestor_docs',[]))
            for field in ('doc','doc_id','document_id','source_doc','source_doc_id','real_doc_id'):
                if value.get(field):
                    ancestors.add(value[field])
            if isinstance(value.get('donor'),str):
                ancestors.add(value['donor'])
            for child in value.values():
                if isinstance(child,(dict,list)):
                    declared(child)
        elif isinstance(value,list):
            for child in value:
                declared(child)
    declared(row)
    legend = row.get('tile_provenance',{}).get('kind')=='legend-swatch-only'
    if not ancestors:
        errors.append('unknown-ancestry:'+str(did))
    for ancestor in ancestors:
        allowed = context['splits'].get(ancestor)=='train' or (legend and ancestor==did and split=='validation')
        if real and ancestor==did and split=='validation':
            allowed = True
        if not allowed:
            errors.append('held-ancestor:'+str(ancestor))
    if real or row.get('source_pixel_sha256'):
        refs.append(row)
    for ref in refs:
        current = context['records'].get(ref.get('sample_id'))
        if (not current or not current.get('usable_for_generation') or current.get('quality_issues')
                or current.get('split')=='test' or context['splits'].get(current.get('document_id')) in ('test','unknown')
                or current.get('annotation_version')!=ref.get('annotation_version')
                or (ref.get('legend_version') and current.get('legend_version')!=ref['legend_version'])
                or (ref.get('source_pixel_sha256') and current.get('pixel_sha256')!=ref['source_pixel_sha256'])
                or current.get('pixel_sha256') in context['forbidden']):
            errors.append('quarantined-or-changed-source:'+str(ref.get('sample_id')))
        elif real and (current.get('document_id')!=did or current.get('annotation_id')!=row.get('annotation_id')
                or current.get('box_xyxy')!=row.get('source_box',row.get('box'))):
            errors.append('source-identity:'+str(ref.get('sample_id')))
    return errors


def require_current_policy(folder,context):
    archived = context['archived_policy']
    if archived!=context['policy_sha256']:
        raise ValueError('Release current split policy digest changed')
    for path,mtime,size in context['source_stamps']:
        st = Path(path).stat()
        if (st.st_mtime_ns,st.st_size)!=(mtime,size):
            refreshed = eligibility_context(folder)
            if refreshed['policy_sha256']!=context['policy_sha256']:
                raise ValueError('Current split policy changed during image extraction')
            # Refresh before source_errors/decoded-hash checks. Unrelated human
            # additions must not cancel a frozen cohort's evaluation.
            context.clear()
            context.update(refreshed)
            break


def checked_rgb(folder, row, context=None):
    if context is not None:
        require_current_policy(folder,context)
        errors = source_errors(row,context,real=row.get('kind')=='real')
        if errors:
            raise ValueError('Unsafe image source: '+','.join(errors))
    with Image.open(Path(folder)/row['file']) as image:
        if image.format!='PNG' or image.width*image.height>16_000_000:
            raise ValueError('Input must be a bounded PNG crop: '+row['file'])
        rgb = np.asarray(image.convert('RGB')).copy()
    actual = pixel_hash(rgb)
    if not row.get('pixel_sha256') or actual!=row['pixel_sha256']:
        raise ValueError('Image pixel hash mismatch: '+row['file'])
    if context and actual in context['forbidden']:
        raise ValueError('Quarantined image pixels: '+row['file'])
    box = row.get('crop_box')
    if box and [rgb.shape[1],rgb.shape[0]]!=[box[2]-box[0],box[3]-box[1]]:
        raise ValueError('Image crop geometry mismatch: '+row['file'])
    return rgb


def implementation_hash(function):
    """Hash loaded executable code, excluding source positions and debug lines."""
    import hashlib,marshal,types
    def normalized(code):
        return code.replace(co_filename='',co_firstlineno=1,co_linetable=b'',
            co_consts=tuple(normalized(v) if isinstance(v,types.CodeType) else v for v in code.co_consts))
    return digest([hashlib.sha256(marshal.dumps(normalized(function.__code__))).hexdigest(),
                   function.__defaults__,function.__kwdefaults__])


def pair_dependency(folder, reals, context):
    """Bind the complete requested cohort and all rendering/augmentation inputs."""
    import hashlib
    from . import learn,render,realism,candidates
    folder = Path(folder)
    stats = read(folder/'stats.json',{})
    nuisance = read(folder/'nuisance.json',{})
    measurements = read(folder/'realism.json',{})
    for row in stats.get('classes',[])+nuisance.get('components',[])+measurements.get('measurements',[]):
        errors = source_errors(row,context)
        if errors:
            raise ValueError('Unsafe cache dependency: '+','.join(errors))
    assets = set()
    for row in stats.get('classes',[])+nuisance.get('components',[]):
        for key in ('model_file','quilt_file','mask','colours'):
            if row.get(key):
                assets.add(row[key])
        if row.get('render_model'):
            assets.add(row['render_model']['asset']['path'])
    hashes = []
    for rel in sorted(assets):
        with (folder/rel).open('rb') as source:
            h = hashlib.file_digest(source,'sha256').hexdigest()
        hashes.append([rel,h])
    code = {Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (render,realism)}
    # Fitting/proposal changes cannot alter a frozen model bank. Bind the exact
    # decoders used here, including the metadata-binding helper they call.
    code.update({name:implementation_hash(function) for name,function in
        [('learn.load_model',learn.load_model),('learn.model_binding',learn.model_binding),
         ('candidates.load_data',candidates.load_data)]})
    from . import dataset
    for name,function in [('dataset_augment',dataset.dataset_augment),('local_model',dataset.local_model),
        ('save_rgb',dataset.save_rgb),('pixel_hash',dataset.pixel_hash),('digest',dataset.digest),
        ('frame_for',dataset.DatasetBuilder.frame_for),('nuisance_for',dataset.DatasetBuilder.nuisance_for),
        ('overlay',dataset.DatasetBuilder.overlay)]:
        code['dataset.'+name] = implementation_hash(function)
    code['dataset_eval.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    references = {r['sample_id'] for r in reals}
    for row in stats.get('classes',[])+measurements.get('measurements',[]):
        references.update(r['sample_id'] for r in row.get('normalized_sources',[]))
        if row.get('sample_id'):
            references.add(row['sample_id'])
    fields = ('sample_id','document_id','annotation_id','annotation_version','legend_version',
              'box_xyxy','split','quality','quality_issues','usable_for_generation','pixel_sha256')
    sources = [{key:context['records'][sid].get(key) for key in fields} for sid in sorted(references)]
    return digest([sorted(reals,key=lambda r:(r['doc'],r['annotation_id'])),stats.get('classes'),
        read(folder/'realism.json'),nuisance,hashes,code,context['policy_sha256'],
        sources,sorted(context['forbidden'])])


def validate_pairs(folder, saved, reals, context, pixels=True):
    lookup = {(r['doc'],r['annotation_id']):r for r in reals}
    if len(lookup)!=len(reals):
        raise ValueError('Duplicate requested real identity')
    seen = set()
    for pair in saved.get('pairs',[]):
        real = pair['real']
        key = (real['doc'],real['annotation_id'])
        if key in seen or key not in lookup or digest(real)!=digest(lookup[key]):
            raise ValueError('Discriminator cache real cohort/source identity mismatch')
        seen.add(key)
        for variant in ('real','clean','augmented'):
            row = pair[variant]
            errors = source_errors(row,context,real=variant=='real')
            if errors:
                raise ValueError('Unsafe discriminator cache: '+','.join(errors))
            if row['doc']!=real['doc'] or row['class_id']!=real['class_id'] or row['split']!=real['split']:
                raise ValueError('Discriminator pair class/document/split mismatch')
            if pixels:
                rgb = checked_rgb(folder,row,context)
                if [rgb.shape[1],rgb.shape[0]]!=pair['native_size']:
                    raise ValueError('Discriminator native size mismatch')
    missing = {(r['doc'],r['annotation_id']) for r in saved.get('unavailable',[])}
    if seen & missing or seen | missing != lookup.keys():
        raise ValueError('Discriminator cache requested cohort mismatch')


def current_test_hashes():
    from . import paths
    return {r['pixel_sha256'] for r in read(paths.DATA/'normalized/current.json',{}).get('records',[])
            if r['split']=='test' or not r.get('usable_for_generation') or r.get('quality_issues')}


def leak_audit(folder):
    """Audit truth, fitted residuals and transitive ancestors, including archives."""
    from . import paths
    folder = Path(folder)
    current = read(paths.DATA/'normalized/current.json',{})
    records = {r['sample_id']:r for r in current.get('records',[])}
    by_annotation = {(r['document_id'],r['annotation_id']):r for r in records.values()}
    test_hashes = {r['pixel_sha256'] for r in records.values() if r['split']=='test'}
    eligibility = read(folder/'eligibility.json',{})
    wells = read(folder/'wells.json',{})
    reals = read(folder/'real.json',[])
    measurements = read(folder/'realism.json',{}).get('measurements',[])
    manifest = [json.loads(s) for s in (folder/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    errors = []
    context = eligibility_context(folder)
    archived_policy = eligibility.get('policy_sha256') or digest(read(folder/'split-policy.json',{}))
    if archived_policy!=context['policy_sha256']:
        errors.append('policy:current-policy-digest-changed')
    for kind, rows in [('real',reals),('measurement',measurements)]:
        for row in rows:
            source = records.get(row.get('sample_id')) or by_annotation.get((row['doc'],row.get('annotation_id')))
            if not source or not source.get('usable_for_generation') or source.get('quality_issues'):
                errors.append(f'{kind}:quarantined-or-unknown:{row.get("sample_id")}')
                continue
            if source['annotation_version'] != row.get('annotation_version'):
                errors.append(f'{kind}:stale-version:{row.get("sample_id")}')
            if row.get('source_pixel_sha256') != source['pixel_sha256'] or source['pixel_sha256'] in test_hashes:
                errors.append(f'{kind}:test-or-changed-pixels:{row.get("sample_id")}')
            if kind=='measurement' and row['split']!='train':
                errors.append('measurement:held-out')
    classes = read(folder/'stats.json',{}).get('classes',[])
    for row in classes + measurements:
        errors.extend('model:'+error for error in source_errors(row,context))
        ancestors = row.get('ancestor_docs')
        legend = row.get('tile_provenance',{}).get('kind')=='legend-swatch-only'
        if not ancestors:
            errors.append('model:unknown-ancestry:'+row.get('class_id','?'))
        for did in ancestors or []:
            split = wells.get(did,{}).get('split')
            if split=='test' or (split!='train' and not (legend and did==row['doc'])):
                errors.append('model:held-ancestor:'+did)
        for ref in row.get('normalized_sources',[]):
            source = records.get(ref['sample_id'])
            if (not source or not source.get('usable_for_generation') or source.get('quality_issues')
                    or source['annotation_version']!=ref['annotation_version']
                    or ref.get('source_pixel_sha256',source['pixel_sha256'])!=source['pixel_sha256']):
                errors.append('model:revoked-source:'+ref['sample_id'])
        if row.get('model_file') and not source_errors(row,context):
            with Image.open(folder/row['model_file']) as image:
                rgb = np.asarray(image.convert('RGB'))
            if pixel_hash(rgb) in test_hashes:
                errors.append('model:test-identical-tile:'+row['class_id'])
            if row.get('model_pixel_sha256') and pixel_hash(rgb)!=row['model_pixel_sha256']:
                errors.append('model:pixel-hash-mismatch:'+row['class_id'])
            if row.get('render_model'):
                from . import learn
                try:
                    learn.load_model(row['render_model'],root=folder,signature=row['signature'])
                except (ValueError,OSError,KeyError) as error:
                    errors.append('model:invalid-full-model:'+str(error))
    for row in manifest:
        errors.extend('synthetic:'+error for error in source_errors(row,context))
        if row.get('pixel_sha256') in test_hashes:
            errors.append('synthetic:test-identical:'+row['sample_id'])
        if row['split']=='test' or wells.get(row['doc'],{}).get('split')=='test':
            errors.append('synthetic:test-document')
        if row.get('format')=='finite-legend' or row.get('kind')=='reference':
            errors.append('synthetic:finite-reference-in-training-manifest')
    support = read(folder/'eval-swatches.json',[])
    for row in support:
        errors.extend('reference:'+error for error in source_errors(row,context))
        if row.get('format')!='finite-legend':
            continue
        p = row.get('tile_provenance',{})
        if (row.get('synthesis_eligible') or row.get('renderable') or not row.get('recognition_eligible')
                or row.get('file')!=p.get('raw_file') or p.get('real_crop_inputs')
                or p.get('kind')!='legend-swatch-only' or row.get('split')!='validation'):
            errors.append('reference:unsafe-contract:'+row['class_id'])
        if source_errors(row,context):
            continue
        with Image.open(folder/row['file']) as image:
            rgb = np.asarray(image.convert('RGB'))
        ph = pixel_hash(rgb)
        if ph!=row.get('pixel_sha256') or ph!=p.get('raw_pixel_sha256') or ph in context['forbidden']:
            errors.append('reference:changed-or-quarantined-pixels:'+row['class_id'])
    nuisance = read(folder/'nuisance.json',{}).get('components',[])
    for row in nuisance:
        errors.extend('nuisance:'+error for error in source_errors(row,context))
        if row['split']!='train' or any(wells.get(d,{}).get('split')!='train' for d in row['ancestor_docs']):
            errors.append('nuisance:held-ancestor')
    for row in nuisance+measurements:
        if source_errors(row,context):
            continue
        for field in ('mask','colours'):
            if not row.get(field+'_pixel_sha256'):
                continue
            try:
                with Image.open(folder/row[field]) as im:
                    rgb = np.asarray(im if field=='mask' else im.convert('RGB'))
                    if pixel_hash(rgb)!=row[field+'_pixel_sha256']:
                        errors.append('nuisance:'+field+'-pixel-hash-mismatch:'+row['id'])
            except OSError as error:
                errors.append('nuisance:'+str(error))
    mismatches, checked = 0, set()
    for kind, rows in [('real',reals),('synthetic',manifest),('support',support)]:
        for row in rows:
            row_errors = source_errors(row,context,real=kind=='real')
            errors.extend(kind+':'+error for error in row_errors)
            if not row.get('file') or row_errors:
                continue
            key = (row['file'],row.get('pixel_sha256'))
            if key in checked:
                continue
            checked.add(key)
            try:
                checked_rgb(folder,row,context)
            except (ValueError,OSError) as error:
                errors.append(kind+':'+str(error))
                mismatches += 1
    pairs = read(folder/'discriminator-pairs.json')
    if pairs is not None:
        try:
            # Validate the ENTIRE metadata inventory before the first pair read.
            validate_pairs(folder,pairs,reals,context,pixels=False)
            validate_pairs(folder,pairs,reals,context)
            if pairs.get('schema')==2 and pairs.get('dependency')!=pair_dependency(folder,reals,context):
                raise ValueError('Discriminator cache stale dependencies')
        except (ValueError,KeyError,OSError) as error:
            errors.append('discriminator:'+str(error))
            mismatches += 'pixel hash mismatch' in str(error)
    support_binding = read(folder/'evaluation-support.json')
    if support_binding is not None:
        dependency = digest([classes,read(folder/'realism.json'),'finite-legend-native-v1'])
        if support_binding.get('schema')==2:
            try:
                dependency = pair_dependency(folder,[r for r in reals if r['split']=='validation'],context)
            except (ValueError,OSError,KeyError) as error:
                errors.append('support:'+str(error))
                dependency = None
        if support_binding.get('dependency')!=dependency or support_binding.get('inventory_sha256')!=digest(support):
            errors.append('support:stale-dependency-or-inventory')
    return dict(passed=not errors,errors=sorted(set(errors)),real_n=len(reals),measurement_n=len(measurements),
        model_n=len(classes),synthetic_n=len(manifest),normalized_snapshot=current.get('snapshot'),
        archived_snapshot=eligibility.get('snapshot'),quarantined_admitted=sum('quarantined' in e for e in errors),
        policy_sha256=context['policy_sha256'],pixel_mismatches=mismatches,
        current_held_synthetic_n=sum(context['splits'].get(r['doc'])=='test' for r in manifest))


def cluster_interval(values, seed=20261108, weights=None):
    """Percentile bootstrap over independent WELL means, in percentage points."""
    a = np.asarray(values,float)
    if not len(a):
        return dict(estimate=None,ci95=[None,None],wells=0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(a),size=(10000,len(a)))
    w = np.ones(len(a)) if weights is None else np.asarray(weights,float)
    estimates = (a[draws]*w[draws]).sum(1)/w[draws].sum(1)
    low,high = np.quantile(estimates,[.025,.975])*100
    return dict(estimate=round(float(np.average(a,weights=w)*100),3),ci95=[round(float(low),3),round(float(high),3)],
        wells=len(a),resamples=10000,unit='well',method='percentile whole-well bootstrap; fragile with few wells')


def prediction_uncertainty(run, truth):
    by = defaultdict(list)
    lookup = {(r['doc'],r['annotation_id']):r for r in truth}
    for p in run['predictions']:
        r = lookup[(p['doc'],p['annotation_id'])]
        by[r['well']].append(int(p['actual']==p['predicted']))
    run['well_macro'] = cluster_interval([np.mean(v) for v in by.values()])
    run['box_weighted'] = cluster_interval([np.mean(v) for v in by.values()],weights=[len(v) for v in by.values()])
    run['per_well'] = [dict(well=k,n=len(v),accuracy=round(float(np.mean(v)*100),3)) for k,v in sorted(by.items())]
    run['confidence_limits'] = 'Few wells; intervals are exploratory, not evidence of equivalence.'
    return run


def compare_versions(folder, baseline):
    """Pair unchanged eligible annotations; label growth never becomes regression."""
    folder,baseline = Path(folder),Path(baseline)
    new,old = read(folder/'eval.json',{}),read(baseline/'eval.json',{})
    coverage_errors = []
    def truths(path):
        rows = [r for r in read(path/'real.json',[]) if r['split']=='validation']
        lookup = {(r['doc'],r['annotation_id']):r for r in rows}
        if len(lookup)!=len(rows):
            coverage_errors.append(path.name+': duplicate truth IDs')
        return lookup
    truth,prior = truths(folder),truths(baseline)
    common = {k for k,r in truth.items() if k in prior and r['box']==prior[k]['box']
        and r['name']==prior[k]['name'] and r.get('annotation_version')==prior[k].get('annotation_version',r.get('annotation_version'))}
    results = {}
    for algorithm in ('matcher','logistic'):
        for variant in ('clean','augmented'):
            def predictions(e, expected, label):
                rows = e.get('closed_set_within_document',{}).get(algorithm,{}).get(variant,{}).get('predictions',[])
                lookup = {(p['doc'],p['annotation_id']):p for p in rows}
                if len(rows)!=len(lookup):
                    coverage_errors.append(f'{label}/{algorithm}/{variant}: duplicate prediction IDs')
                missing,extra = expected.keys()-lookup.keys(),lookup.keys()-expected.keys()
                if missing or extra:
                    coverage_errors.append(f'{label}/{algorithm}/{variant}: missing={len(missing)}, extra={len(extra)}')
                return lookup
            a,b = predictions(old,prior,baseline.name),predictions(new,truth,folder.name)
            keys = sorted(common & a.keys() & b.keys())
            per = defaultdict(list)
            excluded = []
            for k in keys:
                if a[k]['actual'] != b[k]['actual'] or a[k]['candidate_count'] != b[k]['candidate_count']:
                    excluded.append(dict(doc=k[0],annotation_id=k[1],reason='actual class or candidate count changed'))
                    continue
                per[truth[k]['well']].append((int(a[k]['actual']==a[k]['predicted']),int(b[k]['actual']==b[k]['predicted'])))
            av = [np.mean([x[0] for x in v]) for v in per.values()]
            bv = [np.mean([x[1] for x in v]) for v in per.values()]
            results[f'{algorithm}/{variant}'] = dict(baseline=cluster_interval(av),candidate=cluster_interval(bv),
                delta=cluster_interval(np.array(bv)-np.array(av)),common_n=sum(map(len,per.values())),
                box_weighted_baseline=cluster_interval(av,weights=[len(v) for v in per.values()]),
                box_weighted_candidate=cluster_interval(bv,weights=[len(v) for v in per.values()]),
                box_weighted_delta=cluster_interval(np.array(bv)-np.array(av),weights=[len(v) for v in per.values()]),
                excluded_predictions=excluded,
                per_well=[dict(well=w,n=len(v),baseline=100*np.mean([x[0] for x in v]),candidate=100*np.mean([x[1] for x in v])) for w,v in sorted(per.items())])
    primary = results['matcher/augmented']
    lower = primary['delta']['ci95'][0]
    qualified = not coverage_errors and not primary['excluded_predictions'] and lower is not None and primary['delta']['wells']>=3 and lower>=-2
    return dict(baseline=baseline.name,candidate=folder.name,results=results,primary='matcher/augmented',
        noninferiority_tolerance_points=2,qualified=qualified,status='qualified' if qualified else 'inconclusive or regression',
        common_n=primary['common_n'],candidate_truth_n=len(truth),baseline_truth_n=len(prior),
        intended_common_n=len(common),coverage_errors=coverage_errors,
        changed_population=len(truth)!=len(prior) or len(common)!=len(truth),
        baseline_annotation_versions_available=all('annotation_version' in r for r in prior.values()),
        cohort_matching='ID, box, canonical name, current normalized eligibility, and annotation version when archived',
        decision='paired common unchanged annotations only; never compare different truth populations')


def finalize_evaluation(folder, result):
    folder = Path(folder)
    audit = leak_audit(folder)
    result['leak_audit'] = audit
    real = read(folder/'real.json',[])
    train_hashes = {r['pixel_sha256'] for r in real if r['split']=='train'}
    lookup = {(r['doc'],r['annotation_id']):r for r in real if r['split']=='validation'}
    for runs in result['closed_set_within_document'].values():
        for run in runs.values():
            subgroups = defaultdict(list)
            for p in run['predictions']:
                row = lookup[(p['doc'],p['annotation_id'])]
                key = 'shared_exact_crop_pixels' if row['pixel_sha256'] in train_hashes else 'no_exact_crop_overlap'
                subgroups[key].append(p['actual']==p['predicted'])
            run['pixel_overlap_cohorts'] = {k:dict(n=len(v),accuracy=100*float(np.mean(v))) for k,v in subgroups.items()}
            run['overlap_limit'] = 'Exact crop equality is a shared-symbol diagnostic; no claim of copied pages or unseen motif generalization.'
    _write(folder/'leak-audit.json',audit)
    probes = result['discriminator']
    checks = {}
    for kind in ('rail_only','edge_only'):
        for variant in ('clean','augmented'):
            run = probes[kind][variant]
            accuracy = run.get('accuracy')
            checks[f'{kind}/{variant}'] = accuracy is not None and abs(accuracy-50)<=10
    result['shortcut_gate'] = dict(passed=all(checks.values()),checks=checks,
        tolerance_points=10,policy='All clean/augmented rail-only and edge-only point accuracies must lie in [40,60]; CIs reported, not proof of equivalence')
    _write(folder/'eval.json',result)
    baseline_name = clean_baseline(folder.parent,folder)
    if baseline_name and baseline_name!=folder.name and (folder.parent/baseline_name/'eval.json').exists():
        result['baseline_comparison'] = compare_versions(folder,folder.parent/baseline_name)
        _write(folder/'comparison.json',result['baseline_comparison'])
    qualified = result.get('baseline_comparison',{}).get('qualified',False)
    result['utility_qualified'] = bool(audit['passed'] and result['shortcut_gate']['passed'] and qualified)
    result['status'] = 'evaluation completed'
    result['promotion_status'] = 'qualified' if result['utility_qualified'] else 'withheld: audit, shortcuts, or paired noninferiority not qualified'
    return result


def publish(root, folder):
    """Use the promoted baseline and paired CIs; preserve legacy API/ZIP history."""
    from . import dataset_loop
    root,folder = Path(root),Path(folder)
    e = read(folder/'eval.json',{})
    baseline = clean_baseline(root,folder)
    audit = leak_audit(folder)
    comparison = compare_versions(folder,root/baseline) if baseline and baseline!=folder.name else e.get('baseline_comparison',{})
    flags = []
    baseline_clean = bool(baseline and leak_audit(root/baseline)['passed'])
    if not baseline_clean:
        flags.append('A clean baseline is required; quarantined versions are reference only')
    if not audit['passed']:
        flags.append('Leak audit failed')
    if not e.get('shortcut_gate',{}).get('passed'):
        flags.append('Shortcut probe gate failed or unavailable')
    if not comparison.get('qualified'):
        flags.append('Paired within-document noninferiority not established at -2 points')
    delta_ci = comparison.get('results',{}).get('matcher/augmented',{}).get('delta',{}).get('ci95',[None,None])
    regression = delta_ci[1] is not None and delta_ci[1]<-2
    if regression:
        flags.append('REGRESSION: paired within-document 95% CI is entirely below -2 points')
    if comparison.get('changed_population'):
        flags.append('Human population changed; comparison uses only common unchanged annotations')
    eligible = bool(baseline_clean and audit['passed'] and e.get('shortcut_gate',{}).get('passed') and comparison.get('qualified'))
    e.update(status='evaluation completed',utility_qualified=eligible,promotion_status='qualified' if eligible else 'withheld',
        leak_audit=audit,baseline_comparison=comparison)
    _write(folder/'eval.json',e)
    # Legacy publisher only packages and updates history: completed != passed,
    # so its old unpaired thresholds cannot change the promoted pointer.
    report = dataset_loop._dataset_original_publish(root,folder)
    baseline_classes = {r['class_id']:r for r in read(root/baseline/'stats.json',{}).get('classes',[])} if baseline else {}
    candidate_classes = {r['class_id']:r for r in read(folder/'stats.json',{}).get('classes',[])}
    changes = dict(added=sorted(candidate_classes.keys()-baseline_classes.keys()),
        removed=sorted(baseline_classes.keys()-candidate_classes.keys()),
        changed=sorted(k for k in baseline_classes.keys()&candidate_classes.keys() if digest(baseline_classes[k])!=digest(candidate_classes[k])))
    report.update(previous=baseline,eligible_latest=eligible,flags=flags,regression=regression,
        deltas=dict(macro=comparison.get('results',{}).get('matcher/augmented',{}).get('delta',{}).get('estimate'),discriminator=None),changes=changes,
        comparison=comparison,leak_audit=audit,policy='Promoted baseline; paired whole-well 95% CI lower bound >= -2 points; leak and shortcut gates')
    _write(folder/'publication.json',report)
    if eligible:
        _write(root/'latest.json',dict(version=folder.name))
        (root/'latest').write_text(folder.name+'\n',encoding='utf-8')
    history = read(root/'history.json',[])
    for row in history:
        if row['version']==folder.name:
            row['publication'] = report
            row['status'] = e['status']
    _write(root/'history.json',history)
    (root/'CHANGELOG.md').write_text('\n'.join(['# Dataset changelog','',
        'Promotion uses paired well confidence intervals against the promoted baseline; revoked versions are archived diagnostics.','']+
        [f"- {r['version']}: {r.get('status')}; {json.dumps(r.get('publication',{}).get('flags',[]))}" for r in history])+'\n',encoding='utf-8')
    # Refresh the ZIP after adding the final publication decision.
    import zipfile
    temporary = root/(folder.name+'.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                archive.write(path,(Path(folder.name)/path.relative_to(folder)).as_posix())
    temporary.replace(root/(folder.name+'.zip'))
    return report


def clean_baseline(root, folder):
    """A revoked promotion pointer cannot authorize another release."""
    root,folder = Path(root),Path(folder)
    registry = read(root/'quarantine.json',{})
    choices = [read(root/'latest.json',{}).get('version'),
               read(root/'clean-baseline.json',{}).get('version'),
               read(folder/'stats.json',{}).get('controlled_baseline')]
    for name in choices:
        if name and name!=folder.name and name not in registry:
            return name
    return None


def revoke_unsafe_versions(root):
    """Retain old artifacts for comparison, remove them from training publication."""
    from . import paths
    root = Path(root)
    index = read(paths.DATA/'normalized/current.json',{})
    rejected = {(r['document_id'],r['annotation_id']) for r in index.get('records',[]) if r.get('quality_issues')}
    test_hashes = {r['pixel_sha256'] for r in index.get('records',[]) if r['split']=='test'}
    registry = read(root/'quarantine.json',{})
    for folder in root.glob('v[0-9]*'):
        if not folder.is_dir():
            continue
        rows = read(folder/'real.json',[])
        bad = sum((r['doc'],r['annotation_id']) in rejected for r in rows)
        duplicates = sum(r['pixel_sha256'] in test_hashes for r in rows)
        context = eligibility_context(folder)
        policy_errors = []
        eligibility = read(folder/'eligibility.json',{})
        if (eligibility.get('policy_sha256') or digest(read(folder/'split-policy.json',{})))!=context['policy_sha256']:
            policy_errors.append('Current split policy digest differs from release')
        inventory = read(folder/'stats.json',{}).get('classes',[])+rows
        inventory += read(folder/'realism.json',{}).get('measurements',[])
        inventory += read(folder/'nuisance.json',{}).get('components',[])
        inventory += read(folder/'eval-swatches.json',[])
        manifest_path = folder/'manifest.jsonl'
        if manifest_path.exists():
            inventory += [json.loads(s) for s in manifest_path.read_text(encoding='utf-8').splitlines()]
        for pair in read(folder/'discriminator-pairs.json',{}).get('pairs',[]):
            inventory.extend(pair[v] for v in ('real','clean','augmented'))
        for row in inventory:
            policy_errors.extend(source_errors(row,context,real=row.get('kind')=='real'))
        if bad or duplicates or policy_errors:
            registry[folder.name] = dict(rejected_rows=bad,test_identical_rows=duplicates,
                policy_errors=sorted(set(policy_errors)),
                reason='Revoked current policy/source eligibility; archive only, not a training release')
    _write(root/'quarantine.json',registry)
    latest = read(root/'latest.json',{}).get('version')
    if latest in registry:
        _write(root/'promotion-baseline.json',dict(version=latest,revoked=True,policy='Historical comparison only'))
        _write(root/'latest.json',dict(version=None,revoked=latest,reason=registry[latest]['reason']))
        (root/'latest').unlink(missing_ok=True)
    baseline = read(root/'promotion-baseline.json',{}).get('version')
    if baseline and (root/baseline).is_dir():
        (root/baseline/'keep').write_text('Preserve historical promoted baseline for paired comparisons. Quarantined, not a release.\n',encoding='utf-8')
    return registry


def configure_loop():
    """Install dataset-owned policy into the shared scheduler without editing it."""
    from . import dataset_loop,paths
    if hasattr(dataset_loop,'_dataset_original_publish'):
        return
    dataset_loop._dataset_original_publish = dataset_loop.publish
    original_snapshot,original_trigger = dataset_loop.snapshot,dataset_loop.trigger
    def snapshot(*args,**kwargs):
        value = original_snapshot(*args,**kwargs)
        dependency_files = [paths.PREFILL,paths.DATA/'normalized/split-policy.json',paths.DATA/'normalized/current.json',
            Path(__file__),Path(__file__).with_name('dataset.py'),Path(__file__).with_name('realism.py')]
        dependency_files += list(paths.PROPOSALS.glob('*/*.png'))+list(paths.CANDIDATES.glob('*.json'))
        stamps = []
        for path in sorted(dependency_files):
            try:
                st = path.stat()
                stamps.append([str(path),st.st_size,st.st_mtime_ns])
            except FileNotFoundError:
                pass
        index = read(paths.DATA/'normalized/current.json',{})
        critical = [read(paths.DATA/'normalized/split-policy.json',{}),index.get('snapshot'),
            [(r['sample_id'],r['annotation_version'],r.get('usable_for_generation'),r.get('quality_issues'),r['pixel_sha256'])
             for r in index.get('records',[])]]
        value.update(files=digest([value['files'],stamps]),eligibility=digest(critical))
        return value
    def trigger(baseline,current,last_build,now):
        result = original_trigger(baseline,current,last_build,now)
        if baseline.get('eligibility')!=current.get('eligibility'):
            result.update(due=True,any_change=True,reason='Immediate eligibility/policy invalidation')
        return result
    def metrics(e):
        run = e.get('closed_set_within_document',{}).get('matcher',{}).get('augmented',{})
        return dict(macro=run.get('well_macro',{}).get('estimate',run.get('mean_document_accuracy')),
            discriminator=e.get('discriminator',{}).get('augmented',{}).get('accuracy'))
    dataset_loop.publish = publish
    dataset_loop.snapshot = snapshot
    dataset_loop.trigger = trigger
    dataset_loop.metrics = metrics


def rate(correct, n):
    return dict(n=int(n), correct=int(correct), accuracy=round(100 * correct / n, 3) if n >= MIN_N else None,
                flag='too few' if n < MIN_N else None)


def summarize(truth, predictions, training_names):
    if len(truth)!=len(predictions):
        raise ValueError('Truth and predictions must have identical lengths')
    counts, good, conf = Counter(), Counter(), Counter()
    classes = {}
    for r, pred in zip(truth, predictions):
        name, cid = r['name'], r['class_id']
        counts[name] += 1
        good[name] += name == pred
        classes.setdefault(cid, dict(name=name, counts=Counter()))['counts'][name == pred] += 1
        if name != pred:
            conf[(name, str(pred))] += 1
    per_name = [dict(name=k, **rate(good[k], n), unseen_in_training=k not in training_names) for k, n in sorted(counts.items())]
    supported = [x for x in per_name if x['accuracy'] is not None]
    closed = [i for i, r in enumerate(truth) if r['name'] in training_names]
    per_class = [dict(class_id=k, name=v['name'], **rate(v['counts'][True], sum(v['counts'].values())))
                 for k, v in sorted(classes.items())]
    out = rate(sum(good.values()), len(truth))
    out.update(macro_accuracy=(round(float(np.mean([x['accuracy'] for x in supported])), 3) if supported else None),
               macro_classes=len(supported), macro_n=sum(x['n'] for x in supported),
               per_name=per_name, per_class=per_class,
               training_names=len(training_names), unseen_real_boxes=sum(r['name'] not in training_names for r in truth),
               closed_set=rate(sum(predictions[i] == truth[i]['name'] for i in closed), len(closed)),
               confusions=[dict(actual=a, predicted=b, n=n) for (a, b), n in conf.most_common(40)])
    return out


class Features:
    def __init__(self, folder, native=False):
        self.folder = folder
        self.cache = {}
        self.native = native
        self.source_stamp = None

    def get(self, row):
        from . import paths
        stamp = tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in
                      (paths.DATA/'normalized/split-policy.json',paths.DATA/'normalized/current.json'))
        if stamp!=self.source_stamp:
            self.context = eligibility_context(self.folder)
            self.source_stamp = stamp
        rel = row['file']
        rgb = checked_rgb(self.folder,row,self.context)
        key = (rel,row['pixel_sha256'])
        if key not in self.cache:
            image = Image.fromarray(rgb)
            # No letterbox padding shortcuts. Fixed small native-content crops.
            down = np.asarray(image.resize((12, 20), Image.Resampling.BILINEAR), dtype=np.float32).ravel() / 255
            native = score.features(rgb)
            self.cache[key] = (native, native if self.native else down)
        return self.cache[key]


def fit_predict(features, train, truth, algorithm='matcher'):
    names = sorted({r['name'] for r in train})
    if not train or not truth:
        return [None] * len(truth)
    which = 0 if algorithm == 'matcher' else 1
    x = np.array([features.get(r)[which] for r in train])
    xt = np.array([features.get(r)[which] for r in truth])
    y = np.array([r['name'] for r in train])
    if algorithm == 'matcher':
        centers = np.array([x[y == name].mean(0) for name in names])
        return [names[int(np.square(centers - f).sum(1).argmin())] for f in xt]
    if len(names) == 1:
        return names * len(truth)
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    model = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', max_iter=150,
                           tol=.001, random_state=31, n_jobs=1, class_weight='balanced'))
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(x, y)
        return model.predict(xt).tolist()


def classifier_run(features, train, truth, algorithm):
    for row in truth:
        assert_truth(row)
    predictions = fit_predict(features, train, truth, algorithm)
    result = summarize(truth, predictions, {r['name'] for r in train})
    result.update(algorithm=algorithm, train_n=len(train), train_document_classes=len({r['class_id'] for r in train}))
    result['predictions'] = [dict(doc=r['doc'],annotation_id=r['annotation_id'],actual=r['name'],predicted=p)
        for r,p in zip(truth,predictions)]
    prediction_uncertainty(result,truth)
    return result


def _legacy_paired_discriminator(features, synthetic, reals, variant, source=None):
    """Balance by real box; held wells never train the discriminator."""
    if not importlib.util.find_spec('sklearn'):
        return dict(n=0, accuracy=None, flag='sklearn unavailable')
    fake = [r for r in synthetic if r['variant'] == variant and (source is None or r['source'] == source)]
    groups = defaultdict(list)
    for r in fake:
        groups[(r['split'], r['name'])].append(r)
    rng = np.random.default_rng(19)
    data = {'train': [], 'validation': []}
    used = {'train': set(), 'validation': set()}
    for row in sorted(reals, key=lambda r: (r['doc'], r['annotation_id'])):
        split = row['split']
        if split not in data:
            continue
        choices = [r for r in groups[(split, row['name'])] if r['file'] not in used[split]]
        if not choices:
            continue
        same = [r for r in choices if r['doc'] == row['doc']]
        choices = same or choices
        f = choices[int(rng.integers(len(choices)))]
        used[split].add(f['file'])
        data[split].append((row, f))
    train, test = data['train'], data['validation']
    if len(train) < MIN_N or len(test) < MIN_N:
        return dict(n=2 * len(test), real_n=len(test), fake_n=len(test), train_n=2 * len(train),
                    accuracy=None, flag='too few', source=source)
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    def vectors(pairs):
        # Colour/repeat + tiny image. Labels/splits are never features.
        return np.array([np.r_[*features.get(r)] for pair in pairs for r in pair]), np.tile([1, 0], len(pairs))
    x, y = vectors(train)
    xt, yt = vectors(test)
    model = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', max_iter=200,
                         tol=.001, random_state=11, n_jobs=1))
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(x, y)
        pred = model.predict(xt)
    out = rate(int((pred == yt).sum()), len(yt))
    out.update(real_n=len(test), fake_n=len(test), train_n=len(y), train_real_n=len(train),
               variant=variant, source=source, model='CPU logistic (SGD), RGB + colour/repeat',
               test_wells=sorted({r['well'] for pair in test for r in pair}),
               distance_from_chance=round(abs(float(out['accuracy']) - 50), 3))
    return out


def paired_inventory(folder, reals):
    """One frozen doc/class/native-size pair per available real, shared arms."""
    from . import candidates, render
    from .dataset import DatasetBuilder, dataset_augment, save_rgb, local_model
    folder = Path(folder)
    cache = folder/'discriminator-pairs.json'
    saved = read(cache)
    context = eligibility_context(folder)
    for real in reals:
        errors = source_errors(real,context,real=True)
        if errors:
            raise ValueError('Unsafe requested real: '+','.join(errors))
    if saved is not None:
        validate_pairs(folder,saved,reals,context,pixels=False)
        if saved.get('schema')!=2 or saved.get('dependency')!=pair_dependency(folder,reals,context):
            raise ValueError('Discriminator cache stale dependencies; rebuild in a new release')
        validate_pairs(folder,saved,reals,context)
        return saved
    classes = {r['class_id']:r for r in read(folder/'stats.json')['classes']}
    measurements = [m for m in read(folder/'realism.json')['measurements'] if m['split']=='train']
    rows, missing = [], []
    nuisance = read(folder/'nuisance.json',{}).get('components',[])
    for real in sorted(reals,key=lambda r:(r['doc'],r['annotation_id'])):
        meta = classes.get(real['class_id'])
        if not meta or not meta.get('renderable',True):
            missing.append(dict(doc=real['doc'],annotation_id=real['annotation_id'],class_id=real['class_id']))
            continue
        rgb = checked_rgb(folder,real,context)
        h,w = rgb.shape[:2]
        errors = source_errors(meta,context)
        if errors:
            raise ValueError('Unsafe discriminator model: '+','.join(errors))
        with Image.open(folder/meta['model_file']) as im:
            tile = np.asarray(im.convert('RGB')).copy()
        model = dict(meta.get('rendering',dict(kind='tile',anchored=False,phase=[0,0])),tile=tile)
        if meta.get('render_model'):
            from . import learn
            model = learn.load_model(meta['render_model'],root=folder,signature=meta['signature'])
        elif meta.get('quilt_file'):
            model = candidates.load_data(folder/meta['quilt_file'])
        rng = np.random.default_rng(int(digest([real['doc'],real['annotation_id'],'paired-v1'])[:16],16))
        sized = dict(meta,width=w)
        rails,rail_source = DatasetBuilder.frame_for(sized,measurements,rng)
        clean = render.sample(local_model(model),rails,h,rng,y0=0)
        measurement = DatasetBuilder.nuisance_for(meta,measurements,rng)
        contexts = [n for n in nuisance if n['family']==meta['family']] or nuisance
        overlay_source = contexts[int(rng.integers(len(contexts)))] if contexts else None
        overlay = DatasetBuilder.overlay(folder,overlay_source) if overlay_source else None
        augmented,params = dataset_augment(clean,measurement,overlay,rng)
        render.frame(augmented,rails)
        fake_rows = {}
        for variant,rgb in [('clean',clean),('augmented',augmented)]:
            row = dict(meta,kind='synthetic',format='strip',variant=variant,
                augmentation=params if variant=='augmented' else {},paired_annotation_id=real['annotation_id'])
            key = digest([real['doc'],real['annotation_id'],variant,'pair-v1'])[:24]
            row.update(save_rgb(folder,f'discriminator/{key}.png',rgb,False))
            fake_rows[variant] = row
        rows.append(dict(real=real,**fake_rows,native_size=[w,h],
            identity=bool(np.array_equal(clean,augmented)),rail_source_id=(rail_source or {}).get('id'),
            nuisance_source_id=(overlay_source or {}).get('id')))
    saved = dict(schema=2,dependency=pair_dependency(folder,reals,context),pairs=rows,unavailable=missing,
        policy='Exact document/class/native width/height; identical base content and frame in both arms; train-only style')
    _write(cache,saved)
    validate_pairs(folder,saved,reals,context)
    return saved


def paired_discriminator(features, synthetic, reals, variant, source=None, probe='full', nonlinear=False, control=None):
    """Fixed models and identical paired inventory, without name/log substitution."""
    if not importlib.util.find_spec('sklearn'):
        return dict(n=0,accuracy=None,flag='sklearn unavailable')
    inventory = paired_inventory(features.folder,reals)
    pairs = [p for p in inventory['pairs'] if source is None or p[variant]['source']==source]
    train = [p for p in pairs if p['real']['split']=='train']
    test = [p for p in pairs if p['real']['split']=='validation']
    if len(train)<MIN_N or len(test)<MIN_N:
        return dict(n=2*len(test),accuracy=None,real_n=len(test),fake_n=len(test),train_n=2*len(train),flag='unsupported source / too few')
    context = eligibility_context(features.folder)
    def vector(row):
        rgb = checked_rgb(features.folder,row,context)
        dark = rgb.max(2)<80
        if probe=='rail':
            return np.array([max(dark[:,0].mean(),dark[:,-1].mean())])
        if probe=='edge':
            edges = np.concatenate([rgb[:,:3],rgb[:,-3:]],axis=1)
            return np.asarray(Image.fromarray(edges).resize((6,32)),np.float32).ravel()/255
        return np.r_[*features.get(row)]
    def vectors(pp):
        result = []
        for p in pp:
            a,b = vector(p['real']),vector(p[variant])
            if control=='negative':
                b = a.copy()
            elif control=='positive':
                b = np.ones_like(a)*5
            result.extend([a,b])
        return np.asarray(result),np.tile([1,0],len(pp))
    x,y = vectors(train)
    xt,yt = vectors(test)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from threadpoolctl import threadpool_limits
    estimator = (RandomForestClassifier(n_estimators=64,max_depth=5,min_samples_leaf=5,random_state=11,n_jobs=1)
        if nonlinear else LogisticRegression(C=.1,max_iter=500,random_state=11,n_jobs=1))
    model = make_pipeline(StandardScaler(),estimator)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(x,y)
        pred = model.predict(xt)
        probability = model.predict_proba(xt)[:,1]
    by = defaultdict(list)
    for i,p in enumerate(test):
        by[p['real']['well']].extend((pred[2*i:2*i+2]==yt[2*i:2*i+2]).astype(int).tolist())
    out = rate(int((pred==yt).sum()),len(yt))
    out.update(real_n=len(test),fake_n=len(test),train_n=len(y),variant=variant,source=source,probe=probe,
        model='random forest depth 5' if nonlinear else 'logistic C=0.1',
        balanced_accuracy=round(float(balanced_accuracy_score(yt,pred)*100),3),
        auroc=round(float(roc_auc_score(yt,probability)),5),well_macro=cluster_interval([np.mean(v) for v in by.values()]),
        per_well=[dict(well=w,n=len(v),accuracy=100*np.mean(v)) for w,v in sorted(by.items())],
        identity_held_pairs=sum(p['identity'] for p in test),held_pair_count=len(test),
        missing_pairs=len(inventory['unavailable']),control=control,
        distance_from_chance=round(abs(float(out['accuracy'])-50),3))
    return out


def owner_audit(folder):
    snapshot = read(folder / 'lab-snapshot.json', {})
    lookup = {r['file']: r for r in snapshot.get('patterns', [])}
    votes = ratings()
    by = defaultdict(Counter)
    by[('all', 'all')]
    for row in snapshot.get('patterns', []):
        source = row.get('source', 'unknown')
        method = row.get('method') or row.get('tile_source', 'unknown')
        for key in [(source, 'all'), ('all', method), (source, method)]:
            by[key]
    wrong_classes = Counter()
    wrong_evidence = defaultdict(list)
    unassigned = 0
    for file, vote in votes.items():
        verdict = vote.get('verdict')
        if verdict not in ('good', 'bad'):
            continue
        row = lookup.get(file, {})
        source = vote.get('source') or row.get('source', 'unknown')
        method = vote.get('method') or row.get('method') or row.get('tile_source', 'unknown')
        for key in [('all', 'all'), (source, 'all'), ('all', method), (source, method)]:
            by[key][verdict] += 1
        if row and verdict == 'bad':
            wrong_classes[row['pattern_id']] += 1
            wrong_evidence[row['pattern_id']].append(dict(file=file, note=vote.get('note', ''), at=vote.get('at')))
        if not row:
            unassigned += 1
    records = []
    for (source, method), counts in sorted(by.items()):
        records.append(dict(source=source, method=method, real=counts['good'], wrong=counts['bad'],
                            **rate(counts['good'], sum(counts.values()))))
    return dict(rows=records, wrong_classes=dict(wrong_classes), wrong_evidence=dict(wrong_evidence), unassigned_to_current_class=unassigned,
                policy='Latest owner Real/Wrong per file; cleared excluded; historical source/method retained when available.')


def evaluate(folder):
    folder = Path(folder)
    preflight = leak_audit(folder)
    if not preflight['passed']:
        raise ValueError('Evaluation refused revoked sources before model fitting: '+json.dumps(preflight['errors']))
    start = time.monotonic()
    manifest = [json.loads(s) for s in (folder / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    real = read(folder / 'real.json', [])
    wells = read(folder / 'wells.json', {})
    for r in manifest + real:
        if wells.get(r['doc'], {}).get('split') == 'test' or r['split'] == 'test':
            raise ValueError('Test well in dataset')
        if r.get('donor') and wells[r['donor']]['split'] != 'train' and r['split'] == 'train':
            raise ValueError('Held-out donor leaked into training')
    train_wells = {r['well'] for r in manifest + real if r['split'] == 'train'}
    val_wells = {r['well'] for r in manifest + real if r['split'] == 'validation'}
    if train_wells & val_wells:
        raise ValueError('A well appears in both training and validation')
    truth = [r for r in real if r['split'] == 'validation']
    for r in truth:
        assert_truth(r)
    strips = [r for r in manifest if r['format'] == 'strip']
    train = [r for r in strips if r['split'] == 'train']
    features = Features(folder,native=True)
    algorithms = ['matcher'] + (['logistic'] if importlib.util.find_spec('sklearn') else [])
    result = dict(schema=1, version=folder.name, evaluated_at=time.time(), truth_n=len(truth),
        truth_wells=sorted({r['well'] for r in truth}), truth_policy='Human-accepted validation wells only; machine evidence refused.',
        truth_fingerprint=digest([(r['doc'], r['annotation_id'], r['box'], r['name']) for r in truth]),
        target='normalized lithology name across wells; document class IDs retained in per_class',
        min_class_n=MIN_N, fake_to_real={}, clean_vs_augmented={}, ablation={})
    for alg in algorithms:
        clean = [r for r in train if r['variant'] == 'clean']
        augmented = [r for r in train if r['variant'] == 'augmented']
        print('EVAL', alg, 'train', len(augmented), 'real', len(truth), flush=True)
        a = classifier_run(features, augmented, truth, alg)
        c = classifier_run(features, clean, truth, alg)
        result['fake_to_real'][alg] = a
        result['clean_vs_augmented'][alg] = dict(clean=c, augmented=a,
            macro_delta_points=(round(a['macro_accuracy'] - c['macro_accuracy'], 3)
                                if a['macro_accuracy'] is not None and c['macro_accuracy'] is not None else None))
        legend = [r for r in train if r['variant'] == 'legend-only']
        column = [r for r in train if r['variant'] == 'real-learned']
        common = {r['class_id'] for r in legend} & {r['class_id'] for r in column}
        legend = [r for r in legend if r['class_id'] in common]
        column = [r for r in column if r['class_id'] in common]
        names = {r['name'] for r in legend}
        ab_truth = [r for r in truth if r['name'] in names]
        l = classifier_run(features, legend, ab_truth, alg)
        r = classifier_run(features, column, ab_truth, alg)
        result['ablation'][alg] = dict(legend_only=l, real_learned=r, matched_document_classes=len(common),
            task='cross-well training-generator ablation on supported names',
            representation='native phase-independent colour/repeat features in both arms',
            truth_ids=[dict(doc=r['doc'],annotation_id=r['annotation_id']) for r in ab_truth],
            selected_class_ids=sorted(common),excluded_truth_n=len(truth)-len(ab_truth),
            training_class_coverage=[len(common),len({r['class_id'] for r in train})],
            macro_delta_legend_minus_column=(round(l['macro_accuracy'] - r['macro_accuracy'], 3)
                                            if l['macro_accuracy'] is not None and r['macro_accuracy'] is not None else None))
    discriminator_features = Features(folder)
    result['discriminator'] = dict(
        augmented=paired_discriminator(discriminator_features, strips, real, 'augmented'),
        clean=paired_discriminator(discriminator_features, strips, real, 'clean'),
        nonlinear={v:paired_discriminator(discriminator_features,strips,real,v,nonlinear=True) for v in ('clean','augmented')},
        rail_only={v:paired_discriminator(discriminator_features,strips,real,v,probe='rail') for v in ('clean','augmented')},
        edge_only={v:paired_discriminator(discriminator_features,strips,real,v,probe='edge') for v in ('clean','augmented')},
        controls={c:paired_discriminator(discriminator_features,strips,real,'augmented',control=c) for c in ('negative','positive')},
        by_source={source: paired_discriminator(discriminator_features, strips, real, 'augmented', source)
                   for source in sorted({r['source'] for r in strips})})
    ca = {r['well']:r['accuracy']/100 for r in result['discriminator']['clean'].get('per_well',[])}
    aa = {r['well']:r['accuracy']/100 for r in result['discriminator']['augmented'].get('per_well',[])}
    result['discriminator']['paired_delta'] = cluster_interval([aa[w]-ca[w] for w in sorted(ca.keys()&aa.keys())])
    result['discriminator']['inventory_policy'] = 'Both arms use the same exact document/class/native-size cohort; identity counts and unsupported coverage are explicit.'
    result['owner_audit'] = owner_audit(folder)
    result['seconds'] = round(time.monotonic() - start, 2)
    primary = result['fake_to_real']['matcher']
    result['status'] = 'evaluation completed' if primary['macro_accuracy'] is not None else 'insufficient evidence'
    result['limits'] = [
        'Names, not unseen document IDs, are predicted across wells. Unseen lithologies count as misses.',
        'Ablation is paired on available saved training column models; native legend extraction can have seam risks.',
        'Existing tile selection saw human source boxes; validation wells and their donors are excluded from all classifier training.',
        'Discriminator uses distinct files, balanced real/fake pairs and held wells. Synthetic validation tiles can derive from validation source crops.',
        'Residual overlays are appearance proxies; semantic text/curve/depth labels are unavailable.',
        'Whole-well bootstrap intervals are exploratory with few wells; repeated crops do not increase independent sample size.',
        'Real-vs-real control duplicates each real vector; it checks pipeline bias, not independent real-domain equivalence.',
        'Recognition is legend-conditioned oracle-column crop recognition; detector transfer utility and continuous-window fidelity are not measured.']
    _write(folder / 'eval.json', result)
    (folder / 'eval.md').write_text(markdown(result), encoding='utf-8')
    print('PASS eval', folder.name, 'status', result['status'], 'seconds', result['seconds'], flush=True)
    print(json.dumps(dict(a=primary['macro_accuracy'], n=primary['n'], macro_classes=primary['macro_classes'],
                         b=result['ablation']['matcher']['macro_delta_legend_minus_column'],
                         c=result['clean_vs_augmented']['matcher']['macro_delta_points'],
                         d=result['discriminator']['augmented']), ensure_ascii=False), flush=True)
    from .dataset import add_document_evaluation
    return add_document_evaluation(folder, result)


def markdown(e):
    lines = [f"# {e['version']} evaluation", '', f"Status: {e['status']}. Human validation boxes: {e['truth_n']} in {len(e['truth_wells'])} wells.",
             '', 'Every accuracy is a percentage; macro averages only labels with at least three real boxes.', '',
             '| Model / experiment | Train n | Real n | Macro labels / boxes | Macro % | Top-1 % |',
             '|---|---:|---:|---|---:|---:|']
    if e.get('baseline_comparison'):
        comparison = e['baseline_comparison']
        primary = comparison['results']['matcher/augmented']
        lines[2:2] = [f"Promotion: {e.get('promotion_status')}. Baseline {comparison['baseline']}; common unchanged boxes {comparison['common_n']}.",
            f"Primary well-macro: baseline {primary['baseline']['estimate']}% {primary['baseline']['ci95']}; candidate {primary['candidate']['estimate']}% {primary['candidate']['ci95']}.",
            f"Paired delta {primary['delta']['estimate']} points, 95% well-bootstrap CI {primary['delta']['ci95']}. Tolerance -2 points.",'']
    def show(alg, name, r):
        value = lambda v: 'too few' if v is None else str(v)
        lines.append(f"| {alg} / {name} | {r['train_n']} | {r['n']} | {r['macro_classes']} / {r['macro_n']} | {value(r['macro_accuracy'])} | {value(r['accuracy'])} |")
    for alg in e['fake_to_real']:
        show(alg, 'augmented', e['fake_to_real'][alg])
        show(alg, 'clean', e['clean_vs_augmented'][alg]['clean'])
        show(alg, 'legend only (paired)', e['ablation'][alg]['legend_only'])
        show(alg, 'real learned (paired)', e['ablation'][alg]['real_learned'])
    lines += ['', '## Real versus fake discriminator', '', '| Variant / source | Train n | Held real n | Held fake n | Accuracy % (50 is chance) |', '|---|---:|---:|---:|---:|']
    for key, r in [('clean', e['discriminator']['clean']), ('augmented', e['discriminator']['augmented']), *e['discriminator']['by_source'].items()]:
        lines.append(f"| {key} | {r.get('train_n', 0)} | {r.get('real_n', 0)} | {r.get('fake_n', 0)} | {r['accuracy'] if r['accuracy'] is not None else 'too few'} |")
    lines += ['', '| Shortcut / variant | Accuracy % | 95% well-macro CI | Held pairs |', '|---|---:|---|---:|']
    for kind in ('rail_only','edge_only','nonlinear','controls'):
        for variant,r in e['discriminator'].get(kind,{}).items():
            lines.append(f"| {kind} / {variant} | {r['accuracy']} | {r.get('well_macro',{}).get('ci95')} | {r.get('held_pair_count',0)} |")
    lines += ['', '## Per document class (augmented matcher)', '', '| Class | Lithology | Real n | Top-1 % |', '|---|---|---:|---:|']
    for r in e['fake_to_real']['matcher']['per_class']:
        lines.append(f"| {r['class_id']} | {r['name']} | {r['n']} | {r['accuracy'] if r['accuracy'] is not None else 'too few'} |")
    lines += ['', '## Confusions (augmented matcher)', '', '| Human | Predicted | n |', '|---|---|---:|']
    for r in e['fake_to_real']['matcher']['confusions']:
        lines.append(f"| {r['actual']} | {r['predicted']} | {r['n']} |")
    lines += ['', '## Owner audit', '', '| Source | Method | Real | Wrong | Rated n | % Real |', '|---|---|---:|---:|---:|---:|']
    for r in e['owner_audit']['rows']:
        lines.append(f"| {r['source']} | {r['method']} | {r['real']} | {r['wrong']} | {r['n']} | {r['accuracy'] if r['accuracy'] is not None else 'too few'} |")
    lines += ['', '## Limits', ''] + [f'- {s}' for s in e['limits']]
    return '\n'.join(lines) + '\n'
