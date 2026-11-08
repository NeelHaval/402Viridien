"""Fixed-split real-crop classification, deliberately separate from intervals.

Targets are exact source ``target.key`` strings. No geological aliases, class
merges, thickness labels, boundaries or depth scores are inferred from crops.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import uuid

import numpy as np
from PIL import Image

from .data import sha256
from .groups import normalize_group
from .evaluate import _predict, _model_source, discover_config_assets

FEATURE_VERSION = 'native-rgb-texture-v1'
METRIC_VERSION = 'crop-set-classification-v1'


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _root(data_root):
    root = Path(data_root).resolve()
    for candidate in (root, root/'real', root/'viridien'/'data'/'real', root/'data'/'real'):
        if (candidate/'manifest.json').is_file():
            return candidate
    raise ValueError(f'PR54 real/manifest.json not found under {root}')


def _safe(root, name):
    if not isinstance(name, str):
        raise ValueError('asset path must be text')
    path = PurePosixPath(name)
    if path.is_absolute() or '\\' in name or '..' in path.parts or any(':' in part for part in path.parts):
        raise ValueError(f'unsafe crop asset path: {name}')
    result = root.joinpath(*path.parts).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError(f'missing or unsafe crop asset: {name}')
    return result


def _integers(value, count, name):
    if not isinstance(value,list) or len(value)!=count or any(type(x) is not int for x in value):
        raise ValueError(f'{name} must contain {count} integers')
    return value


def audit_crops(data_root):
    """Validate fixed memberships, permissions, native pixels and safe paths.

    Returns root/manifest/train/test/holds/stats/files. Holds remain saved and
    audited but cannot enter fitting or test scoring. JSONL exports must exactly
    match their authoritative manifest rows, including labels and quality flags.
    """
    root = _root(data_root)
    manifest = _read(root/'manifest.json')
    if manifest.get('schema') != 'strata.real.v1' or not isinstance(manifest.get('records'),list):
        raise ValueError('expected strata.real.v1 manifest')
    policy = _read(root/'split-policy.json')
    if policy != manifest.get('split_policy') or policy.get('schema') != 1 or not isinstance(policy.get('seed'),str):
        raise ValueError('split-policy.json disagrees with fixed manifest policy')
    documents = policy.get('document_groups',{})
    seen_declarations=policy.get('previously_seen_groups',[])
    if not isinstance(seen_declarations,list) or any(not isinstance(group,str) or not group for group in seen_declarations):
        raise ValueError('previously_seen_groups must be a list of nonempty well identities')
    previously_seen = {normalize_group(group) for group in seen_declarations}
    ids, groups, assets, native_cache = set(), {}, {}, {}
    rows = manifest['records']
    all_test_pixels = {r.get('pixel_sha256') for r in rows if r.get('split')=='test'}
    taxonomy = {}
    for row in rows:
        ident = row.get('sample_id')
        if not isinstance(ident,str) or not ident or ident in ids:
            raise ValueError('invalid or duplicate sample_id')
        ids.add(ident)
        group, split = row.get('source_group'), row.get('split')
        if not isinstance(group,str) or not group or split not in ('train','test'):
            raise ValueError(f'{ident}: source_group and fixed train/test split required')
        if documents.get(row.get('document_id')) != group:
            raise ValueError(f'{ident}: document-to-well mapping changed')
        expected = 'test' if int(hashlib.sha256((policy['seed']+':'+group).encode()).hexdigest()[:8],16)%5==0 else 'train'
        canonical_group=normalize_group(group)
        if split != expected or canonical_group in groups and groups[canonical_group] != split:
            raise ValueError(f'{ident}: fixed well split changed or leaks across splits')
        groups[canonical_group]=split
        status = 'retrospective' if canonical_group in previously_seen else 'prospective'
        if row.get('evaluation_status') != status:
            raise ValueError(f'{ident}: evaluation_status disagrees with previous-use policy')
        issues = row.get('quality_issues')
        if not isinstance(issues,list) or row.get('quality') not in ('accepted','review'):
            raise ValueError(f'{ident}: quality flags required')
        eligible = row['quality']=='accepted' and not issues
        if (not row.get('accepted') or row.get('kind')!='real' or
            row.get('usable_for_generation') is not (eligible and split=='train') or
            row.get('usable_for_test') is not (eligible and split=='test')):
            raise ValueError(f'{ident}: invalid real/quality-hold permission')
        if row['quality'] != ('review' if issues else 'accepted'):
            raise ValueError(f'{ident}: quality and issues disagree')
        if row.get('source_flags') and 'source_flags' not in issues:
            raise ValueError(f'{ident}: source flags are not held')
        if row['usable_for_generation'] and row.get('pixel_sha256') in all_test_pixels:
            raise ValueError(f'{ident}: exact test pixels enter training')
        x0,y0,x1,y1 = _integers(row.get('box_xyxy'),4,'source crop box')
        width,height = _integers(row.get('source_size'),2,'source_size')
        if not (0<=x0<x1<=width and 0<=y0<y1<=height):
            raise ValueError(f'{ident}: source crop outside document bounds')
        targets = row.get('targets')
        if not isinstance(targets,list) or not targets:
            raise ValueError(f'{ident}: nonempty exact target list required')
        keys=[]
        for target in targets:
            key = target.get('key')
            if not isinstance(key,str) or not key or not isinstance(target.get('id'),str):
                raise ValueError(f'{ident}: exact target key/id required')
            if target['id'] in taxonomy and taxonomy[target['id']]!=key:
                raise ValueError(f'{ident}: label ID maps to inconsistent keys')
            taxonomy[target['id']]=key;keys.append(key)
        if len(keys)!=len(set(keys)):
            raise ValueError(f'{ident}: duplicate target key')
        paths={name:_safe(root,row.get(name)) for name in ('native_png','normalized_png','valid_mask_png')}
        for path in paths.values():
            if str(path) not in assets:
                assets[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size,'role':'crop_asset'}
        native = str(paths['native_png'])
        if native not in native_cache:
            with Image.open(native) as image:
                if image.width*image.height>12_000_000:
                    raise ValueError(f'{ident}: crop exceeds 12M pixel resource cap')
                rgb=image.convert('RGB')
                native_cache[native]=(rgb.size,hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode()+rgb.tobytes()).hexdigest())
        size,pixel = native_cache[native]
        if size!=(x1-x0,y1-y0) or pixel!=row.get('pixel_sha256'):
            raise ValueError(f'{ident}: native dimensions or widthxheight RGB pixel hash mismatch')
        normalization=row.get('normalization',{})
        content=_integers(normalization.get('content_xyxy'),4,'normalization content box')
        if normalization.get('source_size') != list(size) or normalization.get('target_size') != [224,224]:
            raise ValueError(f'{ident}: invalid normalization dimensions')
        a,b,c,d=content
        if not (0<=a<c<=224 and 0<=b<d<=224):
            raise ValueError(f'{ident}: invalid normalization content bounds')
        with Image.open(paths['normalized_png']) as image:
            if image.size!=(224,224):
                raise ValueError(f'{ident}: normalized image must be 224x224')
            with Image.open(native) as source:
                expected_rgb=Image.new('RGB',(224,224),'white')
                expected_rgb.paste(source.convert('RGB').resize((c-a,d-b),Image.Resampling.LANCZOS),(a,b))
            if not np.array_equal(np.asarray(image.convert('RGB')),np.asarray(expected_rgb)):
                raise ValueError(f'{ident}: normalized pixels disagree with authoritative native crop and letterbox transform')
        with Image.open(paths['valid_mask_png']) as mask:
            if mask.size!=(224,224):
                raise ValueError(f'{ident}: valid mask must be 224x224')
            actual=np.asarray(mask.convert('L'))
            expected_mask=np.zeros((224,224),np.uint8);expected_mask[b:d,a:c]=255
            if not np.array_equal(actual,expected_mask):
                raise ValueError(f'{ident}: mask does not match explicit letterbox content')
    train=[r for r in rows if r['usable_for_generation']]
    test=[r for r in rows if r['usable_for_test']]
    holds=[r for r in rows if r['quality']=='review']
    if not train or not test:
        raise ValueError('classification needs nonempty usable fixed train and test sets')
    for name,expected in (('train',train),('test',test),('needs-review',holds)):
        actual=[json.loads(line) for line in (root/(name+'.jsonl')).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if actual!=expected:
            raise ValueError(f'{name}.jsonl differs from fixed manifest')
    if _read(root/'generator-sources.json')!=train:
        raise ValueError('generator sources include held/test records or differ from fixed train')
    stats={'accepted':len(rows),'train':len(train),'test':len(test),'needs_review':len(holds),
           'test_groups':len({r['source_group'] for r in test})}
    if manifest.get('stats')!=stats:
        raise ValueError('manifest counts disagree with audited rows')
    declared={x['id']:x['key'] for x in manifest.get('labels',[])}
    if declared!=taxonomy:
        raise ValueError('manifest label taxonomy disagrees with target IDs/keys')
    for name in ('manifest.json','train.jsonl','test.jsonl','needs-review.jsonl','split-policy.json','generator-sources.json'):
        path=root/name;assets[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size,'role':'crop_metadata'}
    fingerprint=hashlib.sha256(json.dumps(sorted((Path(k).relative_to(root).as_posix(),v['sha256']) for k,v in assets.items())).encode()).hexdigest()
    return {'root':root,'manifest':manifest,'train':train,'test':test,'holds':holds,'stats':stats,'files':assets,
            'dataset_sha256':fingerprint,'status_counts':dict(Counter(r['evaluation_status'] for r in test))}


def _features(path):
    with Image.open(path) as image:
        if image.width*image.height>12_000_000:
            raise ValueError('crop exceeds 12M pixel resource cap')
        rgb=np.asarray(image.convert('RGB')).astype(np.float32)/255
    # Native aspect and motif scale are preserved; no stretch or padded pixels.
    gray=rgb.mean(2);flat=rgb.reshape(-1,3)
    descriptor=np.r_[flat.mean(0),flat.std(0),np.quantile(flat,[.1,.5,.9],axis=0).ravel(),
                     (gray<.25).mean(),(gray<.7).mean(),(gray>.95).mean(),
                     np.abs(np.diff(gray,axis=1)).mean() if gray.shape[1]>1 else 0,
                     np.abs(np.diff(gray,axis=0)).mean() if len(gray)>1 else 0]
    for axis in (0,1):
        profile=(gray<.75).mean(axis=axis);profile=profile-profile.mean();denom=float(profile@profile)+1e-9
        descriptor=np.r_[descriptor,[float(profile[:-lag]@profile[lag:])/denom if len(profile)>lag else 0 for lag in (2,4,8,16)]]
    return descriptor


def fit_prototypes(audit, destination):
    """Fit only audited usable training rows; no access to test labels here."""
    values,labels=defaultdict(list),set()
    for row in audit['train']:
        descriptor=_features(_safe(audit['root'],row['native_png']))
        for target in row['targets']:
            values[target['key']].append(descriptor);labels.add(target['key'])
    order=sorted(labels)
    all_features=np.asarray([item for group in values.values() for item in group])
    scale=np.maximum(all_features.std(0),.025)
    training={'schema_version':1,'feature_version':FEATURE_VERSION,'labels':order,
              'prototypes':[np.mean(values[label],axis=0).tolist() for label in order],
              'scale':scale.tolist(),'training_counts':{label:len(values[label]) for label in order},
              'training_groups':sorted({normalize_group(r['source_group']) for r in audit['train']}),
              'training_sample_ids':[r['sample_id'] for r in audit['train']],
              'training_pixels':sorted({r['pixel_sha256'] for r in audit['train']})}
    training['training_sha256']=hashlib.sha256(json.dumps(audit['train'],sort_keys=True).encode()).hexdigest()
    _json(destination,training)
    return training


def predict(request, config):
    """Builtin classifier plugin. Return exact label-key strings, possibly []."""
    artifact=_read(config['prototype_path'])
    if artifact.get('feature_version')!=FEATURE_VERSION:
        raise ValueError('prototype feature version mismatch')
    labels=artifact['labels']
    if labels!=request['candidate_labels']:
        raise ValueError('candidate taxonomy disagrees with trained artifact')
    query=_features(request['crop'])
    distances=np.linalg.norm((np.asarray(artifact['prototypes'])-query)/np.asarray(artifact['scale']),axis=1)
    best=int(np.argmin(distances))
    reject=float(config.get('reject_distance',1e30))
    if not math.isfinite(reject) or reject<0:
        raise ValueError('reject_distance must be finite and nonnegative')
    if distances[best]>reject:
        return []
    return [labels[best]]


def _metrics(cases):
    if not cases:
        return {'cases':0,'exact_set_accuracy':None,'micro_f1':None,'macro_f1':None,'completion_rate':None,
                'errors':0,'empty_predictions':0,'unknown_rate':None,'per_class':{}}
    counts=defaultdict(lambda:Counter(tp=0,fp=0,fn=0,support=0))
    correct=0;unknown=0;predictions=0
    for case in cases:
        truth=set(case['targets']);pred=set(case['predicted']) if not case['error'] else set()
        correct+=int(not case['error'] and truth==pred)
        predictions+=len(pred);unknown+=len(pred-set(case['candidate_labels']))
        for label in truth|pred:
            counts[label]['tp']+=int(label in truth and label in pred)
            counts[label]['fp']+=int(label not in truth and label in pred)
            counts[label]['fn']+=int(label in truth and label not in pred)
            counts[label]['support']+=int(label in truth)
    per_class={}
    for label,count in sorted(counts.items()):
        tp,fp,fn=count['tp'],count['fp'],count['fn']
        per_class[label]={**count,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
                          'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    tp=sum(c['tp'] for c in counts.values());fp=sum(c['fp'] for c in counts.values());fn=sum(c['fn'] for c in counts.values())
    errors=sum(bool(c['error']) for c in cases)
    return {'cases':len(cases),'exact_set_accuracy':correct/len(cases),'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'macro_f1':float(np.mean([c['f1'] for c in per_class.values()])),'completion_rate':1-errors/len(cases),
            'errors':errors,'empty_predictions':sum(not c['predicted'] and not c['error'] for c in cases),
            'unknown_rate':unknown/predictions if predictions else 0.,'unknown_predictions':unknown,
            'multilabel_targets':sum(len(c['targets'])>1 for c in cases),'per_class':per_class}


def _scorer_hash():
    # Keep the score definition independently identifiable from classifier and
    # reporting changes in this module. The complete source is also hashed.
    return hashlib.sha256((METRIC_VERSION+'\n'+inspect.getsource(_metrics)).encode()).hexdigest()


def _quality_gates(overall,min_score):
    failures=(['classifier errors'] if overall['errors'] else [])+(['empty predictions'] if overall['empty_predictions'] else [])
    if min_score is not None and overall['exact_set_accuracy']<min_score:
        failures.append('exact-set accuracy below min_score')
    return {'passed':not failures,'min_score':min_score,'failures':failures}


def classify(data_root, model='builtin:prototype', config=None, output='runs/crops', timeout=60, min_score=None):
    """Audit, train, classify fixed test crops and save an immutable run.

    Custom module:function plugins receive only crop and training candidate
    keys; return list[str]. Failure/timeout cases stay in every metric denominator.
    Workers inherit the shared credential allowlist and hard process-tree timeout.
    """
    if not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('timeout must be finite and positive')
    if min_score is not None and (not isinstance(min_score,(int,float)) or not math.isfinite(min_score) or not 0<=min_score<=1):
        raise ValueError('min_score must be between zero and one')
    config={} if config is None else dict(config)
    declarations=config.get('training_groups',[])
    if not isinstance(declarations,list) or any(not isinstance(group,str) or not group.strip() for group in declarations):
        raise ValueError('training_groups must be a list of nonempty well identities')
    audit=audit_crops(data_root)
    test_groups={normalize_group(r['source_group']) for r in audit['test']}
    if test_groups & {normalize_group(group) for group in declarations}:
        raise ValueError('declared model training groups overlap fixed crop test groups')
    if model!='builtin:prototype' and (':' not in model or model.startswith('builtin:')):
        raise ValueError('classifier must be builtin:prototype or module:function')
    run=Path(output).resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:10])
    run.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    training=fit_prototypes({'train':audit['train'],'root':audit['root']},run/'prototype.json')
    worker_config={**config,'prototype_path':str(run/'prototype.json')}
    actual_model='viridian402.crops:predict' if model=='builtin:prototype' else model
    cases=[]
    for row in audit['test']:
        # No sample/document/group/annotation identifiers, target labels or test
        # metadata reach model request. Candidate vocabulary is train-only.
        request={'crop':str(_safe(audit['root'],row['native_png'])),'candidate_labels':training['labels']}
        predicted,error,seconds,peak=_predict(actual_model,request,worker_config,timeout)
        if not error:
            if not isinstance(predicted,list) or any(not isinstance(label,str) or not label for label in predicted):
                error='classifier must return a list of nonempty exact label-key strings'
            elif len(predicted)!=len(set(predicted)):
                error='classifier returned duplicate labels'
        cases.append({'sample_id':row['sample_id'],'source_group':normalize_group(row['source_group']),
                      'evaluation_status':row['evaluation_status'],'targets':sorted(t['key'] for t in row['targets']),
                      'predicted':sorted(predicted) if not error else [],'candidate_labels':training['labels'],
                      'error':error,'seconds':seconds,'peak_rss_mb':peak,'pixel_sha256':row['pixel_sha256']})
    overall=_metrics(cases)
    overall.update(seconds=time.perf_counter()-started,peak_worker_rss_mb=max((c['peak_rss_mb'] for c in cases),default=0))
    summary={'schema_version':1,'task':'crop_classification','metric_version':METRIC_VERSION,
             'run_id':run.name,'scorer_sha256':_scorer_hash(),
             'snapshot':audit['manifest'].get('snapshot'),'dataset_sha256':audit['dataset_sha256'],'model':model,
             'counts':audit['stats'],'overall':overall,
             'by_evaluation_status':{status:_metrics([c for c in cases if c['evaluation_status']==status]) for status in ('prospective','retrospective')},
             'by_group':{group:_metrics([c for c in cases if c['source_group']==group]) for group in sorted(test_groups)},
             'training_groups':training['training_groups'],'training_labels':training['labels'],
             'declared_model_training_groups':sorted({normalize_group(group) for group in declarations}),
             'model_training_history':'train-only prototype fit' if model=='builtin:prototype' else 'declared' if 'training_groups' in config else 'unknown',
             'test_only_labels':sorted({t['key'] for r in audit['test'] for t in r['targets']}-set(training['labels'])),
             'quality_gates':_quality_gates(overall,min_score),
             'limitations':['Crop classification does not score interval boundaries, depths, component positions, or bed thickness.',
                            'Retrospective and prospective statuses are inherited from the source previous-use policy; this run is not a new blinded evaluation.',
                            'Exact source target keys remain distinct; untrained test labels are retained in denominators.',
                            'Accepted source labels include user review and machine-origin suggestions; acceptance is not independent geological verification.']}
    files=dict(audit['files'])
    for path,role in ((Path(__file__),'classifier_source'),(Path(__file__).with_name('evaluate.py'),'worker_source'),
                      (run/'prototype.json','trained_artifact')):
        files[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size,'role':role}
    source=_model_source(actual_model)
    if source:
        for path in source.parent.rglob('*.py'):
            if not any(p in ('.git','__pycache__','.venv','venv') for p in path.parts):
                files[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size,'role':'model_source'}
    for name in ('requirements.lock','pyproject.toml'):
        path=Path(__file__).resolve().parents[2]/name
        if path.is_file():files[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size,'role':'dependencies'}
    for path in discover_config_assets(config):
        files[str(path)]={'sha256':sha256(path),'bytes':path.stat().st_size,'role':'model_config_asset'}
    _json(run/'provenance.json',{'dataset_sha256':audit['dataset_sha256'],'model':model,'worker_model':actual_model,
          'config':config,'worker_config':worker_config,'files':files,'python':sys.version,'platform':platform.platform(),
          'evaluation':{'timeout_seconds':timeout,'min_score':min_score},
          'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata.get('Name'))),
          'training_sha256':training['training_sha256'],'test_sha256':sha256(audit['root']/'test.jsonl')})
    _json(run/'audit.json',{'snapshot':summary['snapshot'],'counts':audit['stats'],'test_statuses':audit['status_counts'],
                           'dataset_sha256':audit['dataset_sha256'],'asset_count':sum(v['role']=='crop_asset' for v in audit['files'].values())})
    with (run/'per_case.jsonl').open('w',encoding='utf-8') as handle:
        for case in cases:handle.write(json.dumps(case,sort_keys=True,allow_nan=False)+'\n')
    summary['per_case_sha256']=sha256(run/'per_case.jsonl')
    summary['prototype_sha256']=sha256(run/'prototype.json')
    summary['provenance_sha256']=sha256(run/'provenance.json')
    summary['audit_sha256']=sha256(run/'audit.json')
    _json(run/'summary.json',summary)
    report=['# Real crop classification','',f"Snapshot `{summary['snapshot']}`; model `{model}`.",'',
            f"Usable training crops: {audit['stats']['train']}; fixed test crops: {audit['stats']['test']}; quality holds excluded: {audit['stats']['needs_review']}.",'',
            '| Status | Cases | Exact-set accuracy | Micro F1 | Macro F1 | Errors |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for status,metrics in [('all',overall),*summary['by_evaluation_status'].items()]:
        fmt=lambda x:'unavailable' if x is None else f'{x:.4f}'
        report.append(f"| {status} | {metrics['cases']} | {fmt(metrics['exact_set_accuracy'])} | {fmt(metrics['micro_f1'])} | {fmt(metrics['macro_f1'])} | {metrics['errors']} |")
    report+=['','Metric denominators include failed and empty predictions. Macro F1 averages labels present in test targets or predictions; train-only absent labels do not create artificial zero classes.','',
             f"Configured checks: {'PASS' if summary['quality_gates']['passed'] else 'FAIL'}; minimum exact-set accuracy: {min_score if min_score is not None else 'not configured (execution and nonempty-output checks only)' }.",'',
             'Test-only exact labels: '+(', '.join(summary['test_only_labels']) or 'none')+'.','',*summary['limitations']]
    (run/'summary.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return run,summary


def _load_crop_run(path):
    """Load a complete saved crop run, rejecting changed cases/artifacts."""
    path=Path(path)
    if path.is_file():path=path.parent
    summary=_read(path/'summary.json')
    if summary.get('task')!='crop_classification':
        raise ValueError('run is not a crop-classification result')
    cases=[json.loads(line) for line in (path/'per_case.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(cases)!=summary['overall']['cases'] or len({c['sample_id'] for c in cases})!=len(cases):
        raise ValueError('crop run case count differs from summary or IDs are duplicated')
    for name,key in (('per_case.jsonl','per_case_sha256'),('prototype.json','prototype_sha256'),('provenance.json','provenance_sha256')):
        if not summary.get(key) or sha256(path/name)!=summary[key]:
            raise ValueError(f'crop run {name} checksum mismatch or missing')
    provenance=_read(path/'provenance.json')
    if provenance.get('model')!=summary.get('model') or provenance.get('dataset_sha256')!=summary.get('dataset_sha256'):
        raise ValueError('crop run attribution differs from recorded provenance')
    minimum=summary['quality_gates'].get('min_score')
    if minimum is not None and (isinstance(minimum,bool) or not isinstance(minimum,(int,float)) or not math.isfinite(minimum) or not 0<=minimum<=1):
        raise ValueError('invalid saved minimum score')
    if provenance.get('evaluation',{}).get('min_score',minimum)!=minimum:
        raise ValueError('crop score gate differs from recorded evaluation')
    if summary['quality_gates']!=_quality_gates(_metrics(cases),minimum):
        raise ValueError('crop quality gates differ from saved cases')
    training=_read(path/'prototype.json')
    if summary.get('training_groups')!=training.get('training_groups') or summary.get('training_labels')!=training.get('labels'):
        raise ValueError('crop summary training claims differ from hashed prototype artifact')
    if any(case.get('candidate_labels')!=training.get('labels') for case in cases):
        raise ValueError('crop candidate taxonomy differs from hashed prototype artifact')
    expected_test_only=sorted({label for case in cases for label in case['targets']}-set(training['labels']))
    if summary.get('test_only_labels')!=expected_test_only:
        raise ValueError('crop summary test-only labels differ from saved cases')
    audit=_read(path/'audit.json')
    if summary.get('audit_sha256') and sha256(path/'audit.json')!=summary['audit_sha256']:
        raise ValueError('crop audit checksum mismatch')
    if summary.get('counts')!=audit.get('counts') or audit.get('dataset_sha256')!=summary.get('dataset_sha256'):
        raise ValueError('crop summary counts differ from recorded audit')
    declarations=provenance.get('config',{}).get('training_groups',[])
    if 'declared_model_training_groups' in summary and summary['declared_model_training_groups']!=sorted({normalize_group(group) for group in declarations}):
        raise ValueError('crop model training declaration differs from provenance')
    history='train-only prototype fit' if provenance['model']=='builtin:prototype' else 'declared' if 'training_groups' in provenance.get('config',{}) else 'unknown'
    if 'model_training_history' in summary and summary['model_training_history']!=history:
        raise ValueError('crop model training history differs from recorded model/config')
    for key,value in _metrics(cases).items():
        if summary['overall'].get(key)!=value:
            raise ValueError(f'crop run aggregate differs from saved cases: {key}')
    statuses={status:_metrics([c for c in cases if c['evaluation_status']==status]) for status in ('prospective','retrospective')}
    groups={group:_metrics([c for c in cases if c['source_group']==group]) for group in sorted({c['source_group'] for c in cases})}
    if statuses!=summary['by_evaluation_status'] or groups!=summary['by_group']:
        raise ValueError('crop run status/group aggregates differ from saved cases')
    return summary,cases


def load_crop_run(path):
    try:return _load_crop_run(path)
    except (KeyError,TypeError,AttributeError,OverflowError) as error:
        raise ValueError(f'invalid crop run schema: {error}') from error


def list_crops(output):
    result=[]
    for path in sorted(Path(output).glob('*/summary.json')):
        try:
            summary,_=load_crop_run(path.parent)
            result.append({'path':str(path.parent.resolve()),'run_id':summary['run_id'],'task':'crop_classification',
                           'model':summary['model'],'snapshot':summary['snapshot'],
                           'exact_set_accuracy':summary['overall']['exact_set_accuracy'],
                           'passed':summary['quality_gates']['passed']})
        except (OSError,ValueError,KeyError):continue
    return result


def compare_crops(left,right):
    """Paired descriptive differences; failed crops remain in denominators."""
    left,lc=load_crop_run(left);right,rc=load_crop_run(right)
    for key in ('dataset_sha256','metric_version','scorer_sha256','snapshot'):
        if not left.get(key) or left.get(key)!=right.get(key):
            raise ValueError(f'crop runs are not comparable: {key} differs or missing')
    lmap={c['sample_id']:c for c in lc};rmap={c['sample_id']:c for c in rc}
    if set(lmap)!=set(rmap):
        raise ValueError('crop runs are not comparable: case sets differ')
    paired=[]
    for ident in sorted(lmap):
        a,b=lmap[ident],rmap[ident]
        for key in ('source_group','evaluation_status','targets','pixel_sha256','candidate_labels'):
            if a.get(key)!=b.get(key):
                raise ValueError(f'crop runs are not comparable: {key} differs for {ident}')
        correct=lambda c: int(not c['error'] and set(c['predicted'])==set(c['targets']))
        paired.append({'sample_id':ident,'source_group':a['source_group'],'evaluation_status':a['evaluation_status'],
                       'exact_set_delta':correct(b)-correct(a),'left_error':a['error'],'right_error':b['error']})
    keys=('exact_set_accuracy','micro_f1','macro_f1','completion_rate')
    delta=lambda a,b:{key:b[key]-a[key] if a.get(key) is not None and b.get(key) is not None else None for key in keys}
    return {'task':'crop_classification','left':left['run_id'],'right':right['run_id'],
            'direction':'right minus left; higher accuracy/F1/completion is better',
            'overall_deltas':delta(left['overall'],right['overall']),
            'by_evaluation_status':{status:delta(left['by_evaluation_status'][status],right['by_evaluation_status'][status]) for status in ('prospective','retrospective')},
            'by_group':{group:delta(left['by_group'][group],right['by_group'][group]) for group in left['by_group']},
            'paired':paired,
            'note':'Descriptive paired comparison of this fixed crop snapshot; crop F1 does not measure interval quality.'}
