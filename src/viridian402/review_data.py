"""Download immutable review assets and apply the checked exclusion overlay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import runpy
import shutil

PROFILES = {'dev':'review-dev', 'human':'review-human', 'synthetic':'review-synthetic',
            'unlabelled':'review-unlabelled', 'heldout-inputs':'review-heldout-inputs',
            'heldout-references':'review-heldout-references'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def contained(root, value):
    if not isinstance(value,str) or '\\' in value or ':' in value:
        raise ValueError('Expected a portable relative asset path')
    path=(root/value).resolve()
    if not path.is_relative_to(root.resolve()) or Path(value).is_absolute():
        raise ValueError('Asset escapes the dataset directory')
    return path


def metadata(root):
    manifest=read(root/'metadata/DATASET.json')
    overlay=read(root/'metadata/dataset-containment-overlay-v1.json')
    policy=read(root/'metadata/v2/dataset-metadata-v2/POLICY.json')
    if digest(root/'metadata/DATASET.json')!=overlay['appliesTo']['frozenManifestSha256']:
        raise ValueError('Overlay belongs to another dataset snapshot')
    if policy['autoLabelAllowed'] is not False or policy['modelApiCallsAllowed'] is not False:
        raise ValueError('Expected the checked metadata-only v2 policy')
    if policy['protectedRegistrySha256']!=overlay['authority']['sha256']:
        raise ValueError('Metadata policy and exclusion overlay use different registries')
    return manifest,overlay,policy


def download(profile, root, checkout, *, allow_heldout_references=False):
    root=Path(root).resolve(); checkout=Path(checkout).resolve()
    if profile=='heldout-references' and not allow_heldout_references:
        raise ValueError('Held-out references require --allow-heldout-references and a separate --data-dir for the scorer')
    if profile not in PROFILES:raise ValueError('Unknown review download profile')
    marker=root/'SCORER_REFERENCES_ONLY'
    if profile!='heldout-references' and marker.exists():
        raise ValueError('This directory is reserved for held-out scorer references')
    if profile=='heldout-references':
        if root.exists() and any(root.iterdir()) and not marker.exists():
            raise ValueError('Held-out references need a fresh dedicated directory')
        root.mkdir(parents=True,exist_ok=True);marker.touch()
    helpers=runpy.run_path(str(checkout/'scripts/datasets.py'))
    lock=checkout/'configs/review-data.lock.json'
    entries=read(lock)['bundles']
    cache=root/'downloads'
    meta=root/'metadata';meta.mkdir(parents=True,exist_ok=True)
    for name,target in [('review-manifest',meta),('review-overlay',meta),
                        ('review-evidence',meta/'v1'),('review-metadata-v2',meta/'v2')]:
        archive=helpers['fetch'](name,target,lock,extract=False,cache_dir=cache,use_gh=True)
        if name in ['review-manifest','review-overlay']:
            dest=target/entries[name]['asset_name']
            if dest.exists() and digest(dest)!=entries[name]['sha256']:
                raise ValueError('Existing metadata differs; use a fresh data directory')
            shutil.copyfile(archive,dest)
        else:helpers['safe_extract'](archive,target)
    manifest,overlay,policy=metadata(root)
    entry=entries[PROFILES[profile]]
    expanded=sum(info['bytes'] for info in manifest['files'].values() if info['bundle']==entry['asset_name'])
    cached=cache/(entry['sha256']+Path(entry['asset_name']).suffix)
    needed=expanded+(0 if cached.exists() else entry['size'])+64*1024**2
    if shutil.disk_usage(root).free<needed:
        raise RuntimeError(f'Need {needed/1024**3:.2f} GiB free for this archive. Choose --data-dir on a drive with more space.')
    archive=helpers['fetch'](PROFILES[profile],root,lock,extract=False,cache_dir=cache,use_gh=True)
    helpers['safe_extract'](archive,root)
    manifest,overlay,policy=metadata(root)
    bundle_name=entries[PROFILES[profile]]['asset_name']
    checked=0
    for name,info in manifest['files'].items():
        if info['bundle']!=bundle_name:continue
        path=contained(root,name)
        if path.stat().st_size!=info['bytes'] or digest(path)!=info['sha256']:
            raise ValueError('Extracted file hash mismatch: '+name)
        checked+=1
    if profile=='human':
        ids=set(overlay['humanSelection']['includedRecordIds'])
        selected=[r for r in manifest['records'] if r['id'] in ids]
        if len(selected)!=15:raise ValueError('Expected 15 eligible human diagnostic pages')
        (meta/'human-eligible.json').write_text(json.dumps(dict(records=selected,
            pages=15,approvedBoxes=422,blindVerification=False,fullIntervalScoring=False),indent=2),encoding='utf-8')
    print(f'DOWNLOAD_OK profile={profile} files={checked} auto_label=false independent_accuracy=false')
    return root


def development_manifest(root, *, quick=True):
    root=Path(root).resolve()
    if (root/'SCORER_REFERENCES_ONLY').exists():raise ValueError('Scorer directory cannot be a model input root')
    original,overlay,policy=metadata(root)
    permissions={r['recordId']:r for r in overlay['records']}
    selected=[];groups=set()
    for record in original['records']:
        if record['selection']!='ext_dev' or not record['ready']:continue
        permission=permissions.get(record['id'])
        if not permission or permission['protectedWell'] or permission['denyTrain']:
            raise ValueError('Development record failed the protected-well screen: '+record['id'])
        group=record.get('groupId')
        if not group or group not in permission['groups']:raise ValueError('Unbound development well identity')
        if quick and group in groups:continue
        item=dict(id=record['id'],source='external_labelled',split='dev',group=group,
            window=[record['y0'],record['y1']],truth_window=[record['y0'],record['y1']],
            label_provenance=dict(kind='programmatic source-log extraction; not human-certified',
                                  independent_accuracy=False),tags=['exposed-development'])
        for key in ['image','annotation','truth']:
            path=contained(root,record[key]);info=original['files'][record[key]]
            if digest(path)!=info['sha256']:raise ValueError('Development input hash mismatch: '+record['id'])
            item[key]=record[key];item[key+'_sha256']=info['sha256']
        selected.append(item);groups.add(group)
        if quick and len(selected)==2:break
    if not selected or (quick and len(selected)!=2):raise ValueError('Required development cases are missing')
    dest=root/('quick-dev.json' if quick else 'outside-dev.json')
    dest.write_text(json.dumps(dict(schema_version=1,name='review-dev-v1-with-containment-v2-notice',
        records=selected,independent_accuracy=False,autoLabelAllowed=False),indent=2),encoding='utf-8')
    return dest
