"""Download immutable review assets and apply the checked exclusion overlay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil
import zipfile

from .groups import normalize_group

PROFILES = {'dev':'review-dev', 'human':'review-human', 'synthetic':'review-synthetic',
            'unlabelled':'review-unlabelled', 'heldout-inputs':'review-heldout-inputs',
            'heldout-references':'review-heldout-references'}
DEFAULT_LOCK=Path(__file__).resolve().parents[2]/'configs/review-data.lock.json'
POLICY_MEMBER='dataset-metadata-v2/POLICY.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    def nonfinite(value):raise ValueError('Metadata JSON must contain finite values')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=nonfinite)


def contained(root, value):
    if not isinstance(value,str) or not value or '\\' in value or ':' in value:
        raise ValueError('Expected a portable relative asset path')
    lexical=root/value
    path=lexical.resolve()
    if not path.is_relative_to(root.resolve()) or path==root.resolve() or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Asset escapes the dataset directory')
    if lexical.absolute()!=path:raise ValueError('Dataset asset paths must not traverse symlinks')
    return path


def _verified(path,entry):
    if (not isinstance(entry,dict) or not isinstance(entry.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',entry['sha256'])
            or type(entry.get('size')) is not int or entry['size']<1):raise ValueError('Metadata lock entry is invalid')
    if not path.is_file() or path.is_symlink() or path.stat().st_size!=entry['size'] or digest(path)!=entry['sha256']:
        raise ValueError('Pinned metadata checksum or size mismatch: '+path.name)


def _scorer_zone(root):
    return any((parent/'SCORER_REFERENCES_ONLY').exists() or (parent/'SCORER_REFERENCES_ONLY').is_symlink() for parent in (root,*root.parents))


def metadata(root, *, lock_path=None):
    root=Path(root).resolve()
    entries=read(lock_path or DEFAULT_LOCK)['bundles']
    manifest_path=contained(root,'metadata/DATASET.json')
    overlay_path=contained(root,'metadata/dataset-containment-overlay-v1.json')
    _verified(manifest_path,entries['review-manifest']);_verified(overlay_path,entries['review-overlay'])
    version=entries['review-metadata-v2']
    archive=contained(root,'downloads/'+version['sha256']+Path(version['asset_name']).suffix)
    _verified(archive,version)
    policy_path=contained(root,'metadata/v2/'+POLICY_MEMBER)
    with zipfile.ZipFile(archive) as bundle:
        member=bundle.getinfo(POLICY_MEMBER)
        if member.file_size>1024**2:raise ValueError('Metadata policy exceeds the supported budget')
        with bundle.open(member) as stream:policy_bytes=stream.read(1024**2+1)
    if not policy_path.is_file() or policy_path.stat().st_size!=len(policy_bytes) or digest(policy_path)!=hashlib.sha256(policy_bytes).hexdigest():
        raise ValueError('Pinned metadata policy differs from the verified archive')
    manifest=read(manifest_path);overlay=read(overlay_path);policy=read(policy_path)
    if digest(manifest_path)!=overlay['appliesTo']['frozenManifestSha256']:
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
    if profile!='heldout-references' and _scorer_zone(root):
        raise ValueError('This directory is reserved for held-out scorer references')
    if profile=='heldout-references':
        if root.exists() and any(root.iterdir()) and not marker.exists():
            raise ValueError('Held-out references need a fresh dedicated directory')
        root.mkdir(parents=True,exist_ok=True);marker.touch()
    helpers=runpy.run_path(str(checkout/'scripts/datasets.py'))
    lock=checkout/'configs/review-data.lock.json'
    entries=read(lock)['bundles']
    cache=contained(root,'downloads')
    meta=contained(root,'metadata');meta.mkdir(parents=True,exist_ok=True)
    for name,target in [('review-manifest',meta),('review-overlay',meta),
                        ('review-evidence',meta/'v1'),('review-metadata-v2',meta/'v2')]:
        archive=helpers['fetch'](name,target,lock,extract=False,cache_dir=cache,use_gh=True)
        if name in ['review-manifest','review-overlay']:
            dest=contained(root,(target/entries[name]['asset_name']).relative_to(root).as_posix())
            if dest.exists() and digest(dest)!=entries[name]['sha256']:
                raise ValueError('Existing metadata differs; use a fresh data directory')
            if not dest.exists():shutil.copyfile(archive,dest)
        else:helpers['safe_extract'](archive,target)
    manifest,overlay,policy=metadata(root,lock_path=lock)
    entry=entries[PROFILES[profile]]
    expanded=sum(info['bytes'] for info in manifest['files'].values() if info['bundle']==entry['asset_name'])
    cached=contained(root,'downloads/'+entry['sha256']+Path(entry['asset_name']).suffix)
    valid_cache=cached.is_file() and cached.stat().st_size==entry['size'] and digest(cached)==entry['sha256']
    needed=expanded+(0 if valid_cache else entry['size'])+64*1024**2
    if shutil.disk_usage(root).free<needed:
        raise RuntimeError(f'Need {needed/1024**3:.2f} GiB free for this archive. Choose --data-dir on a drive with more space.')
    archive=helpers['fetch'](PROFILES[profile],root,lock,extract=False,cache_dir=cache,use_gh=True)
    helpers['safe_extract'](archive,root)
    manifest,overlay,policy=metadata(root,lock_path=lock)
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
        contained(root,'metadata/human-eligible.json').write_text(json.dumps(dict(records=selected,
            pages=15,approvedBoxes=422,blindVerification=False,fullIntervalScoring=False),indent=2),encoding='utf-8')
    print(f'DOWNLOAD_OK profile={profile} files={checked} auto_label=false independent_accuracy=false')
    return root


def development_manifest(root, *, quick=True,lock_path=None):
    root=Path(root).resolve()
    if _scorer_zone(root):raise ValueError('Scorer directory cannot be a model input root')
    original,overlay,policy=metadata(root,lock_path=lock_path)
    permissions={r['recordId']:r for r in overlay['records']}
    blocked_groups={normalize_group(group) for permission in overlay['records'] if permission['protectedWell'] or permission['denyTrain'] for group in permission['groups']}
    dev_bundle=read(lock_path or DEFAULT_LOCK)['bundles']['review-dev']['asset_name']
    selected=[];groups=set()
    for record in original['records']:
        if record['selection']!='ext_dev' or record['ready'] is not True:continue
        permission=permissions.get(record['id'])
        if not permission or permission['protectedWell'] or permission['denyTrain']:
            raise ValueError('Development record failed the protected-well screen: '+record['id'])
        group=record.get('groupId')
        if not group or normalize_group(group) not in {normalize_group(value) for value in permission['groups']}:raise ValueError('Unbound development well identity')
        group=normalize_group(group)
        if group in blocked_groups:raise ValueError('Development group failed the protected-well screen: '+record['id'])
        if quick and group in groups:continue
        item=dict(id=record['id'],source='external_labelled',split='dev',group=group,
            window=[record['y0'],record['y1']],truth_window=[record['y0'],record['y1']],
            label_provenance=dict(kind='programmatic source-log extraction; not human-certified',
                                  independent_accuracy=False),tags=['exposed-development'])
        for key in ['image','annotation','truth']:
            info=original['files'][record[key]]
            if info['bundle']!=dev_bundle:raise ValueError('Development input belongs to another bundle: '+record['id'])
            path=contained(root,record[key])
            if digest(path)!=info['sha256']:raise ValueError('Development input hash mismatch: '+record['id'])
            item[key]=record[key];item[key+'_sha256']=info['sha256']
        selected.append(item);groups.add(group)
        if quick and len(selected)==2:break
    if not selected or (quick and len(selected)!=2):raise ValueError('Required development cases are missing')
    dest=contained(root,'quick-dev.json' if quick else 'outside-dev.json')
    dest.write_text(json.dumps(dict(schema_version=1,name='review-dev-v1-with-containment-v2-notice',
        records=selected,independent_accuracy=False,autoLabelAllowed=False),indent=2),encoding='utf-8')
    return dest
