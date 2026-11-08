"""Versioned well-level split permissions, derived from metadata only."""
import hashlib,json,re
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
REGISTRY=BASE/'split-registry-v1.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(value):
    s=str(value or '').replace('\\','/').split('/')[-1].casefold()
    for ext in ['.tiff','.tif','.json']:s=s.removesuffix(ext)
    if s.startswith('current:'):s=s[8:]
    if 'd-582' in s:return 'well:d-582'
    if 'mc765' in s and 'p011' in s:return 'well:mc765_p011'
    m=re.search(r'(?:api:)?(608\d{7,11})',s)
    if m:return 'api:'+m.group(1)[:10]
    if 'critter' in s and '201' in s:return 'document:crittercreek2011807h'
    if 'rionueces' in re.sub(r'[^a-z0-9]','',s):return 'document:rionueces3744unit1212'
    if 'shell' in s and '08852' in s:return 'document:shell08852ke001'
    return s
def registry():
    j=read(REGISTRY)
    if digest(ROOT/'viridien_work/bakeoff/eval_set.json')!=j['evalMetadataSha256']:raise ValueError('Protected split metadata changed; rebuild versioned registry')
    return j
def identity(record,reg=None):
    reg=reg or registry();aliases=reg['aliases'];values=[]
    for key in ['groupId','baseWell','parentGroup','base_document','baseDocument','baseImage','base_image','originalSourceImage']:
        if record.get(key):values.append(record[key])
    for key in ['ancestors','ancestor_wells','training_groups']:
        values.extend(record.get(key,[]) if isinstance(record.get(key,[]),list) else [])
    lineage=record.get('lineage',{})
    if isinstance(lineage,dict):
        for k in ['baseWell','groupId','baseDocument','sourceDocumentId','base_image','baseImage']:
            if lineage.get(k):values.append(lineage[k])
        for k in ['ancestors','ancestor_wells']:values.extend(lineage.get(k,[]) if isinstance(lineage.get(k,[]),list) else [])
    for v in [record.get('image'),record.get('imagePath'),record.get('paths',{}).get('image'),record.get('log')]:
        if v:values.append(v)
    result=set()
    for v in values:
        if not isinstance(v,str):continue
        c=canonical(v);c=aliases.get(c,c)
        if c in aliases:c=aliases[c]
        explicit=v in [record.get('groupId'),record.get('baseWell'),record.get('parentGroup')] and v!=record.get('id') and c not in ['unknown','none','null','unresolved','missing']
        if c and (explicit or c.startswith(('api:','well:','document:')) or canonical(v) in aliases):result.add(c)
    return sorted(result)
def permission(record,reg=None):
    reg=reg or registry();groups=identity(record,reg);protected=sorted(set(groups)&set(reg['protectedGroups']))
    return dict(registryVersion=reg['version'],registrySha256=digest(REGISTRY),groups=groups,protectedGroups=protected,protectedWell=bool(protected),allowTrain=bool(groups) and not protected,allowTune=bool(groups) and not protected,allowLabelGeneration=bool(groups) and not protected,allowConfirm=bool(groups) and not protected,evaluationAllowed=True,truthIsolatedRequired=bool(protected),independentAccuracyClaimAllowed=False if protected else None,reason='Protected evaluation well or descendant' if protected else ('Missing well identity: training denied' if not groups else 'No protected-well match'))
def require_training(record):
    p=permission(record)
    if not p['allowTrain']:raise ValueError(p['reason']+': '+str(record.get('id','unknown')))
    return p
def contain(record,reg=None):
    p=permission(record,reg);record['splitPolicy']=p;record['trainingAllowed']=p['allowTrain']
    if p['protectedWell']:
        record['protectedWell']=True;record['protectedForTraining']=True
        record['priorSplit']=record.get('priorSplit',record.get('split'));record['split']='protected-eval-exposed'
        record['priorTruthStatus']=record.get('priorTruthStatus',record.get('truthStatus'));record['labelsPreviouslyExposed']=True
        record['labelsHeldOut']=True;record['showTruthInReview']=False;record['modelReadableTruth']=False
        record['reviewPolicy']=dict(sourceOnly=True,allowTruth=False,allowConfirm=False,allowTrain=False,allowLabelGeneration=False)
        record['truth']=None;record['truthBoxes']=[];record['intervals']=[];record['reviewSuggestions']=[];record.pop('modelReview',None)
        record['boxes']=record.get('sourceBoxes',record.get('boxes',[]))
        record['boxes']=[b for b in record['boxes'] if b.get('label') in ['depth_column','interpreted_lithology_column','lithology_legend']]
        record['ready']=False;record['diagnosticOnly']=True;record['reviewPriority']=0
        record['thumbnailKind']='source-only';record['imageKind']='source-only';record['boxesKind']='source-only'
        for m in record.get('metricReadiness',{}).values():
            if isinstance(m,dict):m.update(ready=False,reason='Protected well: existing labels exposed; scoring requires isolated truth and exposure declaration')
        if not any(x.get('code')=='protected_eval_well' for x in record.setdefault('issues',[])):record['issues'].append(dict(code='protected_eval_well',severity='error',message='Training, tuning, label generation and review confirmation are blocked for this evaluation well and descendants. Existing labels are exposed; they are not unseen truth.'))
    return record
def build():
    path=ROOT/'viridien_work/bakeoff/eval_set.json';cases=read(path)
    documents=read(ROOT/'viridien_work/pattern_lab/data/documents.json')
    aliases={canonical(d['id']):canonical(d['name']) for d in documents}
    for d in documents:aliases[canonical(d['name'])]=canonical(d['name'])
    groups=sorted({canonical(c['log']) for c in cases})
    j=dict(version='protected-wells-v1',evalMetadata=str(path),evalMetadataSha256=digest(path),metadataOnly=True,protectedLogs=[c['log'] for c in cases],protectedGroups=groups,aliases=aliases,policy='Deny train/tune/label-generation/review-confirmation for protected wells and descendants. Eval only with isolated truth; prior labels remain exposed.',frozenParentChanged=False)
    if REGISTRY.exists() and read(REGISTRY)!=j:raise ValueError('Versioned registry exists with different metadata; use a new version')
    REGISTRY.write_text(json.dumps(j,indent=2),encoding='utf-8')
    print('SPLIT_REGISTRY_BUILT logs='+str(len(cases))+' protected_wells='+str(len(groups))+' metadata_only=true')
if __name__=='__main__':build()
