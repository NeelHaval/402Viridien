from audit import *
import model,tifffile
from genlab import score
from sklearn.linear_model import SGDClassifier
result={}
for version in ('v1','v5'):
    folder,manifest,pairs=match(version,'augmented')
    by_mode=defaultdict(list)
    for r,f in pairs:
        ar,af=load(folder/r['file']),load(folder/f['file'])
        h=min(len(ar),len(af),144);w=min(ar.shape[1],af.shape[1])
        for a in (ar,af):
            for mode in ('original','remove-edge3','same-size-interior'):
                b=a
                if mode=='remove-edge3': b=a[:,3:-3]
                if mode=='same-size-interior':
                    left=(a.shape[1]-w)//2;b=a[:h,left+3:left+w-3]
                down=np.asarray(Image.fromarray(b).resize((12,20),Image.Resampling.BILINEAR),dtype=np.float32).ravel()/255
                by_mode[mode].append(np.r_[score.features(b),down])
    y=np.tile([1,0],len(pairs));tr=np.repeat([r['split']=='train' for r,f in pairs],2)
    result[version]={}
    for mode,x in by_mode.items():
        x=np.array(x)
        clf=make_pipeline(StandardScaler(),SGDClassifier(loss='log_loss',max_iter=200,tol=.001,random_state=11,n_jobs=1))
        clf.fit(x[tr],y[tr]);pred=clf.predict(x[~tr])
        result[version][mode]=dict(correct=int((pred==y[~tr]).sum()),n=int((~tr).sum()),accuracy=float((pred==y[~tr]).mean()))
# Matched native stroke widths, no geometric interpolation.
images=read(OUT/'image-sources.json');r=next(r for r in images if r['class_id']=='7dc28eda27f5c429:23')
weights={}
for kind in ('real','clean','augmented'):
    a=load(ROOT/'data/genlab/datasets/v5'/r[kind]);a=a[:96,3:-3];ink=a.max(2)<80
    lengths=[]
    for col in ink.T:
        edges=np.diff(np.r_[False,col,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        lengths.extend((ends-starts).tolist())
    weights[kind]=dict(median_vertical_ink_run=float(np.median(lengths)),run_distribution=dict(Counter(lengths)),dark_fraction=float(ink.mean()),palette=len(np.unique(a.reshape(-1,3),axis=0)))
result['stroke_width']=weights
store=model.Store();policy=read(ROOT/'data/normalized/split-policy.json');headers=[]
for entry in store.catalog:
    doc=store.get(entry['id']);group=policy['document_groups'][doc['id']]
    if int(hashlib.sha256((policy['seed']+':'+group).encode()).hexdigest()[:8],16)%5==0:continue
    with tifffile.TiffFile(model.DATASET/'images'/doc['image_name']) as f:
        p=f.pages[0]
        headers.append(dict(doc=doc['id'],compression=str(p.compression.name),bits=p.bitspersample,photometric=p.photometric.name))
result['tiff_headers']=headers
result['tiff_summary']=dict(n=len(headers),compression=dict(Counter(h['compression'] for h in headers)),bits=dict(Counter(str(h['bits']) for h in headers)),photometric=dict(Counter(h['photometric'] for h in headers)))
# Capture source fingerprints because builders may edit concurrently.
result['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'genlab/dataset.py',ROOT/'genlab/realism.py',ROOT/'genlab/assemble/noise.py',ROOT/'genlab/assemble/compose.py']}
(OUT/'ablations.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='tiff_headers'}),flush=True)
