from audit import *
from types import SimpleNamespace
from genlab import dataset,realism
from genlab.assemble import compose,noise
result={}
policy=read(ROOT/'data/normalized/split-policy.json')
def istest(did):
    group=policy['document_groups'][did]
    return int(hashlib.sha256((policy['seed']+':'+group).encode()).hexdigest()[:8],16)%5==0
for version in ('v1','v5'):
    folder,manifest,pairs=match(version,'augmented',exact=True)
    feats=[]; ys=[]; train=[]
    for r,f in pairs:
        ar,af=load(folder/r['file']),load(folder/f['file'])
        h=min(len(ar),len(af),144)
        feats.extend([features(ar[:h]),features(af[:h])]);ys.extend([1,0]);train.extend([r['split']=='train']*2)
    x=np.array([[f['rail']] for f in feats]);y=np.array(ys);tr=np.array(train)
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500)).fit(x[tr],y[tr])
    result[version]=dict(exact_pairs=len(pairs),validation_n=int((~tr).sum()),
        rail_accuracy=float(balanced_accuracy_score(y[~tr],clf.predict(x[~tr]))),
        real_edge_over80pct=int(sum(f['rail']>.8 for i,f in enumerate(feats) if not tr[i] and y[i]==1)),
        fake_edge_over80pct=int(sum(f['rail']>.8 for i,f in enumerate(feats) if not tr[i] and y[i]==0)),
        pair_docs_test=sum(istest(r['doc']) or istest(f['doc']) for r,f in pairs))
    if version=='v5':
        examples=[]
        for did in ['3515004b6f643c87','5fbeaf2a0650e013','952caeadf1d8467c','d368760651ded938','b9ba393ab59601f2','7dc28eda27f5c429']:
            pp=[(r,f) for r,f in pairs if r['doc']==did and r['split']=='validation']
            if pp: examples.append(pp[0])
        clean={(f['class_id'],f['file'].split('/')[-1]):f for f in manifest if f['variant']=='clean'}
        canvas=Image.new('RGB',(1000,110+len(examples)*340),'#eeeeee');d=ImageDraw.Draw(canvas)
        d.text((10,8),'v5 EXACT CLASS: native scale x2; top-left crop; shared h/w per row; no resize distortion',fill='black')
        for col,title in enumerate(['Real accepted crop','Clean fake','Augmented fake']):d.text((15+330*col,40),title,fill='black')
        meta=[]
        for i,(r,f) in enumerate(examples):
            cc=[c for c in manifest if c['class_id']==f['class_id'] and c['variant']=='clean' and c['format']=='strip']
            # Select corresponding native height (four distinct heights in v5).
            af=load(folder/f['file']);c=next(c for c in cc if load(folder/c['file']).shape==af.shape)
            ars=[load(folder/r['file']),load(folder/c['file']),af]
            h=min(144,*[len(a) for a in ars]);w=min(a.shape[1] for a in ars)
            for j,a in enumerate(ars):
                canvas.paste(Image.fromarray(a[:h,:w]).resize((w*2,h*2),Image.Resampling.NEAREST),(15+j*330,100+i*340))
            d.text((15,80+i*340),r['class_id']+' '+r['name']+f' | {w}x{h} px',fill='black')
            meta.append(dict(real=r['file'],clean=c['file'],augmented=f['file'],class_id=r['class_id'],native_wh=[w,h]))
        canvas.save(OUT/'real-clean-augmented.png')
        (OUT/'image-sources.json').write_text(json.dumps(meta,indent=1),encoding='utf-8')
# Exercise the current 5% cap on an actual usable residual, retaining its measurement.
folder=ROOT/'data/genlab/datasets/v5'
ms=read(folder/'realism.json')['measurements']
candidates=[m for m in ms if m['overlay_usable'] and m['overlay_fraction']>.10 and not m['rails']]
if not candidates: candidates=[m for m in ms if m['overlay_usable'] and m['overlay_fraction']>.10]
m=max(candidates,key=lambda m:m['overlay_fraction'])
mask=load(folder/m['mask'])[:,:,0];colors=load(folder/m['colours'])
clean=np.full(colors.shape,220,np.uint8)
aug,params=dataset.dataset_augment(clean,m,(mask,colors),np.random.default_rng(19))
oy,ox=params['overlay_offset'];yy=(np.arange(len(mask))+oy)%len(mask);xx=(np.arange(mask.shape[1])+ox)%mask.shape[1]
shift=mask[yy[:,None],xx[None,:]]>0
actual=(aug!=realism.augment(clean,dict(m,blur=min(.7,m['blur']),noise=min(6,m['noise']),drift=np.clip(m['drift'],-12,12).tolist()),None,np.random.default_rng(19))[0]).any(2)
# Reconstruct exact thinning RNG before augment consumes it.
rng=np.random.default_rng(19);keep=rng.choice(np.flatnonzero(mask),max(1,int(mask.size*.05)),replace=False)
thinned=np.zeros(mask.shape,bool);thinned.flat[keep]=True
def components(b):
    labels,n=ndimage.label(b);sizes=np.bincount(labels.ravel())[1:]
    return dict(pixels=int(b.sum()),components=n,single_pixel_fraction=float(sizes[sizes==1].sum()/max(1,sizes.sum())),largest=int(sizes.max(initial=0)))
result['thinning']=dict(id=m['id'],class_id=m['class_id'],source=components(mask>0),capped=components(thinned),params=params)
canvas=Image.new('RGB',(1000,520),'white');draw=ImageDraw.Draw(canvas)
for j,(title,a) in enumerate([('Actual residual source',colors),('Original residual mask',np.repeat(mask[:,:,None],3,2)),('5% randomly thinned mask',np.repeat((thinned*255).astype(np.uint8)[:,:,None],3,2)),('Augmented neutral fill',aug)]):
    h=min(len(a),200);w=a.shape[1];canvas.paste(Image.fromarray(a[:h]).resize((2*w,2*h),Image.Resampling.NEAREST),(10+j*245,70));draw.text((10+j*245,15),title,fill='black')
canvas.save(OUT/'residual-thinning.png')
# Isolate the live assembler scan field from overlays and boundaries.
lib=read(noise.NOISE/'measurements.json')
nonzero=[]
for mm in lib['measurements']:
    with np.load(noise.NOISE/mm['residual']) as cached:
        field,valid=cached['residual'],cached['valid']
    n=int(np.count_nonzero(field[valid]));
    if n: nonzero.append((n,mm))
m=max(nonzero,key=lambda x:x[0])[1]
cached=np.load(noise.NOISE/m['residual']);h,w=cached['residual'].shape[:2]
one=dict(m,overlay_fraction=0)
fake_context=SimpleNamespace(realism=dict(measurements=[one],boundaries={},rails={}),family=lambda did:m['family'])
source=np.full((1024,w,3),128,np.uint8)
aug,params,events=compose.Composer.appearance(fake_context,m['doc_id'],source,[],np.random.default_rng(19),2)
result['assembler']=dict(measurements=len(lib['measurements']),nonzero_fields=len(nonzero),
    test_sources=sum(istest(mm['source_doc']) or istest(mm['doc_id']) for mm in lib['measurements']),
    field_id=m['id'],field_shape=[h,w],lag_equal_fraction=float((aug[:512-h]==aug[h:512]).all(2).mean()),
    residual_std=float(aug[:512].std()),blur_counts=dict(Counter(mm['blur'] for mm in lib['measurements'])))
canvas=Image.new('RGB',(600,1100),'white');d=ImageDraw.Draw(canvas)
d.text((5,5),f'Live assembler isolated scan field; x2; period {h} source rows',fill='black')
canvas.paste(Image.fromarray(aug[:512]).resize((w*2,1024),Image.Resampling.NEAREST),(20,50))
for y0 in range(0,512,h):d.line((15,50+y0*2,20+2*w,50+y0*2),fill='red',width=1)
canvas.save(OUT/'assembler-period.png')
(OUT/'probes.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
