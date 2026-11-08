import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from genlab.dataset import *
from genlab.dataset_eval import Features,fit_predict,summarize
from genlab.swatch import remove_frame
f=ROOT/'v1'
rows=[json.loads(s) for s in (f/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
t=[r for r in rows if r['format']=='strip' and r['variant']=='clean' and r['split']=='train']
truth=[r for r in read(f/'real.json') if r['split']=='validation']
feat=Features(f)
class Trim(Features):
    def get(self,row):
        rgb=np.asarray(Image.open(self.folder/row['file']).convert('RGB'))
        rgb,_=remove_frame(rgb)
        return score.features(rgb),np.asarray(Image.fromarray(rgb).resize((12,20),Image.Resampling.BILINEAR),dtype=np.float32).ravel()/255
for alg in ['matcher','logistic']:
    for mode,features in [('original',feat),('frame-removed',Trim(f))]:
        p=fit_predict(features,t,truth,alg)
        print(alg,mode,summarize(truth,p,{r['name'] for r in t})['accuracy'])
x=np.array([feat.get(r)[0] for r in t]); y=np.array([r['name'] for r in t])
names=sorted(set(y)); centers=np.array([x[y==n].mean(0) for n in names])
for name in ['shale','anhydrite']:
    rr=[r for r in truth if r['name']==name]
    for r in rr[:2]:
        dist=((centers-feat.get(r)[0])**2).sum(1)
        print('distances',name,r['doc'],[(names[i],round(float(dist[i]),3)) for i in dist.argsort()[:5]],
              'true',round(float(dist[names.index(name)]),3))
# Nearest document prototype rather than a mean over different colour conventions.
groups=sorted({r['class_id'] for r in t})
cc=np.array([np.mean([feat.get(r)[0] for r in t if r['class_id']==c],0) for c in groups])
lookup={r['class_id']:r['name'] for r in t}
p=[lookup[groups[int(((cc-feat.get(r)[0])**2).sum(1).argmin())]] for r in truth]
print('cross-log document prototypes',summarize(truth,p,set(y))['accuracy'],'shale->anhydrite',sum(r['name']=='shale' and n=='anhydrite' for r,n in zip(truth,p)))
print('train names',len(names),'unseen boxes',sum(r['name'] not in names for r in truth))
