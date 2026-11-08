import csv, sys, glob, os, itertools
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bakeoff'))
def load(p):
    rows=[r for r in csv.DictReader(open(p,encoding='utf8'))]
    return [(int(float(r['top_y'])),int(float(r['bottom_y'])),r['lithology'].strip().lower()) for r in rows]
def rowlab(rows,y0,y1):
    a=[None]*(y1-y0)
    for t,b,l in rows:
        for y in range(max(t,y0),min(b,y1)): a[y-y0]=l
    return a
def bnd(rows,y0,y1): return sorted({y for t,b,_ in rows for y in (t,b) if y0<y<y1})
def f1(p,g,tol):
    used=set();m=0
    for x in g:
        c=[j for j,y in enumerate(p) if j not in used and abs(y-x)<=tol]
        if c: j=min(c,key=lambda j:abs(p[j]-x)); used.add(j); m+=1
    P=m/len(p) if p else 0; R=m/len(g) if g else 0
    return 0 if P+R==0 else 2*P*R/(P+R)
def norm(l): return {'sand and sandstone':'sandstone'}.get(l,l)
L='0000000009_d-582_0000020435-0001_log06_fe_5in_md'
g=load(f'gt/gold/{L}.csv'); y0,y1=4404,11195
G=rowlab(g,y0,y1); tol=max(4,min(20,0.01*(y1-y0)))
for lab in 'AB':
    p=load(f'gt/silver/{lab}/{L}.csv'); P=rowlab(p,y0,y1)
    idx=[i for i,x in enumerate(G) if x]
    acc=sum(1 for i in idx if P[i] and norm(P[i])==norm(G[i]))/len(idx)
    unk=sum(1 for i in idx if not P[i] or P[i]=='unknown')/len(idx)
    print(lab,len(p),'intervals','row_acc=%.3f'%acc,'unknown=%.3f'%unk,'bF1=%.3f'%f1(bnd(p,y0,y1),bnd(g,y0,y1),tol), sorted({l for *_,l in p}))
print('gold',len(g),sorted({l for *_,l in g}))
