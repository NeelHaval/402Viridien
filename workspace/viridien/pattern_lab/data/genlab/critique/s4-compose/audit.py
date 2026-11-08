import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter
from genlab.assemble import compose, noise

ROOT = Path('data/genlab/critique/s4-compose')
OUT = Path('data/genlab/assemble/out/starter')
read = noise.read

def summary():
    c = compose.Composer()
    try:
        for did in ['5709fe9e86ce942b','97adb9d2c69d860b','4a056c8ff25cf749','d368760651ded938']:
            doc = c.docs[did]
            choices, skipped = c.available(did)
            print('DOC',did,doc['name'],'bbox',doc['boxes']['interpreted_lithology_column'])
            print('AVAILABLE',[(k,c.stats['classes'][k]['name']) for k in choices])
            print('SKIPPED',[(r['class_id'],c.stats['classes'][r['class_id']]['name'],r['reason']) for r in skipped])
            b=c.stats['levels']['doc:'+did]
            print('COMPOSITION',b['composition'])
            for p in sorted(OUT.glob('*/truth.json')):
                t=read(p)
                if t['base_doc'] != did: continue
                print('OUTPUT',p.parent.name,'origin',t['real_depth_span'],'zones',t['zone_start_fraction'],t['zone_span_fraction'])
                print('ROW_SHARES',Counter({k:sum(l['bottom']-l['top'] for l in t['layers'] if l['class_id']==k)/t['height'] for k in set(l['class_id'] for l in t['layers'])}))
                print('FIRST',[(l['top'],l['bottom'],l['class_id'],l['zone']) for l in t['layers'][:7]])
                break
    finally: c.close()

def pages():
    c=compose.Composer()
    try:
        for p in sorted(OUT.glob('page-*/page-bands.json')):
            folder=p.parent
            if not (folder/'bbox_annotation.json').exists():
                print('INCOMPLETE_PAGE',folder.name); continue
            t=read(folder/'truth.json'); doc=c.docs[t['base_doc']]
            x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column']
            with Image.open(folder/'column.png') as im: col=np.asarray(im.convert('RGB'))
            changed=0; total=0; outside=0; best=None; bands=read(p)['bands']
            for b in bands:
                with Image.open(folder/b['file']) as im: band=np.asarray(im.convert('RGB'))
                a,z=max(y0,b['top']),min(y1,b['bottom'])
                if a>=z: continue
                embedded=band[a-b['top']:z-b['top'],x0:x1]
                different=np.any(embedded != col[a-y0:z-y0],axis=2)
                n=int(different.sum()); changed+=n; total+=different.size
                if best is None or n>best[0]: best=(n,b,a,z)
            n,b,a,z=best
            real=c.windows.get(doc,[x0-70,a,x1+70,z])
            with Image.open(folder/b['file']) as im: syn=np.asarray(im.convert('RGB'))[a-b['top']:z-b['top'],x0-70:x1+70]
            canvas=Image.new('RGB',(real.shape[1]*2+30,len(real)+42),'white')
            d=ImageDraw.Draw(canvas); d.text((4,4),'Real page',fill='black');d.text((real.shape[1]+20,4),'Synthetic page',fill='black')
            canvas.paste(Image.fromarray(real),(0,32));canvas.paste(Image.fromarray(syn),(real.shape[1]+20,32))
            dest=ROOT/('page-'+t['base_doc']+'.png');canvas.save(dest)
            print('PAGE_DIFF',folder.name,'changed',changed,'pixels',total,'fraction',changed/total,'crossing',t.get('page'),'max_band',b['top'],'picture',str(dest))
            with Image.open(folder/b['file']) as im: full=np.asarray(im.convert('RGB'))
            native=c.windows.get(doc,[0,b['top'],doc['width'],b['bottom']])
            outside=int(np.any(full[:,:x0]!=native[:,:x0],axis=2).sum()+np.any(full[:,x1:]!=native[:,x1:],axis=2).sum())
            print('OUTSIDE_DIFF_MAX_BAND',outside)
    finally: c.close()

def stretches():
    c=compose.Composer()
    try:
        for did,top in [('5709fe9e86ce942b',18368),('97adb9d2c69d860b',6373),('4a056c8ff25cf749',32599)]:
            folders=sorted(p.parent for p in OUT.glob('column-*/truth.json') if read(p)['base_doc']==did)[:2]
            c.sheet(did,folders,ROOT/('stretch-'+did+'.png'),rows=1200,depth_top=top)
            print('SHEET',did,read(ROOT/('stretch-'+did+'.json')))
            for f in folders:
                t=read(f/'truth.json'); offset=top-t['real_depth_span'][0]
                ls=[l for l in t['layers'] if l['top']<offset+1200 and l['bottom']>offset]
                print('STRETCH',f.name,'offset',offset,'zones',sorted(set(l['zone'] for l in ls)),'layers',len(ls),
                      'same_neighbors',sum(a['class_id']==b['class_id'] for a,b in zip(ls,ls[1:])),
                      'shares',{k:sum(max(0,min(l['bottom'],offset+1200)-max(l['top'],offset)) for l in ls if l['class_id']==k)/1200 for k in set(l['class_id'] for l in ls)})
                print('MODELS',[(k,c.models[v['pattern_id']]['kind'],c.models[v['pattern_id']].get('anchored'),c.models[v['pattern_id']].get('tile',np.zeros((0,0))).shape) for k,v in c.available(did)[0].items()])
    finally:c.close()

def crossings():
    c=compose.Composer()
    try:
        for p in OUT.glob('page-*/page-bands.json'):
            t=read(p.parent/'truth.json');doc=c.docs[t['base_doc']]
            x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column']
            found=[];lost=[]
            for band in read(p)['bands']:
                a,b=max(y0,band['top']),min(y1,band['bottom'])
                if a>=b:continue
                real=c.windows.get(doc,[x0-6,a,x1+6,b])
                with Image.open(p.parent/band['file']) as im:fake=np.asarray(im.convert('RGB'))[a-band['top']:b-band['top'],x0-6:x1+6]
                rd=real.max(2)<100;fd=fake.max(2)<100
                candidate=(rd[:,9:-9].mean(1)>.85)&((rd[:,:6].mean(1)>.8)|(rd[:,-6:].mean(1)>.8))
                broken=fd[:,9:-9].mean(1)<.85
                found.extend((np.where(candidate)[0]+a).tolist());lost.extend((np.where(candidate&broken)[0]+a).tolist())
            print('GRID_CROSSINGS',doc['id'],'real',len(found),'lost',len(lost),'example',lost[:20])
            mm=[m for m in c.realism['measurements'] if m['source_doc']==doc['id'] and m['overlay_usable']]
            coverage=np.zeros(y1-y0,bool)
            for m in mm:coverage[max(0,m['box'][1]-y0):min(y1-y0,m['box'][3]-y0)]=True
            print('CROSSING_LIBRARY_COVERAGE',doc['id'],int(coverage.sum()),len(coverage),'fraction',float(coverage.mean()))
    finally:c.close()

def probes():
    classes={'d:a':dict(norm='a'), 'd:b':dict(norm='b')}
    block=dict(n_layers=2,zones=[dict(name='all',top_frac=0,bottom_frac=1)],
        composition={'all':{'d:a':.5,'d:b':.5}}, transitions={},
        thickness={k:dict(dist='lognormal',mu=float(np.log(h)),sigma=0,n=100) for k,h in [('d:a',10),('d:b',100)]},
        confidence=dict(composition='medium',transitions='low'),gaps={})
    stats=dict(classes=classes,levels={'doc:d':block})
    choices={k:dict(pattern_source='document') for k in classes}
    s=compose.Sampler(stats,'d','f',choices,173)
    seq=s.sequence(10**9,max_layers=10000)
    counts=Counter(l['class_id'] for l in seq);rows=Counter()
    for l in seq:rows[l['class_id']]+=l['bottom']-l['top']
    print('FALLBACK_ROW_BIAS','requested_row_shares',block['composition']['all'],'layer_counts',dict(counts),'row_shares',{k:v/sum(rows.values()) for k,v in rows.items()})
    c=compose.Composer()
    try:
        did='4a056c8ff25cf749'; f=OUT/'column-mc765-colour-grid-4a056c8ff25cf749-00';t=read(f/'truth.json')
        choices,_=c.available(did); data=c.models[choices[did+':0']['pattern_id']]
        from genlab import render
        sampler=compose.Sampler(c.stats,did,c.family(did),choices,t['seed'])
        sampled=sampler.sequence(t['height'],start_frac=t['zone_start_fraction'],span_frac=t['zone_span_fraction'],clip_to_zone=True)
        assert [(l['top'],l['bottom'],l['class_id']) for l in sampled]==[(l['top'],l['bottom'],l['class_id']) for l in t['layers']]
        rng=sampler.rng;x0,_,x1,_=c.docs[did]['boxes']['interpreted_lithology_column'];origin=t['real_depth_span'][0]
        phases=[]
        for l in t['layers']:
            if l['class_id']==compose.GAP:
                with Image.open(noise.NOISE/c.realism['gap_sources'][did]['file']) as im:rng.integers(im.height)
                continue
            a,b=l['top'],l['bottom'];before=rng.bit_generator.state
            rendered=render.layer(data,x0,x1,origin+a,b-a,rng)
            rr=np.random.default_rng();rr.bit_generator.state=before
            phases.append((l,int(rr.integers(data['tile'].shape[0])),int(rr.integers(data['tile'].shape[1]))))
        repeat_pairs=[(a,b) for a,b in zip(phases,phases[1:]) if a[0]['bottom']==b[0]['top']]
        print('MC765_PHASE_RESETS',len(repeat_pairs),'different_phases',sum(a[1:]!=b[1:] for a,b in repeat_pairs),'period',data['tile'].shape[:2])
        did='d368760651ded938';doc=c.docs[did];x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column'];f=OUT/'page-texas-rock-types-box-d368760651ded938-01';top=40015;bottom=40095
        real=c.windows.get(doc,[x0-40,top,x1+40,bottom]);synth=np.empty_like(real)
        for b in read(f/'page-bands.json')['bands']:
            a,z=max(top,b['top']),min(bottom,b['bottom'])
            if a>=z:continue
            with Image.open(f/b['file']) as im:synth[a-top:z-top]=np.asarray(im.convert('RGB'))[a-b['top']:z-b['top'],x0-40:x1+40]
        canvas=Image.new('RGB',(real.shape[1]*2+20,len(real)+32),'white');d=ImageDraw.Draw(canvas);d.text((0,0),'Real y40015-40095',fill='black');d.text((real.shape[1]+20,0),'Synthetic same rows',fill='black');canvas.paste(Image.fromarray(real),(0,24));canvas.paste(Image.fromarray(synth),(real.shape[1]+20,24));canvas.save(ROOT/'crossing-40055.png')
        print('CROSSING_IMAGE',str(ROOT/'crossing-40055.png'))
    finally:c.close()

def sources():
    import hashlib
    c=compose.Composer()
    try:
        bad=[];n=0
        for p in OUT.glob('*/truth.json'):
            t=read(p);n+=1
            bad.extend((str(p),did) for did in t['source_docs'] if not noise.training(c.docs[did]))
        print('STARTER_SOURCE_AUDIT',n,'outputs','test_sources',bad)
        print('NOISE_SOURCE_AUDIT',len(c.realism['measurements']),'measurements','test_sources',
              [m['id'] for m in c.realism['measurements'] if not noise.training(c.docs[m['source_doc']])])
        for p in [Path('genlab/assemble/compose.py'),Path('genlab/assemble/noise.py'),Path('data/genlab/assemble/stats.json')]:
            print('SHA256',str(p),hashlib.sha256(p.read_bytes()).hexdigest())
    finally:c.close()

if __name__=='__main__':
    globals()[sys.argv[1]]()
