"""Check final pages in bounded bands and save native-scale visual evidence."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import numpy as np
from PIL import Image,ImageDraw
from genlab.assemble import compose,noise

root=compose.OUT/'starter2';manifest=noise.read(root/'manifest.json')
snapshot=root/'generation-stats.json'
assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==manifest['stats_sha256']
c=compose.Composer(stats_path=snapshot,state_path=manifest['state_path'])
totals=Counter();pages=[]
try:
    for entry in manifest['outputs']:
        folder=Path(entry['path']);t=noise.read(folder/'truth.json')
        assert t['stats_sha256']==manifest['stats_sha256']
        assert t['label_snapshot']==manifest['label_snapshot']
        assert t['coverage']['supported_composition_mass']>=.95-1e-9
        assert all(noise.training(c.docs[d]) for d in t['source_docs'])
        assert not any(a['class_id']==b['class_id'] for a,b in zip(t['layers'],t['layers'][1:]))
        assert t['real_depth_span'][1]<=c.docs[t['base_doc']]['boxes']['interpreted_lithology_column'][3]
        with Image.open(folder/'mask.png') as im:mask=np.asarray(im)
        lookup={cid:int(i) for i,cid in t['class_index'].items()}
        for layer in t['layers']:
            assert np.all(mask[layer['top']:layer['bottom']]==lookup[layer['class_id']])
        totals['ignored_rows']+=int((mask==255).all(1).sum())
        totals['rows']+=t['height'];totals['samples']+=1
        if t['mode']!='page':continue
        doc=c.docs[t['base_doc']];x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column']
        with Image.open(folder/'column.png') as im:col=np.asarray(im.convert('RGB'))
        crossing_rows=0;lost_rows=0;outside=0;embedded=0
        for band in noise.read(folder/'page-bands.json')['bands']:
            real=c.windows.get(doc,[0,band['top'],doc['width'],band['bottom']])
            with Image.open(folder/band['file']) as im:syn=np.asarray(im.convert('RGB'))
            outside+=int(np.any(real[:,:x0]!=syn[:,:x0],2).sum()+np.any(real[:,x1:]!=syn[:,x1:],2).sum())
            a,b=max(y0,band['top']),min(y1,band['bottom'])
            if a>=b:continue
            r=real[a-band['top']:b-band['top'],x0-6:x1+6]
            take=c.crossing_mask(r);rows=take.any(1)
            target=syn[a-band['top']:b-band['top'],x0:x1]
            lost_rows+=int((rows&((target.max(2)<100)[:,3:-3].mean(1)<.85)).sum())
            crossing_rows+=int(rows.sum())
            embedded+=int(np.any(target!=col[a-y0:b-y0],2).sum())
        assert outside==0 and embedded==0 and lost_rows==0
        pages.append(dict(id=t['synth_id'],crossing_rows=crossing_rows,lost=lost_rows,
            outside_different_pixels=outside,embedded_different_pixels=embedded,
            supervised_rows=int((mask!=255).any(1).sum()),context_consistent=t['context_consistent']))
        # A native 512-row page context window, no whole TIFF decode.
        top=y0+min(30466,max(0,t['height']-512))
        selected=next(b for b in noise.read(folder/'page-bands.json')['bands'] if b['top']<=top<b['bottom'])
        with Image.open(folder/selected['file']) as im:
            patch=im.crop((max(0,x0-160),0,min(doc['width'],x1+160),im.height))
        patch.save(root/('view-'+t['synth_id']+'.png'))
    first=next(Path(e['path']) for e in manifest['outputs'] if e['mode']=='column')
    manifest['scale_variants']=c.scale_variants(first,noise.read(first/'truth.json'))
    for entry in manifest['outputs']:
        if entry['mode']!='column':continue
        folder=Path(entry['path']);t=noise.read(folder/'truth.json')
        t['supervised_rows']=sum(l['bottom']-l['top'] for l in t['layers'] if not l.get('ignore'))
        noise.write(folder/'truth.json',t)
    # Directly compare the same training Hadrian crossing criterion on old pages.
    old_losses=[]
    for path in (compose.OUT/'starter').glob('page-*/page-bands.json'):
        t=noise.read(path.parent/'truth.json')
        if t['base_doc']!='5709fe9e86ce942b':continue
        doc=c.docs[t['base_doc']];x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column'];found=lost=0
        for band in noise.read(path)['bands']:
            a,b=max(y0,band['top']),min(y1,band['bottom'])
            if a>=b:continue
            real=c.windows.get(doc,[x0-6,a,x1+6,b]);rows=c.crossing_mask(real).any(1)
            with Image.open(path.parent/band['file']) as im:syn=np.asarray(im.convert('RGB'))[a-band['top']:b-band['top'],x0:x1]
            found+=int(rows.sum());lost+=int((rows&((syn.max(2)<100)[:,3:-3].mean(1)<.85)).sum())
        old_losses.append(dict(id=t['synth_id'],crossing_rows=found,lost=lost))
    # Shared-channel numerical report; all pixel values below are training only.
    manifest['verification']=dict(totals=totals,pages=pages,before_crossings=old_losses)
    noise.write(root/'manifest.json',manifest)
    noise.write(Path(__file__).with_name('page-checks.json'),manifest['verification'])
    print(json.dumps(manifest['verification'],indent=2))
    print('PASS starter2 masks, provenance, gates, source bounds, pages, crossings and native embedded columns')
finally:c.close()
