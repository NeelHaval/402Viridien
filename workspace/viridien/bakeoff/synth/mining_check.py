"""Reproducible colour audit and native-aspect visual mining check."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import random
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
from synth.common import DATA, HOME, read_json, write_json, local_file, font
from synth.colour import profile, colour_gate


def top_colours(rgb, non_ink=False):
    pixels=np.asarray(rgb).reshape(-1,3).astype(int)
    bins=pixels//32
    codes=bins@np.array([64,8,1])
    counts=np.bincount(codes,minlength=512)
    colours=[]
    for i in np.argsort(-counts):
        if not counts[i]: break
        colour=pixels[codes==i].mean(axis=0)
        if non_ink and colour.max()<60: continue
        colours.append(colour)
        if len(colours)==2: break
    return colours


def audit(baseline=False):
    patterns=read_json(DATA/'catalog.json')['patterns']
    candidates=read_json(DATA/'candidates.json')
    result={}
    for family in sorted({p['family'] for p in patterns}):
        count=colour_pass=gate_pass=0
        for p in patterns:
            if p['family'] != family: continue
            e=p['swatches'][0]
            with Image.open(local_file(e.get('native_file',e['file']))) as im:
                native=np.asarray(im.convert('RGB'))
            ref=profile(native)
            main=(top_colours(native,non_ink=True) or top_colours(native))[0]
            for c in candidates.get(p['pid'],[]):
                with Image.open(local_file(c['file'])) as im: crop=np.asarray(im.convert('RGB'))
                count+=1
                colour_pass+=any(np.max(abs(main-colour))<=48 for colour in top_colours(crop))
                gate_pass+=colour_gate(ref,profile(crop))[0]
        result[family]=dict(candidates=count,colour_pass_rate=colour_pass/max(1,count),
                            lab_gate_pass_rate=gate_pass/max(1,count))
    out=HOME/'out'
    write_json(out/('mining_before.json' if baseline else 'mining_after.json'),result)
    print(result,flush=True)
    if baseline: return
    from synth.mine import calibrate, source_intervals
    report=read_json(DATA/'mining_report.json',{})
    report['calibration']=calibrate(patterns,source_intervals())
    report['output_colour_audit']=result
    report['baseline_colour_audit']=read_json(out/'mining_before.json',{})
    for counts in report.get('pattern_counts',{}).values():
        if counts['candidates']==30 and counts['target']>30:
            counts['shortfall']=['review cap (30)']
    write_json(DATA/'mining_report.json',report)
    rng=random.Random(42)
    rows=[]
    for family in sorted(result):
        if family.startswith('D-582'): continue
        available=[p for p in patterns if p['family']==family and candidates.get(p['pid'])]
        available=available or [p for p in patterns if p['family']==family]
        for p in rng.sample(available,min(2,len(available))):
            e=p['swatches'][0]
            images=[local_file(e.get('native_file',e['file']))]+[local_file(c['file']) for c in candidates.get(p['pid'],[])[:8]]
            thumbs=[]
            for path in images:
                with Image.open(path) as im:
                    thumbs.append(im.convert('RGB').resize((max(1,round(im.width*128/im.height)),128),Image.Resampling.NEAREST))
            rows.append((p,thumbs))
    render(rows,out/'mining_check.png')
    write_json(out/'mining_check_patterns.json',[p['pid'] for p,_ in rows])
    regression=[]
    for p in patterns:
        label=p['label'].lower().replace(' ','')
        wanted=(p['family'].startswith('Gulf') and label=='dolomite'
                or p['family'].startswith('MC761') and label=='limestone'
                or p['family'].startswith('MC765') and label in ('sand-mss','limestone-sl')
                or p['family'].startswith('Texas') and label=='salt')
        if not wanted: continue
        e=p['swatches'][0]
        files=[local_file(e.get('native_file',e['file']))]+[local_file(c['file']) for c in candidates[p['pid']][:8]]
        images=[]
        for path in files:
            with Image.open(path) as im:
                images.append(im.convert('RGB').resize((max(1,round(im.width*128/im.height)),128),Image.Resampling.NEAREST))
        regression.append((p,images))
    if regression: render(regression,out/'mining_regressions.png')


def render(rows,path):
    width=max(900,max((330+sum(im.width+16 for im in ims) for _,ims in rows),default=900))
    canvas=Image.new('RGB',(width,max(190,190*len(rows))),'white');d=ImageDraw.Draw(canvas)
    for row,(p,images) in enumerate(rows):
        y=row*190
        d.text((8,y+10),p['family'],font=font(16),fill='black')
        d.text((8,y+34),p['label'][:35],font=font(16),fill='black')
        d.text((8,y+58),p['legend_type'][-6:],font=font(14),fill='black')
        x=330
        for i,im in enumerate(images):
            d.text((x,y), 'swatch' if i==0 else str(i),font=font(14),fill='black')
            canvas.paste(im,(x,y+24));x+=im.width+16
    canvas.save(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',action='store_true')
    audit(**vars(parser.parse_args()))
