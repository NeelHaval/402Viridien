"""Reproducible before/after visual audit; artifacts live under genlab's output."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import candidates, learn, paths, render, score

BASE = paths.OUT / 'scratch' / 'stage3-baseline'
PRIMARY = [('952caeadf1d8467c', 5), ('4a056c8ff25cf749', 0),
           ('59c21186a4a3c9e9', 0), ('7b0d6b0a3e062ece', 0),
           ('fb5281e7ab92595f', 0), ('3515004b6f643c87', 2), ('3515004b6f643c87', 3)]
SEAMS = [('4a056c8ff25cf749', 12), ('3515004b6f643c87', 6), ('1f585adbef18c764', 10)]


def capture():
    BASE.mkdir(parents=True, exist_ok=True)
    for report in candidates.reports():
        target = BASE / (report['doc_id'] + '.json')
        if target.exists():
            continue
        target.write_text(json.dumps(report, indent=1), encoding='utf-8')
        for e in report['entries']:
            if e.get('data_file'):
                shutil.copy2(paths.CANDIDATES / e['data_file'], BASE / e['data_file'])
    if not (BASE / 'state.json').exists():
        shutil.copy2(paths.STATE, BASE / 'state.json')
    snapshot_verdicts()
    print('PASS baseline preserved', len(list(BASE.glob('*.npz'))), 'models', flush=True)


def snapshot_verdicts():
    dest = BASE / 'verdicts'
    dest.mkdir(exist_ok=True)
    for p in paths.VERDICTS.glob('*.json'):
        shutil.copy2(p, dest / p.name)


def font(size=14):
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def before():
    import model
    crops = learn.Crops(model.Images())
    store = model.Store()
    oldstate = json.loads((BASE / 'state.json').read_text(encoding='utf-8'))
    cards = {(c['doc_id'], c['legend_id']): c for c in oldstate['cards']}
    targets = set(PRIMARY)
    for p in paths.VERDICTS.glob('*.json'):
        report = json.loads((BASE / p.name).read_text(encoding='utf-8'))
        verdicts = json.loads(p.read_text(encoding='utf-8'))
        targets.update((p.stem, e['legend_id']) for e in report['entries']
                       if verdicts.get(str(e['legend_id']), {}).get('verdict') in ('reject', 'unsure'))
    for did, lid in sorted(targets):
        report = json.loads((BASE / (did + '.json')).read_text(encoding='utf-8'))
        e = next(x for x in report['entries'] if x['legend_id'] == lid)
        doc = store.get(did)
        dest = BASE / f'{did}-{lid}'
        dest.mkdir(exist_ok=True)
        if (dest / 'old.png').exists():
            continue
        if not e.get('data_file'):
            continue
        data = candidates.load_data(BASE / e['data_file'])
        sw = crops.get(doc, e['swatch_box'])
        Image.fromarray(sw).save(dest / 'swatch.png')
        ev = e.get('human_evidence') or e.get('evidence') or []
        realdoc = doc
        if e['tier'] == 'borrowed':
            realdoc = store.get(e['donor']['doc_id'])
            ev = e['donor']['evidence']
        for i, v in enumerate(ev[:5]):
            Image.fromarray(crops.get(realdoc, v['box'])[:240]).save(dest / f'real-{i}.png')
        card = cards.get((did, lid))
        if card and (paths.OUT / card['samples'][0]).exists():
            shutil.copy2(paths.OUT / card['samples'][0], dest / 'old.png')
        else:
            col = doc['boxes']['interpreted_lithology_column']
            rails = dict(x0=col[0], x1=col[2], mask=[False] * (col[2]-col[0]), ink=[0, 0, 0])
            Image.fromarray(render.sample(data, rails, 240, np.random.default_rng(0))).save(dest / 'old.png')
    print('PASS before images', len(list(BASE.glob('*/old.png'))), flush=True)


def before_sheets():
    selected = []
    for did, lid in PRIMARY:
        selected.append((did, lid, 'Named entry'))
    for p in sorted(paths.VERDICTS.glob('*.json')):
        report = json.loads((BASE / p.name).read_text(encoding='utf-8'))
        verdicts = json.loads(p.read_text(encoding='utf-8'))
        for e in report['entries']:
            v = verdicts.get(str(e['legend_id']), {})
            if v.get('verdict') in ('reject', 'unsure') and e['tier'] in ('human', 'verified', 'borrowed'):
                if (p.stem, e['legend_id']) not in PRIMARY:
                    selected.append((p.stem, e['legend_id'], v['note']))
    # One scale (2x native) for all pixels, including swatches and narrow crops.
    for start in range(0, len(selected), 8):
        page = Image.new('RGB', (1500, 8 * 390), 'white')
        draw = ImageDraw.Draw(page)
        for row, (did, lid, note) in enumerate(selected[start:start+8]):
            dest = BASE / f'{did}-{lid}'
            report = json.loads((BASE / (did + '.json')).read_text(encoding='utf-8'))
            e = next(x for x in report['entries'] if x['legend_id'] == lid)
            y = row * 390
            draw.text((8, y+3), f'{did}:{lid} {e["label"]} ({e["tier"]}) | {note}', font=font(), fill='black')
            x = 8
            for label, path in [('Swatch', dest/'swatch.png'),
                                *[(f'Real {i}', dest/f'real-{i}.png') for i in range(3)],
                                ('Old generated', dest/'old.png')]:
                if not path.exists():
                    continue
                im = Image.open(path).convert('RGB')
                im = im.crop((0, 0, min(160, im.width), min(164, im.height)))
                draw.text((x, y+25), label, font=font(), fill='black')
                im = im.resize((im.width*2, im.height*2), Image.Resampling.NEAREST)
                page.paste(im, (x, y+48))
                x += max(100, im.width) + 16
            draw.line((0,y+385,1500,y+385), fill='#cccccc')
        page.save(paths.OUT/'scratch'/f'stage3-before-{start//8}.png')
    print('PASS before contact sheets', len(selected), 'entries', flush=True)


def audit():
    """Render rejected entries diagnostically, without changing their verdicts."""
    import model
    store = model.Store()
    docs, groups = learn.human_examples(store)
    crops = learn.Crops(model.Images())
    targets = list(PRIMARY)
    verdicts = {}
    for p in paths.VERDICTS.glob('*.json'):
        vv = json.loads(p.read_text(encoding='utf-8'))
        old = json.loads((BASE / p.name).read_text(encoding='utf-8'))
        for e in old['entries']:
            v = vv.get(str(e['legend_id']), {})
            key = (p.stem, e['legend_id'])
            verdicts[key] = v
            if v.get('verdict') in ('reject', 'unsure') and e['tier'] in ('human', 'verified', 'borrowed') and key not in targets:
                targets.append(key)
    targets += [key for key in SEAMS if key not in targets]
    results = []
    for did, lid in targets:
        old = json.loads((BASE / (did+'.json')).read_text(encoding='utf-8'))
        new = json.loads((paths.CANDIDATES / (did+'.json')).read_text(encoding='utf-8'))
        oe, ne = [next(e for e in r['entries'] if e['legend_id']==lid) for r in (old, new)]
        doc = docs[did]['doc']
        dest = BASE / f'{did}-{lid}'
        if not (dest/'old.png').exists():
            continue
        olddata = candidates.load_data(BASE/oe['data_file'])
        newdata = candidates.load_data(paths.CANDIDATES/ne['data_file']) if ne.get('data_file') else None
        sw, swdata, repeat = candidates.swatch(crops.get(doc, ne['swatch_box']))
        Image.fromarray(sw).save(dest/'new-swatch.png')
        # Withheld sources get a diagnostic of their own swatch; it is never
        # silently promoted into the generated dataset.
        withdrawn = oe['tier']=='borrowed' and ne['tier'] not in ('human','verified','borrowed')
        diagnostic = newdata or dict(kind='tile',tile=sw,bg=learn.background(sw),anchored=False,phase=(0,0))
        if withdrawn:
            diagnostic = swdata
        exs = groups.get((did,lid), []) if oe['tier']=='human' else candidates.examples(doc,oe)
        width = Image.open(dest/'old.png').width
        rails = dict(x0=0,x1=width,mask=[False]*width,ink=[0,0,0])
        humans = [e for (d,_),ee in groups.items() if d==did for e in ee]
        if humans:
            actual_rails = learn.rails(crops,docs,did,humans)
            if actual_rails['x1']-actual_rails['x0']==width:
                rails = actual_rails
        elif exs:
            col = docs[did]['column']
            ev = [dict(e,box=[col[0],e['box'][1],col[2],e['box'][3]]) for e in exs]
            actual_rails = learn.rails(crops,docs,did,ev)
            if actual_rails['x1']-actual_rails['x0']==width:
                rails = actual_rails
        Image.fromarray(render.sample(diagnostic,rails,240,np.random.default_rng(0))).save(dest/'new.png')
        if diagnostic['kind']=='tile':
            Image.fromarray(render.repeat_preview(diagnostic,width)).save(dest/'preview.png')
        newreal_doc = doc
        newreal = ne.get('human_evidence') or ne.get('evidence') or []
        if ne['tier']=='borrowed':
            newreal_doc = docs[ne['donor']['doc_id']]['doc']
            newreal = ne['donor']['evidence']
        for i,ev in enumerate(newreal[:5]):
            Image.fromarray(crops.get(newreal_doc,ev['box'])[:240]).save(dest/f'new-real-{i}.png')
        baseline_score = selection_score = held_old = held_new = None
        if exs and oe['tier'] in ('human','verified'):
            from .selection import crop_scores
            baseline_score = crop_scores(olddata,crops,doc,exs)
            selection_score = crop_scores(newdata,crops,doc,exs) if newdata else None
            held_old = score.realism(oe['model'],olddata,crops,doc,exs)
            # Compare the identical evaluation subset, even if new voters differ.
            held_new = score.realism(oe['model'],newdata,crops,doc,exs) if newdata else None
        original_sw, _, _ = candidates.swatch(crops.get(doc,oe['swatch_box']))
        swold = candidates.pixel_match(original_sw,olddata) if olddata['kind']=='tile' else None
        swnew = candidates.pixel_match(original_sw,newdata) if newdata and newdata['kind']=='tile' and not withdrawn else None
        inkold = candidates.symbol_match(original_sw,olddata) if olddata['kind']=='tile' else None
        inknew = candidates.symbol_match(original_sw,newdata) if newdata and newdata['kind']=='tile' and not withdrawn else None
        results.append(dict(doc_id=did,legend_id=lid,label=oe['label'],old_tier=oe['tier'],new_tier=ne['tier'],
                            verdict=verdicts.get((did,lid)),old_all=baseline_score,new_all=selection_score,
                            old_held=held_old,new_held=held_new,old_swatch_match=swold,new_swatch_match=swnew,
                            old_symbol_match=inkold,new_symbol_match=inknew,withdrawn=withdrawn,
                            withheld=(withdrawn or verdicts.get((did,lid),{}).get('verdict')=='reject' or
                                      ne['tier'] not in ('human','verified','borrowed')),
                            tile_source=ne.get('model',{}).get('tile_source'),tile_scores=ne.get('model',{}).get('tile_scores'),
                            seam_risk=repeat['seam_risk'],directory=dest.relative_to(paths.OUT).as_posix()))
        print(did,lid,oe['label'],oe['tier'],'->',ne['tier'],
              'all', (baseline_score or {}).get('median'), '->', (selection_score or {}).get('median'),
              'swatch',swold,'->',swnew,'withdrawn',withdrawn,flush=True)
    (paths.OUT/'scratch/stage3-audit.json').write_text(json.dumps(results,indent=1),encoding='utf-8')
    for start in range(0,len(results),7):
        contact(results[start:start+7],paths.OUT/'scratch'/f'stage3-after-{start//7}.png')
    contact(results[:len(PRIMARY)],paths.OUT/'scratch/stage3-before-after.png',primary=True)
    print('PASS visual audit sheets',len(results),'entries',flush=True)


def contact(rows,path,primary=False):
    rowheight = 550 if primary else 390
    page = Image.new('RGB',(1500,rowheight*len(rows)),'white')
    draw = ImageDraw.Draw(page)
    for index,r in enumerate(rows):
        dest = paths.OUT/r['directory']
        y = index*rowheight
        status = 'WITHHELD; new image is diagnostic' if r.get('withheld',r['withdrawn']) else f'tile: {r["tile_source"]}'
        draw.text((8,y+2),f'{r["doc_id"]}:{r["legend_id"]} {r["label"]} | {status}',font=font(),fill='black')
        x = 8
        names = [('Real crop',dest/'real-0.png'),('Old generated',dest/'old.png'),
                 ('New generated',dest/'new.png'),('New repeat preview',dest/'preview.png')]
        if not primary:
            names = [('Swatch',dest/'swatch.png')]+names
            names.append(('New real crop',dest/'new-real-0.png'))
        for label,p in names:
            if not p.exists():
                continue
            im = Image.open(p).convert('RGB')
            im = im.crop((0,0,min(im.width,180),min(im.height,240 if primary else 164)))
            im = im.resize((im.width*2,im.height*2),Image.Resampling.NEAREST)
            draw.text((x,y+25),label,font=font(),fill='black')
            page.paste(im,(x,y+48))
            x += max(im.width,150)+18
        draw.line((0,y+rowheight-5,1500,y+rowheight-5),fill='#cccccc')
    page.save(path)


def primary():
    import model
    from .selection import choose
    docs, groups = learn.human_examples(model.Store())
    crops = learn.Crops(model.Images())
    rows = []
    for did,lid in PRIMARY:
        doc = docs[did]['doc']
        e = next(e for e in json.loads((BASE/(did+'.json')).read_text(encoding='utf-8'))['entries'] if e['legend_id']==lid)
        exs = groups.get((did,lid)) or candidates.examples(doc,e)
        m,d = learn.learn(crops,docs,(did,lid),exs)
        _,s,r = candidates.swatch(crops.get(doc,e['swatch_box']))
        m,d = choose(crops,doc,exs,m,d,s,r['seam_risk'])
        dest = BASE/f'{did}-{lid}'
        width = Image.open(dest/'old.png').width
        all_exs = [v for (did2,_),ee in groups.items() if did2==did for v in ee]
        rails = learn.rails(crops,docs,did,all_exs) if all_exs else dict(x0=0,x1=width,mask=[False]*width,ink=[0,0,0])
        Image.fromarray(render.sample(d,rails,240,np.random.default_rng(0))).save(dest/'new.png')
        if d['kind']=='tile':
            Image.fromarray(render.repeat_preview(d,width)).save(dest/'preview.png')
        rows.append(dict(doc_id=did,legend_id=lid,label=e['label'],tile_source=m['tile_source'],
                         withdrawn=False,directory=dest.relative_to(paths.OUT).as_posix()))
        print(did,lid,m['tile_source'],m.get('tile_size'),m['tile_scores'],flush=True)
    contact(rows,paths.OUT/'scratch/stage3-before-after.png',primary=True)


def finish():
    from .stage3_findings import FINDINGS
    results = json.loads((paths.OUT/'scratch/stage3-audit.json').read_text(encoding='utf-8'))
    def metric(value):
        if not value:
            return 'N/A'
        tag = 'held-out' if value.get('held_out') else 'source only'
        return f'{value["median"]*100:.1f}% ({tag}, n={value.get("crops",value.get("held_out"))})'
    lines = ['# Stage 3 visual audit', '',
             'Opened the before and after PNG contact sheets for every entry below. Rejected entries remain withheld. ',
             'Held-out scores use the same original evaluation subset and exact best cyclic RGB match for both models. ',
             'A one-box entry has no held-out evaluation. Borrowed entries have no target-column held-out evaluation. ',
             'Tile selection uses all crops; these voting hold-outs are not a separate untouched final validation set. ',
             'Both versions were rescored with the exact matcher. Old UI chips used approximate phase alignment and mislabeled training fallback crops as held-out. ',
             'All-crop selection scores and target-swatch comparisons are recorded separately in stage3-audit.json. ',
             'High pixel matches can hide a bad wrap; the visual findings take precedence over such scores.', '']
    rereview_keys = {'952caeadf1d8467c:5','3515004b6f643c87:2','16c9064b18b77b61:1',
                     '1bac8dda58a39eaf:0','663e1eb0b13336ca:2','97adb9d2c69d860b:14',
                     'aeb414a4367bafd4:3','4a056c8ff25cf749:12','3515004b6f643c87:6'}
    rereview = []
    for r in results:
        key = f'{r["doc_id"]}:{r["legend_id"]}'
        if r['new_tier']=='swatch':
            r['tile_source'] = 'legend swatch'
        elif r['new_tier']=='borrowed':
            report = json.loads((paths.CANDIDATES/(r['doc_id']+'.json')).read_text(encoding='utf-8'))
            entry = next(e for e in report['entries'] if e['legend_id']==r['legend_id'])
            donor = entry['donor']
            report = json.loads((paths.CANDIDATES/(donor['doc_id']+'.json')).read_text(encoding='utf-8'))
            entry = next(e for e in report['entries'] if e['legend_id']==donor['legend_id'])
            r['tile_source'] = entry['model'].get('tile_source','column')
        finding = FINDINGS[key]
        r['finding'] = finding
        lines += [f'## {key} — {r["label"]}', '', f'Reviewer note: {(r["verdict"] or {}).get("note","Named owner defect; no reviewer note.")}',
                  '', finding, '', f'Old → new: {metric(r["old_held"])} → {metric(r["new_held"])}.', '']
        if r['old_all'] and r['new_all']:
            lines += [f'All original source crops: {r["old_all"]["median"]*100:.1f}% → {r["new_all"]["median"]*100:.1f}%.', '']
        if r['old_tier']=='borrowed' or (r['doc_id'],r['legend_id']) in SEAMS:
            a,b = r['old_swatch_match'],r['new_swatch_match']
            lines += [f'Target-swatch comparison (not held-out): {a*100:.1f}% → '+(f'{b*100:.1f}%.' if b is not None else 'withheld.'), '']
        if key in rereview_keys and (r['verdict'] or {}).get('verdict')=='reject':
            rereview.append(dict(doc_id=r['doc_id'],legend_id=r['legend_id'],reason=finding))
    (paths.OUT/'scratch/stage3-audit.json').write_text(json.dumps(results,indent=1),encoding='utf-8')
    (paths.OUT/'scratch/stage3-audit.md').write_text('\n'.join(lines),encoding='utf-8')
    path = paths.OUT/'jobs/rereview.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(rereview,indent=1),encoding='utf-8')
    for start in range(0,len(results),7):
        contact(results[start:start+7],paths.OUT/'scratch'/f'stage3-after-{start//7}.png')
    contact(results[:len(PRIMARY)],paths.OUT/'scratch/stage3-before-after.png',primary=True)
    print('PASS audit report',len(results),'entries;',len(rereview),'rereview requests',flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['capture', 'before', 'before-sheets', 'audit', 'primary', 'finish'])
    action = ap.parse_args().action
    if action == 'capture':
        capture()
    elif action == 'before':
        snapshot_verdicts()
        before()
    elif action == 'before-sheets':
        before_sheets()
    elif action == 'audit':
        audit()
    elif action == 'primary':
        primary()
    else:
        finish()


if __name__ == '__main__':
    main()
