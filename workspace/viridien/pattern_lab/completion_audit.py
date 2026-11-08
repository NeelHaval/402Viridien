"""Read-only measurements of a frozen annotation snapshot and its image objects."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from feedback import canonical, descriptor, distance, fill_difference

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/completion-audit-20261108'


def main():
    snap=json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    records=manifest['records'];native_root=Path(snap['normalization_root'])
    docs={d['id']:d for d in snap['documents']}
    new=[r for r in records if r['annotation_id'].startswith('queue-')]
    by_hash=collections.defaultdict(list);by_set=collections.defaultdict(list)
    classes=collections.defaultdict(list);invalid=[];visual=[];legend_images={}
    for row in records:
        by_hash[row['pixel_sha256']].append(row)
        for target in row['targets']:classes[target['key']].append(row)
        if row in new:by_set[row['pattern_set']].append(row)
        try:
            with Image.open(native_root/row['native_png']) as im:
                im=im.convert('RGB');pixels=im.tobytes()
                digest=hashlib.sha256(f'{im.width}x{im.height}:'.encode()+pixels).hexdigest()
                if digest!=row['pixel_sha256']:invalid.append([row['sample_id'],'native hash'])
                if im.size!=tuple(row['normalization']['source_size']):invalid.append([row['sample_id'],'native size'])
                if row in new and len(row['targets'])==1:
                    doc=docs[row['document_id']];target=row['targets'][0]
                    entry=next(e for e in doc['legend'] if e['id']==target['legend_id'])
                    key=hashlib.sha256((doc['id']+':'+','.join(map(str,entry['box']))).encode()).hexdigest()
                    path=ROOT/'data/review-crops'/(key+'.png')
                    if path.exists():
                        if key not in legend_images:
                            with Image.open(path) as legend:legend_images[key]=descriptor(legend.convert('RGB'))
                        feature=descriptor(im)
                        visual.append(dict(sample_id=row['sample_id'],annotation_id=row['annotation_id'],pattern_set=row['pattern_set'],document_id=doc['id'],label=target['printed'],box=row['box_xyxy'],distance=distance(feature,legend_images[key]),fill_difference=fill_difference(feature,legend_images[key]),legend_png=str(path.resolve())))
            with Image.open(native_root/row['normalized_png']) as im:
                if im.size!=(224,224) or im.mode!='RGB':invalid.append([row['sample_id'],'normalized size/mode'])
            with Image.open(native_root/row['valid_mask_png']) as im:
                if im.size!=(224,224) or im.mode!='L' or not set(np.unique(np.asarray(im))).issubset({0,255}):invalid.append([row['sample_id'],'mask'])
        except Exception as error:invalid.append([row['sample_id'],str(error)])
    duplicates=[v for v in by_hash.values() if len(v)>1]
    duplicate_conflicts=[v for v in duplicates if len({tuple(sorted(t['key'] for t in r['targets'])) for r in v})>1]
    safe_train={r['pixel_sha256'] for r in records if r['usable_for_generation']}
    safe_test={r['pixel_sha256'] for r in records if r['usable_for_test']}
    train_groups={r['source_group'] for r in records if r['usable_for_generation']}
    test_groups={r['source_group'] for r in records if r['usable_for_test']}
    set_diversity=[]
    for key,rows in sorted(by_set.items(),key=lambda kv:int(kv[0].split('-')[1])):
        set_diversity.append(dict(set=int(key.split('-')[1])+1,examples=len(rows),documents=len({r['document_id'] for r in rows}),source_groups=len({r['source_group'] for r in rows}),pixel_hashes=len({r['pixel_sha256'] for r in rows}),labels=sorted({t['key'] for r in rows for t in r['targets']})))
    class_rows=[]
    for key,rows in sorted(classes.items()):
        clean=[r for r in rows if r['quality']=='accepted']
        class_rows.append(dict(key=key,new=sum(r in new for r in rows),total=len(rows),train=sum(r['usable_for_generation'] for r in rows),test=sum(r['usable_for_test'] for r in rows),documents=len({r['document_id'] for r in clean}),source_groups=len({r['source_group'] for r in clean}),unique_pixels=len({r['pixel_sha256'] for r in clean})))
    overlap=[]
    for i,one in enumerate(new):
        for two in new[i+1:]:
            if one['document_id']!=two['document_id']:continue
            a=one['box_xyxy'];b=two['box_xyxy'];area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            ratio=area/max(1,min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])))
            if ratio>=.1:overlap.append(dict(one=one['annotation_id'],two=two['annotation_id'],fraction=round(ratio,3),label_conflict={t['key'] for t in one['targets']}!={t['key'] for t in two['targets']}))
    def compact(rows):return [dict(sample_id=r['sample_id'],annotation_id=r['annotation_id'],document=r['document_name'],labels=[t['key'] for t in r['targets']],split=r['split'],usable_for_generation=r['usable_for_generation']) for r in rows]
    metrics=dict(snapshot=manifest['snapshot'],new_examples=len(new),new_reviewers=dict(collections.Counter(r['reviewer'] for r in new)),new_documents=len({r['document_id'] for r in new}),completed_sets=sum(g['accepted']>=5 for g in snap['sets']),remaining_sets=[g for g in snap['sets'] if g['accepted']<5],new_usable_train=sum(r['usable_for_generation'] for r in new),new_usable_test=sum(r['usable_for_test'] for r in new),stats=manifest['stats'],asset_errors=invalid,quality_flags=[dict(annotation_id=r['annotation_id'],issues=r['quality_issues']) for r in records if r['quality_issues']],unique_new_pixels=len({r['pixel_sha256'] for r in new}),unique_all_pixels=len(by_hash),duplicate_hash_groups=len(duplicates),duplicate_rows=sum(len(v) for v in duplicates),test_rows=sum(r['usable_for_test'] for r in records),unique_test_pixels=len(safe_test),safe_train_test_hash_intersection=sorted(safe_train&safe_test),safe_train_test_group_intersection=sorted(train_groups&test_groups),evaluation_status=dict(collections.Counter(r['evaluation_status'] for r in records if r['usable_for_test'])),new_overlapping_pairs=overlap,duplicate_label_conflicts=[compact(v) for v in duplicate_conflicts],set_diversity=set_diversity,classes=class_rows,visual_distance_warning='Descriptor distance ranks inspection candidates. It is not a correctness score.',visual_ranked=sorted(visual,key=lambda r:r['distance'],reverse=True))
    (OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=1),encoding='utf-8')
    # One representative per set, plus all holds and high-distance inspection candidates.
    selected={rows[0]['sample_id']:rows[0] for rows in by_set.values()}
    by_id={r['sample_id']:r for r in new}
    for row in new:
        if row['quality_issues']:selected[row['sample_id']]=row
    for v in metrics['visual_ranked'][:16]:selected[v['sample_id']]=by_id[v['sample_id']]
    selected=list(selected.values());vmap={v['sample_id']:v for v in visual}
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    except OSError:font=ImageFont.load_default()
    sheet_index=[]
    for start in range(0,len(selected),8):
        sheet=Image.new('RGB',(1000,1000),'white');draw=ImageDraw.Draw(sheet)
        page=selected[start:start+8]
        for j,row in enumerate(page):
            x=(j%2)*500;y=(j//2)*250
            draw.rectangle((x+3,y+3,x+496,y+246),outline='#b7bec8')
            title=f"Set {int(row['pattern_set'].split('-')[1])+1}: "+' / '.join(t['printed'] for t in row['targets'])
            draw.text((x+12,y+10),title[:64],font=font,fill='black')
            draw.text((x+12,y+30),row['annotation_id'],font=font,fill='black')
            draw.text((x+12,y+50),row['document_name'][:61],font=font,fill='#35445b')
            with Image.open(native_root/row['native_png']) as im:
                im=im.convert('RGB');im.thumbnail((230,148),Image.Resampling.NEAREST);sheet.paste(im,(x+12,y+77))
            info=vmap.get(row['sample_id'])
            if info:
                with Image.open(info['legend_png']) as im:
                    im=im.convert('RGB');factor=min(210/im.width,100/im.height);im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.NEAREST);sheet.paste(im,(x+270,y+85))
                draw.text((x+270,y+194),'Selected source legend',font=font,fill='#35445b')
            draw.text((x+12,y+228),'HOLD: '+','.join(row['quality_issues']) if row['quality_issues'] else 'Accepted; visual correctness not yet verified',font=font,fill='#9c3917' if row['quality_issues'] else '#35445b')
        name=f'contact-{start//8+1:02}.png';sheet.save(OUT/name)
        sheet_index.append(dict(file=name,samples=[r['annotation_id'] for r in page]))
    (OUT/'contact-index.json').write_text(json.dumps(sheet_index,indent=1),encoding='utf-8')
    print(json.dumps({k:metrics[k] for k in ['new_examples','completed_sets','new_documents','new_usable_train','new_usable_test','unique_new_pixels','unique_all_pixels','duplicate_hash_groups','test_rows','unique_test_pixels','evaluation_status']}))
    print('Asset errors:',len(invalid),'overlap pairs:',len(overlap),'conflicting exact hashes:',len(duplicate_conflicts),'contact sheets:',len(sheet_index))


if __name__=='__main__':main()
