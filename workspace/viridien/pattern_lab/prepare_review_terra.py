"""Small blind Terra packet. Training pages and synthetic diagnostics only."""
import collections
import hashlib
import json
from pathlib import Path
import sys

from export_dataset_review import BASE,DATA,ROOT,DEST,ASSETS,read,save,sha,digest
from PIL import Image,ImageDraw,ImageFont
from model import Images
from grid_lab import cached_crop,paste_fit

PACK=ASSETS/'terra-first-pass'
PROMPT='''Label the numbered crops using only each crop's own source legend. Preserve raw legend identities. Use the native crop and its surrounding source rows; a crop can cross beds. Return JSON items with itemId, legendIds, status (match/mixed/uncertain/bad_crop), confidence (high/medium/low), reason, boundaryIssue (boolean). For mixed crops list every visible source legend ID; do not invent merged rock names. Mark insufficient symbols, cut-off beds, uncertain legend extraction or almost identical fills as uncertain. Confidence is your uncalibrated judgement, not empirical accuracy. No human labels, heldout truth or evaluation scores are supplied. These are training-source checks and synthetic diagnostics, not permission to approve a label. Read only this packet and its listed PNGs. Do not inspect other stores, truth files, review suggestions or source labels. Do not call an API or run another model. Write only result.json in this packet. Complete by 07:52 UTC; stop after at most 20 items.'''


def main():
    PACK.mkdir(parents=True,exist_ok=True)
    context=read(ASSETS/'source-context.json')['documents'];inventory=read(DEST/'labeller-inventory.json');suggestions=read(DEST/'review-suggestions.json')['records']
    old=read(DATA/'terra-grid-20261108/blind/packets.json');seen={(i['document_id'],tuple(i['source_box'])) for i in old['items']}
    support=read(DATA/'completion-audit-20261108/opus-support.json');selected=collections.defaultdict(list)
    # Flagged human data are review inputs, never answer keys.
    for e in support['confirmed_visual_label_disagreement']:
        ident=e['document_id'];b=e['box']
        if ident in context and (ident,tuple(b)) not in seen:selected[ident].append(dict(bbox=b,sourceKind='current',reason='structural-or-label-inconsistency'))
    # Gulf pages without a broad cached Terra pass. Limit to two pages/six crops.
    for ident in ['17a317978b2c9139','663e1eb0b13336ca','16c9064b18b77b61']:
        if ident not in context:continue
        p=DATA/'reviews'/(ident+'.json')
        if not p.exists():p=DATA/'seeds'/(ident+'.json')
        doc=read(p);count=0
        for segment in doc['segments']:
            b=segment['box']
            if segment.get('approved') or segment.get('status')=='rejected' or b[3]-b[1]<32 or b[3]-b[1]>180 or (ident,tuple(b)) in seen:continue
            selected[ident].append(dict(bbox=b,sourceKind='current',reason='uncached Gulf training-source crop'));count+=1
            if count>=3:break
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);images=Images();groups=[];items=[];counter=0
    # Synthetic diagnostics keep the parent page and donor lineage explicit.
    generated=read(DATA/'genlab/generated/manifest.json');synthetic=[]
    for ident in ['17a317978b2c9139','16c9064b18b77b61']:
        p=next((p for p in generated['patterns'] if p.get('doc_id')==ident and p.get('source_doc_id')==ident and not p.get('donor')),None)
        if p and ident in context:
            source=DATA/'genlab'/p['file']
            if source.exists():synthetic.append((ident,source,p))
    for ident,rows in selected.items():
        if not rows:continue
        doc=context[ident];source_doc=dict(id=ident,name=doc['name'],image_name=Path(doc['image']).name,width=doc['size'][0],height=doc['size'][1],boxes=dict(interpreted_lithology_column=doc['column']),legend=doc['legend'])
        groupdir=PACK/ident;groupdir.mkdir(exist_ok=True)
        # Enlarge only for display; authoritative evidence retains native pixels.
        legends=[e for e in doc['legend'] if not e.get('deprecated')]
        panels=[]
        for first in range(0,len(legends),16):
            es=legends[first:first+16];panel=Image.new('RGB',(1024,55+((len(es)+1)//2)*88),'white');d=ImageDraw.Draw(panel);d.text((12,12),'Source legend IDs and raw captions',font=font,fill='black')
            for j,e in enumerate(es):
                x=(j%2)*512;y=50+(j//2)*88;d.text((x+8,y+5),'L'+str(e['id']),font=font,fill='black');paste_fit(panel,cached_crop(source_doc,e['box'],images),(x+66,y,150,75));d.text((x+226,y+5),e['label_printed'][:29],font=font,fill='black')
            out=groupdir/f'legend-{first//16}.png';panel.save(out);panels.append(str(out.resolve()))
        group_items=[]
        for r in rows:
            counter+=1;item_id=f'T{counter:03}';box=r['bbox'];crop=cached_crop(source_doc,box,images);path=groupdir/(item_id+'.png');crop.save(path)
            col=doc['column'];cb=[box[0],max(col[1],box[1]-35),box[2],min(col[3],box[3]+35)];context_crop=cached_crop(source_doc,cb,images);cp=groupdir/(item_id+'-context.png');context_crop.save(cp)
            item=dict(itemId=item_id,recordId=doc['recordId'],documentId=ident,sourceImageSha256=doc['imageSha256'],bbox=box,crop=str(path.resolve()),cropSha256=sha(path),context=str(cp.resolve()),contextBox=cb,contextSha256=sha(cp),sourceKind='current',split='train');items.append(item);group_items.append(item)
        # Attach synthetic examples of this document to the same source legend.
        for sid,source,p in synthetic:
            if sid!=ident:continue
            counter+=1;item_id=f'T{counter:03}'
            with Image.open(source) as im:im=im.convert('RGB');w,h=im.size;im.crop((0,0,w,min(h,180))).save(groupdir/(item_id+'.png'))
            path=groupdir/(item_id+'.png');rid='synthetic-'+sha(source)[:20]
            item=dict(itemId=item_id,recordId=rid,parentRecordId=doc['recordId'],documentId=ident,sourceImageSha256=sha(source),bbox=[0,0,w,min(h,180)],crop=str(path.resolve()),cropSha256=sha(path),context=str(path.resolve()),contextBox=[0,0,w,min(h,180)],contextSha256=sha(path),sourceKind='synthetic',split='synthetic-diagnostic',diagnosticOnly=True);items.append(item);group_items.append(item)
            save(groupdir/(item_id+'-lineage.json'),dict(source=str(source.resolve()),parentRecordId=doc['recordId'],lineage=p.get('lineage'),sourceGroup=p.get('source_group'),intendedLabel=p['label'],templateTruth=True,humanConfirmed=False))
        if not group_items:continue
        board=Image.new('RGB',(1024,60+((len(group_items)+3)//4)*285),'white');draw=ImageDraw.Draw(board);draw.text((12,12),'Numbered crops: crop at left; nearby rows at right',font=font,fill='black')
        for j,item in enumerate(group_items):
            x=(j%4)*256;y=52+(j//4)*285;draw.text((x+10,y+5),item['itemId'],font=font,fill='black')
            with Image.open(item['crop']) as im:paste_fit(board,im,(x+8,y+35,112,235))
            with Image.open(item['context']) as im:paste_fit(board,im,(x+128,y+35,112,235))
        board_path=groupdir/'board.png';board.save(board_path);groups.append(dict(documentId=ident,legendPanels=panels,board=str(board_path.resolve()),itemIds=[i['itemId'] for i in group_items],rawLegend=[dict(id=e['id'],label=e['label_printed']) for e in legends]))
    if images.current:images.current.close()
    (PACK/'prompt.txt').write_text(PROMPT,encoding='utf-8')
    save(PACK/'packet.json',dict(model='gpt-6.1-terra',prompt=PROMPT,promptSha256=sha(PACK/'prompt.txt'),groups=groups,items=items,heldoutExcluded=True,maxItems=20,existingIdenticalTerraQueriesExcluded=True))
    print('TERRA_PACKET',len(items),'items',len(groups),'training source pages',str((PACK/'packet.json').resolve()))


if __name__=='__main__':main()
