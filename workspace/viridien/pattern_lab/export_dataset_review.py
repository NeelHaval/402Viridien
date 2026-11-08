"""Export cached model evidence and a conservative human training subset.

Read only existing page stores. Never change human labels or evaluate a model.
"""
import collections
import hashlib
import json
from pathlib import Path
import time

BASE=Path(__file__).resolve().parent
DATA=BASE/'data'
ROOT=BASE.parents[1]
DEST=ROOT/'management/dataset-review/inputs'
ASSETS=ROOT/'management/dataset-review/labeller-assets'
DATASET=ROOT/'viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def boxdict(b,**kw):return dict(x0=b[0],y0=b[1],x1=b[2],y1=b[3],bbox=list(b),**kw)


def main():
    manifest_path=DATA/'completion-audit-20261108/manifest.json';manifest=read(manifest_path)
    support=read(DATA/'completion-audit-20261108/opus-support.json');flagged={r['sample_id']:r for r in support['confirmed_visual_label_disagreement']}
    heldout_docs={r['document_id'] for r in manifest['records'] if r['split']!='train'}
    heldout_groups={r['source_group'] for r in manifest['records'] if r['split']!='train'}
    catalog=read(DATA/'documents.json');docmap={};paths={}
    for row in catalog:
        path=DATA/'reviews'/(row['id']+'.json')
        if not path.exists():path=DATA/'seeds'/(row['id']+'.json')
        docmap[row['id']]=read(path);paths[row['id']]=path
    e2e=read(DATA/'jev-e2e-20261108/manifest.json');known={Path(c['image']).name:c for c in e2e['cases']}
    train=collections.defaultdict(list);sealed=0
    for r in manifest['records']:
        if r['split']!='train':sealed+=1;continue
        if r['source_group'] in heldout_groups:continue
        train[r['document_id']].append(r)
    packets=read(DATA/'terra-grid-20261108/blind/packets.json');answers={b['batch_id']:b for b in read(DATA/'terra-grid-20261108/terra-results.json')['batches']}
    items={i['item_id']:i for i in packets['items']};suggestions=[];records=[];bydoc={};model_calls=0
    for ident,rows in train.items():
        if ident in heldout_docs or ident not in docmap:continue
        doc=docmap[ident];image=DATASET/'images'/doc['image_name'];annotation=DATASET/'bbox_annotation'/(Path(doc['image_name']).stem+'.json')
        if not image.exists() or not annotation.exists():continue
        h=known.get(image.name,{}).get('image_sha256') or sha(image)
        b=doc['boxes']['interpreted_lithology_column'];rid='page-'+h[:20]
        legend={e['id']:e for e in doc['legend']};segments={s['id']:s for s in doc['segments']};truth=[];issues=[];quarantine=[]
        for r in rows:
            seg=segments.get(r['annotation_id']);why=[]
            if r['sample_id'] in flagged:why.append(flagged[r['sample_id']]['seen'])
            if r['box_xyxy'][3]-r['box_xyxy'][1]<32:why.append('Fewer than 32 source rows; weak repeat evidence')
            if r.get('quality_issues'):why.extend(map(str,r['quality_issues']))
            if not seg or seg.get('box')!=r['box_xyxy'] or not seg.get('approved'):why.append('Current human store no longer confirms this exact saved box')
            if seg and set(seg.get('legend_ids',[]))!={t['legend_id'] for t in r['targets']}:why.append('Current source label IDs differ from frozen export')
            if any(t['legend_id'] not in legend or t['printed']!=legend[t['legend_id']]['label_printed'] for t in r['targets']):why.append('Current source caption differs from the frozen export')
            if why:
                quarantine.append(dict(sampleId=r['sample_id'],bbox=r['box_xyxy'],reasons=why));issues.append(dict(bbox=r['box_xyxy'],explanation='; '.join(why),sampleId=r['sample_id']))
            else:truth.append(boxdict(r['box_xyxy'],lithology=' / '.join(t['printed'] for t in r['targets']),legendIds=[t['legend_id'] for t in r['targets']],sampleId=r['sample_id'],humanConfirmed=True,pixelSha256=r['pixel_sha256']))
        truth_path=ASSETS/'truth'/(rid+'.json');ignore_path=ASSETS/'ignore'/(rid+'.json')
        save(truth_path,dict(schema='partial-human-box-truth-v1',recordId=rid,sourceImageSha256=h,boxes=truth,coverage='Only listed boxes are labelled; all other page pixels are ignored.'))
        save(ignore_path,dict(recordId=rid,default='ignore-outside-human-boxes',quarantined=quarantine))
        family=known.get(image.name,{}).get('family') or ('Gulf source log' if doc['name'].startswith('608') else 'Other current source log')
        group=rows[0]['source_group'];record=dict(id=rid,sourceKind='current',current=True,synthetic=False,outside=False,family=family,split='train',truthStatus='partial-human-confirmed-boxes',humanConfirmed=bool(truth),diagnosticOnly=False,ready=bool(truth),image=str(image.resolve()),annotation=str(annotation.resolve()),truth=str(truth_path.resolve()),ignore=str(ignore_path.resolve()),width=doc['width'],height=doc['height'],originalWidth=doc['width'],originalHeight=doc['height'],window=[b[1],b[3]],boxes=truth,legend=[dict(id=e['id'],rawLabel=e['label_printed'],bbox=e['box']) for e in doc['legend'] if not e.get('deprecated')],baseWell=group,groupId=group,documentId=ident,lineage=dict(sourceImageSha256=h,sourceAnnotationSha256=sha(annotation),humanStoreSha256=sha(paths[ident]),humanStoreRevision=doc['revision'],normalizedManifestSha256=sha(manifest_path),normalization='Raw native pixel coordinates; no resize. Partial truth; ignore outside listed boxes.'),sourceHashes=dict(image=h,annotation=sha(annotation)),manifest=str(manifest_path.resolve()),existingThumbnail=None,existingReport=None,knownIssues=issues,eval=dict(command=None,manifest=str(truth_path.resolve()),contract='Partial box classification/generation only. Do not score full-page segmentation against sparse boxes.'),reviewPriority=100 if issues else 20,rawOutputsImmutable=True)
        records.append(record);bydoc[ident]=record
    # Cached Terra remaining-task proposals only. Evaluation/heldout crops are not exported.
    for batch in packets['batches']:
        if batch['purpose']!='remaining' or batch['document_id'] not in bydoc:continue
        answer=answers.get(batch['batch_id'])
        if not answer:continue
        record=bydoc[batch['document_id']];doc=docmap[batch['document_id']];legend={e['id']:e['label_printed'] for e in doc['legend']}
        for row in answer['matrix']:
            for cell in row:
                if not cell or cell.get('item_id') not in items:continue
                item=items[cell['item_id']];ids=cell['legend_ids'];bad=any(i not in legend for i in ids);status=cell['status'];confidence=cell['confidence'];reason=cell.get('reason','')
                if bad:reason+=' Cached legend IDs are absent from the current legend.'
                priority=100 if bad or status in ['bad_crop','mixed'] else 90 if status=='uncertain' or confidence=='low' else 65 if confidence=='medium' else 30
                crop=DATA/'terra-grid-20261108/blind'/item['crop']
                suggestions.append(dict(id='terra-cache-'+digest([record['id'],item['source_box']])[:20],recordId=record['id'],documentId=item['document_id'],sourceHashes=record['sourceHashes'],sourceImageSha256=record['sourceHashes']['image'],bbox=item['source_box'],interval=dict(topY=item['source_box'][1],bottomY=item['source_box'][3]),suggestedLithology=' / '.join(legend.get(i,'UNRESOLVED LEGEND ID '+str(i)) for i in ids),rawLegendIds=ids,model='gpt-6.1-terra',modelConfidence=dict(value=confidence,calibrated=False,empiricalAccuracy=None),uncertaintyReason=reason,status=status,disagreement=None,reviewPriority=priority,evidenceCropPath=str(crop.resolve()) if crop.exists() else None,evidenceGridPath=batch['grid'],legendContextPaths=batch['legends'],modelInputHash=answer.get('input_hash',batch['input_hash']),promptHash=digest('terra-grid-v1: source legend only; row-major; match/mixed/uncertain/bad_crop; high/medium/low; no approvals'),promptHashScope='Saved contract text only; original raw prompt not separately archived',imageHash=sha(crop) if crop.exists() else None,cached=True,humanConfirmed=False,truthStatus='AI suggestion; human review required',autoPromoted=False))
    # Add exact cached Jev choices for current training pages. Preserve source caption IDs.
    for export in sorted((DATA/'jev-e2e-20261108/boat-results/exports').glob('*/export.json')):
        data=read(export);record=next((r for r in records if r['sourceHashes']['image']==data['image_sha256']),None)
        decisions=DATA/'jev-e2e-20261108/boat-decisions'/(data['case_id']+'.json')
        if not record or not decisions.exists():continue
        d=read(decisions);root=export.parent;boxes_path=root/'boxes.json'
        if not boxes_path.exists():continue
        boxes=read(boxes_path);lookup={c['pixel_sha256']:c for c in boxes['queries']}
        record['existingReport']=str((root/'report.html').resolve()) if (root/'report.html').exists() else None
        record['existingThumbnail']=str((root/'source-context.png').resolve()) if (root/'source-context.png').exists() else None
        for chunk in data['chunks']:
            c=d['by_pixel'].get(chunk['pixel_sha256']);q=lookup.get(chunk['pixel_sha256'])
            if not c or not q:continue
            bbox=[q['box'][0],chunk['y0'],q['box'][2],chunk['y1']];path=root/'crops'/(chunk['pixel_sha256']+'.png');conf=c.get('confidence')
            structural=c.get('quality') in ['mixed','bad_crop'];priority=100 if structural else 90 if c.get('review') else 40
            suggestions.append(dict(id='jev-cache-'+digest([record['id'],bbox])[:20],recordId=record['id'],documentId=record['documentId'],sourceHashes=record['sourceHashes'],sourceImageSha256=record['sourceHashes']['image'],bbox=bbox,interval=dict(topY=bbox[1],bottomY=bbox[3]),suggestedLithology=c['label'],rawLegendIds=[c['legend_id']],model='jev-1.14.0',modelConfidence=dict(value=conf,calibrated=False,empiricalAccuracy=None),uncertaintyReason='; '.join(filter(None,[c.get('quality'),'Model requested review' if c.get('review') else None])),status=c.get('quality'),disagreement=None,reviewPriority=priority,evidenceCropPath=str(path.resolve()) if path.exists() else None,modelInputHash=chunk['pixel_sha256'],promptHash=sha(BASE.parent/'bakeoff/strategies/jev_sourcecards.py'),promptHashScope='Full archived request builder source',imageHash=sha(path) if path.exists() else None,cached=True,humanConfirmed=False,truthStatus='AI suggestion; human review required',autoPromoted=False))
    seen={};dedup=[]
    for r in suggestions:
        key=(r['recordId'],tuple(r['bbox']),r['model'])
        if key not in seen:dedup.append(r);seen[key]=r
    for r in dedup:
        peers=[p for p in dedup if p['recordId']==r['recordId'] and p['bbox']==r['bbox'] and p['model']!=r['model']]
        if peers:
            disagreement=any(p['suggestedLithology'].casefold()!=r['suggestedLithology'].casefold() for p in peers);r['disagreement']=disagreement
            if disagreement:r['reviewPriority']=max(95,r['reviewPriority']);r['uncertaintyReason']+='; Models disagree on this exact box'
    dedup.sort(key=lambda r:(-r['reviewPriority'],str(r['modelConfidence']['value']),r['id']))
    for r in records:
        r.update(page=Path(r['image']).stem,set='labeller-current-human',origin='current',imagePath=r['image'],annotationPath=r['annotation'],truthPath=r['truth'],ignorePath=r['ignore'],manifestPath=r['manifest'],thumbnail=r['existingThumbnail'],report=r['existingReport'],evalManifest=r['truth'],evalSuite='partial-human-boxes',usage='train-box-classification-and-pattern-generation',labelsHeldOut=False,showTruthInReview=True,modelReadableTruth=False,aiSuggested=False)
    summary=dict(records=len(records),ready=sum(r['ready'] for r in records),readyHumanBoxes=sum(len(r['boxes']) for r in records),cachedSuggestions=len(dedup),cachedTerra=sum(r['model']=='gpt-6.1-terra' for r in dedup),cachedJev=sum(r['model']=='jev-1.14.0' for r in dedup),planned='Conservative existing human train subset plus cached proposals; bounded Terra follow-up',available=len(records),heldoutAggregateOnly=sealed,sealedCount=None,sealedCountStatus='Not inspected by this scoped exporter; root catalog owns sealed totals',heldoutExcluded=True,missingBlockers=['Unlabelled pixels are ignored. This subset is not full-page segmentation truth.','Known bad or stale human boxes are quarantined. No model guess is human truth.','Some parent/base-well train/test leakage was reported; do not use this train subset as an independent test.'])
    save(DEST/'labeller-inventory.json',dict(name='labeller',schema='dataset-review-inventory-v1',owner='labeller',createdUtc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),summary=summary,records=records))
    save(DEST/'review-suggestions.json',dict(schema='dataset-review-suggestions-v1',owner='labeller',sort='Structural errors, model disagreements, low confidence first',confidenceWarning='Model confidence is uncalibrated. It is not measured empirical accuracy.',summary=summary,records=dedup))
    save(ASSETS/'source-context.json',dict(documents={i:dict(name=docmap[i]['name'],image=str((DATASET/'images'/docmap[i]['image_name']).resolve()),size=[docmap[i]['width'],docmap[i]['height']],legend=docmap[i]['legend'],column=docmap[i]['boxes']['interpreted_lithology_column'],split='train',recordId=r['id'],imageSha256=r['sourceHashes']['image']) for i,r in bydoc.items()},heldoutExcluded=True))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
