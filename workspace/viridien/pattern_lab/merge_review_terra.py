"""Join the bounded Terra result to the shared queue. Never edit source labels."""
import collections
import shutil
from pathlib import Path
from PIL import Image
from export_dataset_review import BASE,DATA,ROOT,DEST,ASSETS,read,save,sha,digest,boxdict

PACK=ASSETS/'terra-first-pass'


def main():
    packet=read(PACK/'packet.json');result=read(PACK/'result.json')
    assert result['model']==packet['model']=='gpt-6.1-terra'
    items={i['itemId']:i for i in packet['items']};answers={i['itemId']:i for i in result['items']}
    assert len(items)==len(packet['items'])==len(answers)==len(result['items'])<=20
    assert set(items)==set(answers)
    inventory=read(DEST/'labeller-inventory.json');queue=read(DEST/'review-suggestions.json')
    for name in ['labeller-inventory.json','review-suggestions.json']:
        target=ASSETS/'frozen-cached-v1'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(DEST/name,target)
    records={r['id']:r for r in inventory['records']};groups={g['documentId']:g for g in packet['groups']};fresh=[]
    for item_id,item in items.items():
        cell=answers[item_id];group=groups[item['documentId']];legend={e['id']:e['label'] for e in group['rawLegend']}
        ids=cell['legendIds'];assert isinstance(ids,list) and all(type(i)==int and i in legend for i in ids)
        assert cell['status'] in ['match','mixed','uncertain','bad_crop'];assert cell['confidence'] in ['high','medium','low']
        assert isinstance(cell['boundaryIssue'],bool) and isinstance(cell['reason'],str)
        if cell['status']=='match':assert len(ids)==1
        assert sha(item['crop'])==item['cropSha256'] and sha(item['context'])==item['contextSha256']
        labels=[legend[i] for i in ids];parent=records.get(item.get('parentRecordId') or item['recordId'])
        assert parent and parent['split']=='train' and not parent['diagnosticOnly']
        if item['sourceKind']=='synthetic':
            lineage_path=PACK/item['documentId']/(item_id+'-lineage.json');lineage=read(lineage_path);source=Path(lineage['source'])
            assert sha(source)==item['sourceImageSha256']
            with Image.open(source) as im:w,h=im.size
            annotation=ASSETS/'synthetic-annotations'/(item['recordId']+'.json')
            save(annotation,dict(image_name=source.name,width=w,height=h,bounding_boxes=dict(interpreted_lithology_column=[0,0,w,h]),legendReferencePage=parent['image']))
            r=dict(id=item['recordId'],page=source.stem,set='labeller-synthetic-diagnostic',sourceKind='synthetic',origin='synthetic',current=False,synthetic=True,outside=False,family=parent['family'],split='synthetic-diagnostic',truthStatus='procedural-template; not human-confirmed',humanConfirmed=False,diagnosticOnly=True,ready=False,image=str(source),imagePath=str(source),annotation=str(annotation),annotationPath=str(annotation),truth=str(lineage_path),truthPath=str(lineage_path),ignore=None,ignorePath=None,width=w,height=h,originalWidth=w,originalHeight=h,window=[0,min(h,180)],boxes=[boxdict(item['bbox'],lithology=lineage['intendedLabel'],humanConfirmed=False,role='synthetic-template')],baseWell=parent['baseWell'],groupId=parent['groupId'],parentRecordId=parent['id'],lineage=lineage,sourceHashes=dict(image=item['sourceImageSha256'],annotation=sha(annotation)),manifest=str((DATA/'genlab/generated/manifest.json').resolve()),manifestPath=str((DATA/'genlab/generated/manifest.json').resolve()),existingThumbnail=item['crop'],thumbnail=item['crop'],existingReport=None,report=None,knownIssues=[dict(bbox=item['bbox'],explanation='Synthetic diagnostic reuses its training parent. It is not independent test data or human truth.')],eval=dict(command=None,manifest=str(lineage_path),contract='Synthetic texture diagnostic only'),evalManifest=str(lineage_path),evalSuite='synthetic-diagnostic',reviewPriority=50,usage='diagnostic-only',labelsHeldOut=False,showTruthInReview=True,modelReadableTruth=False,aiSuggested=True,rawOutputsImmutable=True)
            records[r['id']]=r
        else:
            assert item['recordId'] in records and item['sourceImageSha256']==parent['sourceHashes']['image']
        r=records[item['recordId']];peers=[]
        for prior in queue['records']:
            if prior['recordId']==r['id'] and prior['bbox']==item['bbox']:
                peers.append(dict(model=prior['model'],suggestedLithology=prior['suggestedLithology'],relation='exact original box',suggestionId=prior['id']))
        # Compare to saved human entries only after the blind model call.
        doc_path=DATA/'reviews'/(item['documentId']+'.json')
        if not doc_path.exists():doc_path=DATA/'seeds'/(item['documentId']+'.json')
        doc=read(doc_path);doc_labels={e['id']:e['label_printed'] for e in doc['legend']}
        if item['sourceKind']=='current':
            for seg in doc['segments']:
                if seg.get('approved') and seg['box']==item['bbox']:
                    peers.append(dict(model='saved-human-review',suggestedLithology=' / '.join(doc_labels.get(i,'Unknown source ID') for i in seg['legend_ids']),relation='exact original box; human label not supplied to Terra',annotationId=seg['id']))
        suggestion=' / '.join(labels) if labels else None
        disagree=bool(suggestion and any(p['suggestedLithology'] and p['suggestedLithology'].strip().casefold()!=suggestion.strip().casefold() for p in peers)) if peers else None
        structural=cell['boundaryIssue'] or cell['status'] in ['bad_crop','mixed']
        priority=120 if structural else 115 if disagree else 110 if cell['status']=='uncertain' or cell['confidence']=='low' else 90 if cell['confidence']=='medium' else 50
        image_inputs=dict(crop=item['cropSha256'],context=item['contextSha256'],legendPanels={p:sha(p) for p in group['legendPanels']},board=sha(group['board']))
        fresh.append(dict(id='terra-first-pass-'+item_id,recordId=r['id'],documentId=item['documentId'],sourceHashes=r['sourceHashes'],sourceImageSha256=item['sourceImageSha256'],bbox=item['bbox'],interval=dict(topY=item['bbox'][1],bottomY=item['bbox'][3]),suggestedLithology=suggestion,rawLegendIds=ids,model='gpt-6.1-terra',modelConfidence=dict(value=cell['confidence'],calibrated=False,empiricalAccuracy=None,scope='Confidence in the stated status and legend decision; the model can confidently abstain'),uncertaintyReason=cell['reason'],status=cell['status'],boundaryIssue=cell['boundaryIssue'],disagreement=disagree,disagreementEvidence=peers,reviewPriority=priority,evidenceCropPath=item['crop'],contextCropPath=item['context'],contextBox=item['contextBox'],legendContextPaths=group['legendPanels'],modelInputHash=digest(dict(prompt=packet['promptSha256'],images=image_inputs)),promptHash=packet['promptSha256'],promptPath=str((PACK/'prompt.txt').resolve()),imageHash=item['cropSha256'],imageInputHashes=image_inputs,rawModelOutputPath=str((PACK/'result.json').resolve()),rawModelOutputSha256=sha(PACK/'result.json'),inputPacketSha256=sha(PACK/'packet.json'),cached=False,humanConfirmed=False,truthStatus='AI suggestion; human review required',autoPromoted=False,diagnosticOnly=item['sourceKind']=='synthetic',parentRecordId=item.get('parentRecordId')))
    suggestions={s['id']:s for s in queue['records']}
    suggestions.update({s['id']:s for s in fresh});ordered=sorted(suggestions.values(),key=lambda r:(-r['reviewPriority'],r['id']))
    summary=dict(inventory['summary'],records=len(records),ready=sum(r['ready'] for r in records.values()),freshTerra=len(fresh),freshTerraStructural=sum(s['boundaryIssue'] or s['status'] in ['bad_crop','mixed'] for s in fresh),freshTerraUncertain=sum(s['status']=='uncertain' for s in fresh),freshTerraDisagreements=sum(s['disagreement'] is True for s in fresh),syntheticDiagnostics=sum(r['sourceKind']=='synthetic' for r in records.values()),totalSuggestions=len(ordered),empiricalAccuracy=None,empiricalAccuracyNote='No accuracy estimate was made from this risk-selected training review pass.')
    inventory.update(summary=summary,records=list(records.values()),terraPass=dict(resultPath=str((PACK/'result.json').resolve()),resultSha256=sha(PACK/'result.json'),promptSha256=packet['promptSha256'],items=len(fresh),humanAnnotationsChanged=0))
    queue.update(summary=summary,records=ordered,terraPass=inventory['terraPass'])
    save(DEST/'labeller-inventory.json',inventory);save(DEST/'review-suggestions.json',queue)
    save(ASSETS/'terra-merge-receipt.json',dict(status='verified',items=len(fresh),summary=summary,inventorySha256=sha(DEST/'labeller-inventory.json'),suggestionsSha256=sha(DEST/'review-suggestions.json'),humanAnnotationsChanged=0))
    print('TERRA_MERGE_OK',len(fresh),'items',len(records),'records',len(ordered),'suggestions; human annotations changed=0')


if __name__=='__main__':main()
