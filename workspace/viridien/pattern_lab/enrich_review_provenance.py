"""Keep source legend namespaces distinct and order the cached review queue."""
from pathlib import Path
from export_dataset_review import BASE,DATA,DEST,ASSETS,read,save,sha,digest
from grid_lab import request_hash


def main():
    inventory=read(DEST/'labeller-inventory.json');queue=read(DEST/'review-suggestions.json');records={r['id']:r for r in inventory['records']}
    books={};exports={}
    for path in (DATA/'jev-e2e-20261108/boat-results/exports').glob('*/export.json'):
        data=read(path);namespace='jev-native:'+data['image_sha256'][:20]+':'+digest([(e['id'],e['label'],e['sha256']) for e in data['entries']])[:16]
        books[namespace]=dict(sourceImageSha256=data['image_sha256'],sourceExportSha256=sha(path),kind='frozen native source legend',entries=[dict(id=e['id'],rawLabel=e['label'],nativeCropPath=str((path.parent/e['path']).resolve()),imageSha256=e['sha256']) for e in data['entries']])
        exports[str(path.parent.resolve())]=namespace
    packets=read(DATA/'terra-grid-20261108/blind/packets.json');legacy={}
    for batch in packets['batches']:
        if batch['purpose']!='remaining':continue
        # Verifies the saved geometry, source ID vocabulary, grid and legend pixels.
        assert request_hash(batch)==batch['input_hash']
        namespace='terra-grid:'+batch['input_hash'];legend_panels=[dict(path=p,imageSha256=sha(p)) for p in batch['legends']]
        books[namespace]=dict(kind='saved blind Terra source legend panels',legendPanels=legend_panels,allowedIds=batch['allowed_ids'],captionMapping='Cached IDs are raw. Suggested caption uses the current document mapping; compare the saved panels before acceptance.')
        for item_id in batch['item_ids']:legacy[(batch['document_id'],item_id)]=(namespace,batch)
    legacy_by_box={}
    for item in packets['items']:
        value=legacy.get((item['document_id'],item['item_id']))
        if value:legacy_by_box[(item['document_id'],tuple(item['source_box']))]=value
    fresh_packet=read(ASSETS/'terra-first-pass/packet.json');fresh_groups={g['documentId']:g for g in fresh_packet['groups']}
    for s in queue['records']:
        if s['model']=='jev-1.14.0':
            namespace=exports[str(Path(s['evidenceCropPath']).parent.parent.resolve())]
            s['sourceLegendBook']=namespace;s['sourceLegendReferences']=[namespace+'/L'+str(i) for i in s['rawLegendIds']]
            s['promptHashScope']='Saved request-builder source hash. Exact historical request text is not separately archived for every semantic-cache choice.'
            s['historicalPromptComplete']=False
        elif s['cached']:
            value=legacy_by_box[(s['documentId'],tuple(s['bbox']))];namespace,batch=value
            s['sourceLegendBook']=namespace;s['sourceLegendReferences']=[namespace+'/L'+str(i) for i in s['rawLegendIds']]
            s['sourceLegendCaptionMappingProvisional']=True;s['historicalPromptComplete']=False
            s['gridImageSha256']=sha(batch['grid']);s['savedRequestHashVerified']=True
        else:
            s['sourceLegendBook']='terra-first-pass:'+s['inputPacketSha256']+':'+s['documentId']
            s['sourceLegendReferences']=[s['sourceLegendBook']+'/L'+str(i) for i in s['rawLegendIds']]
            s['historicalPromptComplete']=True
            group=fresh_groups[s['documentId']]
            books[s['sourceLegendBook']]=dict(kind='Bounded blind Terra pass; current source legend',entries=[dict(id=e['id'],rawLabel=e['label']) for e in group['rawLegend']],legendPanels=[dict(path=p,imageSha256=sha(p)) for p in group['legendPanels']],inputPacketSha256=s['inputPacketSha256'])
        s['family']=records[s['recordId']]['family']
        value=s['modelConfidence']['value']
        if isinstance(value,(int,float)):
            s['confidenceOrderBand']='low' if value<.5 else 'medium' if value<.9 else 'high';s['withinModelConfidenceOrder']=value
        else:s['confidenceOrderBand']=value or 'unknown';s['withinModelConfidenceOrder']={'low':0,'medium':1,'high':2}.get(value,-1)
        if s.get('boundaryIssue') or s.get('status') in ['mixed','bad_crop']:s['reviewPriority']=120
        elif s.get('disagreement') is True:s['reviewPriority']=115
        elif s.get('status')=='uncertain' or s['confidenceOrderBand']=='low':s['reviewPriority']=110
    bands={'unknown':-1,'low':0,'medium':1,'high':2}
    queue['records'].sort(key=lambda s:(-s['reviewPriority'],bands[s['confidenceOrderBand']],s['model'],s['withinModelConfidenceOrder'],s['id']))
    for n,s in enumerate(queue['records'],1):s['reviewOrder']=n
    used={s['sourceLegendBook'] for s in queue['records']};books={k:v for k,v in books.items() if k in used}
    queue['legendBooks']=books
    queue['sort']='Structural errors, model disagreements, uncertain/low confidence first. Confidence bands are review heuristics, not calibrated accuracy.'
    queue['summary']['legacyPromptProvenanceLimit']='Legacy suggestions retain exact image/input hashes and builder/contract hashes. Full historical prompt text was not archived for every call. New 17-item Terra pass has an archived prompt and complete input hashes.'
    save(DEST/'review-suggestions.json',queue)
    save(ASSETS/'provenance-receipt.json',dict(status='verified',legendBooks=len(books),suggestions=len(queue['records']),suggestionsSha256=sha(DEST/'review-suggestions.json'),sharedLegendIdsAcrossDocumentsMerged=False))
    print('PROVENANCE_OK legend_books='+str(len(books))+' suggestions='+str(len(queue['records']))+' namespace_merges=0')


if __name__=='__main__':main()
