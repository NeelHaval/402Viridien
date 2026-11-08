"""Use the root catalog's existing source-page IDs. Preserve old IDs as aliases."""
from pathlib import Path
from export_dataset_review import DEST,ASSETS,read,save,sha


def main():
    inv=read(DEST/'labeller-inventory.json');queue=read(DEST/'review-suggestions.json');mapping={}
    for r in inv['records']:
        if r['sourceKind']=='current':mapping[r['id']]='current:'+Path(r['imagePath']).stem
    for r in inv['records']:
        if r['id'] in mapping:
            old=r['id'];r['id']=mapping[old];r['alsoPublishedAs']=list(dict.fromkeys(r.get('alsoPublishedAs',[])+[old]));r['truthRecordId']=old
        if r.get('parentRecordId') in mapping:r['parentRecordId']=mapping[r['parentRecordId']]
    for s in queue['records']:
        if s['recordId'] in mapping:
            s['originalRequestRecordId']=s.get('originalRequestRecordId',s['recordId']);s['recordId']=mapping[s['recordId']]
        if s.get('parentRecordId') in mapping:s['parentRecordId']=mapping[s['parentRecordId']]
    inv['recordIdPolicy']='Current pages update the root current:<source stem> record. The first cached page-hash IDs remain aliases; do not add them as more pages.'
    inv['recordIdAliases']=mapping;queue['recordIdAliases']=mapping
    save(DEST/'labeller-inventory.json',inv);save(DEST/'review-suggestions.json',queue)
    save(ASSETS/'record-id-aliases.json',mapping)
    ready=[]
    for r in inv['records']:
        if r['ready']:ready.append(dict(id=r['id'],alsoPublishedAs=r.get('alsoPublishedAs',[]),image=r['imagePath'],annotation=r['annotationPath'],truth=r['truthPath'],ignore=r['ignorePath'],split=r['split'],groupId=r['groupId'],sourceHashes=r['sourceHashes'],truthSha256=sha(r['truthPath']),ignoreSha256=sha(r['ignorePath'])))
    save(ASSETS/'ready-subset-manifest.json',dict(schema='frozen-partial-human-train-v1',records=ready,ready=len(ready),humanBoxes=inv['summary']['readyHumanBoxes'],defaultIgnoreOutsideHumanBoxes=True,usage='Sparse human box classification and training prototypes; not independent full-page segmentation truth',sourceInventorySha256=sha(DEST/'labeller-inventory.json')))
    print('CANONICAL_IDS_OK current_pages='+str(len(mapping))+' extra_page_count=0')


if __name__=='__main__':main()
