"""Validate the labeller handoff without model calls or a new scorer."""
import collections
from pathlib import Path
from export_dataset_review import DEST,ASSETS,read,save,sha


def main():
    inventory=read(DEST/'labeller-inventory.json');queue=read(DEST/'review-suggestions.json');records={r['id']:r for r in inventory['records']}
    assert len(records)==len(inventory['records'])
    assert len({(r['sourceHashes']['image'],tuple(r['window'])) for r in records.values()})==len(records)
    human=0
    for r in records.values():
        assert r['split'] not in ['test','verify','sealed','shared_verify']
        assert Path(r['imagePath']).is_file() and Path(r['annotationPath']).is_file()
        assert Path(r['truthPath']).is_file()
        if r['ready']:
            assert r['humanConfirmed'] and not r['diagnosticOnly'] and r['split']=='train'
            assert read(r['ignorePath'])['default']=='ignore-outside-human-boxes'
        for box in r['boxes']:
            x0,y0,x1,y1=[box[k] for k in ['x0','y0','x1','y1']]
            assert 0<=x0<x1<=r['width'] and 0<=y0<y1<=r['height']
            human+=bool(box['humanConfirmed'])
    ids=set();priorities=[];fresh=0;missing_evidence=0
    for s in queue['records']:
        assert s['id'] not in ids;ids.add(s['id']);assert s['recordId'] in records
        r=records[s['recordId']];assert s['sourceImageSha256']==r['sourceHashes']['image']
        assert not s['humanConfirmed'] and not s['autoPromoted']
        assert s['sourceLegendBook'] in queue['legendBooks']
        assert s['modelConfidence']['calibrated'] is False and s['modelConfidence']['empiricalAccuracy'] is None
        x0,y0,x1,y1=s['bbox'];assert 0<=x0<x1<=r['width'] and 0<=y0<y1<=r['height']
        priorities.append(s['reviewPriority']);assert s.get('promptHash') and s.get('modelInputHash')
        p=s.get('evidenceCropPath')
        if not p or not Path(p).is_file():missing_evidence+=1
        if not s['cached']:
            fresh+=1;assert sha(p)==s['imageHash'];assert sha(s['rawModelOutputPath'])==s['rawModelOutputSha256']
    assert priorities==sorted(priorities,reverse=True)
    assert missing_evidence==0
    assert human==inventory['summary']['readyHumanBoxes']
    report=dict(status='verified',records=len(records),ready=sum(r['ready'] for r in records.values()),humanBoxes=human,suggestions=len(ids),freshTerra=fresh,missingEvidence=missing_evidence,sourceCounts=dict(collections.Counter(r['sourceKind'] for r in records.values())),inventorySha256=sha(DEST/'labeller-inventory.json'),suggestionsSha256=sha(DEST/'review-suggestions.json'),heldoutExposed=False,humanLabelsChanged=0)
    save(ASSETS/'validation.json',report)
    print('LABELLER_CHECK_OK records='+str(len(records))+' ready='+str(report['ready'])+' human_boxes='+str(human)+' suggestions='+str(len(ids))+' fresh_terra='+str(fresh)+' missing_evidence=0')


if __name__=='__main__':main()
