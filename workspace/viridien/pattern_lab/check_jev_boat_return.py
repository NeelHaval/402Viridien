"""Check returned receipts and portable reports. Does not run inference/scoring."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'data/jev-e2e-20261108'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cases={c['case_id']:c for c in read(ROOT/'manifest.json')['cases']}
    checked=[]
    for ident in ['ee53c8e06c27030e','ff55e97bd8089596']:
        p=ROOT/'boat-results/exports'/ident;c=cases[ident];data=read(p/'export.json');receipt=read(p/'receipt.json');evaluation=read(p/'evaluation.json');er=read(p/'evaluation-receipt.json');rr=read(p/'report-receipt.json');boxes=read(p/'boxes.json')
        assert data['case_id']==evaluation['case_id']==ident
        for field in ['image_sha256','annotation_sha256','window']:assert data[field]==c[field],field
        assert evaluation['truth_sha256']==c['truth_sha256']
        assert sha(p/'export.json')==receipt['export_sha256']
        for name,digest in receipt['artifacts'].items():assert sha(p/name)==digest,name
        for name,digest in rr['artifacts'].items():assert sha(p/name)==digest,name
        assert sha(p/'evaluation.json')==er['evaluation_sha256']
        assert sha(p/'prediction.csv')==er['prediction_sha256']==evaluation['prediction_sha256']
        for record in [receipt,er]:
            env=record['environment'];assert len(env['affinity'])==2 and env['threads']==2 and env['seed']==0
            assert env['packages']['numpy']=='2.2.6'
        for policy in ['strict_harness','clay_shale_equivalent_v1']:
            m=evaluation[policy];assert abs(m['score']-m['row_acc']*m['boundary_f1'])<1e-12
        for box in boxes['queries']:
            x0,y0,x1,y1=box['box'];assert x0<x1 and c['window'][0]<=y0<y1<=c['window'][1]
            assert [x0,x1]==[boxes['source_column'][0],boxes['source_column'][2]]
        assert len(boxes['queries'])==len(data['chunks'])
        page=(p/'report.html').read_text(encoding='utf-8');assert page.lower().startswith('<!doctype html>')
        assert page.count('src="data:image/png;base64,')==3
        assert 'src="file:' not in page and 'src="http' not in page
        checked.append(dict(case_id=ident,queries=len(boxes['queries']),strict=evaluation['strict_harness']['score'],equivalent=evaluation['clay_shale_equivalent_v1']['score'],report_sha256=sha(p/'report.html'),metric_sha256=rr['metric_sha256'],grading_sha256=rr['grading_sha256']))
    output=dict(status='verified',cases=checked,source_manifest_sha256=sha(ROOT/'boat-packet/source-manifest.json'),finished_local_checkpoint_sha256=sha(ROOT/'archive/results-before-boat.json'))
    (ROOT/'returned-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('BOAT_RETURN_OK cases=2 PNGs=6 portable_reports=2 boxes='+str(sum(c['queries'] for c in checked)))


if __name__=='__main__':main()
