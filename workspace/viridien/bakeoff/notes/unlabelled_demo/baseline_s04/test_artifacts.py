"""Check the saved demo without running a strategy or reading held-out truth."""
from pathlib import Path
import csv
import json
import re

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def test_selection_has_current_dev_size_and_no_inherited_weights():
    choice=read('selection.json')
    assert choice['selected']=='s04'
    assert choice['inherited_weights_used'] is False
    assert len(choice['ranking'])==5
    assert all(r['windows']==51 and r['wells']==5 for r in choice['ranking'])
    assert choice['ranking'][0]['well_macro_score']>choice['ranking'][1]['well_macro_score']


def test_inventory_accounts_for_every_input_page():
    with (HERE/'inventory.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    summary=read('summary.json');results=read('results.json')
    assert len(rows)==summary['input_pages']==651
    assert len({r['id'] for r in rows})==len(rows)
    counts={r['id']:r['status'] for r in results}
    for r in rows:
        assert r['status']==counts.get(r['id'],'not_run_missing_reviewed_boxes')


def test_intervals_and_confidence_have_same_rows_and_stay_in_column():
    boxes={r['id']:r['column'] for r in read('boxes.json')}
    count=0
    for result in read('results.json'):
        if result['status']!='predicted': continue
        name=result['id'];column=boxes[name]
        with (HERE/'predictions'/(name+'.csv')).open() as f:
            reader=csv.DictReader(f)
            assert reader.fieldnames==['top_y','bottom_y','lithology','cluster_id','top_depth','bottom_depth']
            rows=list(reader)
        conf=read('confidence/'+name+'.json')
        assert len(rows)==len(conf)==result['intervals']
        previous=column[1]
        for r,c in zip(rows,conf):
            a,b=int(r['top_y']),int(r['bottom_y'])
            assert previous<=a<b<=column[3];previous=b
            assert (a,b,r['lithology'])==(c['top_y'],c['bottom_y'],c['lithology'])
            assert r['lithology'] in result['legend_labels']
            assert r['top_depth']==r['bottom_depth']==''
            assert c['confidence'] is None and c['confidence_status']=='unverified'
        count+=len(rows)
    assert count==read('summary.json')['intervals']


def test_all_gallery_links_are_local_and_exist():
    page=(HERE/'index.html').read_text(encoding='utf-8')
    assert 'linear-gradient' not in page and 'radial-gradient' not in page
    for link in re.findall(r'(?:src|href)="([^"]+)"',page):
        assert '://' not in link
        dest=(HERE/link).resolve()
        assert dest.is_relative_to(HERE) and dest.exists(),link


def test_blocks_have_no_prediction_file():
    for r in read('results.json'):
        if r['status'].startswith('blocked'):
            assert r['intervals']==0
            assert not (HERE/'predictions'/(r['id']+'.csv')).exists()


def test_full_page_inference_and_view_coverage():
    boxes={r['id']:r['column'] for r in read('boxes.json')}
    for r in read('results.json'):
        if r['status']!='predicted':continue
        views=r['windows'];column=boxes[r['id']]
        assert views[0]['y0']==column[1] and views[-1]['y1']==column[3]
        assert all(a['y1']==b['y0'] for a,b in zip(views,views[1:]))
        assert all(0<v['y1']-v['y0']<=4000 for v in views)
        assert r['inference_seconds']<60
