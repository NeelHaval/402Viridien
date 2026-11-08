"""Check demo files. Do not run inference or read interval truth."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import html
import json
import math
import re
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def test_frozen_choice_and_baseline():
    choice=read('selection.json')
    assert choice['selected']=='s19' and choice['selection_sweep'] is False
    frozen=ROOT/choice['source']
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==choice['source_sha256']
    record=json.loads(frozen.read_text(encoding='utf-8'))
    assert record['chosen_strategy']=='s19'
    for name,digest in read('summary.json')['source_hashes'].items():
        if name in record['source_fingerprints']:
            assert digest==record['source_fingerprints'][name]
    assert read('baseline_s04/summary.json')['strategy']=='s04'
    assert read('baseline_s04/summary.json')['intervals']==64
    assert read('baseline_s04/selection.json')['selected']=='s04'
    assert (HERE/'baseline_s04/REPORT.md').is_file()


def test_inventory_and_actual_counts():
    with (HERE/'inventory.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    summary=read('summary.json');results=read('results.json')
    assert len(rows)==summary['input_pages']==651
    assert len({r['id'] for r in rows})==len(rows)
    assert len(results)==summary['reviewed_pages']==17
    assert summary['skipped_pages']==634
    assert summary['strategy']=='s19' and summary['thread_limit']==1
    assert summary['torch_threads']==summary['torch_interop_threads']==1
    assert summary['peak_rss_gib'] < 4_000_000_000/2**30
    for name,digest in summary['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert dict(Counter(r['status'] for r in results))==summary['statuses']
    assert summary['failures']==sum(r['status']=='error' for r in results)==0
    counts={r['id']:r['status'] for r in results}
    for r in rows:
        assert r['status']==counts.get(r['id'],'not_run_missing_reviewed_boxes')
    for folder,pattern,key in [('predictions','*.csv','prediction_files'),
            ('confidence','*.json','confidence_files'),('legends','*.png','legend_images'),
            ('overlays','*.png','overlays'),('contact_sheets','*.jpg','contact_sheets')]:
        assert len(list((HERE/folder).glob(pattern)))==summary[key]
    assert summary['prediction_files']==summary['confidence_files']==summary['predicted_pages']
    assert summary['windows']==sum(len(r.get('windows',[])) for r in results)
    predicted={r['id'] for r in results if r['status']=='predicted'}
    assert {p.stem for p in (HERE/'predictions').glob('*.csv')}==predicted
    assert {p.stem for p in (HERE/'confidence').glob('*.json')}==predicted


def test_prediction_format_and_native_scores():
    boxes={r['id']:r['column'] for r in read('boxes.json')}
    count=0
    for result in read('results.json'):
        if result['status']!='predicted': continue
        name=result['id'];column=boxes[name]
        with (HERE/'predictions'/(name+'.csv')).open(encoding='utf-8',newline='') as f:
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
            assert r['lithology'] in result['legend_labels']+['Unknown']
            assert int(r['cluster_id'])>=0
            assert r['top_depth']==r['bottom_depth']==''
            assert c['confidence_status']=='unverified'
            assert c['confidence_source']==('unknown' if c['confidence'] is None else 'native')
            assert set(c['native_scores'])=={'match_cost','match_margin','boundary_probability'}
            assert all(math.isfinite(v) for v in c['native_scores'].values())
            assert 0<=c['native_scores']['boundary_probability']<=1
            assert not any(k.startswith('raw_descriptor') for k in c)
        count+=len(rows)
    assert count==read('summary.json')['intervals']


def test_gallery_links_are_local_portable_and_exist():
    for gallery in ['index.html','baseline_s04/index.html']:
        path=HERE/gallery
        page=path.read_text(encoding='utf-8')
        assert 'linear-gradient' not in page and 'radial-gradient' not in page
        for link in re.findall(r'(?:src|href)="([^"]+)"',page):
            link=html.unescape(link)
            assert '://' not in link and '\\' not in link
            dest=(path.parent/link).resolve()
            assert dest.is_relative_to(HERE) and dest.is_file(),link
    assert 'S19.' in (HERE/'index.html').read_text(encoding='utf-8')


def test_blocks_have_no_prediction_file():
    for r in read('results.json'):
        if r['status'].startswith('blocked'):
            assert r['intervals']==0
            assert not (HERE/'predictions'/(r['id']+'.csv')).exists()
            assert not (HERE/'confidence'/(r['id']+'.json')).exists()


def test_full_page_inference_and_view_coverage():
    boxes={r['id']:r['column'] for r in read('boxes.json')}
    for r in read('results.json'):
        if r['status']!='predicted':continue
        views=r['windows'];column=boxes[r['id']]
        assert views[0]['y0']==column[1] and views[-1]['y1']==column[3]
        assert all(a['y1']==b['y0'] for a,b in zip(views,views[1:]))
        assert all(0<v['y1']-v['y0']<=4000 for v in views)
        for v in views:
            assert '\\' not in v['overlay']
            assert (HERE/v['overlay']).is_file()


def test_images_are_valid_and_overlays_have_white_background():
    for folder,pattern in [('overlays','*.png'),('legends','*.png'),('contact_sheets','*.jpg')]:
        for path in (HERE/folder).glob(pattern):
            with Image.open(path) as im:
                im.verify()
    for path in (HERE/'overlays').glob('*.png'):
        with Image.open(path) as im:
            assert im.size==(1000,1280) and im.mode=='RGB'
            assert im.getpixel((0,0))==(255,255,255)


def test_report_matches_last_summary():
    summary=read('summary.json')
    report=(HERE/'REPORT.md').read_text(encoding='utf-8')
    last=(f"Last summary: {summary['predicted_pages']} predicted pages; {summary['intervals']} intervals; "
          f"{summary['failures']} errors; {summary['blocked_pages']} blocked pages; {summary['skipped_pages']} skipped pages.")
    assert last in report
    for target in re.findall(r'\]\(([^)]+)\)',report):
        assert '\\' not in target
        dest=(HERE/target).resolve()
        assert dest.is_relative_to(HERE) and dest.exists(),target
