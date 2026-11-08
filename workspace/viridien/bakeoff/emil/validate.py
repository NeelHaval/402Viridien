"""Audit leakage controls, evaluator identity and exact outside window conversion."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))


def main():
    nb=json.loads((WORK/'emil/drive/HTB2026.ipynb').read_text(encoding='utf-8'))
    result={}
    result['notebook_sha256']=hashlib.sha256((WORK/'emil/drive/HTB2026.ipynb').read_bytes()).hexdigest()
    assert result['notebook_sha256']==json.loads((HERE/'port_manifest.json').read_text())['notebook_sha256']
    evidence={str(i):dict(execution_count=nb['cells'][i].get('execution_count'),
                         text=[''.join(o['text']) for o in nb['cells'][i].get('outputs',[]) if 'text' in o])
              for i in [21,25,28,35,40,43,47,51,52,53]}
    (HERE/'notebook_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    result['saved_execution_order']={k:v['execution_count'] for k,v in evidence.items()}
    names={'segmentation':'evaluate_layers_colab_cell.py','classification':'evaluate_segment_classes_cell.py','legend':'evaluate_legend_cell.py'}
    for name,filename in names.items():
        actual=(HERE/'evaluators'/f'{name}.py').read_bytes()
        expected=(WORK/'emil/drive/eval_cells'/filename).read_bytes()
        assert actual==expected
        result[name+'_sha256']=hashlib.sha256(actual).hexdigest()
    training=json.loads((HERE/'training.json').read_text())
    splits=json.loads((WORK/'emil/splits.json').read_text())
    labelled={path.stem for path in (WORK/'emil/labels').glob('*.json') if json.loads(path.read_text()).get('layers')}
    assert labelled.isdisjoint(training['pages'])
    assert set(training['pages'])<=set(splits['train'])
    assert set(training['pages']).isdisjoint(splits['test'])
    assert not any(s.startswith('0000000009') for s in training['pages'])
    result['real_labels_excluded']=sorted(labelled)
    result['training_pages']=len(training['pages'])
    windows=json.loads((WORK/'external_tests/eval_set.json').read_text())
    n=0
    for path in (HERE/'external_labels').glob('*.json'):
        data=json.loads(path.read_text()); name=path.stem
        with (WORK/'external_tests/truth'/f'{name}.csv').open(encoding='utf-8',newline='') as f:truth=list(csv.DictReader(f))
        expected=[]
        for w in [w for w in windows if w['log']==name]:
            for r in truth:
                a,b=max(w['y0'],int(r['top_y'])),min(w['y1'],int(r['bottom_y']))
                if b>a:expected.append((a,b,r['lithology']))
        got=[(l['top_px'],l['bottom_px'],l['composition'][0]['type']) for l in data['layers']]
        assert sorted(got)==sorted(expected),name
        mapping={e['id']:e['label_printed'] for e in data['legend']}
        assert all(mapping[l['composition'][0]['legend_id']]==l['composition'][0]['type'] for l in data['layers'])
        n+=len(got)
    result.update(external_pages=len(list((HERE/'external_labels').glob('*.json'))),external_windows=len(windows),external_intervals=n)
    # Inspect the actual source, not a duplicated fairness implementation.
    tree=ast.parse((HERE/'pipeline.py').read_text())
    loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_legend')
    branch=next(n for n in loader.body if isinstance(n,ast.If))
    expression=ast.unparse(branch.test)
    assert "'auto'" in expression and "'detect'" not in expression
    result['legacy_auto_label_branch']=expression
    result['fair_mode']='detect (explicit; never auto)'
    computational=['trim_borders','raw_row_features','zscore','row_features','to_table',
                   'boundary_score','robust_threshold','ink_boundaries','BoundaryNet','heatmap',
                   'cnn_probs','cnn_boundaries','cluster_beds','detect_swatches','run_hist',
                   'describe','descriptor_distance','prototypes','split_mixed','merge_named','lithology_layers']
    original_nodes={}
    for cell in nb['cells']:
        if cell['cell_type']!='code': continue
        source='\n'.join(s for s in ''.join(cell['source']).splitlines() if not s.startswith('!'))
        try: nodes=ast.parse(source).body
        except SyntaxError: continue
        original_nodes.update({n.name:n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))})
    local_nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in computational:
        assert ast.dump(original_nodes[name],include_attributes=False)==ast.dump(local_nodes[name],include_attributes=False),name
    result['unchanged_algorithm_definitions']=computational
    from emil.bootstrap import installed_arrow
    installed_arrow()
    from emil import pipeline as p
    pages=p.build_train_pages()
    result['live_training_rule_pages']=[a.stem for a,b in pages]
    result['prior_retrain_selection_difference']=dict(live_only=sorted({a.stem for a,b in pages}-set(training['pages'])),retrain_only=sorted(set(training['pages'])-{a.stem for a,b in pages}))
    assert labelled.isdisjoint({a.stem for a,b in pages})
    result['main_checkpoint_sha256']=hashlib.sha256(p.DEFAULT_MODEL.read_bytes()).hexdigest()
    result['main_checkpoint']=str(p.DEFAULT_MODEL.relative_to(WORK))
    result['training_provenance_limit']='Reconstructed selection from live rule + supplied splits; original checkpoint contains no training-page metadata.'
    # Check converted legend geometry against metadata and the immutable dataset view.
    import math
    missing_metadata = []
    for path in (HERE/'external_labels').glob('*.json'):
        data=json.loads(path.read_text())
        metadata_path=WORK/Path(data['provenance']['extraction'].replace('\\','/'))/'meta.json'
        if metadata_path.exists():
            meta=json.loads(metadata_path.read_text())
            rocks=[e for e in meta['legend_entries'] if e['kind']=='rock']
            scale=meta['render']['px_per_pt']
            bounds=[e[k] for e in rocks for k in ('bbox_pt','label_bbox_pt')]
            left=max(0,math.floor((min(b[0] for b in bounds)-3)*scale))
            top=max(0,math.floor((min(b[1] for b in bounds)-18)*scale))
            dx,dy=(10-left,-top) if data['provenance']['y_shift'] else (0,0)
            for e,g in zip(rocks,data['legend']):
                b=e['bbox_pt']
                assert g['box']==[math.floor(b[0]*scale)+dx,math.floor(b[1]*scale)+dy,math.ceil(b[2]*scale)+dx,math.ceil(b[3]*scale)+dy]
        else:
            missing_metadata.append(path.stem)
        assert json.loads((HERE/'external_view/test'/path.name).read_text())==data
        assert json.loads((HERE/'external_view/dataset/bbox_annotation'/path.name).read_text())==json.loads((WORK/'external_tests/bbox_annotation'/path.name).read_text())
        view_image=HERE/'external_view/dataset/images'/(path.stem+'.tif')
        source_image=WORK/'external_tests/images'/(path.stem+'.tif')
        assert view_image.stat().st_size==source_image.stat().st_size
        if not view_image.samefile(source_image):
            def digest_file(filename):
                h=hashlib.sha256()
                with filename.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
                return h.hexdigest()
            assert digest_file(view_image)==digest_file(source_image)
    result['converter_metadata_available'] = not missing_metadata
    result['external_conversion']='Reused; exact intervals, legend IDs/names and dataset view validated on VM. Original converter metadata absent: geometry validation relies on retained laptop validation evidence.'
    result['prior_validation_evidence']='live_validation.log'

    (HERE/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
