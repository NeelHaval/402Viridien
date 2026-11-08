"""Static notebook checks plus a deliberately small pure-helper extraction.

NEVER execute notebook cells, setup/install commands, training, OCR, downloads,
paid APIs or full prediction pipelines. Passing these checks does not validate
trained notebook weights, GPU execution or full geological model quality.
"""
import ast
from collections import Counter
import copy
import difflib
import json
from pathlib import Path
import re

import numpy as np
import pytest

NOTEBOOK=Path(__file__).resolve().parents[2]/'notebooks/main_pipeline.ipynb'


def _cells():
    notebook=json.loads(NOTEBOOK.read_text(encoding='utf-8-sig'))
    return [ast.parse(''.join(cell['source']),filename=f'{NOTEBOOK.name}:cell-{index}')
            for index,cell in enumerate(notebook['cells']) if cell['cell_type']=='code']


def _pure(names,constants=()):
    """Compile only explicitly selected def nodes; never a cell's statements."""
    matches=[]
    for tree in _cells():
        functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        if set(names)<=functions.keys():matches.append((tree,functions))
    assert len(matches)==1,f'expected one notebook cell containing helpers {names}'
    tree,functions=matches[0]
    namespace={'np':np,'re':re,'Counter':Counter,'difflib':difflib}
    for name in constants:
        assignments=[node for node in tree.body if isinstance(node,ast.Assign) and
                     any(isinstance(target,ast.Name) and target.id==name for target in node.targets)]
        assert len(assignments)==1
        namespace[name]=ast.literal_eval(assignments[0].value)
    selected=[copy.deepcopy(functions[name]) for name in names]
    forbidden={'open','eval','exec','compile','__import__','input','Popen','system','run','train','fit',
               'load','save','load_state_dict','download','read_text','write_text','read_bytes','write_bytes','fromfile','tofile','memmap'}
    for function in selected:
        assert not function.decorator_list
        assert not function.returns and all(argument.annotation is None for argument in function.args.args)
        assert all(isinstance(default,ast.Constant) for default in function.args.defaults)
        for node in ast.walk(function):
            assert not isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal))
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                assert name not in forbidden,f'helper acquired a side-effecting call: {name}'
    module=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    exec(compile(module,f'{NOTEBOOK.name}:approved-pure-helpers','exec'),namespace)
    return namespace


def test_every_code_cell_is_python_ast_parseable_without_executing():
    cells=_cells()
    assert len(cells)>=20
    assert any(isinstance(node,ast.FunctionDef) and node.name=='record_errors' for tree in cells for node in tree.body)
    assert any(isinstance(node,ast.FunctionDef) and node.name=='train_cnn' for tree in cells for node in tree.body)
    # The training function is recognized statically, never imported/executed.


def test_record_validation_handles_missing_fields_overlap_and_legend_references():
    validate=_pure(['record_errors'],['SCHEMA_VERSION'])['record_errors']
    record={'schema_version':1,'stem':'fixture','image_name':'fixture.png','written_by':'isolated-stress',
            'legend':[{'id':0,'box':[0,0,10,10],'label_printed':'Shale'}],
            'layers':[{'top_px':20,'bottom_px':30,'composition':[{'legend_id':0,'type':'Shale','proportion':1.0}]}]}
    assert validate(record)==[]
    assert len(validate({}))==6
    duplicate=copy.deepcopy(record);duplicate['legend'].append(copy.deepcopy(duplicate['legend'][0]))
    assert any('unique int' in error for error in validate(duplicate))
    overlap=copy.deepcopy(record);overlap['layers'].append({'top_px':29,'bottom_px':40})
    assert any('not overlap' in error for error in validate(overlap))
    unknown=copy.deepcopy(record);unknown['layers'][0]['composition'][0]['legend_id']=8
    assert any('not in this record' in error for error in validate(unknown))
    inverted=copy.deepcopy(record);inverted['layers'][0]['top_px']=31
    assert any('top_px < bottom_px' in error for error in validate(inverted))


def test_notebook_legend_iou_translation_symmetry_and_one_to_one(stress_seed,stress_iterations):
    early=_pure(['box_iou'])['box_iou']
    helpers=_pure(['iou','match_boxes','norm_text','text_scores','_box'])
    iou=helpers['iou'];rng=np.random.default_rng(stress_seed)
    for _ in range(stress_iterations):
        for _ in range(32):
            x,y=rng.integers(-100,100,size=2);w,h=rng.integers(1,80,size=2)
            a=[int(x),int(y),int(x+w),int(y+h)]
            b=[int(x+w//2),int(y+h//2),int(x+w+w//2),int(y+h+h//2)]
            assert iou(a,a)==1 and early(a,a)==1
            assert 0<=iou(a,b)<=1 and iou(a,b)==pytest.approx(iou(b,a))
            assert early(a,b)==pytest.approx(iou(a,b))
            offset=int(rng.integers(-500,500))
            assert iou([v+offset for v in a],[v+offset for v in b])==pytest.approx(iou(a,b))
    assert iou([0,0,0,0],[0,0,0,0])==0
    assert iou([0,0,10,10],[10,0,20,10])==0  # exclusive touching edge
    gt=[([0,0,10,10],'Shale'),([20,0,30,10],'Sand')]
    predicted=[([0,0,10,10],'Shale'),([0,0,10,10],'Duplicate'),([20,0,30,10],'Sand')]
    pairs=helpers['match_boxes'](gt,predicted,.5)
    assert len(pairs)==2 and len({pair[0] for pair in pairs})==2 and len({pair[1] for pair in pairs})==2
    exact,similarity=helpers['text_scores'](['Shale','Shale','Sand'],[' shale. ','Sand'])
    assert exact==2 and 0<=similarity<=1


def test_notebook_classification_metrics_perfect_wrong_and_missing_predictions():
    names=['norm','seg_class','rasterise','confusion','div','class_metrics','segment_level','evaluate']
    helpers=_pure(names,['BG','CLASS_KEYS'])
    evaluate=helpers['evaluate'];classes=['background','shale','sand']
    truth=[(10,20,'shale'),(20,30,'sand')]
    perfect=evaluate(truth,truth,classes)
    assert perfect['pixel_accuracy']==1 and perfect['macro_f1']==1
    assert perfect['n_pixels']==20 and perfect['segment_level']['segments_correct']==2
    wrong=evaluate(truth,[(10,20,'sand'),(20,30,'shale')],classes)
    assert wrong['pixel_accuracy']==0 and wrong['macro_f1']==0
    missing=evaluate(truth,[],classes)
    assert missing['pixel_accuracy']==0 and missing['per_class']['shale']['support_px']==10
    assert missing['segment_level']['segment_accuracy']==0
    assert helpers['seg_class']({'composition':[{'type':'Sand','proportion':.2},{'type':' Shale ','proportion':.8}]})=='shale'
    # Empty matrices are explicitly unavailable in this notebook helper, not 1.
    unavailable=helpers['class_metrics'](np.zeros((3,3),dtype=int),classes)
    assert np.isnan(unavailable['pixel_accuracy'])
