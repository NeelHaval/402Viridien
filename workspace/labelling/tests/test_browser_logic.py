"""Cross-check persisted grouping between browser JS and Python using Node."""
import json
import shutil
import subprocess
from pathlib import Path
import pytest
from test_labelslib import good_label,box,Y1,X1
import labelslib as L


def test_assignee_selection_does_not_reopen_another_persons_page():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for browser selection checks')
    source = (Path(__file__).resolve().parents[1] / 'labelling/app.js').read_text()
    script = 'const {pageForAssignee}=await import("data:text/javascript;base64,"+Buffer.from(' + json.dumps(source) + ').toString("base64"));\n'
    script += '''
const pages=[{stem:'reserved',in_progress:true},{stem:'e1',assignee:'E'},
  {stem:'e2',assignee:'E'},{stem:'b1',assignee:'B'},{stem:'n1',assignee:'N'}];
console.log(JSON.stringify([
  pageForAssignee(pages,'E','reserved').stem,
  pageForAssignee(pages,'B','e1').stem,
  pageForAssignee(pages,'N','b1').stem,
  pageForAssignee(pages,'E','e2').stem,
  pageForAssignee(pages,'in_progress','e1').stem,
  pageForAssignee(pages,'all',null).stem
]));
'''
    result = subprocess.run([node, '--input-type=module', '-e', script],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ['e1', 'b1', 'n1', 'e2', 'reserved', 'reserved']


def test_next_page_stays_with_the_selected_labeller_and_wraps():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for browser selection checks')
    source = (Path(__file__).resolve().parents[1] / 'labelling/app.js').read_text()
    script = 'const {nextPageForAssignee}=await import("data:text/javascript;base64,"+Buffer.from(' + json.dumps(source) + ').toString("base64"));\n'
    script += '''
const pages=[{stem:'reserved',in_progress:true},{stem:'e1',assignee:'E'},
  {stem:'b1',assignee:'B'},{stem:'e2',assignee:'E'}];
console.log(JSON.stringify([
  nextPageForAssignee(pages,'E','reserved').stem,
  nextPageForAssignee(pages,'E','e1').stem,
  nextPageForAssignee(pages,'E','e2').stem,
  nextPageForAssignee(pages,'B','e1').stem,
  nextPageForAssignee(pages,'N','e1'),
  nextPageForAssignee(pages,'all','reserved').stem
]));
'''
    result = subprocess.run([node, '--input-type=module', '-e', script],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ['e1', 'e2', 'e1', 'b1', None, 'e1']


def test_browser_grouping_matches_python():
    node=shutil.which('node')
    if not node: pytest.skip('Node is needed for browser logic parity checks')
    source=(Path(__file__).resolve().parents[1]/'labelling/app.js').read_text()
    lab=good_label();anchors=lab['depth_calibration']['anchors']
    cases=[
        [box(Y1,Y1+100,0,[0,610]),box(Y1,Y1+100,1,[610,830]),box(Y1,Y1+100,2,[830,1000])],
        [box(Y1,Y1+100,1,[X1,X1+30]),box(Y1,Y1+100,1,[X1+30,X1+70]),box(Y1,Y1+100,2,[X1+70,X1+100])],
        [box(Y1,Y1+100,None,[0,1]),box(Y1,Y1+100,2,[1,1000])],
    ]
    script='const {groupBoxes,depthFn}=await import("data:text/javascript;base64,"+Buffer.from('+json.dumps(source)+').toString("base64"));\n'
    script+='console.log(JSON.stringify('+json.dumps(cases)+'.map(bs=>groupBoxes(bs,'+json.dumps(lab['legend'])+','+json.dumps(anchors)+'))));\n'
    result=subprocess.run([node,'--input-type=module','-e',script],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==[L.group_boxes(bs,lab['legend'],anchors) for bs in cases]
