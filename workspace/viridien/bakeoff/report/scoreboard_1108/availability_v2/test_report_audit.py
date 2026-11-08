"""Reject stale availability reports without reading protected truth files."""
import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from report_contract import validate_report,validate_parent_audit,sha


def fixture(tmp_path):
    base=tmp_path/'bakeoff'/'report'/'scoreboard';base.mkdir(parents=True)
    scorer=base.parents[1]/'harness'/'score.py';scorer.parent.mkdir();scorer.write_text('pass\n',encoding='utf-8')
    (base/'index_baseline_v1.html').write_text('<p>Historical baseline</p>',encoding='utf-8')
    (base/'scores.json').write_text(json.dumps({'records':[{'case_id':'case','strategy':'s19_combine'}]}),encoding='utf-8')
    (base/'raw_results.json').write_text('{}',encoding='utf-8')
    receipt={k:sha(v) for k,v in {'preserved_html_sha256':base/'index_baseline_v1.html','scores_json_sha256':base/'scores.json','raw_results_json_sha256':base/'raw_results.json','shared_scorer_sha256':scorer}.items()}
    receipt['original_html_sha256']=receipt['preserved_html_sha256']
    (base/'baseline_v1_preservation_receipt.json').write_text(json.dumps(receipt),encoding='utf-8')
    d={'version':'availability_v2','complete':True,'selection_promoted':False,'generalization_verified':False,'pins':{},'records':[{'case_id':'case','strategy':'s19_combine','structural_supervision':{'filled_rows':3},'views':{}}],'summaries':[]}
    path=base/'derived.json';audit_path=base/'audit.json'
    def save(data=d):
        path.write_text(json.dumps(data),encoding='utf-8')
        a={'complete':True,'errors':[],'exact_keys':True,'derived_sha256':sha(path),'source_pins':{},'records':len(data['records']),'expected':1}
        audit_path.write_text(json.dumps(a),encoding='utf-8')
    save()
    return base,d,path,audit_path,save


def test_stale_derived_bytes_rejected_even_when_json_still_valid(tmp_path):
    base,d,path,audit,_=fixture(tmp_path)
    assert validate_report(path,audit,base)[0]['complete']
    path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='derived bytes'):validate_report(path,audit,base)


def test_duplicate_derived_key_rejected_even_with_fresh_audit_hash(tmp_path):
    base,d,path,audit,save=fixture(tmp_path);d['records'].append(copy.deepcopy(d['records'][0]));save()
    with pytest.raises(ValueError,match='duplicate'):validate_report(path,audit,base)


def test_changed_baseline_rejected(tmp_path):
    base,d,path,audit,_=fixture(tmp_path)
    (base/'scores.json').write_text('{"records":[]}',encoding='utf-8')
    with pytest.raises(ValueError,match='Preserved baseline changed'):validate_report(path,audit,base)


def test_new_view_cannot_publish_a_champion(tmp_path):
    base,d,path,audit,save=fixture(tmp_path);d['selection_promoted']=True;save()
    with pytest.raises(ValueError,match='cannot promote'):validate_report(path,audit,base)


def test_parent_clearance_cannot_be_reused_for_changed_data(tmp_path):
    base,d,path,audit,_=fixture(tmp_path)
    parent=base/'parent_audit.json'
    parent.write_text(json.dumps(dict(errors=[],derived_sha256=sha(path),records=1,original_scores_unchanged=True,selection_promoted=False,inference_started=False)),encoding='utf-8')
    assert validate_parent_audit(parent,path,1)['records']==1
    path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='stale'):validate_parent_audit(parent,path,1)
