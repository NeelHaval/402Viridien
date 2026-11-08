import importlib.util,json,sys,textwrap
from pathlib import Path
import pytest
p=Path(__file__).with_name('bounded_local.py');spec=importlib.util.spec_from_file_location('bounded_local',p);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def test_stuck_owned_child_cannot_block_later_case_or_lose_completed(tmp_path):
    raw=tmp_path/'records.json';b.atomic(raw,{'records':[]});progress=tmp_path/'progress.json';script=tmp_path/'worker.py'
    script.write_text(textwrap.dedent('''
        import json,os,sys,time
        from pathlib import Path
        raw=Path(sys.argv[1]);progress=Path(sys.argv[2])
        def save(p,v):
            q=p.with_suffix('.tmp');q.write_text(json.dumps(v),encoding='utf-8');q.replace(p)
        for case in ('first','stuck','later'):
            data=json.loads(raw.read_text(encoding='utf-8'))
            if any(r['case_id']==case for r in data['records']):continue
            save(progress,dict(pid=os.getpid(),case_id=case,strategy='s',attempt_id=case,phase='strategy_run',attempt_started_perf=time.perf_counter()))
            if case=='stuck':
                while True:time.sleep(.05)
            data['records'].append(dict(case_id=case,strategy='s',marker='kept'));save(raw,data)
        '''),encoding='utf-8')
    done=lambda:{(r['case_id'],r['strategy']) for r in b.read(raw)['records']}
    def timeout(attempt,receipt):
        data=b.read(raw);data['records'].append(dict(case_id=attempt['case_id'],strategy='s',error='timeout',seconds=None));b.atomic(raw,data)
    result=b.supervise({(c,'s') for c in ('first','stuck','later')},done,lambda:[sys.executable,str(script),str(raw),str(progress)],progress,timeout,tmp_path/'logs',cap=.7,poll=.02)
    assert result['complete']
    assert len(b.read(raw)['records'])==3
    assert b.read(raw)['records'][0]['marker']=='kept'
    assert b.read(raw)['records'][1]['seconds'] is None
    assert b.read(raw)['records'][2]['case_id']=='later'
    assert len(result['events'])==2

def test_two_no_progress_exits_stop_finite_queue(tmp_path):
    with pytest.raises(RuntimeError,match='Two owned-child exits'):
        b.supervise({('x','s')},lambda:set(),lambda:[sys.executable,'-c','pass'],tmp_path/'none',lambda*a:None,tmp_path/'logs',cap=2,poll=.01)
    receipt=b.read(tmp_path/'logs'/'controller_receipt.json')
    assert len(receipt['events'])==2
