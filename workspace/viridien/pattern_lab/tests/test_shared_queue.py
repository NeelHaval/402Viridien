import copy,json,sys,uuid
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Store,write
from review_queue import ReviewQueue,version

@pytest.fixture
def queue(tmp_path):
    catalog=[];groups=[]
    for name in ('a','b','c'):
        doc=dict(id=name,name=name,width=100,height=3000,revision=0,boxes={'interpreted_lithology_column':[10,0,90,3000]},legend=[dict(id=1,label_printed='Sand',box=[0,0,10,10])],segments=[])
        write(tmp_path/'seeds'/(name+'.json'),doc);catalog.append(dict(id=name,name=name))
    write(tmp_path/'documents.json',catalog)
    for number in range(3):
        candidates=[]
        for row in range(4):
            for name in ('a','b','c'):
                ident=f't{number}-{name}-{row}';top=100+number*800+row*100
                segment=dict(id=ident,box=[10,top,90,top+48],legend_ids=[1],status='suggested',approved=False,example=False,comments=[],provenance='test source')
                candidates.append(dict(id=ident,document_id=name,segment=segment))
        groups.append(dict(key=f'g{number}',title=f'Pattern {number}',names=['Sand'],document_count=3,candidates=candidates))
    return ReviewQueue(Store(tmp_path),dict(groups=groups,references=[]))

def assign(queue,session):return queue.bundle(session['token'],1)['sets'][0]
def operation(task,action='accept'):
    return dict(op_id=uuid.uuid4().hex,task_id=task['id'],version=task['version'],segment=copy.deepcopy(task['segment']),action=action)

def test_set_alternates_documents_and_does_not_grow_on_refresh(queue):
    s=queue.create();g=assign(queue,s)
    assert [t['document_id'] for t in g['tasks'][:3]]==['a','b','c']
    for _ in range(3):assert len(assign(queue,s)['tasks'])==7

def test_reviewers_receive_different_sets(queue):
    one=assign(queue,queue.create('one'));two=assign(queue,queue.create('two'))
    assert one['key']!=two['key']
    assert not ({t['id'] for t in one['tasks']}&{t['id'] for t in two['tasks']})

def test_existing_accepted_areas_are_excluded(queue):
    doc=queue.store.get('a');approved=dict(id='existing',box=[10,100,90,148],legend_ids=[1],approved=True,status='approved',example=False,comments=[])
    queue.store.save('a',0,segment=approved)
    group=assign(queue,queue.create())
    assert 't0-a-0' not in [t['id'] for t in group['tasks']]

def test_five_examples_finish_set_and_sixth_is_refused(queue):
    s=queue.create();g=assign(queue,s)
    for task in g['tasks'][:5]:assert queue.apply(s['token'],operation(task))['ok']
    result=queue.apply(s['token'],operation(g['tasks'][5]))
    assert result['quota'] and not result['ok']
    assert queue.counts()[g['key']]==5
    assert assign(queue,s)['key']!=g['key']

def test_draft_saves_without_counting_as_an_accepted_example(queue):
    s=queue.create();t=assign(queue,s)['tasks'][0];op=operation(t,'draft');op['segment']['comments']=[dict(text='Check this edge.')]
    result=queue.apply(s['token'],op)
    assert result['ok'] and not result['segment']['approved']
    assert result['segment']['comments'][0]['text']=='Check this edge.'
    assert queue.counts()['g0']==0
    restored=ReviewQueue(queue.store,queue.index)
    assert restored.current(queue.tasks[t['id']])['comments'][0]['text']=='Check this edge.'

def test_retry_after_restart_is_idempotent(queue):
    s=queue.create();t=assign(queue,s)['tasks'][0];op=operation(t)
    first=queue.apply(s['token'],op);revision=queue.store.get(t['document_id'])['revision']
    restarted=ReviewQueue(queue.store,queue.index)
    assert restarted.apply(s['token'],op)==first
    assert queue.store.get(t['document_id'])['revision']==revision
    assert restarted.counts()['g0']==1

def test_different_boxes_merge_even_when_document_revision_changes(queue):
    s=queue.create();g=assign(queue,s);one=g['tasks'][0];two=g['tasks'][3]
    assert one['document_id']==two['document_id']
    assert queue.apply(s['token'],operation(one))['ok']
    assert queue.apply(s['token'],operation(two))['ok']
    assert queue.counts()['g0']==2

def test_same_box_conflict_does_not_replace_saved_label(queue):
    s=queue.create();t=assign(queue,s)['tasks'][0]
    first=operation(t);assert queue.apply(s['token'],first)['ok']
    stale=operation(t,'draft');stale['segment']['box'][1]+=10
    result=queue.apply(s['token'],stale)
    assert result['conflict'] and not result['ok']
    assert queue.current(queue.tasks[t['id']])['box']==t['segment']['box']

def test_other_reviewer_cannot_use_an_assignment(queue):
    one=queue.create();two=queue.create();t=assign(queue,one)['tasks'][0]
    with pytest.raises(PermissionError):queue.apply(two['token'],operation(t))

def test_state_poll_sees_labels_saved_by_another_client(queue):
    s=queue.create();t=assign(queue,s)['tasks'][0]
    queue.apply(s['token'],operation(t))
    update=queue.sync(s['token'],[t['id']])
    assert update['tasks'][0]['segment']['approved']
    assert update['counts']['g0']==1

def test_recovery_after_json_save_before_database_commit(queue):
    s=queue.create();t=assign(queue,s)['tasks'][0];op=operation(t)
    queue.apply(s['token'],op);revision=queue.store.get(t['document_id'])['revision']
    with queue.db() as db:db.execute('DELETE FROM operations WHERE id=?',(op['op_id'],))
    assert queue.apply(s['token'],op)['ok']
    assert queue.store.get(t['document_id'])['revision']==revision

def test_offline_draft_can_reclaim_an_unassigned_expired_set(queue):
    s=queue.create();t=assign(queue,s)['tasks'][0]
    with queue.db() as db:db.execute('DELETE FROM leases')
    result=queue.apply(s['token'],operation(t,'draft'))
    assert result['ok'] and result['segment']['approved'] is False

def test_reassigned_offline_work_becomes_a_conflict_not_a_stuck_save(queue):
    s=queue.create();t=assign(queue,s)['tasks'][0]
    with queue.db() as db:db.execute('DELETE FROM leases')
    other=queue.create();assign(queue,other)
    result=queue.apply(s['token'],operation(t,'draft'))
    assert result['conflict'] and not result['ok'] and 'offline' in result['error']


def test_excluding_loaded_sets_assigns_fresh_work_and_keeps_other_reviewers_separate(queue):
    one=queue.create('one');first=assign(queue,one)
    second=queue.bundle(one['token'],1,exclude=[first['key']])['sets'][0]
    assert first['key']!=second['key']
    third=assign(queue,queue.create('two'))
    assert third['key'] not in {first['key'],second['key']}

def test_upload_receipts_are_visible_and_idempotent(queue):
    one=queue.create('one');task=assign(queue,one)['tasks'][0];op=operation(task)
    first=queue.apply(one['token'],op);again=queue.apply(one['token'],op)
    assert first['saved_at']==again['saved_at']
    own=queue.upload_status(one['token']);assert len(own)==1
    assert own[0]['saves']==1 and own[0]['last_save']==first['saved_at']
    two=queue.create('two');assert len(queue.upload_status(two['token']))==1
    assert len(queue.upload_status(two['token'],True))==2


def test_focus_resumes_a_box_from_another_assigned_set(queue):
    one=queue.create('one');groups=queue.bundle(one['token'],3)['sets']
    target=groups[-1]['tasks'][1]
    resumed=queue.bundle(one['token'],1,focus=target['id'])['sets'][0]
    assert resumed['key']==groups[-1]['key']
    assert any(t['id']==target['id'] for t in resumed['tasks'])


def test_all_candidates_skipped_moves_to_another_set_even_with_old_focus(queue):
    reviewer=queue.create();group=assign(queue,reviewer);anchor=group['tasks'][0]
    ids=[t['id'] for t in queue.groups[group['key']]['candidates']]
    with queue.db() as db:
        db.executemany('INSERT OR IGNORE INTO skips VALUES(?,?)',[(reviewer['id'],i) for i in ids])
    next_group=queue.bundle(reviewer['token'],1,focus=anchor['id'])['sets'][0]
    assert next_group['key']!=group['key']
    assert queue.counts()[group['key']]==0
    assert queue.current(queue.tasks[anchor['id']]) is None
