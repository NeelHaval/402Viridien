import copy,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feedback import descriptor,distance,refine_box,Feedback
from legend_repairs import callaway_entries,install_repairs,CALLAWAY
from model import Store,write
from review_queue import ReviewQueue
from review_media import ReviewMedia

def pattern(dashes=False,colour='white'):
    im=Image.new('RGB',(100,120),colour);p=ImageDraw.Draw(im)
    for y in range(8,116,12):
        for x in range(8,95,24):p.rectangle((x,y,x+(14 if dashes else 2),y+2),fill='black')
    return im

def test_shape_descriptor_separates_dots_and_dashes_with_same_fill():
    dots=pattern();dash=pattern(True)
    assert distance(descriptor(dots),descriptor(dots.crop((0,12,100,108))))<distance(descriptor(dots),descriptor(dash))

def test_box_growth_stops_at_colour_boundary():
    im=pattern(colour='yellow');ImageDraw.Draw(im).rectangle((0,0,99,29),fill='green')
    box=refine_box(im,[10,1000,110,1120],[10,1050,110,1070],1.4)
    assert box[1]>=1030 and box[3]<=1120 and box[3]-box[1]>20

def test_source_checked_callaway_entries_are_separate_and_unique():
    rows=callaway_entries()
    assert len(rows)==32 and len({r['id'] for r in rows})==32
    assert all(r['box'][3]-r['box'][1]==14 for r in rows)
    assert {r['label_printed'] for r in rows}>={'Shale','Shale gray','Chalk','Cement','Salt','Sandstone'}

def test_legend_repair_keeps_accepted_boxes_and_legacy_ids(tmp_path):
    legend=[dict(id=i,label_printed='CEMENT CHALK' if i==0 else 'Merged row',box=[0,0,10,10]) for i in range(5)]
    accepted=dict(id='human',box=[10,10,20,30],legend_ids=[0],status='approved',approved=True,example=True,comments=[dict(text='Keep this')])
    doc=dict(id='a',name=CALLAWAY,width=1300,height=6000,revision=0,legend=legend,segments=[accepted]);write(tmp_path/'documents.json',[dict(id='a',name=CALLAWAY)]);write(tmp_path/'seeds/a.json',doc)
    store=Store(tmp_path);assert install_repairs(store)==['a'];saved=store.get('a')
    assert saved['segments']==[accepted] and len(saved['legend'])==37
    assert all(e['deprecated'] for e in saved['legend'][:5]);assert install_repairs(store)==[]

def make_feedback(tmp_path):
    doc=dict(id='a',name='a',width=100,height=500,revision=0,boxes={'interpreted_lithology_column':[0,0,100,500]},legend=[dict(id=1,label_printed='Sandstone',box=[0,0,100,120]),dict(id=2,label_printed='Shale',box=[0,120,100,240])],segments=[])
    write(tmp_path/'documents.json',[dict(id='a',name='a')]);write(tmp_path/'seeds/a.json',doc)
    queue=ReviewQueue(Store(tmp_path),dict(groups=[],references=[]))
    class Images:
        def crop(self,doc,box):return pattern(box[1]==120,'yellow' if box[1]!=120 else 'green').resize((box[2]-box[0],box[3]-box[1]))
    media=ReviewMedia(queue,Images());return doc,media.feedback

def test_accepted_colour_mismatch_is_withheld_without_changing_labels(tmp_path):
    doc,f=make_feedback(tmp_path)
    s=dict(id='human',box=[0,260,100,380],legend_ids=[2],status='approved',approved=True,example=True,comments=[])
    f.queue.store.save('a',0,segment=s);f.refresh()
    assert len(f.flagged)==1 and not f.refs
    assert f.queue.store.get('a')['segments'][0]==s

def test_acceptance_refreshes_training_and_uses_only_document_legend_ids(tmp_path):
    doc,f=make_feedback(tmp_path);f.refresh();assert not f.refs
    s=dict(id='human',box=[0,260,100,380],legend_ids=[1],status='approved',approved=True,example=True,comments=[])
    f.queue.store.save('a',0,segment=s);f.refresh();assert len(f.refs)==1
    prediction=f.predict(doc,descriptor(pattern(colour='yellow')))
    assert prediction['choices'][0]['legend_ids']==[1] and prediction['status']=='close'

def test_enrichment_never_changes_saved_human_boxes_or_labels(tmp_path):
    doc,f=make_feedback(tmp_path);f.refresh()
    segment=dict(id='saved',box=[0,260,100,300],legend_ids=[2],comments=[dict(text='Keep')])
    task=dict(segment=copy.deepcopy(segment),context_box=[0,240,100,360],version='saved-human-hash')
    f.enrich(doc,task,pattern(colour='yellow'))
    assert task['segment']==segment and 'prediction' in task

def test_new_guess_does_not_select_a_label_before_human_review(tmp_path):
    doc,f=make_feedback(tmp_path);f.refresh()
    task=dict(segment=dict(id='new',box=[0,260,100,300],legend_ids=[2],suggestion='wrong old guess'),context_box=[0,240,100,360],version='new')
    f.enrich(doc,task,pattern(colour='yellow'))
    assert not task['segment']['legend_ids'] and task['prediction']['choices']
