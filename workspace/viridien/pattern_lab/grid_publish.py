"""Attach validated Terra suggestions to unsaved tasks. Never approve labels."""
import copy
import json
import argparse
from pathlib import Path

from grid_lab import OUT, request_hash, save, validate_response
from model import HOME, Store, read
from review_queue import ReviewQueue
from verification import BedGuesses


def prediction(item,cell,base,doc,input_hash):
    result=copy.deepcopy(base)
    result['box']=list(item['source_box'])
    old=result['prediction'];allowed={e['id']:e for e in doc['legend'] if not e.get('deprecated')}
    ids=cell['legend_ids'];clean=cell['status']=='match' and len(ids)==1
    first=old.get('choices',[{}])[0].get('legend_ids',[]) if old.get('choices') else []
    agree=clean and not old.get('ambiguous') and ids==first
    choices=[]
    ordered=list(ids)+[ident for c in old.get('choices',[]) for ident in c['legend_ids']]
    for ident in dict.fromkeys(ordered):
        if ident not in allowed:continue
        label=allowed[ident]['label_printed']
        choices.append(dict(legend_ids=[ident],label=label,reference=dict(kind='legend',label=label,document_name=doc['name'])))
    labels=[allowed[i]['label_printed'] for i in ids]
    warning=cell['reason']
    if cell['status']!='match':warning='Terra marked this crop '+cell['status'].replace('_',' ')+'. '+warning
    if clean and first and not agree:warning+=' The bed model suggests '+old['choices'][0]['label']+'. Compare both with the source legend.'
    result['prediction']=dict(engine='terra-grid',model_version='gpt-6.1-terra:grid-v1',model_label=' / '.join(labels) or 'Choose from source legend',
                              confidence=None,terra_confidence=cell['confidence'],terra_status=cell['status'],
                              model_agreement=agree,input_hash=input_hash,warning=warning,choices=choices,
                              ambiguous=not(clean and cell['confidence']=='high'),status='close' if agree and cell['confidence']=='high' else 'check',
                              training_examples=None,family=old.get('family','Source legend'))
    return result


def publish():
    packets=read(OUT/'blind/packets.json');results=read(OUT/'terra-results.json')
    answers={r['batch_id']:r for r in results['batches']};items={i['item_id']:i for i in packets['items']}
    pack_path=HOME/'static/verification-predictions.txt';pack=read(pack_path);queue=ReviewQueue(Store())
    backup=OUT/'private/pre-terra-predictions.txt'
    if not backup.exists():backup.write_bytes(pack_path.read_bytes())
    baseline=read(backup)
    updates={};skipped=[];statuses={};bed_model=None
    # Validate every remaining batch before changing the public pack.
    for batch in packets['batches']:
        if batch['purpose']!='remaining':continue
        response=answers[batch['batch_id']];digest=request_hash(batch)
        if response.get('input_hash')!=digest:raise ValueError('Source pixels or request changed: '+batch['batch_id'])
        for cell in validate_response(batch,response):
            item=items[cell['item_id']];tid=item['task_id'];task=queue.task_payload(tid)
            if task['version']!=item['source_version'] or task['segment'].get('approved') or task['segment'].get('status')=='rejected':
                skipped.append(tid);continue
            base=baseline['tasks'].get(tid)
            if base and base['prediction'].get('engine')=='terra-grid':
                if bed_model is None:bed_model=BedGuesses(HOME.parent/'bakeoff/analysis/categories')
                base=bed_model.proposal(task)
            if not base:
                base=dict(task_id=tid,document_id=task['document_id'],source_version=task['version'],bed=[item['source_box'][1],item['source_box'][3]],
                          prediction=dict(choices=[],family='Source legend'))
            updates[tid]=prediction(item,cell,base,queue.document(task['document_id']),digest)
            statuses[cell['status']]=statuses.get(cell['status'],0)+1
    pack['tasks'].update(updates);pack['terra_grid']=dict(model='gpt-6.1-terra',schema=1,tasks=len(updates),human_confirmation_required=True)
    temp=pack_path.with_suffix('.tmp');temp.write_text(json.dumps(pack,ensure_ascii=False,separators=(',',':')),encoding='utf-8');temp.replace(pack_path)
    report=dict(published_suggestions=len(updates),skipped_changed_or_saved=skipped,statuses=statuses,annotations_changed=0)
    save(OUT/'publish.json',report);print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out-dir',type=Path);args=parser.parse_args()
    if args.out_dir:OUT=args.out_dir.resolve()
    publish()
