"""Read cached v4 bed predictions without running inference in a review request."""
import bisect
import hashlib
import json
from pathlib import Path

from feedback import canonical


class BedGuesses:
    def __init__(self, root):
        self.root=Path(root);self.cache={}
        conf=json.loads((self.root/'conf.json').read_text(encoding='utf-8'))
        self.families={r['log']:r['family'] for r in conf['logs']}
        self.version='v4-beds-'+hashlib.sha256((self.root/'conf.json').read_bytes()).hexdigest()[:12]

    @staticmethod
    def key(name):return name.replace('#','_').replace(' ','_')[:90]

    def load(self,name):
        if name not in self.cache:
            key=self.key(name);bed_file=self.root/'beds'/(key+'.json');windows=self.root/'classified_v4'/(key+'.json')
            if not bed_file.exists() or not windows.exists():self.cache[name]=None
            else:
                beds=json.loads(bed_file.read_text(encoding='utf-8'))['beds']
                scores={}
                for r in json.loads(windows.read_text(encoding='utf-8')):
                    if r.get('state')=='ok' and r.get('bed') and r.get('conf') is not None:
                        scores[tuple(r['bed'])]=float(r['conf'])
                self.cache[name]=(beds,[b['top'] for b in beds],scores)
        return self.cache[name]

    @staticmethod
    def entries(label,legend):
        # A slash with spaces denotes model twins. Do not split printed Clay/Shale.
        names=(label or '').split(' / ')
        return list(dict.fromkeys(e['id'] for name in names for e in legend
                                  if not e.get('deprecated') and canonical(e['label_printed'])==canonical(name)))

    def proposal(self,task):
        loaded=self.load(task['document_name'])
        if not loaded:return None
        beds,tops,scores=loaded;box=task['segment']['box'];y=(box[1]+box[3])//2
        i=bisect.bisect_right(tops,y)-1
        if i<0 or y>=beds[i]['bottom']:return None
        bed=beds[i];family=self.families.get(task['document_name'],'Unknown layout')
        ids=self.entries(bed.get('label'),task['legend']);second=self.entries(bed.get('second'),task['legend'])
        confidence=scores.get((bed['top'],bed['bottom']))
        warning=[]
        if bed.get('state')!='ok':warning.append('The bed model could not name this crop. Choose from the source legend.')
        if not ids:warning.append('The model label does not map to a current legend choice.')
        if len(ids)>1:warning.append('The model found matching legend symbols. Choose the correct label manually.')
        if 'D-582' in family:warning.append('This model is weak on the D-582 layout. Check the source symbol closely.')
        if 'FREEMAN' in task['document_name'].upper():warning.append('Freeman has known look-alike legend choices. Check the symbol, not only its colour.')
        if bed['bottom']-bed['top']<24:warning.append('This bed is thin. Check that the box contains a full motif.')
        choices=[]
        for ident in dict.fromkeys(ids+second):
            entry=next(e for e in task['legend'] if e['id']==ident)
            choices.append(dict(legend_ids=[ident],label=entry['label_printed'],reference=dict(kind='legend',label=entry['label_printed'],document_name=task['document_name'])))
        context=task['context_box'];top=max(context[1],bed['top']);bottom=min(context[3],bed['bottom'])
        proposed=[box[0],top,box[2],bottom] if bottom-top>=24 else None
        return dict(task_id=task['id'],document_id=task['document_id'],source_version=task['version'],
                    box=proposed,bed=[bed['top'],bed['bottom']],
                    prediction=dict(engine='v4-beds',model_version=self.version,model_label=bed.get('label'),
                                    confidence=confidence,family=family,ambiguous=len(ids)!=1 or bed.get('state')!='ok',
                                    status='close' if ids and len(ids)==1 and confidence is not None and confidence>=.8 and not warning else 'check',
                                    warning=' '.join(warning),choices=choices,training_examples=None))


def build_manifest(bundle,model,held_sets=()):
    proposals={}
    for group in bundle['sets']:
        for task in group['tasks']:
            proposal=model.proposal(task)
            if proposal:proposals[task['id']]=proposal
    return dict(schema=1,model_version=model.version,held_sets=list(held_sets),tasks=proposals)
