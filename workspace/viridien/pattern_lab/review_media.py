"""Bake review crops into a small portable HTML file."""
import base64, hashlib, io, json, threading, uuid, time
from pathlib import Path
from model import HOME
from feedback import Feedback
from legend_repairs import legend_warning
from legend_palette import colours

class ReviewMedia:
    def __init__(self,queue,images):
        self.queue=queue;self.images=images;self.cache=queue.store.data/'review-crops';self.cache.mkdir(exist_ok=True)
        self.lock=threading.RLock()
        self.feedback=Feedback(queue,self)
        self.proposal_provider=None
        self.palettes={}
    def image(self,doc,box):
        ident=hashlib.sha256((doc['id']+':'+','.join(map(str,box))).encode()).hexdigest()
        path=self.cache/(ident+'.png')
        with self.lock:
            if not path.exists():
                image=self.images.crop(doc,box);buffer=io.BytesIO();image.save(buffer,format='PNG')
                temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');temp.write_bytes(buffer.getvalue())
                for attempt in range(8):
                    try:temp.replace(path);break
                    except PermissionError:
                        if attempt==7:raise
                        time.sleep(.025*(attempt+1))
        return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
    @staticmethod
    def signature(task):
        legend=[{k:e.get(k) for k in ('id','box','label_printed','deprecated')} for e in task['legend']]
        return hashlib.sha256(json.dumps([task['context_box'],legend],sort_keys=True).encode()).hexdigest()[:16]
    def hydrate(self,bundle,refresh=True):
        if refresh:self.feedback.refresh()
        tasks=[t for group in bundle['sets'] for t in group['tasks']]
        by_doc={}
        for task in tasks:by_doc.setdefault(task['document_id'],[]).append(task)
        for ident,rows in by_doc.items():
            doc=self.queue.document(ident)
            legend=[]
            for e in doc['legend']:
                key=(doc['id'],tuple(e['box']))
                if key not in self.palettes:self.palettes[key]=colours(self.feedback.cached_image(doc,e['box']))
                legend.append(dict(e,image=self.image(doc,e['box']),colours=self.palettes[key]))
            overview=self.image(doc,doc['boxes']['lithology_legend']) if doc['boxes'].get('lithology_legend') else None
            for task in rows:
                task['media_version']=self.signature(task)
                task['image']=self.image(doc,task['context_box']);task['legend']=legend
                task['legend_overview']=overview;task['legend_warning']=legend_warning(legend)
                prepared=self.proposal_provider(task) if self.proposal_provider else None
                if prepared:task.update(segment=prepared['segment'],prediction=prepared['prediction'])
                else:self.feedback.enrich(doc,task,self.feedback.cached_image(doc,task['context_box']))
                for choice in task['prediction']['choices'][:1]:
                    ref=choice['reference'];refdoc=self.queue.document(ref['document_id'])
                    ref['image']=self.image(refdoc,ref['box']);ref['document_name']=refdoc['name']
        return bundle
    def html(self,boot):
        template=(HOME/'static/review.html').read_text(encoding='utf-8')
        css=(HOME/'static/review.css').read_text(encoding='utf-8')
        js='\n'.join((HOME/'static'/name).read_text(encoding='utf-8') for name in ('review-connect.js','review-details.js','review.js'))
        payload=json.dumps(boot,ensure_ascii=False).replace('<','\\u003c')
        return template.replace('/* REVIEW_CSS */',css).replace('/* REVIEW_JS */',js).replace('null /* REVIEW_BOOT */',payload)
