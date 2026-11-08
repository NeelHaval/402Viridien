"""Prepare normalized snapshots and future suggestions outside save requests."""
import copy,hashlib,json,threading,time
from collections import deque
from model import Images,read,write
from normalized import SplitPolicy,NormalizedDataset

class Background:
    def __init__(self,queue,foreground,autostart=True,images=None):
        from review_media import ReviewMedia
        self.queue=queue;self.policy=SplitPolicy(queue.store);self.media=ReviewMedia(queue,images or Images())
        self.media.feedback.policy=self.policy
        foreground.feedback=self.media.feedback
        foreground.proposal_provider=self.proposal
        self.dataset=NormalizedDataset(queue,self.media,self.policy)
        self.event=threading.Event();self.stop=threading.Event();self.lock=threading.RLock();self.todo=deque();self.queued=set();self.cache={}
        self.root=queue.store.data/'background';self.root.mkdir(exist_ok=True)
        self.state=dict(building=True,error=None,last_ready=None,pending=0,training_examples=0,snapshot=None,dataset=None)
        if self.dataset.last:self.state.update(snapshot=self.dataset.last['snapshot'],dataset=self.dataset.last['stats'])
        self.seen=None;self.generation=None
        if autostart:self.thread=threading.Thread(target=self.loop,name='strata-background',daemon=True);self.thread.start()
    def notify(self,ids=()):
        with self.lock:
            for ident in ids:
                if ident in self.queue.tasks and ident not in self.queued and len(self.todo)<100:
                    self.todo.append(ident);self.queued.add(ident)
            self.state['pending']=len(self.todo)
        self.event.set()
    def status(self):
        with self.lock:return copy.deepcopy(self.state)
    def loop(self):
        while not self.stop.is_set():
            try:self.run_once()
            except Exception as error:
                with self.lock:self.state.update(error=str(error),building=False)
            self.event.wait(3);self.event.clear()
    def run_once(self):
        fingerprint=self.dataset.fingerprint()
        if fingerprint!=self.seen:
            with self.lock:self.state.update(building=True,error=None)
            manifest=self.dataset.build();self.media.feedback.stamp=None;self.media.feedback.refresh()
            self.generation=manifest['snapshot'];self.seen=fingerprint
            with self.lock:self.state.update(building=False,last_ready=time.time(),training_examples=len(self.media.feedback.refs),snapshot=manifest['snapshot'],dataset=manifest['stats'])
            with self.queue.db() as db:
                ids=[i for r in db.execute('SELECT tasks FROM leases') for i in json.loads(r[0])]
            self.notify(ids)
        # Work in bounded batches. A new save interrupts after this batch.
        for _ in range(7):
            with self.lock:
                if not self.todo:break
                ident=self.todo.popleft();self.queued.discard(ident);self.state['pending']=len(self.todo)
            if ident not in self.queue.tasks:continue
            with self.queue.lock:task=self.queue.task_payload(ident);revision=self.queue.document(task['document_id'])['revision']
            if task['segment'].get('approved'):continue
            bundle=self.media.hydrate(dict(sets=[dict(tasks=[task])]),refresh=False)
            task=bundle['sets'][0]['tasks'][0]
            # Images stay in the shared crop cache; only the proposal is stored here.
            value=dict(source_version=task['version'],document_revision=revision,generation=self.generation,segment=task['segment'],prediction=task['prediction'])
            with self.lock:self.cache[ident]=value
            write(self.root/(ident+'.json'),value)
        write(self.root/'status.json',self.status())
        if self.todo:self.event.set()
    def proposal(self,task):
        with self.lock:value=self.cache.get(task['id'])
        if value is None:
            path=self.root/(task['id']+'.json')
            if path.exists():value=read(path)
        revision=self.queue.document(task['document_id'])['revision']
        if value and value['source_version']==task['version'] and value['document_revision']==revision and value['generation']==self.generation:
            return copy.deepcopy(value)
        self.notify([task['id']]);return None
    def close(self):self.stop.set();self.event.set()
