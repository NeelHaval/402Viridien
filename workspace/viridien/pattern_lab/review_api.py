"""HTTP endpoints for shared reviewers. Only the host can issue reviewer keys."""
import json
from urllib.parse import urlparse
from review_queue import ReviewQueue,version
from review_media import ReviewMedia
from legend_repairs import install_repairs
from background import Background
from pathlib import Path
import time

class ReviewAPI:
    def __init__(self,store,images,share_url):
        install_repairs(store)
        self.queue=ReviewQueue(store);self.media=ReviewMedia(self.queue,images);self.share_url=share_url.rstrip('/')
        self.worker=Background(self.queue,self.media)
    def local(self,handler):
        return handler.client_address[0] in ('127.0.0.1','::1',urlparse(self.share_url).hostname)
    def token(self,handler):
        value=handler.headers.get('Authorization','')
        if not value.startswith('Bearer '):raise PermissionError('Open a reviewer link or a portable reviewer file.')
        token=value[7:];self.queue.session(token);return token
    def get(self,handler,path,query):
        if path=='/api/review/info':
            if not self.local(handler):raise PermissionError('Open this page on the host to share work.')
            return handler.respond(dict(server_url=self.share_url,summary=self.queue.summary()))
        if path=='/api/review/bundle':
            bundle=self.queue.bundle(self.token(handler),int(query.get('sets',3)),query.get('exclude','').split(',')[:100],query.get('focus'))
            self.worker.notify([t['id'] for g in bundle['sets'] for t in g['tasks']])
            known=dict(v.split(':',1) for v in query.get('known','').split(',')[:100] if ':' in v)
            for group in bundle['sets']:group['tasks']=[t for t in group['tasks'] if known.get(t['id'])!=self.media.signature(t)]
            return handler.respond(self.media.hydrate(bundle,refresh=False))
        if path=='/api/review/state':
            result=self.queue.sync(self.token(handler),query.get('tasks','').split(','));result['background']=self.worker.status();result['reviewers']=self.queue.upload_status(self.token(handler),self.local(handler))
            return handler.respond(result)
        if path=='/api/review/session':
            s=self.queue.session(self.token(handler));return handler.respond(dict(session={k:s[k] for k in ('id','name','token')},host=self.local(handler)))
        if path=='/api/review/dataset':
            self.token(handler)
            if not self.local(handler):raise PermissionError('Export the full normalized dataset on the host.')
            state=self.worker.status()
            if not state['snapshot'] or not (self.queue.store.data/'exports'/('normalized-'+state['snapshot']+'.zip')).exists():return handler.respond(dict(ready=False,background=state))
            return handler.respond(dict(ready=True,url='/download/normalized-'+state['snapshot']+'.zip',background=state))
        if path=='/api/review/improve':
            tasks=self.queue.sync(self.token(handler),query.get('tasks','').split(','))['tasks'][:7]
            bundle=self.media.hydrate(dict(sets=[dict(tasks=tasks)]),refresh=False)
            return handler.respond(dict(tasks=bundle['sets'][0]['tasks']))
        raise ValueError('Unknown review route.')
    def post(self,handler,path,data):
        if path in ('/api/review/start','/api/review/portable'):
            if not self.local(handler):raise PermissionError('Create reviewer files on the host computer.')
            session=self.queue.create(data.get('name','Shared reviewer'))
            if path.endswith('/start'):return handler.respond(session)
            return self.portable(handler,session,int(data.get('sets',5)))
        token=self.token(handler)
        if path=='/api/review/handoff':
            s=self.queue.session(token)
            return handler.respond(dict(url=self.share_url+'/review#review='+token+('&task='+str(data['task_id']) if data.get('task_id') in self.queue.tasks else ''),name=s['name']))
        if path=='/api/review/file':
            s=self.queue.session(token);return self.portable(handler,{k:s[k] for k in ('id','name','token')},int(data.get('sets',5)),data.get('task_id'))
        if path=='/api/review/save':
            try:result=self.queue.apply(token,data)
            except ValueError as error:
                task=self.queue.tasks.get(data.get('task_id'));current=self.queue.current(task) if task else None
                result=dict(ok=False,validation=True,task_id=data.get('task_id'),error=str(error),version=version(current),segment=current or (task['segment'] if task else None))
            if result.get('ok'):self.worker.notify([data['task_id']])
            result['saved_at']=time.time();return handler.respond(result)
        raise ValueError('Unknown review route.')
    def portable(self,handler,session,sets,focus=None):
            bundle=self.media.hydrate(self.queue.bundle(session['token'],sets,focus=focus),refresh=False)
            boot=dict(server=self.share_url,session=session,bundle=bundle,portable=True,task_id=focus)
            body=self.media.html(boot).encode('utf-8')
            file=self.queue.store.data/'exports'/('reviewer-'+session['id']+'.html')
            file.parent.mkdir(exist_ok=True);file.write_bytes(body)
            return handler.respond(body,content_type='text/html; charset=utf-8',filename=file.name)
