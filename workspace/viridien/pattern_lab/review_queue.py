"""Shared pattern queue, leases, idempotent saves, and per-box conflicts."""
from __future__ import annotations
import bisect, copy, hashlib, json, secrets, sqlite3, threading, time
from contextlib import contextmanager
from pathlib import Path
from model import BAKEOFF, Store, read, write

TARGET=5

def version(segment):
    if segment is None:return 'new'
    value={k:v for k,v in segment.items() if not k.startswith('_')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]

def overlaps(a,b):
    return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])

def prepare_index(store):
    path=store.data/'pattern-queue.json'
    if path.exists():return read(path)
    source=read(BAKEOFF/'analysis/families_all/families.json')
    docs={c['name']:store.get(c['id']) for c in store.catalog}
    groups=[];references=[]
    for family in source['families']:
        key='family-'+str(family['id']);candidates=[]
        for name,y in family['members']:
            doc=docs.get(name)
            if not doc:continue
            col=doc['boxes']['interpreted_lithology_column'];top=max(col[1],int(y));bottom=min(col[3],int(y)+48)
            if bottom-top<8:continue
            parent=next((s for s in doc['segments'] if s['box'][1]<=top<s['box'][3]),None)
            ident='queue-'+hashlib.sha256(f'{key}:{doc["id"]}:{y}'.encode()).hexdigest()[:20]
            segment=dict(id=ident,box=[col[0],top,col[2],bottom],legend_ids=parent['legend_ids'][:] if parent else [],status='suggested',approved=False,example=False,comments=[],provenance='column pattern cluster; human review needed',suggestion=parent.get('suggestion','') if parent else '',source_segment_id=parent['id'] if parent else None,queue_pattern_key=key)
            candidates.append(dict(id=ident,document_id=doc['id'],segment=segment))
        if candidates:
            groups.append(dict(key=key,title=f'Pattern set {family["id"]+1}',names=family.get('legend_lookalikes',[])[:4],candidates=candidates,document_count=len({c['document_id'] for c in candidates})))
    # Existing clean source examples count towards the same five-example limit.
    for doc in docs.values():
        for segment in doc['segments']:
            if not segment.get('example') or not segment.get('approved'):continue
            possibilities=[(abs((segment['box'][1]+segment['box'][3])/2-(c['segment']['box'][1]+24)),g['key']) for g in groups for c in g['candidates'] if c['document_id']==doc['id'] and overlaps(segment['box'],c['segment']['box'])]
            if possibilities:references.append(dict(document_id=doc['id'],segment_id=segment['id'],key=min(possibilities)[1]))
    result=dict(schema=1,target=TARGET,groups=groups,references=references)
    write(path,result);return result

class ReviewQueue:
    def __init__(self,store,index=None):
        self.store=store;self.lock=store.lock;self.index=index or prepare_index(store)
        self.groups={g['key']:g for g in self.index['groups']}
        self.tasks={c['id']:dict(c,key=g['key']) for g in self.groups.values() for c in g['candidates']}
        self.cache={};self.db_path=store.data/'shared-reviews.sqlite3'
        with self.db() as db:
            db.executescript('CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,token TEXT UNIQUE,name TEXT,created REAL); CREATE TABLE IF NOT EXISTS leases(pattern TEXT PRIMARY KEY,session TEXT,expires REAL,tasks TEXT); CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY,session TEXT,response TEXT); CREATE TABLE IF NOT EXISTS skips(session TEXT,task TEXT,PRIMARY KEY(session,task));')
            db.execute('CREATE TABLE IF NOT EXISTS reviewer_status(id TEXT PRIMARY KEY,last_seen REAL,last_save REAL,saves INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS assignments(session TEXT,task TEXT,pattern TEXT,PRIMARY KEY(session,task))')
            for lease in db.execute('SELECT * FROM leases').fetchall():
                db.executemany('INSERT OR IGNORE INTO assignments VALUES(?,?,?)',[(lease['session'],i,lease['pattern']) for i in json.loads(lease['tasks'])])

    @contextmanager
    def db(self):
        connection=sqlite3.connect(self.db_path,timeout=30)
        connection.row_factory=sqlite3.Row
        try:
            yield connection;connection.commit()
        except BaseException:
            connection.rollback();raise
        finally:connection.close()

    def document(self,ident):
        path=self.store.data/'reviews'/(ident+'.json')
        if not path.exists():path=self.store.data/'seeds'/(ident+'.json')
        stamp=path.stat().st_mtime_ns
        if ident not in self.cache or self.cache[ident][0]!=stamp:self.cache[ident]=(stamp,self.store.get(ident))
        return self.cache[ident][1]

    def session(self,token,db=None):
        if db is None:
            with self.db() as db:return self.session(token,db)
        row=db.execute('SELECT * FROM sessions WHERE token=?',(token,)).fetchone()
        if not row:raise PermissionError('This reviewer key is not valid.')
        return dict(row)

    def create(self,name='Reviewer'):
        ident=secrets.token_hex(12);token=secrets.token_urlsafe(32)
        with self.lock,self.db() as db:db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(ident,token,str(name)[:80],time.time()))
        return dict(id=ident,token=token,name=str(name)[:80])

    def current(self,task):
        doc=self.document(task['document_id'])
        return next((s for s in doc['segments'] if s['id']==task['id']),None)

    def counts(self):
        counts={k:0 for k in self.groups};seen=set()
        for c in self.store.catalog:
            for s in self.document(c['id'])['segments']:
                key=s.get('queue_pattern_key')
                if key in counts and s.get('approved') and s.get('example'):
                    counts[key]+=1;seen.add((c['id'],s['id']))
        for r in self.index.get('references',[]):
            if (r['document_id'],r['segment_id']) in seen:continue
            s=next((s for s in self.document(r['document_id'])['segments'] if s['id']==r['segment_id']),None)
            if s and s.get('approved') and s.get('example'):counts[r['key']]+=1
        return counts

    def available(self,task,skips):
        if task['id'] in skips:return False
        current=self.current(task)
        if current and current['status'] in ('approved','rejected'):return False
        box=(current or task['segment'])['box']
        return not any(s['id']!=task['id'] and s.get('approved') and overlaps(s['box'],box) for s in self.document(task['document_id'])['segments'])

    def spread(self,group,amount,skips):
        pools={}
        for task in group['candidates']:
            if self.available(task,skips):pools.setdefault(task['document_id'],[]).append(task['id'])
        result=[]
        # One example per document in each round. Nearby chunks do not fill a set.
        while pools and len(result)<amount:
            for ident in list(pools):
                if len(result)>=amount:break
                result.append(pools[ident].pop(0))
                if not pools[ident]:del pools[ident]
        return result

    def bundle(self,token,sets=3,exclude=(),focus=None):
        with self.lock,self.db() as db:
            session=self.session(token,db);now=time.time();counts=self.counts()
            skips={r[0] for r in db.execute('SELECT task FROM skips WHERE session=?',(session['id'],))}
            db.execute('DELETE FROM leases WHERE expires<?',(now,))
            leases={r['pattern']:dict(r) for r in db.execute('SELECT * FROM leases')}
            chosen=[];preferred=(self.tasks.get(focus) or {}).get('key')
            order=sorted(self.groups.values(),key=lambda g:(-1 if g['key']==preferred and (g['key'] not in leases or leases[g['key']]['session']==session['id']) else 0 if g['key'] in leases and leases[g['key']]['session']==session['id'] else 1,counts[g['key']],-g['document_count'],-len(g['candidates'])))
            for group in order:
                key=group['key'];lease=leases.get(key)
                if key in exclude or counts[key]>=TARGET or (lease and lease['session']!=session['id']):continue
                ids=json.loads(lease['tasks']) if lease else []
                remaining=[i for i in ids if i not in skips and self.available(self.tasks[i],skips)]
                extra=self.spread(group,max(0,TARGET-counts[key]+2-len(remaining)),skips|set(ids))
                ids=list(dict.fromkeys(ids+extra))
                if not remaining and not extra:continue
                db.execute('INSERT OR REPLACE INTO leases VALUES(?,?,?,?)',(key,session['id'],now+7*86400,json.dumps(ids)))
                db.executemany('INSERT OR IGNORE INTO assignments VALUES(?,?,?)',[(session['id'],i,key) for i in ids])
                chosen.append(dict(key=key,title=group['title'],names=group['names'],accepted=counts[key],target=TARGET,document_count=group['document_count'],tasks=[self.task_payload(i) for i in ids if i not in skips and (self.available(self.tasks[i],skips) or (self.current(self.tasks[i]) or {}).get('approved'))]))
                if len(chosen)>=max(1,min(10,sets)):break
            return dict(session={k:session[k] for k in ('id','name','token')},sets=chosen,summary=self.summary(counts))

    def task_payload(self,ident):
        task=self.tasks[ident];doc=self.document(task['document_id']);current=self.current(task)
        segment=copy.deepcopy(current or task['segment']);col=doc['boxes']['interpreted_lithology_column']
        rows=max(96,round((col[2]-col[0])*360/220));mid=(segment['box'][1]+segment['box'][3])//2
        top=max(col[1],min(mid-rows//2,col[3]-rows));bottom=min(col[3],top+rows)
        return dict(id=ident,document_id=doc['id'],document_name=doc['name'],pattern_key=task['key'],segment=segment,version=version(current),legend=copy.deepcopy(doc['legend']),context_box=[col[0],top,col[2],bottom])

    def summary(self,counts=None):
        counts=counts or self.counts()
        return dict(pattern_sets=len(counts),complete=sum(n>=TARGET for n in counts.values()),accepted=sum(min(n,TARGET) for n in counts.values()),target=TARGET)

    def sync(self,token,ids):
        with self.lock:
            self.session(token);counts=self.counts()
            return dict(summary=self.summary(counts),counts=counts,tasks=[self.task_payload(i) for i in ids[:100] if i in self.tasks])

    def upload_status(self,token,all_reviewers=False):
        with self.lock,self.db() as db:
            session=self.session(token,db)
            db.execute('INSERT INTO reviewer_status(id,last_seen,saves) VALUES(?,?,0) ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen',(session['id'],time.time()))
            return [dict(r) for r in db.execute('SELECT s.name,r.last_seen,r.last_save,r.saves FROM sessions s LEFT JOIN reviewer_status r ON s.id=r.id WHERE ? OR s.id=?',(int(all_reviewers),session['id']))]

    def apply(self,token,operation):
        with self.lock,self.db() as db:
            session=self.session(token,db);op_id=operation.get('op_id');ident=operation.get('task_id')
            if not isinstance(op_id,str) or not 8<=len(op_id)<=100:raise ValueError('Invalid save operation.')
            previous=db.execute('SELECT session,response FROM operations WHERE id=?',(op_id,)).fetchone()
            if previous:
                if previous['session']!=session['id']:raise PermissionError('Save operation belongs to another reviewer.')
                return json.loads(previous['response'])
            if ident not in self.tasks:raise ValueError('Unknown review box.')
            task=self.tasks[ident];key=task['key'];current=self.current(task);action=operation.get('action')
            lease=db.execute('SELECT * FROM leases WHERE pattern=?',(key,)).fetchone()
            assigned=db.execute('SELECT 1 FROM assignments WHERE session=? AND task=?',(session['id'],ident)).fetchone()
            if not assigned:raise PermissionError('This box was not assigned to this reviewer.')
            if lease and lease['session']!=session['id']:
                return dict(ok=False,conflict=True,reassigned=True,task_id=ident,error='This set was assigned to another reviewer while this device was offline. Your draft is kept.',version=version(current),segment=current or task['segment'])
            if not lease:
                db.execute('INSERT INTO leases VALUES(?,?,?,?)',(key,session['id'],time.time()+7*86400,json.dumps([ident])))
            if current and current.get('_last_op')==op_id:
                result=dict(ok=True,op_id=op_id,task_id=ident,version=version(current),segment=current)
            elif operation.get('version')!=version(current):
                return dict(ok=False,conflict=True,task_id=ident,error='This box changed on another machine. Your draft is kept.',version=version(current),segment=current or task['segment'])
            elif action=='skip':
                db.execute('INSERT OR IGNORE INTO skips VALUES(?,?)',(session['id'],ident));result=dict(ok=True,op_id=op_id,task_id=ident,version=version(current),skipped=True)
            else:
                segment=copy.deepcopy(operation.get('segment',{}))
                if segment.get('id')!=ident:raise ValueError('Review box ID does not match.')
                if action not in ('draft','accept','reject'):raise ValueError('Invalid review action.')
                if current is None and not self.available(task,set()):return dict(ok=False,conflict=True,task_id=ident,error='This area already has accepted labels. Your draft is kept.',version='new',segment=task['segment'])
                if action=='accept':
                    retired={e['id'] for e in self.document(task['document_id'])['legend'] if e.get('deprecated')}
                    if retired.intersection(segment.get('legend_ids',[])):raise ValueError('Choose a separate corrected legend entry. The old merged rows cannot label a new example.')
                    if not (current or {}).get('approved') and self.counts()[key]>=TARGET:return dict(ok=False,quota=True,task_id=ident,error='This pattern already has five accepted examples. Your draft is kept.')
                    segment.update(status='approved',approved=True,example=True)
                elif action=='reject':segment.update(status='rejected',approved=False,example=False)
                else:segment.update(status='suggested',approved=False,example=False)
                segment.update(queue_pattern_key=key,_last_op=op_id,_reviewer=session['name'])
                doc=self.store.get(task['document_id']);self.store.save(doc['id'],doc['revision'],segment=segment)
                self.cache.pop(doc['id'],None);result=dict(ok=True,op_id=op_id,task_id=ident,version=version(segment),segment=segment)
            result['saved_at']=time.time()
            db.execute('INSERT INTO reviewer_status(id,last_seen,last_save,saves) VALUES(?,?,?,1) ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen,last_save=excluded.last_save,saves=saves+1',(session['id'],result['saved_at'],result['saved_at']))
            db.execute('INSERT INTO operations VALUES(?,?,?)',(op_id,session['id'],json.dumps(result)))
            db.execute('UPDATE leases SET expires=? WHERE pattern=?',(time.time()+7*86400,key))
            return result
