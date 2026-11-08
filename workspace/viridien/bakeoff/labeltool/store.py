"""Append-only decisions, durable undo, and last-decision-wins CSV materialization."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid
from labeltool.common import DATA, HUMAN, FIELDS, read_json, overlay, normalize


class Store:
    def __init__(self, directory=HUMAN, metadata=DATA/'logs'):
        self.directory = Path(directory)
        self.metadata = Path(metadata)
        self.path = self.directory/'labels.jsonl'
        self.lock = threading.RLock()
        self.events = []
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                if line.strip(): self.events.append(json.loads(line))
        # Rebuild CSVs on startup, recovering a crash between journal and CSV.
        for name in {e['log'] for e in self.events if e.get('action')=='label'}:
            self.rebuild(name)

    def active(self):
        return [e for e in self.active_decisions() if e['action']=='label']

    def active_decisions(self):
        undone = {e['target'] for e in self.events if e.get('action')=='undo'}
        return [e for e in self.events if e.get('action') in ('label','wrong') and e['event_id'] not in undone]

    def completed(self):
        return {e['item_id'] for e in self.active_decisions() if not e.get('window_id')}

    def completed_windows(self):
        return {e['window_id'] for e in self.active() if e.get('window_id')}

    def narrowed(self):
        return [e for e in self.active() if e.get('candidates')]

    def append(self, *events):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open('a',encoding='utf-8') as f:
            f.write(''.join(json.dumps(event,ensure_ascii=False)+'\n' for event in events))
            f.flush()
            os.fsync(f.fileno())
        self.events.extend(events)

    def save(self, item, payload):
        with self.lock:
            request_id = str(payload.get('request_id') or uuid.uuid4())
            previous = next((e for e in self.events if e.get('event_id')==request_id),None)
            if previous: return previous
            top, bottom = int(payload.get('top_y',item['top_y'])), int(payload.get('bottom_y',item['bottom_y']))
            label = str(payload.get('label','')).strip()
            if not label or len(label)>160: raise ValueError('Choose a label (1–160 characters)')
            meta = read_json(self.metadata/(item['log']+'.json'),{})
            box = meta.get('box',[0,0,0,2**31])
            if not box[1]<=top<bottom<=box[3]: raise ValueError('Boundary must be inside the column; bottom must exceed top')
            if item.get('truth_only'): raise ValueError('Gold is for agreement statistics only')
            mode = str(payload.get('mode','uncertain'))
            if mode not in ('uncertain','spot','disputed','walk','window','strip'): raise ValueError('Unknown mode')
            event = dict(action='label', event_id=request_id, item_id=item['id'], log=item['log'],
                top_y=top, bottom_y=bottom, label=label,
                suggestion_shown=str(payload.get('suggestion_shown',item.get('suggestion','Unknown'))),
                jev2=str(payload.get('jev2') or '')[:160],
                boundary_edited=(top!=item['top_y'] or bottom!=item['bottom_y']), mode=mode,
                tier=item['tier'], ms_taken=max(0,min(86400000,int(payload.get('ms_taken',0)))),
                timestamp=datetime.now(timezone.utc).isoformat())
            self.append(event)
            self.rebuild(item['log'])
            return event

    def save_batch(self, window, intervals, payload, machine=()):
        """One window's painted intervals as a single batch; unpainted rows are not recorded.

        `machine` holds the window's machine segments (top_y, bottom_y, label, tier) shown.
        """
        with self.lock:
            batch = str(payload.get('request_id') or uuid.uuid4())
            previous = [e for e in self.events if e.get('batch_id')==batch and e.get('action')=='label']
            if previous: return previous
            if not intervals: raise ValueError('Paint at least one interval before saving')
            if len(intervals)>200: raise ValueError('Too many intervals in one window')
            rows=[]
            for raw in intervals:
                top,bottom=int(raw['top_y']),int(raw['bottom_y'])
                label=str(raw.get('label') or '').strip()
                candidates=[str(c).strip() for c in raw.get('candidates') or [] if str(c).strip()]
                candidates=list(dict.fromkeys(candidates))
                if candidates and label not in candidates: candidates.insert(0,label)
                if not label or len(label)>160: raise ValueError('Every painted interval needs a label (1–160 characters)')
                if len(candidates)==1: candidates=[]
                if len(candidates)>3: raise ValueError('Unsure answers name 2–3 labels')
                if not window['top_y']<=top<bottom<=window['bottom_y']:
                    raise ValueError('Intervals must lie inside the window; bottom must exceed top')
                rows.append((top,bottom,label,candidates,bool(raw.get('adopted'))))
            rows.sort()
            if any(a[1]>b[0] for a,b in zip(rows,rows[1:])): raise ValueError('Painted intervals overlap')
            if window.get('truth_only'): raise ValueError('Gold is for agreement statistics only')
            meta=read_json(self.metadata/(window['log']+'.json'),{})
            box=meta.get('box',[0,0,0,2**31])
            if not (box[1]<=rows[0][0] and rows[-1][1]<=box[3]): raise ValueError('Boundary must be inside the column')
            now=datetime.now(timezone.utc).isoformat()
            ms=max(0,min(86400000,int(payload.get('ms_taken',0))))
            edges={e for m in machine for e in (m['top_y'],m['bottom_y'])}
            events=[]
            for index,(top,bottom,label,candidates,adopted) in enumerate(rows):
                overlap=lambda m:max(0,min(bottom,m['bottom_y'])-max(top,m['top_y']))
                shown=max(machine,key=overlap,default=None)
                event=dict(action='label',event_id=f'{batch}:{index}',batch_id=batch,window_id=window['id'],
                    item_id=f"{window['id']}:{top}:{bottom}",log=window['log'],top_y=top,bottom_y=bottom,label=label,
                    suggestion_shown=shown['label'] if shown and overlap(shown) else 'Unknown',
                    boundary_edited=not (top in edges and bottom in edges),adopted=adopted,mode='window',
                    tier=shown['tier'] if shown and overlap(shown) else 'low',ms_taken=ms,timestamp=now)
                if candidates: event['candidates']=candidates
                events.append(event)
            self.append(*events)
            self.rebuild(window['log'])
            return events

    def undo(self):
        """Undo the latest active decision; a window batch is undone as a whole."""
        with self.lock:
            active = self.active_decisions()
            if not active: return None
            decision = active[-1]
            batch = decision.get('batch_id')
            targets = [e for e in active if e.get('batch_id')==batch] if batch else [decision]
            now = datetime.now(timezone.utc).isoformat()
            self.append(*[dict(action='undo',event_id=str(uuid.uuid4()),target=e['event_id'],log=e['log'],
                               timestamp=now) for e in targets])
            self.rebuild(decision['log'])
            return dict(decision,batch=targets) if batch else decision

    def rows(self, name):
        rows=[]
        for event in self.active():
            if event['log']==name:
                row=dict(top_y=event['top_y'],bottom_y=event['bottom_y'],lithology=event['label'])
                if event.get('candidates'): row['candidates']=list(event['candidates'])
                rows=overlay(rows,row)
        return rows

    def rebuild(self, name):
        self.directory.mkdir(parents=True,exist_ok=True)
        # Narrowed ("one of these") rows still override older decisions but are not ground truth.
        rows=[r for r in self.rows(name) if not r.get('candidates')]
        calib=read_json(self.metadata/(name+'.json'),{}).get('calibration')
        clusters={}
        for row in rows:
            key=normalize(row['lithology'])
            row['cluster_id']=clusters.setdefault(key,len(clusters))
            for edge in ('top','bottom'):
                row[edge+'_depth']=round(calib['a']*row[edge+'_y']+calib['b'],1) if calib else ''
        path=self.directory/(name+'.csv')
        temp=path.with_suffix('.csv.tmp')
        with temp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            writer.writeheader();writer.writerows(rows)
        temp.replace(path)
        return rows
