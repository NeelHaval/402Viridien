#!/usr/bin/env python3
"""Allocate untouched pages to E/B/N, preserving any started pages and annotations.

Writes labelling_assignments.json and LABELLING_ASSIGNMENTS.md only.
Existing allocations are reused; --overwrite reallocates untouched pages explicitly.
"""
import argparse
import json
from pathlib import Path
import labelslib as L

ASSIGNMENTS = L.ROOT/'labelling_assignments.json'


def started(label):
    return bool(label.get('legend') or label.get('layers') or label.get('depth_calibration',{}).get('anchors'))


def allocate(stems,protected):
    initials=['E','B','N']
    remaining=[s for s in stems if s not in protected]
    quotas={a:len(remaining)//3+(i<len(remaining)%3) for i,a in enumerate(initials)}
    counts={a:0 for a in initials};work={a:0 for a in initials};assignments={}
    def height(stem):
        box=L.bbox(stem);return box[3]-box[1]
    for stem in sorted(remaining,key=lambda s:(-height(s),s)):
        owner=min((a for a in initials if counts[a]<quotas[a]),key=lambda a:(work[a],counts[a],initials.index(a)))
        assignments[stem]=owner;counts[owner]+=1;work[owner]+=height(stem)
    return {'protected_stems':list(protected),'assignments':dict(sorted(assignments.items()))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args(argv)
    if ASSIGNMENTS.exists() and not args.overwrite:
        print('Existing allocation kept. Use --overwrite only to reallocate.');return
    stems=L.label_stems();protected=[]
    for stem in stems:
        path=L.label_path(stem)
        if path.exists() and started(json.loads(path.read_text())):protected.append(stem)
    result=allocate(stems,protected)
    ASSIGNMENTS.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Labelling assignments','', 'Started pages are reserved for the person already working on them. No label files are modified.','', '## In progress — do not reassign','']
    lines += [f'- `{s}`' for s in protected] or ['- None']
    for initial in ['E','B','N']:
        pages=[s for s,a in result['assignments'].items() if a==initial]
        lines+=['',f'## {initial} — {len(pages)} pages','']+[f'- `{s}` ({stems[s]})' for s in pages]
    lines+=['','## Working together','',
      'Choose your initial in the tool to filter your assigned pages. Each page has one owner. Teammates can start with `http://127.0.0.1:8765/?assignee=E` (or B / N) on their own machine.',
      'For separate computers, pull this branch and follow README.md. Prep restores missing dataset inputs and builds the local cache. Run the server locally, then return only the assigned `labels/<stem>.json` files. Preserve the current in-progress page.',
      'The localhost address works on the machine running the server. It is not a remotely shared website.','']
    (L.ROOT/'LABELLING_ASSIGNMENTS.md').write_text('\n'.join(lines))
    print('Reserved:',len(protected));print({a:list(result['assignments'].values()).count(a) for a in ['E','B','N']})

if __name__=='__main__':main()
