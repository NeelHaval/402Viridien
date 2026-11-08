"""Stage only source files and declared unfinished development cases for boat."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent;BAKE=HERE.parent/'bakeoff';OUT=HERE/'data/jev-e2e-20261108'
JOB='jev-e2e-cpu-20261108T0645';REMOTE='/home/user/viridian402-jobs/'+JOB;PACK=OUT/'boat-packet'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    PACK.mkdir(parents=True,exist_ok=True);files={}
    def copy(src,dest):
        path=PACK/dest;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,path);files[dest]=sha(path)
    copy(HERE/'jev_boat_worker.py','worker.py')
    for name in ['scoreboard_v4.py','jev_sourcecards.py','grading_policy.py','legend_identity.py']:
        copy(BAKE/'strategies'/name,'source/strategies/'+name)
    for p in (BAKE/'strategies/scoreboard_v4_snapshot').glob('*'):
        if p.is_file():copy(p,'source/strategies/scoreboard_v4_snapshot/'+p.name)
    for name in ['io.py','score.py']:copy(BAKE/'harness'/name,'source/harness/'+name)
    for name in ['source/harness/__init__.py','source/strategies/__init__.py']:
        path=PACK/name;path.write_text('',encoding='utf-8');files[name]=sha(path)
    manifest=read(OUT/'manifest.json');done={r['case_id'] for r in read(OUT/'results.json')['records']}
    selected=[c for c in manifest['cases'] if c['case_id']=='ee53c8e06c27030e' and c['case_id'] not in done]
    if len(selected)!=1:raise ValueError('First remote case is already done or missing')
    for c in selected:
        c=dict(c,remote_image='/home/user/work/labeller-live/viridien/dataset/images/'+Path(c['image']).name,remote_annotation='/home/user/work/labeller-live/viridien/dataset/bbox_annotation/'+Path(c['annotation']).name)
        # Strip local truth paths from the inference packet; scoring uses a separate upload.
        c.pop('truth_sources',None);c.pop('truth_csv',None)
        p=PACK/'cases'/(c['case_id']+'.json');p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(c),encoding='utf-8')
    wrapper='''#!/bin/bash
set -u
umask 077
cd "$(dirname "$0")"
stage="$1"
case_id="$2"
printf 'queued\\n' > "$stage-$case_id.state"
(
  flock -w 1800 9 || exit 75
  printf 'started\\n' > "$stage-$case_id.state"
  timeout --signal=TERM --kill-after=10 900s env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$PWD/home" TMPDIR="$PWD/tmp" /usr/bin/python3 -X utf8 "$PWD/worker.py" "$stage" "$case_id"
  rc=$?
  printf '%s\\n' "$rc" > "$stage-$case_id.exit"
  if [ "$rc" = 0 ]; then printf 'completed\\n'; else printf 'failed\\n'; fi > "$stage-$case_id.state"
  exit "$rc"
) 9>/home/user/viridian402-jobs/v4-cpu-worker.lock
'''
    (PACK/'run_locked.sh').write_text(wrapper,encoding='utf-8',newline='\n');files['run_locked.sh']=sha(PACK/'run_locked.sh')
    (PACK/'source-manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    (OUT/'boat-location.json').write_text(json.dumps(dict(remote=REMOTE,packet=str(PACK.resolve()),first_case=selected[0]['case_id'],source_fingerprint=sha(PACK/'source-manifest.json')),indent=2),encoding='utf-8')
    print('PACKET',str(PACK),'files',len(files),'remote',REMOTE)


if __name__=='__main__':main()
