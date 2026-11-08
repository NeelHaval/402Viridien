"""Audit native-row coverage and every high-confidence decision on all 51 logs."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labeltool.common import DATA, GOLD, read_json, write_json, normalize
from harness import io


def verify():
    rows=[json.loads(line) for line in (DATA/'queue.jsonl').read_text(encoding='utf-8').splitlines() if line]
    assert len({r['id'] for r in rows})==len(rows)
    names=io.list_logs()
    assert len(names)==51
    assert set(names)=={r['log'] for r in rows}
    sources=set()
    for name in names:
        meta=read_json(DATA/'logs'/(name+'.json'))
        group=sorted([r for r in rows if r['log']==name],key=lambda r:r['top_y'])
        assert group[0]['top_y']==meta['box'][1]
        assert group[-1]['bottom_y']==meta['box'][3]
        assert all(a['bottom_y']==b['top_y'] for a,b in zip(group,group[1:]))
        assert all(r['bottom_y']-r['top_y']>=12 for r in group)
        assert all(r['truth_only']==(name==GOLD) for r in group)
        assert meta['legend_parser']=='harness.legend_v2'
        assert meta['calibration'] and not meta['errors']
        for row in group:
            sources.update(s['source'] for s in row['suggestions'])
            if row['tier']=='high':
                assert row['d1']<=12 and row['margin']>=15
                labels={normalize(s['label']) for s in row['suggestions'] if not s.get('truth')}
                assert len(labels)==1
                assert not row['disputed']
    result=dict(status='passed',logs=51,intervals=len(rows),reviewable=sum(not r['truth_only'] for r in rows),
                truth_only=sum(r['truth_only'] for r in rows),source_names=sorted(sources),
                coverage='Full native boxes; no gaps, overlaps or <12-row slivers',calibrated=51,legend_v2=51)
    write_json(DATA/'coverage.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':verify()
