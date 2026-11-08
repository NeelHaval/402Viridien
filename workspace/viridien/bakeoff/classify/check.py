"""Input and scoring invariants on real TRUE patches; no paid calls."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from PIL import Image
from classify.data import datasets, prepare, stratified, label_key, OUT, vocab_names, dump, panel
from classify.bench import hybrid, summarize
from classify.api import APIs


def main():
    case=prepare(datasets(False)[0])
    assert len(case['patches'])==17
    assert len(case['vocab'])==10
    assert label_key('Sand and Sandstone')==label_key('Sandstone')
    width=case['info']['column_box'][2]-case['info']['column_box'][0]
    for p in case['patches']:
        assert p['rgb'].shape[1]==width
        assert 12<=len(p['rgb'])<=160
        assert p['top_y']<=p['patch_y0']<p['patch_y1']<=p['bottom_y']
        assert abs((p['patch_y0']+p['patch_y1'])-(p['top_y']+p['bottom_y']))<=1
    fake=[dict(top_y=i*20,bottom_y=i*20+20,lithology='rare' if i==0 else 'common') for i in range(100)]
    sampled=stratified(fake)
    assert len(sampled)==40 and any(r['lithology']=='rare' for r in sampled)
    fixture=OUT/'test-fixtures'/'vocab.json'
    dump(fixture,dict(entries=[dict(label='Rock A',box=[0,0,10,10]),dict(label='Rock B',box=[10,0,20,10])]))
    assert vocab_names(fixture)==['Rock A','Rock B']
    # Check ranking extraction from typed JEV probabilities without contacting an API.
    apis=APIs()
    p=np.arange(1,len(case['vocab'])+1,dtype=float);p/=p.sum()
    apis.request=lambda *args:(dict(response=dict(answers=dict(rock=dict(probabilities={str(i):float(v) for i,v in enumerate(p)}))),
                                   seconds=.1,calls=1,status='ok'),dict(cache_hit=False,key='fixture'))
    predictions,_=apis.jev(case,'jev_choice')
    assert all(np.allclose(v,p) for v in predictions)
    apis.request=lambda *args:(dict(response=None,seconds=.1,calls=1,error='fixture failure'),dict(cache_hit=False,key='fixture'))
    predictions,_=apis.jev(case,'jev_choice')
    assert all(v is None for v in predictions), 'API failures must never become fallback guesses'
    sizes=[]
    def terra_fixture(c,patches,name):
        sizes.append(len(patches))
        answers=[dict(patch=i+1,ranking=[dict(label=v,probability=float(p[j])) for j,v in enumerate(c['vocab'])])
                 for i in range(len(patches))]
        return dict(response=dict(predictions=answers),seconds=.1,calls=1,status='ok'),dict(cache_hit=False,key='fixture')
    apis.request=terra_fixture
    predictions,_=apis.terra(case)
    assert sizes==[8,8,1] and all(np.allclose(v,p) for v in predictions)
    im=panel(case,case['patches'][:1],'choice')
    assert max(im.size)<=1024
    results=OUT/'results.csv'
    if results.exists():
        df=pd.read_csv(results,keep_default_na=False,dtype={'source':str})
        assert not df.duplicated(['source','patch_id','classifier']).any()
        for _,r in df.iterrows():
            probs=json.loads(r.probabilities)
            if r.predicted:
                assert abs(sum(probs.values())-1)<1e-5
                assert r.predicted in probs
                assert 1<=int(float(r.rank_of_truth))<=int(r.n_labels)
                assert (int(float(r.rank_of_truth))==1)==bool(r.correct)
        # Changing CONSENSUS truths must not change the TRUE-fitted gate policy.
        records=df.to_dict('records')
        records=[r for r in records if r['classifier']!='hybrid']
        _,policy=hybrid(records)
        fake_consensus=[dict(r,source='CONSENSUS',patch_id='fixture_'+r['patch_id'],truth='irrelevant',correct=False)
                        for r in records if r['source']=='TRUE']
        mutated=records+fake_consensus
        _,other=hybrid(mutated)
        assert policy==other
    print('PASS: 17 central full-width patches; 10-way vocabulary; alias; stratified cap; vocab fallback; typed API ranking and failure handling; probability/rank consistency; no consensus leakage in hybrid tuning.')


if __name__=='__main__':main()
