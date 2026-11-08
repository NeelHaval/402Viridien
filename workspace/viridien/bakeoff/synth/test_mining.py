"""pytest tests use only local fixture images and an injected JEV client."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import threading
import time
import uuid
from collections import Counter, defaultdict
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import numpy as np
import pytest
from PIL import Image
from synth.common import HOME, read_json, write_json, relative, local_file
from synth.patterns import build_catalog, pattern_id
from synth.mine import (diverse, crop_windows, quality, repeat_height, Embedder,
                        Verifier, mine, source_intervals, verification_panel)
from synth.colour import profile, colour_gate, contrastive, PaletteMatcher
from synth.sheets import make_sheets
from synth.review import ReviewStore, handler_for


@pytest.fixture
def tmp_path():
    # pytest's Windows mkdir(mode=0o700) creates inaccessible ACLs in this
    # sandbox. Use ordinary directories, entirely under synth/, instead.
    path = HOME / 'cache' / 'test-cases' / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def fixture_data(tmp_path):
    data = tmp_path / 'data'
    swatch = tmp_path / 'swatch.png'
    rgb = np.full((48, 24, 3), 240, np.uint8)
    rgb[::6, ::3] = (30, 60, 90)
    Image.fromarray(rgb).save(swatch)
    p = dict(pid='gulf-123456/shale', label='Shale', family='Gulf 3-panel', legend_type='gulf-123456',
             logs=['a', 'b', 'c', 'd'], swatches=[dict(file=relative(swatch), log='a')], n_target=20, share=.1,
             candidate_count=4, status='mined')
    cs = []
    for i in range(4):
        path = tmp_path / f'crop-{i}.png'
        Image.fromarray(np.roll(rgb, i, axis=0)).save(path)
        cs.append(dict(file=relative(path), log=chr(97+i), top_y=108, bottom_y=156,
                       col_w=24, h=48, source='gold', score=1-i/10))
    entry = make_sheets(p, cs, data / 'sheets')
    write_json(data / 'catalog.json', {'patterns': [p]})
    write_json(data / 'sheets' / 'index.json', {p['pid']: entry})
    write_json(data / 'selected.json', {p['pid']: dict(picks=[1, 3], notes='Mock reviewer: clear hatch.', issues=['Check example 3'])})
    return data, p, cs


def test_catalogue_ids_are_stable():
    sw = {'MC765_A': [dict(label='Shale', file='a.png'), dict(label='Sand and Sandstone', file='b.png')],
          'MC765_B': [dict(label='SANDSTONE', file='b.png'), dict(label='shale', file='a.png')]}
    mix = [dict(log=name, col_h=100, share={'Shale': .5}) for name in sw]
    a = build_catalog(mix, sw)
    b = build_catalog(list(reversed(mix)), {k: list(reversed(v)) for k, v in sw.items()})
    assert a == b
    assert len(a['legend_types']) == 1 and len(a['patterns']) == 2
    expected = 'mc765-colour-grid-' + hashlib.sha1(b'sandstone|shale').hexdigest()[:6] + '/shale'
    assert pattern_id('MC765 colour grid', ['Shale', 'Sand and Sandstone'], 'SHALE') == expected
    assert all(p['logs'] == ['MC765_A', 'MC765_B'] for p in a['patterns'])
    assert sum(p['rows'] for p in a['patterns']) == 100
    assert len(a['no_candidates']) == 2


def candidates_by_counts(counts):
    basis = np.eye(sum(counts.values()), dtype=np.float32)
    out = []
    for log, count in counts.items():
        for j in range(count):
            out.append(dict(log=log, top_y=j*100, score=1-len(out)/1000, _embedding=basis[len(out)]))
    return out


@pytest.mark.parametrize('counts', [dict(a=40,b=12,c=12,d=12), dict(a=30,b=1,c=1,d=1), dict(a=10,b=10,c=10), dict(a=12,b=12,c=12,d=12,e=12)])
def test_strict_candidate_diversity(counts):
    result = diverse(candidates_by_counts(counts))
    assert len(result) <= 40
    if len(counts) < 4:
        assert result
        assert max(Counter(c['log'] for c in result).values()) <= -(-len(result) // len(counts))
    else:
        assert result
        assert max(Counter(c['log'] for c in result).values()) <= len(result)*.25


def test_cosine_dedupe_across_logs_and_final_30():
    cs = candidates_by_counts(dict(a=15,b=15,c=15,d=15,e=15))
    cs.append(dict(cs[0], log='other', top_y=900))
    near = cs[1]['_embedding'].copy(); near[2] = .1; near /= np.linalg.norm(near)
    cs.append(dict(cs[1], log='other', top_y=1000, _embedding=near))
    result = diverse(cs, limit=30)
    assert len(result) == 30
    vec = np.stack([c['_embedding'] for c in result])
    assert (vec @ vec.T - np.eye(len(vec))).max() <= .97
    assert max(Counter(c['log'] for c in result).values()) <= len(result)*.25


def test_crop_never_crosses_interval_boundary():
    rng = np.random.default_rng(9)
    for _ in range(150):
        top = int(rng.integers(100, 10000)); bottom = top+int(rng.integers(1, 1200))
        height = max(48, 3*int(rng.integers(1, 150)))
        for a, b in crop_windows(top, bottom, height, 33):
            assert top+8 <= a < b <= bottom-8
            assert b-a == height
    assert crop_windows(100, 163, 48, 8) == []
    assert crop_windows(100, 164, 48, 8) == [(108,156)]


def test_blank_grid_and_native_repeat_detection():
    blank = np.full((96,60,3),255,np.uint8)
    assert not quality(blank)
    rule = blank.copy(); rule[40:42] = 0
    assert not quality(rule)
    pattern = blank.copy(); pattern[::12, 8:52:4] = 0
    assert quality(pattern)
    assert repeat_height(pattern) == 12


def test_sheet_numbering_matches_index(fixture_data):
    data, p, cs = fixture_data
    index = read_json(data/'sheets'/'index.json')[p['pid']]
    assert list(index['candidates']) == ['1','2','3','4']
    for n, c in index['candidates'].items():
        assert c['file'] == cs[int(n)-1]['file']
        with Image.open(local_file(c['sheet'])) as sheet, Image.open(local_file(c['file'])) as crop:
            assert sheet.width <= 1600
            x,y = c['display_xy']; w,h = c['display_size']
            expected = crop.resize((w,h), Image.Resampling.NEAREST)
            assert np.array_equal(np.asarray(sheet.crop((x,y,x+w,y+h))), np.asarray(expected))
            assert abs(w/h-c['col_w']/c['h']) < .02


def test_two_sheets_preserve_global_numbering(fixture_data):
    data,p,cs = fixture_data
    long = [dict(cs[i%4], h=500, col_w=24) for i in range(30)]
    index = make_sheets(p,long,data/'two-sheets',max_width=800)
    assert len(index['sheets']) == 2
    assert [c['number'] for c in index['candidates'].values()] == list(range(1,31))
    assert len({c['display_scale'] for c in index['candidates'].values()}) == 1


class MockJEV:
    key = 'mock-only'
    total = 0
    lock = threading.Lock()
    def ask(self, state, assets, questions):
        with self.lock: type(self).total += 1
        assert 'QUERY PATCH' in state['task']
        assert list(assets) == ['benchmark_image']
        n = len(questions['rock']['criteria'])
        return {'answers': {'rock': {'probabilities': {str(i): 1.0 if i == 0 else 0.0 for i in range(n)}}}}


def test_mocked_jev_budget_cache_and_agreement(fixture_data, tmp_path):
    data,p,cs=fixture_data
    MockJEV.total=0
    verifier=Verifier(max_calls=2,workers=8,client_factory=MockJEV,cache=tmp_path/'jev')
    verifier.run([(p,c) for c in cs], [p])
    assert verifier.calls == MockJEV.total == 2
    assert sum(c['verification']['status']=='ok' for c in cs) == 2
    assert all(c['verification'].get('agreement') is True for c in cs if c['verification']['status']=='ok')
    again=Verifier(max_calls=0,client_factory=lambda: pytest.fail('Cache must avoid client construction'),cache=tmp_path/'jev')
    for c in cs:
        if c['verification']['status']=='ok':
            assert again.verify(p,c,[p])['cache_hit']
    assert again.calls == 0
    cached=next(c for c in cs if c['verification']['status']=='ok')
    with Image.open(local_file(cached['file'])) as im: changed=im.convert('RGB')
    changed.putpixel((0,0),(255,0,255));changed.save(local_file(cached['file']))
    assert again.verify(p,cached,[p])['status']=='budget_exhausted'


def test_jev_failure_retries_without_caching_and_eight_failure_circuit(fixture_data,tmp_path):
    _,p,cs=fixture_data
    class Blocked(MockJEV):
        def ask(self,*args): raise RuntimeError('blocked')
    delays=[]
    verifier=Verifier(client_factory=Blocked,cache=tmp_path/'blocked',sleeper=delays.append)
    a=verifier.verify(p,cs[0],[p]); b=verifier.verify(p,cs[0],[p])
    assert a['status']==b['status']=='unavailable'
    assert verifier.calls == 6 and not verifier.stopped
    assert delays == [2,5,10,2,5,10]
    assert not list((tmp_path/'blocked').glob('*.json'))
    verifier.verify(p,cs[1],[p])
    assert verifier.calls == 8 and verifier.stopped
    assert verifier.verify(p,cs[2],[p])['status']=='circuit_open'


def test_jev_retry_recovers_resets_failures_and_ignores_old_failed_cache(fixture_data,tmp_path):
    _,p,cs=fixture_data
    class Flaky(MockJEV):
        attempts=0
        def ask(self,*args):
            self.attempts+=1
            if self.attempts <= 2: raise RuntimeError('temporary')
            return super().ask(*args)
    delays=[]
    v=Verifier(client_factory=Flaky,cache=tmp_path/'flaky',sleeper=delays.append)
    key=v.request_spec(p,cs[0],[p])[3]
    write_json(v.cache/(key+'.json'),dict(status='unavailable',key=key))
    assert v.verify(p,cs[0],[p])['status']=='ok'
    assert v.calls==3 and v.consecutive_failures==0 and not v.stopped
    assert delays==[2,5]
    assert v.verify(p,cs[0],[p])['cache_hit']


def test_colour_gate_and_contrastive_margin():
    right=np.full((80,40,3),(240,220,30),np.uint8); right[::8,::4]=0
    wrong=np.full_like(right,(30,190,90)); wrong[::8,::4]=0
    ref=profile(right)
    assert colour_gate(ref,profile(right))[0]
    assert not colour_gate(ref,profile(wrong))[0]
    # Enough of the right fill alone cannot excuse a different dominant fill.
    mixed=np.full_like(right,(240,220,30));mixed[:41]=(30,190,90)
    mixed_pass,mixed_share,_=colour_gate(ref,profile(mixed))
    assert mixed_share >= .5*ref['share'] and not mixed_pass
    shifted=np.clip(right.astype(int)+np.array([-10,-10,0]),0,255).astype(np.uint8)
    assert colour_gate(ref,profile(shifted))[0]
    assert contrastive({'a':.8,'b':.9},'a')<0
    assert contrastive({'a':.9,'b':.8},'a')>0
    assert contrastive({'a':.9,'b':.9,'c':.8},'a',{'b'})>0
    assert contrastive({'a':.9,'b':.9},'a')==0
    references=[ref,profile(wrong),profile(np.zeros_like(right))]
    matcher=PaletteMatcher(references,[[0,0,0]]*3)
    batched=matcher(profile(right))
    for i,r in enumerate(references):
        expected=colour_gate(r,profile(right))
        assert batched[0][i]==expected[0]
        assert batched[1][i]==pytest.approx(expected[1])
        assert batched[2][i]==pytest.approx(expected[2])


def test_gallery_and_choice_use_benchmark_panels(fixture_data,monkeypatch):
    import synth.mine as module
    from synth.common import norm
    _,p,cs=fixture_data
    rocks=[dict(p,pid=p['pid']+str(i),label='Rock '+chr(65+i)) for i in range(16)]
    boxes={norm(r['label']):[i%4*100,i//4*100,i%4*100+35,i//4*100+35] for i,r in enumerate(rocks)}
    monkeypatch.setitem(module._LEGENDS,'a',(Image.new('RGB',(400,400),'white'),[0,0,400,400],boxes))
    gallery,_,small=verification_panel(rocks[0],cs[0],rocks[:15])
    choice,vocab,large=verification_panel(rocks[0],cs[0],rocks)
    assert small=='jev_gallery' and large=='jev_choice' and len(vocab)==16
    assert gallery.width==choice.width==1024
    # A column mined in another legend still uses the target's native choice
    # legend; it must not require nonexistent labels in the source log.
    outside=dict(cs[0],log='different-legend')
    wide,_,method=verification_panel(rocks[0],outside,rocks)
    assert method=='jev_choice' and wide.tobytes()==choice.tobytes()
    from classify.data import panel
    from types import SimpleNamespace
    with Image.open(local_file(cs[0]['file'])) as im: query=np.asarray(im.convert('RGB'))
    with Image.open(local_file(p['swatches'][0]['file'])) as im: swatch=np.asarray(im.convert('RGB'))
    ordered=sorted(rocks,key=lambda r:r['pid'])
    entries=[SimpleNamespace(label=r['label'],swatch=swatch,swatch_box=boxes[norm(r['label'])]) for r in ordered]
    case=dict(entries=entries,vocab=[r['label'] for r in ordered],legend=Image.new('RGB',(400,400),'white'),info=dict(legend_box=[0,0,400,400]))
    assert choice.tobytes()==panel(case,[dict(rgb=query)],'choice').tobytes()
    smallentries=[e for e in entries if e.label!=rocks[-1]['label']]
    smallcase=dict(entries=smallentries,vocab=[e.label for e in smallentries])
    assert gallery.tobytes()==panel(smallcase,[dict(rgb=query)],'gallery').tobytes()


def test_embedding_cache_is_content_addressed(tmp_path):
    calls=[]
    def backend(images):
        calls.append(len(images)); return np.tile(np.array([[1,2,3]],np.float32),(len(images),1))
    im=np.full((48,20,3),120,np.uint8)
    embed=Embedder(tmp_path/'embeddings.sqlite',backend)
    first=embed([im,im]); embed.close()
    second=Embedder(tmp_path/'embeddings.sqlite',lambda _: pytest.fail('Cached vector should be reused'))
    assert np.array_equal(first,second([im,im]));second.close()
    assert calls == [1]


def test_review_save_undo_restart_and_approved_rebuild(fixture_data):
    data,p,cs=fixture_data;pid=p['pid'];store=ReviewStore(data)
    assert read_json(data/'approved.json') == {}
    store.save(pid,'approved',[1,3])
    approved=read_json(data/'approved.json')[pid]
    assert len(approved)==2
    assert local_file(approved[0]['file']).read_bytes() == local_file(cs[0]['file']).read_bytes()
    assert local_file(approved[1]['file']).name=='02.png'
    store.save(pid,'bad',[1])
    assert not list((data/'approved').glob('*/*.png'))
    assert read_json(data/'approved.json')=={}
    store.undo(pid)
    assert len(read_json(data/'approved.json')[pid])==2
    (data/'approved.json').write_text('{}',encoding='utf8')
    restored=ReviewStore(data)
    assert restored.decisions()[pid]['status']=='approved'
    assert len(read_json(data/'approved.json')[pid])==2
    restored.undo(pid)
    assert restored.decisions()=={} and read_json(data/'approved.json')=={}
    assert [e['kind'] for e in restored.events]==['decision','decision','undo','undo']
    with pytest.raises(ValueError): restored.save(pid,'approved',[99])


def test_review_http_first_screen_save_and_undo(fixture_data):
    data,p,cs=fixture_data
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(ReviewStore(data)))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def get(path):
        with urlopen(base+path,timeout=2) as r:return r.read()
    def post(path,body):
        req=Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=2) as r:return json.load(r)
    try:
        started=time.perf_counter()
        assert b'Does the column match the legend?' in get('/')
        assert json.loads(get('/api/bootstrap'))['patterns'][0]['pid']==p['pid']
        assert json.loads(get('/api/pattern?pid='+p['pid']))['picks']==[1,3]
        assert time.perf_counter()-started<1
        post('/api/save',dict(pid=p['pid'],status='approved',picks=[1]))
        assert len(read_json(data/'approved.json')[p['pid']])==1
        post('/api/undo',dict(pid=p['pid']))
        assert read_json(data/'approved.json')=={}
    finally:server.shutdown();server.server_close();worker.join()


def test_source_trust_filters_mock_and_disputed(tmp_path,monkeypatch):
    import synth.mine as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    q=tmp_path/'labeltool'/'data';q.mkdir(parents=True)
    rows=[dict(log='x',top_y=0,bottom_y=100,suggestions=[dict(source='JEV/cluster',label='Shale',tier='medium'),dict(source='colour/Lab',label='Clay',tier='low'),dict(source='silver/A',label='Sand',tier='high')]),
          dict(log='x',top_y=100,bottom_y=200,mock=True,suggestions=[dict(source='JEV/interval',label='Sand',tier='high')])]
    (q/'prefill.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
    result=source_intervals()
    assert len(result['x'])==1 and result['x'][0]['source']=='prefill/JEV'


def test_native_mining_pipeline_with_mocked_jev(fixture_data,tmp_path,monkeypatch):
    import synth.mine as module
    data,p,_=fixture_data
    p['swatches']=[dict(p['swatches'][0],log=name) for name in p['logs']]
    pixels={name:np.random.default_rng(i).integers(220,250,(512,24,3),dtype=np.uint8) for i,name in enumerate(p['logs'])}
    class Log:
        def __init__(self,name):
            self.name=name;self.boxes={'interpreted_lithology_column':[50,100,74,612]}
        def close(self):pass
    def read(log,box,y0=None,y1=None,**kwargs):
        return pixels[log.name][(100 if y0 is None else y0)-100:(612 if y1 is None else y1)-100].copy()
    def backend(images):
        return np.stack([np.random.default_rng(int.from_bytes(hashlib.sha256(im.tobytes()).digest()[:8],'big')).normal(size=128) for im in images])
    real_embed,real_verify=module.Embedder,module.Verifier
    monkeypatch.setattr(module,'DATA',data)
    monkeypatch.setattr(module,'CACHE',tmp_path/'scan-cache')
    monkeypatch.setattr(module,'scan_identity',lambda *args:hashlib.sha256(b''.join(pixels[n].tobytes() for n in sorted(pixels))).hexdigest())
    monkeypatch.setattr(module.io,'open_log',Log)
    monkeypatch.setattr(module.io,'read_rgb',read)
    monkeypatch.setattr(module,'Embedder',lambda:real_embed(tmp_path/'integration.sqlite',backend))
    monkeypatch.setattr(module,'Verifier',lambda **kw:real_verify(**kw,client_factory=MockJEV,cache=tmp_path/'integration-jev'))
    monkeypatch.setattr(module,'source_intervals',lambda:defaultdict(list,{name:[dict(top_y=100,bottom_y=612,lithology='Shale',source='gold')] for name in p['logs']}))
    monkeypatch.setattr(module,'detected_intervals',lambda rgb,offset:[(100,612)])
    catalog=dict(patterns=[p],no_candidates=[p['pid']])
    report=mine(catalog,max_calls=3,workers=2,resume_scan=True)
    assert report['logs']==4 and report['errors']==[]
    assert report['actual_jev_calls']==3
    results=read_json(data/'candidates.json')[p['pid']]
    assert results and len(results)<=30
    assert max(Counter(c['log'] for c in results).values())<=len(results)*.25
    for c in results:
        assert c['interval_top']+8<=c['top_y']<c['bottom_y']<=c['interval_bottom']-8
        assert c['h']==max(48,3*c['repeat_height'])
        with Image.open(local_file(c['file'])) as im:
            assert im.size==(24,c['h'])
            assert np.array_equal(np.asarray(im),pixels[c['log']][c['top_y']-100:c['bottom_y']-100])

    resumed=mine(catalog,no_jev=True,workers=2,resume_scan=True)
    assert resumed['scan_resumed'] and resumed['logs']==4
    again=read_json(data/'candidates.json')[p['pid']]
    assert [(c['file'],c['top_y'],c['margin']) for c in again]==[(c['file'],c['top_y'],c['margin']) for c in results]

    # Changing only fill colour eliminates every candidate; quotas never
    # repopulate a failed gate, even for trusted seed intervals.
    for name in pixels: pixels[name][:]=(30,180,80)
    empty=mine(catalog,dry_run=True,workers=2)
    assert empty['sheet_candidates']==0
    assert empty['actual_jev_calls']==0
    assert read_json(data/'candidates.json')[p['pid']]==[]
    assert 'no colour match' in empty['pattern_counts'][p['pid']]['shortfall']
