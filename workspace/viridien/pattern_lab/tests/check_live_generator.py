"""Check live generated records against the real training snapshot."""
import hashlib,io,json,sys,urllib.request,zipfile
from pathlib import Path
from PIL import Image
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Store
from normalized import SplitPolicy,letterbox
data=Path(__file__).resolve().parents[1]/'data';store=Store();policy=SplitPolicy(store)
state=json.load(urllib.request.urlopen('http://127.0.0.1:8781/state.json',timeout=20))
assert state['status']['error'] is None,state['status']
body=urllib.request.urlopen('http://127.0.0.1:8781/dataset.zip',timeout=30).read()
with zipfile.ZipFile(io.BytesIO(body)) as z:
    assert z.testzip() is None
    manifest=json.loads(z.read('manifest.json'));index=policy.load_index()
    docs={c['name']:store.get(c['id']) for c in store.catalog}
    assert manifest['kind']=='synthetic' and manifest['split']=='train'
    for record in manifest['patterns']:
        assert record['kind']=='synthetic' and record['split']=='train'
        assert policy.split(docs[record['document']])=='train'
        if record['source']=='human':assert record['normalized_sources']
        for source in record['normalized_sources']:
            row=index[source['sample_id']]
            assert row['usable_for_generation'] and row['annotation_version']==source['annotation_version']
            assert record['label_ids'][0] in [t['id'] for t in row['targets']]
        native=Image.open(io.BytesIO(z.read(record['native_png']))).convert('RGB')
        rgb,mask,transform=letterbox(native)
        assert transform==record['normalization']
        assert np.array_equal(np.asarray(rgb),np.asarray(Image.open(io.BytesIO(z.read(record['normalized_png'])))))
        assert np.array_equal(np.asarray(mask),np.asarray(Image.open(io.BytesIO(z.read(record['valid_mask_png'])))))
        assert hashlib.sha256(f'{native.width}x{native.height}:'.encode()+native.tobytes()).hexdigest()==record['pixel_sha256']
result=dict(result='PASS',synthetic_patterns=len(manifest['patterns']),synthetic_columns=len(manifest['columns']),human_patterns=state['human_pattern_count'],machine_verified_patterns=state['verified_pattern_count'],training_source_boxes=state['human_boxes'],test_groups_used=0,all_normalized_images_match_native=True)
(data/'continuation-generator-checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
archive=data/'exports'/('synthetic-'+hashlib.sha256(body).hexdigest()[:20]+'.zip');archive.write_bytes(body)
print(json.dumps(dict(result,archive=str(archive))),flush=True)
