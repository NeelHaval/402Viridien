from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from strategies import jev_sourcecards as api
from strategies import s30_jev_sourcecards as S


class TestSourcecards(unittest.TestCase):
    def test_dark_rock_not_blank(self):
        self.assertFalse(S.blank(np.zeros((48,60,3),dtype=np.uint8)))
        self.assertTrue(S.blank(np.full((48,60,3),255,dtype=np.uint8)))

    def test_identity_keeps_separate_and_genuine_compound(self):
        a=np.zeros((10,10,3),np.uint8);C=SimpleNamespace(_native=lambda:{'x':[('Clay',a),('Shale',a),('Clay/Shale',a)]},NOT_ROCK=__import__('re').compile('NEVER'))
        e,source=S.source_entries(C,SimpleNamespace(name='x'),{})
        self.assertEqual([x['label'] for x in e],['Clay','Shale','Clay/Shale'])
        self.assertEqual([x['id'] for x in e],[0,1,2])

    def test_payload_has_only_local_source_options(self):
        a=np.zeros((30,20,3),np.uint8);e=[dict(id=1,label='Clay',pixels=a),dict(id=2,label='Shale',pixels=a)]
        q=[dict(item_id='Q1',pixels=a,candidates=[2])];b,files,key=api.payload(e,q)
        self.assertEqual(set(b['questions']['Q1_label']['criteria']),{'L2'})
        self.assertEqual(set(b['state']['source_legend']),{'L2'})
        self.assertEqual(len(files),1)
        self.assertEqual(api.payload(e,q)[2],key)
        q[0]['pixels']=np.ones((30,20,3),np.uint8)
        self.assertNotEqual(api.payload(e,q)[2],key)

    def test_parser_rejects_foreign_choices_and_preserves_review(self):
        q=[dict(item_id='Q1',candidates=[2])];e=[dict(id=2,label='Shale')]
        raw=dict(model=api.MODEL,answers={'Q1_label':dict(choice='L2',confidence=.99,probabilities={'L2':.99}),'Q1_quality':dict(choice='match'),'Q1_fits':dict(noul=.1)})
        c=api.parse(raw,q,e)[0];self.assertTrue(c['review']);self.assertEqual(c['label'],'Shale')
        q[0]['candidates']=[1]
        with self.assertRaises(ValueError):api.parse(raw,q,e)

    def test_source_pixel_shape_is_part_of_key(self):
        self.assertNotEqual(api.image_hash(np.zeros((2,6,3),np.uint8)),api.image_hash(np.zeros((3,4,3),np.uint8)))

    def test_consecutive_chunks_keep_their_own_predictions(self):
        a=np.zeros((48,20,3),np.uint8);b=np.full((48,20,3),100,np.uint8)
        c=SimpleNamespace();entries=[dict(id=0,label='Clay',pixels=a),dict(id=1,label='Shale',pixels=b)]
        def segment(log,ctx,classifier,window):
            beds=[]
            for n,image in enumerate([a,b]):
                p=classifier.classify(image,log.name,check=False)
                beds.append(dict(top=n*48,bottom=(n+1)*48,state=p['state'],label=p.get('label'),chunks=[]))
            return dict(beds=beds)
        def classify(entries,queries,cache):
            cells={q['item_id']:dict(label=entries[n]['label'],quality='match',confidence=.99,review=False) for n,q in enumerate(queries)}
            return cells,dict(fresh_requests=1,cached_requests=0,failures=[])
        with patch.object(S.adapter,'make_classifier',return_value=c),patch.object(S,'source_entries',return_value=(entries,'test')),patch.object(S.adapter,'segment',side_effect=segment),patch.object(api,'ranks',return_value=[[0],[1]]),patch.object(api,'classify',side_effect=classify):
            rows,_=S.run(SimpleNamespace(name='x'),(0,96),dict(calib=None,cache_dir='unused'))
        self.assertEqual([r['lithology'] for r in rows],['Clay','Shale'])


if __name__=='__main__':unittest.main()
