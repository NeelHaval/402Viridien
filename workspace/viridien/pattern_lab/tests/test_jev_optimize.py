import json
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_optimize import body, card, label_ids, overlap, parse, score


class TestJevOptimize(unittest.TestCase):
    def test_overlap_excludes_shared_pixels_only(self):
        self.assertTrue(overlap([0,0,10,20],[5,19,15,30]))
        self.assertFalse(overlap([0,0,10,20],[0,20,10,30]))

    def test_twins_do_not_receive_partial_credit(self):
        truth={'x':dict(legend_ids=[1],status='match')}
        result=score([dict(item_id='x',legend_ids=[1,2],status='uncertain')],[dict(item_id='x')],truth)
        self.assertEqual(result['correct'],0)
        self.assertEqual(label_ids('Clay / Shale',[dict(id=1,label_printed='Clay'),dict(id=2,label_printed='Shale')]),[])

    def test_companion_noul_blocks_confident_forced_choice(self):
        raw={'answers':{'x_label':dict(choice='L2',confidence=.99,probabilities={'L2':.99}),'x_fits':dict(noul=.1),'x_quality':dict(choice='match')}}
        cell=parse(raw,[dict(item_id='x')],{2})[0]
        self.assertEqual(cell['route'],'review')
        raw['answers']['x_fits']['noul']=.99
        self.assertEqual(parse(raw,[dict(item_id='x')],{2})[0]['route'],'suggest')
        with self.assertRaises(ValueError):parse(raw,[dict(item_id='x')],{3})

    def test_document_local_visual_refs_and_no_truth(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.png';Image.new('RGB',(30,20),'red').save(p)
            data={'sources':{'doc':{'legend':[dict(id=7,label='Clay',path=str(p))]}}}
            items=[dict(item_id='T1',document_id='doc',crop=str(p),truth_secret='not sent')]
            payload,files=body(data,items,'tile',True)
            text=json.dumps(payload)
            self.assertNotIn('truth_secret',text);self.assertNotIn('not sent',text)
            self.assertEqual(set(payload['questions']['T1_label']['criteria']),{'L7'})
            self.assertEqual(len(files),1)  # Identical source pixels are uploaded once.
            again,again_files=body(data,items,'tile',True)
            self.assertEqual(payload,again)
            with self.assertRaises(ValueError):body(data,items+[dict(item_id='x',document_id='other')],'tile')
            self.assertEqual(card(p,True).size,(280,560))
            self.assertEqual(card(p,False).size,(280,280))


if __name__=='__main__': unittest.main()
