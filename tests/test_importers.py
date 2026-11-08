import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from viridian402.data import load_manifest,save_intervals,sha256
from viridian402.importers import import_source


class ImporterTests(unittest.TestCase):
    def test_portable_copies_suite_tags_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';base=source/'viridien';dataset=root/'challenge'
            (dataset/'images').mkdir(parents=True);(dataset/'bbox_annotation').mkdir()
            boxes={'width':64,'height':128,'bounding_boxes':{'interpreted_lithology_column':[10,20,40,128],
                'depth_column':[0,20,8,128],'lithology_legend':[0,0,64,18]}}
            def page(folder,log,color,external=False):
                (folder/'images').mkdir(parents=True,exist_ok=True);(folder/'bbox_annotation').mkdir(exist_ok=True)
                Image.new('RGB',(64,128),color).save(folder/'images'/(log+'.tif'))
                (folder/'bbox_annotation'/(log+'.json')).write_text(json.dumps(boxes))
                if external:
                    (folder/'truth').mkdir(exist_ok=True)
                    save_intervals(folder/'truth'/(log+'.csv'),[{'top_y':20,'bottom_y':128,'lithology':'Shale','cluster_id':0}])
            human=base/'human_labels';human.mkdir(parents=True)
            page(dataset,'human', 'red');(human/'bbox_annotation').mkdir();(human/'truth').mkdir()
            (human/'bbox_annotation'/'human.json').write_text(json.dumps(boxes))
            save_intervals(human/'truth'/'human.csv',[{'top_y':20,'bottom_y':128,'lithology':'Shale','cluster_id':0}])
            (human/'eval_set.json').write_text(json.dumps([{'log':'human','y0':20,'y1':128,'tier':'human'}]))
            (human/'manifest.csv').write_text('id,labeller,snapshot,labelled_y\nhuman,person,2026-11-08,20-128\n')
            gold=base/'bakeoff'/'gt'/'gold';gold.mkdir(parents=True)
            page(dataset,'gold','green');save_intervals(gold/'gold.csv',[{'top_y':20,'bottom_y':128,'lithology':'Sandstone','cluster_id':1}])
            (base/'bakeoff'/'eval_set.json').write_text(json.dumps([{'log':'gold','y0':20,'y1':128,'tier':'gold'}]))
            for name,log,color,agency in [('external_shared','ext_nl_one','blue','nl'),('external_shared_verify','ext_verify','yellow','kgs')]:
                folder=base/name;page(folder,log,color,True)
                (folder/'eval_set.json').write_text(json.dumps([{'log':log,'y0':20,'y1':128}]))
                with (folder/'manifest.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,['id','source','well','source_url','notes']);writer.writeheader()
                    writer.writerow({'id':log,'source':agency,'well':log,'source_url':'https://archive.example/record.pdf','notes':'Metric source; truth depths converted to ft (m/0.3048)'})
            before={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
            imported=import_source(source,dataset,root/'out',include_unlabelled=False)
            records=load_manifest(imported)['records'];self.assertEqual(len(records),4)
            diagnostic=[r for r in records if r['split']=='verify'][0]
            self.assertIn('diagnostic_only',diagnostic['tags']);self.assertFalse(diagnostic['fresh_holdout'])
            metric=[r for r in records if r.get('image_depth_unit')=='m'][0]
            self.assertEqual(metric['depth_unit'],'ft')
            self.assertEqual(before,{str(p):sha256(p) for p in source.rglob('*') if p.is_file()})
            self.assertTrue(all(Path(r['image']).is_relative_to(root/'out') for r in records))


if __name__=='__main__':unittest.main()
