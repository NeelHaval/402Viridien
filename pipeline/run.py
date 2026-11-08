"""Run the submitted S19 model on organiser images and bounding boxes."""
import argparse,csv,json,os,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
sys.path.insert(0,str(Path(__file__).parent/'s19'))
sys.path.insert(1,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from harness.io import Log,read_rgb
from harness.depth import calibrate
from ml.combine.runtime import run
from scripts.challenge_export import _write_csv,NOTEBOOK_FIELDS

def main():
 p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--boxes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 images=sorted(x for x in a.images.iterdir() if x.suffix.lower() in ('.tif','.tiff','.png','.jpg','.jpeg'))
 for image in images[:a.limit]:
  annotation=a.boxes/(image.stem+'.json')
  data=json.loads(annotation.read_text(encoding='utf-8'));boxes=data.get('bounding_boxes',data)
  log=Log(Image.open(image),boxes,image.stem);box=list(map(int,boxes['interpreted_lithology_column']));status={}
  try: calib=calibrate(log);status['calibration']=vars(calib)
  except Exception as exc: calib=None;status['depth_unavailable']=str(exc)
  rows=run(log,(box[1],box[3]),{'calib':calib,'device':'cpu','cache':str(a.out/'cache')})
  for row in rows:
   for edge in ('top','bottom'): row.setdefault(edge+'_depth','')
  _write_csv(a.out/(image.stem+'.csv'),NOTEBOOK_FIELDS,rows)
  canvas=Image.fromarray(read_rgb(log,'interpreted_lithology_column'));canvas=canvas.convert('RGB');canvas2=Image.new('RGB',(canvas.width+420,canvas.height),'white');canvas2.paste(canvas,(0,0));draw=ImageDraw.Draw(canvas2)
  for row in rows:
   y=int(row['top_y'])-box[1];draw.line((0,y,canvas2.width,y),fill='red',width=2);draw.text((canvas.width+5,y),f"{row['lithology']} / {row['cluster_id']} / {row['top_depth']}",fill='black')
  canvas2.save(a.out/(image.stem+'.png'));status['rows']=len(rows);(a.out/(image.stem+'.json')).write_text(json.dumps(status,indent=2),encoding='utf-8');log.close();print(image.name,len(rows),status,flush=True)
if __name__=='__main__': main()
