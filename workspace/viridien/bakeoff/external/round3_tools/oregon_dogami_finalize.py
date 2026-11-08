import json,os
from PIL import Image,ImageDraw
root='external_untrusted/round3/oregon_dogami';path=root+'/triage.json';data=json.load(open(path)); cells=[]
for d in data['documents']:
 name=os.path.basename(d['path']); bad=name.startswith('Burgess2_');d['is_log']=not bad;d['has_lith_column']=not bad;d['has_legend']=not bad;d['legend_page']=1 if not bad else None;d['layout_guess']='bw_hatch' if not bad else 'other';d['born_digital_wellsight']=False
 d['notes']=('Electrical resistivity/porosity log; mud characteristics header, but no lithology column or lithology legend observed.' if bad else 'Scanned monochrome mud log with lithology symbols in the page 1 header and a lithology track; bw_hatch is a broad visual layout guess. Thin tracks and small legend symbols may be difficult to read at thumbnail resolution.')
 for t in d.pop('_thumbs'): cells.append((name[:-4]+' '+os.path.basename(t).split('_')[-1][:-4],t))
 d.pop('_text_hint',None)
with open(path,'w') as f:json.dump(data,f,indent=2)
w=1200;cw=200;ch=290;sheet=Image.new('RGB',(w,((len(cells)+5)//6)*ch),'white');draw=ImageDraw.Draw(sheet)
for j,(name,p) in enumerate(cells):
 im=Image.open(p);im.thumbnail((190,255));x=j%6*cw+(cw-im.width)//2;y=j//6*ch;sheet.paste(im,(x,y));draw.text((j%6*cw+3,y+257),name[:30],fill='grey')
sheet.save(root+'/contact_sheet.png')
for i in range(4):os.remove(root+'/_thumbs/review%d.png'%i)
print(len(data['documents']),len(cells),sum(d['has_legend'] and d['has_lith_column'] and d['is_log'] for d in data['documents']))
