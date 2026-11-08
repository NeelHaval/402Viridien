from pathlib import Path
import requests,time,json,fitz
r=Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work');out=r/'external_untrusted/round6/r6_europe_africa/sodir';s=requests.Session();s.headers['User-Agent']='Mozilla/5.0 (research crawler)';recs=[]
for u in ['https://factpages.sodir.no/pbl/wellbore_documents/4423_34_7_32_COMPLETION_REPORT.PDF','https://factpages.sodir.no/pbl/wellbore_documents/4939_6608_11_4_COMPLETION_REPORT.PDF']:
 time.sleep(1.1)
 try:
  x=s.get(u,timeout=60);x.raise_for_status();p=out/u.split('/')[-1];p.write_bytes(x.content);d=fitz.open(p);hits=[i for i,pg in enumerate(d) if 'FORMATION EVALUATION LOG' in pg.get_text().upper()];recs.append(dict(url=u,path=str(p.relative_to(r)),bytes=len(x.content),pages=len(d),hits=hits));print(p.name,len(d),hits,flush=True)
 except Exception as e:print(str(e)[:150],flush=True)
 (out/'more_records.json').write_text(json.dumps(recs,indent=2),encoding='utf-8')
# Text and dimensions for the best candidate, used to select image regions.
d=fitz.open(out/'5093_7131_4_1_COMPLETION_GEOLOGICAL_REPORT.PDF');p=d[105]
print('BEST',p.rect,flush=True)
(out/'best_text.txt').write_text(p.get_text(),encoding='utf-8')
for j,y in enumerate([0,p.rect.height*.35,p.rect.height*.65]):
 clip=fitz.Rect(0,y,p.rect.width,min(p.rect.height,y+p.rect.width*1.4));z=min(1024/clip.height,1024/clip.width);p.get_pixmap(matrix=fitz.Matrix(z,z),clip=clip).save(out/f'best_region_{j}.png')
