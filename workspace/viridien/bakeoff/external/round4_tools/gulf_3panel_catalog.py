import csv,datetime,io,sys,zipfile,collections,re
from pathlib import Path
p=Path(sys.argv[1]); rows=collections.Counter(); wells=set(); sample=[]; total=0
with zipfile.ZipFile(p) as z, z.open('FRSWellDataRawData/mv_frs_file_inventory.txt') as f:
    for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',errors='replace',newline='')):
        typ=r['DOC_TYPE'].lower()
        if not re.search(r'\bmud\s*log\b',typ) or r['REGION']!='G' or r['FILE_EXT'].lower() not in ('.tif','.tiff'): continue
        try: date=datetime.datetime.strptime(r['RELEASABLE_DATE'].split(' ')[0],'%m/%d/%Y').date()
        except ValueError: continue
        if date>datetime.date(2026,11,8): continue
        total+=1; wells.add(r['API']); rows['all']+=1
        try: wd=int(r['WATER_DEPTH'])
        except ValueError: wd=0
        try: run=datetime.datetime.strptime(r['RUN_DATE'].split(' ')[0],'%m/%d/%Y').year
        except ValueError: run=0
        if wd>=1000 and run>=2005:
            rows['deep_recent']+=1
            if len(sample)<8: sample.append({k:r[k] for k in ('DOC_ID','DOC_TYPE','API','AREA','BLOCK','RUN_DATE','WATER_DEPTH')})
print('released Gulf mud-log TIFF rows',total,'distinct APIs',len(wells),'deep >=1000ft run >=2005',rows['deep_recent'])
print(sample)
