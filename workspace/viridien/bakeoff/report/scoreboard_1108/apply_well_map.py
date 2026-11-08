from pathlib import Path
p=Path(__file__).with_name('score_all.py');s=p.read_text(encoding='utf-8')
s=s.replace("paths=[Path(__file__),REPORT/'evaluation_protocol.json',", "paths=[Path(__file__),REPORT/'evaluation_protocol.json',REPORT/'reviewed_dev_wells.json',")
s=s.replace("def summarize(records):\n    summaries=[];pairs=[]", "def summarize(records):\n    mapping_path=REPORT/'reviewed_dev_wells.json'\n    mapping=read(mapping_path)['log_to_group'] if mapping_path.exists() else {}\n    records=[{**r,'well':mapping.get(r['log'],r['well']) if r['suite']=='ext_dev' else r['well']} for r in records]\n    summaries=[];pairs=[]")
s=s.replace('First six legacy records were input-validated','Legacy records were input-validated')
p.write_text(s,encoding='utf-8')
