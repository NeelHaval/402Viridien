from pathlib import Path
p=Path(__file__).with_name('score_all.py');s=p.read_text(encoding='utf-8')
s=s.replace("rec['over_budget_60s']=rec.get('seconds',0)>60", "for timing in ('seconds','legend_seconds','depth_seconds','preprocess_seconds'):rec.setdefault(timing,None)\n            rec['over_budget_60s']=rec['seconds']>60 if rec['seconds'] is not None else None")
s=s.replace("seconds={rec.get('seconds',0):.2f}", "seconds={rec.get('seconds')}")
p.write_text(s,encoding='utf-8')
