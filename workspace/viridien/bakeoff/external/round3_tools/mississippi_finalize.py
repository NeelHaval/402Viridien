import json,pathlib,re
from collections import Counter
root=pathlib.Path('external_untrusted/round3/mississippi'); s=json.loads((root/'progress.json').read_text())
legend_ids={'mdl123045000360000a','mdl123059200030000a','mdl523045000100000a','mdl523045000110000a','mdl523045000150000a','mdl523045000310000a','mdl523045000310000b','mdl523045000380000a','mdl523045000400000a','mdl523045200720000a','mdl523047200120000a','mdl123131200120000a','mdl523131200120000a','mdl223131200170000a'}
# Segments of these wells visibly continue the same log whose first segment includes a legend.
well_legend={'2304500036':'mdl123045000360000a','2304500011':'mdl523045000110000a','2304520072':'mdl523045200720000a','2304720012':'mdl523047200120000a'}
nonlog={'mdl523045000100000a-091':'blank end page with state watermark', 'mdl523045000150000a-072':'appended show report', 'mdl523045000400000a-008':'appended handwritten core record', 'mdl523045000400000a-009':'appended handwritten core record'}
logs={x['id']:x for x in s['logs']}
for d in s['documents']:
 stem=pathlib.Path(d['path']).stem; lid,num=stem.rsplit('-',1); num=int(num); l=logs[lid]; no=nonlog.get(stem)
 d['is_log']=not bool(no); d['has_lith_column']=False if no or stem=='mdl523045200720000a-001' else True
 d['has_legend']= True if lid in legend_ids and num==1 else (False if no or num==1 or l['page_count']==2 else None)
 d['legend_page']=1 if d['has_legend'] else None
 d['layout_guess']='core_description' if 'core record' in (no or '') else ('other' if no else 'bw_hatch')
 details=[f"{l['county']}; API {l['api']}; viewer log {lid}, page {num} of {l['page_count']}; descriptor {l['descriptor']}."]
 if no: details.append(no+'.')
 else:
  details.append('Scanned black-and-white mud log; lithology classification based on visual samples from this viewer log, not independent inspection of every page.')
  if lid in legend_ids: details.append(f"Printed lithology legend is on viewer page 1: {lid}-001.png.")
  elif l['api'][:10] in well_legend: details.append('A related segment of this well has a printed legend: '+well_legend[l['api'][:10]]+'-001.png; verify pattern consistency before use.')
  else: details.append('No lithology legend confirmed in the sampled pages of this segment.')
  if d['has_legend'] is None: details.append('Legend presence on this individual PNG was not independently checked.')
  if stem=='mdl523045200720000a-001': details.append('Header and legend; lithology track is empty on this page, populated on page 2.')
  if lid.startswith('mdl523047'): details.append('Some pages have severe scan shading.')
 d['notes']=' '.join(details)
more='At least 3 additional viewer logs: the 27-page Pearl River log https://geology.deq.ms.gov/Energy/Data02/MudLogs/PearlRiver/mdl123109200430000a.xml was deferred for time, and George well 23039200070000 has at least two more indexed segments (c and d). Other counties and unexamined folders may have more; total unknown.'
out={'slug':'mississippi','documents':s['documents'],'blocked':s['blocked'],'more_available':more}
(root/'triage.json').write_text(json.dumps(out,indent=2)+'\n')
print('viewer logs',len(logs),'directly confirmed legend logs',len(legend_ids),'PNG documents',len(s['documents']),'legend and lithology on same PNG',sum(d['has_legend'] is True and d['has_lith_column'] is True for d in s['documents']))
print('counties',dict(Counter(x['county'] for x in s['logs'])),'bytes',sum(x['bytes'] for x in s['documents']))
print('missing thumbnails',sum(not (root/'_thumbs'/pathlib.Path(d['path']).name).exists() for d in s['documents']))
