import json,pathlib
root=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3')
for slug in ('coso_forge','utah_forge','dhi_wister'):
 d=root/slug; docs=[]
 for m in json.loads((d/'technical_metadata.json').read_text()):
  name=m['file']; ps=m['pages']; vector=any(p['drawings'] or p['text_chars'] for p in ps); td='_TD_Logging_Data' in name
  if slug=='coso_forge':
   url='https://gdr.openei.org/files/758/52B-7%20Lithology%20Log.pdf' if name=='52B-7.pdf' else 'https://gdr.openei.org/files/807/WestFlankMudLogs.zip'
   legend=not td; layout='wellsight_striplog' if vector else 'bw_hatch'
   note=('Archive member: '+name+'. ' if name!='52B-7.pdf' else 'Direct PDF; byte-identical to archive member 52B-7 Lithology Log.pdf. ')
   if td: note+='No legend found in the PDF; related 33A-7 versions have a legend on page 1. '
   if vector: note+='WellSight-style layout is a visual inference; producer and creator do not identify WellSight. '
   else: note+='Scanned pages are sideways; header and patterned column are visible. '
  elif slug=='utah_forge':
   url='https://gdr.openei.org/files/1292/16A%2878%29-32%20Mud%20Log%20Final%20130-10987%27.pdf';legend=True;layout='other';note='Horizon Well Logging geothermal mud log; raster PDF with three very tall pages. Lithology symbol legend in page 1 header. '
  else:
   url='https://gdr.openei.org/files/904/wister%2012-27%20composite%20log.pdf';legend=True;layout='wellsight_striplog';note='DHI composite mud log; WellSight-style layout is a visual inference, not verified software provenance. '
  docs.append({'url':url,'path':str((d/name).relative_to(root.parent.parent)),'bytes':m['bytes'],'pages':len(ps),'page_sizes_points':[[p['size_points'][0],p['size_points'][1]] for p in ps], 'raster_or_vector':'vector' if vector else 'raster','producer':m['metadata'].get('producer') or None,'creator':m['metadata'].get('creator') or None,'is_log':True,'has_lith_column':True,'has_legend':legend,'legend_page':1 if legend else None,'layout_guess':layout,'born_digital_wellsight':bool(vector),'notes':note+'Per-page PyMuPDF drawing, text, image counts and sizes are in technical_metadata.json.'})
 more={'coso_forge':'The 11 PDFs in the six-well archive were fetched in full; additional 52B-7 direct PDF is a duplicate. Other catalog submissions may contain related logs; additional unique count unknown.','utah_forge':'0 more requested 16A(78)-32 mud-log PDFs identified; catalog lists other wireline and processed logs.','dhi_wister':'0 more requested Wister 12-27 composite PDFs identified; additional related logs not surveyed.'}[slug]
 (d/'triage.json').write_text(json.dumps({'slug':slug,'documents':docs,'blocked':[],'more_available':more},indent=2)+'\n')
 print(slug,len(docs),sum(x['has_legend'] for x in docs),sum(x['born_digital_wellsight'] for x in docs))
