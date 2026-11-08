import json,pathlib
root=pathlib.Path('/home/user/lith/viridien_work/external_untrusted/round3/nwt_orogo'); ins=json.loads((root/'inspection.json').read_text()); dl=json.loads((root/'downloads.json').read_text()); docs=[]
notes=[
 'Standalone Cameron J-74 STRIP.LOG; header and ROCK TYPES legend on p1, lithology track starts p2. Vector text and hundreds of drawings on log pages. Pattern extraction and depth registration remain unverified.',
 'Mirror Lake P-20 geological report with two embedded STRIP.LOG sections, pp65–98 (vertical depth) and pp99–118 (horizontal); ROCK TYPES legends on pp65 and 100. Vector text and drawings, with embedded photos/images; p110 horizontal lithology track runs left to right. Test extraction separately for each orientation.',
 'WellSight STRIP.LOG-generated gamma correlation log, not a lithology strip log: inspected pp1, 2 and 16, and no lithology column or ROCK TYPES legend found. Filename says TVD; first-page header says measured depth. Vector text/curves. Unsuitable for lithology matching despite WellSight provenance.',
 'Continental Laboratories hydrocarbon well log. Patterned lithology column and depth track visible on pp1 and 16; p1 says lithologic symbols are CANSTRAT compatible, but no printed rock-type key found in text or visual survey. Vector drawings and text, not a WellSight document.'
]
for i,x in enumerate(ins):
 pagesizes={}
 for p in x['page_details']:
  k=' × '.join(map(str,p['size_points']))+' pt'; pagesizes.setdefault(k,[]).append(p['page'])
 docs.append({'url':x['url'],'path':'external_untrusted/round3/nwt_orogo/'+x['path'],'bytes':x['bytes'],'pages':x['pages'],'page_size_points':pagesizes,'raster_or_vector':'vector (log pages contain text and drawings; some embedded images)' if i in (0,1,2) else 'vector (log pages contain text and drawings)','producer':x['producer'],'creator':x['creator'],'is_log':True,'has_lith_column':i!=2,'has_legend':i in (0,1),'legend_page': [1] if i==0 else [65,100] if i==1 else None,'layout_guess': 'wellsight_striplog' if i<3 else 'other','born_digital_wellsight':i<3,'notes':notes[i]})
result={'slug':'nwt_orogo','documents':docs,'blocked':dl['blocked'],'more_available':'Unknown total. Two additional P-20 standalone SL PDF links were found but returned HTTP 504; they may be retried. The registry lists 685 wells and 21 records were sampled; approximately 664 records remain unexamined, not a count of additional logs. Initial discovery searched filenames for strip/mud/lithology, which can miss abbreviated _SL_ files.'}
(root/'triage.json').write_text(json.dumps(result,indent=2)+'\n')
