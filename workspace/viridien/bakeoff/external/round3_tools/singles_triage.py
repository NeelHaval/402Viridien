import os,json,pymupdf,glob
root='external_untrusted/round3/'
for p in glob.glob(root+'singles/_thumbs/0885_p*.png')+glob.glob(root+'singles/*.overview.png'): os.unlink(p)
data={
'stratagraph':[
('Measured_Depth_Log_Example.pdf','https://stratagraph.com/wp-content/uploads/2019/11/Measured_Depth_Log_Example.pdf','raster',True,True,True,1,'other','Colour lithology track and legend. GeoDraft output rasterized.'),
('New-Mexico-Sample-Logs-1INCH-HORIZONTAL.pdf','https://stratagraph.com/wp-content/uploads/2019/11/New-Mexico-Sample-Logs-1INCH-HORIZONTAL.pdf','vector',True,True,True,1,'other','Horizontal GeoDraft log with colour lithology and legend; page 1 includes an image and vector artwork.'),
('Engineering_Log_Example.pdf','https://stratagraph.com/wp-content/uploads/2019/11/Engineering_Log_Example.pdf','raster',True,False,False,None,'other','Engineering curves; no lithology column observed.'),
('Geopressure_Log_Example.pdf','https://stratagraph.com/wp-content/uploads/2019/11/Geopressure_Log_Example.pdf','raster',True,False,False,None,'other','Geopressure curves; no lithology column observed.')],
'precision_wl':[
('log-01.pdf','https://pwl-inc.com/wp-content/uploads/2018/06/log-01.pdf','raster',True,True,True,1,'other','Colour lithology column and lithology symbol key; raster images.'),
('log-02.pdf','https://pwl-inc.com/wp-content/uploads/2018/06/log-02.pdf','vector',True,True,True,1,'other','Vector lithology column and printed lithology symbols. No evidence of WellSight.'),
('log-03.pdf','https://pwl-inc.com/wp-content/uploads/2018/06/log-03.pdf','raster',True,False,False,None,'other','Horizontal trajectory and curves; no lithology column or rock legend apparent.')],
'singles':[
('0885_log_pages_60-71.pdf','https://cmscontent.nrs.gov.bc.ca/geoscience/Coal/COALReports/0885.pdf','raster',True,True,True,1,'wellsight_striplog','Extracted original pages 60–71 of 71; legend original page 60. Scanned WellSight STRIP.LOG with OCR overlay, not born digital. Original download 27264142 bytes.'),
('NYSERDA_00449_log_page_32.pdf','https://esogis.nysm.nysed.gov/sites/default/files/NYSERDA/NYSERDA_00449_Report.pdf','raster',True,True,False,None,'bw_hatch','Extracted original page 32 of 275, Figure 2-3; patterned lithology and descriptions, no key on figure. Original download 7097361 bytes. Download used TLS without certificate verification after certificate verification failed.'),
('Bilyara1.pdf','https://www.occam.com.au/static/sample-data/Prints/Bilyara1.pdf','vector',True,False,False,None,'other','Composite wireline curves; no patterned lithology track or rock legend apparent.')],
}
blocked={'stratagraph':[],'precision_wl':[],'singles':['GSQ cr_84155_1.pdf: HTTP 403 Forbidden at https://gsq-horizon.s3-ap-southeast-2.amazonaws.com/QDEX/84155/cr_84155_1.pdf','New Brunswick GR 2014-2: HTTP 403 Forbidden at https://leglibbibcat.legnb.ca/e-repository/monographs/31000000055636/31000000055636.pdf','Quebec 2008OA25702.pdf: HTTP 403 Blocked at https://gq.mines.gouv.qc.ca/documents/examine/2008OA25702/2008OA25702.pdf','Pennsylvania Harry Dewey #1: publication page timed out twice at https://maps.dcnr.pa.gov/publications/Default.aspx?id=980']}
more={'stratagraph':'0 known: all four gallery PDFs fetched.','precision_wl':'0 known: all three services-page sample PDFs fetched.','singles':'4 named documents remain inaccessible (three HTTP 403, one timeout); unknown whether alternative routes would work. No broader count established.'}
for slug,items in data.items():
 docs=[]
 for name,url,rv,log,lith,legend,legpage,layout,note in items:
  path=root+slug+'/'+name; d=pymupdf.open(path); sizes=[[round(p.rect.width,2),round(p.rect.height,2)] for p in d]; m=d.metadata
  docs.append(dict(url=url,path=path,bytes=os.path.getsize(path),pages=len(d),page_sizes_points=sizes,raster_or_vector=rv,producer=m.get('producer'),creator=m.get('creator'),is_log=log,has_lith_column=lith,has_legend=legend,legend_page=legpage,layout_guess=layout,born_digital_wellsight=False,notes=note))
 with open(root+slug+'/triage.json','w') as f: json.dump(dict(slug=slug,documents=docs,blocked=blocked[slug],more_available=more[slug]),f,indent=2); f.write('\n')
