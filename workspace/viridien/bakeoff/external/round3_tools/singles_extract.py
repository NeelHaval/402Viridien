import pymupdf,os
root='external_untrusted/round3/singles/'
for src,dst,a,b in [('0885.pdf','0885_log_pages_60-71.pdf',59,70),('NYSERDA_00449_Report.pdf','NYSERDA_00449_log_page_32.pdf',31,31)]:
 d=pymupdf.open(root+src); out=pymupdf.open(); out.insert_pdf(d,from_page=a,to_page=b); out.set_metadata({k:v for k,v in d.metadata.items() if k in ('author','title','subject','keywords','creator','producer','creationDate','modDate','trapped')}); out.save(root+dst,garbage=4,deflate=True); out.close(); d.close(); os.remove(root+src)
