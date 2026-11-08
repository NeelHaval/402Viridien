import pathlib,sys
p=pathlib.Path(__file__).with_name('sample.py')
code=p.read_text(encoding='utf-8')
start=code.index('items=[');end=code.index('records=[]')
code=code[:start]+'''items=[
('mexico','https://rondasmexico.energia.gob.mx/media/1161/lista-pozos-r1l04-2015.pdf'),
('mexico','https://rondasmexico.energia.gob.mx/media/1216/18-abr-2016.pdf'),
('trinidad','https://www.energy.gov.tt/wp-content/uploads/2013/12/Data-Package-Listing.pdf')]
'''+code[end:]
code=code.replace("base/'sample_index.json'","base/'extra_index.json'")
exec(compile(code,str(p),'exec'))
