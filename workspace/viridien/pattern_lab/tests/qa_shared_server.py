"""Use copies of all review seeds for shared-client browser checks."""
import shutil,sys
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import DATA,Store,Images,read,write
from server import create_handler
root=DATA/'qa-shared';root.mkdir(exist_ok=True)
catalog=read(DATA/'documents.json');write(root/'documents.json',catalog)
(root/'seeds').mkdir(exist_ok=True)
for doc in catalog:shutil.copyfile(DATA/'seeds'/(doc['id']+'.json'),root/'seeds'/(doc['id']+'.json'))
shutil.copyfile(DATA/'pattern-queue.json',root/'pattern-queue.json')
print('Shared review test server: http://127.0.0.1:8777',flush=True)
ThreadingHTTPServer(('0.0.0.0',8777),create_handler(Store(root),Images(),'http://192.168.0.92:8777')).serve_forever()
