"""Local pattern review server. Run: python -X utf8 server.py --port 8776."""
from __future__ import annotations
import argparse
import csv
import io
import json
import mimetypes
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from PIL import Image
from urllib.parse import parse_qs, urlparse
from model import HOME, DATA, Store, Images, Conflict, prepare, write
from review_api import ReviewAPI


class CropSource(Image.Image):
    def __init__(self, images, doc):
        super().__init__()
        self.images, self.doc = images, doc
        self._size = (doc['width'], doc['height'])
    def crop(self, box):
        return self.images.crop(self.doc, list(map(int, box)))


def create_handler(store, images, share_url='http://127.0.0.1:8776'):
    review=ReviewAPI(store,images,share_url)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if len(args)>1 and str(args[1]).startswith(('4','5')):
                super().log_message(fmt, *args)

        def respond(self, value, code=200, content_type='application/json', filename=None):
            body = json.dumps(value, ensure_ascii=False).encode() if content_type=='application/json' else value
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store' if content_type=='application/json' else 'private, max-age=60')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if self.path.startswith('/api/review/'):
                self.send_header('Access-Control-Allow-Origin',self.headers.get('Origin','*'))
                self.send_header('Vary','Origin')
                self.send_header('Access-Control-Allow-Private-Network','true')
            if filename:
                self.send_header('Content-Disposition', 'attachment; filename="'+filename+'"')
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            if not self.path.startswith('/api/review/'):
                return self.respond({'error':'Unsupported request'},405)
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin',self.headers.get('Origin','*'))
            self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS')
            self.send_header('Access-Control-Allow-Headers','Authorization,Content-Type')
            self.send_header('Access-Control-Allow-Private-Network','true')
            self.end_headers()

        def png(self, image, metadata=None):
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            self.respond(buf.getvalue(), content_type='image/png')

        def do_GET(self):
            try:
                self.get()
            except PermissionError as error:
                self.respond({'error':str(error)},403)
            except (ValueError, KeyError, StopIteration) as error:
                self.respond({'error': str(error) or 'Item not found'}, 400)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.respond({'error': str(error)}, 500)

        def get(self):
            url = urlparse(self.path)
            query = {k:v[0] for k,v in parse_qs(url.query).items()}
            path = url.path
            if path in ('/','/review'):
                boot=dict(server='http://'+self.headers.get('Host','127.0.0.1:8776'),portable=False)
                return self.respond(review.media.html(boot).encode('utf-8'),content_type='text/html; charset=utf-8')
            if path.startswith('/api/review/'):
                return review.get(self,path,query)
            if path.startswith('/api/') and path!='/api/health' and not review.local(self):
                raise PermissionError('Open a portable reviewer file to label from another machine.')
            if path=='/api/revisions':
                return self.respond({c['id']:review.queue.document(c['id'])['revision'] for c in store.catalog})
            if path=='/favicon.ico':
                return self.respond(b'',204,content_type='image/x-icon')
            if path=='/api/documents':
                return self.respond(store.catalog)
            if path=='/api/health':
                return self.respond({'ok':True, 'documents':len(store.catalog)})
            if path.startswith('/api/'):
                doc = store.get(query['doc'])
                if path=='/api/document':
                    return self.respond(doc)
                if path=='/api/export':
                    return self.respond(json.dumps(doc, indent=2, ensure_ascii=False).encode(), content_type='application/json; charset=utf-8', filename=doc['id']+'-review.json')
                if path=='/api/crop':
                    box = list(map(int, query['box'].split(',')))
                    if (box[2]-box[0])*(box[3]-box[1]) > 24000000:
                        raise ValueError('Use a smaller image window')
                    image = images.crop(doc, box)
                    if 'width' in query:
                        width = max(1, min(1600, int(query['width'])))
                        image = image.resize((width, max(1, round(image.height*width/image.width))))
                    return self.png(image)
                if path=='/api/generate' or path=='/api/generation-info':
                    segment = next(s for s in doc['segments'] if s['id']==query['segment'])
                    if not review.worker.policy.allowed(doc,segment):
                        raise ValueError('This source is reserved for testing, needs a label check, or is not in the current training snapshot. It cannot generate training patterns.')
                    from generator.core import generate
                    width = max(16,min(800,int(query.get('width',180))))
                    height = max(16,min(4096,int(query.get('height',480))))
                    image, meta = generate(CropSource(images,doc), segment['box'], width=width,height=height,seed=int(query.get('seed',0)),variation=max(0,min(1,float(query.get('variation',0)))))
                    if path=='/api/generation-info':
                        return self.respond(meta)
                    return self.png(image)
                raise ValueError('Unknown API route')
            if path.startswith('/download/'):
                filename = path.rsplit('/',1)[1]
                if not filename.endswith('.zip') or Path(filename).name != filename:
                    raise ValueError('Invalid dataset name')
                file = store.data / 'exports' / filename
                return self.respond(file.read_bytes(), content_type='application/zip', filename=filename)
            file = (HOME / ('generator/'+path.split('/generator/',1)[1] if path.startswith('/generator/') else 'static/'+('index.html' if path=='/editor' else path.lstrip('/')))).resolve()
            if not file.is_relative_to(HOME / 'static') and not file.is_relative_to(HOME / 'generator'):
                raise ValueError('Invalid file path')
            if not file.is_file():
                return self.respond({'error':'File not found'},404)
            return self.respond(file.read_bytes(), content_type=mimetypes.guess_type(file.name)[0] or 'application/octet-stream')

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                origin = self.headers.get('Origin')
                cross_review=path.startswith('/api/review/') and path not in ('/api/review/start','/api/review/portable')
                if origin and urlparse(origin).netloc != self.headers.get('Host') and not cross_review:
                    return self.respond({'error':'Use this local editor to save changes'},403)
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 5000000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                if path.startswith('/api/review/'):
                    return review.post(self,path,data)
                if not review.local(self):
                    raise PermissionError('Use the assigned reviewer key to save work.')
                if path=='/api/save':
                    revision = store.save(data['document_id'],data['revision'],segment=data.get('segment'),delete_id=data.get('delete_id'),legend=data.get('legend'))
                    return self.respond({'revision':revision})
                if path=='/api/generation-review':
                    doc = store.get(data['document_id'])
                    segment = next((s for s in doc['segments'] if s['id']==data['segment_id']), None)
                    if segment is None:
                        raise ValueError('Unknown source segment')
                    if type(data.get('approved')) is not bool or not isinstance(data.get('comment',''),str):
                        raise ValueError('Invalid sample review')
                    data['source_box'] = segment['box']
                    data['legend_ids'] = segment['legend_ids']
                    data['document_revision'] = doc['revision']
                    data['timestamp'] = time.time()
                    with store.lock:
                        with (store.data / 'generation-reviews.jsonl').open('a',encoding='utf-8') as out:
                            out.write(json.dumps(data,ensure_ascii=False)+'\n')
                    return self.respond({'saved':True})
                if path=='/api/dataset':
                    doc = store.get(data['document_id'])
                    count = int(data.get('count_per_example',3))
                    if not 1 <= count <= 100:
                        raise ValueError('Use 1 to 100 samples per pattern source')
                    examples = [dict(s,document_id=doc['id']) for s in doc['segments'] if s.get('example') and s.get('approved') and s['status']=='approved' and review.worker.policy.allowed(doc,s)]
                    if not examples:
                        raise ValueError('No current training sources are ready. Test sources and examples that need review are excluded. Wait for the background snapshot after accepting a source.')
                    if len(examples)*count > 200:
                        raise ValueError('Export at most 200 samples at one time')
                    from generator.core import create_dataset
                    outdir = store.data / 'exports' / ('patterns-'+doc['id']+'-'+uuid.uuid4().hex[:8])
                    outdir.mkdir(parents=True)
                    manifest = create_dataset(CropSource(images,doc),examples,outdir,count_per_example=count,seed=int(data.get('seed',0)))
                    write(outdir/'document-legend.json',dict(document_id=doc['id'],image_name=doc['image_name'],legend=doc['legend']))
                    archive = outdir.with_suffix('.zip')
                    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
                        for file in outdir.rglob('*'):
                            if file.is_file():
                                zipped.write(file,file.relative_to(outdir))
                    return self.respond({'download_url':'/download/'+archive.name,'url':'/download/'+archive.name,'examples':len(examples),'samples':len(examples)*count,'manifest':manifest})
                raise ValueError('Unknown API route')
            except PermissionError as error:
                self.respond({'error':str(error)},403)
            except Conflict as error:
                self.respond({'error':str(error)},409)
            except (ValueError,KeyError,StopIteration) as error:
                self.respond({'error':str(error) or 'Item not found'},400)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
                pass
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.respond({'error':str(error)},500)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8776)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--share-url',default=None)
    args = parser.parse_args()
    docs = prepare(force=args.prepare)
    if args.prepare:
        print(f'Prepared {len(docs)} documents.',flush=True)
        return
    share_url=args.share_url or f'http://127.0.0.1:{args.port}'
    handler=create_handler(Store(),Images(),share_url)
    server = ThreadingHTTPServer((args.host,args.port),handler)
    share_host=urlparse(share_url).hostname
    if args.host in ('127.0.0.1','localhost') and share_host not in ('127.0.0.1','localhost','::1'):
        shared=ThreadingHTTPServer((share_host,args.port),handler)
        threading.Thread(target=shared.serve_forever,daemon=True).start()
        print(f'Shared review: {share_url}',flush=True)
    print(f'Pattern review: http://127.0.0.1:{args.port} ({len(docs)} documents)',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
