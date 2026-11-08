"""Pillow-compatible region facade. No full-page RGB allocation for TIFF."""
from PIL import Image
from app.tools.read_upload import read_upload

class RegionImage:
    def __init__(self,path):
        self.scan=read_upload(str(path))
        self.width,self.height=self.scan.width,self.scan.height
        self.size=(self.width,self.height)
        self.mode='RGB'
    def crop(self,box):
        x0,y0,x1,y1=map(int,box)
        if (x1-x0)*(y1-y0)>32_000_000:
            raise ValueError('Requested image region exceeds 32 megapixels')
        return Image.fromarray(self.scan.reader.crop(y0,y1,x0,x1)).convert('RGB')
    def thumbnail_image(self,size):
        scale=min(1,size[0]/self.width,size[1]/self.height)
        return Image.fromarray(self.scan.reader.crop(0,self.height,scale=scale)).convert('RGB')
    def close(self):self.scan.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
