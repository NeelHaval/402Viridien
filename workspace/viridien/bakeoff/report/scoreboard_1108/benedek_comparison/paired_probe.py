import importlib.metadata as m
for k in ("easyocr","torch","torchvision","opencv-python-headless","scikit-image","numpy","pandas","python-bidi","pyclipper","Shapely"):
 try: print(k,m.version(k))
 except m.PackageNotFoundError: print(k,"MISSING")
