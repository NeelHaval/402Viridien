from PIL import Image
import sys
im=Image.open(sys.argv[1]);im.thumbnail((1024,1024));im.save(sys.argv[2])
