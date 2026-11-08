from audit import *
row=next(r for r in read(OUT/'image-sources.json') if r['class_id']=='7dc28eda27f5c429:23')
canvas=Image.new('RGB',(1000,390),'white');d=ImageDraw.Draw(canvas)
d.text((12,10),'Same class, native pixels enlarged x6; 49 x 48 source pixels per panel',fill='black')
for j,key in enumerate(('real','clean','augmented')):
    a=load(ROOT/'data/genlab/datasets/v5'/row[key])[:48,:49]
    canvas.paste(Image.fromarray(a).resize((294,288),Image.Resampling.NEAREST),(12+j*330,60))
    d.text((12+j*330,35),key+(' | strokes 2 native px' if key=='real' else ' | strokes 1 native px'),fill='black')
canvas.save(OUT/'stroke-width.png')
