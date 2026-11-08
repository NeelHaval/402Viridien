import json
from PIL import Image, ImageDraw
D='analysis/legend_families2/'
G=json.load(open(D+'groups.json',encoding='utf-8'))['groups']
H=26; rows=[]
for g in G:
    ims=[Image.open(D+it['file']) for it in g['items'][:26]]
    ims=[i.resize((max(1,i.width*H//i.height),H)) for i in ims]
    r=Image.new('RGB',(1500,H+2),'white'); d=ImageDraw.Draw(r)
    d.text((2,2),f"{g['id']} {g['colour'][:16]} n={len(g['items'])}",fill='black'); x=150
    for i in ims: r.paste(i.crop((0,0,min(i.width,50),H)),(x,1)); x+=min(i.width,50)+3
    rows.append(r)
half=(len(rows)+1)//2
for part,rs in ((1,rows[:half]),(2,rows[half:])):
    s=Image.new('RGB',(1500,sum(r.height for r in rs)),'white'); y=0
    for r in rs: s.paste(r,(0,y)); y+=r.height
    s.save(f'notes/legend2_overview{part}.png')
print(len(G), 'singletons', sum(len(g['items'])==1 for g in G))
