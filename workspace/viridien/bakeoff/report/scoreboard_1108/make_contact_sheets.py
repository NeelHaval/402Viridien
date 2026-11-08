"""Create bounded labelled sheets for the final diagnostic figures."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent

def font(size):
    try:return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    except OSError:return ImageFont.load_default()

def sheet(items,path,columns,cell):
    rows=(len(items)+columns-1)//columns
    canvas=Image.new('RGB',(columns*cell[0],rows*cell[1]),'white');draw=ImageDraw.Draw(canvas)
    for index,item in enumerate(items):
        x=index%columns*cell[0];y=index//columns*cell[1]
        name=item['strategy']+' / '+str(item.get('rank',item.get('grade','comparison')))
        draw.text((x+7,y+5),name,fill='#193849',font=font(12))
        draw.text((x+7,y+23),str(item.get('log','log06'))[:36],fill='#405c6c',font=font(10))
        render=item['render']
        if render['status']=='ok':
            img=Image.open(HERE/render['path']).convert('RGB');img.thumbnail((cell[0]-14,cell[1]-46))
            canvas.paste(img,(x+(cell[0]-img.width)//2,y+43))
        else:draw.text((x+7,y+58),'RENDER '+render['status'],fill='#a12b20',font=font(14))
        draw.rectangle((x,y,x+cell[0]-1,y+cell[1]-1),outline='#b8cbd5')
    if max(canvas.size)>1024:canvas.thumbnail((1024,1024))
    canvas.save(path)
    return dict(path=path.name,width=canvas.width,height=canvas.height,items=len(items))

def main():
    data=json.loads((HERE/'report_data.json').read_text(encoding='utf-8'))
    worst=sorted(data['worst'],key=lambda item:(item['rank'],item['strategy']))
    results=[sheet(worst,HERE/'worst_contact_sheet.png',4,(256,333))]
    extra=data['grade_comparison_views']+data['class_identity_fix']['log06_views']
    results.append(sheet(extra,HERE/'grade_contact_sheet.png',2,(512,333)))
    (HERE/'visual_review_sheets.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Contact sheets:',json.dumps(results))
if __name__=='__main__':main()
