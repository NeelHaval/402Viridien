"""Broad colour hints from source pixels. These are filters, not rock labels."""
import numpy as np

NAMES={'neutral':'White / grey','yellow':'Yellow','orange':'Orange / brown','red':'Red / pink','green':'Green','blue':'Blue / cyan','purple':'Purple','black':'Black'}

def colours(image):
    image=image.convert('RGB');image.thumbnail((160,160))
    rgb=np.asarray(image).reshape(-1,3).astype(float)
    hsv=np.asarray(image.convert('HSV')).reshape(-1,3).astype(float)
    hue=hsv[:,0]*360/255;chroma=rgb.max(1)-rgb.min(1)
    ink=(rgb.max(1)<60)&(chroma<22);neutral=(chroma<30)|(hsv[:,1]<44)
    masks={
        'neutral':neutral&~ink,
        'yellow':(hue>=45)&(hue<75)&~neutral,
        'orange':(hue>=15)&(hue<45)&~neutral,
        'red':((hue<15)|(hue>=325))&~neutral,
        'green':(hue>=75)&(hue<170)&~neutral,
        'blue':(hue>=170)&(hue<270)&~neutral,
        'purple':(hue>=270)&(hue<325)&~neutral,
        'black':ink,
    }
    result=[];visible=max(1,(~ink).sum())
    for key,mask in masks.items():
        threshold=.45*len(rgb) if key=='black' else (.12 if key=='neutral' else .025)*visible
        if mask.sum()>=max(1,threshold):
            result.append(dict(id=key,label=NAMES[key],rgb=np.median(rgb[mask],axis=0).round().astype(int).tolist()))
    return result or [dict(id='unknown',label='Other / unknown',rgb=[150,150,150])]
