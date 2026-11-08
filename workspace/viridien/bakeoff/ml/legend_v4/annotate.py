"""Manual development legend labels from the saved, non-eval image crops.

Coordinates refer to visible printed swatch frames, not parser detections.
"""
import json
from harness import io
from ml.legend_v4.experiment import HERE,domain_names,family_map

def save(family,boxes,labels):
    name=next(n for n in domain_names() if family_map()[n]==family)
    log=io.open_log(name);dx,dy=log.boxes['lithology_legend'][:2];log.close()
    if family == "Texas 'Rock Types' box":
        boxes=[(324,107,334,120) if t=='COAL' else b for b,t in zip(boxes,labels)]
    entries=[dict(id=i,box=[a+dx,b+dy,c+dx,d+dy],label_printed=t) for i,((a,b,c,d),t) in enumerate(zip(boxes,labels))]
    assert len(boxes)==len(labels)
    folder=HERE/'manual_truth';folder.mkdir(exist_ok=True)
    (folder/(name+'.json')).write_text(json.dumps(dict(log=name,source='Manual read of permitted non-eval legend image',legend=entries),indent=2))

def grid(xs,ys,w,h):return [(x,y,x+w,y+h) for y in ys for x in xs]
save('D-582 1-bit grid',grid([74,327,579,831],[122,354],170,120)+grid([74,327],[621],170,118),
     ['Limestone','Dolomite','Anhydrite','Halite','Coal','Clay','Shale','Siltstone','Silty Sandstone','Sand and Sandstone'])
# This exporter has flattened the sections: the first three rows are rocks.
save('Gulf 3-panel',grid([39,514,989,1463,1938],[52,157,261],95,72),
     ['Siltstone','Sandstone','Halite','Shale','Clay','Limestone','Dolomite','Marl','Dolomitic limestone','Chalk','Gypsum','Anhydrite','Chert','Volcanics','Granite'])
save('MC761 mineral/lithology',grid([506,666],[85,136,187,238,289],38,36)+[(506,340,544,376),(666,34,704,70)],
     ['Sandstone','Marlstone','Siltstone','Salt','Shale','Breccia','Conglomerate','Claystone','Limestone','No Sample','Dolomite','Anhydrite'])
# No Sample is a printed non-rock cell. Keep it for exact-name checks, and flag it.
save('Texas \'Rock Types\' box',grid([51],[123,140,157,174,191,208,225,242],90,13)+grid([324],[107,123,140,157,174,191,208,225,242],90,13)+grid([623],[107,123,140,157,174,191,208,225,242],90,13)+grid([903],[107,123,140,157,174,191,208],90,13),
 ['ANHYDRITE','BENTONITE','BRECCIA','CEMENT','CHALK','CHERT','CLAY CHOKE SAND','CLAYSTONE','COAL','CONGLOMERATE','DOLOMITE','DOLOMITIC LIMESTONE','GRANITE','GYPSUM','IGNEOUS','LIMESTONE','SIDERITE or LIMONITE','MARLSTONE','METAMORPHIC','NO SAMPLE','SALT','SALT- PEPPER SAND','SANDSTONE','SHALE','SHALE COLORED','SHALE GRAY','SHALY SANDSTONE','SHALY SILTSTONE','SILTSTONE','SILTY SHALE','TILL','TUFF','WELDED TUFF'])
save('MC765 colour grid',grid([32,346,661,976,1291],[4,74,145,215],64,48)+grid([32,346,661],[285],64,48),
 ['Clay-S','Clay-SCS','Clay-M','Clay-CCS','Clay-SACS','Sand-SIS','Sand-SS','Sand-FS','Sand-MS','Sand-CS','Sand-FSS','Sand-MSS','Limestone-AL','Limestone-SL','Limestone-L','Dolomites-AD','Dolomites-CD','Dolomites-D','Others-H','Others-A','Others-GY','Others-E','Others-CL'])
save('One-off layout',grid([8,307],[51,111,171,232,292,352,412,472],75,38),
 ['SHALE','CONGLOMERATE','SILTSTONE','COAL','SANDSTONE','SALT','LIMESTONE','VOLCANICS','DOLOMITE','GRANITE','ANHYDRITE','METAMORPHIC','BENTONITE','TUFF','CHERT','IGNEOUS'])
