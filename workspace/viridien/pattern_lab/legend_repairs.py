"""Source-checked legend repairs. Old IDs remain available for saved annotations."""
import copy

CALLAWAY='Callaway 3-33 mudlog to TD.PDF'
REVISION='callaway-rows-v1'

def callaway_entries():
    # Native source pixels, checked against the full Rock Types panel.
    columns=[(77,168,1,['Anhydrite','Bentonite','Breccia','Cement','Chalk','Chert','Clay choke sand','Claystone']),
             (350,441,0,['Coal','Conglomerate','Dolomite','Dolomitic limestone','Granite','Gypsum','Igneous','Limestone','Siderite or limonite']),
             (649,740,0,['Marlstone','Metamorphic',None,'Salt','Salt-pepper sand','Sandstone','Shale','Shale colored','Shale gray']),
             (930,1020,0,['Shaly sandstone','Shaly siltstone','Siltstone','Silty shale','Till','Tuff','Welded tuff'])]
    result=[]
    for left,right,start,names in columns:
        for offset,name in enumerate(names):
            if name is None:continue
            row=start+offset;top=2185+round(row*16.65)
            result.append(dict(id=1000+len(result),label_printed=name,box=[left,top,right,top+14],source_checked=True,repair=REVISION))
    return result

def install_repairs(store):
    """Append corrected IDs, retire merged choices, and preserve every saved box."""
    changed=[]
    for meta in store.catalog:
        if meta.get('name')!=CALLAWAY:continue
        doc=store.get(meta['id'])
        if any(e.get('repair')==REVISION for e in doc['legend']):continue
        # Do not replace a legend that another reviewer has already corrected.
        if len(doc['legend'])!=5 or not any('CEMENT CHALK'==e['label_printed'] for e in doc['legend']):continue
        old=copy.deepcopy(doc['legend'])
        for entry in old:entry.update(deprecated=True,quality_warning='Old merged legend row. Choose a separate source-checked entry.')
        store.save(doc['id'],doc['revision'],legend=old+callaway_entries());changed.append(doc['id'])
    return changed

def legend_warning(legend):
    bad=[e for e in legend if not e.get('deprecated') and (len(e['label_printed'].split())>6 or len(e['label_printed'])>65)]
    return 'Some machine legend choices may contain several rows. Check the full source legend.' if bad else ''
