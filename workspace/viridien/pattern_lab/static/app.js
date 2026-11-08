'use strict';
const $=id=>document.getElementById(id);
const S={catalog:[],doc:null,selected:null,pattern:null,filter:'all',mode:'select',y:0,span:240,view:null,pending:[],saving:false,failed:false,tab:'editor',token:0,drag:null};
document.addEventListener('error',event=>{
  const image=event.target;
  if(!(image instanceof HTMLImageElement)||!image.getAttribute('src'))return;
  const attempt=Number(image.dataset.retry||0);
  if(attempt>=2)return;
  const failed=image.src;image.dataset.retry=String(attempt+1);
  setTimeout(()=>{if(image.isConnected&&image.src===failed){const url=new URL(failed);url.searchParams.set('retry',String(attempt+1));image.src=url.href;}},150*(attempt+1));
},true);
const clone=value=>JSON.parse(JSON.stringify(value));
function node(tag,attrs={},text){const e=document.createElement(tag);for(const [k,v]of Object.entries(attrs)){if(k==='class')e.className=v;else e.setAttribute(k,v);}if(text!==undefined)e.textContent=text;return e;}
function error(value){$('error').hidden=!value;$('error').textContent=value||'';}
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await r.json();if(!r.ok)throw Error(data.error||`Request failed (${r.status})`);return data;}
function crop(box,width=120){return `/api/crop?doc=${S.doc.id}&box=${box.join(',')}&width=${width}`;}
function selected(){return S.doc?.segments.find(s=>s.id===S.selected);}
function labels(s){return s.legend_ids.map(id=>S.doc.legend.find(e=>e.id===id)?.label_printed||'Unknown label').join(' + ')||s.suggestion||'Choose a label';}
function status(s){return s.status==='approved'?'Accepted':s.status==='rejected'?'Rejected':'Needs review';}
function visible(){return S.doc.segments.filter(s=>(S.pattern===null||s.legend_ids.includes(S.pattern))&&(S.filter==='all'||(S.filter==='examples'?s.example:s.status===S.filter)));}
function saveStatus(){const text=S.failed?'Save failed. Your edits stay in this page.':S.saving||S.pending.length?'Saving changes…':'Saved to local review files';$('save-state').textContent=text;$('save-state').style.color=S.failed?'#9e3945':'';$('retry').hidden=!S.failed;}
function queue(change){S.pending.push({document_id:S.doc.id,doc:S.doc,...clone(change)});processSaves();}
async function processSaves(){if(S.saving||S.failed)return;S.saving=true;saveStatus();try{while(S.pending.length){const op=S.pending[0];const {doc,...body}=op;const result=await api('/api/save',{...body,revision:doc.revision});doc.revision=result.revision;S.pending.shift();}error('');}catch(e){S.failed=true;error(e.message);}finally{S.saving=false;saveStatus();}}
async function flush(){if(S.failed)return false;while(S.saving||S.pending.length){await new Promise(r=>setTimeout(r,80));if(S.failed)return false;}return true;}
function edit(mutator,reset=true){const s=selected();if(!s)return;mutator(s);if(reset){s.approved=false;s.status='suggested';s.example=false;}queue({segment:s});renderLegend();renderQueue();renderInspector();renderOverlay();}
function cropPreviewBox(s,maxHeight=180){const b=[...s.box];if(b[3]-b[1]>maxHeight){const mid=Math.round((b[1]+b[3])/2);b[1]=mid-Math.floor(maxHeight/2);b[3]=b[1]+maxHeight;}return b;}
async function loadDocument(id){const token=++S.token;const doc=await api(`/api/document?doc=${id}`);if(token!==S.token)return;S.doc=doc;S.pattern=null;S.selected=doc.segments.find(s=>s.example)?.id||doc.segments[0]?.id;S.filter='all';$('status-filter').value='all';$('document').value=id;$('doc-meta').textContent=`${doc.legend.length} legend entries · ${doc.source_layer_count?doc.source_layer_count+' supplied layers · ':''}${doc.segments.length} boxes`;$('legend-source').textContent=doc.legend_source;center();renderLegend();renderLegendEditor();renderQueue();renderInspector();renderLab();localStorage.setItem('strata-document',id);saveStatus();}
function renderLegend(){if(!S.doc)return;const q=$('search').value.toLowerCase();const host=$('legend-list');host.replaceChildren();$('pattern-count').textContent=S.doc.legend.length;for(const entry of S.doc.legend){if(!entry.label_printed.toLowerCase().includes(q))continue;const list=S.doc.segments.filter(s=>s.legend_ids.includes(entry.id));const approved=list.filter(s=>s.approved).length;const examples=list.filter(s=>s.example).length;const b=node('button',{class:'legend-item'+(S.pattern===entry.id?' active':''),'aria-pressed':String(S.pattern===entry.id)});const img=node('img',{src:crop(entry.box,80),alt:'',loading:'lazy'});const text=node('span');text.append(node('span',{class:'name'},entry.label_printed),node('small',{},`${list.length} boxes · ${approved} accepted`),node('small',{},examples?`${examples} pattern source${examples===1?'':'s'}`:'No pattern source yet'));b.append(img,text);b.onclick=()=>{S.pattern=entry.id;const first=visible().find(s=>s.example)||visible()[0];if(first)S.selected=first.id;renderLegend();renderQueue();renderInspector();center();renderLab();};host.append(b);}const entry=S.doc.legend.find(e=>e.id===S.pattern);$('view-title').textContent=entry?entry.label_printed:'All document patterns';$('view-subtitle').textContent=entry?'Check the segment edges, then select the correct legend labels.':'Review the saved guesses against this document’s legend.';}
function renderLegendEditor(){const host=$('legend-editor');host.replaceChildren();for(const e of S.doc.legend){const label=node('label',{},`Label ${e.id}`);const input=node('input',{value:e.label_printed,'data-legend-id':e.id,'aria-label':`Name for legend ${e.id}`});label.append(input);host.append(label);}}
function renderQueue(){const list=visible();const host=$('segment-list');host.replaceChildren();$('queue-count').textContent=`${list.length} boxes`;if(!list.length){host.append(node('p',{class:'notice'},'No boxes in this view. Draw a new box, or change the review filter.'));return;}const index=Math.max(0,list.findIndex(s=>s.id===S.selected));const start=Math.max(0,index-10);for(const s of list.slice(start,start+50)){const b=node('button',{class:'segment-card'+(s.id===S.selected?' active':''),'data-segment':s.id});b.append(node('img',{src:crop(cropPreviewBox(s,100),40),alt:'',loading:'lazy'}));const text=node('span');text.append(node('strong',{},labels(s)),node('small',{},`${s.box[1]}–${s.box[3]} px`),node('small',{},status(s)+(s.example?' · Source':'')));b.append(text);b.onclick=()=>select(s.id);host.append(b);}if(list.length>50)host.append(node('p',{class:'hint'},`Showing ${start+1}–${Math.min(start+50,list.length)}. Use Previous and Next to move through all boxes.`));}
function select(id){$('comment').value='';S.selected=id;center();renderQueue();renderInspector();renderLab();}
function renderInspector(){const s=selected();const statusEl=$('segment-status');statusEl.className='badge '+(s?.status||'');statusEl.textContent=s?status(s):'No selection';$('segment-origin').textContent=s?`Source: ${s.provenance}${s.reason?'\n'+s.reason:''}`:'Draw a box to start.';for(const id of ['top','bottom','left','right','accept','reject','use-example','split','delete','add-comment'])$(id).disabled=!s;$('selection-preview').replaceChildren();$('label-choices').replaceChildren();$('comments').replaceChildren();if(!s)return;const [x0,y0,x1,y1]=s.box;$('top').value=y0;$('bottom').value=y1;$('left').value=x0;$('right').value=x1;$('selection-preview').append(node('img',{src:crop(cropPreviewBox(s,100),100),alt:'Selected pattern at its centre'}));$('use-example').checked=Boolean(s.example);$('example-hint').textContent=s.approved?'This source can supply generated samples.':'Accept the box before you use it as a source.';for(const entry of S.doc.legend){const label=node('label');const check=node('input',{type:'checkbox',value:entry.id});check.checked=s.legend_ids.includes(entry.id);check.onchange=()=>edit(seg=>{seg.legend_ids=check.checked?[...seg.legend_ids,entry.id]:seg.legend_ids.filter(i=>i!==entry.id);});label.append(check,node('img',{src:crop(entry.box,40),alt:'',loading:'lazy'}),node('span',{},entry.label_printed));$('label-choices').append(label);}for(const comment of s.comments||[])$('comments').append(node('p',{},comment.text));}
function setMode(mode){S.mode=mode;$('select-mode').classList.toggle('active',mode==='select');$('draw-mode').classList.toggle('active',mode==='draw');$('plane').classList.toggle('draw',mode==='draw');}
function center(){const s=selected();const col=S.doc?.boxes.interpreted_lithology_column;if(!col)return;S.y=s?Math.round((s.box[1]+s.box[3]-S.span)/2):col[1];renderView();}
const TILES={key:null,items:[],frame:0,layout:null};
const TILE_ROWS=1024;
function loadVisibleTiles(){
  if(!TILES.layout||!S.view)return;
  const {bounds,scale}=TILES.layout;
  const first=Math.max(0,Math.floor((S.view[1]-bounds[1])/TILE_ROWS)-1);
  const last=Math.min(TILES.items.length-1,Math.ceil((S.view[3]-bounds[1])/TILE_ROWS)+1);
  const current=Math.max(0,Math.min(TILES.items.length-1,Math.floor((S.view[1]-bounds[1])/TILE_ROWS)));
  const old=$('scan');if(old)old.removeAttribute('id');
  for(let i=0;i<TILES.items.length;i++){
    const tile=TILES.items[i];
    if(i>=first&&i<=last){
      if(!tile.image.getAttribute('src'))tile.image.src=tile.url;
    }else if(i<first-2||i>last+2){
      tile.image.removeAttribute('src');
    }
    if(i===current)tile.image.id='scan';
  }
  $('image-loading').hidden=Boolean($('scan')?.naturalWidth);
}
function updateScrollView(){
  if(!S.doc||!TILES.layout||S.tab!=='editor')return;
  const {bounds,scale,rows}=TILES.layout;
  S.y=Math.round(bounds[1]+$('viewer').scrollTop/scale);
  S.span=rows;
  S.view=[bounds[0],S.y,bounds[2],Math.min(bounds[3],Math.ceil(S.y+rows))];
  $('position').value=S.y;
  $('coordinates').textContent=`Rows ${S.y.toLocaleString()}–${S.view[3].toLocaleString()} · Column x ${S.doc.boxes.interpreted_lithology_column[0]}–${S.doc.boxes.interpreted_lithology_column[2]} px`;
  const ruler=$('ruler');ruler.replaceChildren();
  const anchors=S.doc.depth_calibration?.anchors||[];
  for(let i=0;i<=8;i++){
    const row=Math.round(S.y+rows*i/8);
    if(row>bounds[3])continue;
    let value=row.toLocaleString();
    if(anchors.length>=2){const a=anchors[0],b=anchors[1];value=Math.round(a.depth_ft+(row-a.y)*(b.depth_ft-a.depth_ft)/(b.y-a.y)).toLocaleString()+' ft';}
    const text=node('span',{},value);text.style.top=`${i*12.5}%`;ruler.append(text);
  }
  loadVisibleTiles();renderOverlay();
}
function renderView(){
  if(!S.doc||S.tab!=='editor')return;
  const viewer=$('viewer');
  const layout=StrataViewport.layout(S.doc,viewer.clientWidth,viewer.clientHeight,Number($('window-size').value),$('context').checked);
  const {bounds,scale,rows}=layout;
  const target=Math.max(bounds[1],Math.min(S.y,bounds[3]-rows));
  const key=S.doc.id+':'+bounds.join(',');
  const plane=$('plane');
  plane.style.width=`${layout.width}px`;plane.style.height=`${layout.height}px`;
  TILES.layout=layout;S.scale=scale;S.span=rows;
  if(key!==TILES.key){
    TILES.key=key;TILES.items=[];$('tiles').replaceChildren();
    const documentId=S.doc.id;
    for(let top=bounds[1];top<bounds[3];top+=TILE_ROWS){
      const bottom=Math.min(bounds[3],top+TILE_ROWS);
      const frame=node('div',{class:'scan-tile'});
      const image=node('img',{alt:'Source document strip',draggable:'false'});
      const box=[bounds[0],top,bounds[2],bottom];
      const item={frame,image,top,bottom,url:`/api/crop?doc=${documentId}&box=${box.join(',')}`};
      image.fetchPriority='high';image.onload=()=>{if(image.id==='scan')$('image-loading').hidden=true;};
      image.onerror=()=>{if(image.id==='scan'){$('image-loading').textContent='This strip did not load. Move the view to retry.';$('image-loading').hidden=false;}};
      frame.append(image);$('tiles').append(frame);TILES.items.push(item);
    }
  }
  for(const tile of TILES.items){tile.frame.style.top=`${(tile.top-bounds[1])*scale}px`;tile.frame.style.height=`${(tile.bottom-tile.top)*scale}px`;}
  $('position').min=bounds[1];$('position').max=Math.max(bounds[1],Math.floor(bounds[3]-rows));
  viewer.scrollTop=(target-bounds[1])*scale;
  updateScrollView();
}
$('viewer').addEventListener('scroll',()=>{
  if(TILES.frame)return;
  TILES.frame=requestAnimationFrame(()=>{TILES.frame=0;updateScrollView();});
},{passive:true});
function svg(tag,attrs){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e;}
function renderOverlay(){if(!S.view||!S.doc)return;const host=$('overlay');const [vx,vy,vr,vb]=S.view;const bounds=TILES.layout.bounds;host.setAttribute('viewBox',`${bounds[0]} ${bounds[1]} ${bounds[2]-bounds[0]} ${bounds[3]-bounds[1]}`);host.replaceChildren();const scale=TILES.layout.scale;for(const s of S.doc.segments){const [x0,y0,x1,y1]=s.box;if(y1<vy||y0>vb||x1<vx||x0>vr)continue;const chosen=s.id===S.selected;const color=chosen?'#135cc7':s.status==='approved'?'#24835f':s.status==='rejected'?'#af4257':'#b7761e';const rect=svg('rect',{x:x0,y:y0,width:x1-x0,height:y1-y0,fill:chosen?'#1d6fe518':'transparent',stroke:color,'stroke-width':chosen?2.5:1,'vector-effect':'non-scaling-stroke','data-id':s.id,'data-edge':'move',cursor:'pointer'});host.append(rect);if(chosen){const size=8/scale;for(const[edge,x,y]of[['top',(x0+x1)/2,y0],['bottom',(x0+x1)/2,y1],['left',x0,(y0+y1)/2],['right',x1,(y0+y1)/2]]){const h=svg('rect',{x:x-size/2,y:y-size/2,width:size,height:size,rx:1/scale,fill:'#fff',stroke:color,'stroke-width':2,'vector-effect':'non-scaling-stroke','data-id':s.id,'data-edge':edge,cursor:edge==='top'||edge==='bottom'?'ns-resize':'ew-resize'});host.append(h);}}}if(S.drag?.draft){const b=S.drag.draft;host.append(svg('rect',{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:'#1d6fe522',stroke:'#135cc7','stroke-width':2,'stroke-dasharray':'5 3','vector-effect':'non-scaling-stroke'}));}}
function point(event){
  const transform=$('overlay').getScreenCTM();
  const source=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());
  return [Math.round(source.x),Math.round(source.y)];
}
$('overlay').addEventListener('pointerdown',event=>{if(!S.doc||event.button!==0)return;event.preventDefault();const p=point(event);if(S.mode==='draw')S.drag={type:'draw',start:p,draft:[...p,...p]};else{const id=event.target.dataset.id;if(!id)return;S.selected=id;renderInspector();renderQueue();const s=selected();S.drag={type:event.target.dataset.edge||'move',start:p,original:[...s.box],changed:false};}event.currentTarget.setPointerCapture(event.pointerId);renderOverlay();});
$('overlay').addEventListener('pointermove',event=>{const d=S.drag;if(!d)return;const p=point(event),w=S.doc.width,h=S.doc.height;if(d.type==='draw'){d.draft=[Math.max(0,Math.min(d.start[0],p[0])),Math.max(0,Math.min(d.start[1],p[1])),Math.min(w,Math.max(d.start[0],p[0])),Math.min(h,Math.max(d.start[1],p[1]))];}else{const b=[...d.original],dx=p[0]-d.start[0],dy=p[1]-d.start[1];if(d.type==='top')b[1]=Math.max(0,Math.min(b[3]-1,b[1]+dy));if(d.type==='bottom')b[3]=Math.min(h,Math.max(b[1]+1,b[3]+dy));if(d.type==='left')b[0]=Math.max(0,Math.min(b[2]-1,b[0]+dx));if(d.type==='right')b[2]=Math.min(w,Math.max(b[0]+1,b[2]+dx));if(d.type==='move'){const mx=Math.max(-b[0],Math.min(dx,w-b[2])),my=Math.max(-b[1],Math.min(dy,h-b[3]));b[0]+=mx;b[2]+=mx;b[1]+=my;b[3]+=my;}selected().box=b;d.changed=b.some((v,i)=>v!==d.original[i]);}renderOverlay();});
function finishDrag(event){const d=S.drag;if(!d)return;S.drag=null;if(d.type==='draw'){if(d.draft[2]-d.draft[0]>=2&&d.draft[3]-d.draft[1]>=2){const s={id:crypto.randomUUID(),box:d.draft,legend_ids:S.pattern===null?[]:[S.pattern],status:'suggested',approved:false,example:false,provenance:'manual selection',comments:[]};S.doc.segments.push(s);S.selected=s.id;queue({segment:s});setMode('select');}}else if(d.changed){edit(()=>{});}renderLegend();renderQueue();renderInspector();renderOverlay();renderLab();}
$('overlay').addEventListener('pointerup',finishDrag);$('overlay').addEventListener('pointercancel',()=>{if(S.drag?.original&&selected())selected().box=S.drag.original;S.drag=null;renderOverlay();renderInspector();});
function moveView(delta){if(TILES.layout)$('viewer').scrollTop+=delta*TILES.layout.scale;}
$('position').oninput=event=>{if(TILES.layout)$('viewer').scrollTop=(Number(event.target.value)-TILES.layout.bounds[1])*TILES.layout.scale;};
$('window-size').value='240';$('window-size').onchange=()=>{S.span=Number($('window-size').value);center();};$('context').onchange=renderView;$('center').onclick=center;
$('find-top').onclick=()=>{if(selected()){S.y=selected().box[1]-S.span/4;renderView();}};$('find-bottom').onclick=()=>{if(selected()){S.y=selected().box[3]-S.span*3/4;renderView();}};
$('select-mode').onclick=()=>setMode('select');$('draw-mode').onclick=()=>setMode('draw');
$('search').oninput=renderLegend;$('all-patterns').onclick=()=>{S.pattern=null;renderLegend();renderQueue();renderLab();};
$('status-filter').onchange=event=>{S.filter=event.target.value;renderQueue();};
function navigate(delta){const list=visible();if(!list.length)return;const current=list.findIndex(s=>s.id===S.selected);select(list[(Math.max(0,current)+delta+list.length)%list.length].id);}
$('previous').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);
$('box-form').onsubmit=event=>{event.preventDefault();const box=[Number($('left').value),Number($('top').value),Number($('right').value),Number($('bottom').value)];if(box.some(v=>!Number.isInteger(v))||box[0]<0||box[1]<0||box[2]<=box[0]||box[3]<=box[1]||box[2]>S.doc.width||box[3]>S.doc.height){error('Enter a box inside the source image. Bottom and right must be larger than top and left.');return;}edit(s=>{s.box=box;});center();renderLab();};
$('accept').onclick=()=>{const s=selected();if(!s?.legend_ids.length){error('Choose at least one legend label before you accept the box.');return;}edit(s=>{s.status='approved';s.approved=true;if(s.queue_pattern_key)s.example=true;},false);renderLab();};
$('reject').onclick=()=>{edit(s=>{s.status='rejected';s.approved=false;s.example=false;},false);renderLab();};
$('use-example').onchange=event=>{const s=selected();if(!s?.approved){event.target.checked=false;error('Accept this segment first.');return;}edit(s=>{s.example=event.target.checked;},false);renderLab();};
$('delete').onclick=()=>{const s=selected();if(!s)return;S.doc.segments=S.doc.segments.filter(v=>v.id!==s.id);queue({delete_id:s.id});S.selected=visible()[0]?.id;renderLegend();renderQueue();renderInspector();renderOverlay();renderLab();};
$('split').onclick=()=>{const s=selected();if(!s||s.box[3]-s.box[1]<4){error('This box is too short to split.');return;}const lower=clone(s),mid=Math.floor((s.box[1]+s.box[3])/2);lower.id=crypto.randomUUID();lower.box[1]=mid;lower.status='suggested';lower.approved=false;lower.example=false;lower.provenance='manual split';lower.comments=[];edit(seg=>{seg.box[3]=mid;});S.doc.segments.push(lower);queue({segment:lower});renderQueue();renderOverlay();};
$('add-comment').onclick=()=>{const text=$('comment').value.trim();if(!text||!selected())return;edit(s=>{s.comments.push({id:crypto.randomUUID(),text,created_at:new Date().toISOString()});},false);$('comment').value='';};
$('save-legend').onclick=()=>{const legend=clone(S.doc.legend);for(const input of $('legend-editor').querySelectorAll('input')){const name=input.value.trim();if(!name){error('Enter a name for each legend entry.');return;}legend.find(e=>e.id===Number(input.dataset.legendId)).label_printed=name;}S.doc.legend=legend;queue({legend});renderLegend();renderInspector();renderQueue();renderLab();};
$('retry').onclick=()=>{S.failed=false;processSaves();};
$('document').onchange=async event=>{const id=event.target.value;if(!await flush()){event.target.value=S.doc.id;return;}try{await loadDocument(id);}catch(e){error(e.message);event.target.value=S.doc.id;}};
$('export').onclick=async()=>{if(!S.doc)return;const file=new Blob([JSON.stringify(S.doc,null,2)],{type:'application/json'});const a=node('a',{href:URL.createObjectURL(file),download:S.doc.id+'-review.json'});a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
$('dataset').onclick=async()=>{if(!await flush())return;const b=$('dataset');b.disabled=true;b.textContent='Building dataset…';try{const result=await api('/api/dataset',{document_id:S.doc.id,count_per_example:3,seed:0});const a=node('a',{href:result.url,download:''});a.click();error('');$('save-state').textContent=`Dataset: ${result.samples} samples from ${result.examples} sources`;}catch(e){error(e.message);}finally{b.disabled=false;b.textContent='Export pattern dataset';}};
async function renderLab(){if(S.tab!=='lab')return;if(!await flush())return;if(S.tab!=='lab')return;const host=$('pattern-lab');if(!selected()){host.replaceChildren(node('p',{class:'notice'},'Select a segment to compare its source pattern.'));return;}if(window.PatternLab){window.PatternLab.mount(host,{document:S.doc,segment:selected()});}else{host.replaceChildren(node('p',{class:'notice'},'The pattern lab module is loading. Select Pattern lab again to retry.'));}}
function tab(value){S.tab=value;$('editor-view').hidden=value!=='editor';$('lab-view').hidden=value!=='lab';$('editor-tab').classList.toggle('active',value==='editor');$('lab-tab').classList.toggle('active',value==='lab');if(value==='lab')renderLab();else renderView();}
$('editor-tab').onclick=()=>tab('editor');$('lab-tab').onclick=async()=>{if(await flush())tab('lab');};
$('help-toggle').onclick=()=>$('help').showModal();$('help-close').onclick=()=>$('help').close();
document.addEventListener('keydown',event=>{if(/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)||$('help').open)return;if(event.key===']')navigate(1);if(event.key==='[')navigate(-1);if(event.key==='Escape')setMode('select');});
window.addEventListener('beforeunload',event=>{if(S.pending.length||S.failed){event.preventDefault();event.returnValue='';}});let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(S.tab==='editor')renderView();},200);});
(async()=>{try{S.catalog=await api('/api/documents');for(const doc of S.catalog)$('document').append(node('option',{value:doc.id},(doc.supplied?'✓ ':'')+doc.name));const saved=new URLSearchParams(location.search).get('doc')||localStorage.getItem('strata-document');await loadDocument(S.catalog.some(d=>d.id===saved)?saved:S.catalog[0].id);const requested=new URLSearchParams(location.search).get('segment');if(requested&&S.doc.segments.some(s=>s.id===requested))select(requested);}catch(e){error(e.message);$('save-state').textContent='Could not open the review data';}})();
window.Strata={state:S,flush,select,renderView,renderOverlay,tiles:TILES,moveView};

setInterval(async()=>{if(!S.doc||S.pending.length||S.failed||S.drag||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;try{const revisions=await api('/api/revisions');if(revisions[S.doc.id]!==S.doc.revision){const fresh=await api('/api/document?doc='+S.doc.id);if(S.pending.length||S.drag)return;S.doc=fresh;renderLegend();renderQueue();renderInspector();renderOverlay();saveStatus();}}catch(e){/* A reconnect will fetch the next revision. */}},5000);
