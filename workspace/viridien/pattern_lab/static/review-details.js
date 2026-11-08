(function(){
'use strict';

const colourNames={neutral:'White / grey',yellow:'Yellow',orange:'Orange / brown',red:'Red / pink',green:'Green',blue:'Blue / cyan',purple:'Purple',black:'Black'};
function boxColours(pixels){
  const buckets=Object.fromEntries(Object.keys(colourNames).map(k=>[k,[]]));let total=0,ink=0;
  for(let i=0;i<pixels.length;i+=4){
    if(pixels[i+3]<128)continue;total++;
    const rgb=[pixels[i],pixels[i+1],pixels[i+2]],hi=Math.max(...rgb),lo=Math.min(...rgb),chroma=hi-lo;
    let key;
    if(hi<60&&chroma<22){key='black';ink++;}
    else if(chroma<30||chroma/Math.max(1,hi)*255<44)key='neutral';
    else{let hue=hi===rgb[0]?((rgb[1]-rgb[2])/chroma)%6:hi===rgb[1]?(rgb[2]-rgb[0])/chroma+2:(rgb[0]-rgb[1])/chroma+4;hue=(hue*60+360)%360;
      key=hue<15||hue>=325?'red':hue<45?'orange':hue<75?'yellow':hue<170?'green':hue<270?'blue':'purple';}
    buckets[key].push(rgb);
  }
  const result=[],visible=Math.max(1,total-ink);
  for(const[key,values]of Object.entries(buckets)){
    const threshold=key==='black'?.45*total:(key==='neutral'?.12:.025)*visible;
    if(values.length<Math.max(1,threshold))continue;
    const rgb=[0,1,2].map(c=>{const channel=values.map(v=>v[c]).sort((a,b)=>a-b);const m=Math.floor(channel.length/2);return Math.round(channel.length%2?channel[m]:(channel[m-1]+channel[m])/2);});
    result.push({id:key,label:colourNames[key],rgb});
  }
  return result;
}
function suggestEntries(legend,colours,prediction,limit=8){
  const entries=legend.filter(e=>!e.deprecated),chromatic=colours.filter(c=>!['neutral','black','unknown'].includes(c.id)),used=chromatic.length?chromatic:colours.filter(c=>c.id!=='unknown');
  const matches=e=>(e.colours||[]).some(c=>used.some(v=>v.id===c.id));
  const distance=e=>Math.min(Infinity,...(e.colours||[]).flatMap(c=>used.filter(v=>v.id===c.id).map(v=>Math.hypot(...c.rgb.map((n,i)=>n-v.rgb[i])))));
  const colourMatches=entries.filter(matches).sort((a,b)=>distance(a)-distance(b)||a.label_printed.localeCompare(b.label_printed));
  const result=[],seen=new Set();
  function add(entry,kind){if(!entry||seen.has(entry.id)||result.length>=limit)return;seen.add(entry.id);result.push({entry,kind,matchesColour:matches(entry)});}
  for(const[index,choice]of (prediction?.choices||[]).entries())for(const id of choice.legend_ids||[])add(entries.find(e=>e.id===id),index===0?'best':'pattern');
  for(const e of colourMatches)add(e,'colour');
  return {entries:result,colourMatchCount:colourMatches.length,colours:used};
}
function applyBedProposal(task,proposal){
  if(!proposal||proposal.document_id!==task.document_id||proposal.source_version!==task.version||task.segment.approved||task.segment.status==='rejected')return task;
  const allowed=new Map(task.legend.filter(e=>!e.deprecated).map(e=>[e.id,e]));
  const prediction=JSON.parse(JSON.stringify(proposal.prediction));
  prediction.choices=prediction.choices.filter(c=>c.legend_ids.length===1&&allowed.has(c.legend_ids[0])).map(c=>({...c,reference:{...c.reference,image:allowed.get(c.legend_ids[0]).image}}));
  if(!prediction.choices.length){task.verification_warning=prediction.warning||'No v4 model choice maps to this legend.';return task;}
  task.prediction=prediction;task.segment={...task.segment,suggestion:prediction.model_label||prediction.choices[0].label};
  const b=proposal.box,c=task.context_box;
  if(task.version==='new'&&b?.length===4&&b.every(Number.isInteger)&&b[0]>=c[0]&&b[1]>=c[1]&&b[2]<=c[2]&&b[3]<=c[3]&&b[0]<b[2]&&b[1]<b[3])task.segment.box=[...b];
  task.verification={source_version:task.version,box:[...task.segment.box],bed:proposal.bed,model_version:prediction.model_version};
  return task;
}
function confirmableChoice(task,answer){
  const p=task?.prediction,choice=p?.choices?.[0],v=task?.verification;
  if(!['v4-beds','terra-grid'].includes(p?.engine)||p.ambiguous||!v||v.source_version!==answer?.version||answer.segment.approved||answer.segment.status==='rejected'||choice?.legend_ids?.length!==1)return null;
  if(p.engine==='terra-grid'&&(p.terra_status!=='match'||p.terra_confidence!=='high'))return null;
  if(v.box.join(',')!==answer.segment.box.join(','))return null;
  const id=choice.legend_ids[0];if(!task.legend.some(e=>e.id===id&&!e.deprecated))return null;
  if(answer.segment.legend_ids.length&&answer.segment.legend_ids.join(',')!==String(id))return null;
  return choice;
}
if(typeof module!=='undefined'&&module.exports){module.exports={boxColours,suggestEntries,applyBedProposal,confirmableChoice};return;}
window.ReviewModel={apply:applyBedProposal,choice:confirmableChoice};

const $=id=>document.getElementById(id);
function el(tag,text,cls){const e=document.createElement(tag);if(text)e.textContent=text;if(cls)e.className=cls;return e;}
let dialog,view,title,zoom,image,full,current;
function enlarge(entry,task){
  if(!dialog){
    dialog=el('dialog',null,'legend-dialog');title=el('h2');const close=el('button','Close');close.onclick=()=>dialog.close();
    const heading=el('div',null,'legend-dialog-heading');heading.append(title,close);
    const controls=el('div',null,'legend-dialog-controls');const label=el('label','Zoom ');zoom=el('input');zoom.type='range';zoom.min='1';zoom.max='4';zoom.step='.25';zoom.setAttribute('aria-label','Legend zoom');label.append(zoom);
    full=el('button','Full source legend');full.onclick=()=>show(null,current);controls.append(label,full);
    view=el('div',null,'legend-large-view');image=el('img');image.alt='Enlarged source legend';view.append(image);dialog.append(heading,controls,view);document.body.append(dialog);
    zoom.oninput=()=>{image.style.width=Math.round(image.naturalWidth*Number(zoom.value))+'px';};
    image.onload=()=>{const base=title.dataset.kind==='entry'?Math.max(2,Math.min(4,360/image.naturalWidth)):1;zoom.value=String(base);image.style.width=Math.round(image.naturalWidth*base)+'px';};
    dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  }
  function show(item,source){current=source;title.textContent=item?item.label_printed:source.document_name+' · source legend';title.dataset.kind=item?'entry':'source';full.hidden=!item||!source.legend_overview;image.src=item?item.image:source.legend_overview;}
  if(!entry&&!task.legend_overview)return;
  show(entry,task);if(!dialog.open)dialog.showModal();
}
let filter={document:null,text:'',colour:'all'},refreshSelection=()=>{},activeAuto=null,autoFrame=null;
const imageCache=new Map();
function sourceImage(src){if(!imageCache.has(src)){const ready=new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(Error('Source image unavailable'));img.src=src;});imageCache.set(src,ready);if(imageCache.size>3)imageCache.delete(imageCache.keys().next().value);}return imageCache.get(src);}
function refreshBox(task,answer){window.SharedReview?.refreshConfirm?.();if(activeAuto?.task.id!==task.id)return;activeAuto.answer=answer;if(autoFrame!==null)return;autoFrame=requestAnimationFrame(()=>{autoFrame=null;if(activeAuto?.task.id===task.id)updateAutomatic(activeAuto);});}
function syncAutomatic(){if(!activeAuto)return;for(const input of activeAuto.list.querySelectorAll('input')){input.checked=activeAuto.answer.segment.legend_ids.includes(Number(input.value));input.closest('.legend-row').classList.toggle('is-selected',input.checked);}}
function showAutomatic(view,colours,error=''){if(activeAuto!==view)return;const ranked=suggestEntries(view.task.legend,colours,view.task.prediction);view.host.dataset.box=view.boxKey;view.palette.replaceChildren();for(const c of colours){const chip=el('span',null,'box-colour');chip.dataset.colour=c.id;const dot=el('span',null,'colour-dot');dot.style.backgroundColor='rgb('+c.rgb.join(',')+')';chip.append(dot,document.createTextNode(c.label));view.palette.append(chip);}
view.note.textContent=error||(!colours.length?'No box colours found. Best pattern matches are shown.':ranked.colourMatchCount?ranked.entries.length+' suggestions · '+ranked.colourMatchCount+' colour matches.':'No legend colours match this box. Best pattern matches are shown.');view.list.replaceChildren();
for(const item of ranked.entries){const e=item.entry,row=el('div',null,'legend-row automatic-row'),label=el('label'),input=el('input');input.type='checkbox';input.value=String(e.id);input.onchange=()=>{const main=$('labels').querySelector('input[value="'+e.id+'"]');if(main&&main.checked!==input.checked)main.click();};const img=el('img');img.src=e.image;img.alt='';const text=el('span');text.append(el('span',e.label_printed,'automatic-label'),el('small',item.kind==='best'?'Best guess':item.kind==='pattern'?'Next pattern match':'Matches box colour'));if(item.kind!=='colour'&&colours.length&&!item.matchesColour)text.append(el('small','Colour differs — check this guess','quality-warning'));label.append(input,img,text);const zoom=el('button','Enlarge');zoom.type='button';zoom.setAttribute('aria-label','Enlarge suggested '+e.label_printed);zoom.onclick=()=>enlarge(e,view.task);row.append(label,zoom);view.list.append(row);}
if(!ranked.entries.length)view.list.append(el('p','Use the full legend below to choose a label.'));syncAutomatic();}
async function updateAutomatic(view){const box=[...view.answer.segment.box],key=box.join(','),source=view.task.image;if(view.boxKey===key&&view.source===source)return;view.boxKey=key;view.source=source;const context=[...view.task.context_box];try{const img=await sourceImage(source);if(activeAuto!==view||view.boxKey!==key||view.source!==source)return;const left=Math.max(0,box[0]-context[0]),top=Math.max(0,box[1]-context[1]),width=Math.min(context[2]-context[0]-left,box[2]-box[0]),height=Math.min(context[3]-context[1]-top,box[3]-box[1]);if(width<=0||height<=0)throw Error('Empty crop');const canvas=document.createElement('canvas'),scale=Math.min(1,160/Math.max(width,height));canvas.width=Math.max(1,Math.round(width*scale));canvas.height=Math.max(1,Math.round(height*scale));const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;ctx.drawImage(img,left,top,width,height,0,0,canvas.width,canvas.height);showAutomatic(view,boxColours(ctx.getImageData(0,0,canvas.width,canvas.height).data));}catch(e){if(activeAuto===view&&view.boxKey===key)showAutomatic(view,[],'Could not read box colours. Best pattern matches are shown.');}}
function clear(){activeAuto=null;for(const id of ['legend-automatic','full-legend-title','legend-view-tools','legend-filters','prediction-info'])if($(id))$(id).hidden=true;}
function render(task,answer,choose){
  for(const id of ['legend-automatic','full-legend-title','legend-view-tools','legend-filters','prediction-info'])if($(id))$(id).hidden=false;
  if(filter.document!==task.document_id)filter={document:task.document_id,text:'',colour:'all'};
  let toolbar=$('legend-view-tools');if(!toolbar){toolbar=el('div',null,'legend-view-tools');toolbar.id='legend-view-tools';$('labels').before(toolbar);}
  toolbar.replaceChildren();const all=el('button','Expand source legend');all.disabled=!task.legend_overview;all.onclick=()=>enlarge(null,task);toolbar.append(all);
  const update=el('button','Update guesses');update.onclick=()=>window.SharedReview.improve();toolbar.append(update);
  if(task.legend_warning)toolbar.append(el('p',task.legend_warning,'quality-warning'));
  let automatic=$('legend-automatic');if(!automatic){automatic=el('section',null,'legend-automatic');automatic.id='legend-automatic';automatic.setAttribute('aria-labelledby','legend-automatic-title');toolbar.before(automatic);}
  automatic.replaceChildren();const title=el('h4','Suggested for this box');title.id='legend-automatic-title';automatic.append(title,el('p','Based on the selected box colours and the current pattern guesses. Choose labels to select them.','automatic-help'));
  const boxPalette=el('div',null,'box-colours');boxPalette.id='box-colours';boxPalette.setAttribute('aria-label','Colours in the selected box');const note=el('p','Reading box colours…','automatic-note');note.id='automatic-note';const list=el('div');list.id='automatic-choices';automatic.append(boxPalette,note,list);
  activeAuto={task,answer,host:automatic,palette:boxPalette,note,list,boxKey:null};updateAutomatic(activeAuto);
  let fullHeading=$('full-legend-title');if(!fullHeading){fullHeading=el('h4','All legend labels');fullHeading.id='full-legend-title';toolbar.before(fullHeading);}

  for(const entry of task.legend.filter(e=>!e.deprecated||answer.segment.legend_ids.includes(e.id))){
    const input=[...$('labels').querySelectorAll('input')].find(e=>e.value===String(entry.id));if(!input)continue;
    const label=input.closest('label');const row=el('div',null,'legend-row');label.before(row);row.append(label);
    const button=el('button','Enlarge');button.type='button';button.setAttribute('aria-label','Enlarge '+entry.label_printed);button.onclick=()=>enlarge(entry,task);row.append(button);
    if(entry.deprecated)row.append(el('p','Old merged choice. Select a separate corrected entry.','quality-warning'));
  }
  let filters=$('legend-filters');if(!filters){filters=el('div',null,'legend-filters');filters.id='legend-filters';$('labels').before(filters);}
  filters.replaceChildren();const entries=task.legend.filter(e=>!e.deprecated||answer.segment.legend_ids.includes(e.id));
  const searchLabel=el('label','Find a label');const search=el('input');search.type='search';search.id='legend-search';search.placeholder='Search names';search.value=filter.text;searchLabel.append(search);filters.append(searchLabel);
  const palette=el('div',null,'legend-colours');palette.setAttribute('role','group');palette.setAttribute('aria-label','Filter legend by colour');
  const groups=new Map();for(const e of entries)for(const c of e.colours||[{id:'unknown',label:'Other / unknown',rgb:[150,150,150]}]){if(!groups.has(c.id))groups.set(c.id,{...c,count:0});groups.get(c.id).count++;}
  const buttons=[];function colourButton(id,label,rgb){const b=el('button');b.type='button';b.dataset.colour=id;b.setAttribute('aria-pressed',String(filter.colour===id));if(rgb){const dot=el('span',null,'colour-dot');dot.style.backgroundColor='rgb('+rgb.join(',')+')';dot.setAttribute('aria-hidden','true');b.append(dot);}b.append(document.createTextNode(label));b.onclick=()=>{filter.colour=id;apply();};palette.append(b);buttons.push(b);}
  colourButton('all','All colours');for(const c of groups.values())colourButton(c.id,c.label+' ('+c.count+')',c.rgb);filters.append(palette);
  const summary=el('p');summary.id='legend-result-count';summary.setAttribute('role','status');summary.setAttribute('aria-live','polite');const clear=el('button','Clear filters');clear.type='button';clear.id='legend-clear';clear.onclick=()=>{filter.text='';filter.colour='all';search.value='';apply();search.focus();};
  const result=el('div',null,'legend-filter-result');result.append(summary,clear);filters.append(result);
  const selected=el('div',null,'legend-selected');selected.id='legend-selected';selected.setAttribute('aria-label','Selected legend labels');filters.append(selected);
  const empty=el('p','No choices match. Clear the filters to see all labels.','legend-no-results');empty.id='legend-no-results';$('labels').append(empty);
  function apply(){
    const query=filter.text.trim().toLocaleLowerCase();let shown=0;
    for(const e of entries){const row=$('labels').querySelector('input[value="'+e.id+'"]').closest('.legend-row');const checked=answer.segment.legend_ids.includes(e.id);row.classList.toggle('is-selected',checked);row.hidden=!(e.label_printed.toLocaleLowerCase().includes(query)&&(filter.colour==='all'||(e.colours||[{id:'unknown'}]).some(c=>c.id===filter.colour)));if(!row.hidden)shown++;}
    for(const b of buttons)b.setAttribute('aria-pressed',String(b.dataset.colour===filter.colour));
    summary.textContent=shown+' of '+entries.length+' choices · '+answer.segment.legend_ids.length+' selected';clear.hidden=!query&&filter.colour==='all';empty.hidden=shown>0;selected.replaceChildren();
    for(const e of entries.filter(e=>answer.segment.legend_ids.includes(e.id))){const b=el('button',e.label_printed+' ×');b.type='button';b.setAttribute('aria-label','Remove '+e.label_printed);b.onclick=()=>{$('labels').querySelector('input[value="'+e.id+'"]').click();};selected.append(b);}
    syncAutomatic();
  }
  search.oninput=()=>{filter.text=search.value;apply();};refreshSelection=apply;apply();
  let panel=$('prediction-info');if(!panel){panel=el('div',null,'prediction-info');panel.id='prediction-info';automatic.append(panel);}
  panel.replaceChildren();const p=task.prediction;if(!p){panel.append(el('p','Update guesses to compare this crop with accepted examples.'));return;}
  panel.append(el('strong',p.status==='close'?'Closest visual match':'Suggestion needs a check'));
  if(task.verification_warning)panel.append(el('p',task.verification_warning,'quality-warning'));
  if(p.warning)panel.append(el('p',p.warning,'quality-warning'));
  panel.append(el('p',p.engine==='terra-grid'?`GPT-6.1 Terra · ${p.terra_confidence} model confidence · ${p.model_agreement?'same label as the bed model':'check against the source legend'}. Confidence is the model assessment, not measured accuracy.`:p.engine==='v4-beds'?`v4 bed model · ${p.confidence===null?'unscored':Math.round(p.confidence*100)+'% model score'} · ${p.family}. This score is a suggestion, not a measured accuracy.`:`${p.training_examples} accepted examples used. These are suggestions; acceptance stays manual.`));
  const ref=p.choices?.[0]?.reference;if(ref?.image){const details=el('details');details.append(el('summary',ref.kind==='legend'?'Compare with the source legend':'Compare with the accepted example'));const img=el('img');img.src=ref.image;img.alt=ref.label+' accepted source example';details.append(el('p',ref.document_name+' · '+ref.label),img);panel.append(details);}
}
window.ReviewDetails={render,enlarge,refreshSelection:()=>refreshSelection(),refreshBox,clear};
})();
