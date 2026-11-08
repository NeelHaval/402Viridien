(function(){
'use strict';
const $=id=>document.getElementById(id);
let api;
function status(bg,reviewers){
  if(reviewers)$('reviewer-uploads').textContent=reviewers.map(r=>r.name+': '+(r.last_save?'last upload '+new Date(r.last_save*1000).toLocaleTimeString():'no new upload recorded')).join(' · ');
  if(!bg)return;
  $('background-state').textContent=bg.error?'Background needs a retry. Labels still save.':bg.building?'Background: preparing the next data snapshot…':`Background: ${bg.training_examples} learning sources · ${bg.pending} suggestions waiting`;
  if(bg.dataset)$('dataset-counts').textContent=`${bg.dataset.train} training sources · ${bg.dataset.test} real test samples · ${bg.dataset.needs_review} need a check`;
}
function mount(context){
  api=context;const r=api.state();$('server-address').value=r.server;
  $('device-link').onclick=async()=>{try{await api.flush();if(Object.keys(api.state().pending).length)throw Error('Some changes are still local. Reconnect or export work before switching devices.');const result=await api.request('/api/review/handoff',{method:'POST',body:JSON.stringify({task_id:api.state().taskId})});$('connection-panel').open=true;$('continue-address').value=result.url;$('continue-address').focus();$('continue-address').select();api.notice('Open this private link on the other device. This computer must stay on and be reachable.');}catch(e){api.notice(e.message);}};
  $('copy-device-link').onclick=async()=>{const value=$('continue-address').value;if(!value)return;try{await navigator.clipboard.writeText(value);api.notice('Device link copied.');}catch(e){$('continue-address').focus();$('continue-address').select();api.notice('Copy the selected device link.');}};
  $('reconnect').onclick=async()=>{try{const url=new URL($('server-address').value);if(!['http:','https:'].includes(url.protocol)||url.username||url.password)throw Error('Enter an HTTP or HTTPS server address.');api.state().server=url.origin;api.persist();await api.sync();api.notice('Connection address saved. Check the save status above.');}catch(e){api.notice(e.message);}};
  $('reviewer-update').onclick=async()=>{const b=$('reviewer-update');b.disabled=true;try{await api.flush();if(Object.keys(api.state().pending).length)throw Error('Changes are still local. Use Export work, or reconnect before making this file.');const r=api.state();const response=await fetch(r.server+'/api/review/file',{method:'POST',headers:{'Authorization':'Bearer '+r.session.token,'Content-Type':'application/json'},body:JSON.stringify({sets:5,task_id:api.state().taskId}),signal:AbortSignal.timeout(90000)});if(!response.ok)throw Error((await response.json()).error);api.download(await response.text(),'strata-continue.html','text/html');api.notice('This file keeps the same reviewer and work assignment. Later offline changes stay in that browser; Export work makes a transferable copy.');}catch(e){api.notice(e.message);}finally{b.disabled=false;}};
  $('normalized-export').hidden=!r.host||Boolean(window.REVIEW_BOOT?.portable);
  $('normalized-export').onclick=async()=>{try{const result=await api.request('/api/review/dataset');if(!result.ready)throw Error('The first normalized snapshot is still building. You can continue labelling.');const a=document.createElement('a');a.href=api.state().server+result.url;a.download='';a.click();api.notice(result.background.building?'Downloading the last complete snapshot. A newer snapshot is being prepared.':'Downloading real train/test samples, native crops, normalized images, masks, and generator source records.');}catch(e){api.notice(e.message);}};
}
window.ReviewConnect={mount,status};
})();
