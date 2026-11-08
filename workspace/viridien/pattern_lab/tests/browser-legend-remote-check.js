async page=>{
  const check=(v,m)=>{if(!v)throw Error(m);},errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.setViewportSize({width:1600,height:1000});await page.reload();
  await page.waitForFunction(()=>window.SharedReview?.current()&&document.getElementById('legend-search'),null,{timeout:120000});
  await page.waitForFunction(()=>!document.getElementById('more-work').disabled);
  for(const value of await page.locator('#labels input:checked').evaluateAll(xs=>xs.map(x=>x.value)))await page.locator('#labels input[value="'+value+'"]').uncheck();
  const choices=page.locator('#labels input');check(await choices.count()>2,'Need multiple real legend choices.');
  await choices.nth(0).check();await choices.nth(1).check();
  const selected=await page.evaluate(()=>[...SharedReview.state().answers[SharedReview.current().id].segment.legend_ids]);
  const colour=await page.locator('.legend-colours button:not([data-colour="all"])').first().getAttribute('data-colour');
  await page.locator('.legend-colours button[data-colour="'+colour+'"]').click();
  const filtered=await page.evaluate(()=>({visible:[...document.querySelectorAll('#labels .legend-row')].filter(r=>!r.hidden).length,total:document.querySelectorAll('#labels .legend-row').length,selected:SharedReview.state().answers[SharedReview.current().id].segment.legend_ids}));
  check(filtered.visible>0&&filtered.visible<=filtered.total,'Colour filtering must keep matching choices.');
  check(JSON.stringify(filtered.selected)===JSON.stringify(selected),'Colour filtering must keep selections.');
  await page.locator('#legend-search').fill('no-such-lithology-12345');check(await page.locator('#legend-no-results').isVisible(),'Empty search needs a recovery message.');
  check(await page.locator('#legend-selected button').count()===selected.length,'Selected labels must stay visible with no matching rows.');
  await page.locator('#legend-clear').click();check(await page.locator('#labels .legend-row:not([hidden])').count()===filtered.total,'Clear must restore all choices.');
  const before=await page.locator('#labels input:checked').count();await page.locator('#labels .legend-row button').first().click();await page.keyboard.press('Escape');
  check(await page.locator('#labels input:checked').count()===before,'Enlarge must preserve selected labels.');
  const target=page.locator('.strip-card').nth(1);const targetId=await target.getAttribute('data-task');await target.focus();await page.keyboard.press('Space');
  check(await page.evaluate(id=>document.activeElement?.dataset.task===id&&SharedReview.current().id===id,targetId),'Keyboard selection must retain focus.');
  await page.locator('#legend-search').scrollIntoViewIfNeeded();await page.screenshot({path:'C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/pattern_lab/data/ux-audit/legend-desktop.png'});
  const download=page.waitForEvent('download',{timeout:120000});await page.locator('#portable').click();const file=await download;
  const filePath='C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/pattern_lab/data/ux-audit/remote-reviewer-check.html';await file.saveAs(filePath);
  const context=await page.context().browser().newContext({viewport:{width:1440,height:1000},hasTouch:true,permissions:['local-network-access']});
  const remote=await context.newPage();remote.on('pageerror',e=>errors.push(e.message));await remote.goto('file:///'+filePath);
  await remote.waitForFunction(()=>window.SharedReview?.current()&&document.getElementById('legend-search'),null,{timeout:120000});
  const saved=async(id,text,approved)=>remote.waitForFunction(async({id,text,approved})=>{const r=SharedReview.state();try{const data=await(await fetch(r.server+'/api/review/state?tasks='+id,{headers:{Authorization:'Bearer '+r.session.token}})).json();const s=data.tasks[0]?.segment;return Boolean(s&&(text===null||s.comments.some(c=>c.text===text))&&(approved===undefined||s.approved===approved));}catch(e){return false;}},{id,text,approved},{timeout:20000});
  const id=await remote.evaluate(()=>SharedReview.current().id);const start=Date.now();
  await remote.locator('#comment').fill('Automatic file-to-Wi-Fi save.');await saved(id,'Automatic file-to-Wi-Fi save.');const automaticSaveMs=Date.now()-start;
  // Commit a request, lose its response, then type a newer draft before retry.
  let first=true,committed;const commit=new Promise(r=>committed=r);const ops=[];
  await context.route('**/api/review/save',async route=>{ops.push(JSON.parse(route.request().postData()).op_id);if(first){first=false;await route.fetch();committed();await new Promise(r=>setTimeout(r,600));await route.abort();}else await route.continue();});
  await remote.locator('#comment').fill('First committed request.');await commit;await remote.locator('#comment').fill('Newer draft after a lost response.');
  await saved(id,'Newer draft after a lost response.');await remote.waitForFunction(()=>!Object.keys(SharedReview.state().pending).length&&!Object.keys(SharedReview.state().inflight).length);
  check(ops.length>=3&&ops[0]===ops[1]&&ops[1]!==ops[2],'Lost responses must retry the same operation before sending the newer draft.');
  check(await remote.evaluate(()=>!Object.keys(SharedReview.state().conflicts).length),'Lost responses must recover without a false conflict.');await context.unroute('**/api/review/save');
  // Delay old state until after a new draft has saved.
  let firstState=true,stateCaptured,releaseState;const captured=new Promise(r=>stateCaptured=r),release=new Promise(r=>releaseState=r);
  await context.route('**/api/review/state?*',async route=>{if(firstState){firstState=false;const response=await route.fetch();stateCaptured();await release;await route.fulfill({response});}else await route.continue();});
  await remote.evaluate(()=>{window.delayedSync=SharedReview.sync();});await captured;
  await remote.locator('#comment').fill('Saved after an older state request.');await saved(id,'Saved after an older state request.');await remote.waitForFunction(()=>!Object.keys(SharedReview.state().pending).length);await remote.locator('#source-name').click();releaseState();await remote.evaluate(()=>window.delayedSync);
  check(await remote.locator('#comment').inputValue()==='Saved after an older state request.','Stale state must not replace a newer saved draft.');await context.unroute('**/api/review/state?*');
  // A true remote conflict must retain a second local edit and permit reapply.
  await remote.evaluate(async id=>{const r=SharedReview.state();const headers={Authorization:'Bearer '+r.session.token,'Content-Type':'application/json'};const data=await(await fetch(r.server+'/api/review/state?tasks='+id,{headers})).json();const t=data.tasks[0];t.segment.comments=[{id:'other-device',text:'Another device changed this box.'}];await fetch(r.server+'/api/review/save',{method:'POST',headers,body:JSON.stringify({task_id:id,op_id:crypto.randomUUID(),version:t.version,action:'draft',segment:t.segment})});},id);
  let conflictReceived,releaseConflict;const received=new Promise(r=>conflictReceived=r),conflictGate=new Promise(r=>releaseConflict=r);let hold=true;
  await context.route('**/api/review/save',async route=>{if(hold){hold=false;const response=await route.fetch();conflictReceived();await conflictGate;await route.fulfill({response});}else await route.continue();});
  await remote.locator('#comment').fill('Earlier local conflict draft.');await received;await remote.locator('#comment').fill('Latest local conflict draft.');releaseConflict();
  await remote.getByRole('button',{name:'Keep my draft',exact:true}).waitFor();await remote.getByRole('button',{name:'Keep my draft',exact:true}).click();await saved(id,'Latest local conflict draft.');
  check(await remote.evaluate(()=>SharedReview.state().history.some(h=>h.operation.segment.comments.some(c=>c.text==='Earlier local conflict draft.')&&h.pending.segment.comments.some(c=>c.text==='Latest local conflict draft.'))),'Both conflict drafts must remain in history.');await context.unroute('**/api/review/save');
  // Loaded sets must not starve an explicit fresh-work request.
  const oldKeys=await remote.evaluate(()=>SharedReview.state().sets.map(g=>g.key));const refillStart=Date.now();await remote.evaluate(()=>SharedReview.more(true));
  const refill=await remote.evaluate(keys=>({newKeys:SharedReview.state().sets.filter(g=>!keys.includes(g.key)).length,unique: new Set(SharedReview.state().sets.flatMap(g=>g.tasks.map(t=>t.id))).size,total:SharedReview.state().sets.reduce((n,g)=>n+g.tasks.length,0)}),oldKeys);
  check(refill.newKeys>0&&refill.unique===refill.total,'Fresh work must add a new set without duplicate tasks.');const refillMs=Date.now()-refillStart;
  const group=await remote.evaluate(()=>SharedReview.state().setKey),advanceStart=Date.now();
  for(let n=0;n<5;n++){const task=await remote.evaluate(()=>SharedReview.current().id);if(!await remote.locator('#labels input:checked').count())await remote.locator('#labels input').first().check();await remote.locator('#accept').click();await saved(task,null,true);}
  check(await remote.evaluate(key=>SharedReview.state().setKey!==key,group),'Five accepts must open the next prefetched set.');const fiveAcceptsMs=Date.now()-advanceStart;
  // Synthesized touch and cancelled pointer drag at phone width.
  await remote.waitForLoadState('networkidle');await remote.setViewportSize({width:390,height:844});const handle=remote.locator('svg[data-task] rect[data-edge="bottom"]').first();await handle.scrollIntoViewIfNeeded().catch(()=>handle.scrollIntoViewIfNeeded());const bounds=await handle.boundingBox();
  const original=await remote.evaluate(()=>[...SharedReview.state().answers[SharedReview.current().id].segment.box]);const cdp=await context.newCDPSession(remote);
  const touch=(x,y)=>[{x,y,radiusX:3,radiusY:3,force:1}];const x=bounds.x+bounds.width/2,y=bounds.y+bounds.height/2;
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:touch(x,y)});await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:touch(x,y+12)});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
  const changed=await remote.evaluate(()=>SharedReview.state().answers[SharedReview.current().id].segment.box);check(changed[3]>original[3]&&changed[1]===original[1],'Touch drag must resize the bottom only.');const nextHandle=await remote.locator('svg[data-task] rect[data-edge="bottom"]').first().boundingBox();const cancelX=nextHandle.x+nextHandle.width/2,cancelY=nextHandle.y+nextHandle.height/2;await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:touch(cancelX,cancelY)});await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:touch(cancelX,cancelY+8)});await cdp.send('Input.dispatchTouchEvent',{type:'touchCancel',touchPoints:[]});await cdp.detach();
  check(await remote.evaluate(b=>JSON.stringify(SharedReview.state().answers[SharedReview.current().id].segment.box)===JSON.stringify(b),changed),'Cancelled drag must keep the prior box.');
  await remote.locator('#legend-search').evaluate(e=>e.scrollIntoView({block:'start'}));check(await remote.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Phone layout must fit.');await remote.screenshot({path:'C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/pattern_lab/data/ux-audit/legend-phone.png'});
  // No explicit flush or export: offline work must upload after reconnect and reload.
  const offlineId=await remote.evaluate(()=>SharedReview.current().id);await context.setOffline(true);await remote.locator('#comment').fill('Offline automatic retry after reload.');await remote.reload();await remote.waitForFunction(()=>window.SharedReview?.current());await context.setOffline(false);await saved(offlineId,'Offline automatic retry after reload.');
  const failContext=await page.context().browser().newContext();const fail=await failContext.newPage();await fail.addInitScript(()=>{indexedDB.open=()=>{throw Error('Offline storage denied for test');};});await fail.goto('http://127.0.0.1:8777');await fail.waitForFunction(()=>window.SharedReview?.current()&&!document.getElementById('more-work').disabled,null,{timeout:120000});
  check((await fail.locator('#pack-state').textContent()).includes('Offline image storage failed'),'Cache failure must not claim offline readiness.');await failContext.close();await context.close();check(errors.length===0,'Page errors: '+errors.join('; '));
  return {result:'PASS',colourFiltering:true,selectionPreserved:true,keyboardFocus:true,automaticFileToWifiSave:true,automaticSaveMs,lostResponseRecovered:true,newestConflictDraftSaved:true,staleStateProtected:true,newWorkRefill:true,refillMs,fiveAcceptsMs,touchDrag:true,offlineAutomaticRetry:true,cacheFailureVisible:true,phoneWidth:390,errors};
}
