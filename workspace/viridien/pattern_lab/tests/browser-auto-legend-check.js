async page=>{
  const check=(v,m)=>{if(!v)throw Error(m);},errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.setViewportSize({width:1500,height:1000});await page.reload();
  await page.waitForFunction(()=>window.SharedReview?.current()&&document.querySelector('#automatic-choices input'),null,{timeout:120000});
  const original=await page.evaluate(()=>{const t=SharedReview.current(),a=SharedReview.state().answers[t.id];return{task:t.id,box:[...a.segment.box],ids:[...a.segment.legend_ids],guess:t.prediction.choices[0].legend_ids.find(id=>t.legend.some(e=>e.id===id&&!e.deprecated))};});
  check(await page.locator('#automatic-choices input').first().inputValue()===String(original.guess),'The best guess must be first.');
  check(await page.evaluate(()=>document.getElementById('legend-automatic').compareDocumentPosition(document.getElementById('legend-filters'))&Node.DOCUMENT_POSITION_FOLLOWING),'Automatic suggestions must be above manual filters.');
  check(await page.evaluate(ids=>JSON.stringify(SharedReview.state().answers[SharedReview.current().id].segment.legend_ids)===JSON.stringify(ids),original.ids),'Showing suggestions must not choose labels.');
  const autoIds=await page.locator('#automatic-choices input').evaluateAll(xs=>xs.map(x=>x.value));
  await page.locator('#legend-search').fill('no-such-label-457');
  check(JSON.stringify(await page.locator('#automatic-choices input').evaluateAll(xs=>xs.map(x=>x.value)))===JSON.stringify(autoIds),'Manual filters must not hide the automatic section.');
  const input=page.locator('#automatic-choices input').first(),id=await input.inputValue(),checked=await input.isChecked();if(checked)await input.uncheck();else await input.check();
  check(await page.locator('#labels input[value="'+id+'"]').isChecked()===!checked,'Suggested selection must update its full-legend checkbox.');
  await page.waitForFunction(async({task,id,checked})=>{const r=SharedReview.state();const d=await(await fetch(r.server+'/api/review/state?tasks='+task,{headers:{Authorization:'Bearer '+r.session.token}})).json();return d.tasks[0].segment.legend_ids.includes(Number(id))===checked;},{task:original.task,id,checked:!checked});
  await page.locator('#legend-clear').click();
  await page.locator('#automatic-choices .legend-row button').first().click();await page.keyboard.press('Escape');
  check(await page.locator('#labels input[value="'+id+'"]').isChecked()===!checked,'Enlarging a suggested swatch must keep the chosen labels.');
  // Check the real coordinate-control path before a deterministic colour fixture.
  await page.locator('#box-details summary').click();await page.locator('#bottom').fill(String(original.box[3]-1));await page.locator('#bottom').press('Tab');
  await page.waitForFunction(()=>document.getElementById('legend-automatic').dataset.box===SharedReview.state().answers[SharedReview.current().id].segment.box.join(','));
  await page.locator('#bottom').fill(String(original.box[3]));await page.locator('#bottom').press('Tab');await page.evaluate(()=>SharedReview.flush());
  await page.waitForFunction(()=>!Object.keys(SharedReview.state().pending).length&&!Object.keys(SharedReview.state().inflight).length);
  // The full source contains blue and yellow, but only the current selection must count.
  await page.evaluate(()=>{const t=SharedReview.current(),a=SharedReview.state().answers[t.id],c=t.context_box;window.fixtureBackup={image:t.image,box:[...a.segment.box]};const image=document.createElement('canvas');image.width=c[2]-c[0];image.height=c[3]-c[1];const ctx=image.getContext('2d');ctx.fillStyle='blue';ctx.fillRect(0,0,image.width,image.height);ctx.fillStyle='yellow';ctx.fillRect(0,Math.floor(image.height/2),image.width,image.height);t.image=image.toDataURL();a.segment.box=[c[0],c[1]+1,c[2],c[1]+Math.floor(image.height/2)-1];ReviewDetails.refreshBox(t,a);});
  await page.waitForFunction(()=>document.querySelector('#box-colours [data-colour="blue"]')&&!document.querySelector('#box-colours [data-colour="yellow"]'));
  await page.evaluate(()=>{const t=SharedReview.current(),a=SharedReview.state().answers[t.id],c=t.context_box;a.segment.box=[c[0],c[1]+Math.floor((c[3]-c[1])/2)+1,c[2],c[3]-1];ReviewDetails.refreshBox(t,a);});
  await page.waitForFunction(()=>document.querySelector('#box-colours [data-colour="yellow"]')&&!document.querySelector('#box-colours [data-colour="blue"]'));
  await page.evaluate(()=>{const t=SharedReview.current(),a=SharedReview.state().answers[t.id];t.image=fixtureBackup.image;a.segment.box=fixtureBackup.box;ReviewDetails.refreshBox(t,a);});
  await page.locator('.strip-card[data-task="'+original.task+'"]').click();
  await page.waitForFunction(()=>document.querySelector('#automatic-choices input'));
  await page.locator('#box-details summary').click();
  await page.locator('#legend-automatic').evaluate(e=>e.scrollIntoView({block:'start'}));
  await page.screenshot({path:'C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/pattern_lab/data/ux-audit/auto-desktop.png'});
  await page.setViewportSize({width:390,height:844});await page.locator('#legend-automatic').evaluate(e=>e.scrollIntoView({block:'start'}));
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'The automatic list must fit a phone.');
  check(await page.locator('#automatic-choices button').first().evaluate(e=>e.getBoundingClientRect().height>=44),'Enlarge must retain a 44 pixel touch target.');
  await page.screenshot({path:'C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/pattern_lab/data/ux-audit/auto-phone.png'});
  // Open a new person's reviewer link in a fresh browser, without a local HTML file.
  const url=await page.evaluate(async()=>{const post=async(path,body,token)=>{const headers={'Content-Type':'application/json'};if(token)headers.Authorization='Bearer '+token;return (await fetch('/api/review/'+path,{method:'POST',headers,body:JSON.stringify(body)})).json();};const session=await post('start',{name:'Direct browser link test'});return(await post('handoff',{},session.token)).url;});
  const context=await page.context().browser().newContext({viewport:{width:390,height:844}}),remote=await context.newPage();remote.on('pageerror',e=>errors.push(e.message));
  await context.route('**/api/review/session',async route=>{const response=await route.fetch(),body=await response.json();body.host=false;await route.fulfill({response,json:body});});
  await remote.goto(url);await remote.waitForFunction(()=>window.SharedReview?.current()&&document.querySelector('#automatic-choices input'),null,{timeout:120000});
  check(await remote.locator('#editor-link').isHidden(),'A remote reviewer must not see host-only editor controls.');
  await remote.locator('#comment').fill('Direct browser link saves without HTML.');
  await remote.waitForFunction(async()=>{const r=SharedReview.state(),t=SharedReview.current();const d=await(await fetch(r.server+'/api/review/state?tasks='+t.id,{headers:{Authorization:'Bearer '+r.session.token}})).json();return d.tasks[0].segment.comments.some(c=>c.text==='Direct browser link saves without HTML.');});
  check(await remote.evaluate(()=>!location.hash),'The private sign-in fragment must be removed after login.');await context.close();check(errors.length===0,'Page errors: '+errors.join('; '));
  return{result:'PASS',automaticSectionAboveFilters:true,bestGuessFirst:true,selectedBoxColoursOnly:true,updatesAfterResize:true,manualFiltersIndependent:true,selectionSynced:true,automaticSave:true,phoneWidth:390,directReviewerLink:true,hostControlsHidden:true,errors};
}
