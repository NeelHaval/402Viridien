async page => {
  const check=(v,message)=>{if(!v)throw Error(message);},errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const session=await(await page.request.post('http://127.0.0.1:8777/api/review/start',{data:{name:'QA verification'}})).json();
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('http://127.0.0.1:8777/review?qa='+Date.now()+'#review='+session.token);
  await page.waitForFunction(()=>window.SharedReview?.current()&&document.querySelector('.scan-wrap img')?.naturalWidth>0,null,{timeout:90000});
  const destination=await page.evaluate(()=>{for(const g of SharedReview.state().sets)for(const t of g.tasks){if(t.prediction?.engine==='terra-grid'&&ReviewModel.choice(t,SharedReview.state().answers[t.id]))return {set:g.title,task:t.id};}return null;});
  check(destination,'A clean Terra proposal must be loaded.');
  await page.locator('#sets button').filter({hasText:destination.set}).click();
  await page.locator('.strip-card[data-task="'+destination.task+'"]').click();
  await page.locator('#confirm-guess:not([disabled])').waitFor({timeout:30000});
  check((await page.locator('#prediction-info').textContent()).includes('GPT-6.1 Terra'),'Terra must be named in the reviewer page.');
  const before=await page.evaluate(()=>{const r=SharedReview.state(),t=SharedReview.current(),a=r.answers[t.id];return {task:t.id,ids:[...a.segment.legend_ids],guess:t.prediction.choices[0].legend_ids,box:[...a.segment.box],model:t.prediction.model_version,inputHash:t.prediction.input_hash};});
  check(before.model.startsWith('gpt-6.1-terra:'),'The Terra grid model must be active.');
  check(await page.locator('#held-sets').textContent().then(t=>t.includes('44')&&t.includes('52')),'Known problem sets must be explained.');
  check((await page.locator('.scan-wrap img').evaluateAll(xs=>xs.map(x=>x.getBoundingClientRect().width))).every(w=>w===220),'Every source strip must use 220 display pixels.');
  // A quick double click must confirm one box, not the next unseen box too.
  const base=(await(await page.request.get('http://127.0.0.1:8777/api/review/state?tasks=',{headers:{Authorization:'Bearer '+session.token}})).json()).summary.accepted;
  await page.locator('#confirm-guess').dblclick();
  await page.waitForFunction(()=>!Object.keys(SharedReview.state().pending).length&&!Object.keys(SharedReview.state().inflight).length,null,{timeout:20000});
  const saved=await(await page.request.get('http://127.0.0.1:8777/api/review/state?tasks='+before.task,{headers:{Authorization:'Bearer '+session.token}})).json();
  check(saved.summary.accepted===base+1,'Double-clicking must save one accepted box.');
  const segment=saved.tasks[0].segment;
  check(segment.approved&&JSON.stringify(segment.legend_ids)===JSON.stringify(before.guess),'The explicitly confirmed guess must reach the server.');
  check(segment.review_method==='confirmed_model_guess'&&segment.model_assistance.version===before.model&&segment.model_assistance.input_hash===before.inputHash,'Model-assisted review provenance must be saved.');
  check(JSON.stringify(segment.box)===JSON.stringify(before.box),'Saved source coordinates must match the displayed box.');
  await page.screenshot({path:'output/playwright/terra-grid/desktop.png'});
  // A manual box edit must invalidate the old quick-confirm choice.
  await page.locator('#box-details summary').click();
  const top=await page.locator('#top').inputValue();await page.locator('#top').fill(String(Number(top)+1));await page.locator('#top').press('Tab');
  check(await page.locator('#confirm-guess').isDisabled(),'A moved box must not confirm a stale model guess.');
  await page.setViewportSize({width:390,height:844});
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'The verification controls must fit a phone.');
  await page.locator('#labels').evaluate(e=>e.scrollIntoView({block:'start'}));
  await page.screenshot({path:'output/playwright/terra-grid/phone.png'});
  check(errors.length===0,'Page errors: '+errors.join('; '));
  return {result:'PASS',checks:['terra_grid_guesses','held_set_explanation','equal_strip_width','single_confirm','double_click_guard','server_receipt','model_provenance','source_coordinates','stale_box_guard','phone_width'],pageErrors:errors};
}
