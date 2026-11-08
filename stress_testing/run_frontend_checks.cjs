/* Offline browser contracts against a built frontend and mocked API.
 * node stress_testing/run_frontend_checks.cjs --playwright-module <module-or-path>
 * Optional --browser-executable <Chrome path>. No external target is accepted.
 */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const ROOT = path.resolve(__dirname, '..');
function option(name) { const index = process.argv.indexOf(name); return index < 0 ? undefined : process.argv[index + 1]; }
const playwright = require(option('--playwright-module') || 'playwright');
const dist = path.join(ROOT, 'demo/frontend/dist');
if (!fs.existsSync(path.join(dist, 'index.html'))) throw Error('Build demo/frontend first');
const output = path.resolve(option('--output') || path.join(__dirname, '.runs'),
  `frontend-${new Date().toISOString().replace(/[:.]/g, '-')}-${crypto.randomBytes(4).toString('hex')}`);
fs.mkdirSync(output, {recursive:true});
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jXioAAAAASUVORK5CYII=', 'base64');
const sample = (id='one', name='Test well', status='ready') => ({id,name,status,lat:27.5,lon:-90,block:'GC 100',
  located:true,scores:status==='ready'?{drilling:40,cost:50,price:60,co2:70}:null,validation_confidence:0.9});
const detail = well => ({...well,total_depth_ft:100,column:[],validation:{confidence:0.9,components:{},flags:[]},warnings:[],error:null});
const answer = (route, value, status=200) => route.fulfill({status,contentType:'application/json',body:JSON.stringify(value)});
const results=[]; let externalIntercepts=0;
const fingerprint = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const sourceFiles=['demo/frontend/src/App.tsx','demo/frontend/src/api.ts','demo/frontend/package-lock.json'];
const sources=Object.fromEntries(sourceFiles.map(file=>[file,fingerprint(path.join(ROOT,file))]));
const assets={};
function assetHashes(directory) {
  for(const item of fs.readdirSync(directory,{withFileTypes:true})) {
    const file=path.join(directory,item.name);
    if(item.isDirectory())assetHashes(file);else assets[path.relative(dist,file).replace(/\\/g,'/')]=fingerprint(file);
  }
}
assetHashes(dist);
let browser, origin;
const server=http.createServer((req,res)=>{
  try {
    const relative=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    const file=path.resolve(dist,'.'+(relative==='/'?'/index.html':relative));
    if (!file.startsWith(dist+path.sep) || !fs.statSync(file).isFile()) {res.writeHead(404);res.end();return;}
    const ext=path.extname(file);res.setHeader('content-type',({'.js':'application/javascript','.css':'text/css','.html':'text/html','.png':'image/png'})[ext]||'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  } catch {res.writeHead(404);res.end();}
});
async function scenario(name, api, check) {
  const context=await browser.newContext({viewport:{width:1100,height:800},reducedMotion:'reduce'});
  const page=await context.newPage();page.setDefaultTimeout(7000);
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/*', async route=>{
    const url=new URL(route.request().url());
    if (url.origin===origin && url.pathname.startsWith('/api/')) return api(route,url,page);
    if (url.origin!==origin) {
      externalIntercepts++;
      if (url.pathname.endsWith('/style.json')) return answer(route,{version:8,sources:{},layers:[]});
      return route.abort();
    }
    return route.continue();
  });
  const started=Date.now();
  try {
    await page.goto(origin,{waitUntil:'domcontentloaded'});
    await check(page);
    assert.deepEqual(errors,[], 'Uncaught browser errors');
    results.push({name,passed:true,seconds:(Date.now()-started)/1000});
    if (name==='upload completes') await page.screenshot({path:path.join(output,'upload-complete.png')});
  } catch(error) {
    results.push({name,passed:false,error:String(error),browser_errors:errors,seconds:(Date.now()-started)/1000});
    await page.screenshot({path:path.join(output,name.replace(/\W/g,'-')+'-failure.png')}).catch(()=>{});
  } finally {await context.close();}
}
async function upload(page) {
  await page.getByRole('button',{name:'Upload well',exact:true}).click();
  await page.locator('input[type=file]').first().setInputFiles({name:'tiny.png',mimeType:'image/png',buffer:png});
  await page.getByRole('button',{name:'Upload',exact:true}).click();
}
async function main() {
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));origin=`http://127.0.0.1:${server.address().port}`;
  browser=await playwright.chromium.launch({headless:true,executablePath:option('--browser-executable'),
    args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  await scenario('empty dialog and keyboard',r=>answer(r,[]),async page=>{
    await page.getByText('No wells yet. Upload a scan to put one on the map.').waitFor();
    await page.getByRole('button',{name:'Upload well',exact:true}).click();
    assert.equal(await page.getByRole('button',{name:'Upload',exact:true}).isDisabled(),true);
    await page.keyboard.press('Escape');await page.getByRole('dialog').waitFor({state:'hidden'});
  });
  await scenario('network failure is visible',r=>r.abort(),async page=>{
    await page.getByText('Cannot reach the server',{exact:true}).waitFor();
    await page.getByRole('button',{name:'Dismiss message'}).click();
    await page.getByText('Cannot reach the server',{exact:true}).waitFor({state:'hidden'});
  });
  let submitted=false, polls=0;
  await scenario('upload completes',(r,u)=>{
    if(u.pathname==='/api/upload'){submitted=true;return answer(r,{id:'one',status:'queued'});}
    if(u.pathname==='/api/wells') return answer(r,submitted?[sample('one','Test well',++polls===1?'queued':'ready')]:[]);
    return answer(r,detail(sample()));
  },async page=>{
    await page.getByText('No wells yet. Upload a scan to put one on the map.').waitFor();await upload(page);
    await page.getByRole('button',{name:'Test well, ready',exact:true}).waitFor();
    await page.getByRole('button',{name:'Test well, ready',exact:true}).click();
    await page.getByRole('link',{name:'Download CSV'}).waitFor();
    assert.equal(await page.getByRole('link',{name:'Download CSV'}).getAttribute('href'),'/api/wells/one/csv');
  });
  await scenario('upload rejection keeps dialog',(r,u)=>u.pathname==='/api/upload'?answer(r,{error:{code:'too_large',message:'Fixture upload rejected'}},413):answer(r,[]),async page=>{
    await page.getByText('No wells yet. Upload a scan to put one on the map.').waitFor();await upload(page);
    await page.getByRole('alert').filter({hasText:'Fixture upload rejected'}).waitFor();
    assert.equal(await page.getByRole('button',{name:'Upload',exact:true}).isEnabled(),true);
  });
  const attack='<img src=x onerror="globalThis.STRESS_INJECTED=1">';
  await scenario('metadata remains text',(r,u)=>answer(r,u.pathname==='/api/wells'?[sample('one',attack)]:detail(sample('one',attack))),async page=>{
    await page.getByRole('button',{name:attack+', ready',exact:true}).click();
    await page.getByRole('heading',{name:attack,exact:true}).waitFor();
    assert.equal(await page.locator('.sidebar img').count(),0);
    assert.equal(await page.evaluate(()=>globalThis.STRESS_INJECTED),undefined);
  });
  let initialRoute, initialSent=false;
  await scenario('late list response cannot stop polling',async(r,u)=>{
    if(u.pathname==='/api/upload') return answer(r,{id:'one',status:'queued'});
    if(u.pathname==='/api/wells' && !initialSent){initialSent=true;initialRoute=r;return;}
    if(u.pathname==='/api/wells') return answer(r,[sample('one','Race well','queued')]);
    return answer(r,detail(sample('one','Race well')));
  },async page=>{
    await page.getByRole('button',{name:'Upload well',exact:true}).waitFor();
    await page.waitForFunction(()=>document.querySelector('main'));
    await upload(page);await page.getByRole('button',{name:'Race well, queued',exact:true}).waitFor();
    await answer(initialRoute,[]);
    await page.waitForTimeout(250);
    assert.equal(await page.getByRole('button',{name:'Race well, queued',exact:true}).count(),1,'Late initial response removed the queued well');
  });
  let deferred;
  await scenario('late details do not replace selection',async(r,u)=>{
    if(u.pathname==='/api/wells') return answer(r,[sample('one','First'),{...sample('two','Second'),lon:-89}]);
    if(u.pathname==='/api/wells/one'){deferred=r;return;}
    return answer(r,detail(sample('two','Second')));
  },async page=>{
    await page.getByRole('button',{name:'First, ready',exact:true}).click();
    await page.getByRole('button',{name:'Second, ready',exact:true}).click();
    await page.getByRole('heading',{name:'Second',exact:true}).waitFor();
    assert.ok(deferred,'first request was never issued');await answer(deferred,detail(sample('one','First')));
    await page.waitForTimeout(150);assert.equal(await page.getByRole('heading',{name:'Second',exact:true}).count(),1);
  });
}
main().catch(error=>results.push({name:'browser setup',passed:false,error:String(error)})).finally(async()=>{
  if(browser)await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
  const summary={schema_version:1,passed:results.length===7&&results.every(x=>x.passed),results,
    source_file_sha256:sources,built_asset_sha256:assets,node:process.version,
    external_requests_intercepted:externalIntercepts,backend:'mocked API; separate Python tests exercise real FastAPI',
    script_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex')};
  fs.writeFileSync(path.join(output,'summary.json'),JSON.stringify(summary,null,2)+'\n');
  console.log(JSON.stringify({...summary,output},null,2));process.exitCode=summary.passed?0:1;
});
