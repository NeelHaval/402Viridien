const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require.resolve('../static/review.js'),'utf8');
function app(){
  const storage=new Map(),elements=new Map();
  const element=id=>{if(!elements.has(id))elements.set(id,{style:{},classList:{add(){},remove(){}},replaceChildren(){},append(){}});return elements.get(id);};
  const localStorage={getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)};
  const window={addEventListener(){}};
  const context={window,document:{getElementById:element,activeElement:null,createElement:()=>({setAttribute(){},append(){}})},localStorage,sessionStorage:localStorage,location:{origin:'http://example.test'},crypto:require('node:crypto').webcrypto,setTimeout(){},clearTimeout(){},console,AbortSignal,fetch:async(url,options)=>{const op=JSON.parse(options.body);return {ok:true,json:async()=>({ok:true,deferred_set:'family-43',task_id:op.task_id,op_id:op.op_id,saved_at:123})};}};
  // Run the actual UI functions without starting network or browser storage work.
  vm.runInNewContext(source.replace('storageKey:key};start();','storageKey:key,choose};'),context);
  const ui=window.SharedReview,r=ui.state();r.session={id:'reviewer',token:'test'};
  const make=(key,status='suggested')=>({key,title:key,accepted:0,document_count:1,tasks:[{id:key+'-box',document_name:key,context_box:[0,0,10,50],legend:[],segment:{id:key+'-box',status,approved:false,legend_ids:[],comments:[],box:[0,1,10,8]},version:'new'}]});
  r.sets=[make('family-43','rejected'),make('family-45')];r.setKey='family-43';r.taskId='family-43-box';
  return {ui,r,storage};
}
test('rejecting the last loaded box opens the next available set',()=>{
  const {ui,r}=app();ui.choose();assert.equal(r.setKey,'family-45');assert.equal(ui.current().id,'family-45-box');
});
test('an exhausted queue clears selection instead of trapping the reviewer',()=>{
  const {ui,r}=app();r.sets.pop();ui.choose();assert.equal(r.setKey,null);assert.equal(r.taskId,null);
});
test('a skipped final box opens the next available set',()=>{
  const {ui,r}=app();r.sets[0].tasks[0].segment.status='suggested';r.answers['family-43-box']={segment:r.sets[0].tasks[0].segment,skipped:true};ui.choose();assert.equal(r.setKey,'family-45');
});
test('a conflict on the last loaded box does not trap navigation or remove its pending draft',()=>{
  const {ui,r}=app();r.sets[0].tasks[0].segment.status='suggested';r.conflicts['family-43-box']={};r.pending['family-43-box']={op_id:'draft'};
  ui.choose();assert.equal(r.setKey,'family-45');assert.equal(r.pending['family-43-box'].op_id,'draft');
});
test('the current set stays selected while it has a usable box',()=>{
  const {ui,r}=app();r.sets[0].tasks[0].segment.status='suggested';ui.choose();assert.equal(r.setKey,'family-43');
});
