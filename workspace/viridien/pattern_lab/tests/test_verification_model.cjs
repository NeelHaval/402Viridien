const {test}=require('node:test');
const assert=require('node:assert/strict');
const {applyBedProposal,confirmableChoice}=require('../static/review-details.js');
const make=()=>({id:'t',document_id:'a',version:'new',context_box:[10,100,50,200],segment:{id:'t',box:[10,120,50,168],legend_ids:[],approved:false,status:'suggested'},legend:[{id:1,label_printed:'Shale',image:'legend'}]});
const proposal=()=>({document_id:'a',source_version:'new',box:[10,110,50,190],prediction:{engine:'v4-beds',model_version:'v4',model_label:'Shale',choices:[{legend_ids:[1],label:'Shale',reference:{kind:'legend'}}]}});
test('cached bed proposal changes guesses and a new box but never chooses a label',()=>{
  const t=make();applyBedProposal(t,proposal());assert.deepEqual(t.segment.legend_ids,[]);assert.deepEqual(t.segment.box,[10,110,50,190]);assert.equal(t.prediction.choices[0].reference.image,'legend');
  assert.equal(confirmableChoice(t,{version:'new',segment:t.segment}).label,'Shale');
});
test('saved labels and version mismatches are kept',()=>{
  const t=make();t.segment.approved=true;const before=JSON.stringify(t);applyBedProposal(t,proposal());assert.equal(JSON.stringify(t),before);
  t.segment.approved=false;t.version='saved-version';applyBedProposal(t,proposal());assert.equal(t.prediction,undefined);
});
test('a proposal outside the loaded source strip cannot move the box',()=>{
  const t=make(),p=proposal();p.box=[10,0,50,300];applyBedProposal(t,p);assert.deepEqual(t.segment.box,[10,120,50,168]);
});
test('retired or foreign legend IDs never become verification choices',()=>{
  const t=make(),p=proposal();p.prediction.choices[0].legend_ids=[99];applyBedProposal(t,p);assert.equal(t.verification,undefined);assert.equal(t.prediction,undefined);
});
test('box changes, ambiguous symbols, and a different manual selection disable quick confirmation',()=>{
  const t=make();applyBedProposal(t,proposal());const a={version:'new',segment:structuredClone(t.segment)};a.segment.box[1]++;assert.equal(confirmableChoice(t,a),null);a.segment=structuredClone(t.segment);t.prediction.ambiguous=true;assert.equal(confirmableChoice(t,a),null);t.prediction.ambiguous=false;a.segment.legend_ids=[2];assert.equal(confirmableChoice(t,a),null);
});

test('Terra only offers quick confirmation for a clean high-confidence crop',()=>{
  const t=make(),p=proposal();Object.assign(p.prediction,{engine:'terra-grid',terra_status:'match',terra_confidence:'high'});
  applyBedProposal(t,p);const a={version:'new',segment:t.segment};assert.equal(confirmableChoice(t,a).label,'Shale');
  t.prediction.terra_status='mixed';assert.equal(confirmableChoice(t,a),null);
  t.prediction.terra_status='match';t.prediction.terra_confidence='medium';assert.equal(confirmableChoice(t,a),null);
});
