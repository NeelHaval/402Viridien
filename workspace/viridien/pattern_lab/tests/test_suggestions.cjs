const {test}=require('node:test');
const assert=require('node:assert/strict');
const {boxColours,suggestEntries}=require('../static/review-details.js');
const swatch=(id,key,rgb,extra={})=>({id,label_printed:'Label '+id,colours:[{id:key,rgb}],...extra});
function pixels(...bands){return Uint8ClampedArray.from(bands.flatMap(([rgb,count])=>Array.from({length:count},()=>[...rgb,255]).flat()));}

test('box palette identifies each broad colour and ignores a thin black frame',()=>{
  for(const [rgb,key] of [[[255,255,255],'neutral'],[[130,130,130],'neutral'],[[255,255,0],'yellow'],[[255,130,0],'orange'],[[255,0,0],'red'],[[0,255,0],'green'],[[0,255,255],'blue'],[[160,0,255],'purple']]){
    const found=boxColours(pixels([rgb,95],[[0,0,0],5])).map(c=>c.id);assert.deepEqual(found,[key]);
  }
  assert.deepEqual(boxColours(pixels([[0,0,0],100])).map(c=>c.id),['black']);
  assert.deepEqual(boxColours(new Uint8ClampedArray(40)),[]);
});
test('mixed box colours retain each family',()=>{
  assert.deepEqual(new Set(boxColours(pixels([[255,255,255],40],[[255,255,0],30],[[0,255,255],30])).map(c=>c.id)),new Set(['neutral','yellow','blue']));
});
test('best guess stays first even if its colour differs from the box',()=>{
  const legend=[swatch(1,'red',[255,0,0]),swatch(2,'blue',[0,200,240]),swatch(3,'neutral',[255,255,255]),swatch(4,'blue',[0,0,255])];
  const result=suggestEntries(legend,[{id:'blue',rgb:[0,195,235]},{id:'neutral',rgb:[255,255,255]}],{choices:[{legend_ids:[1]}]});
  assert.deepEqual(result.entries.map(e=>e.entry.id),[1,2,4]);assert.equal(result.entries[0].kind,'best');assert.equal(result.entries[0].matchesColour,false);assert.equal(result.colourMatchCount,2);
});
test('retired entries and duplicate guesses do not enter the shortlist',()=>{
  const legend=[swatch(1,'blue',[0,0,255]),swatch(2,'blue',[0,0,255],{deprecated:true})];const before=JSON.stringify(legend);
  const result=suggestEntries(legend,[{id:'blue',rgb:[0,0,255]}],{choices:[{legend_ids:[2,1]},{legend_ids:[1]}]});
  assert.deepEqual(result.entries.map(e=>e.entry.id),[1]);assert.equal(JSON.stringify(legend),before);
});
test('no colour match does not turn unrelated labels into suggestions',()=>{
  const legend=[swatch(1,'red',[255,0,0]),swatch(2,'neutral',[255,255,255])];
  assert.deepEqual(suggestEntries(legend,[{id:'blue',rgb:[0,0,255]}],null).entries,[]);
  assert.equal(suggestEntries(legend,[],{choices:[{legend_ids:[1]}]}).entries[0].entry.id,1);
});
test('shortlist stays bounded when many legend entries share a colour',()=>{
  const legend=Array.from({length:20},(_,i)=>swatch(i,'yellow',[255,255,0]));const result=suggestEntries(legend,[{id:'yellow',rgb:[255,255,0]}],null,8);
  assert.equal(result.entries.length,8);assert.equal(result.colourMatchCount,20);assert.equal(new Set(result.entries.map(e=>e.entry.id)).size,8);
});
