'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {layout}=require('../static/viewport.js');
const hadrian={boxes:{interpreted_lithology_column:[525,1528,582,49449],depth_column:[369,1528,449,49449]}};
const wide={boxes:{interpreted_lithology_column:[2972,4392,3269,92668],depth_column:[1271,4392,1407,92668]}};
test('normal Hadrian view uses the supplied 57-pixel column without padding',()=>{
 const v=layout(hadrian,570,560,240,false);
 assert.deepEqual(v.bounds,[525,1528,582,49449]);
 assert.equal(v.width,133);
 assert.equal(v.height/v.scale,47921);
});
test('a wide column fits the viewer and has the same scale on both axes',()=>{
 const v=layout(wide,350,560,240,false);
 assert.deepEqual(v.bounds,[2972,4392,3269,92668]);
 assert.ok(v.width<=350-24+1e-8);
 assert.ok(Math.abs(v.width/297-v.height/88276)<1e-12);
});
test('context includes the existing depth and lithology bounds',()=>{
 const v=layout(hadrian,570,560,240,true);
 assert.deepEqual(v.bounds,[369,1528,582,49449]);
});
test('wide context stays visible on a phone',()=>{
 const v=layout(wide,240,450,240,true);
 assert.deepEqual(v.bounds,[1271,4392,3269,92668]);
 assert.ok(v.width<=216+1e-8);
 assert.ok(v.rows>240);
});
test('different row windows keep the original source bounds',()=>{
 for(const rows of [240,600,1200,2400])assert.deepEqual(layout(hadrian,570,560,rows,false).bounds,[525,1528,582,49449]);
});
