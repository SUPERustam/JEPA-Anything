const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const root=__dirname;
const {eccentricAnomaly,orbitalPoint,blockAverage,validateRun}=require('./app.js');
const data=JSON.parse(fs.readFileSync(path.join(root,'data/run.json')));
validateRun(data);
assert.equal(data.provenance.exit_code,0);
assert.equal(data.provenance.optimizer_steps,10000);
assert(data.provenance.environment_prefix.endsWith('/envs/menv'));
for(const [file,hash] of Object.entries(data.provenance.artifact_sha256)){
  if(file==='checkpoint.pt')continue; // Checkpoint stays on the remote machine.
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'data',file))).digest('hex'),hash,file);
}
for(const [file,hash] of Object.entries(data.provenance.source_sha256))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'data/training-source',file))).digest('hex'),hash,file);
for(const [file,hash] of Object.entries(data.evaluation_provenance.source_sha256))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.resolve(root,'../../..',file))).digest('hex'),hash,file);
for(const [file,hash] of Object.entries(data.evaluation_provenance.artifact_sha256))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'data/controls',file))).digest('hex'),hash,file);
assert.equal(data.evaluation_provenance.checkpoint_sha256,data.provenance.artifact_sha256['checkpoint.pt']);
assert.equal(data.evaluation.prediction_evaluation.transitions,208845);
assert(data.evaluation.prediction_evaluation.mse.learned<data.evaluation.prediction_evaluation.mse.teacher_persistence);
assert.deepEqual(data.evaluation.harmonic_flags.trained[0],[11,13,15]);
assert(Math.abs(data.evaluation.untrained_aggregate.slope-data.evaluation.learned_aggregate.slope)<1e-6);
for(const e of [.05,.15,.3]) for(let M=0;M<=2*Math.PI;M+=.05){const E=eccentricAnomaly(M,e);assert(Math.abs(E-e*Math.sin(E)-M)<1e-12);}
const a=2,p=orbitalPoint(a,0,.25);assert(Math.abs(p.x)<1e-12);assert(Math.abs(p.y-a)<1e-12);
const start=orbitalPoint(a,.3,0),end=orbitalPoint(a,.3,1);assert(Math.abs(start.x-end.x)<1e-12);assert(Math.abs(start.y-end.y)<1e-12);
const mean=blockAverage([{step:1,total:2},{step:2,total:4},{step:3,total:6}],'total',2);assert.deepEqual(mean,[[1.5,3],[3,6]]);
assert.equal(blockAverage(data.metrics,'total',50).length,200);
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
for(const match of html.matchAll(/(?:src|href)="([^"#]+)"/g)){const url=match[1];if(!url.startsWith('http'))assert(fs.existsSync(path.join(root,url)),url);}
for(const match of fs.readFileSync(path.join(root,'app.js'),'utf8').matchAll(/\$\('([^']+)'\)/g))assert(html.includes(`id="${match[1]}"`),match[1]);
const harmonics=data.frequencies.filter(r=>r.factor_0_frequency/r.theory_frequency>1.8);assert.equal(harmonics.length,3);
console.log('PASS: menv provenance, source/artifact hashes, run counts, solver parity, averaging, linked files, DOM targets, and three factor-0 harmonics');
(async()=>{
  const {boot}=require('./app.js');
  const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{value:'',checked:false,textContent:'',innerHTML:'',handlers:{},attrs:{},setAttribute(k,v){this.attrs[k]=v;},addEventListener(k,fn){this.handlers[k]=fn;},classList:{add(){}}}]));
  for(const [id,value] of Object.entries({axis:'2',eccentricity:'.15',phase:'0','loss-metric':'total',step:'10000','frequency-series':'aggregate','orbit-index':'0'}))elements[id].value=value;
  elements.smooth.checked=true;
  global.document={getElementById:id=>elements[id]};global.fetch=async()=>({ok:true,json:async()=>data});global.requestAnimationFrame=()=>{};
  await boot();
  assert(elements['loss-readout'].textContent.includes('0.01758786'));
  elements.axis.value='4';elements.axis.handlers.input();assert(elements.period.textContent.includes('50.265'));
  elements.eccentricity.value='.3';elements.eccentricity.handlers.input();assert(elements.period.textContent.includes('50.265'));
  elements.step.value='1';elements.step.handlers.input();assert(elements['loss-readout'].textContent.includes('0.03864766'));
  elements['loss-metric'].value='factor_activity';elements['loss-metric'].handlers.change();assert(elements['loss-readout'].textContent.includes('not recorded'));
  elements.smooth.checked=false;elements.smooth.handlers.change();
  elements['frequency-series'].value='factor0';elements['frequency-series'].handlers.change();assert(elements['fit-stats'].innerHTML.includes('-1.528294'));
  elements['frequency-series'].value='untrained';elements['frequency-series'].handlers.change();assert(elements['fit-stats'].innerHTML.includes('-1.499882'));
  assert(elements['prediction-score'].textContent.includes('0.02399'));
  assert(elements['baseline-source'].textContent.includes('reconstructed'));
  elements['frequency-series'].value='factor0';elements['frequency-series'].handlers.change();
  elements.residual.checked=true;elements.residual.handlers.change();
  elements['orbit-index'].value=String(data.frequencies.indexOf(harmonics[0]));elements['orbit-index'].handlers.input();assert(Number(elements['frequency-readout'].textContent.split('ratio = ')[1])>1.8);
  elements.play.handlers.click();assert.equal(elements.play.attrs['aria-pressed'],'true');elements.play.handlers.click();assert.equal(elements.play.attrs['aria-pressed'],'false');
  for(const id of ['orbit-chart','loss-chart','frequency-chart'])assert(!/NaN|undefined/.test(elements[id].innerHTML),id);
  console.log('PASS: orbit sliders, fixed-a period, play/pause, loss components, smoothing, exact step readout, factor selection, harmonic ratio');
})().catch(error=>{console.error(error);process.exitCode=1;});
