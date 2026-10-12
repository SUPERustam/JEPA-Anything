'use strict';
const colors = {green:'#2e695e',orange:'#aa602d',gray:'#6d7971',grid:'#dce0d5',ink:'#39463f'};
function eccentricAnomaly(mean, e) {
  let E = mean;
  for (let i=0;i<50;i++) { const delta=(E-e*Math.sin(E)-mean)/(1-e*Math.cos(E)); E-=delta; if(Math.abs(delta)<1e-13) return E; }
  throw Error('Orbit solver did not converge');
}
function orbitalPoint(a,e,fraction) {
  const E=eccentricAnomaly(2*Math.PI*fraction,e), n=a**-1.5, rate=n/(1-e*Math.cos(E));
  return {x:a*(Math.cos(E)-e), y:a*Math.sqrt(1-e*e)*Math.sin(E),
    vx:-a*Math.sin(E)*rate, vy:a*Math.sqrt(1-e*e)*Math.cos(E)*rate};
}
function blockAverage(rows,key,size) {
  const points=[];
  for(let i=0;i<rows.length;i+=size) {
    const chunk=rows.slice(i,i+size);
    points.push([chunk.reduce((s,r)=>s+r.step,0)/chunk.length,chunk.reduce((s,r)=>s+r[key],0)/chunk.length]);
  }
  return points;
}
function validateRun(data) {
  if(data.config.orbits!==256 || data.config.samples!==4096 || data.config.smoke || data.metrics.length!==10000 || data.frequencies.length!==32) throw Error('Unexpected run configuration or sample counts');
  if(data.metrics.some((row,i)=>row.step!==i+1 || Object.values(row).some(v=>!Number.isFinite(v)))) throw Error('Invalid stored metrics');
  if(data.frequencies.some(row=>Object.values(row).some(v=>!Number.isFinite(v)) || row.a<=0 || row.theory_frequency<=0)) throw Error('Invalid frequency data');
  if(data.normalization.training_ids.length!==205 || data.normalization.validation_ids.length!==51 || data.normalization.training_ids.some(id=>data.normalization.validation_ids.includes(id))) throw Error('Invalid whole-orbit split');
  if(!data.evaluation || data.evaluation.optimizer_steps!==10000 || data.evaluation_frequencies.length!==32 || !data.evaluation.baseline_source) throw Error('Matched evaluation controls are missing');
  if(data.evaluation_frequencies.some(row=>Object.values(row).some(v=>!Number.isFinite(v)))) throw Error('Invalid matched frequencies');
  if(Object.values(data.evaluation.prediction_evaluation.mse).some(v=>!Number.isFinite(v) || v<0)) throw Error('Invalid prediction scores');
}
function chart(svg,{series,xLabel,yLabel,logX=false,logY=false,xDomain,yDomain,highlight}) {
  const W=960,H=360,m={left:88,right:25,top:24,bottom:60};
  const all=series.flatMap(s=>s.points), transformX=x=>logX?Math.log10(x):x, transformY=y=>logY?Math.log10(y):y;
  let xmin=transformX(xDomain?xDomain[0]:Math.min(...all.map(p=>p[0]))), xmax=transformX(xDomain?xDomain[1]:Math.max(...all.map(p=>p[0])));
  let ymin=transformY(yDomain?yDomain[0]:Math.min(...all.map(p=>p[1]))), ymax=transformY(yDomain?yDomain[1]:Math.max(...all.map(p=>p[1])));
  if(!yDomain) { const padding=(ymax-ymin || 1)*.1; ymin-=padding; ymax+=padding; }
  if(xmax===xmin) xmax=xmin+1;
  if(ymax===ymin) ymax=ymin+1;
  const X=x=>m.left+(transformX(x)-xmin)/(xmax-xmin)*(W-m.left-m.right),Y=y=>H-m.bottom-(transformY(y)-ymin)/(ymax-ymin)*(H-m.top-m.bottom);
  const label=(value,log)=>{const v=log?10**value:value;return Math.abs(v)<.001&&v!==0?v.toExponential(1):Number(v.toPrecision(4)).toLocaleString('en-US');};
  let markup='';
  for(let i=0;i<=4;i++) {
    const xx=m.left+i*(W-m.left-m.right)/4, yy=m.top+i*(H-m.top-m.bottom)/4;
    markup+=`<path d="M${xx},${m.top}V${H-m.bottom} M${m.left},${yy}H${W-m.right}" stroke="${colors.grid}"/>`;
    markup+=`<text x="${xx}" y="${H-m.bottom+25}" text-anchor="middle">${label(xmin+i*(xmax-xmin)/4,logX)}</text>`;
    markup+=`<text x="${m.left-12}" y="${yy+4}" text-anchor="end">${label(ymax-i*(ymax-ymin)/4,logY)}</text>`;
  }
  for(const s of series) {
    if(s.line) markup+=`<path d="${s.points.map((p,i)=>`${i?'L':'M'}${X(p[0]).toFixed(2)},${Y(p[1]).toFixed(2)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="${s.width||2}" ${s.dashed?'stroke-dasharray="7 5"':''}/>`;
    if(s.dots) markup+=s.points.map(p=>`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="${s.color}"/>`).join('');
  }
  if(highlight) markup+=`<circle cx="${X(highlight[0])}" cy="${Y(highlight[1])}" r="7" fill="none" stroke="${colors.ink}" stroke-width="2"/>`;
  markup+=`<text x="${W/2}" y="${H-9}" text-anchor="middle">${xLabel}</text><text transform="translate(18 ${H/2}) rotate(-90)" text-anchor="middle">${yLabel}</text>`;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.innerHTML=`<g font-family="system-ui,sans-serif" font-size="12" fill="${colors.ink}">${markup}</g>`;
}
async function boot() {
  const $=id=>document.getElementById(id);
  const response=await fetch('data/run.json');
  if(!response.ok) throw Error(`Run data could not load (${response.status})`);
  const data=await response.json(); validateRun(data);
  $('status').textContent='Verified remote menv run · 12 October 2026 · CPU · 205 training / 51 validation orbits · controls use saved data';
  let playing=false, lastTime=0, elapsed=0;
  function drawOrbit() {
    const a=+$('axis').value,e=+$('eccentricity').value,fraction=+$('phase').value;
    $('axis-value').textContent=a.toFixed(2);$('ecc-value').textContent=e.toFixed(2);$('phase-value').textContent=fraction.toFixed(3);
    const period=2*Math.PI*a**1.5;
    $('period').textContent=period.toFixed(3)+' time units';$('orbit-frequency').textContent=(1/period).toFixed(6)+' cycles / time';
    const X=x=>340+x*36,Y=y=>175-y*36, point=orbitalPoint(a,e,fraction), samples=[];
    for(let i=0;i<=160;i++) samples.push(orbitalPoint(a,e,i/160));
    let dots=''; for(let i=0;i<48;i++) {const p=orbitalPoint(a,e,i/48);dots+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="2.5" fill="${colors.green}" opacity=".6"/>`;}
    $('orbit-chart').innerHTML=`<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="${colors.orange}"/></marker></defs><path d="M70 175H590M340 25V325" stroke="${colors.grid}"/><path d="${samples.map((p,i)=>`${i?'L':'M'}${X(p.x)},${Y(p.y)}`).join(' ')}" fill="none" stroke="${colors.green}" stroke-width="1.5"/>${dots}<circle cx="340" cy="175" r="7" fill="${colors.orange}"/><text x="350" y="195" fill="${colors.gray}" font-size="12">gravity focus</text><circle cx="${X(point.x)}" cy="${Y(point.y)}" r="7" fill="${colors.green}"/><path d="M${X(point.x)},${Y(point.y)}l${point.vx*30},${-point.vy*30}" stroke="${colors.orange}" stroke-width="2" marker-end="url(#arrow)"/><text x="25" y="330" fill="${colors.gray}" font-size="12">Fixed spatial scale · arrow: velocity direction</text>`;
  }
  ['axis','eccentricity','phase'].forEach(id=>$(id).addEventListener('input',drawOrbit));
  function frame(time) {
    if(!playing) return;
    if(lastTime) elapsed+=(time-lastTime)/1000;
    lastTime=time;
    // Illustration time is accelerated; one real second advances two simulation time units.
    const period=2*Math.PI*(+$('axis').value)**1.5;
    $('phase').value=(elapsed*2/period)%1;drawOrbit();requestAnimationFrame(frame);
  }
  $('play').addEventListener('click',()=>{
    playing=!playing;lastTime=0;
    elapsed=+$('phase').value*2*Math.PI*(+$('axis').value)**1.5/2;
    $('play').textContent=playing?'Pause orbit':'Play orbit';$('play').setAttribute('aria-pressed',String(playing));
    if(playing) requestAnimationFrame(frame);
  });
  function drawLoss() {
    const key=$('loss-metric').value, size=$('smooth').checked?50:1, rows=data.metrics, index=+$('step').value-1, row=rows[index];
    const train=blockAverage(rows,key,size), valKey='validation_'+key;
    const series=[{points:train,line:true,color:colors.green}];
    if(valKey in row) series.push({points:blockAverage(rows,valKey,size),line:true,color:colors.orange});
    const upper=Math.max(...rows.map(r=>r[key]),...(valKey in row?rows.map(r=>r[valKey]):[0]))*1.05;
    chart($('loss-chart'),{series,xLabel:'Optimizer step',yLabel:key==='total'?'Composite objective':key.replaceAll('_',' '),xDomain:[1,10000],yDomain:[0,upper||.1],highlight:[row.step,row[key]]});
    $('step-value').textContent=row.step.toLocaleString('en-US');
    $('loss-readout').textContent=`Step ${row.step.toLocaleString('en-US')} · training ${row[key].toFixed(8)}`+(valKey in row?` · validation ${row[valKey].toFixed(8)}`:' · validation component not recorded');
  }
  ['loss-metric','smooth','step'].forEach(id=>$(id).addEventListener(id==='step'?'input':'change',drawLoss));
  const evaluation=data.evaluation;
  const seriesMap={aggregate:['learned_aggregate_frequency',evaluation.learned_aggregate],untrained:['untrained_aggregate_frequency',evaluation.untrained_aggregate],raw:['raw_state_frequency',evaluation.raw_state_physics_check],...Object.fromEntries([0,1,2,3].map(i=>['factor'+i,['factor_'+i+'_frequency',evaluation.learned_factors[i]]]))};
  function drawFrequency() {
    const [key,fit]=seriesMap[$('frequency-series').value], ratio=$('residual').checked, rows=data.evaluation_frequencies,index=+$('orbit-index').value, row=rows[index];
    const theory=rows.map(r=>[r.a,ratio?1:r.theory_frequency]);
    const points=rows.map(r=>[r.a,ratio?r[key]/r.theory_frequency:r[key]]);
    const series=[{points:theory,line:true,color:colors.orange},{points,dots:true,color:colors.green}];
    if(!ratio) series.unshift({points:rows.map(r=>[r.a,10**fit.intercept_log10*r.a**fit.slope]),line:true,color:colors.gray,dashed:true});
    chart($('frequency-chart'),{series,xLabel:'Semimajor axis a (log scale)',yLabel:ratio?'Extracted / theoretical frequency':'Frequency (cycles / time, log scale)',logX:true,logY:!ratio,highlight:points[index],...(ratio?{yDomain:[.85,Math.max(1.15,...points.map(p=>p[1]))+.08]}:{})});
    $('fit-stats').innerHTML=[['Log–log slope',fit.slope.toFixed(6)],['R² in log space',fit.r_squared_log10.toFixed(10)],['Mean relative error',(100*fit.mean_relative_frequency_error).toFixed(5)+'%'],['Largest relative error',(100*fit.max_relative_frequency_error).toFixed(5)+'%']].map(([name,value])=>`<div><span>${name}</span><strong>${value}</strong></div>`).join('');
    $('orbit-index-value').textContent=`${index+1} / 32`;
    $('frequency-readout').textContent=`Orbit ${index+1} · a = ${row.a.toFixed(4)} · e = ${row.e.toFixed(4)} · extracted = ${row[key].toFixed(7)} · theory = ${row.theory_frequency.toFixed(7)} · ratio = ${(row[key]/row.theory_frequency).toFixed(5)}`;
  }
  ['frequency-series','residual','orbit-index'].forEach(id=>$(id).addEventListener(id==='orbit-index'?'input':'change',drawFrequency));
  $('hero-slope').textContent=data.analysis.learned_aggregate.slope.toFixed(6).replace('-','−');
  $('full-slope').textContent=$('hero-slope').textContent;
  $('untrained-slope').textContent=evaluation.untrained_aggregate.slope.toFixed(6).replace('-','−');
  const prediction=evaluation.prediction_evaluation;
  $('prediction-score').textContent=prediction.mse.learned.toFixed(5);
  $('persistence-score').textContent=prediction.mse.teacher_persistence.toFixed(5);
  $('prediction-improvement').textContent=(100*prediction.improvement_over_teacher_persistence).toFixed(2)+'%';
  $('baseline-source').textContent=evaluation.baseline_source;
  drawOrbit();drawLoss();drawFrequency();
}
if(typeof document!=='undefined') boot().catch(error=>{const status=document.getElementById('status');status.classList.add('error');status.textContent='Interactive data error: '+error.message;console.error(error);});
if(typeof module!=='undefined') module.exports={eccentricAnomaly,orbitalPoint,blockAverage,validateRun,boot};
