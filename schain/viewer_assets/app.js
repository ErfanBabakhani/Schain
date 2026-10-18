/* Offline, deterministic tier explorer. No external libraries, servers or telemetry. */
'use strict';
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const colors={Company:'#087f8c',Mine:'#ad6b22',Product:'#526ed3',Material:'#8555af',Location:'#448951',Person:'#bf5271'};
const text=(tag,value,cls)=>{const el=document.createElement(tag);el.textContent=String(value??'');if(cls)el.className=cls;return el;};
const svgEl=(tag,attrs={})=>{const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,String(v)));return el;};
const safeUrl=value=>{try{const u=new URL(value);return ['https:','http:'].includes(u.protocol)?u.href:null;}catch{return null;}};
let bundle,byId,selected=null,focus=null,currentNodes=[],currentEdges=[],positions=new Map(),page=0,view=[0,0,1000,600],fullView=[0,0,1000,600],drag=null;
const loaded=[];
function validate(data){
 if(!data||!Array.isArray(data.nodes)||!Array.isArray(data.edges))throw Error('Expected a SChain JSON object with nodes and edges arrays.');
 const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 const ids=new Set(),nodes=[];for(const n of data.nodes){if(!object(n)||typeof n.id!=='string'||!n.id||ids.has(n.id))throw Error('Node IDs must be unique non-empty strings.');ids.add(n.id);
  for(const key of ['name','type'])if(n[key]!=null&&typeof n[key]!=='string')throw Error('Node '+key+' must be a string when present.');
  if(n.aliases!=null&&(!Array.isArray(n.aliases)||n.aliases.some(a=>typeof a!=='string')))throw Error('Node aliases must be an array of strings.');
  if(n.discovery_tier!=null&&(!Number.isInteger(n.discovery_tier)||n.discovery_tier<0))throw Error('Node discovery_tier must be a non-negative integer or null.');
  nodes.push({...n,name:n.name||n.id,type:n.type||'Unknown'});
 }
 for(const key of ['metadata','statistics'])if(data[key]!=null&&!object(data[key]))throw Error('Graph '+key+' must be an object.');
 if(data.metadata?.generated_at!=null&&typeof data.metadata.generated_at!=='string')throw Error('metadata.generated_at must be a string.');
 if(data.documents!=null&&(!Array.isArray(data.documents)||data.documents.some(d=>!object(d))))throw Error('Graph documents must be an array of objects.');
 const es=new Set();for(const e of data.edges){if(!object(e)||!ids.has(e.source)||!ids.has(e.target))throw Error('An edge references a missing entity.');if(typeof e.id!=='string'||!e.id||es.has(e.id))throw Error('Edge IDs must be unique non-empty strings.');es.add(e.id);
  if(e.quality!=null&&!object(e.quality))throw Error('Edge quality must be an object.');
  if(e.source_doc_ids!=null&&(!Array.isArray(e.source_doc_ids)||e.source_doc_ids.some(d=>typeof d!=='string')))throw Error('Edge source_doc_ids must be an array of strings.');
  if(e.evidence!=null&&(!Array.isArray(e.evidence)||e.evidence.some(item=>!object(item))))throw Error('Edge evidence must be an array of objects.');
 }
 return {...data,nodes};
}
function option(select,value,label){const o=text('option',label);o.value=value;select.append(o);}
function isChecked(e){return e.quality?.evidence_validated===true;}
function stateName(e){return e.quality?.contested||e.quality?.status==='contested'?'Contested':isChecked(e)?'Model-checked':'Unchecked / unknown';}
function docs(e){return new Set((e.source_doc_ids||e.evidence?.map(x=>x.doc_id)||[]).filter(Boolean)).size||Number(e.support_count)||0;}
function load(data,name){bundle=validate(data);byId=new Map(bundle.nodes.map(n=>[n.id,n]));selected=null;focus=null;page=0;
 $('dataset').textContent=name;
 $('banner').textContent=bundle.metadata?.demo?'SYNTHETIC DEMO · Fictional companies and relationships, for interface testing only.':'Model checks indicate textual support, not independent factual verification. This graph is a dated observation, not a complete supply-chain inventory.';
 for(const id of ['seed','relation','type','tier'])$(id).replaceChildren();
 option($('seed'),'','All configured seeds');bundle.nodes.filter(n=>n.is_seed).sort((a,b)=>a.name.localeCompare(b.name)).forEach(n=>option($('seed'),n.id,n.name));
 option($('relation'),'suppliesTo','Supply relationships');option($('relation'),'','All relationships');[...new Set(bundle.edges.map(e=>e.relation))].filter(x=>x!=='suppliesTo').sort().forEach(x=>option($('relation'),x,x));
 option($('type'),'','All types');[...new Set(bundle.nodes.map(n=>n.type))].sort().forEach(x=>option($('type'),x,x));
 option($('tier'),'','All available tiers');[...new Set(bundle.nodes.map(n=>n.discovery_tier).filter(x=>Number.isInteger(x)&&x>=0))].sort((a,b)=>a-b).forEach(x=>option($('tier'),String(x),'Through tier '+x));
 reset();
}
function reset(){for(const id of ['seed','type','tier','quality'])$(id).value='';$('relation').value='suppliesTo';$('support').value='1';$('search').value='';$('context').checked=true;focus=null;selected=null;page=0;render();showIntro();}
function neighbors(id,edges,hops=1,upstream=false){
 const adjacency=new Map();for(const e of edges){if(!adjacency.has(e.target))adjacency.set(e.target,[]);adjacency.get(e.target).push(e.source);if(!upstream){if(!adjacency.has(e.source))adjacency.set(e.source,[]);adjacency.get(e.source).push(e.target);}}
 const seen=new Set([id]),queue=[[id,0]];for(let head=0;head<queue.length;head++){const [node,depth]=queue[head];if(depth>=hops)continue;for(const other of adjacency.get(node)||[])if(!seen.has(other)){seen.add(other);queue.push([other,depth+1]);}}return seen;
}

function filtered(){
 const relation=$('relation').value,type=$('type').value,tier=$('tier').value,quality=$('quality').value,query=$('search').value.trim().toLocaleLowerCase();
 let edges=bundle.edges.filter(e=>(!relation||e.relation===relation)&&docs(e)>=Number($('support').value));
 edges=edges.filter(e=>!quality||(quality==='checked'&&isChecked(e))||(quality==='unchecked'&&!isChecked(e))||(quality==='contested'&&stateName(e)==='Contested'));
 let allowed=new Set(bundle.nodes.filter(n=>(!type||n.type===type)&&(tier===''||n.discovery_tier==null||n.discovery_tier<=Number(tier))&&($('context').checked||n.discovery_tier!=null)).map(n=>n.id));
 if($('seed').value){const ns=neighbors($('seed').value,bundle.edges.filter(e=>e.relation==='suppliesTo'),bundle.nodes.length,true);allowed=new Set([...allowed].filter(x=>ns.has(x)));}
 if(query){const matching=new Set(bundle.nodes.filter(n=>[n.name,...(n.aliases||[])].some(x=>String(x).toLocaleLowerCase().includes(query))).map(n=>n.id));const ns=new Set(matching);for(const e of edges)if(matching.has(e.source)||matching.has(e.target)){ns.add(e.source);ns.add(e.target);}allowed=new Set([...allowed].filter(x=>ns.has(x)));}
 if(focus){const ns=neighbors(focus,edges,1);allowed=new Set([...allowed].filter(x=>ns.has(x)));}
 edges=edges.filter(e=>allowed.has(e.source)&&allowed.has(e.target));
 const used=new Set(edges.flatMap(e=>[e.source,e.target]));
 let nodes=bundle.nodes.filter(n=>allowed.has(n.id)&&(used.has(n.id)||n.is_seed||n.id===focus||(query&&[n.name,...(n.aliases||[])].some(x=>String(x).toLocaleLowerCase().includes(query)))));
 return{nodes,edges};
}
function render(){const f=filtered();currentNodes=f.nodes;currentEdges=f.edges;page=0;
 $('nodeCount').textContent=currentNodes.length.toLocaleString();$('edgeCount').textContent=currentEdges.length.toLocaleString();$('docCount').textContent=new Set(currentEdges.flatMap(e=>e.source_doc_ids||[])).size.toLocaleString();$('checkedCount').textContent=currentEdges.filter(isChecked).length.toLocaleString();
 $('focusLabel').textContent=focus?'Neighborhood: '+byId.get(focus)?.name:'';$('clearFocus').hidden=!focus;
 $('viewInfo').textContent=`${bundle.nodes.length.toLocaleString()} entities / ${bundle.edges.length.toLocaleString()} relationships in file · ${bundle.metadata?.generated_at?.slice(0,10)||'date unavailable'}`;
 draw();drawTable();if(selected&&!currentEdges.some(e=>e.id===selected)){selected=null;showIntro();}
}
function draw(preserveView=false){
 const svg=$('graph');svg.replaceChildren();positions=new Map();
 const defs=svgEl('defs'),marker=svgEl('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});marker.append(svgEl('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#7994a8'}));defs.append(marker);svg.append(defs);
 const nodes=[...currentNodes].sort((a,b)=>(Number(b.is_seed)-Number(a.is_seed))||((a.discovery_tier??999)-(b.discovery_tier??999))||String(a.name).localeCompare(String(b.name))).slice(0,300),allowed=new Set(nodes.map(n=>n.id));
 const edges=currentEdges.filter(e=>allowed.has(e.source)&&allowed.has(e.target)).slice(0,1000);
 const buckets=new Map();for(const n of nodes){const k=Number.isInteger(n.discovery_tier)?n.discovery_tier:'context';if(!buckets.has(k))buckets.set(k,[]);buckets.get(k).push(n);}
 const keys=[...buckets.keys()].sort((a,b)=>a==='context'?1:b==='context'?-1:b-a);
 const colWidth=250,w=Math.max(800,keys.length*colWidth+60),h=Math.max(520,...[...buckets.values()].map(b=>b.length*92+130));
 fullView=[0,0,w,h];if(!preserveView)view=[...fullView];svg.setAttribute('viewBox',view.join(' '));
 svg.append(svgEl('rect',{x:0,y:0,width:w,height:h,fill:'#f7fafc'}));
 keys.forEach((k,i)=>{const x=40+i*colWidth;if(i%2===0)svg.append(svgEl('rect',{x:x-20,y:30,width:colWidth-5,height:h-55,rx:12,fill:'#edf3f7'}));const label=svgEl('text',{x:x+12,y:62,fill:'#5b7185','font-family':'Segoe UI, Arial, sans-serif','font-size':15,'font-weight':600});label.textContent=k==='context'?'UNTIERED CONTEXT':k===0?'TIER 0 · SEEDS':'UPSTREAM · TIER '+k;svg.append(label);buckets.get(k).sort((a,b)=>String(a.name).localeCompare(String(b.name))).forEach((n,j)=>positions.set(n.id,{x:x+100,y:125+j*92}));});
 for(const e of edges){const a=positions.get(e.source),b=positions.get(e.target);const dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy)||1;const end={x:b.x-dx/len*24,y:b.y-dy/len*24};let d;
 if(e.source===e.target)d=`M ${a.x-12} ${a.y-15} C ${a.x-65} ${a.y-85},${a.x+65} ${a.y-85},${a.x+15} ${a.y-18}`;
 else if(a.x===b.x)d=`M ${a.x+18} ${a.y} C ${a.x+92} ${a.y},${b.x+92} ${b.y},${b.x+20} ${b.y}`;
 else d=`M ${a.x+dx/len*23} ${a.y+dy/len*23} C ${(a.x+b.x)/2} ${a.y},${(a.x+b.x)/2} ${b.y},${end.x} ${end.y}`;
 const g=svgEl('g',{'data-edge':e.id,tabindex:0,role:'button','aria-label':`${byId.get(e.source).name} ${e.relation} ${byId.get(e.target).name}. Inspect evidence.`});
 const path=svgEl('path',{d,fill:'none',stroke:e.id===selected?'#087f8c':stateName(e)==='Contested'?'#bd843d':'#8da5b7','stroke-width':e.id===selected?3:1.7,'marker-end':'url(#arrow)'});if(!isChecked(e))path.setAttribute('stroke-dasharray','6 4');
 const hit=svgEl('path',{d,fill:'none',stroke:'transparent','stroke-width':17,cursor:'pointer'});g.append(path,hit);const title=svgEl('title');title.textContent=`${byId.get(e.source).name} → ${byId.get(e.target).name} · ${e.relation} · ${docs(e)} source documents`;g.append(title);g.addEventListener('click',ev=>{ev.stopPropagation();inspectEdge(e);});g.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();inspectEdge(e);}});svg.append(g);
 }
 for(const n of nodes){const p=positions.get(n.id),g=svgEl('g',{'data-node':n.id,tabindex:0,role:'button','aria-label':n.name+'. Inspect entity',cursor:'pointer'});if(n.is_seed)g.append(svgEl('circle',{cx:p.x,cy:p.y,r:27,fill:'none',stroke:colors[n.type]||'#677','stroke-width':1.3}));g.append(svgEl('circle',{cx:p.x,cy:p.y,r:20,fill:colors[n.type]||'#677',stroke:'#fff','stroke-width':3}));const letter=svgEl('text',{x:p.x,y:p.y+5,'text-anchor':'middle',fill:'#fff','font-size':14,'font-weight':700,'font-family':'Segoe UI, Arial, sans-serif'});letter.textContent=n.type==='Company'?'C':String(n.type||'?')[0];g.append(letter);const name=svgEl('text',{x:p.x,y:p.y+44,'text-anchor':'middle',fill:'#213b51','font-size':14,'font-weight':600,'font-family':'Segoe UI, Arial, sans-serif'});name.textContent=String(n.name||n.id).length>27?String(n.name).slice(0,25)+'…':n.name||n.id;g.append(name);const title=svgEl('title');title.textContent=n.name+' · '+n.type;g.append(title);g.addEventListener('click',ev=>{ev.stopPropagation();inspectNode(n);});g.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();inspectNode(n);}});svg.append(g);}
 $('empty').hidden=nodes.length!==0;
 if(nodes.length<currentNodes.length||edges.length<currentEdges.length)$('viewInfo').textContent+=` · Drawing limited to ${nodes.length} nodes / ${edges.length} edges; filter further or use Relationships.`;
}
function showIntro(){$('details').replaceChildren(text('h2','Select a relationship'),text('p','Inspect its exact source quote and evidence status. Select a node to focus on its immediate connections.','muted'));}
function pill(parent,label,warn=false){parent.append(text('span',label,'pill'+(warn?' warning':'')));}
function inspectNode(n){selected=null;const box=$('details');box.replaceChildren(text('h2',n.name));pill(box,n.type);pill(box,n.discovery_tier==null?'Untiered context':'Tier '+n.discovery_tier);if(n.is_seed)pill(box,'Configured seed');const list=text('dl','');for(const [k,v]of [['Suppliers',bundle.edges.filter(e=>e.relation==='suppliesTo'&&e.target===n.id).length],['Customers',bundle.edges.filter(e=>e.relation==='suppliesTo'&&e.source===n.id).length],['Aliases',(n.aliases||[]).join(', ')||'None']]){list.append(text('dt',k),text('dd',v));}box.append(list);const b=text('button','Focus on this entity and its neighbors');b.onclick=()=>{focus=n.id;render();inspectNode(n);};box.append(b);box.append(text('p','Counts describe observed relationships in this file. A missing path does not prove the absence of a supplier.','note'));draw(true);}
function inspectEdge(e){selected=e.id;const box=$('details');box.replaceChildren(text('h2',`${byId.get(e.source).name} → ${byId.get(e.target).name}`));pill(box,e.relation);pill(box,stateName(e),!isChecked(e));pill(box,docs(e)+' source document'+(docs(e)===1?'':'s'));
 box.append(text('p','Model-checked means the validator judged the quote to support the relation. It does not establish external truth or current contractual status.','note'));
 const evidence=e.evidence||[];if(!evidence.length)box.append(text('p','No source quote is available in this export.','muted'));
 for(const item of evidence){const card=text('section','','evidence');card.append(text('strong',item.document_title||item.doc_id||'Untitled source'));card.append(text('blockquote',item.quote||'Quote unavailable'));
 pill(card,item.content_source==='title_snippet'?'Title / snippet only':item.content_source||'Content type unknown',item.content_source==='title_snippet');
 if(item.evidence_quote_grounded===true)pill(card,'Quote matched source');else if(item.evidence_quote_grounded===false)pill(card,'Quote did not match source',true);else pill(card,'Source match unknown',true);
 const url=safeUrl(item.canonical_source_url||item.source_url);if(url){const a=text('a','Open source ↗');a.href=url;a.target='_blank';a.rel='noopener noreferrer';card.append(text('p',''),a);}
 const meta=[item.source_domain,item.provider,Number.isInteger(item.chunk_index)?'Chunk '+item.chunk_index:null,item.extracted_at?'Extracted '+item.extracted_at:null,item.model?'Model: '+item.model:null].filter(Boolean).join(' · ');card.append(text('p',meta));box.append(card);
 }draw(true);drawTable();
}
function drawTable(){const body=$('rows');body.replaceChildren();const max=Math.max(1,Math.ceil(currentEdges.length/100));page=Math.min(page,max-1);for(const e of currentEdges.slice(page*100,(page+1)*100)){const tr=text('tr','');tr.tabIndex=0;for(const v of [byId.get(e.source).name,e.relation,byId.get(e.target).name,docs(e),stateName(e)])tr.append(text('td',v));tr.onclick=()=>inspectEdge(e);tr.onkeydown=ev=>{if(ev.key==='Enter')inspectEdge(e);};body.append(tr);}$('page').textContent=`Page ${page+1} of ${max}`;$('prev').disabled=page===0;$('next').disabled=page>=max-1;}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function zoom(factor){const [x,y,w,h]=view;view=[x+w*(1-factor)/2,y+h*(1-factor)/2,w*factor,h*factor];$('graph').setAttribute('viewBox',view.join(' '));}
$('zoomIn').onclick=()=>zoom(.8);$('zoomOut').onclick=()=>zoom(1.25);$('fit').onclick=()=>{view=[...fullView];$('graph').setAttribute('viewBox',view.join(' '));};
$('graph').addEventListener('wheel',ev=>{ev.preventDefault();zoom(ev.deltaY>0?1.1:.9);},{passive:false});
$('graph').addEventListener('pointerdown',ev=>{if(ev.target.closest('[data-node],[data-edge]'))return;drag={x:ev.clientX,y:ev.clientY,view:[...view]};$('graph').setPointerCapture(ev.pointerId);});
$('graph').addEventListener('pointermove',ev=>{if(!drag)return;const r=$('graph').getBoundingClientRect(),scale=Math.max(drag.view[2]/r.width,drag.view[3]/r.height);view=[drag.view[0]-(ev.clientX-drag.x)*scale,drag.view[1]-(ev.clientY-drag.y)*scale,...drag.view.slice(2)];$('graph').setAttribute('viewBox',view.join(' '));});
for(const evt of ['pointerup','pointercancel'])$('graph').addEventListener(evt,()=>drag=null);
for(const id of ['seed','relation','type','tier','quality','support','context'])$(id).onchange=render;
let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(render,140);};
$('clearFocus').onclick=()=>{focus=null;render();};$('reset').onclick=reset;
$('prev').onclick=()=>{page--;drawTable();};$('next').onclick=()=>{page++;drawTable();};
function tab(isGraph){$('canvas').hidden=!isGraph;$('tableWrap').hidden=isGraph;$('graphTab').setAttribute('aria-pressed',String(isGraph));$('tableTab').setAttribute('aria-pressed',String(!isGraph));for(const id of ['zoomIn','zoomOut','fit','saveSvg'])$(id).hidden=!isGraph;}
$('graphTab').onclick=()=>tab(true);$('tableTab').onclick=()=>tab(false);
$('saveSvg').onclick=()=>{const copy=$('graph').cloneNode(true);copy.setAttribute('viewBox',fullView.join(' '));copy.setAttribute('width',fullView[2]);copy.setAttribute('height',fullView[3]);download('supply-network.svg',new XMLSerializer().serializeToString(copy),'image/svg+xml');};
$('saveJson').onclick=()=>{const ids=new Set(currentEdges.flatMap(e=>e.source_doc_ids||[]));const out={...bundle,metadata:{...bundle.metadata,filtered_view:true,original_graph_fingerprint:bundle.metadata?.graph_fingerprint,graph_fingerprint:null},statistics:{nodes:currentNodes.length,edges:currentEdges.length,documents:ids.size},nodes:currentNodes,edges:currentEdges,documents:(bundle.documents||[]).filter(d=>ids.has(d.doc_id))};download('filtered-graph.json',JSON.stringify(out,null,2),'application/json');};
$('openGraph').onclick=()=>$('upload').click();
$('upload').onchange=async ev=>{const errors=[];for(const file of ev.target.files){try{if(file.size>50*1024*1024)throw Error('File exceeds the 50 MB viewer limit.');const data=validate(JSON.parse(await file.text()));loaded.push({name:file.name,data});option($('files'),String(loaded.length-1),file.name);}catch(e){errors.push(file.name+': '+e.message);}}if(loaded.length){$('files').value=String(loaded.length-1);load(loaded.at(-1).data,loaded.at(-1).name);}if(errors.length)$('banner').textContent=errors.join(' ');ev.target.value='';};
$('files').onchange=()=>{const item=loaded[Number($('files').value)];load(item.data,item.name);};
for(const [name,color]of Object.entries(colors)){const l=text('span','','legendItem'),d=text('span','','dot');d.style.background=color;l.append(d,text('span',name));$('legend').append(l);}
try{const data=JSON.parse($('graphData').textContent);loaded.push({name:data.metadata?.demo?'Synthetic demo':'Embedded graph',data:validate(data)});option($('files'),'0',loaded[0].name);load(data,loaded[0].name);}catch(e){$('banner').textContent='Cannot open graph: '+e.message;}
