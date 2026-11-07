(() => {
  'use strict';
  if (document.body.dataset.mode !== 'server') return;
  const $ = id => document.getElementById(id);
  const explorer = window.SChainExplorer;
  const activeStates = new Set(['queued', 'running', 'pausing']);
  const limits = [
    ['max_tier', 'Maximum discovery tier', 3, 0, 10, 'Only supply relationships advance a tier.'],
    ['max_documents_per_entity', 'Documents per entity', 5, 1, 100],
    ['max_documents_per_run', 'Documents per run', 10, 1, 1000],
    ['max_total_documents', 'Total document budget', 100, 1, 100000],
    ['llm_max_concurrency', 'LLM concurrency', 1, 1, 8],
    ['retrieval_concurrency', 'Retrieval concurrency', 2, 1, 8],
    ['max_minutes', 'Session duration (minutes)', 30, 0.01, 1440, 'A new time allowance for each execution session.']
  ];
  const session = { base: '', token: '', connected: false, projects: [], request: null, localProxy: false, csrf: '', connecting: false };
  let current = null, timer = null, editing = false, saving = false;
  let reconnectTimer = null, reconnectAttempt = 0;

  for (const [key, title, value, min, max, hint] of limits) {
    const label = document.createElement('label'); label.textContent = title;
    const input = document.createElement('input');
    Object.assign(input, { id: key, type: 'number', min, max, value, required: true, className: 'field', step: key === 'max_minutes' ? 'any' : '1' });
    label.append(input);
    if (hint) { const small = document.createElement('small'); small.textContent = hint; label.append(small); }
    $('projectLimits').append(label);
  }

  function message(text = '') {
    $('appMessage').textContent = text; $('appMessage').hidden = !text;
  }
  function abortProject() {
    clearTimeout(timer);
    current?.controller.abort(); current = null;
    $('projectActivity').hidden = true;
    $('liveActivity').hidden = true;
    $('liveDocuments').replaceChildren(); $('liveEvents').replaceChildren();
    $('projectSummary').textContent = 'Select a project to inspect its graph.';
    $('projectDialog').close();
  }
  function disconnected(text = 'Connect to your server to open a project.') {
    clearTimeout(reconnectTimer);
    abortProject(); session.request?.abort(); session.connected = false; session.token = '';
    session.projects = []; $('projectSelect').replaceChildren(new Option('Select a project', ''));
    $('projectState').textContent = 'Disconnected'; $('apiToken').value = '';
    explorer.clear('Project workspace', text); updateControls();
  }
  function errorText(body, fallback) {
    if (typeof body?.detail === 'string') return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map(e => `${e.loc.slice(1).join(' · ')}: ${e.msg}`).join('\n');
    return fallback;
  }
  async function api(path, options = {}, signal = current?.controller.signal) {
    const authentication = session.localProxy ? { 'X-SChain-CSRF': session.csrf } : { Authorization: 'Bearer ' + session.token };
    const response = await fetch(session.base + path, {
      ...options, signal, cache: 'no-store',
      headers: { ...authentication, ...options.headers }
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const error = Error(errorText(body, `Request failed (${response.status})`));
      error.status = response.status; throw error;
    }
    return body;
  }
  function handleError(error, context = current) {
    if (error.name === 'AbortError' || context !== current) return;
    if (session.localProxy && [401, 403].includes(error.status)) {
      disconnected('The local proxy connection needs attention.');
      message(error.message);
      if (error.status === 403) scheduleReconnect(error);
      return;
    }
    if (error.status === 401) {
      disconnected('Authentication expired. Connect again with the server token.');
      $('connectionError').textContent = error.message; $('connectionDialog').showModal();
    } else message(error instanceof TypeError ? 'Cannot reach the API. Check the SSH tunnel, endpoint and allowed local origin.' : error.message);
  }
  function updateControls() {
    const project = current?.project, active = activeStates.has(project?.status);
    $('projectSelect').disabled = !session.connected;
    $('newProject').disabled = !session.connected || saving;
    $('manageProject').disabled = !project;
    $('startRun').disabled = !project || active || current.busy;
    $('pauseRun').disabled = !project || !['queued', 'running'].includes(project.status) || current.busy;
    $('resumeRun').disabled = !project || !['paused', 'failed', 'completed'].includes(project.status) || current.busy;
    $('documentInput').disabled = !project || active || current.busy;
    $('uploadDocument').disabled = !project || active || current.busy;
    const locked = editing && !!project?.runs.length;
    if (locked) { $('projectSeeds').value = project.seeds.join('\n'); $('max_tier').value = project.limits.max_tier; }
    $('identityNote').hidden = !locked;
    for (const input of $('projectForm').querySelectorAll('input,textarea')) input.disabled = saving || (editing && active) || (locked && ['projectSeeds', 'max_tier'].includes(input.id));
    $('saveProject').disabled = saving || (editing && active);
  }
  function projectPath(context) { return '/projects/' + encodeURIComponent(context.id); }
  function showProject(context) {
    const project = context.project;
    const option = [...$('projectSelect').options].find(item => item.value === context.id);
    if (option) option.textContent = project.name;
    $('datasetTitle').textContent = project.name;
    $('projectState').textContent = project.status;
    const explanation = { queued: 'Waiting for the server worker', running: 'Processing on the server', pausing: 'Graceful pause requested; finishing the current unit', paused: 'Paused', completed: 'Last job completed', failed: 'Last job failed; inspect recent errors', ready: 'Ready to start' }[project.status];
    const stats = project.statistics;
    const documents = Object.entries(stats.documents).map(([key, value]) => `${value} ${key.toLowerCase()}`).join(', ');
    $('projectSummary').textContent = `${explanation} · ${stats.nodes} entities · ${stats.edges} relationships · ${documents || 'No documents'}`;
    updateControls(); renderActivity(context);
  }
  function row(root, title, detail) {
    const element = document.createElement('div'); element.className = 'activity-row';
    const heading = document.createElement('strong'); heading.textContent = title;
    const note = document.createElement('span'); note.textContent = detail;
    element.append(heading, note); root.append(element); return element;
  }
  function renderActivity(context) {
    $('documentList').replaceChildren(); $('runList').replaceChildren();
    for (const doc of context.documents) row($('documentList'), `${doc.title || doc.doc_id} · ${doc.status}${doc.stage ? ' · ' + (stageLabels[doc.stage] || doc.stage) : ''}`, [doc.error, doc.source_url, doc.origin_entity, doc.updated_at].filter(Boolean).join(' · '));
    if (!context.documents.length) $('documentList').textContent = 'No processed documents yet.';
    for (const run of context.project.runs) row($('runList'), `${run.kind} · ${run.status}`, [run.created_at, run.error, run.summary?.stop_reason].filter(Boolean).join(' · '));
    for (const error of context.project.errors) row($('runList'), error.doc_id || error.entity_key, error.error);
    if (!context.project.runs.length) $('runList').textContent = 'No runs yet.';
    $('moreDocuments').hidden = context.documentLimit >= 200 || context.documents.length < context.documentLimit;
  }
  const stageLabels = {
    retrieved: 'Document received', extraction: 'Extracting relationships', resolution: 'Resolving entities',
    validation: 'Validating relationships', facts_persistence: 'Saving validated facts',
    resolved: 'Validation completed', commit: 'Graph commit', failed: 'Document failed', chunk_started: 'Processing chunk',
    chunk_extracted: 'Chunk extracted', run_running: 'Run started', run_completed: 'Run completed',
    run_paused: 'Run paused', run_failed: 'Run failed', run_interrupted: 'Worker interrupted'
  };
  function showLive(context, activity) {
    context.cursor = activity.cursor; context.moreActivity = activity.has_more;
    $('liveActivity').hidden = false;
    $('liveStatus').textContent = `${activeStates.has(context.project.status) ? 'Live' : 'Up to date'} · ${new Date().toLocaleTimeString()}`;
    $('pendingCount').textContent = `${activity.pending_relationships} pending relationships`;
    const activeSignature = JSON.stringify([activity.active_documents, activity.entities]);
    if (context.activeSignature !== activeSignature) {
      context.activeSignature = activeSignature; $('liveDocuments').replaceChildren();
      for (const doc of activity.active_documents) row($('liveDocuments'), doc.title || doc.doc_id, `${stageLabels[doc.stage] || doc.stage} · ${doc.source_url || doc.doc_id}`);
      if (!activity.active_documents.length) {
        for (const entity of activity.entities) row($('liveDocuments'), entity.entity_name, `Expanding Tier ${entity.tier} · retrieving or processing documents`);
        if (!activity.entities.length) $('liveDocuments').textContent = activeStates.has(context.project.status) ? 'Waiting for the next document or worker update.' : 'No documents being processed.';
      }
    }
    if (activity.events.length) {
      const events = new Map(context.events.map(event => [event.id, event]));
      const root = $('liveEvents'), previousHeight = root.scrollHeight, previousScroll = root.scrollTop;
      if (!context.events.length) root.replaceChildren();
      for (const event of activity.events) {
        if (events.has(event.id)) continue;
        events.set(event.id, event);
        const p = event.payload;
        const details = [p.title, Number.isInteger(p.chunk_index) ? `chunk ${p.chunk_index + 1}${p.chunk_total ? '/' + p.chunk_total : ''}` : '',
          p.relationships != null ? `${p.relationships} extracted relationships` : '',
          p.accepted != null ? `${p.accepted} accepted, ${p.rejected} rejected` : '', p.cached ? 'cached' : '', p.error].filter(Boolean);
        root.prepend(row(root, `${new Date(event.created_at).toLocaleTimeString()} · ${stageLabels[event.stage] || event.stage}`, details.join(' · ')));
      }
      while (root.children.length > 100) root.lastElementChild.remove();
      if (previousScroll > 0) root.scrollTop = previousScroll + root.scrollHeight - previousHeight;
      context.events = [...events.values()].slice(-100);
    } else if (!context.events.length) $('liveEvents').textContent = 'No activity recorded yet.';
  }
  function populateProjects() {
    $('projectSelect').replaceChildren(new Option('Select a project', ''));
    for (const project of session.projects) $('projectSelect').add(new Option(project.name, project.id));
    if (current) $('projectSelect').value = current.id;
    if (!session.projects.length) explorer.setEmptyState('No projects yet', 'Create a project to start building a supply chain knowledge graph.', { label: 'New Project', run: () => openProject(false) });
  }
  async function graphSnapshot(context) {
    for (let attempt = 0; attempt < 3; attempt++) {
      const nodes = new Map(), edges = new Map(), documents = new Map();
      let offset = 0, nodeOffset = 0, first = null, changed = false;
      do {
        const page = await api(`${projectPath(context)}/graph?limit=5000&offset=${offset}&node_limit=5000&node_offset=${nodeOffset}`, {}, context.controller.signal);
        if (context !== current) return null;
        if (!first) first = page;
        if (page.revision !== first.revision) { changed = true; break; }
        page.nodes.forEach(n => nodes.set(n.id, n)); page.edges.forEach(e => edges.set(e.id, e));
        (page.documents || []).forEach(d => documents.set(d.doc_id, d));
        const pagination = page.pagination;
        if (!Number.isInteger(pagination.total_nodes)) throw Error('Update the backend to load all nodes and graph pages. No partial graph was displayed.');
        const nextEdge = pagination.next_edge_offset, nextNode = pagination.next_node_offset;
        if (nextEdge < pagination.total_edges || nextNode < pagination.total_nodes) {
          if (nextEdge <= offset && nextNode <= nodeOffset) throw Error('Graph pagination made no progress.');
          message(`Loading graph: ${nodes.size} entities, ${edges.size} relationships…`);
        }
        offset = nextEdge; nodeOffset = nextNode;
        if (offset >= pagination.total_edges && nodeOffset >= pagination.total_nodes) break;
      } while (context === current);
      if (!changed) return { ...first, nodes: [...nodes.values()], edges: [...edges.values()], documents: [...documents.values()] };
    }
    throw Error('The graph changed while loading. Waiting for the next refresh; the previous complete graph is retained.');
  }
  async function poll(context = current) {
    if (!context || context !== current || context.polling) return;
    clearTimeout(timer); context.polling = true;
    try {
      const [project, documents, activity] = await Promise.all([
        api(projectPath(context), {}, context.controller.signal),
        api(`${projectPath(context)}/documents?limit=${context.documentLimit}`, {}, context.controller.signal),
        api(`${projectPath(context)}/activity?limit=100${context.cursor == null ? '' : '&after=' + context.cursor}`, {}, context.controller.signal)
      ]);
      if (context !== current) return;
      context.project = project; context.documents = documents; showProject(context);
      if (context.revision !== project.graph_revision) {
        const graph = await graphSnapshot(context);
        if (!graph || context !== current) return;
        explorer.setGraph(graph, { name: project.name, preserveView: !!context.revision });
        context.revision = graph.revision;
        explorer.setEmptyState('No graph entities yet', 'The graph will appear as seeds and extracted relationships are published.');
      }
      showLive(context, activity);
      message('');
      context.failures = 0;
    } catch (error) { context.failures = (context.failures || 0) + 1; $('liveStatus').textContent = 'Updates interrupted; retrying'; handleError(error, context); }
    finally {
      context.polling = false;
      if (context === current) {
        const interval = document.hidden ? 15000 : context.moreActivity ? 100 : activeStates.has(context.project.status) ? 1000 : 5000;
        timer = setTimeout(() => poll(context), Math.min(30000, interval * 2 ** Math.min(context.failures || 0, 5)));
      }
    }
  }
  function selectProject(id) {
    abortProject(); message('');
    explorer.clear(id ? 'Loading project…' : 'Select a project', id ? 'Reading the latest committed graph from SQLite.' : 'Choose a server project from the selector.');
    const project = session.projects.find(p => p.id === id);
    if (project) {
      current = { id, project, controller: new AbortController(), documents: [], documentLimit: 50, revision: '', polling: false, busy: false, cursor: null, events: [] };
      showProject(current); poll(current);
    } else { $('projectState').textContent = session.connected ? 'Connected' : 'Disconnected'; updateControls(); populateProjects(); }
  }
  function openProject(edit) {
    editing = edit; $('projectError').textContent = ''; $('uploadMessage').textContent = '';
    const project = edit ? current?.project : null;
    $('projectDialogTitle').textContent = edit ? 'Project settings and activity' : 'New project';
    $('projectName').value = project?.name || ''; $('projectSeeds').value = project?.seeds.join('\n') || '';
    for (const [key, , value] of limits) $(key).value = project?.limits[key] ?? value;
    $('saveProject').textContent = edit ? 'Save settings' : 'Create project';
    $('projectActivity').hidden = !edit; updateControls();
    $('projectDialog').showModal();
  }

  async function connectSession(base, token) {
    disconnected(); session.base = base; session.token = token;
    const request = new AbortController(); session.request = request;
    const projects = await api('/projects', {}, request.signal);
    if (session.request !== request) return;
    session.projects = projects; session.connected = true; $('apiToken').value = '';
    reconnectAttempt = 0; $('connectionDialog').close(); selectProject('');
  }
  function scheduleReconnect(error) {
    if (!session.localProxy || reconnectAttempt >= 5 || (error.status && ![403, 502, 503, 504].includes(error.status))) return;
    const delay = Math.min(30000, 2000 * 2 ** reconnectAttempt++);
    clearTimeout(reconnectTimer);
    message(`${error.message} Retrying in ${delay / 1000}s; you can also use Reconnect.`);
    reconnectTimer = setTimeout(connectLocalProxy, delay);
  }
  async function connectLocalProxy() {
    if (session.connecting) return;
    session.connecting = true; $('connectButton').disabled = true;
    try {
      const response = await fetch('/schain-dev-config', { cache: 'no-store', credentials: 'omit', redirect: 'error' });
      if (!response.ok && !session.localProxy) return;
      if (!response.ok) throw Error('Local development server unavailable. Restart it and use Reconnect.');
      const config = await response.json();
      if (config.mode !== 'local-proxy' || config.api_prefix !== '/api' || !/^[A-Za-z0-9_-]{32,}$/.test(config.csrf_token)) return;
      session.localProxy = true; session.csrf = config.csrf_token;
      $('connectButton').textContent = 'Reconnect';
      await connectSession('/api', '');
    } catch (error) {
      if (session.localProxy && error.name !== 'AbortError') {
        disconnected('Cannot connect through the local proxy. Check the SSH tunnel and backend.');
        message(error.message); scheduleReconnect(error);
      }
    } finally { session.connecting = false; $('connectButton').disabled = false; }
  }
  $('connectionForm').onsubmit = async event => {
    event.preventDefault();
    try {
      const endpoint = new URL($('apiEndpoint').value);
      if (!['http:', 'https:'].includes(endpoint.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(endpoint.hostname) || endpoint.username || endpoint.password || endpoint.search || endpoint.hash) throw Error('Use a loopback HTTP endpoint forwarded by SSH.');
      $('connectionError').textContent = '';
      await connectSession(endpoint.href.replace(/\/$/, ''), $('apiToken').value);
    } catch (error) {
      if (error.name !== 'AbortError') $('connectionError').textContent = error.message;
    }
  };
  $('connectButton').onclick = () => {
    if (session.localProxy) { clearTimeout(reconnectTimer); reconnectAttempt = 0; connectLocalProxy(); }
    else $('connectionDialog').showModal();
  };
  $('newProject').onclick = () => openProject(false);
  $('manageProject').onclick = () => openProject(true);
  $('projectSelect').onchange = event => selectProject(event.target.value);
  document.querySelectorAll('[data-close-dialog]').forEach(button => button.onclick = () => button.closest('dialog').close());
  $('projectForm').onsubmit = async event => {
    event.preventDefault(); if (saving) return;
    const context = current, request = session.request;
    const payload = { name: $('projectName').value.trim(), seeds: $('projectSeeds').value.split('\n').map(x => x.trim()).filter(Boolean), limits: Object.fromEntries(limits.map(([key]) => [key, Number($(key).value)])) };
    if (!payload.name || !payload.seeds.length || new Set(payload.seeds).size !== payload.seeds.length) { $('projectError').textContent = 'Enter a name and at least one unique Tier-0 company.'; return; }
    saving = true; updateControls();
    try {
      const project = await api(editing ? projectPath(context) : '/projects', { method: editing ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }, request.signal);
      const projects = await api('/projects', {}, request.signal);
      if (session.request !== request || current !== context) return;
      session.projects = projects; $('projectDialog').close(); populateProjects();
      if (current?.id === project.id) { current.project = project; await poll(current); }
      else { $('projectSelect').value = project.id; selectProject(project.id); }
    } catch (error) { if (error.name !== 'AbortError' && current === context) { $('projectError').textContent = error.message; handleError(error, context); } }
    finally { saving = false; updateControls(); }
  };
  for (const [id, action] of [['startRun', 'start'], ['pauseRun', 'pause'], ['resumeRun', 'resume']]) $(id).onclick = async () => {
    const context = current; if (!context || context.busy) return;
    context.busy = true; updateControls();
    try {
      const project = await api(`${projectPath(context)}/${action}`, { method: 'POST' }, context.controller.signal);
      if (context !== current) return;
      context.project = project; showProject(context); await poll(context);
    } catch (error) { handleError(error, context); }
    finally { context.busy = false; if (context === current) updateControls(); }
  };
  $('documentForm').onsubmit = async event => {
    event.preventDefault(); const context = current; if (!context || context.busy) return;
    const file = $('documentInput').files[0];
    if (!file || !/\.(txt|md|pdf)$/i.test(file.name) || !file.size || file.size > 10 * 1024 * 1024) { $('uploadMessage').textContent = 'Select a non-empty .txt, .md or .pdf file up to 10 MiB.'; return; }
    context.busy = true; updateControls();
    try {
      const project = await api(`${projectPath(context)}/documents`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name) }, body: file }, context.controller.signal);
      if (context !== current) return;
      context.project = project; $('documentForm').reset(); $('uploadMessage').textContent = 'Queued for the server worker. Processing and publication continue if this page closes.';
      showProject(context); await poll(context);
    } catch (error) { if (context === current) $('uploadMessage').textContent = error.message; handleError(error, context); }
    finally { context.busy = false; if (context === current) updateControls(); }
  };
  $('moreDocuments').onclick = () => { if (current) { current.documentLimit = Math.min(200, current.documentLimit + 50); poll(current); } };
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
  $('openJsonBtn').disabled = true;
  $('openJsonBtn').title = 'Use python -m schain view for a separate offline graph';
  disconnected();
  if (location.protocol === 'file:') message('Serve the generated directory locally with python -m http.server; file:// is unsupported.');
  else connectLocalProxy();
})();
