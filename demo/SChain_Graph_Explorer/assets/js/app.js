(() => {
  'use strict';
  const ENTITY_COLORS = {
    Company: '#827cff',
    Product: '#55d2f2',
    Material: '#67d8aa',
    Location: '#e9bd6d',
    Person: '#e98daf',
    Mine: '#bb95e9'
  };
  const RELATION_COLORS = { suppliesTo: '#6da8ff', produces: '#67d8aa', locatedIn: '#e9bd6d', owns: '#bb95e9' };
  const TIER_COLORS = ['#e9bd6d', '#8e9dff', '#62cbe9', '#67d8aa', '#e98daf', '#bc99ec', '#8d9ab3'];
  const FALLBACK_COLOR = '#8d9ab3';
  const elementById = id => document.getElementById(id);
  const clampValue = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
  const formatCount = n => Number(n || 0).toLocaleString();
  const escapeHtml = value => String(value ?? '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
  const getDomainName = url => {
    try {
      return new URL(url).hostname.replace(/^www\./, '');
    } catch {
      return url || '—';
    }
  };
  const getRelationLabel = r => ({ suppliesTo: 'supplies to', produces: 'produces', locatedIn: 'located in', owns: 'owns' }[r] || r || 'related to');
  const getSourceTypeLabel = v => ({ full_text: 'Full text', title_snippet: 'Title / snippet' }[v] || String(v || 'Document').replaceAll('_', ' '));
  const hashString = value => {
    const text = String(value);
    let hash = 2166136261 >>> 0;
    for (let index = 0; index < text.length; index++) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  };
  const stableFraction = (value, salt = '') => (hashString(`${value}|${salt}`) % 100000) / 100000;
  const state = {
    datasets: new Map(),
    currentName: null,
    data: null,
    nodeMap: new Map(),
    docMap: new Map(),
    viewMode: 'tiered',
    dimension: '3d',
    entityScope: 'all',
    focalId: null,
    entityFilters: new Set(),
    relationFilters: new Set(),
    tierFilters: new Set(),
    selectedNodes: new Set(),
    selectedEdges: new Set(),
    detailMode: false,
    showContext: false,
    focusNodes: new Set(),
    focusEdges: new Set(),
    labels: false,
    sideTab: 'info',
    panelScope: 'visible',
    panelRelation: 'all',
    panelContent: 'all',
    panelScore: 0,
    panelSearch: ''
  };
  class GraphRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d', { alpha: true });
      this.mode2D = false;
      this.nodes = [];
      this.edges = [];
      this.nodeMap = new Map();
      this.projected = new Map();
      this.edgeProjected = [];
      this.camera = {
        yaw: -.42,
        pitch: -.22,
        zoom: 1,
        panX: 0,
        panY: 0
      };
      this.dragging = false;
      this.dragStarted = false;
      this.dragMode = 'rotate';
      this.pointerDown = null;
      this.lastPointer = null;
      this.activePointerId = null;
      this.downHit = null;
      this.hoverNode = null;
      this.hoverEdge = null;
      this.selectedNodes = new Set();
      this.selectedEdges = new Set();
      this.focusNodeIds = new Set();
      this.focusEdgeIds = new Set();
      this.contextMode = false;
      this.allLabels = false;
      this.particles = true;
      this.tiered = false;
      this.stars = Array.from({ length: 72 }, (_, i) => ({ x: stableFraction(i, 'sx'), y: stableFraction(i, 'sy'), a: .04 + stableFraction(i, 'sa') * .14, r: .3 + stableFraction(i, 'sr') * .8 }));
      this.onNodeClick = null;
      this.onEdgeClick = null;
      this.onBlankClick = null;
      this._bind();
      this.resize();
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      requestAnimationFrame(t => this._frame(t));
    }
    _eventPoint(e) {
      const r = this.canvas.getBoundingClientRect();
      return { x: (e.clientX - r.left) * this.dpr, y: (e.clientY - r.top) * this.dpr };
    }
    _bind() {
      const begin = e => {
        if (e.button !== 0 && e.button !== 1 && e.button !== 2) return;
        e.preventDefault();
        const pt = this._eventPoint(e);
        this._updateHover(pt.x, pt.y);
        this.dragging = true;
        this.dragStarted = false;
        this.activePointerId = e.pointerId;
        this.pointerDown = { x: e.clientX, y: e.clientY };
        this.lastPointer = { x: e.clientX, y: e.clientY };
        this.downHit = this.hoverNode ? { kind: 'node', id: this.hoverNode.id } : this.hoverEdge ? { kind: 'edge', id: this.hoverEdge.edge.id } : null;
        this.dragMode = this.mode2D ? 'pan' : ((e.ctrlKey || e.metaKey || e.altKey || e.button === 1 || e.button === 2) ? 'pan' : 'rotate');
        try {
          this.canvas.setPointerCapture?.(e.pointerId);
        } catch {
        }
      };
      const move = e => {
        if (this.dragging && this.activePointerId === e.pointerId) {
          const total = this.pointerDown ? Math.hypot(e.clientX - this.pointerDown.x, e.clientY - this.pointerDown.y) : 0;
          if (!this.dragStarted && total >= 3) {
            this.dragStarted = true;
            this.canvas.classList.add('dragging');
          }
          if (this.dragStarted) {
            const dx = e.clientX - this.lastPointer.x, dy = e.clientY - this.lastPointer.y;
            if (this.dragMode === 'pan') {
              this.camera.panX += dx;
              this.camera.panY += dy;
            }
            else {
              this.camera.yaw += dx * .006;
              this.camera.pitch = clampValue(this.camera.pitch + dy * .005, -1.25, 1.25);
            }
          }
          this.lastPointer = { x: e.clientX, y: e.clientY };
          return;
        }
        const pt = this._eventPoint(e);
        this._updateHover(pt.x, pt.y);
      };
      const end = e => {
        if (!this.dragging || this.activePointerId !== e.pointerId) return;
        const moved = this.pointerDown ? Math.hypot(e.clientX - this.pointerDown.x, e.clientY - this.pointerDown.y) : 99;
        const wasDrag = this.dragStarted || moved >= 5;
        this.dragging = false;
        this.dragStarted = false;
        this.activePointerId = null;
        this.canvas.classList.remove('dragging');
        try {
          this.canvas.releasePointerCapture?.(e.pointerId);
        } catch {
        }
        const pt = this._eventPoint(e);
        this._updateHover(pt.x, pt.y);
        if (!wasDrag && e.button === 0) {
          let node = this.hoverNode, edge = this.hoverEdge?.edge || null;
          if (!node && !edge && this.downHit?.kind === 'node') node = this.nodeMap.get(this.downHit.id) || null;
          if (!node && !edge && this.downHit?.kind === 'edge') edge = this.edges.find(x => x.id === this.downHit.id) || null;
          if (node && this.onNodeClick) this.onNodeClick(node, { shiftKey: e.shiftKey });
          else if (edge && this.onEdgeClick) this.onEdgeClick(edge, { shiftKey: e.shiftKey });
          else if (this.onBlankClick) this.onBlankClick({ shiftKey: e.shiftKey });
        }
        this.downHit = null;
      };
      this.canvas.addEventListener('pointerdown', begin);
      this.canvas.addEventListener('pointermove', move);
      this.canvas.addEventListener('pointerup', end);
      this.canvas.addEventListener('pointercancel', end);
      window.addEventListener('pointermove', e => {
        if (this.dragging && this.activePointerId === e.pointerId && e.target !== this.canvas) move(e);
      });
      window.addEventListener('pointerup', e => {
        if (this.dragging && this.activePointerId === e.pointerId && e.target !== this.canvas) end(e);
      });
      this.canvas.addEventListener('pointerleave', () => {
        if (!this.dragging) {
          this.hoverNode = null;
          this.hoverEdge = null;
          this.canvas.style.cursor = 'grab';
        }
      });
      this.canvas.addEventListener('contextmenu', e => e.preventDefault());
      this.canvas.addEventListener('wheel', e => {
        e.preventDefault();
        this.camera.zoom = clampValue(this.camera.zoom * Math.exp(-e.deltaY * .001), .2, 3.8);
      }, { passive: false });
      this.canvas.addEventListener('dblclick', e => {
        const pt = this._eventPoint(e);
        this._updateHover(pt.x, pt.y);
        if (this.hoverNode) this.focusNode(this.hoverNode.id, 1.55);
      });
    }
    resize() {
      const r = this.canvas.getBoundingClientRect();
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.width = Math.max(1, Math.round(r.width * this.dpr));
      this.height = Math.max(1, Math.round(r.height * this.dpr));
      if (this.canvas.width !== this.width || this.canvas.height !== this.height) {
        this.canvas.width = this.width;
        this.canvas.height = this.height;
      }
    }
    setMode2D(flag) {
      const next = !!flag;
      if (next === this.mode2D) return;
      this.mode2D = next;
      this.camera = next ? {
        yaw: 0,
        pitch: 0,
        zoom: 1,
        panX: 0,
        panY: 0
      } : {
        yaw: -.42,
        pitch: -.22,
        zoom: 1,
        panX: 0,
        panY: 0
      };
      this.fit();
    }
    setGraph(nodes, edges, positions, { tiered = false, focusNodeIds = null, focusEdgeIds = null, contextMode = false, fit = true } = {}) {
      this.nodes = nodes.map(n => ({ ...n, _pos: positions.get(n.id) || { x: 0, y: 0, z: 0, tier: null } }));
      this.nodeMap = new Map(this.nodes.map(n => [n.id, n]));
      this.edges = edges.filter(e => this.nodeMap.has(e.source) && this.nodeMap.has(e.target));
      this.tiered = !!tiered;
      this.focusNodeIds = new Set(focusNodeIds || []);
      this.focusEdgeIds = new Set(focusEdgeIds || []);
      this.contextMode = !!contextMode;
      this.hoverNode = null;
      this.hoverEdge = null;
      if (fit) this.fit();
    }
    setSelection(nodeIds, edgeIds) {
      this.selectedNodes = new Set(nodeIds || []);
      this.selectedEdges = new Set(edgeIds || []);
    }
    fit() {
      if (!this.nodes.length) return;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, maxZ = 1;
      for (const n of this.nodes) {
        const p = n._pos;
        minX = Math.min(minX, p.x);
        maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y);
        maxY = Math.max(maxY, p.y);
        maxZ = Math.max(maxZ, Math.abs(p.z || 0));
      }
      const w = Math.max(190, maxX - minX + 210), h = Math.max(190, maxY - minY + 210), cw = this.width / this.dpr, ch = this.height / this.dpr;
      if (this.mode2D) {
        this.camera.zoom = clampValue(Math.min(cw / w, ch / h) * .88, .2, 1.65);
        this.camera.panX = -(minX + maxX) / 2 * this.camera.zoom;
        this.camera.panY = -(minY + maxY) / 2 * this.camera.zoom;
      } else {
        this.camera.zoom = clampValue(Math.min(cw, ch) / Math.max(660, Math.max(w, h, maxZ * 2) * 1.55), .34, 1.45);
        this.camera.panX = 0;
        this.camera.panY = 0;
      }
    }
    focusNode(id, zoom = 1.5) {
      const p = this.projected.get(id);
      if (!p) return;
      const cx = this.width / 2, cy = this.height / 2;
      this.camera.panX += (cx - p.x) / this.dpr;
      this.camera.panY += (cy - p.y) / this.dpr;
      this.camera.zoom = clampValue(Math.max(this.camera.zoom, zoom), .3, 3);
    }
    _project(pos) {
      const { zoom, panX, panY } = this.camera;
      if (this.mode2D) return { x: this.width / 2 + panX * this.dpr + pos.x * zoom * this.dpr, y: this.height / 2 + panY * this.dpr + pos.y * zoom * this.dpr, z: 0, scale: zoom * this.dpr };
      const { yaw, pitch } = this.camera, cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = pos.x * cy - pos.z * sy,
        z1 = pos.x * sy + pos.z * cy,
        y2 = pos.y * cp - z1 * sp,
        z2 = pos.y * sp + z1 * cp,
        perspective = clampValue(900 / (900 + z2), .46, 1.8),
        scale = zoom * perspective * this.dpr;
      return { x: this.width / 2 + panX * this.dpr + x1 * scale, y: this.height / 2 + panY * this.dpr + y2 * scale, z: z2, scale };
    }
    _frame(t) {
      this._draw(t);
      requestAnimationFrame(x => this._frame(x));
    }
    _draw(t) {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.width, this.height);
      this._drawBackground(ctx);
      if (!this.nodes.length) return;
      this.projected.clear();
      for (const n of this.nodes) this.projected.set(n.id, this._project(n._pos));
      if (this.tiered) this._drawTierGuides(ctx);
      this.edgeProjected = [];
      for (const e of this.edges) {
        const a = this.projected.get(e.source), b = this.projected.get(e.target);
        if (!a || !b) continue;
        this.edgeProjected.push({ edge: e, a, b });
        this._drawEdge(ctx, e, a, b, t);
      }
      const ordered = this.mode2D ? [...this.nodes] : [...this.nodes].sort((a, b) => (this.projected.get(b.id)?.z || 0) - (this.projected.get(a.id)?.z || 0));
      for (const n of ordered) this._drawNode(ctx, n, this.projected.get(n.id));
    }
    _drawBackground(ctx) {
      ctx.save();
      for (const s of this.stars) {
        ctx.globalAlpha = s.a;
        ctx.fillStyle = '#a6bce8';
        ctx.beginPath();
        ctx.arc(s.x * this.width, s.y * this.height, s.r * this.dpr, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
    _drawTierGuides(ctx) {
      const groups = new Map();
      for (const n of this.nodes) {
        const t = n._pos?.tier;
        if (t === null || t === undefined) continue;
        if (!groups.has(t)) groups.set(t, []);
        groups.get(t).push(n);
      }
      const tiers = [...groups.keys()].sort((a, b) => a - b);
      if (this.mode2D) {
        ctx.save();
        for (const t of tiers) {
          const arr = groups.get(t), ps = arr.map(n => this.projected.get(n.id)).filter(Boolean);
          if (!ps.length) continue;
          const xs = ps.map(p => p.x), ys = ps.map(p => p.y), x = (Math.min(...xs) + Math.max(...xs)) / 2, y = (Math.min(...ys) + Math.max(...ys)) / 2;
          const widthHint = Math.max(...arr.map(n => Number(n._pos?._tierWidth || 0))) || 0,
            heightHint = Math.max(...arr.map(n => Number(n._pos?._tierHeight || 0))) || 0,
            w = Math.max(170 * this.dpr, widthHint ? widthHint * this.scale * .92 : (Math.max(...xs) - Math.min(...xs)) + 120 * this.dpr),
            h = Math.max(150 * this.dpr, heightHint ? heightHint * this.scale * .92 : (Math.max(...ys) - Math.min(...ys)) + 120 * this.dpr);
          ctx.fillStyle = 'rgba(89,107,139,.035)';
          ctx.strokeStyle = 'rgba(118,139,171,.12)';
          ctx.lineWidth = this.dpr;
          this._roundRect(ctx, x - w / 2, y - h / 2, w, h, 16 * this.dpr);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = 'rgba(150,166,190,.45)';
          ctx.font = `${10 * this.dpr}px Inter, sans-serif`;
          ctx.fillText(t === 99 ? 'Other' : `Tier ${t}`, x - w / 2 + 12 * this.dpr, y - h / 2 + 18 * this.dpr);
        }
        ctx.restore();
        return;
      }
      const planeColors = ['233,190,109', '142,157,255', '98,203,233', '103,216,170', '233,141,175', '188,153,236'];
      ctx.save();
      for (const t of tiers) {
        const arr = groups.get(t), p0 = arr[0]?._pos;
        if (!p0) continue;
        let maxY = 150, maxZ = 150;
        for (const n of arr) {
          maxY = Math.max(maxY, Math.abs(n._pos.y) + 70);
          maxZ = Math.max(maxZ, Math.abs(n._pos.z) + 70);
        }
        const x = p0.x,
          corners = [{ x, y: -maxY, z: -maxZ }, { x, y: maxY, z: -maxZ }, { x, y: maxY, z: maxZ }, { x, y: -maxY, z: maxZ }].map(p => this._project(p));
        const rgb = t === 99 ? '130,145,170' : (planeColors[t % planeColors.length] || '130,145,170');
        ctx.beginPath();
        ctx.moveTo(corners[0].x, corners[0].y);
        for (let i = 1; i < corners.length; i++)ctx.lineTo(corners[i].x, corners[i].y);
        ctx.closePath();
        ctx.fillStyle = `rgba(${rgb},.025)`;
        ctx.strokeStyle = `rgba(${rgb},.15)`;
        ctx.lineWidth = this.dpr;
        ctx.fill();
        ctx.stroke();
        const label = this._project({ x, y: -maxY, z: maxZ });
        ctx.fillStyle = `rgba(${rgb},.55)`;
        ctx.font = `${10 * this.dpr}px Inter, sans-serif`;
        ctx.fillText(t === 99 ? 'Other' : `Tier ${t}`, label.x + 8 * this.dpr, label.y - 7 * this.dpr);
      }
      ctx.restore();
    }
    _edgeInFocus(e) {
      if (!this.contextMode) return true;
      if (this.focusEdgeIds.has(e.id)) return true;
      return this.focusNodeIds.has(e.source) && this.focusNodeIds.has(e.target);
    }
    _drawEdge(ctx, e, a, b, t) {
      const selected = this.selectedEdges.has(e.id),
        related = selected || this.selectedNodes.has(e.source) || this.selectedNodes.has(e.target),
        inFocus = this._edgeInFocus(e),
        isContext = this.contextMode && !inFocus;
      const c = isContext ? '#7a8492' : (RELATION_COLORS[e.relation] || FALLBACK_COLOR);
      ctx.save();
      ctx.globalAlpha = isContext ? .11 : (selected ? 1 : related ? .82 : .38);
      ctx.lineWidth = (selected ? 3 : related ? 1.8 : 1) * this.dpr;
      ctx.strokeStyle = c;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      if (!isContext) {
        const dx = b.x - a.x,
          dy = b.y - a.y,
          len = Math.hypot(dx, dy) || 1,
          ux = dx / len,
          uy = dy / len,
          arrow = selected ? 7 : 5,
          px = b.x - ux * 11 * this.dpr,
          py = b.y - uy * 11 * this.dpr;
        ctx.fillStyle = c;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px - ux * arrow * this.dpr - uy * arrow * .6 * this.dpr, py - uy * arrow * this.dpr + ux * arrow * .6 * this.dpr);
        ctx.lineTo(px - ux * arrow * this.dpr + uy * arrow * .6 * this.dpr, py - uy * arrow * this.dpr - ux * arrow * .6 * this.dpr);
        ctx.closePath();
        ctx.fill();
        if (this.particles && !this.mode2D && e.relation === 'suppliesTo' && !selected) {
          const phase = (t * .00012 + stableFraction(e.id, 'p')) % 1, x = a.x + (b.x - a.x) * phase, y = a.y + (b.y - a.y) * phase;
          ctx.globalAlpha = .45;
          ctx.fillStyle = c;
          ctx.beginPath();
          ctx.arc(x, y, 1.5 * this.dpr, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }
    _nodeInFocus(n) {
      return !this.contextMode || this.focusNodeIds.has(n.id);
    }
    _drawNode(ctx, n, p) {
      const selected = this.selectedNodes.has(n.id),
        edgeEndpoint = [...this.selectedEdges].some(id => {
          const e = this.edges.find(x => x.id === id);
          return e && (e.source === n.id || e.target === n.id);
        }),
        inFocus = this._nodeInFocus(n),
        isContext = this.contextMode && !inFocus,
        hover = this.hoverNode?.id === n.id,
        c = isContext ? '#707a88' : (ENTITY_COLORS[n.type] || FALLBACK_COLOR),
        base = (n.is_seed ? 7.4 : 5.8),
        r = (base + (selected ? 2.5 : edgeEndpoint ? 1.4 : hover ? 1.1 : 0)) * clampValue(p.scale / this.dpr, .72, 1.28) * this.dpr;
      p.hitRadius = Math.max(9 * this.dpr, r + 4 * this.dpr);
      ctx.save();
      ctx.globalAlpha = isContext ? .22 : 1;
      if (selected || edgeEndpoint) {
        ctx.globalAlpha = isContext ? .32 : 1;
        ctx.strokeStyle = selected ? '#ffffff' : '#9aa8ff';
        ctx.lineWidth = 1.6 * this.dpr;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 5 * this.dpr, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.shadowColor = isContext ? 'transparent' : c;
      ctx.shadowBlur = (selected ? 18 : hover ? 12 : 6) * this.dpr;
      ctx.fillStyle = c;
      ctx.strokeStyle = isContext ? 'rgba(180,190,205,.25)' : 'rgba(255,255,255,.4)';
      ctx.lineWidth = .75 * this.dpr;
      this._nodeShape(ctx, n.type, p.x, p.y, r);
      ctx.shadowBlur = 0;
      if ((this.allLabels || selected || hover || edgeEndpoint || n.is_seed) && (!isContext || selected || hover)) {
        const label = n.name || n.id, fs = (this.mode2D ? 10.5 : 9.5) * this.dpr;
        ctx.font = `${selected ? '650' : '560'} ${fs}px Inter, sans-serif`;
        const tw = ctx.measureText(label).width, pad = 5 * this.dpr, x = p.x + 9 * this.dpr, y = p.y - r - 7 * this.dpr;
        ctx.fillStyle = 'rgba(7,11,16,.82)';
        this._roundRect(ctx, x - pad, y - fs, pad * 2 + tw, fs + 8 * this.dpr, 6 * this.dpr);
        ctx.fill();
        ctx.fillStyle = isContext ? '#9ba5b2' : '#e7edf5';
        ctx.fillText(label, x, y + 2 * this.dpr);
      }
      ctx.restore();
    }
    _nodeShape(ctx, type, x, y, r) {
      ctx.beginPath();
      if (type === 'Product') {
        ctx.moveTo(x, y - r);
        ctx.lineTo(x + r, y);
        ctx.lineTo(x, y + r);
        ctx.lineTo(x - r, y);
        ctx.closePath();
      } else if (type === 'Material') {
        ctx.moveTo(x - r * .9, y - r * .6);
        ctx.lineTo(x + r * .9, y - r * .6);
        ctx.lineTo(x + r * .9, y + r * .6);
        ctx.lineTo(x - r * .9, y + r * .6);
        ctx.closePath();
      } else if (type === 'Location') {
        ctx.moveTo(x, y - r);
        ctx.lineTo(x + r * .88, y + r * .75);
        ctx.lineTo(x - r * .88, y + r * .75);
        ctx.closePath();
      } else if (type === 'Mine') {
        for (let i = 0; i < 5; i++) {
          const a = -Math.PI / 2 + i * 2 * Math.PI / 5, px = x + Math.cos(a) * r, py = y + Math.sin(a) * r;
          i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.closePath();
      } else ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    _roundRect(ctx, x, y, w, h, r) {
      const rr = Math.min(r, w / 2, h / 2);
      ctx.beginPath();
      ctx.moveTo(x + rr, y);
      ctx.arcTo(x + w, y, x + w, y + h, rr);
      ctx.arcTo(x + w, y + h, x, y + h, rr);
      ctx.arcTo(x, y + h, x, y, rr);
      ctx.arcTo(x, y, x + w, y, rr);
      ctx.closePath();
    }
    _updateHover(x, y) {
      let best = null, bestD = Infinity;
      for (const n of this.nodes) {
        const p = this.projected.get(n.id);
        if (!p) continue;
        const d = Math.hypot(x - p.x, y - p.y);
        if (d < (p.hitRadius || 10 * this.dpr) + 5 * this.dpr && d < bestD) {
          best = n;
          bestD = d;
        }
      }
      this.hoverNode = best;
      this.hoverEdge = null;
      if (!best) {
        let eb = null, ed = 8 * this.dpr;
        for (const ep of this.edgeProjected) {
          const d = this._pointSegDistance(x, y, ep.a.x, ep.a.y, ep.b.x, ep.b.y);
          if (d < ed) {
            eb = ep;
            ed = d;
          }
        }
        this.hoverEdge = eb;
      }
      this.canvas.style.cursor = (this.hoverNode || this.hoverEdge) ? 'pointer' : 'grab';
    }
    _pointSegDistance(px, py, x1, y1, x2, y2) {
      const dx = x2 - x1, dy = y2 - y1, l2 = dx * dx + dy * dy;
      if (!l2) return Math.hypot(px - x1, py - y1);
      let t = ((px - x1) * dx + (py - y1) * dy) / l2;
      t = clampValue(t, 0, 1);
      return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
    }
  }

  const graphRenderer = new GraphRenderer(elementById('graphCanvas'));
  graphRenderer.onNodeClick = (node, ev) => selectNode(node.id, !!ev?.shiftKey);
  graphRenderer.onEdgeClick = (edge, ev) => selectEdge(edge.id, !!ev?.shiftKey);
  graphRenderer.onBlankClick = ev => {
    if (!ev?.shiftKey) clearSelection();
  };
  function validateGraph(data) {
    if (!data || typeof data !== 'object') throw new Error('JSON root must be an object.');
    if (!Array.isArray(data.nodes) || !Array.isArray(data.edges)) throw new Error('Expected nodes[] and edges[] arrays.');
    const ids = new Set(data.nodes.map(n => n.id)), bad = data.edges.filter(e => !ids.has(e.source) || !ids.has(e.target));
    if (bad.length) throw new Error(`${bad.length} relationships reference missing entities.`);
    return true;
  }
  function registerDataset(name, data, origin = 'loaded') {
    validateGraph(data);
    let finalName = name || `dataset-${state.datasets.size + 1}.json`, base = finalName, i = 2;
    while (state.datasets.has(finalName)) finalName = `${base} (${i++})`;
    state.datasets.set(finalName, { name: finalName, data, origin });
    rebuildDatasetSelect();
    return finalName;
  }
  function rebuildDatasetSelect() {
    const sel = elementById('datasetSelect'), old = state.currentName;
    sel.innerHTML = '';
    for (const [name, item] of state.datasets) {
      const o = document.createElement('option');
      o.value = name;
      o.textContent = name + (item.origin === 'demo' ? ' · bundled' : '');
      sel.appendChild(o);
    }
    if (old && state.datasets.has(old)) sel.value = old;
  }
  function loadDataset(name) {
    const item = state.datasets.get(name);
    if (!item) return;
    validateGraph(item.data);
    state.currentName = name;
    state.data = item.data;
    state.nodeMap = new Map(state.data.nodes.map(n => [n.id, n]));
    state.docMap = new Map((state.data.documents || []).map(d => [d.doc_id, d]));
    state.selectedNodes.clear();
    state.selectedEdges.clear();
    state.focusNodes.clear();
    state.focusEdges.clear();
    state.detailMode = false;
    state.showContext = false;
    state.panelScope = 'visible';
    state.panelRelation = 'all';
    state.panelContent = 'all';
    state.panelScore = 0;
    state.panelSearch = '';
    state.entityScope = 'all';
    state.entityFilters = new Set(state.data.nodes.map(n => n.type).filter(Boolean));
    state.relationFilters = new Set(state.data.edges.map(e => e.relation).filter(Boolean));
    state.tierFilters = new Set(state.data.nodes.map(n => n.discovery_tier).filter(t => t != null).map(String));
    const seedName = state.data.metadata?.seeds?.[0],
      seed = state.data.nodes.find(n => n.is_seed) || state.data.nodes.find(n => n.name === seedName) || state.data.nodes.find(n => n.type === 'Company');
    state.focalId = seed?.id || state.data.nodes[0]?.id || null;
    elementById('datasetSelect').value = name;
    renderAll();
  }
  function countsBy(arr, key) {
    return arr.reduce((o, x) => {
      const k = String(x?.[key] ?? 'Unknown');
      o[k] = (o[k] || 0) + 1;
      return o;
    }, {});
  }
  function selectionCount() {
    return state.selectedNodes.size + state.selectedEdges.size;
  }
  function renderAll() {
    renderHeader();
    renderGraphControls();
    refreshGraph();
    renderSelectionBar();
    renderSide();
  }
  function renderHeader() {
    const seeds = state.data?.metadata?.seeds || [];
    elementById('datasetTitle').textContent = seeds.length ? `${seeds.slice(0, 2).join(' · ')}${seeds.length > 2 ? ` +${seeds.length - 2}` : ''}` : 'Supply chain graph';
    elementById('datasetMeta').textContent = `${formatCount(state.data?.nodes?.length)} entities · ${formatCount(state.data?.edges?.length)} facts`;
  }
  function renderGraphControls() {
    const d = state.data,
      companies = d.nodes.filter(n => n.type === 'Company').sort((a, b) => (Number(b.is_seed) - Number(a.is_seed)) || (a.name || '').localeCompare(b.name || ''));
    elementById('focalCompany').innerHTML = companies.map(n => `<option value="${escapeHtml(n.id)}">${escapeHtml(n.name || n.id)}</option>`).join('');
    if (state.focalId) elementById('focalCompany').value = state.focalId;
    document.querySelector('.view-cluster').classList.toggle('upstream', state.viewMode === 'supply');
    syncModeButtons();
    renderFilters('relationFilters', countsBy(d.edges, 'relation'), state.relationFilters, RELATION_COLORS, true, getRelationLabel);
    renderFilters('tierFilters', countsBy(d.nodes.filter(n => n.discovery_tier != null), 'discovery_tier'), state.tierFilters, Object.fromEntries(TIER_COLORS.map((c, i) => [String(i), c])), false, k => `T${k}`);
  }
  function renderFilters(id, counts, activeSet, colors, relation = false, labeler = x => x) {
    const root = elementById(id);
    root.innerHTML = '';
    Object.keys(counts).sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).forEach(key => {
      const b = document.createElement('button');
      b.className = 'filter-item ' + (activeSet.has(key) ? 'active' : '');
      b.innerHTML = `<span class="filter-label"><i class="${relation ? 'relation-swatch' : 'swatch'}" style="--c:${colors[key] || FALLBACK_COLOR}"></i>${escapeHtml(labeler(key))}</span><span class="filter-count">${formatCount(counts[key])}</span>`;
      b.onclick = () => {
        activeSet.has(key) ? activeSet.delete(key) : activeSet.add(key);
        renderGraphControls();
        refreshGraph();
        renderSide();
      };
      root.appendChild(b);
    });
  }
  function syncModeButtons() {
    document.querySelectorAll('#viewMode button').forEach(b => b.classList.toggle('active', b.dataset.view === state.viewMode));
    document.querySelectorAll('#dimensionMode button').forEach(b => b.classList.toggle('active', b.dataset.dimension === state.dimension));
    document.querySelectorAll('#entityScopeMode button').forEach(b => b.classList.toggle('active', b.dataset.entityScope === state.entityScope));
  }

  function getCoreGraph() {
    const d = state.data,
      nodeIds = new Set(d.nodes.filter(n => n.is_supply_chain_node === true || n.discovery_tier != null || n.is_seed === true).map(n => n.id));
    let edges = d.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target) && (e.is_supply_chain_edge === true || e.relation === 'suppliesTo'));
    if (!edges.length) edges = d.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
    const used = new Set(edges.flatMap(e => [e.source, e.target]));
    for (const n of d.nodes) if (n.is_seed) used.add(n.id);
    return { nodes: d.nodes.filter(n => used.has(n.id)), edges, derivedTier: new Map(), upstream: new Set(used) };
  }
  function getSupplyFocusGraph() {
    const d = state.data, map = state.nodeMap, focal = state.focalId;
    if (!focal) return { nodes: d.nodes, edges: d.edges, derivedTier: new Map(), upstream: new Set() };
    const upstream = new Set([focal]), tier = new Map([[focal, 0]]), q = [focal];
    while (q.length) {
      const target = q.shift(), depth = tier.get(target) || 0;
      if (depth >= 8) continue;
      for (const e of d.edges) {
        if (e.relation !== 'suppliesTo' || e.target !== target) continue;
        const src = map.get(e.source);
        if (!src || !['Company', 'Mine'].includes(src.type)) continue;
        if (!upstream.has(src.id)) {
          upstream.add(src.id);
          tier.set(src.id, depth + 1);
          q.push(src.id);
        }
      }
    }
    const include = new Set(upstream);
    for (const e of d.edges) {
      if (upstream.has(e.source) || upstream.has(e.target)) {
        const t = map.get(e.target);
        if (e.relation === 'suppliesTo' && upstream.has(e.source) && t?.type === 'Company' && !upstream.has(e.target)) continue;
        include.add(e.source);
        include.add(e.target);
      }
    }
    const nodes = d.nodes.filter(n => include.has(n.id)),
      edges = d.edges.filter(e => include.has(e.source) && include.has(e.target)).filter(e => {
        const t = map.get(e.target);
        return !(e.relation === 'suppliesTo' && upstream.has(e.source) && t?.type === 'Company' && !upstream.has(e.target));
      });
    return { nodes, edges, derivedTier: tier, upstream };
  }
  function getBaseGraph() {
    if (state.viewMode === 'tiered') return { nodes: state.data.nodes, edges: state.data.edges, derivedTier: new Map(), upstream: new Set() };
    if (state.viewMode === 'network') return getCoreGraph();
    if (state.viewMode === 'supply') return getSupplyFocusGraph();
    return { nodes: state.data.nodes, edges: state.data.edges, derivedTier: new Map(), upstream: new Set() };
  }
  function applyStandardFilters(base) {
    const tierMap = state.viewMode === 'tiered' ? inferTierMap(base) : null,
      allTierValues = new Set(state.data.nodes.map(n => n.discovery_tier).filter(t => t != null).map(String)),
      allTiersSelected = !allTierValues.size || [...allTierValues].every(t => state.tierFilters.has(t));
    const ids = new Set(base.nodes.filter(n => {
      if (state.entityScope === 'companies' && n.type !== 'Company') return false;
      if (!state.entityFilters.has(n.type)) return false;
      if (state.viewMode === 'tiered') {
        const t = tierMap?.has(n.id) ? tierMap.get(n.id) : 99;
        if (t === 99) return allTiersSelected;
        return !state.tierFilters.size || state.tierFilters.has(String(t));
      }
      return n.discovery_tier == null || state.tierFilters.has(String(n.discovery_tier)) || !state.tierFilters.size;
    }).map(n => n.id));
    const edges = base.edges.filter(e => state.relationFilters.has(e.relation) && ids.has(e.source) && ids.has(e.target));
    return { ...base, nodes: base.nodes.filter(n => ids.has(n.id)), edges };
  }
  function focusSeedIds(nodesSet = state.focusNodes, edgesSet = state.focusEdges) {
    const ids = new Set(nodesSet);
    for (const eid of edgesSet) {
      const e = state.data.edges.find(x => x.id === eid);
      if (e) {
        ids.add(e.source);
        ids.add(e.target);
      }
    }
    return ids;
  }
  function buildFocusGraph(nodesSet = state.focusNodes, edgesSet = state.focusEdges) {
    const seeds = focusSeedIds(nodesSet, edgesSet);
    if (!seeds.size) return { nodes: [], edges: [], seedIds: seeds };
    const firstEdges = state.data.edges.filter(e => seeds.has(e.source) || seeds.has(e.target)), ids = new Set(seeds);
    for (const e of firstEdges) {
      ids.add(e.source);
      ids.add(e.target);
    }
    const edges = state.data.edges.filter(e => ids.has(e.source) && ids.has(e.target));
    return { nodes: state.data.nodes.filter(n => ids.has(n.id)), edges, seedIds: seeds };
  }
  function getVisibleGraph() {
    const base = applyStandardFilters(getBaseGraph());
    if (!state.detailMode) return base;
    const focus = buildFocusGraph();
    if (!state.showContext) return focus;
    const ids = new Set(base.nodes.map(n => n.id));
    for (const n of focus.nodes) ids.add(n.id);
    const edgeMap = new Map(base.edges.map(e => [e.id, e]));
    for (const e of focus.edges) edgeMap.set(e.id, e);
    return { ...base, nodes: state.data.nodes.filter(n => ids.has(n.id)), edges: [...edgeMap.values()].filter(e => ids.has(e.source) && ids.has(e.target)), focusGraph: focus };
  }
  function getPanelGraph() {
    return state.detailMode ? buildFocusGraph() : getVisibleGraph();
  }

  function inferTierMap(graph) {
    const tier = new Map();
    for (const n of graph.nodes) if (n.discovery_tier != null) tier.set(n.id, Number(n.discovery_tier));
    for (const [id, t] of graph.derivedTier || []) if (!tier.has(id)) tier.set(id, Number(t));
    for (let pass = 0; pass < 8; pass++) {
      let changed = false;
      for (const n of graph.nodes) {
        if (tier.has(n.id)) continue;
        const ts = [];
        for (const e of graph.edges) {
          if (e.source === n.id && tier.has(e.target)) ts.push(tier.get(e.target));
          if (e.target === n.id && tier.has(e.source)) ts.push(tier.get(e.source));
        }
        if (ts.length) {
          tier.set(n.id, Math.min(...ts));
          changed = true;
        }
      }
      if (!changed) break;
    }
    return tier;
  }
  function computePositions(graph) {
    if (state.viewMode === 'tiered') return computeLayerPositions(graph);
    if (state.viewMode === 'supply') return computeSupplyPositions(graph);
    return state.dimension === '2d' ? computeForce2DPositions(graph.nodes, graph.edges) : computeForcePositions(graph.nodes, graph.edges);
  }
  function computeLayerPositions(graph) {
    const pos = new Map(), tierMap = inferTierMap(graph), groups = new Map();
    for (const n of graph.nodes) {
      const t = tierMap.has(n.id) ? tierMap.get(n.id) : 99;
      if (!groups.has(t)) groups.set(t, []);
      groups.get(t).push(n);
    }
    const ordered = [...groups.keys()].sort((a, b) => a - b),
      finite = ordered.filter(t => t !== 99),
      maxTier = finite.length ? Math.max(...finite) : 0;
    if (state.dimension === '2d') {
      const specs = ordered.map(t => {
        const arr = [...(groups.get(t) || [])].sort((a, b) => (a.type || '').localeCompare(b.type || '') || (a.name || '').localeCompare(b.name || ''));
        const cols = Math.min(Math.max(3, Math.ceil(Math.sqrt(arr.length))), 6),
          rows = Math.ceil(arr.length / cols),
          gx = 82,
          gy = 88,
          width = Math.max(170, (cols - 1) * gx + 120),
          height = Math.max(150, (rows - 1) * gy + 120);
        return {
          t,
          arr,
          cols,
          rows,
          gx,
          gy,
          width,
          height
        };
      });
      let cursor = 0;
      const centers = new Map();
      for (const spec of specs) {
        centers.set(spec.t, cursor + spec.width / 2);
        cursor += spec.width + 120;
      }
      const total = cursor > 0 ? cursor - 120 : 0;
      for (const spec of specs) {
        const cx = centers.get(spec.t) - total / 2;
        spec.arr.forEach((n, i) => {
          const row = Math.floor(i / spec.cols), col = i % spec.cols;
          pos.set(n.id, {
            x: cx + (col - (spec.cols - 1) / 2) * spec.gx,
            y: (row - (spec.rows - 1) / 2) * spec.gy,
            z: 0,
            tier: spec.t,
            _tierWidth: spec.width,
            _tierHeight: spec.height
          });
        });
      }
      return pos;
    }
    for (const t of ordered) {
      const arr = groups.get(t).sort((a, b) => (a.type || '').localeCompare(b.type || '') || (a.name || '').localeCompare(b.name || '')),
        x = t === 99 ? 280 - (maxTier + 1) * 250 : 280 - Number(t) * 250;
      const cols = Math.max(1, Math.ceil(Math.sqrt(arr.length))), rows = Math.ceil(arr.length / cols), gap = 96;
      arr.forEach((n, i) => {
        const row = Math.floor(i / cols), col = i % cols;
        pos.set(n.id, { x, y: (col - (cols - 1) / 2) * gap, z: (row - (rows - 1) / 2) * gap, tier: t });
      });
    }
    return pos;
  }
  function computeSupplyPositions(graph) {
    const pos = new Map(),
      tierMap = graph.derivedTier || new Map(),
      anchors = graph.nodes.filter(n => graph.upstream?.has(n.id) || n.id === state.focalId),
      groups = new Map();
    anchors.forEach(n => {
      let t = n.discovery_tier;
      if (t == null) t = tierMap.get(n.id);
      if (t == null) t = 0;
      if (!groups.has(t)) groups.set(t, []);
      groups.get(t).push(n);
    });
    for (const [tier, arr0] of [...groups.entries()].sort((a, b) => a[0] - b[0])) {
      const arr = [...arr0].sort((a, b) => (a.name || '').localeCompare(b.name || '')), x = 260 - tier * 240;
      if (state.dimension === '2d') {
        arr.forEach((n, i) => {
          const y = (i - (arr.length - 1) / 2) * 72;
          pos.set(n.id, { x, y, z: 0, tier });
        });
      } else {
        const rr = tier === 0 ? 0 : Math.min(220, 75 + arr.length * 13);
        arr.forEach((n, i) => {
          if (n.id === state.focalId) {
            pos.set(n.id, { x: 260, y: 0, z: 0, tier });
            return;
          }
          const a = i / Math.max(1, arr.length) * Math.PI * 2 + tier * .7;
          pos.set(n.id, { x, y: Math.cos(a) * rr, z: Math.sin(a) * rr, tier });
        });
      }
    }
    const contexts = graph.nodes.filter(n => !pos.has(n.id)), buckets = new Map();
    for (const n of contexts) {
      let p = null;
      for (const e of graph.edges) {
        if (e.source === n.id && pos.has(e.target)) {
          p = e.target;
          break;
        }
        if (e.target === n.id && pos.has(e.source)) {
          p = e.source;
          break;
        }
      }
      p = p || state.focalId;
      if (!buckets.has(p)) buckets.set(p, []);
      buckets.get(p).push(n);
    }
    for (const [pid, arr] of buckets) {
      const p = pos.get(pid) || { x: 0, y: 0, z: 0, tier: 0 };
      arr.forEach((n, i) => {
        const a = i / Math.max(6, arr.length) * Math.PI * 2 + stableFraction(n.id, 'a'),
          rr = 72 + Math.floor(i / 8) * 38,
          dx = {
            Product: 35,
            Material: -20,
            Location: 65,
            Person: 90,
            Mine: -65,
            Company: -40
          }[n.type] || 0;
        if (state.dimension === '2d') pos.set(n.id, { x: p.x + dx + (Math.cos(a) * rr * .55), y: p.y + Math.sin(a) * rr, z: 0, tier: p.tier });
        else pos.set(n.id, { x: p.x + dx + (stableFraction(n.id, 'x') - .5) * 24, y: p.y + Math.cos(a) * rr, z: p.z + Math.sin(a) * rr, tier: p.tier });
      });
    }
    return pos;
  }
  function computeForce2DPositions(nodes, edges) {
    const pos = new Map(), N = Math.max(1, nodes.length);
    nodes.forEach(n => {
      const a = stableFraction(n.id, 'a2') * Math.PI * 2, r = 120 + stableFraction(n.id, 'r2') * 260;
      pos.set(n.id, { x: Math.cos(a) * r, y: Math.sin(a) * r, z: 0, tier: n.discovery_tier ?? null });
    });
    const iterations = N > 160 ? 45 : 92;
    for (let it = 0; it < iterations; it++) {
      const f = new Map(nodes.map(n => [n.id, { x: 0, y: 0 }])), repel = 15000 * (1 - it / iterations * .3);
      for (let i = 0; i < nodes.length; i++)for (let j = i + 1; j < nodes.length; j++) {
        const a = pos.get(nodes[i].id),
          b = pos.get(nodes[j].id),
          dx = a.x - b.x,
          dy = a.y - b.y,
          d2 = dx * dx + dy * dy + 110,
          dist = Math.sqrt(d2),
          q = repel / d2,
          fx = dx / dist * q,
          fy = dy / dist * q,
          fa = f.get(nodes[i].id),
          fb = f.get(nodes[j].id);
        fa.x += fx;
        fa.y += fy;
        fb.x -= fx;
        fb.y -= fy;
      }
      for (const e of edges) {
        const a = pos.get(e.source), b = pos.get(e.target);
        if (!a || !b) continue;
        const dx = b.x - a.x,
          dy = b.y - a.y,
          dist = Math.sqrt(dx * dx + dy * dy) + .001,
          ideal = e.relation === 'suppliesTo' ? 120 : 96,
          k = (dist - ideal) * .021,
          fx = dx / dist * k,
          fy = dy / dist * k,
          fa = f.get(e.source),
          fb = f.get(e.target);
        fa.x += fx;
        fa.y += fy;
        fb.x -= fx;
        fb.y -= fy;
      }
      for (const n of nodes) {
        const p = pos.get(n.id), v = f.get(n.id);
        p.x = (p.x + v.x * .75) * .995;
        p.y = (p.y + v.y * .75) * .995;
      }
    }
    return pos;
  }
  function computeForcePositions(nodes, edges) {
    const pos = new Map(), N = Math.max(1, nodes.length);
    nodes.forEach(n => {
      const u = stableFraction(n.id, 'u'),
        v = stableFraction(n.id, 'v'),
        th = u * Math.PI * 2,
        ph = Math.acos(2 * v - 1),
        r = 180 + stableFraction(n.id, 'r') * 130;
      pos.set(n.id, { x: r * Math.sin(ph) * Math.cos(th), y: r * Math.cos(ph), z: r * Math.sin(ph) * Math.sin(th), tier: n.discovery_tier ?? null });
    });
    const iterations = N > 160 ? 38 : 76;
    for (let it = 0; it < iterations; it++) {
      const f = new Map(nodes.map(n => [n.id, { x: 0, y: 0, z: 0 }])), repel = 12500 * (1 - it / iterations * .35);
      for (let i = 0; i < nodes.length; i++)for (let j = i + 1; j < nodes.length; j++) {
        const a = pos.get(nodes[i].id),
          b = pos.get(nodes[j].id),
          dx = a.x - b.x,
          dy = a.y - b.y,
          dz = a.z - b.z,
          d2 = dx * dx + dy * dy + dz * dz + 90,
          dist = Math.sqrt(d2),
          q = repel / d2,
          fx = dx / dist * q,
          fy = dy / dist * q,
          fz = dz / dist * q,
          fa = f.get(nodes[i].id),
          fb = f.get(nodes[j].id);
        fa.x += fx;
        fa.y += fy;
        fa.z += fz;
        fb.x -= fx;
        fb.y -= fy;
        fb.z -= fz;
      }
      for (const e of edges) {
        const a = pos.get(e.source), b = pos.get(e.target);
        if (!a || !b) continue;
        const dx = b.x - a.x,
          dy = b.y - a.y,
          dz = b.z - a.z,
          dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + .001,
          ideal = e.relation === 'suppliesTo' ? 118 : 94,
          k = (dist - ideal) * .018,
          fx = dx / dist * k,
          fy = dy / dist * k,
          fz = dz / dist * k,
          fa = f.get(e.source),
          fb = f.get(e.target);
        fa.x += fx;
        fa.y += fy;
        fa.z += fz;
        fb.x -= fx;
        fb.y -= fy;
        fb.z -= fz;
      }
      for (const n of nodes) {
        const p = pos.get(n.id), v = f.get(n.id);
        p.x = (p.x + v.x * .72) * .996;
        p.y = (p.y + v.y * .72) * .996;
        p.z = (p.z + v.z * .72) * .996;
      }
    }
    return pos;
  }

  function refreshGraph() {
    if (!state.data) return;
    const g = getVisibleGraph(),
      focus = state.detailMode ? buildFocusGraph() : null,
      focusIds = focus ? new Set(focus.nodes.map(n => n.id)) : new Set(),
      focusEdgeIds = focus ? new Set(focus.edges.map(e => e.id)) : new Set();
    graphRenderer.setMode2D(state.dimension === '2d');
    graphRenderer.allLabels = state.labels || state.detailMode;
    graphRenderer.particles = state.dimension === '3d';
    graphRenderer.setGraph(g.nodes, g.edges, computePositions(g), { tiered: state.viewMode === 'tiered', focusNodeIds: focusIds, focusEdgeIds: focusEdgeIds, contextMode: state.detailMode && state.showContext });
    graphRenderer.setSelection(state.selectedNodes, state.selectedEdges);
    elementById('visibleNodeCount').textContent = formatCount(g.nodes.length);
    elementById('visibleEdgeCount').textContent = formatCount(g.edges.length);
    elementById('graphEmptyState').hidden = !!g.nodes.length;
    renderLegend(g.nodes);
    renderFocusBar();
  }
  function renderLegend(nodes) {
    const ts = [...new Set(nodes.map(n => n.type).filter(Boolean))];
    elementById('graphLegend').innerHTML = ts.map(t => `<span class="legend-item"><i style="--c:${ENTITY_COLORS[t] || FALLBACK_COLOR}"></i>${escapeHtml(t)}</span>`).join('');
  }
  function renderFocusBar() {
    const bar = elementById('focusBar');
    bar.hidden = !state.detailMode;
    if (!state.detailMode) return;
    const f = buildFocusGraph(), labels = focusLabels();
    elementById('focusTitle').textContent = labels.length === 1 ? labels[0] : `${labels.length} selected`;
    elementById('focusMeta').textContent = `${formatCount(f.nodes.length)} entities · ${formatCount(f.edges.length)} facts`;
    elementById('contextToggle').checked = state.showContext;
  }

  function selectNode(id, shift = false) {
    if (!state.nodeMap.has(id)) return;
    if (!shift) {
      state.selectedNodes.clear();
      state.selectedEdges.clear();
      state.selectedNodes.add(id);
    } else {
      state.selectedNodes.has(id) ? state.selectedNodes.delete(id) : state.selectedNodes.add(id);
    }
    state.panelScope = 'visible';
    refreshSelectionUI();
  }
  function selectEdge(id, shift = false) {
    const e = state.data.edges.find(x => x.id === id);
    if (!e) return;
    if (!shift) {
      state.selectedNodes.clear();
      state.selectedEdges.clear();
      state.selectedEdges.add(id);
    } else {
      state.selectedEdges.has(id) ? state.selectedEdges.delete(id) : state.selectedEdges.add(id);
    }
    state.panelScope = 'visible';
    refreshSelectionUI();
  }
  function clearSelection() {
    state.selectedNodes.clear();
    state.selectedEdges.clear();
    state.panelScope = 'visible';
    refreshSelectionUI();
  }
  function refreshSelectionUI() {
    graphRenderer.setSelection(state.selectedNodes, state.selectedEdges);
    renderSelectionBar();
    renderSide();
  }
  function selectionLabels(nodes = state.selectedNodes, edges = state.selectedEdges) {
    const items = [];
    for (const id of nodes) {
      const n = state.nodeMap.get(id);
      if (n) items.push({ kind: 'node', text: n.name || n.id, id });
    }
    for (const id of edges) {
      const e = state.data.edges.find(x => x.id === id);
      if (e) {
        const s = state.nodeMap.get(e.source)?.name || e.source, t = state.nodeMap.get(e.target)?.name || e.target;
        items.push({ kind: 'edge', text: `${s} · ${getRelationLabel(e.relation)} · ${t}`, id });
      }
    }
    return items;
  }
  function focusLabels() {
    return selectionLabels(state.focusNodes, state.focusEdges).map(x => x.text);
  }
  function renderSelectionBar() {
    const count = selectionCount(), bar = elementById('selectionBar');
    bar.hidden = !count;
    if (!count) return;
    const items = selectionLabels();
    elementById('selectionChips').innerHTML = items.slice(0, 4).map(x => `<span class="selection-chip ${x.kind === 'edge' ? 'edge' : ''}">${escapeHtml(x.text)}</span>`).join('') + (items.length > 4 ? `<span class="selection-chip">+${items.length - 4}</span>` : '');
    elementById('openDetailsBtn').textContent = state.detailMode ? 'Refocus' : 'Details';
  }
  function enterDetails() {
    if (!selectionCount()) return;
    state.focusNodes = new Set(state.selectedNodes);
    state.focusEdges = new Set(state.selectedEdges);
    state.detailMode = true;
    state.showContext = false;
    state.panelScope = 'visible';
    refreshGraph();
    renderSelectionBar();
    renderSide();
    setTimeout(() => graphRenderer.fit(), 30);
  }
  function exitDetails() {
    state.detailMode = false;
    state.showContext = false;
    state.focusNodes.clear();
    state.focusEdges.clear();
    state.selectedNodes.clear();
    state.selectedEdges.clear();
    state.panelScope = 'visible';
    refreshGraph();
    renderSelectionBar();
    renderSide();
    graphRenderer.setSelection(state.selectedNodes, state.selectedEdges);
    setTimeout(() => graphRenderer.fit(), 30);
  }

  function getPanelEdges() {
    const g = getPanelGraph();
    if (state.panelScope !== 'selected' || !selectionCount()) return g.edges;
    const selectedIds = new Set(state.selectedNodes), selectedEdgeIds = new Set(state.selectedEdges);
    return g.edges.filter(e => selectedEdgeIds.has(e.id) || selectedIds.has(e.source) || selectedIds.has(e.target));
  }
  function renderSide() {
    if (!state.data) return;
    const g = getPanelGraph(),
      scopeLabel = state.detailMode ? 'Focused graph' : state.viewMode === 'tiered' ? 'Tiered graph' : state.viewMode === 'network' ? 'Network' : state.viewMode === 'supply' ? 'Upstream' : 'All graph';
    elementById('sideTitle').textContent = scopeLabel;
    elementById('sideMeta').textContent = `${formatCount(g.nodes.length)} entities · ${formatCount(g.edges.length)} facts`;
    document.querySelectorAll('#sideTabs button').forEach(b => b.classList.toggle('active', b.dataset.sideTab === state.sideTab));
    elementById('sideTools').hidden = state.sideTab === 'info';
    elementById('panelScopeToggle').hidden = !selectionCount() || state.sideTab === 'info';
    document.querySelectorAll('#panelScopeToggle button').forEach(b => b.classList.toggle('active', b.dataset.panelScope === state.panelScope));
    if (state.sideTab === 'info') renderInfoPanel(g);
    else if (state.sideTab === 'evidence') renderEvidencePanel(g);
    else renderSourcesPanel(g);
    renderPanelRelationOptions();
  }
  function renderInfoPanel(g) {
    const root = elementById('sideContent'), selected = selectionLabels();
    if (selected.length) {
      let html = '';
      const nodeRows = [...state.selectedNodes].map(id => state.nodeMap.get(id)).filter(Boolean);
      if (nodeRows.length) html += `<div class="selected-group"><div class="mini-title">Entities</div><div class="row-list">${nodeRows.map(n => `<div class="selected-row"><i class="entity-dot" style="--c:${ENTITY_COLORS[n.type] || FALLBACK_COLOR}"></i><div class="row-main"><b>${escapeHtml(n.name || n.id)}</b><span>${escapeHtml(n.type || 'Entity')}${n.discovery_tier != null ? ` · Tier ${n.discovery_tier}` : ''}</span></div><span class="tier-badge">${formatCount(n.stats?.edge_count || 0)} links</span></div>`).join('')}</div></div>`;
      const edgeRows = [...state.selectedEdges].map(id => state.data.edges.find(e => e.id === id)).filter(Boolean);
      if (edgeRows.length) html += `<div class="selected-group"><div class="mini-title">Relations</div><div class="row-list">${edgeRows.map(e => factRowHtml(e)).join('')}</div></div>`;
      const rel = getPanelEdges().filter(e => state.selectedNodes.has(e.source) || state.selectedNodes.has(e.target) || state.selectedEdges.has(e.id)).slice(0, 12);
      if (rel.length) html += `<div class="mini-section"><div class="mini-title">Related facts</div><div class="row-list">${rel.map(e => factRowHtml(e)).join('')}</div></div>`;
      root.innerHTML = html || '<div class="side-empty">—</div>';
      bindSideSelections(root);
      return;
    }
    const typeCounts = countsBy(g.nodes, 'type'),
      tierCounts = countsBy(g.nodes.filter(n => n.discovery_tier != null), 'discovery_tier'),
      relCounts = countsBy(g.edges, 'relation');
    const top = [...g.nodes].sort((a, b) => (b.stats?.edge_count || 0) - (a.stats?.edge_count || 0)).slice(0, 8);
    const summaryMarkup = `<div class="summary-grid"><div class="summary-card"><span>Entities</span><strong>${formatCount(g.nodes.length)}</strong></div><div class="summary-card"><span>Facts</span><strong>${formatCount(g.edges.length)}</strong></div><div class="summary-card wide"><span>Tiers</span><strong>${Object.keys(tierCounts).sort((a, b) => a - b).map(t => `T${t} · ${tierCounts[t]}`).join('   ') || '—'}</strong></div></div>`;
    const relationMarkup = `<div class="mini-section"><div class="mini-title">Relations</div><div class="row-list">${Object.entries(relCounts).sort((a, b) => b[1] - a[1]).map(([r, c]) => `<div class="entity-row"><i class="entity-dot" style="--c:${RELATION_COLORS[r] || FALLBACK_COLOR}"></i><div class="row-main"><b>${escapeHtml(getRelationLabel(r))}</b><span>${formatCount(c)} facts</span></div></div>`).join('')}</div></div>`;
    const connectedMarkup = `<div class="mini-section"><div class="mini-title">Most connected</div><div class="row-list">${top.map(n => `<div class="entity-row" data-select-node="${escapeHtml(n.id)}"><i class="entity-dot" style="--c:${ENTITY_COLORS[n.type] || FALLBACK_COLOR}"></i><div class="row-main"><b>${escapeHtml(n.name || n.id)}</b><span>${escapeHtml(n.type || 'Entity')}${n.discovery_tier != null ? ` · Tier ${n.discovery_tier}` : ''}</span></div><span class="tier-badge">${formatCount(n.stats?.edge_count || 0)}</span></div>`).join('')}</div></div>`;
    root.innerHTML = summaryMarkup + relationMarkup + connectedMarkup;
    bindSideSelections(root);
  }
  function factRowHtml(e) {
    const s = state.nodeMap.get(e.source), t = state.nodeMap.get(e.target), c = RELATION_COLORS[e.relation] || FALLBACK_COLOR;
    const factMarkup = `<div class="fact-row" data-select-edge="${escapeHtml(e.id)}"><div class="fact-line"><b>${escapeHtml(s?.name || e.source)}</b><span class="arrow">→</span><span class="relation-pill" style="--c:${c}">${escapeHtml(getRelationLabel(e.relation))}</span><span class="arrow">→</span><b>${escapeHtml(t?.name || e.target)}</b></div>`;
    const factMetadataMarkup = `<div class="fact-meta"><span>${formatCount(e.support_count || e.source_doc_ids?.length || 0)} sources</span><span>${formatCount(e.evidence?.length || e.evidence_count || 0)} evidence</span></div></div>`;
    return factMarkup + factMetadataMarkup;
  }
  function bindSideSelections(root) {
    root.querySelectorAll('[data-select-node]').forEach(el => el.onclick = () => {
      selectNode(el.dataset.selectNode, false);
      setTimeout(() => graphRenderer.focusNode(el.dataset.selectNode, 1.35), 20);
    });
    root.querySelectorAll('[data-select-edge]').forEach(el => el.onclick = () => selectEdge(el.dataset.selectEdge, false));
  }

  function renderPanelRelationOptions() {
    const sel = elementById('panelRelationFilter');
    if (!sel) return;
    const rels = ['all', ...Object.keys(countsBy(state.data.edges, 'relation')).sort()];
    const current = state.panelRelation;
    sel.innerHTML = rels.map(r => `<option value="${escapeHtml(r)}">${escapeHtml(r === 'all' ? 'All' : getRelationLabel(r))}</option>`).join('');
    sel.value = rels.includes(current) ? current : 'all';
    elementById('panelContentFilter').value = state.panelContent;
    elementById('panelScoreFilter').value = String(state.panelScore);
    elementById('panelSearch').value = state.panelSearch;
  }
  function evidenceRows() {
    const q = state.panelSearch.trim().toLowerCase(), edges = getPanelEdges();
    let rows = [];
    for (const e of edges) {
      if (state.panelRelation !== 'all' && e.relation !== state.panelRelation) continue;
      for (const ev of e.evidence || []) {
        const doc = state.docMap.get(ev.doc_id),
          score = Number(ev.source_score ?? doc?.source_effective_score ?? e.quality?.max_source_score ?? 0),
          content = ev.content_source || doc?.content_source || '';
        if (state.panelContent !== 'all' && content !== state.panelContent) continue;
        if (score < Number(state.panelScore || 0)) continue;
        const s = state.nodeMap.get(e.source)?.name || e.source,
          t = state.nodeMap.get(e.target)?.name || e.target,
          blob = [s, t, e.relation, ev.quote, ev.document_title, ev.source_domain, ev.provider, doc?.title, doc?.publisher_name].join(' ').toLowerCase();
        if (q && !blob.includes(q)) continue;
        rows.push({
          edge: e,
          ev,
          doc,
          score,
          content
        });
      }
    }
    return rows.sort((a, b) => b.score - a.score || (b.edge.support_count || 0) - (a.edge.support_count || 0));
  }
  function renderEvidencePanel() {
    const root = elementById('sideContent'), rows = evidenceRows();
    root.innerHTML = rows.map((x, i) => evidenceCardHtml(x, i)).join('') || '<div class="side-empty">No evidence</div>';
    hydrateSourceLinks(root);
  }
  function evidenceCardHtml(x, index) {
    const e = x.edge,
      ev = x.ev,
      doc = x.doc,
      s = state.nodeMap.get(e.source),
      t = state.nodeMap.get(e.target),
      url = resolveEvidenceUrl(ev, doc),
      domain = ev.source_domain || doc?.source_domain || getDomainName(url),
      content = x.content || ev.content_source || doc?.content_source || 'document',
      href = url || '#';
    const headingMarkup = `<article class="evidence-card"><div class="fact-line"><b>${escapeHtml(s?.name || e.source)}</b><span class="arrow">→</span><span class="relation-pill" style="--c:${RELATION_COLORS[e.relation] || FALLBACK_COLOR}">${escapeHtml(getRelationLabel(e.relation))}</span><span class="arrow">→</span><b>${escapeHtml(t?.name || e.target)}</b></div>`;
    const quoteMarkup = `<div class="evidence-quote">“${escapeHtml(ev.quote || 'No quote stored')}”</div>`;
    const metadataMarkup = `<div class="evidence-meta"><span class="source-pill">${escapeHtml(domain || ev.provider || 'source')}</span><span>${escapeHtml(getSourceTypeLabel(content))}</span>${x.score ? `<span class="score-badge">${escapeHtml(Math.round(x.score))}</span>` : ''}${url ? `<a class="evidence-source-link" href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer" data-resolve-source="${escapeHtml(url)}">Source ↗</a>` : ''}</div></article>`;
    return headingMarkup + quoteMarkup + metadataMarkup;
  }
  function renderSourcesPanel() {
    const root = elementById('sideContent'), q = state.panelSearch.trim().toLowerCase(), edges = getPanelEdges(), ids = new Set();
    for (const e of edges) {
      for (const id of e.source_doc_ids || []) ids.add(id);
      for (const ev of e.evidence || []) if (ev.doc_id) ids.add(ev.doc_id);
    }
    let docs = [...ids].map(id => state.docMap.get(id)).filter(Boolean);
    if (state.panelContent !== 'all') docs = docs.filter(d => (d.content_source || '') === state.panelContent);
    docs = docs.filter(d => Number(d.source_effective_score || d.source_base_score || 0) >= Number(state.panelScore || 0));
    if (q) docs = docs.filter(d => [d.title, d.publisher_name, d.source_domain, d.provider].join(' ').toLowerCase().includes(q));
    docs.sort((a, b) => Number(b.source_effective_score || 0) - Number(a.source_effective_score || 0));
    root.innerHTML = docs.map((d, i) => {
      const url = d.canonical_source_url || d.source_url || d.retrieval_url || '',
        body = `<article class="source-row"><b>${escapeHtml(d.title || d.doc_id)}</b><span class="source-domain">${escapeHtml(d.publisher_name || d.source_domain || getDomainName(url))}</span><span>${escapeHtml(getSourceTypeLabel(d.content_source))}${Number(d.source_effective_score) ? ` · score ${escapeHtml(Math.round(Number(d.source_effective_score)))}` : ''}</span></article>`;
      return url ? `<a class="source-row-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" data-resolve-source="${escapeHtml(url)}">${body}</a>` : body;
    }).join('') || '<div class="side-empty">No sources</div>';
    hydrateSourceLinks(root);
  }

  const resolvedSourceCache = new Map();
  let evidenceModalToken = 0;
  function resolveEvidenceUrl(ev, doc) {
    return doc?.canonical_source_url || ev?.canonical_source_url || doc?.source_url || ev?.source_url || doc?.retrieval_url || ev?.retrieval_url || '';
  }
  function isGoogleNewsUrl(url) {
    try {
      return new URL(url, location.href).hostname === 'news.google.com';
    } catch {
      return false;
    }
  }
  function normalizeFragmentText(text) {
    return String(text || '').replace(/[\r\n\t]+/g, ' ').replace(/\s{2,}/g, ' ').trim().replace(/^[“”"']+|[“”"']+$/g, '').slice(0, 220);
  }
  function withTextFragment(url, text) {
    if (!url) return '';
    try {
      const u = new URL(url, location.href);
      u.hash = '';
      const q = normalizeFragmentText(text);
      if (q) u.hash = `:~:text=${encodeURIComponent(q)}`;
      return u.toString();
    } catch {
      return url;
    }
  }
  function previewUrl(url, text) {
    const q = normalizeFragmentText(text);
    if ((location.protocol === 'http:' || location.protocol === 'https:') && location.hostname) {
      return `/evidence-preview?url=${encodeURIComponent(url)}&text=${encodeURIComponent(q)}`;
    }
    return withTextFragment(url, q);
  }
  async function resolveSourceUrlRemote(url, ev, doc) {
    if (!url || !isGoogleNewsUrl(url)) return url;
    if (resolvedSourceCache.has(url)) return resolvedSourceCache.get(url);
    if (!((location.protocol === 'http:' || location.protocol === 'https:') && location.hostname)) return url;
    try {
      const params = new URLSearchParams({ url, title: ev?.document_title || doc?.title || '', publisher: ev?.source_domain || doc?.source_domain || doc?.publisher_name || '' }),
        r = await fetch(`/resolve-source?${params.toString()}`, { cache: 'no-store' });
      if (!r.ok) throw new Error('resolver unavailable');
      const out = await r.json(), direct = out?.url || url;
      resolvedSourceCache.set(url, direct);
      return direct;
    } catch {
      return url;
    }
  }
  function hydrateSourceLinks(root) {
    root.querySelectorAll('a[data-resolve-source]').forEach(async a => {
      const original = a.dataset.resolveSource || a.href;
      if (!original || !isGoogleNewsUrl(original)) return;
      const direct = await resolveSourceUrlRemote(original, null, null);
      if (direct && direct !== original) {
        a.href = direct;
        a.dataset.resolveSource = direct;
      }
    });
  }

  function localEvidenceContext(ev, doc) {
    const candidates = [ev?.context, ev?.chunk_text, ev?.source_text, ev?.text, ev?.content, doc?.context, doc?.chunk_text, doc?.source_text, doc?.text, doc?.content, doc?.full_text, doc?.raw_text, doc?.article_text, doc?.body].filter(v => typeof v === 'string' && v.trim().length > 40);
    if (!candidates.length) return null;
    const text = candidates.sort((a, b) => b.length - a.length)[0].trim(),
      isFull = text.length >= 1200 || ['full_text', 'article_text', 'body', 'raw_text'].some(k => typeof doc?.[k] === 'string' && doc[k].trim().length >= 900);
    return {
      ok: true,
      title: ev?.document_title || doc?.title || 'Evidence source',
      domain: ev?.source_domain || doc?.source_domain || doc?.publisher_name || '',
      context: text,
      paragraphs: text.split(/\n{2,}/).map(x => x.trim()).filter(Boolean).slice(0, 70),
      matched: true,
      origin: 'json',
      full_text: isFull
    };
  }
  function highlightEvidenceText(text, quote) {
    const escaped = escapeHtml(text || '');
    const q = normalizeFragmentText(quote);
    if (!q) return escaped;
    const variants = [q, q.slice(0, 180), q.split(' ').slice(0, 14).join(' ')].filter(x => x.length > 12);
    for (const v of variants) {
      const idx = String(text || '').toLowerCase().indexOf(v.toLowerCase());
      if (idx >= 0) {
        const a = escapeHtml(String(text).slice(0, idx)),
          b = escapeHtml(String(text).slice(idx, idx + v.length)),
          c = escapeHtml(String(text).slice(idx + v.length));
        return `${a}<mark>${b}</mark>${c}`;
      }
    }
    return escaped;
  }
  function renderEvidenceContent(out, quote) {
    const root = elementById('evidenceContent'), paragraphs = (out?.paragraphs?.length ? out.paragraphs : [out?.context || quote || '']).filter(Boolean);
    if (!paragraphs.length) {
      root.innerHTML = '<div class="content-error">Source text unavailable.</div>';
      return;
    }
    const origin = out?.origin === 'json' ? 'Stored article text' : out?.origin === 'web-search' ? 'Public copy found on the web' : out?.origin === 'web' ? 'Readable article text' : 'Evidence context';
    const foot = out?.full_text ? (out?.searched ? 'Full readable public copy located by title/evidence search.' : 'Full readable text extracted from the public source.') : (out?.matched ? 'Evidence text located in the source context.' : 'Closest readable source context shown.');
    root.innerHTML = `<div class="content-source-head"><div class="content-origin">${escapeHtml(origin)}</div><h4>${escapeHtml(out?.title || 'Evidence source')}</h4><div class="content-domain">${escapeHtml(out?.domain || '')}</div></div><div class="content-body">${paragraphs.map(p => `<p>${highlightEvidenceText(p, quote)}</p>`).join('')}</div><div class="content-foot">${escapeHtml(foot)}</div>`;
    requestAnimationFrame(() => root.querySelector('mark')?.scrollIntoView({ block: 'center', behavior: 'instant' }));
  }
  async function openEvidenceModal(ev, doc) {
    const original = resolveEvidenceUrl(ev, doc),
      quote = ev?.quote || ev?.document_title || doc?.title || '',
      title = ev?.document_title || doc?.title || getDomainName(original) || 'Evidence source';
    const token = ++evidenceModalToken;
    elementById('modalDomain').textContent = ev?.source_domain || doc?.source_domain || doc?.publisher_name || getDomainName(original);
    elementById('modalTitle').textContent = title;
    elementById('modalQuote').textContent = quote ? `“${quote}”` : '';
    elementById('evidenceContent').innerHTML = '<div class="content-loading">Searching for the full readable article…</div>';
    elementById('evidenceModal').hidden = false;
    document.body.style.overflow = 'hidden';
    const local = localEvidenceContext(ev, doc);
    if (local?.full_text) {
      if (token !== evidenceModalToken || elementById('evidenceModal').hidden) return;
      renderEvidenceContent(local, quote);
      return;
    }
    if (!original) {
      if (token !== evidenceModalToken) return;
      if (local) renderEvidenceContent(local, quote);
      else elementById('evidenceContent').innerHTML = '<div class="content-error">No stored source text or source URL is available.</div>';
      return;
    }
    if (!((location.protocol === 'http:' || location.protocol === 'https:') && location.hostname)) {
      if (token !== evidenceModalToken) return;
      renderEvidenceContent(local || {
        ok: false,
        title,
        domain: elementById('modalDomain').textContent,
        context: quote,
        paragraphs: [quote],
        origin: 'evidence',
        matched: false
      }, quote);
      return;
    }
    try {
      const params = new URLSearchParams({ url: original, text: quote, title, publisher: ev?.source_domain || doc?.source_domain || doc?.publisher_name || '' }),
        r = await fetch(`/evidence-content?${params.toString()}`, { cache: 'no-store' });
      const out = await r.json();
      if (token !== evidenceModalToken || elementById('evidenceModal').hidden) return;
      if (out?.domain) elementById('modalDomain').textContent = out.domain;
      if (out?.title) elementById('modalTitle').textContent = out.title;
      if (out?.ok && out?.paragraphs?.length) renderEvidenceContent(out, quote);
      else if (local) renderEvidenceContent(local, quote);
      else renderEvidenceContent(out, quote);
    } catch (err) {
      if (token !== evidenceModalToken) return;
      renderEvidenceContent(local || {
        title,
        domain: elementById('modalDomain').textContent,
        context: quote,
        paragraphs: [quote],
        origin: 'evidence',
        matched: false
      }, quote);
    }
  }
  function closeEvidenceModal() {
    evidenceModalToken++;
    elementById('evidenceModal').hidden = true;
    elementById('evidenceContent').innerHTML = '';
    document.body.style.overflow = '';
  }

  function searchNodes(q) {
    q = q.trim().toLowerCase();
    const root = elementById('searchResults');
    if (!q) {
      root.classList.remove('open');
      return;
    }
    const rows = state.data.nodes.filter(n => (n.name || '').toLowerCase().includes(q) || (n.aliases || []).some(a => String(a).toLowerCase().includes(q)) || String(n.id).toLowerCase().includes(q)).sort((a, b) => (b.stats?.edge_count || 0) - (a.stats?.edge_count || 0)).slice(0, 14);
    root.innerHTML = rows.map(n => `<div class="search-row" data-node="${escapeHtml(n.id)}"><b>${escapeHtml(n.name || n.id)}</b><span>${escapeHtml(n.type || 'Entity')}${n.discovery_tier != null ? ` · T${n.discovery_tier}` : ''}</span></div>`).join('') || '<div class="search-row"><span>No match</span></div>';
    root.classList.add('open');
    root.querySelectorAll('[data-node]').forEach(el => el.onclick = e => {
      const id = el.dataset.node, target = state.nodeMap.get(id);
      if (target?.type !== 'Company' && state.entityScope === 'companies') {
        state.entityScope = 'all';
      }
      if (!getVisibleGraph().nodes.some(n => n.id === id)) {
        state.viewMode = 'global';
        syncModeButtons();
        refreshGraph();
      }
      selectNode(id, e.shiftKey);
      elementById('entitySearch').value = '';
      root.classList.remove('open');
      setTimeout(() => graphRenderer.focusNode(id, 1.35), 30);
    });
  }
  async function loadFiles(list) {
    const files = [...list].filter(f => f.name.toLowerCase().endsWith('.json'));
    if (!files.length) {
      toast('No JSON files');
      return;
    }
    let first = null;
    for (const f of files) {
      try {
        const data = JSON.parse(await f.text()), name = registerDataset(f.name, data, 'file');
        first = first || name;
      } catch (err) {
        toast(`Skipped ${f.name}`, err.message);
      }
    }
    if (first) loadDataset(first);
  }
  async function loadLocalCatalog() {
    try {
      const r = await fetch('data/catalog.json', { cache: 'no-store' });
      if (!r.ok) return;
      const c = await r.json();
      for (const file of c.files || []) {
        if (state.datasets.has(file)) continue;
        try {
          const x = await fetch(`data/${encodeURIComponent(file)}`, { cache: 'no-store' });
          if (x.ok) registerDataset(file, await x.json(), 'data-folder');
        } catch {
        }
      }
    } catch {
    }
  }
  function toast(title, msg = '') {
    let s = document.querySelector('.toast-stack');
    if (!s) {
      s = document.createElement('div');
      s.className = 'toast-stack';
      document.body.appendChild(s);
    }
    const d = document.createElement('div');
    d.className = 'toast';
    d.innerHTML = `<b>${escapeHtml(title)}</b>${msg ? `<span>${escapeHtml(msg)}</span>` : ''}`;
    s.appendChild(d);
    setTimeout(() => d.remove(), 2800);
  }

  elementById('brandBtn').onclick = () => {
    state.viewMode = 'tiered';
    state.dimension = '3d';
    state.entityScope = 'all';
    state.detailMode = false;
    state.showContext = false;
    state.focusNodes.clear();
    state.focusEdges.clear();
    clearSelection();
    renderGraphControls();
    refreshGraph();
    renderSide();
  };
  elementById('datasetSelect').onchange = e => loadDataset(e.target.value);
  elementById('openJsonBtn').onclick = () => {
    elementById('jsonInput').click();
    elementById('datasetMenu').open = false;
  };
  elementById('jsonInput').onchange = e => loadFiles(e.target.files);
  elementById('downloadJsonBtn').onclick = () => {
    if (!state.data) return;
    const blob = new Blob([JSON.stringify(state.data, null, 2)], { type: 'application/json' }),
      url = URL.createObjectURL(blob),
      a = document.createElement('a');
    a.href = url;
    a.download = state.currentName || 'schain-graph.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 500);
    elementById('datasetMenu').open = false;
  };
  document.querySelectorAll('#viewMode button').forEach(b => b.onclick = () => {
    state.viewMode = b.dataset.view;
    renderGraphControls();
    refreshGraph();
    renderSide();
    setTimeout(() => graphRenderer.fit(), 25);
  });
  document.querySelectorAll('#dimensionMode button').forEach(b => b.onclick = () => {
    state.dimension = b.dataset.dimension;
    syncModeButtons();
    refreshGraph();
    setTimeout(() => graphRenderer.fit(), 20);
  });
  document.querySelectorAll('#entityScopeMode button').forEach(b => b.onclick = () => {
    state.entityScope = b.dataset.entityScope;
    state.entityFilters = new Set(state.entityScope === 'companies' ? ['Company'] : state.data.nodes.map(n => n.type).filter(Boolean));
    syncModeButtons();
    refreshGraph();
    renderSide();
    setTimeout(() => graphRenderer.fit(), 20);
  });
  elementById('focalCompany').onchange = e => {
    state.focalId = e.target.value;
    state.viewMode = 'supply';
    renderGraphControls();
    refreshGraph();
    renderSide();
  };
  elementById('entitySearch').oninput = e => searchNodes(e.target.value);
  elementById('filterBtn').onclick = e => {
    e.stopPropagation();
    elementById('filterPopover').hidden = !elementById('filterPopover').hidden;
  };
  elementById('relationAll').onclick = () => {
    state.relationFilters = new Set(state.data.edges.map(e => e.relation));
    renderGraphControls();
    refreshGraph();
    renderSide();
  };
  elementById('tierAll').onclick = () => {
    state.tierFilters = new Set(state.data.nodes.map(n => n.discovery_tier).filter(t => t != null).map(String));
    renderGraphControls();
    refreshGraph();
    renderSide();
  };
  elementById('labelsToggle').onchange = e => {
    state.labels = e.target.checked;
    graphRenderer.allLabels = state.labels || state.detailMode;
  };
  elementById('fitBtn').onclick = () => graphRenderer.fit();
  elementById('clearSelectionBtn').onclick = clearSelection;
  elementById('openDetailsBtn').onclick = enterDetails;
  elementById('exitFocusBtn').onclick = exitDetails;
  elementById('contextToggle').onchange = e => {
    state.showContext = e.target.checked;
    refreshGraph();
    renderSide();
    setTimeout(() => graphRenderer.fit(), 20);
  };
  document.querySelectorAll('#sideTabs button').forEach(b => b.onclick = () => {
    state.sideTab = b.dataset.sideTab;
    renderSide();
  });
  document.querySelectorAll('#panelScopeToggle button').forEach(b => b.onclick = () => {
    state.panelScope = b.dataset.panelScope;
    renderSide();
  });
  elementById('panelSearch').oninput = e => {
    state.panelSearch = e.target.value;
    if (state.sideTab === 'evidence') renderEvidencePanel();
    else if (state.sideTab === 'sources') renderSourcesPanel();
  };
  elementById('panelFilterBtn').onclick = e => {
    e.stopPropagation();
    elementById('panelFilterPopover').hidden = !elementById('panelFilterPopover').hidden;
  };
  elementById('panelRelationFilter').onchange = e => {
    state.panelRelation = e.target.value;
    if (state.sideTab === 'evidence') renderEvidencePanel();
  };
  elementById('panelContentFilter').onchange = e => {
    state.panelContent = e.target.value;
    state.sideTab === 'evidence' ? renderEvidencePanel() : renderSourcesPanel();
  };
  elementById('panelScoreFilter').onchange = e => {
    state.panelScore = Number(e.target.value || 0);
    state.sideTab === 'evidence' ? renderEvidencePanel() : renderSourcesPanel();
  };
  elementById('modalCloseBtn').onclick = closeEvidenceModal;
  document.querySelectorAll('[data-modal-close]').forEach(el => el.onclick = closeEvidenceModal);
  document.addEventListener('click', e => {
    if (!elementById('filterPopover').hidden && !e.target.closest('.popover-wrap')) elementById('filterPopover').hidden = true;
    if (!elementById('panelFilterPopover').hidden && !e.target.closest('.panel-filter-wrap')) elementById('panelFilterPopover').hidden = true;
  });
  const stage = elementById('graphStage');
  ['dragenter', 'dragover'].forEach(t => stage.addEventListener(t, e => {
    e.preventDefault();
    elementById('dropOverlay').classList.add('show');
  }));
  ['dragleave', 'drop'].forEach(t => stage.addEventListener(t, e => {
    e.preventDefault();
    if (t === 'drop' && e.dataTransfer?.files?.length) loadFiles(e.dataTransfer.files);
    elementById('dropOverlay').classList.remove('show');
  }));
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !elementById('evidenceModal').hidden) {
      closeEvidenceModal();
      return;
    }
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    if (e.key === 'Escape') {
      if (state.detailMode) exitDetails();
      else clearSelection();
    }
    if (e.key.toLowerCase() === 'f') graphRenderer.fit();
  });

  if (window.SCHAIN_DEMO_DATA) {
    try {
      const name = registerDataset(window.SCHAIN_DEMO_NAME || 'SChain graph.json', window.SCHAIN_DEMO_DATA, 'demo');
      loadDataset(name);
    } catch (err) {
      console.error(err);
      toast('Could not load graph', err.message);
    }
  }
  if (location.protocol === 'http:' || location.protocol === 'https:') loadLocalCatalog();
})();
