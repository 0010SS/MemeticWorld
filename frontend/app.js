// MemeWorld frontend: pixel-art Homewood campus replay, culture dashboard, causal trace explorer.
// Normal demo mode never requests hidden ground truth; Research Debug Mode adds ?debug=1.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const S = {
  runs: [], runId: null, debug: false, manifest: null, frames: [], analysis: null,
  tick: 0, tf: 0, playing: false, speed: 1, sel: null, sprites: {},
  selMeme: null, agentReq: 0, order: {},
};

async function api(path, opts = {}) {
  const sep = path.includes("?") ? "&" : "?";
  const r = await fetch(`/api${path}${S.debug ? sep + "debug=1" : ""}`, opts);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

// ------------------------------------------------------------------ tooltip
const tip = $("#tooltip");
function showTip(e, html) { tip.innerHTML = html; tip.hidden = false; moveTip(e); }
function moveTip(e) { tip.style.left = Math.min(e.clientX + 12, innerWidth - 330) + "px"; tip.style.top = e.clientY + 12 + "px"; }
function hideTip() { tip.hidden = true; }

// ------------------------------------------------------------------- boot
async function boot() {
  $$(".tab").forEach((b) => b.onclick = () => showView(b.dataset.view));
  $("#debugToggle").onchange = async (e) => { S.debug = e.target.checked; $("#debugBadge").hidden = !S.debug; await loadRun(S.runId, true); };
  $("#runSelect").onchange = (e) => loadRun(e.target.value);
  $("#btnPlay").onclick = togglePlay;
  $("#btnBack").onclick = () => setTick(S.tick - 1);
  $("#btnFwd").onclick = () => setTick(S.tick + 1);
  $("#speed").onchange = (e) => S.speed = +e.target.value;
  $("#scrub").oninput = (e) => setTick(+e.target.value);
  $("#btnFirstMeme").onclick = () => jumpTo("first_meme_use");
  $("#btnFirstCross").onclick = () => jumpTo("first_cross_group_transmission");
  $("#modalClose").onclick = () => $("#modal").hidden = true;
  $("#modal").onclick = (e) => { if (e.target.id === "modal") $("#modal").hidden = true; };
  $("#onlyConv").onchange = renderCulture;
  $("#btnAnalyze").onclick = async () => { await fetch(`/api/runs/${S.runId}/analyze`, { method: "POST" }); $("#cultureSummary").innerHTML = `<span class="muted">Analysis launched; reload this run in a minute.</span>`; };
  $("#traceQuery").oninput = renderTraceSearch;
  $("#btnLaunch").onclick = launchRun;
  bindMapControls();
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowRight") setTick(S.tick + 1);
    if (e.code === "ArrowLeft") setTick(S.tick - 1);
    if (e.key === "+" || e.key === "=") zoomBy(1);
    if (e.key === "-") zoomBy(-1);
  });
  const mapReady = loadMap();
  S.runs = await api("/runs");
  const sel = $("#runSelect");
  sel.innerHTML = S.runs.map((r) => `<option value="${r.run_id}">${esc(r.run_id)} (${r.status}${r.has_analysis ? ", analyzed" : ""})</option>`).join("");
  const want = new URLSearchParams(location.search).get("run");
  const first = S.runs.find((r) => r.run_id === want) || S.runs.find((r) => r.status === "finished" && r.has_analysis) || S.runs[0];
  if (first) { sel.value = first.run_id; await loadRun(first.run_id); }
  await mapReady;
  const qs = new URLSearchParams(location.search);
  if (qs.get("tick")) setTick(+qs.get("tick"));
  if (qs.get("view") === "fit") zoomFit();
  if (qs.get("zoom")) setZoom(+qs.get("zoom"));
  if (qs.get("place") && M.data.places[qs.get("place")]) { const b = M.data.places[qs.get("place")].box; centerOn((b[0] + b[2] + 1) / 2 * TILE, (b[1] + b[3] + 1) / 2 * TILE); }
  if (qs.get("frac")) S.tf = S.tick + Math.min(0.999, Math.max(0, +qs.get("frac")));
  if (qs.get("agent")) { S.sel = qs.get("agent"); renderAgent(S.sel); if (qs.get("follow")) { cam.follow = true; $("#followSel").checked = true; } }
  renderRuns();
  requestAnimationFrame(loop);
}

function showView(v) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === "view-" + v));
  if (v === "culture") renderCulture();
  if (v === "trace") renderTraceSearch();
  if (v === "runs") renderRuns();
  if (v === "campus") resizeMap();
}

async function loadRun(id, keepTick = false) {
  if (!id) return;
  S.runId = id;
  const [man, frames] = await Promise.all([api(`/runs/${id}/manifest`), api(`/runs/${id}/frames`)]);
  S.manifest = man; S.frames = frames;
  try { S.analysis = await api(`/runs/${id}/analysis`); } catch { S.analysis = null; }
  S.order = {}; Object.keys(man.agents).sort().forEach((aid, i) => S.order[aid] = i);
  for (const [aid, a] of Object.entries(man.agents)) {
    if (!S.sprites[a.sprite]) { const img = new Image(); img.src = `/ga_assets/characters/${a.sprite}.png`; S.sprites[a.sprite] = img; }
  }
  posCache.length = 0;
  $("#scrub").max = Math.max(0, frames.length - 1);
  if (!keepTick) setTick(0); else setTick(S.tick);
  if (S.sel) renderAgent(S.sel);
  renderCulture();
}

// ================================================================== pixel map
const TILE = 32;
const M = { data: null, images: {}, layers: null, thumb: null, ready: false, cost: null, W: 0, H: 0, CW: 0, CHn: 0, fgHas: null, pathCache: new Map() };
const cam = { x: 0, y: 0, zoom: 0.5, drag: null, moved: false, follow: false };
const ZOOMS = [0.125, 0.25, 0.35, 0.5, 0.75, 1, 1.5, 2];
let dpr = 1;

async function loadMap() {
  const d = await (await fetch("/static/homewood_map.json")).json();
  d.tilesets.sort((a, b) => a.firstgid - b.firstgid);
  M.data = d; M.W = d.width; M.H = d.height;
  const n = d.width * d.height;
  const dec = (arr) => { if (d.encoding !== "rle") return arr; const out = new Int32Array(n); let i = 0; for (let k = 0; k < arr.length; k += 2) { const v = arr[k], r = arr[k + 1]; if (v) out.fill(v, i, i + r); i += r; } return out; };
  M.layers = {}; for (const [name, arr] of Object.entries(d.layers)) M.layers[name] = dec(arr);
  M.cost = dec(d.cost);
  M.CW = Math.ceil(d.width / CH); M.CHn = Math.ceil(d.height / CH);
  M.fgHas = new Uint8Array(M.CW * M.CHn);
  for (const name of d.fg_layers) { const L = M.layers[name]; for (let i = 0; i < n; i++) if (L[i]) M.fgHas[Math.floor(Math.floor(i / d.width) / CH) * M.CW + Math.floor((i % d.width) / CH)] = 1; }
  await Promise.all(d.tilesets.map((ts) => new Promise((res) => {
    const im = new Image(); im.onload = res; im.onerror = res; im.src = ts.image; M.images[ts.name] = im;
  })));
  M.thumb = new Image(); M.thumb.src = d.thumb || "/static/homewood_thumb.png";
  const mini = $("#minimap"); mini.height = Math.round(mini.width * M.H / M.W);
  try { await document.fonts.load("600 14px 'Pixelify Sans'"); } catch { /* fallback font */ }
  chunkCache.clear();
  M.ready = true;
  $("#mapLoading").hidden = true;
  resizeMap();
  zoomQuad();
}

function tileSrc(gid) {
  const ts = M.data.tilesets;
  for (let i = ts.length - 1; i >= 0; i--) if (gid >= ts[i].firstgid) {
    const t = ts[i], l = gid - t.firstgid;
    return [M.images[t.name], (l % t.columns) * TILE, Math.floor(l / t.columns) * TILE];
  }
  return null;
}

// ---- chunked map rendering: 32x32-tile chunks rendered on demand at three detail levels, LRU-cached
const CH = 32;
const chunkCache = new Map();
const CHUNK_MAX = 220;
let chunkBudget = 0;
function levelFor(zoom) { return zoom >= 0.75 ? 1 : zoom >= 0.35 ? 0.5 : 0.25; }
function renderChunk(level, cx, cy, fg) {
  const size = Math.round(CH * TILE * level), ts = TILE * level;
  const c = document.createElement("canvas"); c.width = size; c.height = size;
  const ctx = c.getContext("2d"); ctx.imageSmoothingEnabled = level < 1; if (level < 1) ctx.imageSmoothingQuality = "high";
  const layers = fg ? M.data.fg_layers : ["bottom", "ground", "deco1", "deco2", "floor", "wall", "furn1", "furn2"];
  const x0 = cx * CH, y0 = cy * CH, x1 = Math.min(x0 + CH, M.W), y1 = Math.min(y0 + CH, M.H);
  for (const name of layers) {
    const L = M.layers[name];
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const g = L[y * M.W + x]; if (!g) continue;
      const src = tileSrc(g); if (!src || !src[0]?.naturalWidth) continue;
      ctx.drawImage(src[0], src[1], src[2], TILE, TILE, (x - x0) * ts, (y - y0) * ts, ts, ts);
    }
  }
  return c;
}
function getChunk(level, cx, cy, fg) {
  if (fg && !M.fgHas[cy * M.CW + cx]) return null;
  const k = `${fg ? "f" : "b"}${level}:${cx}:${cy}`;
  let c = chunkCache.get(k);
  if (c) { chunkCache.delete(k); chunkCache.set(k, c); return c; }
  if (chunkBudget <= 0) return null;
  chunkBudget--;
  c = renderChunk(level, cx, cy, fg);
  chunkCache.set(k, c);
  while (chunkCache.size > CHUNK_MAX) chunkCache.delete(chunkCache.keys().next().value);
  return c;
}
function drawChunks(ctx, fg) {
  const [vw, vh] = viewSize(), level = levelFor(cam.zoom), px = CH * TILE;
  if (cam.zoom < 0.2) {   // far out: the pre-rendered thumbnail is sharper than 8 px tiles and costs nothing
    if (!fg && M.thumb?.naturalWidth) ctx.drawImage(M.thumb, 0, 0, M.W * TILE, M.H * TILE);
    return;
  }
  const cx0 = Math.max(0, Math.floor(cam.x / px)), cy0 = Math.max(0, Math.floor(cam.y / px));
  const cx1 = Math.min(M.CW - 1, Math.floor((cam.x + vw / cam.zoom) / px)), cy1 = Math.min(M.CHn - 1, Math.floor((cam.y + vh / cam.zoom) / px));
  const th = M.thumb, tk = th?.naturalWidth ? th.naturalWidth / (M.W * TILE) : 0;
  for (let cy = cy0; cy <= cy1; cy++) for (let cx = cx0; cx <= cx1; cx++) {
    const c = getChunk(level, cx, cy, fg);
    if (c) ctx.drawImage(c, cx * px, cy * px, px, px);
    else if (!fg && tk) ctx.drawImage(th, cx * px * tk, cy * px * tk, px * tk, px * tk, cx * px, cy * px, px, px);
  }
}

// ---- camera
function viewSize() { const c = $("#map"); return [c.clientWidth, c.clientHeight]; }
function resizeMap() {
  const c = $("#map"); dpr = Math.min(2, window.devicePixelRatio || 1);
  const [vw, vh] = viewSize();
  if (c.width !== Math.round(vw * dpr) || c.height !== Math.round(vh * dpr)) { c.width = Math.round(vw * dpr); c.height = Math.round(vh * dpr); }
  clampCam();
}
function clampCam() {
  if (!M.data) return;
  const [vw, vh] = viewSize(), w = M.W * TILE, h = M.H * TILE;
  const maxX = Math.max(-vw / cam.zoom * 0.5, w - vw / cam.zoom + vw / cam.zoom * 0.5), maxY = Math.max(-vh / cam.zoom * 0.5, h - vh / cam.zoom + vh / cam.zoom * 0.5);
  cam.x = Math.max(-vw / cam.zoom * 0.5, Math.min(maxX, cam.x));
  cam.y = Math.max(-vh / cam.zoom * 0.5, Math.min(maxY, cam.y));
}
function centerOn(wx, wy) { const [vw, vh] = viewSize(); cam.x = wx - vw / cam.zoom / 2; cam.y = wy - vh / cam.zoom / 2; clampCam(); }
function setZoom(z, ax, ay) {
  // keep the world point under (ax, ay) screen px fixed
  const [vw, vh] = viewSize(); ax = ax ?? vw / 2; ay = ay ?? vh / 2;
  const wx = cam.x + ax / cam.zoom, wy = cam.y + ay / cam.zoom;
  cam.zoom = z; cam.x = wx - ax / z; cam.y = wy - ay / z; clampCam();
}
function zoomBy(dir, ax, ay) {
  let i = ZOOMS.findIndex((z) => z >= cam.zoom - 1e-6); if (i < 0) i = ZOOMS.length - 1;
  setZoom(ZOOMS[Math.max(0, Math.min(ZOOMS.length - 1, i + dir))], ax, ay);
}
function zoomFit() { const [vw, vh] = viewSize(); cam.zoom = Math.min(vw / (M.W * TILE), vh / (M.H * TILE)); cam.follow = false; $("#followSel").checked = false; centerOn(M.W * TILE / 2, M.H * TILE / 2); }
function zoomQuad() { cam.zoom = 0.35; const q = M.data.places.Quad?.box || [21, 48, 41, 59]; centerOn((q[0] + q[2] + 1) / 2 * TILE, (q[1] + q[3] + 1) / 2 * TILE - 40); }
function screenToWorld(e) { const r = $("#map").getBoundingClientRect(); return { x: cam.x + (e.clientX - r.left) / cam.zoom, y: cam.y + (e.clientY - r.top) / cam.zoom }; }

function bindMapControls() {
  const c = $("#map");
  c.onpointerdown = (e) => { cam.drag = { x: e.clientX, y: e.clientY, cx: cam.x, cy: cam.y }; cam.moved = false; c.setPointerCapture(e.pointerId); c.classList.add("dragging"); };
  c.onpointermove = (e) => {
    if (cam.drag) {
      const dx = e.clientX - cam.drag.x, dy = e.clientY - cam.drag.y;
      if (Math.hypot(dx, dy) > 3) { cam.moved = true; cam.follow = false; $("#followSel").checked = false; }
      cam.x = cam.drag.cx - dx / cam.zoom; cam.y = cam.drag.cy - dy / cam.zoom; clampCam(); hideTip();
    } else onMapHover(e);
  };
  c.onpointerup = (e) => { c.classList.remove("dragging"); if (cam.drag && !cam.moved) onMapClick(e); cam.drag = null; };
  c.onpointerleave = () => { hideTip(); };
  c.onwheel = (e) => { e.preventDefault(); const r = c.getBoundingClientRect(); zoomBy(e.deltaY < 0 ? 1 : -1, e.clientX - r.left, e.clientY - r.top); };
  $("#zoomIn").onclick = () => zoomBy(1); $("#zoomOut").onclick = () => zoomBy(-1);
  $("#zoomFit").onclick = zoomFit; $("#zoomQuad").onclick = zoomQuad;
  $("#followSel").onchange = (e) => { cam.follow = e.target.checked; };
  $("#minimap").onclick = (e) => { const r = e.target.getBoundingClientRect(); centerOn((e.clientX - r.left) / r.width * M.W * TILE, (e.clientY - r.top) / r.height * M.H * TILE); cam.follow = false; $("#followSel").checked = false; };
  new ResizeObserver(resizeMap).observe($("#mapWrap"));
}

// ---- agent placement: one stable spot per agent inside its arena
const posCache = [];
function positions(t) {
  if (posCache[t]) return posCache[t];
  const frame = S.frames[t], out = {};
  if (!frame || !M.data) return out;
  const groups = {};
  for (const [aid, a] of Object.entries(frame.agents)) (groups[a.location + "|" + a.arena] = groups[a.location + "|" + a.arena] || []).push(aid);
  for (const [key, ids] of Object.entries(groups)) {
    const [loc, ar] = key.split("|");
    const pl = M.data.places[loc]; if (!pl) continue;
    const arena = pl.arenas[ar] || Object.values(pl.arenas)[0];
    const spots = arena.spots; if (!spots?.length) continue;
    ids.sort((a, b) => S.order[a] - S.order[b]);
    const used = new Set();
    for (const aid of ids) {
      let k = S.order[aid] % spots.length, n = 0;
      while (used.has(k) && n < spots.length) { k = (k + 1) % spots.length; n++; }
      used.add(k);
      out[aid] = { cx: spots[k][0], cy: spots[k][1] };
    }
  }
  posCache[t] = out;
  return out;
}

// ---- A* over the walk-cost grid (4-neighbour, like Smallville)
function findPath(a, b) {
  const key = a.cx + "," + a.cy + ">" + b.cx + "," + b.cy;
  if (M.pathCache.has(key)) return M.pathCache.get(key);
  const W = M.W, H = M.H, cost = M.cost, start = a.cy * W + a.cx, goal = b.cy * W + b.cx;
  const g = new Float32Array(W * H).fill(Infinity), prev = new Int32Array(W * H).fill(-1), closed = new Uint8Array(W * H);
  const hx = (i) => Math.abs((i % W) - b.cx) + Math.abs(Math.floor(i / W) - b.cy);
  const open = [[hx(start), start]]; g[start] = 0;
  const push = (it) => { open.push(it); let i = open.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (open[p][0] <= open[i][0]) break; [open[p], open[i]] = [open[i], open[p]]; i = p; } };
  const pop = () => { const top = open[0], last = open.pop(); if (open.length) { open[0] = last; let i = 0; for (;;) { let l = 2 * i + 1, r = l + 1, m = i; if (l < open.length && open[l][0] < open[m][0]) m = l; if (r < open.length && open[r][0] < open[m][0]) m = r; if (m === i) break; [open[m], open[i]] = [open[i], open[m]]; i = m; } } return top; };
  let found = false, steps = 0;
  while (open.length && steps++ < 60000) {
    const [, cur] = pop();
    if (cur === goal) { found = true; break; }
    if (closed[cur]) continue; closed[cur] = 1;
    const cx = cur % W, cy = Math.floor(cur / W);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = cx + dx, ny = cy + dy; if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      const ni = ny * W + nx, c = cost[ni]; if (!c && ni !== goal) continue;
      const ng = g[cur] + (c || 1);
      if (ng < g[ni]) { g[ni] = ng; prev[ni] = cur; push([ng + hx(ni), ni]); }
    }
  }
  let path;
  if (found) { path = []; for (let i = goal; i !== -1; i = prev[i]) path.push({ cx: i % W, cy: Math.floor(i / W) }); path.reverse(); }
  else path = [a, b];
  M.pathCache.set(key, path);
  return path;
}
const px = (c) => ({ x: c.cx * TILE + TILE / 2, y: c.cy * TILE + TILE / 2 });

// -------------------------------------------------------------------- loop
let lastT = performance.now();
function loop(now) {
  const dt = (now - lastT) / 1000; lastT = now;
  if (S.playing && S.frames.length) {
    S.tf += dt * S.speed * 0.5;
    if (S.tf >= S.frames.length - 1) { S.tf = S.frames.length - 1; S.playing = false; $("#btnPlay").textContent = "▶"; }
    const t = Math.floor(S.tf);
    if (t !== S.tick) { S.tick = t; onTickChanged(); }
  }
  draw();
  requestAnimationFrame(loop);
}
function togglePlay() { S.playing = !S.playing; $("#btnPlay").textContent = S.playing ? "⏸" : "▶"; if (S.playing && S.tf >= S.frames.length - 1) setTick(0); }
function setTick(t) {
  if (!S.frames.length) return;
  S.tick = Math.max(0, Math.min(S.frames.length - 1, t)); S.tf = S.tick + 0.999; onTickChanged();
}
let agentTimer = null;
function onTickChanged() {
  const f = S.frames[S.tick];
  $("#clock").textContent = f ? f.label : "—";
  $("#scrub").value = S.tick;
  renderFeed(); renderEvents();
  if (S.sel) { clearTimeout(agentTimer); agentTimer = setTimeout(() => renderAgent(S.sel), S.playing ? 600 : 80); }
}
function jumpTo(key) {
  const s = S.analysis?.summary?.[key];
  if (!s) { alert("No such moment in this run (or the run is not analyzed yet)."); return; }
  const t = s.tick ?? s.first_reuse_timestamp;
  showView("campus"); S.playing = false; $("#btnPlay").textContent = "▶"; setTick(t);
}

// -------------------------------------------------------------------- draw
const PIX = "'Pixelify Sans', ui-monospace, monospace";
const drawState = { pos: {}, tick: -1 };

function agentDrawPositions() {
  // world-pixel position, facing and walking state of every agent at the current sub-tick time
  const frame = S.frames[S.tick]; if (!frame) return {};
  const p = S.tf - S.tick;
  const cur = positions(S.tick), prev = S.tick > 0 ? positions(S.tick - 1) : cur;
  const out = {};
  for (const aid of Object.keys(frame.agents)) {
    const to = cur[aid]; if (!to) continue;
    const from = prev[aid] || to;
    let pos = px(to), dir = 0, walking = false;
    if (from.cx !== to.cx || from.cy !== to.cy) {
      const path = findPath(from, to);
      const dur = Math.min(0.9, Math.max(0.3, (path.length - 1) / 24));   // ≈ 12 tiles per second at 1×
      const k = Math.min(1, p / dur);
      if (k < 1) {
        const f = k * (path.length - 1), i = Math.min(path.length - 2, Math.floor(f)), t = f - i;
        const a = px(path[i]), b = px(path[i + 1]);
        pos = { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
        const dx = b.x - a.x, dy = b.y - a.y;
        dir = Math.abs(dx) > Math.abs(dy) ? (dx < 0 ? 1 : 2) : (dy < 0 ? 3 : 0);
        walking = true;
      } else {
        const a = px(path[Math.max(0, path.length - 2)]), b = px(path[path.length - 1]);
        const dx = b.x - a.x, dy = b.y - a.y;
        dir = Math.abs(dx) > Math.abs(dy) ? (dx < 0 ? 1 : 2) : (dy < 0 ? 3 : 0);
      }
    }
    out[aid] = { ...pos, dir, walking, cx: to.cx, cy: to.cy };
  }
  return out;
}

function draw() {
  const cv = $("#map"), ctx = cv.getContext("2d");
  const [vw, vh] = viewSize();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--map-bg").trim() || "#5aa84a";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!M.ready) return;
  const frame = S.frames[S.tick];
  const man = S.manifest;
  const pos = frame ? agentDrawPositions() : {};
  drawState.pos = pos;
  if (cam.follow && S.sel && pos[S.sel]) { const t = pos[S.sel]; cam.x += (t.x - vw / cam.zoom / 2 - cam.x) * 0.15; cam.y += (t.y - vh / cam.zoom / 2 - cam.y) * 0.15; clampCam(); }
  // world layer
  const z = cam.zoom * dpr;
  ctx.setTransform(z, 0, 0, z, -cam.x * z, -cam.y * z);
  ctx.imageSmoothingEnabled = cam.zoom < 0.75;
  chunkBudget = 6;
  drawChunks(ctx, false);
  // active-event outline on buildings
  const active = new Set((frame?.beats || []).map((b) => b.location));
  for (const loc of active) {
    const pl = M.data.places[loc]; if (!pl) continue;
    const [x0, y0, x1, y1] = pl.box;
    ctx.strokeStyle = "#eb6834"; ctx.lineWidth = 3 / cam.zoom; ctx.setLineDash([8 / cam.zoom, 6 / cam.zoom]);
    ctx.strokeRect(x0 * TILE - 2, y0 * TILE - 2, (x1 - x0 + 1) * TILE + 4, (y1 - y0 + 1) * TILE + 4); ctx.setLineDash([]);
  }
  // agents, painter's order by y
  const ids = Object.keys(pos).sort((a, b) => pos[a].y - pos[b].y);
  for (const aid of ids) {
    const a = pos[aid], prof = man.agents[aid], img = S.sprites[prof.sprite];
    const fx = a.walking ? [0, 32, 64][Math.floor(performance.now() / 140) % 3] : 32;
    if (S.sel === aid) {
      ctx.fillStyle = "rgba(42,120,214,.35)"; ctx.beginPath(); ctx.ellipse(a.x, a.y + 8, 14, 7, 0, 0, 7); ctx.fill();
    }
    if (img?.complete && img.naturalWidth) ctx.drawImage(img, fx, a.dir * 32, 32, 32, Math.round(a.x - 16), Math.round(a.y - 22), 32, 32);
    else { ctx.fillStyle = "#2a78d6"; ctx.beginPath(); ctx.arc(a.x, a.y - 6, 9, 0, 7); ctx.fill(); }
  }
  drawChunks(ctx, true);
  // screen-space overlays (labels, names, bubbles, badges)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const sx = (wx) => (wx - cam.x) * cam.zoom, sy = (wy) => (wy - cam.y) * cam.zoom;
  drawContextLabels(ctx, sx, sy, vw, vh);
  drawPlaceLabels(ctx, sx, sy, active);
  for (const aid of ids) {
    const a = pos[aid], x = sx(a.x), y = sy(a.y);
    if (x < -40 || y < -40 || x > vw + 40 || y > vh + 40) continue;
    const nm = man.agents[aid].name.split(" ")[0];
    ctx.font = `600 ${cam.zoom >= 0.75 ? 13 : 11}px ${PIX}`; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const w = ctx.measureText(nm).width + 8, yy = y + 12 * cam.zoom;
    ctx.fillStyle = S.sel === aid ? "#2a78d6" : "rgba(20,20,18,.78)";
    ctx.fillRect(Math.round(x - w / 2), Math.round(yy), Math.round(w), 15);
    ctx.fillStyle = "#fff"; ctx.fillText(nm, Math.round(x), Math.round(yy) + 1);
    if (frame.agents[aid].conversation) { ctx.fillStyle = "#ffd866"; ctx.font = `700 12px ${PIX}`; ctx.fillText("…", Math.round(x + 18), Math.round(y - 34 * cam.zoom)); }
  }
  // speech bubbles (cycle through a conversation's lines within the tick)
  const p = S.tf - S.tick;
  if (frame && p > 0.45) {
    const convs = {};
    for (const u of frame.utterances) (convs[u.conversation_id || u.id] = convs[u.conversation_id || u.id] || []).push(u);
    for (const us of Object.values(convs)) {
      const k = Math.min(us.length - 1, Math.floor(((p - 0.45) / 0.55) * us.length));
      const u = us[k], sp = pos[u.speaker]; if (!sp) continue;
      if (sx(sp.x) < -20 || sx(sp.x) > vw + 20 || sy(sp.y) < -20 || sy(sp.y) > vh + 20) continue;
      ctx.strokeStyle = "rgba(42,120,214,.6)"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      for (const l of u.listeners) { const lp = pos[l]; if (lp) { ctx.beginPath(); ctx.moveTo(sx(sp.x), sy(sp.y) - 10); ctx.lineTo(sx(lp.x), sy(lp.y) - 10); ctx.stroke(); } }
      ctx.setLineDash([]);
      bubble(ctx, sx(sp.x), sy(sp.y) - 26 * cam.zoom, u.text, vw);
    }
  }
  drawMinimap(pos, vw, vh);
}

function drawContextLabels(ctx, sx, sy, vw, vh) {
  if (!M.data.context || cam.zoom < 0.35) return;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const c of M.data.context) {
    if (c.kind === "campus" && cam.zoom < 0.5) continue;
    if (c.kind === "other" && cam.zoom < 1) continue;
    if (c.kind === "green" && cam.zoom < 0.5) continue;
    const x = sx((c.x + 0.5) * TILE), y = sy((c.y + 0.5) * TILE);
    if (x < -120 || x > vw + 120 || y < -20 || y > vh + 20) continue;
    ctx.font = `${c.kind === "campus" ? 600 : 500} ${c.kind === "campus" ? 12 : 11}px ${PIX}`;
    const w = ctx.measureText(c.name).width + 8;
    if (c.kind === "road") { ctx.fillStyle = "rgba(40,40,40,.85)"; ctx.fillRect(Math.round(x - w / 2), Math.round(y - 8), Math.round(w), 16); ctx.fillStyle = "#fff"; }
    else if (c.kind === "green") { ctx.fillStyle = "rgba(255,255,255,.5)"; ctx.fillRect(Math.round(x - w / 2), Math.round(y - 8), Math.round(w), 16); ctx.fillStyle = "#2f5a2a"; }
    else { ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.fillRect(Math.round(x - w / 2), Math.round(y - 8), Math.round(w), 16); ctx.fillStyle = "#222"; }
    ctx.fillText(c.name, Math.round(x), Math.round(y));
  }
}

function drawPlaceLabels(ctx, sx, sy, active) {
  if (cam.zoom < 0.3) return;
  ctx.textBaseline = "top"; ctx.textAlign = "left";
  for (const pl of Object.values(M.data.places)) {
    const [x0, y0, x1] = pl.box;
    const x = sx(x0 * TILE) + 2, y = sy(y0 * TILE) - 20;
    const name = pl.name, sub = pl.label === pl.name ? "" : pl.label;
    ctx.font = `700 13px ${PIX}`; const w1 = ctx.measureText(name).width;
    ctx.font = `500 11px ${PIX}`; const w2 = sub ? ctx.measureText(sub).width : 0;
    const w = w1 + (sub ? w2 + 8 : 0) + 10, maxw = (x1 - x0 + 1) * TILE * cam.zoom;
    ctx.fillStyle = active.has(name) ? "rgba(235,104,52,.92)" : "rgba(20,20,18,.8)";
    ctx.fillRect(Math.round(x), Math.round(y), Math.round(Math.max(w, Math.min(maxw, w))), 18);
    ctx.fillStyle = "#fff"; ctx.font = `700 13px ${PIX}`; ctx.fillText(name, Math.round(x) + 5, Math.round(y) + 2);
    if (sub) { ctx.fillStyle = "#e8e4d4"; ctx.font = `500 11px ${PIX}`; ctx.fillText(sub, Math.round(x) + 5 + w1 + 8, Math.round(y) + 4); }
    if (active.has(name)) { ctx.fillStyle = "#fff"; ctx.font = `700 13px ${PIX}`; ctx.fillText("!", Math.round(x) + Math.round(Math.max(w, Math.min(maxw, w))) - 12, Math.round(y) + 2); }
  }
}

function bubble(ctx, x, y, text, vw) {
  ctx.font = `500 12px ${PIX}`; ctx.textBaseline = "top"; ctx.textAlign = "left";
  const words = text.split(" "), lines = []; let line = "";
  for (const w of words) { if (ctx.measureText(line + " " + w).width > 200 && line) { lines.push(line); line = w; } else line = line ? line + " " + w : w; if (lines.length >= 3) break; }
  if (lines.length < 3 && line) lines.push(line);
  if (lines.length === 3 && text.length > lines.join(" ").length) lines[2] += "…";
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 14, h = lines.length * 15 + 10;
  const bx = Math.round(Math.max(4, Math.min(vw - w - 4, x - w / 2))), by = Math.round(Math.max(4, y - h - 6));
  ctx.fillStyle = "#1c1c1a"; ctx.fillRect(bx - 2, by - 2, w + 4, h + 4);
  ctx.fillStyle = "#fffdf5"; ctx.fillRect(bx, by, w, h);
  ctx.fillStyle = "#1c1c1a"; ctx.fillRect(Math.round(x) - 3, by + h, 6, 4); ctx.fillRect(Math.round(x) - 1, by + h + 4, 2, 3);
  ctx.fillStyle = "#1c1c1a"; lines.forEach((l, i) => ctx.fillText(l, bx + 7, by + 6 + i * 15));
}

function drawMinimap(pos, vw, vh) {
  const c = $("#minimap"), ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  if (M.thumb?.naturalWidth) ctx.drawImage(M.thumb, 0, 0, c.width, c.height);
  const kx = c.width / (M.W * TILE), ky = c.height / (M.H * TILE);
  for (const [aid, a] of Object.entries(pos)) { ctx.fillStyle = aid === S.sel ? "#ffd866" : "#e34948"; ctx.fillRect(Math.round(a.x * kx) - 1, Math.round(a.y * ky) - 1, 3, 3); }
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
  ctx.strokeRect(Math.round(cam.x * kx) + .5, Math.round(cam.y * ky) + .5, Math.round(vw / cam.zoom * kx), Math.round(vh / cam.zoom * ky));
}

function agentAt(e) {
  const { x, y } = screenToWorld(e);
  let best = null, bd = 18 / Math.min(1, cam.zoom);
  for (const [aid, p] of Object.entries(drawState.pos)) { const d = Math.hypot(p.x - x, p.y - 8 - y); if (d < bd) { bd = d; best = aid; } }
  return best;
}
function onMapClick(e) { const a = agentAt(e); if (a) { S.sel = a; renderAgent(a); } }
function onMapHover(e) {
  const a = agentAt(e); const f = S.frames[S.tick];
  if (a && f) { const st = f.agents[a]; showTip(e, `<b>${esc(S.manifest.agents[a].name)}</b><br>${esc(st.activity)}<br><span class="muted">${esc(st.location)} · ${esc(st.arena)}</span>`); }
  else {
    const { x, y } = screenToWorld(e); const cx = Math.floor(x / TILE), cy = Math.floor(y / TILE);
    const pl = Object.values(M.data?.places || {}).find((p) => cx >= p.box[0] && cx <= p.box[2] && cy >= p.box[1] && cy <= p.box[3]);
    if (pl) { const ar = Object.entries(pl.arenas).find(([, a]) => cx >= a.rect[0] && cx <= a.rect[2] && cy >= a.rect[1] && cy <= a.rect[3]); showTip(e, `<b>${esc(pl.name)}</b> · ${esc(pl.label)}${ar ? `<br><span class="muted">${esc(ar[0])}</span>` : ""}`); }
    else hideTip();
  }
}

// ------------------------------------------------------------ side panels
function renderEvents() {
  const f = S.frames[S.tick];
  $("#eventsStrip").innerHTML = (f?.beats || []).map((b) =>
    `<div class="event-chip"><b>${esc(b.location)}</b> ${S.debug ? `<span class="tag gt">${esc(b.latent_type)} · ${esc(b.event_id)}</span>` : ""}${b.facts.map(esc).join(" ")}</div>`).join("");
}
function renderFeed() {
  const out = [];
  for (let t = S.tick; t >= 0 && out.length < 40; t--) for (const u of [...(S.frames[t]?.utterances || [])].reverse()) out.push([t, u]);
  $("#feed").innerHTML = out.map(([t, u]) => `<div class="utt" data-uid="${esc(u.id)}"><span class="who">${esc(name(u.speaker))}</span>
    <span class="meta">→ ${u.listeners.map(name).map(esc).join(", ") || "nobody"} · ${esc(S.frames[t].label)}</span><div>${esc(u.text)}</div></div>`).join("") || `<div class="muted small">No utterances yet.</div>`;
  $$("#feed .utt").forEach((d) => d.onclick = () => openChain(d.dataset.uid));
}
const name = (aid) => S.manifest?.agents[aid]?.name.split(" ")[0] || aid;

async function renderAgent(aid) {
  const req = ++S.agentReq;
  const d = await api(`/runs/${S.runId}/agent/${aid}?tick=${S.tick}`);
  if (req !== S.agentReq) return;
  const p = d.profile, st = d.state || {};
  const rel = Object.entries(p.relationships).filter(([, r]) => r.relation_type !== "stranger")
    .map(([o, r]) => `${esc(name(o))} <span class="muted">(${r.relation_type}, fam ${r.familiarity}, aff ${r.affinity})</span>`).join("<br>");
  const mem = (m) => `<div class="mem ${m.kind}"><div>${esc(m.text)}</div><div class="meta">${esc(m.time?.slice(11, 16))} · ${m.kind} · ${m.source_type} · importance ${m.importance}${m.score ? ` · score ${m.score.s} (rel ${m.score.rel}, rec ${m.score.rec}, imp ${m.score.imp})` : ""}${S.debug && m.originating_event_ids?.length ? ` <span class="tag gt">events ${m.originating_event_ids.join(",")}</span>` : ""}</div></div>`;
  const mods = st.modules && Object.keys(st.modules).length ? `<dt>Modules</dt><dd>${esc(JSON.stringify(st.modules))}</dd>` : "";
  const place = M.data?.places?.[st.location];
  $("#agentPanel").innerHTML = `
    <div class="agent-head"><div class="avatar" style="background-image:url(/ga_assets/characters/${p.sprite}.png)"></div>
      <div><div style="font-weight:700">${esc(p.name)}</div><div class="muted small">${esc(p.demographics.year)} · ${esc(p.demographics.major)} · ${esc(p.demographics.role)}</div>
      <button class="pill" id="btnLocate">locate on map</button></div></div>
    <dl class="kv">
      <dt>Now</dt><dd>${esc(st.activity)} <span class="muted">@ ${esc(st.location)}${place && place.label !== st.location ? ` (${esc(place.label)})` : ""} / ${esc(st.arena)}</span></dd>
      <dt>Goal</dt><dd>${esc(st.goal || "follow the routine")}</dd>
      <dt>Personality</dt><dd>${esc(p.personality.traits.join(", "))}; ${esc(p.personality.communication_style)}</dd>
      <dt>Interests</dt><dd>${esc([...p.interests.topics, ...p.interests.hobbies].join(", "))}</dd>
      <dt>Clubs</dt><dd>${esc(p.interests.clubs.join(", ") || "—")}</dd>
      <dt>Background</dt><dd>${esc(p.background)}</dd>
      <dt>Memories</dt><dd>${d.n_memories} in stream</dd>${mods}
    </dl>
    <details><summary>Relationships</summary><div class="small">${rel}</div></details>
    <details><summary>Routine</summary><div class="small">${p.routine.map((r) => `${r.time} ${esc(r.activity)} <span class="muted">@ ${esc(r.location)}</span>`).join("<br>")}</div></details>
    <details open><summary>Last retrieved memories ${d.retrieved.tick != null ? `<span class="muted small">(tick ${d.retrieved.tick})</span>` : ""}</summary>${d.retrieved.memories.map(mem).join("") || '<div class="muted small">none yet</div>'}</details>
    <details open><summary>Reflections (${d.reflections.length})</summary>${d.reflections.map(mem).join("") || '<div class="muted small">none yet</div>'}</details>
    <details><summary>Recent memories</summary>${d.memories.slice(0, 25).map(mem).join("")}</details>
    <details><summary>Recent conversations (${d.conversations.length})</summary>${d.conversations.map((c) => `<div class="mem chat">${c.transcript.map(([s, t]) => `<b>${esc(s.split(" ")[0])}:</b> ${esc(t)}`).join("<br>")}<div class="meta">${esc(c.time.slice(11, 16))} @ ${esc(c.location)}</div></div>`).join("")}</details>`;
  $("#btnLocate").onclick = () => { const a = drawState.pos[aid]; if (a) { if (cam.zoom < 0.75) cam.zoom = 1; centerOn(a.x, a.y); } cam.follow = true; $("#followSel").checked = true; };
}

// ---------------------------------------------------------------- culture
const STATUS = { established: "●", spreading: "▲", emerging: "○", fading: "▽" };
function orderedCands() {
  const cs = [...(S.analysis?.candidates || [])];
  if ($("#onlyConv").checked) cs.sort((a, b) => (!!b.llm?.is_convention - !!a.llm?.is_convention) || b.score - a.score);
  return cs;
}
function renderCulture() {
  const A = S.analysis;
  if (!A) { $("#cultureSummary").innerHTML = `<span class="muted">This run has not been analyzed yet. Use “Re-run analysis” or <code>python -m backend.cli analyze runs/${esc(S.runId)}</code>.</span>`; $("#cards").innerHTML = ""; $("#memeDetail").innerHTML = ""; return; }
  const s = A.summary;
  $("#cultureSummary").innerHTML = [["Utterances", s.n_utterances], ["Conversations", s.n_conversations], ["World events", s.n_events], ["Candidates", s.n_candidates], ["LLM-classified conventions", s.n_llm_conventions]]
    .map(([l, v]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  const cs = orderedCands();
  $("#cards").innerHTML = cs.slice(0, 16).map((c) => {
    const k = c.card;
    return `<div class="card ${S.selMeme === c.id ? "sel" : ""}" data-id="${c.id}">
      <div class="form">“${esc(c.display_form || c.canonical_form)}”</div>
      <div class="row"><span>First used</span><b>${esc(k.first_used_by)}, ${esc(k.first_used_label)}</b></div>
      <div class="row"><span>Users</span><b>${k.users} / ${k.population}</b></div>
      <div class="row"><span>Uses</span><b>${k.uses}</b></div>
      <div class="row"><span>Transmission depth</span><b>${k.transmission_depth}</b></div>
      <div class="row"><span>Semantic coherence</span><b>${k.semantic_coherence ?? "—"}</b></div>
      <div class="row"><span>Latent-event alignment</span><b>${S.debug ? (k.latent_alignment ?? "—") : '<span class="lock">debug only</span>'}</b></div>
      <div class="row"><span>Status</span><b class="status">${STATUS[k.status] || ""} ${esc(k.status)}</b></div>
      <div class="row"><span>LLM classifier</span><b>${k.is_convention ? "convention" : k.is_convention === false ? "not a convention" : "—"}</b></div>
    </div>`;
  }).join("");
  $$("#cards .card").forEach((d) => d.onclick = () => { S.selMeme = d.dataset.id; renderCulture(); });
  if (!S.selMeme && cs.length) S.selMeme = cs[0].id;
  renderMemeDetail(cs.find((c) => c.id === S.selMeme));
  renderOverview(cs);
}

function renderOverview(cs) {
  // appended at the top of the detail grid: adoption of the top-4 conventions (categorical slots 1-4, fixed order)
  const top = cs.slice(0, 4);
  if (!top.length) return;
  const T = S.manifest.ticks;
  const series = top.map((c, i) => ({ name: c.display_form || c.canonical_form, color: `var(--series-${i + 1})`,
    pts: stepPts(S.analysis.transmission[c.id].adoption_curve.map((p) => [p.tick, p.users]), T) }));
  const html = `<div class="panel chart"><h3>Adoption over time — top candidates</h3>${legend(series)}${lineChart(series, T, S.manifest.config.population_size || 8, "distinct users")}</div>`;
  $("#memeDetail").insertAdjacentHTML("afterbegin", html);
  bindChartHover();
}
function stepPts(curve, T) { const pts = [[0, 0]]; for (const [t, v] of curve) { pts.push([t, pts[pts.length - 1][1]], [t, v]); } pts.push([T, pts[pts.length - 1][1]]); return pts; }
function legend(series) { return `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>`; }

function lineChart(series, xmax, ymax, ylabel) {
  const w = 560, h = 200, m = { l: 34, r: 90, t: 8, b: 26 };
  const X = (x) => m.l + (x / Math.max(1, xmax)) * (w - m.l - m.r), Y = (y) => h - m.b - (y / Math.max(1, ymax)) * (h - m.t - m.b);
  const tpd = S.manifest.ticks_per_day;
  let axis = `<g class="axis"><line x1="${m.l}" x2="${w - m.r}" y1="${h - m.b}" y2="${h - m.b}"/>`;
  for (let d = 0; d * tpd <= xmax; d++) axis += `<line x1="${X(d * tpd)}" x2="${X(d * tpd)}" y1="${m.t}" y2="${h - m.b}" stroke-dasharray="2 3"/><text x="${X(d * tpd) + 3}" y="${h - 8}">Day ${d + 1}</text>`;
  for (let v = 0; v <= ymax; v += Math.max(1, Math.round(ymax / 4))) axis += `<text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  axis += `<text x="${m.l}" y="${m.t + 2}" dy="-0" font-size="10">${esc(ylabel)}</text></g>`;
  const lines = series.map((s) => {
    const last = s.pts[s.pts.length - 1];
    return `<path d="${s.pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("")}" fill="none" stroke="${s.color}" stroke-width="2"/>
      <text x="${X(last[0]) + 4}" y="${Y(last[1]) + 4}" font-size="11" fill="var(--text-2)">${esc(s.name.slice(0, 14))}</text>`;
  }).join("");
  const data = esc(JSON.stringify(series.map((s) => ({ n: s.name, p: s.pts }))));
  return `<svg viewBox="0 0 ${w} ${h}" data-series="${data}" data-xmax="${xmax}" data-ml="${m.l}" data-mr="${m.r}" data-w="${w}" class="hoverable">${axis}${lines}<line class="xhair" y1="${m.t}" y2="${h - m.b}" stroke="var(--muted)" visibility="hidden"/></svg>`;
}
function bindChartHover() {
  $$("svg.hoverable").forEach((svg) => {
    const series = JSON.parse(svg.dataset.series), xmax = +svg.dataset.xmax, ml = +svg.dataset.ml, mr = +svg.dataset.mr, w = +svg.dataset.w;
    svg.onmousemove = (e) => {
      const r = svg.getBoundingClientRect(), sx = (e.clientX - r.left) * w / r.width;
      const t = Math.round(((sx - ml) / (w - ml - mr)) * xmax);
      if (t < 0 || t > xmax) return;
      const xh = svg.querySelector(".xhair"); xh.setAttribute("x1", sx); xh.setAttribute("x2", sx); xh.setAttribute("visibility", "visible");
      const vals = series.map((s) => { let v = 0; for (const p of s.p) if (p[0] <= t) v = p[1]; return `${esc(s.n)}: <b>${v}</b>`; });
      showTip(e, `${esc(S.frames[Math.min(t, S.frames.length - 1)]?.label || "tick " + t)}<br>${vals.join("<br>")}`);
    };
    svg.onmouseleave = () => { hideTip(); svg.querySelector(".xhair")?.setAttribute("visibility", "hidden"); };
  });
}

function renderMemeDetail(c) {
  const box = $("#memeDetail");
  if (!c) { box.innerHTML = ""; return; }
  const A = S.analysis, tr = A.transmission[c.id], sem = A.semantics[c.id] || {}, pr = A.probes?.[c.id], ev = A.evaluation?.[c.id];
  const T = S.manifest.ticks;
  const hl = (t) => { let s = esc(t); for (const v of [...c.variants].sort((a, b) => b.length - a.length)) s = s.replace(new RegExp(`(${v.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "ig"), "<mark>$1</mark>"); return s; };
  const perDay = Object.entries(tr.uses_per_day);
  const maxDay = Math.max(1, ...perDay.map(([, v]) => v));
  const bars = `<svg viewBox="0 0 300 120">${perDay.map(([d, v], i) => { const bh = (v / maxDay) * 80, x = 30 + i * 60;
    return `<rect x="${x}" y="${100 - bh}" width="36" height="${bh}" rx="4" fill="var(--series-1)" data-tip="Day ${d}: ${v} uses"/><text x="${x + 18}" y="${96 - bh}" text-anchor="middle" font-size="11" fill="var(--text-2)">${v}</text><text x="${x + 18}" y="114" text-anchor="middle" font-size="11" fill="var(--muted)">Day ${d}</text>`; }).join("")}</svg>`;
  const adoption = lineChart([{ name: "distinct users", color: "var(--series-1)", pts: stepPts(tr.adoption_curve.map((p) => [p.tick, p.users]), T) }], T, Object.keys(S.manifest.agents).length, "distinct users");
  box.innerHTML = `
    <div class="panel chart"><h3>“${esc(c.display_form || c.canonical_form)}” — adoption</h3>${adoption}<div class="muted small">Uses per day</div>${bars}</div>
    <div class="panel chart"><h3>Propagation network</h3>${network(c, tr)}</div>
    <div class="panel"><h3>Lexical lineage</h3>${lineageHtml(c)}</div>
    <div class="panel"><h3>Meaning over time</h3>
      <div class="small">Observer LLM gloss: <i>${esc(c.llm?.gloss || "—")}</i> ${c.llm ? `(confidence ${c.llm.confidence})` : ""}</div>
      <table class="table"><tr><th>Day</th><th>Uses</th><th>Context similarity to overall meaning</th><th>… to Day-1 meaning</th></tr>
      ${(sem.over_time || []).map((r) => `<tr><td>${r.day}</td><td>${r.n}</td><td>${r.sim_to_overall}</td><td>${r.sim_to_first_day}</td></tr>`).join("")}</table>
      <div class="small muted">Coherence (mean pairwise context similarity, expression masked): ${sem.coherence ?? "—"} · between-group centroid similarity: ${sem.between_group_centroid_sim ?? "—"}</div>
      <div class="small muted">Within-group coherence: ${Object.entries(sem.within_group || {}).map(([g, v]) => `${esc(g)} ${v}`).join(" · ") || "—"}</div></div>
    <div class="panel"><h3>Agent interpretations <span class="muted small">(private probes; never enter agent memory)</span></h3>${probesHtml(pr)}</div>
    <div class="panel"><h3>Latent-event correspondence</h3>${S.debug ? latentHtml(ev) : '<div class="lock">Hidden ground truth. Enable Research debug to see latent-event alignment, precision/recall, generalization and drift.</div>'}</div>
    <div class="panel" style="grid-column:1/-1"><h3>Usages (${c.usage_count})</h3><table class="table"><tr><th>When</th><th>Speaker</th><th>Listeners</th><th>Utterance</th>${S.debug ? "<th>Linked latent types</th>" : ""}<th></th></tr>
      ${c.usages.map((u) => `<tr><td>${esc(S.frames[u.tick]?.label || u.tick)}</td><td>${esc(name(u.speaker))}</td><td>${u.listeners.map(name).map(esc).join(", ")}</td><td>${hl(u.text)}</td>${S.debug ? `<td>${esc(Object.keys(u.latent_types || {}).join(", "))}</td>` : ""}
      <td><button class="pill" data-jump="${u.tick}">campus</button> <button class="pill" data-chain="${esc(u.utterance_id)}">trace</button></td></tr>`).join("")}</table></div>`;
  $$("[data-jump]", box).forEach((b) => b.onclick = () => { showView("campus"); setTick(+b.dataset.jump); });
  $$("[data-chain]", box).forEach((b) => b.onclick = () => openChain(b.dataset.chain));
  $$("[data-tip]", box).forEach((el) => { el.onmousemove = (e) => showTip(e, el.dataset.tip); el.onmouseleave = hideTip; });
  bindChartHover();
}

function network(c, tr) {
  const ids = Object.keys(S.manifest.agents), w = 420, h = 320, cx = w / 2, cy = h / 2, R = 120;
  const P = {}; ids.forEach((a, i) => { const t = (i / ids.length) * Math.PI * 2 - Math.PI / 2; P[a] = { x: cx + R * Math.cos(t), y: cy + R * Math.sin(t) }; });
  const users = new Set(c.speakers), inv = new Set(tr.inventors.filter((x) => !x.partial).map((x) => x.agent));
  const exposed = new Set(c.usages.flatMap((u) => u.listeners));
  const edges = tr.edges.map((e) => {
    const a = P[e.source_agent], b = P[e.target_agent], dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy);
    const ex = b.x - dx / L * 16, ey = b.y - dy / L * 16, col = e.cross_group ? "var(--series-2)" : "var(--series-1)";
    return `<line x1="${a.x}" y1="${a.y}" x2="${ex}" y2="${ey}" stroke="${col}" stroke-width="${1 + 4 * e.confidence}" marker-end="url(#arr${e.cross_group ? 2 : 1})" opacity="${0.35 + 0.65 * e.confidence}"
      data-tip="${esc(name(e.source_agent))} → ${esc(name(e.target_agent))}<br>confidence ${e.confidence}<br>first reuse ${esc(S.frames[e.first_reuse_timestamp]?.label || "")}${e.cross_group ? "<br><b>cross-group</b>" : ""}"/>`;
  }).join("");
  const nodes = ids.map((a) => `<g data-tip="${esc(S.manifest.agents[a].name)}<br>${users.has(a) ? "used it" : exposed.has(a) ? "heard it, never used it" : "never exposed"}${inv.has(a) ? "<br><b>independent originator</b>" : ""}">
    <circle cx="${P[a].x}" cy="${P[a].y}" r="14" fill="${users.has(a) ? "var(--series-1)" : "var(--surface-2)"}" stroke="${inv.has(a) ? "var(--text)" : exposed.has(a) ? "var(--series-1)" : "var(--border)"}" stroke-width="${inv.has(a) ? 3 : 1.5}"/>
    <text x="${P[a].x}" y="${P[a].y + 28}" text-anchor="middle" font-size="11" fill="var(--text-2)">${esc(name(a))}</text></g>`).join("");
  return `<div class="legend"><span><i style="background:var(--series-1)"></i>within-group transmission</span><span><i style="background:var(--series-2)"></i>cross-group transmission</span><span>● filled = user · thick ring = originator</span></div>
    <svg viewBox="0 0 ${w} ${h}"><defs>${[1, 2].map((k) => `<marker id="arr${k}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="var(--series-${k})"/></marker>`).join("")}</defs>${edges}${nodes}</svg>
    <div class="small muted">Depth ${tr.depth} · ${tr.edges.length} candidate transmission edges · ${tr.n_exposed} agents exposed. Edges are plausible exposures before first reuse, confidence-weighted (the most recent speaker is not assumed causal).</div>`;
}
function lineageHtml(c) {
  const v = S.analysis.lineage.variants[c.id] || [];
  const cl = (S.analysis.lineage.candidates || []).filter((e) => e.parent === c.id || e.child === c.id).slice(0, 8);
  return `<div class="small"><b>Variants:</b> ${c.variants.map((x) => `<span class="tag">${esc(x)}</span>`).join("")}</div>
    <table class="table"><tr><th>Parent variant</th><th>Child variant</th><th>Confidence</th></tr>${v.map((e) => `<tr><td>${esc(e.parent)}</td><td>${esc(e.child)}</td><td>${e.confidence}${e.asserted ? "" : " <span class='muted'>(not asserted)</span>"}</td></tr>`).join("") || "<tr><td colspan=3 class='muted'>single variant</td></tr>"}</table>
    <div class="small" style="margin-top:6px"><b>Related candidates</b></div>
    <table class="table">${cl.map((e) => `<tr><td>“${esc(e.parent_form)}” → “${esc(e.child_form)}”</td><td>${e.confidence}${e.asserted ? "" : " <span class='muted'>(low confidence)</span>"}</td><td class="muted small">lex ${e.components.lexical} · ctx ${e.components.context} · time ${e.components.temporal} · exposure ${e.components.exposure_overlap}</td></tr>`).join("") || "<tr><td class='muted'>none</td></tr>"}</table>`;
}
function probesHtml(pr) {
  if (!pr) return `<div class="muted small">Not probed (only the top candidates are probed).</div>`;
  const opts = pr.options.map((o) => `<div class="small"><b>${o.letter}.</b> ${esc(o.text)} ${S.debug ? `<span class="tag gt">${o.family}</span>` : ""}</div>`).join("");
  return `<table class="table"><tr><th>Agent</th><th>Exposure</th><th>Private interpretation</th><th>Best-matching novel situation</th></tr>
    ${Object.entries(pr.agents).map(([a, r]) => `<tr><td>${esc(name(a))}</td><td class="small">${r.used ? "used" : r.heard_before ? "heard" : "never"}</td><td>${esc(r.meaning)}</td><td>${esc(r.match_choice || "—")}${S.debug && r.match_family ? ` <span class="tag gt">${r.match_family}</span>` : ""}</td></tr>`).join("")}</table>
    <details><summary>Novel situations used in the generalization probe</summary>${opts}</details>`;
}
function latentHtml(ev) {
  if (!ev || !ev.best_type) return `<div class="muted small">No usages could be linked to world events.</div>`;
  const d = ev.latent_distribution, tot = Object.values(d).reduce((a, b) => a + b, 0), lt = S.manifest.latent_types || {};
  const rows = Object.entries(d).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<div class="small" style="display:flex;gap:8px;align-items:center"><span style="width:200px">${k} ${esc(lt[k]?.name || "")}</span>
    <svg viewBox="0 0 200 12" width="200" height="12"><rect x="0" y="1" height="10" rx="3" width="${(v / tot) * 200}" fill="var(--series-1)"/></svg><span>${(100 * v / tot).toFixed(0)}%</span></div>`).join("");
  const g = ev.generalization || {};
  return `${rows}<dl class="kv" style="margin-top:8px">
    <dt>Best latent type</dt><dd><span class="tag gt">${ev.best_type}</span> ${esc(lt[ev.best_type]?.description || "")}</dd>
    <dt>Precision</dt><dd>${ev.precision} <span class="muted">(usages linked to it; lift ${ev.lift ?? "—"} over the base rate)</span></dd>
    <dt>Recall</dt><dd>${ev.recall ?? "—"} <span class="muted">(later instances that triggered it)</span></dd>
    <dt>Alignment (F1)</dt><dd>${ev.alignment}</dd>
    <dt>Generalization</dt><dd>${g.holdout_instances ?? 0} held-out instances; spontaneous use rate ${g.spontaneous_use_rate ?? "—"}; probe match accuracy ${g.probe_match_accuracy ?? "—"}</dd>
    <dt>Semantic drift</dt><dd>${ev.drift ? `precision ${ev.drift.precision_first_half} → ${ev.drift.precision_second_half}` : "—"}</dd></dl>`;
}

// ------------------------------------------------------------------ trace
function allUtterances() { const out = []; S.frames.forEach((f, t) => f.utterances.forEach((u) => out.push([t, u]))); return out; }
function renderTraceSearch() {
  const q = $("#traceQuery").value.toLowerCase();
  const hits = allUtterances().filter(([, u]) => !q || u.text.toLowerCase().includes(q)).slice(-80).reverse();
  $("#traceResults").innerHTML = hits.map(([t, u]) => `<div class="utt" data-uid="${esc(u.id)}"><span class="who">${esc(name(u.speaker))}</span> <span class="meta">${esc(S.frames[t].label)}</span><div>${esc(u.text)}</div></div>`).join("");
  $$("#traceResults .utt").forEach((d) => d.onclick = async () => { $("#chain").innerHTML = await chainHtml(d.dataset.uid); });
}
async function openChain(uid) {
  $("#modalContent").innerHTML = `<h2>Causal trace</h2>` + await chainHtml(uid);
  $("#modal").hidden = false;
}
async function chainHtml(uid) {
  let c;
  try { c = await api(`/runs/${S.runId}/chain/${encodeURIComponent(uid)}?depth=3`); } catch (e) { return `<div class="muted">${esc(e.message)}</div>`; }
  const memNode = (m) => {
    if (m.missing) return `<div class="node mem"><span class="kind">memory</span> <span class="muted">(${esc(m.node_id)} not found)</span></div>`;
    let inner = "";
    if (m.world_event) inner += `<div class="node world"><span class="kind">world event (ground truth)</span> <span class="tag gt">${esc(m.world_event.latent_type)} · ${esc(m.world_event.id)} · ${esc(m.world_event.scenario)}</span><div>${esc(m.world_event.narrative)}</div></div>`;
    if (m.observation) inner += `<div class="node obs"><span class="kind">${esc(m.observation.source_type)} observation — what ${esc(name(m.agent))} actually noticed</span><div>${(m.observation.facts || []).map((f) => esc(f.text) + (f.p_attend != null ? ` <span class="muted small">(p=${f.p_attend})</span>` : "")).join("<br>")}</div></div>`;
    inner += (m.from_utterances || []).map(uttNode).join("") + (m.from_memories || []).map(memNode).join("");
    return `<div class="node mem"><span class="kind">${esc(name(m.agent))}'s ${esc(m.kind)} memory · ${esc(m.source_type)}</span><div>${esc(m.text)}</div>${inner}</div>`;
  };
  const uttNode = (u) => `<div class="node utt"><span class="kind">utterance · ${esc(name(u.speaker))} → ${u.listeners.map(name).map(esc).join(", ") || "nobody"} · ${esc(u.time?.slice(11, 16))}</span><div>“${esc(u.text)}”</div>
      ${u.retrieved?.length ? `<div class="small muted" style="margin-left:18px">retrieved memories:</div>${u.retrieved.map(memNode).join("")}` : ""}</div>`;
  return `${uttNode(c)}
    <div class="node lmem"><span class="kind">listener memories formed from this utterance</span>${(c.listener_memories || []).map((m) => `<div><b>${esc(name(m.agent))}</b> (${esc(m.source_type)}): ${esc(m.text)}</div>`).join("") || "<div class='muted'>none</div>"}</div>
    <div class="node utt"><span class="kind">later utterances that retrieved those memories</span>${(c.later_utterances_using_these_memories || []).map((u) => `<div><b>${esc(name(u.speaker))}</b>: ${esc(u.text)}</div>`).join("") || "<div class='muted'>none</div>"}</div>`;
}

// ------------------------------------------------------------------- runs
async function renderRuns() {
  S.runs = await api("/runs");
  $("#runsTable").innerHTML = `<tr><th>Run</th><th>Status</th><th>Config</th><th>LLM</th><th>Modules</th><th>Events</th><th>Conversations</th><th>Utterances</th><th>LLM calls</th><th>Analysis</th></tr>` +
    S.runs.map((r) => `<tr><td><a href="?run=${encodeURIComponent(r.run_id)}">${esc(r.run_id)}</a></td><td>${esc(r.status)}</td><td>${esc(r.run_name)} (seed ${r.seed}, ${r.days}d)</td><td>${esc(r.llm_backend)}</td><td>${esc((r.modules || []).join(", ") || "none")}</td>
      <td>${r.stats?.events ?? ""}</td><td>${r.stats?.conversations ?? ""}</td><td>${r.stats?.utterances ?? ""}</td><td>${r.stats?.llm?.calls ?? ""}</td><td>${r.has_analysis ? "yes" : "—"}</td></tr>`).join("");
  const cmp = await api("/compare");
  const cols = ["condition", "events", "conversations", "utterances", "reflections", "candidates", "llm_conventions", "max_adoption", "mean_depth", "cross_group_edges", "mean_coherence", ...(S.debug ? ["mean_alignment", "mean_lift"] : []), "top_expression"];
  $("#compareTable").innerHTML = `<tr>${cols.map((c) => `<th>${c.replace(/_/g, " ")}</th>`).join("")}</tr>` +
    cmp.map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c] ?? "—")}</td>`).join("")}</tr>`).join("");
  const cfgs = await api("/configs");
  $("#cfgSelect").innerHTML = cfgs.map((c) => `<option>${esc(c)}</option>`).join("");
}
async function launchRun() {
  const q = new URLSearchParams({ config: $("#cfgSelect").value, days: $("#cfgDays").value, backend: $("#cfgBackend").value });
  const r = await fetch(`/api/runs?${q}`, { method: "POST" });
  $("#launchMsg").textContent = r.ok ? "Launched — it will appear in the list shortly (refresh)." : "Launch failed.";
}

boot();
