// MemeWorld frontend: pixel-art Homewood campus replay, culture dashboard, causal trace explorer.
// Normal demo mode never requests hidden ground truth; Research Debug Mode adds ?debug=1.

import { renderCultureTrends } from "./culture.js";
import { mountResearch, refreshResearch, renderMemeticsCulture } from "./research.js";
import { loadRealMap, drawRealMap, drawRealMapLabels, REALMAP_ATTRIBUTION } from "./realmap.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const S = {
  runs: [], runId: null, debug: false, manifest: null, frames: [], analysis: null,
  tick: 0, tf: 0, playing: false, speed: 1, sel: null, sprites: {},
  selMeme: null, agentReq: 0, order: {},
  coop: null,          // GET /runs/<id>/coop (v3 co-op view; enabled=false for v2 / older runs)
  hasCoop: false,      // frames carry a co-op block or the run has co-op mechanisms on
  schema: null,        // GET /schema (trace record registry; debug adds hidden types)
  agentTrace: {},      // per-agent NEED / WORDING / forgetting records (trace endpoint), per run
};

// Fields added over time may be missing in older runs: default them so every view can rely on them.
const AGENT_DEFAULTS = { active: true, role: null, cohort: null, open_matters: 0, wordings: 0 };
function normalizeFrames(frames) {
  for (const f of frames) {
    f.agents = f.agents || {};
    for (const a of Object.values(f.agents)) for (const [k, v] of Object.entries(AGENT_DEFAULTS)) if (a[k] === undefined) a[k] = v;
    f.away = Array.isArray(f.away) ? f.away : Object.keys(f.agents).filter((aid) => f.agents[aid].active === false).sort();
    f.utterances = f.utterances || []; f.beats = f.beats || [];
    if (f.coop) f.coop.jobs = Array.isArray(f.coop.jobs) ? f.coop.jobs : [];
  }
  return frames;
}
const ROLE = { am_crew: ["AM", "AM crew", "#2a78d6"], pm_crew: ["PM", "PM crew", "#eb6834"], stores: ["ST", "Stores", "#1baf7a"] };
const roleName = (r) => ROLE[r]?.[1] || (r ? String(r).replace(/_/g, " ") : "");
const roleBadge = (r) => r ? `<span class="badge role" style="--c:${ROLE[r]?.[2] || "#7a7973"}" title="co-op role">${esc(roleName(r))}</span>` : "";
const cohortBadge = (c) => c ? `<span class="badge ${c === "newcomer" ? "new" : "founder"}" title="roster cohort">${esc(c)}</span>` : "";
const agentBadges = (st) => roleBadge(st?.role) + cohortBadge(st?.cohort) + (st && st.active === false ? `<span class="badge away" title="not on the co-op roster / off campus">away</span>` : "");
const firstName = (aid) => S.manifest?.agents?.[aid]?.name?.split(" ")[0] || aid;
const spriteOf = (aid) => S.manifest?.agents?.[aid]?.sprite;
const headIcon = (aid) => spriteOf(aid) ? `<span class="head" style="background-image:url(/ga_assets/characters/${encodeURIComponent(spriteOf(aid))}.png)"></span>` : `<span class="head"></span>`;

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
  mountResearch({run: () => S.runId, replay: (tick) => { showView("campus"); setTick(Math.max(0,tick)); }});
  const qs = new URLSearchParams(location.search);
  if (qs.get("view") === "research") showView("research");
  $$(".tab").forEach((b) => b.onclick = () => showView(b.dataset.view));
  $("#debugToggle").onchange = async (e) => { S.debug = e.target.checked; $("#debugBadge").hidden = !S.debug; await loadRun(S.runId, true); };
  $("#runSelect").onchange = (e) => loadRun(e.target.value);
  $("#btnPlay").onclick = togglePlay;
  $("#btnBack").onclick = () => setTick(S.tick - 1);
  $("#btnFwd").onclick = () => setTick(S.tick + 1);
  $("#speed").onchange = (e) => S.speed = +e.target.value;
  $("#scrub").oninput = (e) => setTick(+e.target.value);
  $("#worldScrub").oninput = (e) => setTick(+e.target.value);
  $("#worldBack").onclick = () => setTick(S.tick - 1);
  $("#worldFwd").onclick = () => setTick(S.tick + 1);
  $("#btnFirstMeme").onclick = () => jumpTo("first_meme_use");
  $("#btnFirstCross").onclick = () => jumpTo("first_cross_group_transmission");
  $("#modalClose").onclick = () => $("#modal").hidden = true;
  $("#modal").onclick = (e) => { if (e.target.id === "modal") $("#modal").hidden = true; };
  $("#onlyConv").onchange = renderCulture;
  $("#btnAnalyze").onclick = async () => { await fetch(`/api/runs/${S.runId}/analyze`, { method: "POST" }); $("#cultureSummary").innerHTML = `<span class="muted">Analysis launched; reload this run in a minute.</span>`; };
  $("#traceQuery").oninput = renderTraceSearch;
  bindTraceExplorer();
  $("#btnLaunch").onclick = launchRun;
  bindMapControls();
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowRight") setTick(S.tick + 1);
    if (e.code === "ArrowLeft") setTick(S.tick - 1);
    if (e.key === "+" || e.key === "=") zoomBy(1);
    if (e.key === "-") zoomBy(-1);
  });
  const mapReady = loadMap();
  // real-world (OpenStreetMap) layer: Tiles | Real map toggle, remembered per viewer
  try { S.mapMode = localStorage.getItem("mw.mapMode") === "real" ? "real" : "tiles"; } catch { S.mapMode = "tiles"; }
  loadRealMap().catch(() => {});
  const mm = $("#mapMode");
  if (mm) {
    const sync = () => { mm.textContent = S.mapMode === "real" ? "tiles" : "real map"; const at = $("#mapAttrib"); if (at) { at.hidden = S.mapMode !== "real"; at.textContent = REALMAP_ATTRIBUTION; } };
    mm.onclick = () => { S.mapMode = S.mapMode === "real" ? "tiles" : "real"; try { localStorage.setItem("mw.mapMode", S.mapMode); } catch {} sync(); };
    sync();
  }
  S.runs = await api("/runs");
  const sel = $("#runSelect");
  sel.innerHTML = '<option value="">Select a recorded society</option>' + S.runs.map((r) => `<option value="${esc(r.run_id)}">${esc(r.run_id)} (${r.status}${r.has_analysis ? ", analyzed" : ""})</option>`).join("");
  const want = new URLSearchParams(location.search).get("run");
  const first = S.runs.find((r) => r.run_id === want) || (qs.get("view") !== "research" && (S.runs.find((r) => r.status === "finished" && r.has_analysis) || S.runs[0]));
  if (first) { sel.value = first.run_id; await loadRun(first.run_id); }
  await mapReady;
  if (qs.get("tick")) setTick(+qs.get("tick"));
  if (qs.get("view") === "fit") zoomFit();
  if (qs.get("view") === "research") showView("research");
  if (qs.get("zoom")) setZoom(+qs.get("zoom"));
  if (qs.get("place") && M.data.places[qs.get("place")]) { const b = M.data.places[qs.get("place")].box; centerOn((b[0] + b[2] + 1) / 2 * TILE, (b[1] + b[3] + 1) / 2 * TILE); }
  if (qs.get("frac")) S.tf = S.tick + Math.min(0.999, Math.max(0, +qs.get("frac")));
  if (qs.get("agent")) { S.sel = qs.get("agent"); renderAgent(S.sel); if (qs.get("follow")) { cam.follow = true; $("#followSel").checked = true; } }
  renderAwayTray();
  if (qs.get("tab") && $(`.tab[data-view="${qs.get("tab")}"]`)) showView(qs.get("tab"));
  renderRuns();
  requestAnimationFrame(loop);
}

function showView(v) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === "view-" + v));
  if (v === "culture") { renderCulture(); drawTrends(); }
  if (v === "trace") { renderTraceSearch(); renderTypeFilter(); }
  if (v === "runs") renderRuns();
  if (v === "research") refreshResearch();
  if (v === "campus") resizeMap();
}

async function loadRun(id, keepTick = false) {
  if (!id) return;
  S.runId = id;
  const [man, frames] = await Promise.all([api(`/runs/${id}/manifest`), api(`/runs/${id}/frames`)]);
  S.manifest = man; S.frames = normalizeFrames(frames);
  man.agents = man.agents || {};
  // only ask for the analysis when the run list says there is one (a 404 would be a console error)
  const info = S.runs.find((r) => r.run_id === id);
  S.analysis = null;
  if (!info || info.has_analysis) { try { S.analysis = await api(`/runs/${id}/analysis`); } catch { S.analysis = null; } }
  try { S.coop = await api(`/runs/${id}/coop`); } catch { S.coop = null; }
  S.hasCoop = !!(S.coop?.enabled || S.frames.some((f) => f.coop));
  try { S.schema = await api(`/schema`); } catch { S.schema = null; }
  S.agentTrace = {}; jobsCache.clear(); trState.records = null;
  // every agent that ever appears in a frame gets a stable order (newcomers may be missing from older manifests)
  const ids = new Set(Object.keys(man.agents)); for (const f of S.frames) for (const aid of Object.keys(f.agents)) ids.add(aid);
  S.order = {}; [...ids].sort().forEach((aid, i) => S.order[aid] = i);
  for (const a of Object.values(man.agents)) {
    if (a.sprite && !S.sprites[a.sprite]) { const img = new Image(); img.src = `/ga_assets/characters/${a.sprite}.png`; S.sprites[a.sprite] = img; }
  }
  posCache.length = 0; offCache.length = 0;
  $("#coopGrid").hidden = !S.hasCoop;
  $("#scrub").max = Math.max(0, frames.length - 1);
  if (S.sel && !man.agents[S.sel]) { S.sel = null; $("#agentPanel").innerHTML = `<div class="muted">Click an agent on the map to inspect their profile, memories, retrieval and reflections.</div>`; }
  if (!keepTick) setTick(0); else setTick(S.tick);
  if (S.sel) renderAgent(S.sel);
  renderCulture(); drawTrends();
  resetTraceExplorer();
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

// ---- agent placement: one stable spot per agent inside its arena. Generic over the map's places and arenas:
// an unknown arena falls back to the place's first arena with spots, then to a row across the place's centre.
// Inactive agents (frame.away), the 'Away' sentinel and locations missing from the map go to the away tray.
const posCache = [], offCache = [];
const ord = (aid) => S.order[aid] ?? [...String(aid)].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
const boxesOf = (pl) => (pl.boxes?.length ? pl.boxes : [pl.box]);
function placeOf(a) {
  if (!a || a.active === false || !a.location || a.location === "Away") return null;
  return M.data?.places?.[a.location] || null;
}
function spotsFor(pl, ar) {
  const arenas = pl.arenas || {};
  if (arenas[ar]?.spots?.length) return arenas[ar].spots;
  for (const a of Object.values(arenas)) if (a.spots?.length) return a.spots;
  if (!pl._spots) {
    const [x0, y0, x1, y1] = pl.box, cy = Math.round((y0 + y1) / 2), cx = Math.round((x0 + x1) / 2);
    pl._spots = [];
    for (let x = Math.max(x0, cx - 5); x <= Math.min(x1, cx + 5); x++) pl._spots.push([x, cy]);
  }
  return pl._spots;
}
function positions(t) {
  if (posCache[t]) return posCache[t];
  const frame = S.frames[t], out = {}, off = [];
  if (!frame || !M.data) return out;
  const groups = new Map();
  for (const [aid, a] of Object.entries(frame.agents)) {
    const pl = placeOf(a);
    if (!pl) { off.push(aid); continue; }
    const key = a.location + "\u0000" + (a.arena ?? "");
    if (!groups.has(key)) groups.set(key, { pl, ar: a.arena, ids: [] });
    groups.get(key).ids.push(aid);
  }
  for (const { pl, ar, ids } of groups.values()) {
    const spots = spotsFor(pl, ar); if (!spots.length) { off.push(...ids); continue; }
    ids.sort((a, b) => ord(a) - ord(b));
    const used = new Set();
    for (const aid of ids) {
      let k = ord(aid) % spots.length, n = 0;
      while (used.has(k) && n < spots.length) { k = (k + 1) % spots.length; n++; }
      used.add(k);
      out[aid] = { cx: spots[k][0], cy: spots[k][1] };
    }
  }
  posCache[t] = out; offCache[t] = off.sort((a, b) => ord(a) - ord(b));
  return out;
}
function offMap(t) { positions(t); return offCache[t] || []; }

// the "off campus / away" tray: agents that are not on the map this tick
function renderAwayTray() {
  const tray = $("#awayTray"), f = S.frames[S.tick];
  if (!tray || !f || !M.data) { if (tray) tray.hidden = true; return; }
  const off = offMap(S.tick);
  tray.hidden = !off.length;
  if (!off.length) return;
  tray.innerHTML = `<div class="tray-title">off campus / away · ${off.length}</div><div class="tray-list">${off.map((aid) => {
    const a = f.agents[aid] || {}, why = a.active === false ? "away" : a.location && a.location !== "Away" ? `off map: ${a.location}` : "away";
    return `<button class="tray-agent${S.sel === aid ? " sel" : ""}" data-aid="${esc(aid)}" title="${esc(S.manifest.agents[aid]?.name || aid)} · ${esc(why)}${a.role ? " · " + esc(roleName(a.role)) : ""}">${headIcon(aid)}<span>${esc(firstName(aid))}</span>${a.role ? `<i class="dot" style="background:${ROLE[a.role]?.[2] || "#7a7973"}"></i>` : ""}</button>`;
  }).join("")}</div>`;
  $$(".tray-agent", tray).forEach((b) => b.onclick = () => { S.sel = b.dataset.aid; renderAgent(S.sel); renderAwayTray(); });
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
  renderFeed(); renderEvents(); renderAwayTray(); renderCoop();
  if ($("#view-culture").classList.contains("active")) drawTrends();
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
  if (S.mapMode === "real") drawRealMap(ctx, cam, { dpr }); else drawChunks(ctx, false);
  // active-event outline on buildings (every box of a multi-part place)
  const active = new Set((frame?.beats || []).map((b) => b.location));
  for (const loc of active) {
    const pl = M.data.places[loc]; if (!pl) continue;
    ctx.strokeStyle = "#eb6834"; ctx.lineWidth = 3 / cam.zoom; ctx.setLineDash([8 / cam.zoom, 6 / cam.zoom]);
    for (const [x0, y0, x1, y1] of boxesOf(pl)) ctx.strokeRect(x0 * TILE - 2, y0 * TILE - 2, (x1 - x0 + 1) * TILE + 4, (y1 - y0 + 1) * TILE + 4);
    ctx.setLineDash([]);
  }
  // agents, painter's order by y
  const ids = Object.keys(pos).sort((a, b) => pos[a].y - pos[b].y);
  const far = cam.zoom < 0.25;   // far out: sprites are a few pixels; draw a marker dot instead
  for (const aid of ids) {
    const a = pos[aid], img = S.sprites[spriteOf(aid)];
    const fx = a.walking ? [0, 32, 64][Math.floor(performance.now() / 140) % 3] : 32;
    if (S.sel === aid) {
      ctx.fillStyle = "rgba(42,120,214,.35)"; ctx.beginPath(); ctx.ellipse(a.x, a.y + 8, 14, 7, 0, 0, 7); ctx.fill();
    }
    if (far) {
      const r = 5 / cam.zoom, st = frame.agents[aid];
      ctx.fillStyle = "#1c1c1a"; ctx.fillRect(a.x - r - 1 / cam.zoom, a.y - r - 1 / cam.zoom, 2 * r + 2 / cam.zoom, 2 * r + 2 / cam.zoom);
      ctx.fillStyle = S.sel === aid ? "#ffd866" : ROLE[st?.role]?.[2] || "#e34948"; ctx.fillRect(a.x - r, a.y - r, 2 * r, 2 * r);
    } else if (img?.complete && img.naturalWidth) ctx.drawImage(img, fx, a.dir * 32, 32, 32, Math.round(a.x - 16), Math.round(a.y - 22), 32, 32);
    else { ctx.fillStyle = "#2a78d6"; ctx.beginPath(); ctx.arc(a.x, a.y - 6, 9, 0, 7); ctx.fill(); }
  }
  if (S.mapMode !== "real") drawChunks(ctx, true);
  // screen-space overlays (labels, names, bubbles, badges)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const sx = (wx) => (wx - cam.x) * cam.zoom, sy = (wy) => (wy - cam.y) * cam.zoom;
  if (S.mapMode === "real") drawRealMapLabels(ctx, cam, { dpr }); else drawContextLabels(ctx, sx, sy, vw, vh);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const occ = {};
  for (const aid of ids) { const l = frame.agents[aid]?.location; occ[l] = (occ[l] || 0) + 1; }
  drawPlaceLabels(ctx, sx, sy, active, occ, vw, vh);
  for (const aid of ids) {
    const a = pos[aid], x = sx(a.x), y = sy(a.y);
    if (x < -40 || y < -40 || x > vw + 40 || y > vh + 40) continue;
    if (far && S.sel !== aid) continue;   // far out, the place labels carry head counts; only the selection is named
    const st = frame.agents[aid] || {};
    const nm = firstName(aid);
    ctx.font = `600 ${cam.zoom >= 0.75 ? 13 : 11}px ${PIX}`; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const w = ctx.measureText(nm).width + 8, yy = y + (far ? 8 : 12 * cam.zoom);
    // co-op badges (role, newcomer) sit right of the name tag
    const badges = [];
    if (st.role) badges.push([ROLE[st.role]?.[0] || String(st.role).slice(0, 2).toUpperCase(), ROLE[st.role]?.[2] || "#7a7973", "#fff"]);
    if (st.cohort === "newcomer") badges.push(["NEW", "#ffd866", "#1c1c1a"]);
    ctx.font = `700 10px ${PIX}`;
    const bw = badges.map(([t]) => ctx.measureText(t).width + 6), tw = w + bw.reduce((s, v) => s + v + 1, 0);
    let bx = Math.round(x - tw / 2);
    ctx.font = `600 ${cam.zoom >= 0.75 ? 13 : 11}px ${PIX}`;
    ctx.fillStyle = S.sel === aid ? "#2a78d6" : "rgba(20,20,18,.78)";
    ctx.fillRect(bx, Math.round(yy), Math.round(w), 15);
    ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.fillText(nm, bx + 4, Math.round(yy) + 1);
    bx += Math.round(w) + 1;
    ctx.font = `700 10px ${PIX}`;
    badges.forEach(([t, bg, fg], i) => { ctx.fillStyle = bg; ctx.fillRect(bx, Math.round(yy), Math.round(bw[i]), 15); ctx.fillStyle = fg; ctx.fillText(t, bx + 3, Math.round(yy) + 2); bx += Math.round(bw[i]) + 1; });
    ctx.textAlign = "center";
    if (st.conversation) { ctx.fillStyle = "#ffd866"; ctx.font = `700 12px ${PIX}`; ctx.fillText("…", Math.round(x + 18), Math.round(y - 34 * cam.zoom)); }
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
      for (const l of u.listeners || []) { const lp = pos[l]; if (lp) { ctx.beginPath(); ctx.moveTo(sx(sp.x), sy(sp.y) - 10); ctx.lineTo(sx(lp.x), sy(lp.y) - 10); ctx.stroke(); } }
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

// screen rects (canvas CSS px) of the HUD overlays, refreshed at most every 250 ms
let hudCache = { t: 0, rects: [] };
function hudRects() {
  const now = performance.now();
  if (now - hudCache.t > 250) {
    const m = $("#map").getBoundingClientRect();
    hudCache = { t: now, rects: $$("#mapWrap .hud, #debugBadge").filter((el) => !el.hidden && el.offsetParent !== null).map((el) => {
      const r = el.getBoundingClientRect(); return { x: r.left - m.left, y: r.top - m.top, w: r.width, h: r.height };
    }) };
  }
  return [...hudCache.rects];
}
// Every place in homewood_map.json gets a label at every zoom: sim name (+ the real building's name when close
// enough) and a head count of the agents inside. Labels are kept on screen for partly visible places and
// nudged to the first free slot so neighbouring buildings do not cover each other.
function drawPlaceLabels(ctx, sx, sy, active, occ, vw, vh) {
  const detail = cam.zoom >= 0.5 ? 2 : cam.zoom >= 0.2 ? 1 : 0;   // 2: name + real name, 1: name, 0: small name
  const fs = detail ? 13 : 11, fs2 = 11, h = detail ? 18 : 15;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const placed = hudRects();   // HUD boxes over the canvas count as taken
  const hits = (r) => placed.some((p) => r.x < p.x + p.w + 2 && r.x + r.w + 2 > p.x && r.y < p.y + p.h + 1 && r.y + r.h + 1 > p.y);
  const entries = Object.entries(M.data.places).sort(([ka, a], [kb, b]) =>
    (!!(occ[kb] || active.has(kb)) - !!(occ[ka] || active.has(ka))) || a.box[1] - b.box[1] || a.box[0] - b.box[0]);
  ctx.textBaseline = "top"; ctx.textAlign = "left";
  for (const [key, pl] of entries) {
    const [x0, y0, x1, y1] = pl.box;
    const bx0 = sx(x0 * TILE), by0 = sy(y0 * TILE), bx1 = sx((x1 + 1) * TILE), by1 = sy((y1 + 1) * TILE);
    if (bx1 < 0 || bx0 > vw || by1 < 0 || by0 > vh) continue;
    const name = pl.name || key, sub = detail === 2 && pl.label && pl.label !== name ? pl.label : "";
    const n = occ[key] || 0, hot = active.has(key);
    ctx.font = `700 ${fs}px ${PIX}`; const w1 = ctx.measureText(name).width;
    ctx.font = `500 ${fs2}px ${PIX}`; const w2 = sub ? ctx.measureText(sub).width + 8 : 0;
    ctx.font = `700 ${fs2}px ${PIX}`; const wn = n ? ctx.measureText(String(n)).width + 8 : 0;
    const w = Math.round(w1 + w2 + (n ? wn + 4 : 0) + (hot ? 12 : 0) + 10);
    const cx = detail ? bx0 + 2 : (bx0 + bx1) / 2 - w / 2, cy = detail ? by0 - h - 2 : (by0 + by1) / 2 - h / 2;
    const cands = [[cx, cy], [bx0 + 2, by0 + 2], [bx0 + 2, by1 + 2], [bx1 - w - 2, by0 - h - 2]];
    for (let k = 1; k <= 4; k++) cands.push([cx, cy + k * (h + 2)], [cx, cy - k * (h + 2)]);
    // a clamped candidate that collides slides past the obstacle (down, or right), staying next to its building
    const slackY = detail ? h + 4 : 3 * h, slackX = detail ? 0 : w / 2;
    const ok = (c) => c.y >= 2 && c.y + h <= vh - 2 && c.x >= 2 && c.x + w <= vw - 2 &&
      c.y <= Math.max(by1, 2) + slackY && c.y + h >= Math.min(by0, vh) - slackY && c.x <= bx1 + slackX && c.x + w >= bx0 - slackX;
    const slide = (c, dir) => {
      for (let i = 0; i < 8; i++) {
        const p = placed.find((q) => c.x < q.x + q.w + 2 && c.x + c.w + 2 > q.x && c.y < q.y + q.h + 1 && c.y + c.h + 1 > q.y);
        if (!p) return c;
        c = dir === "down" ? { ...c, y: Math.round(p.y + p.h + 2) } : { ...c, x: Math.round(p.x + p.w + 3) };
        if (!ok(c)) return null;
      }
      return null;
    };
    const at = ([px_, py_]) => ({ x: Math.round(clamp(px_, 2, vw - w - 2)), y: Math.round(clamp(py_, 2, vh - h - 2)), w, h });
    let r = null;
    for (const cd of cands) { const c = at(cd); if (!hits(c)) { r = c; break; } }
    for (const dir of ["down", "right"]) for (const cd of cands.slice(0, 3)) { if (!r) r = slide(at(cd), dir); }
    r = r || at([cx, cy]);
    placed.push(r);
    ctx.fillStyle = hot ? "rgba(235,104,52,.92)" : pl.kind === "lawn" ? "rgba(47,90,42,.88)" : "rgba(20,20,18,.8)";
    ctx.fillRect(r.x, r.y, w, h);
    let x = r.x + 5;
    ctx.fillStyle = "#fff"; ctx.font = `700 ${fs}px ${PIX}`; ctx.fillText(name, x, r.y + 2); x += w1;
    if (sub) { ctx.fillStyle = "#e8e4d4"; ctx.font = `500 ${fs2}px ${PIX}`; ctx.fillText(sub, x + 8, r.y + 4); x += w2; }
    if (n) { ctx.fillStyle = "#ffd866"; ctx.fillRect(x + 4, r.y + 2, wn, h - 4); ctx.fillStyle = "#1c1c1a"; ctx.font = `700 ${fs2}px ${PIX}`; ctx.fillText(String(n), x + 8, r.y + (detail ? 3 : 2)); x += wn + 4; }
    if (hot) { ctx.fillStyle = "#fff"; ctx.font = `700 ${fs}px ${PIX}`; ctx.fillText("!", x + 4, r.y + 2); }
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
function onMapClick(e) { const a = agentAt(e); if (a) { S.sel = a; renderAgent(a); renderAwayTray(); if (S.hasCoop && S.frames[S.tick]) renderRoster(S.frames[S.tick]); } }
function onMapHover(e) {
  const a = agentAt(e); const f = S.frames[S.tick];
  if (a && f) { const st = f.agents[a]; showTip(e, `<b>${esc(S.manifest.agents[a]?.name || a)}</b> ${agentBadges(st)}<br>${esc(st.activity)}<br><span class="muted">${esc(st.location)} · ${esc(st.arena)}</span>`); }
  else {
    const { x, y } = screenToWorld(e); const cx = Math.floor(x / TILE), cy = Math.floor(y / TILE);
    const inR = (r) => r && cx >= r[0] && cx <= r[2] && cy >= r[1] && cy <= r[3];
    const hit = Object.entries(M.data?.places || {}).find(([, p]) => boxesOf(p).some(inR));
    if (hit) {
      const [key, pl] = hit, ar = Object.entries(pl.arenas || {}).find(([, a]) => inR(a.rect));
      const here = f ? Object.keys(drawState.pos).filter((aid) => f.agents[aid]?.location === key) : [];
      showTip(e, `<b>${esc(pl.name || key)}</b>${pl.label && pl.label !== pl.name ? ` · ${esc(pl.label)}` : ""}${ar ? `<br><span class="muted">${esc(ar[0])}</span>` : ""}` +
        `<br><span class="muted">${Object.keys(pl.arenas || {}).map(esc).join(" · ")}</span>` +
        (here.length ? `<br>here now: ${here.map((aid) => esc(firstName(aid))).join(", ")}` : ""));
    } else hideTip();
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
    <span class="meta">→ ${(u.listeners || []).map(name).map(esc).join(", ") || "nobody"} · ${esc(S.frames[t].label)}</span><div>${esc(u.text)}</div></div>`).join("") || `<div class="muted small">No utterances yet.</div>`;
  $$("#feed .utt").forEach((d) => d.onclick = () => openChain(d.dataset.uid));
}
const name = (aid) => S.manifest?.agents?.[aid]?.name?.split(" ")[0] || aid;

// ------------------------------------------------------------ co-op panels (v3)
// Everything here is world-side, public co-op state from frame.coop and GET /coop (demo mode); the job truth
// (fault class, cause, regime) appears only in Research Debug Mode, where GET /coop?debug=1 carries it.
const jobsCache = new Map();
const FINAL = new Set(["delivered", "defer", "failed"]);
function dayJobs(t) {
  // every job of the tick's day seen up to t, in its latest state (a finished job appears once in the frames)
  if (jobsCache.has(t)) return jobsCache.get(t);
  const f = S.frames[t]; if (!f) return [];
  let s = t; while (s > 0 && S.frames[s - 1]?.day === f.day) s--;
  const m = new Map();
  for (let k = s; k <= t; k++) for (const j of S.frames[k].coop?.jobs || []) m.set(j.id, { ...j, seen: k });
  const out = [...m.values()].sort((a, b) => (a.slot ?? 0) - (b.slot ?? 0) || String(a.id).localeCompare(String(b.id)));
  jobsCache.set(t, out);
  return out;
}
const coopJob = (id) => (S.coop?.jobs || []).find((j) => j.job === id);
const statusPill = (s) => `<span class="status-pill s-${esc(s)}">${esc(s)}</span>`;
function truthHtml(tr) {
  // debug only: GET /coop?debug=1 adds each job's hidden truth
  if (!S.debug || !tr) return "";
  const p = tr.p ? Object.entries(tr.p).map(([k, v]) => `${esc(k)} ${v}`).join(", ") : "";
  return `<div class="small truth"><span class="tag gt">truth</span> class ${esc(tr.class)} · cause ${esc(tr.cause ?? "—")} · fault ${esc(tr.fault)} · fix ${esc(tr.gt ?? "—")} · regime ${esc(tr.regime)}/${esc(tr.mapping)}${p ? `<br><span class="muted">p(success): ${p}</span>` : ""}</div>`;
}

function renderCoop() {
  if (!S.hasCoop) return;
  const f = S.frames[S.tick]; if (!f) return;
  renderJobs(f); renderBinder(f); renderRoster(f);
}

function renderJobs(f) {
  const t = f.tick ?? S.tick, jobs = dayJobs(S.tick);
  const done = jobs.filter((j) => FINAL.has(j.status)), delivered = done.filter((j) => j.status === "delivered").length;
  const tally = (S.coop?.tallies || []).filter((r) => r.day === f.day && r.tick <= t).pop();
  const regime = S.debug ? (S.coop?.regimes || []).filter((r) => r.tick <= t).pop() : null;
  $("#jobsHead").innerHTML = `Day ${esc(f.day)} · ${delivered} delivered · ${done.length - delivered} not · ${jobs.length - done.length} open${regime ? ` <span class="tag gt">regime ${esc(regime.regime)} · ${esc(regime.mapping)}</span>` : ""}`;
  const now = new Set((f.coop?.jobs || []).map((j) => j.id));
  const card = (j) => {
    const full = coopJob(j.id), upto = (r) => (r.tick ?? 0) <= t;
    const dec = (full?.decisions || []).filter(upto).pop(), asks = (full?.clarifications || []).filter(upto);
    const tried = (j.tried || []).map((x) => `<span class="try o-${esc(x.outcome)}">${esc(x.attempt)}. ${esc(x.action)} → ${esc(x.outcome)}</span>`);
    if (j.status === "running" && (j.attempt || 0) > (j.tried || []).length) {
      const pend = (full?.decisions || []).filter((x) => upto(x) && x.attempt === j.attempt).pop();
      tried.push(`<span class="try o-pending">${esc(j.attempt)}. ${esc(pend?.action || "…")} → running</span>`);
    }
    const decLine = !FINAL.has(j.status) && dec ? `<div class="small job-dec">${dec.choice === "G" || dec.question ? `asks ${esc(name(dec.ask_target))}: “${esc(dec.question || "")}”` : `chose <b>${esc(dec.action)}</b>${dec.reason ? `: ${esc(dec.reason)}` : ""}`}${dec.says_aloud ? ` <span class="muted">(says: “${esc(dec.says_aloud)}”)</span>` : ""}${dec.binder_chosen ? ` <span class="badge founder">from binder</span>` : ""}</div>` : "";
    const askLine = asks.length && !FINAL.has(j.status) ? `<div class="small muted">${asks.map((q) => `${esc(name(q.agent))} asked ${esc(name(q.target))}: “${esc(q.question)}”`).join("<br>")}</div>` : "";
    const opSt = f.agents[j.operator];
    return `<div class="job-card s-${esc(j.status)}${now.has(j.id) ? " now" : ""}" data-job="${esc(j.id)}" title="click for the job's full record">
      <div class="job-top">${statusPill(j.status)} ${headIcon(j.operator)}<b>${esc(firstName(j.operator))}</b> ${roleBadge(opSt?.role)}${cohortBadge(opSt?.cohort)}
        <span class="muted small">${esc(j.id)} · ${esc(String(j.shift || "").toUpperCase())} · attempt ${esc(j.attempt ?? 0)}</span></div>
      <div class="job-proj">${esc(j.project)}</div>
      ${j.symptom ? `<div class="job-sym">“${esc(j.symptom)}”</div>` : ""}
      ${tried.length ? `<div class="job-tries">${tried.join("")}</div>` : ""}
      ${decLine}${askLine}${truthHtml(full?.truth)}
    </div>`;
  };
  const open = jobs.filter((j) => !FINAL.has(j.status)), fin = jobs.filter((j) => FINAL.has(j.status)).reverse();
  $("#jobsPanel").innerHTML = (tally ? `<div class="tally">${esc(tally.text)}</div>` : "") +
    (open.length ? open.map(card).join("") : `<div class="muted small">No job at the laser right now.</div>`) +
    (fin.length ? `<div class="sub-h">Finished today (${fin.length})</div>${fin.map(card).join("")}` : "");
  $$("#jobsPanel [data-job]").forEach((el) => el.onclick = () => openJob(el.dataset.job));
}

function openJob(id) {
  const j = coopJob(id); if (!j) return;
  const t = S.tick, upto = (r) => (r.tick ?? 0) <= t;
  const items = [
    { tick: j.start_tick, html: `<b>started</b> · ${esc(j.project)}${j.symptom ? `<br><i>“${esc(j.symptom)}”</i>` : ""}` },
    ...(j.decisions || []).map((d) => ({ tick: d.tick, html: `<b>decision, attempt ${esc(d.attempt)}</b>: ${d.question ? `ask ${esc(name(d.ask_target))}: “${esc(d.question)}”` : `${esc(d.choice || "")} ${esc(d.action)}`}${d.reason ? ` — ${esc(d.reason)}` : ""}${d.says_aloud ? `<br><span class="muted">says aloud: “${esc(d.says_aloud)}”</span>` : ""}<br><span class="muted small">binder ${d.binder_shown ? "shown" : "not shown"}${d.binder_chosen ? ", chosen from it" : ""}${d.valid === false ? " · invalid reply" : ""}</span>` })),
    ...(j.clarifications || []).map((q) => ({ tick: q.tick, html: `<b>clarification</b>: ${esc(name(q.agent))} asked ${esc(name(q.target))}: “${esc(q.question)}”${q.conversation_id ? ` <a href="#" data-conv="${esc(q.conversation_id)}">conversation</a>` : ""}` })),
    ...(j.attempts || []).map((a) => ({ tick: a.tick, html: `<b>attempt ${esc(a.attempt)}</b>: ${esc(a.action)} → ${statusPill(a.outcome)}${a.text ? `<br>${esc(a.text)}` : ""}` })),
    ...(j.end ? [{ tick: j.end.tick, html: `<b>ended</b>: ${statusPill(j.end.result)}${j.writes?.length ? ` · binder writes ${j.writes.map(esc).join(", ")}` : ""}` }] : []),
  ].sort((a, b) => (a.tick ?? 0) - (b.tick ?? 0));
  const shown = items.filter(upto), later = items.length - shown.length;
  $("#modalContent").innerHTML = `<h2>Job ${esc(j.job)} <span class="muted small">${esc(firstName(j.operator))} · Day ${esc(j.day)} · ${esc(String(j.shift || "").toUpperCase())} shift</span></h2>
    ${truthHtml(j.truth)}
    <div class="job-log">${shown.map((x) => `<div class="node"><span class="kind">${esc(labelAt(x.tick))}</span><div>${x.html}</div></div>`).join("") || '<div class="muted">Not started yet at this point of the replay.</div>'}</div>
    ${later ? `<div class="muted small">${later} later record(s) hidden until the replay reaches them.</div>` : ""}`;
  $("#modal").hidden = false;
  $$("#modalContent [data-conv]").forEach((el) => el.onclick = (e) => { e.preventDefault(); $("#modal").hidden = true; showView("trace"); $("#trConv").value = el.dataset.conv; loadTrace(); });
}

function renderBinder(f) {
  const b = f.coop?.binder, t = f.tick ?? S.tick;
  if (!b) { $("#binderHead").textContent = ""; $("#binderPanel").innerHTML = `<div class="muted small">Records are off in this run: the co-op keeps no binder.</div>`; return; }
  $("#binderHead").textContent = `${b.binder_id || ""}${b.archived ? ` · ${b.archived} archived` : ""}`;
  const tl = (S.coop?.binder?.timeline || []).filter((e) => (e.tick ?? 0) <= t);
  const reads = tl.filter((e) => e.kind === "read").length, declined = tl.filter((e) => e.kind === "write" && e.choice === "none").length;
  const hist = tl.filter((e) => e.kind === "transition" || (e.kind === "write" && e.choice !== "none")).slice(-10).reverse();
  const histRow = (e) => e.kind === "transition"
    ? `<div class="hist" data-tick="${e.tick}"><span class="muted">${esc(labelAt(e.tick))}</span> <b>${e.mode === "wipe" ? "binder replaced" : "binder kept"}</b>${e.mode === "wipe" ? `: ${esc(e.archived_binder_id || "old binder")} archived, ${esc(e.binder_id)} started` : ""}</div>`
    : `<div class="hist" data-tick="${e.tick}"><span class="muted">${esc(labelAt(e.tick))}</span> <b>${esc(e.author_name || name(e.agent))}</b> ${e.choice === "front" ? "rewrote the front page" : "added a log entry"}${e.offer && e.offer !== "job" ? ` <span class="muted">(${esc(e.offer)})</span>` : ""}: ${esc(e.text || "")}</div>`;
  $("#binderPanel").innerHTML = `
    <div class="binder-page front"><div class="bp-h">Front page${b.front?.revisions ? ` <span class="muted">· ${b.front.revisions} revision${b.front.revisions > 1 ? "s" : ""}</span>` : ""}</div>
      ${b.front ? `<div>${esc(b.front.text)}</div><div class="meta">${esc(b.front.author)} · ${esc(b.front.when)}</div>` : `<div class="muted small">blank</div>`}</div>
    <div class="binder-page"><div class="bp-h">Log <span class="muted">· newest first</span></div>
      ${(b.log || []).map((e) => `<div class="log-e"><div>${esc(e.text)}</div><div class="meta">${esc(e.author)} · ${esc(e.when)}</div></div>`).join("") || `<div class="muted small">no entries</div>`}</div>
    <div class="sub-h">Binder over time <span class="muted">· ${reads} read${reads === 1 ? "" : "s"}, ${declined} write offer${declined === 1 ? "" : "s"} declined</span></div>
    ${hist.map(histRow).join("") || `<div class="muted small">nothing written yet</div>`}`;
  $$("#binderPanel [data-tick]").forEach((el) => el.onclick = () => setTick(+el.dataset.tick));
}

const ROLE_ORDER = { am_crew: 0, pm_crew: 1, stores: 2 };
function renderRoster(f) {
  const t = f.tick ?? S.tick, ids = Object.keys(f.agents);
  ids.sort((a, b) => (f.agents[a].active === false) - (f.agents[b].active === false) || (ROLE_ORDER[f.agents[a].role] ?? 9) - (ROLE_ORDER[f.agents[b].role] ?? 9) || firstName(a).localeCompare(firstName(b)));
  const nAway = f.away.length;
  $("#rosterHead").textContent = `${ids.length - nAway} here${nAway ? ` · ${nAway} away` : ""}`;
  const C = S.coop || {}, upto = (r) => (r.tick ?? 0) <= t;
  const ev = [
    ...(C.roster_changes || []).filter(upto).map((r) => ({ tick: r.tick, text: `${esc(firstName(r.agent))} ${r.kind === "arrive" ? "joined" : "left"} (${esc(roleName(r.role))}${r.replaces ? `, replacing ${esc(firstName(r.replaces))}` : ""})` })),
    ...(C.farewells || []).filter(upto).map((r) => ({ tick: r.tick, text: `farewell: ${esc(r.text)}` })),
    ...(C.handovers || []).filter(upto).map((r) => ({ tick: r.tick, text: `handover ${esc(firstName(r.outgoing))} → ${esc(firstName(r.incoming))}` })),
    ...(C.meetings || []).filter(upto).map((r) => ({ tick: r.tick, text: `co-op meeting (${(r.invited || []).length} invited)` })),
    ...(C.cues || []).filter(upto).map((r) => ({ tick: r.tick, text: `${esc(r.arena || "")}: ${esc(r.text)}` })),
    ...(C.tallies || []).filter(upto).map((r) => ({ tick: r.tick, text: `tally: ${esc(r.delivered)} of ${esc(r.total)} delivered` })),
  ].sort((a, b) => b.tick - a.tick).slice(0, 10);
  $("#rosterPanel").innerHTML = ids.map((aid) => {
    const a = f.agents[aid], away = a.active === false;
    return `<div class="member${S.sel === aid ? " sel" : ""}${away ? " is-away" : ""}" data-aid="${esc(aid)}">${headIcon(aid)}
      <div class="m-main"><div><b>${esc(firstName(aid))}</b> ${roleBadge(a.role)}${cohortBadge(a.cohort)}${away ? `<span class="badge away">away</span>` : ""}</div>
      <div class="muted small">${away ? "off the roster" : `${esc(a.location)} · ${esc(a.arena)}`}</div></div></div>`;
  }).join("") + (ev.length ? `<div class="sub-h">Co-op events</div>${ev.map((e) => `<div class="hist" data-tick="${e.tick}"><span class="muted">${esc(labelAt(e.tick))}</span> ${e.text}</div>`).join("")}` : "");
  $$("#rosterPanel .member").forEach((el) => el.onclick = () => { S.sel = el.dataset.aid; renderAgent(S.sel); renderAwayTray(); renderRoster(S.frames[S.tick]); });
  $$("#rosterPanel .hist[data-tick]").forEach((el) => el.onclick = () => setTick(+el.dataset.tick));
}

// per-agent NEED / WORDING / forgetting records, fetched once per run and agent (filtered by tick locally)
function agentTraceRecs(aid) {
  if (!S.agentTrace[aid]) S.agentTrace[aid] = api(`/runs/${S.runId}/trace?agent=${encodeURIComponent(aid)}&type=open_matter,wording,memory_forgotten&limit=200000`).catch(() => []);
  return S.agentTrace[aid];
}
const cfgOn = (k, sub) => { const c = S.manifest?.config?.[k]; return !!(sub ? c?.[sub]?.enabled : c?.enabled); };
// Open matters still on the agent's mind at tick t (memory/need.py): strength of the last open/refresh (1.0),
// multiplied at each 'discussed', decayed with the half-life since it was opened; faded below 0.1.
function openMattersAt(recs, aid, t) {
  const now = Date.parse(S.frames[t]?.time), hl = +(S.manifest.config?.need?.half_life_hours ?? 24) || 24;
  const gone = new Set(recs.filter((r) => r.type === "memory_forgotten" && r.agent === aid && r.tick <= t).map((r) => r.node_id));
  const m = new Map();
  for (const r of recs) {
    if (r.type !== "open_matter" || r.agent !== aid || r.tick > t) continue;
    const cur = m.get(r.node_id);
    if (!cur || r.action === "open" || r.action === "refresh") m.set(r.node_id, { node_id: r.node_id, text: r.text, created: r.time, opened: r.tick, base: r.strength ?? 1, discussed: 0, reason: r.reason });
    else { cur.base = r.strength ?? cur.base; cur.discussed++; }
  }
  const out = [];
  for (const x of m.values()) {
    if (gone.has(x.node_id)) continue;
    const hours = Math.max(0, (now - Date.parse(x.created)) / 3.6e6);
    x.now = Number.isFinite(hours) ? x.base * 0.5 ** (hours / hl) : x.base;
    if (x.now >= 0.1) out.push(x);
  }
  return out.sort((a, b) => b.now - a.now || b.opened - a.opened);
}
function wordingsAt(recs, aid, t) {
  const gone = new Set(recs.filter((r) => r.type === "memory_forgotten" && r.agent === aid && r.tick <= t).map((r) => r.node_id));
  return recs.filter((r) => r.type === "wording" && r.agent === aid && r.tick <= t && !r.self_produced && !gone.has(r.node_id)).reverse();
}
const hhmm = (iso) => String(iso || "").slice(11, 16);
const labelAt = (t) => S.frames[t]?.label || `tick ${t}`;

function coopAgentHtml(aid, st, t) {
  if (!S.hasCoop || !S.coop) return "";
  const C = S.coop, upto = (r) => (r.tick ?? 0) <= t;
  const changes = (C.roster_changes || []).filter((r) => r.agent === aid && upto(r));
  const replacedBy = (C.roster_changes || []).filter((r) => r.replaces === aid && upto(r));
  const jobs = (C.jobs || []).filter((j) => j.operator === aid && (j.start_tick ?? 0) <= t);
  const writes = (C.binder?.timeline || []).filter((e) => e.kind === "write" && e.agent === aid && upto(e) && e.choice !== "none");
  const fw = (C.farewells || []).find((r) => r.agent === aid && upto(r));
  const onb = (C.onboarding || []).find((r) => r.agent === aid && upto(r));
  const hist = [
    ...changes.map((r) => `${esc(labelAt(r.tick))}: ${r.kind === "arrive" ? "joined" : "left"} as ${esc(roleName(r.role))}${r.replaces ? ` (replacing ${esc(firstName(r.replaces))})` : ""}`),
    ...replacedBy.map((r) => `${esc(labelAt(r.tick))}: replaced by ${esc(firstName(r.agent))}`),
    ...(fw ? [`${esc(labelAt(fw.tick))}: farewell: “${esc(fw.text)}”`] : []),
    ...(onb ? [`${esc(labelAt(onb.tick))}: orientation; ties to ${onb.ties.map((x) => esc(firstName(x.with))).join(", ")}`] : []),
  ];
  const jobRow = (j) => {
    const done = j.end && j.end.tick <= t, last = [...(j.attempts || [])].filter(upto).pop();
    return `<div class="mem job"><b>${esc(j.job)}</b> ${esc(j.project)}<div class="meta">${esc(labelAt(j.start_tick))} · ${done ? `<span class="status-pill s-${esc(j.end.result)}">${esc(j.end.result)}</span>` : `<span class="status-pill s-running">in progress</span>`}${last ? ` · last try: ${esc(last.action)} → ${esc(last.outcome)}` : ""}</div></div>`;
  };
  return `<details open><summary>Co-op <span class="muted small">${esc(roleName(st.role) || "no role")}${st.cohort ? " · " + esc(st.cohort) : ""}</span></summary>
    ${hist.length ? `<div class="small">${hist.join("<br>")}</div>` : ""}
    ${jobs.length ? `<div class="small muted" style="margin-top:4px">Jobs operated (${jobs.length})</div>${jobs.slice(-6).reverse().map(jobRow).join("")}` : ""}
    ${writes.length ? `<div class="small muted" style="margin-top:4px">Binder writes (${writes.length})</div>${writes.slice(-4).reverse().map((e) => `<div class="mem binder"><div>${esc(e.text || "")}</div><div class="meta">${esc(labelAt(e.tick))} · ${e.choice === "front" ? "rewrote the front page" : "log entry"}</div></div>`).join("")}` : ""}
    ${!hist.length && !jobs.length && !writes.length ? `<div class="muted small">No co-op activity yet.</div>` : ""}</details>`;
}

async function renderAgent(aid) {
  const req = ++S.agentReq, t = S.tick;
  let d, recs;
  try { [d, recs] = await Promise.all([api(`/runs/${S.runId}/agent/${encodeURIComponent(aid)}?tick=${t}`), agentTraceRecs(aid)]); }
  catch (e) { if (req === S.agentReq) $("#agentPanel").innerHTML = `<div class="muted">Could not load ${esc(aid)}: ${esc(e.message)}</div>`; return; }
  if (req !== S.agentReq) return;
  const p = d.profile || {}, st = { ...AGENT_DEFAULTS, ...(S.frames[t]?.agents?.[aid] || {}), ...(d.state || {}) };
  const demo = p.demographics || {}, pers = p.personality || {}, intr = p.interests || {};
  const rel = Object.entries(p.relationships || {}).filter(([, r]) => r.relation_type !== "stranger")
    .map(([o, r]) => `${esc(name(o))} <span class="muted">(${esc(r.relation_type)}, fam ${r.familiarity}, aff ${r.affinity})</span>`).join("<br>");
  const mem = (m) => `<div class="mem ${m.kind}"><div>${esc(m.text)}</div><div class="meta">${esc(m.time?.slice(11, 16))} · ${m.kind} · ${m.source_type} · importance ${m.importance}${m.score ? ` · score ${m.score.s} (rel ${m.score.rel}, rec ${m.score.rec}, imp ${m.score.imp})` : ""}${S.debug && m.originating_event_ids?.length ? ` <span class="tag gt">events ${m.originating_event_ids.join(",")}</span>` : ""}</div></div>`;
  const mods = st.modules && Object.keys(st.modules).length ? `<dt>Modules</dt><dd>${esc(JSON.stringify(st.modules))}</dd>` : "";
  const place = M.data?.places?.[st.location];
  const away = st.active === false || st.location === "Away";
  // NEED, WORDING, reminding links (v2 mechanisms; hidden rows when the mechanism is off and nothing was logged)
  const matters = openMattersAt(recs, aid, t), words = wordingsAt(recs, aid, t);
  const reminds = (d.remindings || []).filter((m) => m.tick <= t);
  const byNode = {}; for (const m of [...(d.memories || []), ...(d.reflections || [])]) byNode[m.node_id] = m;
  const memText = (nid) => byNode[nid] ? esc(byNode[nid].text) : `<span class="muted">${esc(nid)} (older memory)</span>`;
  const showNeed = cfgOn("need") || st.open_matters > 0 || matters.length, showWord = cfgOn("memory", "verbatim") || st.wordings > 0 || words.length;
  const showRem = cfgOn("reminding") || reminds.length;
  const needHtml = showNeed ? `<details ${matters.length ? "open" : ""}><summary>Open matters (${st.open_matters})</summary>${matters.slice(0, 8).map((m) => `<div class="mem matter"><div>${esc(m.text)}</div>
      <div class="meta"><span class="meter" title="strength ${m.now.toFixed(2)}"><i style="width:${Math.round(Math.min(1, m.now) * 100)}%"></i></span> strength ${m.now.toFixed(2)} · opened ${esc(labelAt(m.opened))}${m.reason ? ` (${esc(m.reason.replace(/_/g, " "))})` : ""}${m.discussed ? ` · talked about ${m.discussed}×` : ""}</div></div>`).join("") || '<div class="muted small">nothing on their mind</div>'}</details>` : "";
  const wordHtml = showWord ? `<details ${words.length ? "open" : ""}><summary>Stuck wordings (${st.wordings})</summary>${words.slice(0, 10).map((w) => `<div class="mem wording"><div>“${esc(w.phrase)}”</div>
      <div class="meta">heard from ${esc(name(w.heard_from))} · ${esc(labelAt(w.tick))} · ${esc(w.source_type)}${w.utterance_id ? ` · <a href="#" data-chain="${esc(w.utterance_id)}">trace</a>` : ""}</div></div>`).join("") || '<div class="muted small">none held</div>'}</details>` : "";
  const remHtml = showRem ? `<details><summary>Reminding links (${reminds.length})</summary>${reminds.slice(0, 8).map((m) => `<div class="mem link">
      <div class="small"><b>felt alike:</b> ${esc(m.what_felt_alike || "—")}</div>
      <div class="small">new: ${memText(m.evidence?.[0])}</div><div class="small">earlier: ${memText(m.evidence?.[1])}</div>
      <div class="meta">${esc(labelAt(m.tick))} · importance ${m.importance ?? "—"}</div></div>`).join("") || '<div class="muted small">none yet</div>'}</details>` : "";
  const personalLabels = { goal: "Personal goal", values: "Values", strengths: "Strengths", blind_spots: "Blind spots",
    stress_response: "Response to stress", coping_strategy: "Coping strategy", social_energy: "Social energy",
    trust_style: "How trust develops", conflict_style: "Approach to conflict", humor_style: "Sense of humor",
    pet_peeves: "Pet peeves", small_joys: "Small joys" };
  const personal = Object.entries(p.personal || {}).map(([key, value]) =>
    `<dt>${esc(personalLabels[key] || key.replaceAll("_", " "))}</dt><dd>${esc(Array.isArray(value) ? value.join(", ") : value)}</dd>`).join("");
  const friends = (p.friend_groups || []).map((g) =>
    `<div class="small"><b>${esc(g.name)}</b><br>${g.members.map(esc).join(", ")}</div>`).join("");
  $("#agentPanel").innerHTML = `
    <div class="agent-head"><div class="avatar" style="background-image:url(/ga_assets/characters/${encodeURIComponent(p.sprite || "")}.png)"></div>
      <div><div style="font-weight:700">${esc(p.name || aid)} ${agentBadges(st)}</div><div class="muted small">${[demo.year, demo.major, demo.role].filter(Boolean).map(esc).join(" · ")}</div>
      <button class="pill" id="btnLocate" ${away ? "disabled title='not on the map'" : ""}>locate on map</button></div></div>
    <dl class="kv">
      <dt>Now</dt><dd>${away ? `<span class="badge away">away</span> <span class="muted">off campus / not on the co-op roster</span>` : `${esc(st.activity)} <span class="muted">@ ${esc(st.location)}${place && place.label && place.label !== st.location ? ` (${esc(place.label)})` : ""} / ${esc(st.arena)}</span>`}</dd>
      ${st.role || st.cohort ? `<dt>Co-op role</dt><dd>${esc(roleName(st.role) || "—")}${st.cohort ? ` · ${esc(st.cohort)}` : ""}</dd>` : ""}
      <dt>Goal</dt><dd>${esc(st.goal || "follow the routine")}</dd>
      ${showNeed ? `<dt>Open matters</dt><dd>${st.open_matters}</dd>` : ""}
      ${showWord ? `<dt>Stuck wordings</dt><dd>${st.wordings}</dd>` : ""}
      ${showRem ? `<dt>Reminding links</dt><dd>${reminds.length}</dd>` : ""}
      <dt>Personality</dt><dd>${esc((pers.traits || []).join(", "))}${pers.communication_style ? `; ${esc(pers.communication_style)}` : ""}</dd>
      <dt>Interests</dt><dd>${esc([...(intr.topics || []), ...(intr.hobbies || [])].join(", ") || "—")}</dd>
      <dt>Clubs</dt><dd>${esc((intr.clubs || []).join(", ") || "—")}</dd>
      <dt>Background</dt><dd>${esc(p.background)}</dd>
      <dt>Memories</dt><dd>${d.n_memories} in stream</dd>${mods}
    </dl>
    ${coopAgentHtml(aid, st, t)}${needHtml}${wordHtml}${remHtml}
    ${personal ? `<details open><summary>Personal traits and motivations</summary><dl class="kv">${personal}</dl></details>` : ""}
    ${friends ? `<details open><summary>Friend groups</summary>${friends}</details>` : ""}
    <details><summary>Relationships</summary><div class="small">${rel}</div></details>
    <details><summary>Routine</summary><div class="small">${(p.routine || []).map((r) => `${esc(r.time)} ${esc(r.activity)} <span class="muted">@ ${esc(r.location)}</span>`).join("<br>")}</div></details>
    <details open><summary>Last retrieved memories ${d.retrieved?.tick != null ? `<span class="muted small">(tick ${d.retrieved.tick})</span>` : ""}</summary>${(d.retrieved?.memories || []).map(mem).join("") || '<div class="muted small">none yet</div>'}</details>
    <details open><summary>Reflections (${(d.reflections || []).length})</summary>${(d.reflections || []).map(mem).join("") || '<div class="muted small">none yet</div>'}</details>
    <details><summary>Recent memories</summary>${(d.memories || []).slice(0, 25).map(mem).join("")}</details>
    <details><summary>Recent conversations (${(d.conversations || []).length})</summary>${(d.conversations || []).map((c) => `<div class="mem chat">${(c.transcript || []).map(([s, tx]) => `<b>${esc(String(s).split(" ")[0])}:</b> ${esc(tx)}`).join("<br>")}<div class="meta">${esc(c.time?.slice(11, 16))} @ ${esc(c.location)}</div></div>`).join("")}</details>`;
  $("#btnLocate").onclick = () => { const a = drawState.pos[aid]; if (a) { if (cam.zoom < 0.75) cam.zoom = 1; centerOn(a.x, a.y); } cam.follow = true; $("#followSel").checked = true; };
  $$("#agentPanel [data-chain]").forEach((el) => el.onclick = (e) => { e.preventDefault(); openChain(el.dataset.chain); });
}

// ---------------------------------------------------------------- culture
// trends dashboard (culture.js); debounced inside, follows the replay tick
function drawTrends() { if (S.runId) renderCultureTrends($("#cultureTrends"), { runId: S.runId, tick: S.tick, api }); }
const STATUS = { established: "●", spreading: "▲", emerging: "○", fading: "▽" };
function orderedCands() {
  const cs = [...(S.analysis?.candidates || [])];
  if ($("#onlyConv").checked) cs.sort((a, b) => (!!b.llm?.is_convention - !!a.llm?.is_convention) || b.score - a.score);
  return cs;
}
function renderCulture() {
  const A = S.analysis;
  if (!A) { $("#cultureSummary").innerHTML = `<span class="muted">This run has not been analyzed yet. Use “Re-run analysis” or <code>python -m backend.cli analyze runs/${esc(S.runId)}</code>.</span>`; $("#cards").innerHTML = ""; $("#memeDetail").innerHTML = ""; return; }
  if (A.mode === "commons") { renderCommonsResearch(A); return; }
  if (A.kind === "memetics") { renderMemeticsCulture(A, S.runId, tick => { showView("campus"); setTick(Math.max(0,tick)); }); return; }
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

// ------------------------------------------------------------------ trace explorer
// Filters go to GET /runs/<id>/trace (type, agent, tick_from/tick_to, conversation, limit); the type list comes
// from GET /schema, which lists hidden (observer-only) types only in Research Debug Mode.
const trState = { types: new Set(), records: null, limit: 500 };
const TR_PRESETS = {
  talk: ["conversation", "utterance", "exposure", "invitation", "decision", "clarification", "handover", "meeting"],
  memory: ["observation", "viewpoint", "memory_encoded", "memory_forgotten", "memory_merged", "memory_link", "reflection", "reminding", "wording", "open_matter", "open_matter_focal", "priming"],
};
const AGENT_KEYS = ["agent", "speaker", "speaker_id", "operator", "stores", "outgoing"];
const TARGET_KEYS = ["target", "ask_target", "incoming", "heard_from"];
const LIST_KEYS = ["listeners", "listener_ids", "participants", "invited", "movers"];

function bindTraceExplorer() {
  $("#trLoad").onclick = loadTrace;
  $("#trNow").onclick = () => { $("#trTo").value = S.tick; loadTrace(); };
  for (const id of ["#trFrom", "#trTo", "#trConv", "#trLimit"]) $(id).onkeydown = (e) => { if (e.key === "Enter") loadTrace(); };
  $("#trAgent").onchange = loadTrace;
  $$(".trace-presets [data-preset]").forEach((b) => b.onclick = () => {
    const types = Object.keys(S.schema?.types || {}), p = b.dataset.preset;
    trState.types = new Set(p === "coop" ? types.filter((k) => S.schema.types[k].since === "v3")
      : p === "world" ? types.filter((k) => S.schema.types[k].layer === "world")
      : (TR_PRESETS[p] || []).filter((k) => types.includes(k)));
    renderTypeFilter();
  });
}
function resetTraceExplorer() {
  const ids = Object.keys(S.order).sort((a, b) => firstName(a).localeCompare(firstName(b)));
  const keep = $("#trAgent").value;
  $("#trAgent").innerHTML = `<option value="">any</option>` + ids.map((aid) => `<option value="${esc(aid)}">${esc(S.manifest.agents[aid]?.name || aid)}</option>`).join("");
  if (ids.includes(keep)) $("#trAgent").value = keep;
  $("#trFrom").placeholder = "0"; $("#trTo").placeholder = String(Math.max(0, S.frames.length - 1));
  // drop selected types this run's schema does not know
  if (S.schema?.types) trState.types = new Set([...trState.types].filter((k) => S.schema.types[k]));
  trState.records = null;
  $("#trResults").innerHTML = ""; $("#trInfo").textContent = "Pick filters and load records.";
  renderTypeFilter();
}
function renderTypeFilter() {
  const box = $("#trTypes"), sc = S.schema;
  if (!sc?.types) { box.innerHTML = `<div class="muted small">Trace schema unavailable: all record types are returned.</div>`; return; }
  const layers = sc.layers?.length ? sc.layers : [...new Set(Object.values(sc.types).map((v) => v.layer))];
  box.innerHTML = layers.map((L) => {
    const ts = Object.entries(sc.types).filter(([, v]) => v.layer === L).sort(([a], [b]) => a.localeCompare(b));
    if (!ts.length) return "";
    return `<div class="type-col"><div class="type-layer l-${esc(L)}">${esc(L)}</div>${ts.map(([k, v]) => {
      const tip = `${v.description || ""}\n${v.section ? "§ " + v.section : ""}\nrequired: ${(v.required || []).join(", ")}${v.optional?.length ? "\noptional: " + v.optional.join(", ") : ""}${v.hidden ? "\nHIDDEN (observer only)" : ""}${v.hidden_fields?.length ? "\nhidden fields: " + v.hidden_fields.join(", ") : ""}`;
      return `<label class="tchk" title="${esc(tip)}"><input type="checkbox" value="${esc(k)}" ${trState.types.has(k) ? "checked" : ""}> ${esc(k)}<sup>${esc(v.since || "")}</sup>${v.hidden ? ` <span class="tag gt">hidden</span>` : ""}</label>`;
    }).join("")}</div>`;
  }).join("");
  $$("#trTypes input").forEach((c) => c.onchange = () => { c.checked ? trState.types.add(c.value) : trState.types.delete(c.value); countTypes(); });
  countTypes();
}
function countTypes() { $("#trTypeCount").textContent = trState.types.size ? `${trState.types.size} selected` : "none selected = all types"; }

async function loadTrace() {
  if (!S.runId) return;
  const q = new URLSearchParams();
  if (trState.types.size) q.set("type", [...trState.types].join(","));
  const ag = $("#trAgent").value; if (ag) q.set("agent", ag);
  const from = $("#trFrom").value, to = $("#trTo").value;
  if (from !== "") q.set("tick_from", Math.max(0, Math.floor(+from)));
  if (to !== "") q.set("tick_to", Math.max(0, Math.floor(+to)));
  const conv = $("#trConv").value.trim(); if (conv) q.set("conversation", conv);
  trState.limit = Math.max(1, Math.floor(+$("#trLimit").value || 500)); q.set("limit", trState.limit);
  $("#trInfo").textContent = "loading…";
  const req = (trState.req = (trState.req || 0) + 1);
  let recs;
  try { recs = await api(`/runs/${S.runId}/trace?${q}`); } catch (e) { $("#trInfo").textContent = `Could not load records: ${e.message}`; return; }
  if (req !== trState.req) return;
  trState.records = Array.isArray(recs) ? recs : [];
  renderTraceTable();
}
const clip = (s, n = 240) => { s = String(s ?? ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; };
const factText = (fs) => (fs || []).map((f) => (typeof f === "string" ? f : f?.text || "")).filter(Boolean).join(" · ");
function recWho(r) {
  const who = AGENT_KEYS.map((k) => r[k]).find((v) => typeof v === "string");
  const tgt = TARGET_KEYS.map((k) => r[k]).find((v) => typeof v === "string" && v !== who);
  const many = LIST_KEYS.map((k) => r[k]).find((v) => Array.isArray(v) && v.length);
  const parts = [];
  if (who) parts.push(`<b>${esc(name(who))}</b>`);
  if (tgt) parts.push(`→ ${esc(name(tgt))}`);
  else if (many) parts.push(`${who ? "→ " : ""}${many.slice(0, 5).map((a) => esc(name(a))).join(", ")}${many.length > 5 ? ` +${many.length - 5}` : ""}`);
  return parts.join(" ");
}
function recSummary(r) {
  switch (r.type) {
    case "utterance": return `“${esc(clip(r.text))}”`;
    case "move": return `${esc(r.frm ?? "")} → ${esc(r.to ?? "")}${r.activity ? ` · ${esc(r.activity)}` : ""}`;
    case "conversation": return `@ ${esc(r.location)} / ${esc(r.arena)} · ${(r.utterance_ids || []).length} lines${r.trigger?.topic ? ` · ${esc(r.trigger.topic)}` : ""}`;
    case "observation": case "coop_fact": case "viewpoint": return `${r.source_type || r.kind ? `<span class="muted">${esc(r.source_type || r.kind)}</span> ` : ""}${esc(clip(factText(r.facts)))}`;
    case "job_start": return `${esc(r.job)} · ${esc(r.project)}`;
    case "job_decision": return `${esc(r.job)} #${esc(r.attempt)}: ${r.question ? `ask “${esc(clip(r.question, 120))}”` : `${esc(r.choice ?? "")} ${esc(r.action ?? "")}`}${r.reason ? ` — ${esc(clip(r.reason, 120))}` : ""}`;
    case "job_attempt": return `${esc(r.job)} #${esc(r.attempt)}: ${esc(r.action)} → ${esc(r.outcome)}`;
    case "job_end": return `${esc(r.job)}: ${esc(r.result)} after ${Array.isArray(r.attempts) ? r.attempts.length : esc(r.attempts)} attempt(s)`;
    case "record_write": return `${esc(r.offer)} → ${esc(r.choice)}${r.text ? `: ${esc(clip(r.text, 180))}` : ""}`;
    case "record_read": return `${esc(r.context)} · shown ${(r.rev_ids || []).length + (r.entry_ids || []).length}, new ${(r.new_ids || []).length}`;
    case "record_transition": return `day ${esc(r.day)}: ${esc(r.mode)}${r.archived_binder_id ? ` (archived ${esc(r.archived_binder_id)})` : ""}`;
    case "roster_change": return `${esc(r.kind)} · ${esc(roleName(r.role))}${r.replaces ? ` · replaces ${esc(name(r.replaces))}` : ""}`;
    case "memory_link": return `${esc(r.mechanism)}: ${esc(r.from)} → ${esc(r.to)}${r.reason ? ` · ${esc(clip(r.reason, 100))}` : ""}`;
    case "open_matter": return `${esc(r.action)} (${r.strength}) · ${esc(clip(r.text, 180))}`;
    case "wording": return `“${esc(r.phrase)}” from ${esc(name(r.heard_from))}`;
    case "reminding": return r.reminded_of ? `felt alike: ${esc(clip(r.what_felt_alike || "", 120))}` : `<span class="muted">no reminding</span>`;
    case "day_plan": return `${Array.isArray(r.plan) ? r.plan.length + " plan items" : esc(clip(JSON.stringify(r.plan), 160))}`;
    case "tally": return esc(clip(r.text));
    case "coop_day": return esc(Object.entries(r.counts || {}).map(([k, v]) => `${k} ${v}`).join(" · "));
    case "handover": return `day ${esc(r.day)} shift handover${r.n_utterances != null ? ` · ${esc(r.n_utterances)} lines` : ""}`;
    case "meeting": return `day ${esc(r.day)} co-op meeting · ${(r.invited || []).length} invited${r.n_utterances != null ? ` · ${esc(r.n_utterances)} lines` : ""}`;
    case "onboarding": return `day ${esc(r.day)} orientation · ties to ${(r.ties || []).map((x) => `${esc(name(x.with))} (${esc(x.relation_type)})`).join(", ")}`;
    case "exposure": return `heard “${esc(clip(r.utterance, 160))}” @ ${esc(r.location)}`;
    case "priming": return `primed: ${(r.phrases || []).map((x) => `“${esc(typeof x === "string" ? x : x?.phrase || JSON.stringify(x))}”`).join(", ")}`;
    case "open_matter_focal": return `focal: ${(r.texts || []).map((x) => esc(clip(x, 90))).join(" · ")}`;
    case "invitation": return `invites ${esc(name(r.target))}${r.until ? ` until ${esc(hhmm(r.until) || r.until)}` : ""}`;
    case "replan": return `${esc(r.reason ?? "")} → ${esc(typeof r.to === "string" ? r.to : JSON.stringify(r.to))}`;
    case "checkpoint": return `${(r.files || []).length} checkpoint file(s)`;
    case "reflection": return `${esc(clip(r.text))}${r.focal_point ? ` <span class="muted">(focal: ${esc(clip(r.focal_point, 80))})</span>` : ""}`;
  }
  for (const k of ["text", "phrase", "says_aloud", "question", "project", "into_text", "focal_point", "decision", "reason", "activity", "mode", "kind", "result"]) {
    const v = r[k];
    if (typeof v === "string" && v) return `${k !== "text" ? `<span class="muted">${esc(k)}:</span> ` : ""}${esc(clip(v))}`;
    if (v && typeof v === "object" && !Array.isArray(v)) return `<span class="muted">${esc(k)}:</span> ${esc(clip(JSON.stringify(v), 160))}`;
  }
  const keys = Object.keys(r).filter((k) => !["id", "type", "tick", "time"].includes(k));
  return `<span class="muted">${esc(keys.slice(0, 8).join(", "))}</span>`;
}
function renderTraceTable() {
  const recs = trState.records || [], types = S.schema?.types || {};
  const counts = {}; for (const r of recs) counts[r.type] = (counts[r.type] || 0) + 1;
  $("#trInfo").innerHTML = `${recs.length} record${recs.length === 1 ? "" : "s"}${recs.length >= trState.limit ? ` <b>(limit reached: narrow the filters or raise the limit)</b>` : ""}` +
    (recs.length ? ` · ${Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<span class="tag">${esc(k)} ${v}</span>`).join("")}` : "");
  $("#trResults").innerHTML = recs.length ? `<table class="table"><tr><th>When</th><th>Type</th><th>Who</th><th>Record</th><th></th></tr>${recs.map((r, i) => {
    const L = types[r.type]?.layer || "", conv = r.conversation_id || (r.type === "conversation" ? r.id : null);
    return `<tr class="tr-row"><td class="nowrap"><a href="#" data-jump="${esc(r.tick)}" title="show this tick on the campus map">${esc(labelAt(r.tick))}</a><div class="muted small">tick ${esc(r.tick)}</div></td>
      <td><span class="ttype l-${esc(L)}" title="${esc(types[r.type]?.description || "")}">${esc(r.type)}</span></td>
      <td class="small">${recWho(r)}</td>
      <td>${recSummary(r)}${conv ? ` <a href="#" class="small" data-conv="${esc(conv)}" title="all records of this conversation">[conv ${esc(conv)}]</a>` : ""}${r.type === "utterance" ? ` <a href="#" class="small" data-chain="${esc(r.id)}">[causal trace]</a>` : ""}</td>
      <td><button class="pill" data-raw="${i}" title="raw record">{ }</button></td></tr><tr class="raw-row" hidden><td colspan="5"></td></tr>`;
  }).join("")}</table>` : `<div class="muted small">No records match.</div>`;
  const box = $("#trResults");
  $$("[data-raw]", box).forEach((b) => b.onclick = () => {
    const row = b.closest("tr").nextElementSibling;
    if (row.hidden && !row.firstElementChild.innerHTML) row.firstElementChild.innerHTML = `<pre class="raw">${esc(JSON.stringify(recs[+b.dataset.raw], null, 1))}</pre>`;
    row.hidden = !row.hidden;
  });
  $$("[data-jump]", box).forEach((a) => a.onclick = (e) => { e.preventDefault(); showView("campus"); setTick(+a.dataset.jump); });
  $$("[data-conv]", box).forEach((a) => a.onclick = (e) => { e.preventDefault(); $("#trConv").value = a.dataset.conv; loadTrace(); });
  $$("[data-chain]", box).forEach((a) => a.onclick = (e) => { e.preventDefault(); openChain(a.dataset.chain); });
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
  const uttNode = (u) => `<div class="node utt"><span class="kind">utterance · ${esc(name(u.speaker))} → ${(u.listeners || []).map(name).map(esc).join(", ") || "nobody"} · ${esc(u.time?.slice(11, 16))}</span><div>“${esc(u.text)}”</div>
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
  const cols = cmp.some((r) => r.mode === "commons")
    ? ["condition", "seed", "records_enabled", "change_enabled", "projects_completed", "projects_unfinished", "failed_deliveries", "record_reads", "newcomers_with_success", "post_boundary_success"]
    : ["condition", "events", "conversations", "utterances", "reflections", "candidates", "llm_conventions", "max_adoption", "mean_depth", "cross_group_edges", "mean_coherence", ...(S.debug ? ["mean_alignment", "mean_lift"] : []), "top_expression"];
  $("#compareTable").innerHTML = `<tr>${cols.map((c) => `<th>${c.replace(/_/g, " ")}</th>`).join("")}</tr>` +
    cmp.map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c] ?? "—")}</td>`).join("")}</tr>`).join("");
  const cfgs = await api("/configs");
  $("#cfgSelect").innerHTML = cfgs.map((c) => `<option>${esc(c)}</option>`).join("");
}
async function launchRun() {
  const q = new URLSearchParams({ config: $("#cfgSelect").value, backend: $("#cfgBackend").value });
  if ($("#cfgDays").value) q.set("days", $("#cfgDays").value);
  const r = await fetch(`/api/runs?${q}`, { method: "POST" });
  $("#launchMsg").textContent = r.ok ? "Launched — it will appear in the list shortly (refresh)." : "Launch failed.";
}

// console / test hook (read-only use: inspect state, drive the replay)
window.MW = { S, M, setTick, renderAgent, showView, zoomFit, loadRun };

boot();
