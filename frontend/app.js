// MemeWorld frontend: campus replay, culture dashboard, causal trace explorer.
// Normal demo mode never requests hidden ground truth; Research Debug Mode adds ?debug=1.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const S = {
  runs: [], runId: null, debug: false, manifest: null, frames: [], analysis: null,
  tick: 0, tf: 0, playing: false, speed: 1, sel: null, sprites: {}, layout: null,
  selMeme: null, agentReq: 0,
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
  $("#btnLaunch").onclick = launchRun;
  const canvas = $("#map");
  canvas.onclick = onMapClick;
  canvas.onmousemove = onMapHover;
  canvas.onmouseleave = hideTip;
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowRight") setTick(S.tick + 1);
    if (e.code === "ArrowLeft") setTick(S.tick - 1);
  });
  S.runs = await api("/runs");
  const sel = $("#runSelect");
  sel.innerHTML = S.runs.map((r) => `<option value="${r.run_id}">${esc(r.run_id)} (${r.status}${r.has_analysis ? ", analyzed" : ""})</option>`).join("");
  const want = new URLSearchParams(location.search).get("run");
  const first = S.runs.find((r) => r.run_id === want) || S.runs.find((r) => r.status === "finished" && r.has_analysis) || S.runs[0];
  if (first) { sel.value = first.run_id; await loadRun(first.run_id); }
  renderRuns();
  requestAnimationFrame(loop);
}

function showView(v) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === "view-" + v));
  if (v === "culture") renderCulture();
  if (v === "trace") renderTraceSearch();
  if (v === "runs") renderRuns();
  if (v === "world") renderWorld();
}

async function loadRun(id, keepTick = false) {
  if (!id) return;
  S.runId = id;
  const [man, frames] = await Promise.all([api(`/runs/${id}/manifest`), api(`/runs/${id}/frames`)]);
  S.manifest = man; S.frames = frames;
  const commons = man.world.mode === "commons";
  $("#worldTab").hidden = !commons;
  $("#btnFirstMeme").hidden = commons;
  $("#btnFirstCross").hidden = commons;
  $("#onlyConv").parentElement.hidden = commons;
  $("#cultureHeading").textContent = commons ? "Inheritance and adaptation" : "Emerging cultural conventions";
  $("#cultureDescription").textContent = commons
    ? "RQ2: How do shared records affect continuity and adaptation? These measures describe behavior; semantic change is not evaluated."
    : "Detected by the external observer only (n-gram statistics + variant grouping + LLM classifier). Nothing on this page is ever shown to agents.";
  if (!commons && $("#view-world").classList.contains("active")) showView("campus");
  try { S.analysis = await api(`/runs/${id}/analysis`); } catch { S.analysis = null; }
  S.layout = computeLayout(man);
  for (const [aid, a] of Object.entries(man.agents)) {
    if (!S.sprites[a.sprite]) { const img = new Image(); img.src = `/ga_assets/characters/${a.sprite}.png`; S.sprites[a.sprite] = img; }
  }
  $("#scrub").max = Math.max(0, frames.length - 1);
  $("#worldScrub").max = Math.max(0, frames.length - 1);
  if (!keepTick) setTick(0); else setTick(S.tick);
  if (S.sel) renderAgent(S.sel);
  renderCulture();
}

// ------------------------------------------------------------------ layout
const W = 1000, H = 620;
function computeLayout(man) {
  const L = {};
  for (const [loc, [x, y]] of Object.entries(man.world.map_pos)) {
    const quad = loc === "Quad";
    const arenas = man.world.arenas[loc];
    const cols = Math.min(arenas.length, loc === "Dorm" ? 4 : 3);
    const rows = Math.ceil(arenas.length / cols);
    const w = quad ? 230 : Math.max(170, cols * 80), h = quad ? 130 : 40 + rows * 60;
    const cx = x * W, cy = y * H;
    const slots = {};
    arenas.forEach((ar, i) => {
      const c = i % cols, r = Math.floor(i / cols);
      slots[ar] = { x: cx - w / 2 + (c + 0.5) * (w / cols), y: cy - h / 2 + 22 + (r + 0.5) * ((h - 34) / rows) };
    });
    L[loc] = { x: cx, y: cy, w, h, slots, label: man.world.labels[loc] };
  }
  return L;
}

function positions(frame) {
  const out = {}, groups = {};
  if (!frame) return out;
  for (const [aid, a] of Object.entries(frame.agents)) {
    const key = a.location + "|" + a.arena;
    (groups[key] = groups[key] || []).push(aid);
  }
  for (const [key, ids] of Object.entries(groups)) {
    const [loc, ar] = key.split("|");
    const L = S.layout[loc]; if (!L) continue;
    const slot = L.slots[ar] || { x: L.x, y: L.y };
    ids.sort();
    ids.forEach((aid, i) => { out[aid] = { x: slot.x + (i - (ids.length - 1) / 2) * 26, y: slot.y + 4 }; });
  }
  return out;
}

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
  renderWorld();
  if (S.sel) { clearTimeout(agentTimer); agentTimer = setTimeout(() => renderAgent(S.sel), S.playing ? 600 : 80); }
}
function jumpTo(key) {
  const s = S.analysis?.summary?.[key];
  if (!s) { alert("No such moment in this run (or the run is not analyzed yet)."); return; }
  const t = s.tick ?? s.first_reuse_timestamp;
  showView("campus"); S.playing = false; $("#btnPlay").textContent = "▶"; setTick(t);
}

// -------------------------------------------------------------------- draw
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function roundRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }

function draw() {
  const cv = $("#map"), ctx = cv.getContext("2d");
  const C = { grass: cssVar("--grass"), path: cssVar("--path"), b: cssVar("--building"), be: cssVar("--building-edge"),
    text: cssVar("--text"), muted: cssVar("--muted"), accent: cssVar("--accent"), acc2: cssVar("--accent-2"),
    bubble: cssVar("--bubble"), btext: cssVar("--bubble-text") };
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = C.grass; ctx.fillRect(0, 0, W, H);
  if (!S.layout) return;
  const man = S.manifest;
  // paths
  ctx.strokeStyle = C.path; ctx.lineWidth = 10; ctx.lineCap = "round";
  for (const [a, nbs] of Object.entries(man.world.graph)) for (const b of nbs) if (a < b) {
    ctx.beginPath(); ctx.moveTo(S.layout[a].x, S.layout[a].y); ctx.lineTo(S.layout[b].x, S.layout[b].y); ctx.stroke();
  }
  const frame = S.frames[S.tick];
  const active = new Set((frame?.beats || []).map((b) => b.location));
  // buildings
  for (const [loc, L] of Object.entries(S.layout)) {
    const x = L.x - L.w / 2, y = L.y - L.h / 2;
    ctx.fillStyle = loc === "Quad" ? C.grass : C.b; ctx.strokeStyle = active.has(loc) ? C.acc2 : C.be; ctx.lineWidth = active.has(loc) ? 3 : 1.5;
    roundRect(ctx, x, y, L.w, L.h, 10); ctx.fill(); ctx.stroke();
    ctx.fillStyle = C.text; ctx.font = "600 13px system-ui"; ctx.textAlign = "left"; ctx.fillText(loc, x + 8, y + 16);
    const lw = ctx.measureText(loc).width;
    ctx.fillStyle = C.muted; ctx.font = "11px system-ui"; ctx.fillText(L.label, x + 14 + lw, y + 16);
    ctx.font = "9.5px system-ui"; ctx.textAlign = "center";
    for (const [ar, p] of Object.entries(L.slots)) if (Object.keys(L.slots).length > 1) ctx.fillText(ar, p.x, p.y + 33);
    if (active.has(loc)) { ctx.fillStyle = C.acc2; ctx.font = "700 14px system-ui"; ctx.fillText("!", x + L.w - 12, y + 17); }
  }
  if (!frame) return;
  // agents, interpolated from the previous frame along the semantic path
  const p = S.tf - S.tick;
  const cur = positions(frame), prev = positions(S.frames[S.tick - 1] || frame);
  const drawPos = {};
  for (const [aid, a] of Object.entries(frame.agents)) {
    const to = cur[aid], from = prev[aid] || to;
    let pos = to, dir = 0, walking = false;
    const k = Math.min(1, p / 0.45);
    if (k < 1 && from && (from.x !== to.x || from.y !== to.y)) {
      const pts = [from, ...(a.path || []).slice(1, -1).map((l) => S.layout[l]).filter(Boolean), to];
      const segs = []; let tot = 0;
      for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y); segs.push(d); tot += d; }
      let dist = k * tot, i = 0;
      while (i < segs.length - 1 && dist > segs[i]) { dist -= segs[i]; i++; }
      const a0 = pts[i], a1 = pts[i + 1], f = segs[i] ? dist / segs[i] : 1;
      pos = { x: a0.x + (a1.x - a0.x) * f, y: a0.y + (a1.y - a0.y) * f };
      const dx = a1.x - a0.x, dy = a1.y - a0.y;
      dir = Math.abs(dx) > Math.abs(dy) ? (dx < 0 ? 1 : 2) : (dy < 0 ? 3 : 0);
      walking = true;
    }
    drawPos[aid] = pos;
    const prof = man.agents[aid], img = S.sprites[prof.sprite];
    const fx = walking ? [0, 32, 64][Math.floor(performance.now() / 150) % 3] : 32;
    if (S.sel === aid) { ctx.fillStyle = C.accent; ctx.globalAlpha = .25; ctx.beginPath(); ctx.arc(pos.x, pos.y, 20, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
    if (img?.complete && img.naturalWidth) { ctx.imageSmoothingEnabled = false; ctx.drawImage(img, fx, dir * 32, 32, 32, pos.x - 16, pos.y - 24, 32, 32); }
    else { ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(pos.x, pos.y - 8, 9, 0, 7); ctx.fill(); }
    ctx.fillStyle = C.text; ctx.font = "600 10px system-ui"; ctx.textAlign = "center"; ctx.fillText(prof.name.split(" ")[0], pos.x, pos.y + 18);
    if (a.conversation) { ctx.fillStyle = C.accent; ctx.fillText("…", pos.x + 14, pos.y - 22); }
  }
  // speech bubbles (cycle through a conversation's lines within the tick)
  if (p > 0.45) {
    const convs = {};
    for (const u of frame.utterances) (convs[u.conversation_id || u.id] = convs[u.conversation_id || u.id] || []).push(u);
    for (const us of Object.values(convs)) {
      const k = Math.min(us.length - 1, Math.floor(((p - 0.45) / 0.55) * us.length));
      const u = us[k], sp = drawPos[u.speaker]; if (!sp) continue;
      ctx.strokeStyle = C.accent; ctx.globalAlpha = 0.5; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      for (const l of u.listeners) { const lp = drawPos[l]; if (lp) { ctx.beginPath(); ctx.moveTo(sp.x, sp.y - 10); ctx.lineTo(lp.x, lp.y - 10); ctx.stroke(); } }
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      bubble(ctx, sp.x, sp.y - 30, u.text, C);
    }
  }
}
function bubble(ctx, x, y, text, C) {
  ctx.font = "11px system-ui";
  const words = text.split(" "), lines = []; let line = "";
  for (const w of words) { if (ctx.measureText(line + " " + w).width > 190 && line) { lines.push(line); line = w; } else line = line ? line + " " + w : w; if (lines.length >= 3) break; }
  if (lines.length < 3 && line) lines.push(line);
  if (lines.length === 3 && text.length > lines.join(" ").length) lines[2] += "…";
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 14, h = lines.length * 14 + 8;
  const bx = Math.max(4, Math.min(W - w - 4, x - w / 2)), by = Math.max(4, y - h);
  ctx.fillStyle = C.bubble; ctx.strokeStyle = C.accent; ctx.lineWidth = 1;
  roundRect(ctx, bx, by, w, h, 7); ctx.fill(); ctx.stroke();
  ctx.fillStyle = C.btext; ctx.textAlign = "left";
  lines.forEach((l, i) => ctx.fillText(l, bx + 7, by + 15 + i * 14));
}
function mapXY(e) { const r = $("#map").getBoundingClientRect(); return { x: (e.clientX - r.left) * W / r.width, y: (e.clientY - r.top) * H / r.height }; }
function agentAt(e) {
  const { x, y } = mapXY(e), pos = positions(S.frames[S.tick]);
  let best = null, bd = 22;
  for (const [aid, p] of Object.entries(pos)) { const d = Math.hypot(p.x - x, p.y - 8 - y); if (d < bd) { bd = d; best = aid; } }
  return best;
}
function onMapClick(e) { const a = agentAt(e); if (a) { S.sel = a; renderAgent(a); } }
function onMapHover(e) {
  const a = agentAt(e); const f = S.frames[S.tick];
  if (a && f) { const st = f.agents[a]; showTip(e, `<b>${esc(S.manifest.agents[a].name)}</b><br>${esc(st.activity)}<br><span class="muted">${esc(st.location)} · ${esc(st.arena)}</span>`); }
  else hideTip();
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
  $("#agentPanel").innerHTML = `
    <div class="agent-head"><div class="avatar" style="background-image:url(/ga_assets/characters/${p.sprite}.png)"></div>
      <div><div style="font-weight:700">${esc(p.name)}</div><div class="muted small">${esc(p.demographics.year)} · ${esc(p.demographics.major)} · ${esc(p.demographics.role)}</div></div></div>
    <dl class="kv">
      <dt>Now</dt><dd>${esc(st.activity)} <span class="muted">@ ${esc(st.location)} / ${esc(st.arena)}</span></dd>
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
  if (A.mode === "commons") { renderCommonsResearch(A); return; }
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
    if (m.observation) inner += `<div class="node obs"><span class="kind">${esc(m.observation.source_type)} observation — what ${esc(name(m.agent))} actually noticed</span><div>${(m.observation.facts || []).map((f) => esc(f.text) + (f.p_attend != null ? ` <span class="muted small">(p=${f.p_attend})</span>` : "")).join("<br>")}</div>${inner.includes("world") ? "" : ""}</div>`;
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

function statsHtml(items) {
  return items.map(([label, value]) => `<div class="stat"><div class="v">${esc(value ?? "—")}</div><div class="l">${esc(label)}</div></div>`).join("");
}

function renderWorld() {
  const frame = S.frames[S.tick], world = frame?.commons;
  if (!world) return;
  $("#worldClock").textContent = frame.label;
  $("#worldScrub").value = S.tick;
  const projects = Object.values(world.projects);
  $("#worldSummary").innerHTML = statsHtml([
    ["Requests completed", `${projects.filter((p) => p.completed !== null).length} / ${projects.length}`],
    ["Components", world.supplies.components], ["Test supplies", world.supplies.test_supplies],
    ["Archive", world.records_enabled ? "Available" : "Unavailable"], ["Outdoor conditions", world.outdoor_condition],
  ]);
  $("#worldProjects").innerHTML = `<table class="table"><tr><th>Project / kit</th><th>Site</th><th>State</th><th>Location / custody</th><th>Attempts</th></tr>` + projects.map((p) => {
    const k = world.kits[p.kit];
    const state = p.completed !== null ? `Complete at tick ${p.completed}` : !k.assembled ? "Needs assembly" : k.calibration === null ? "Needs calibration" : `Calibrated ${k.calibration}; request open`;
    const holder = k.holder ? ` · ${S.manifest.agents[k.holder]?.name || k.holder}` : "";
    return `<tr><td>${esc(p.title)}<div class="muted small">${esc(k.id)} · due tick ${p.due}</div></td><td>${esc(p.site)}</td><td>${esc(state)}</td><td>${esc(k.location + holder)}</td><td>${p.attempts}</td></tr>`;
  }).join("") + "</table>";
  $("#worldActivity").innerHTML = Object.values(world.residents).filter((r) => r.active).map((r) => {
    const op = world.operations[r.id];
    return `<div class="mem"><b>${esc(r.name)}</b> · ${esc(r.location)}<div class="muted small">${op ? `${esc(op.intention.action)}${op.intention.kit ? " · " + esc(op.intention.kit) : ""} · until tick ${op.due}` : "Available"}${r.joined ? ` · joined at tick ${r.joined}` : ""}</div></div>`;
  }).join("");
  const records = Object.values(world.records);
  $("#worldRecords").innerHTML = !world.records_enabled ? '<p class="muted">The shared archive is unavailable in this condition.</p>'
    : !records.length ? '<p class="muted">No shared records have been written yet.</p>'
    : records.map((r) => `<details class="record-history"><summary>${esc(r.versions.at(-1).title)} <span class="muted">${esc(r.id)} · ${r.versions.length} version(s)</span></summary>${r.versions.map((v) => {
      const reads = (S.analysis?.record_access || []).filter((e) => e.record === r.id && e.version === v.version && e.tick <= S.tick);
      const readers = [...new Set(reads.map((e) => S.manifest.agents[e.actor]?.name || e.actor))];
      return `<article class="mem"><div class="meta">Version ${v.version} · ${esc(S.manifest.agents[v.author]?.name || v.author)} · tick ${v.tick}</div><div class="record-text">${esc(v.text)}</div><div class="meta">${S.analysis?.mode === "commons" ? `Read ${reads.length} time(s) by ${esc(readers.join(", ") || "nobody yet")}` : "Read history available after analysis"}</div></article>`;
    }).join("")}</details>`).join("");
}

function renderCommonsResearch(A) {
  const s = A.summary;
  $("#cultureSummary").innerHTML = statsHtml([["Projects completed", s.projects_completed], ["Unfinished", s.projects_unfinished], ["Failed deliveries", s.failed_deliveries], ["Record versions", s.record_versions], ["Explicit reads", s.record_reads], ["Local utterances", s.n_utterances]]);
  $("#cards").innerHTML = "";
  $("#memeDetail").innerHTML = `<div class="panel table-scroll"><h3>Before and after the common intervention boundary</h3><p class="muted small">The same boundary is used in stable controls. These are descriptive summaries, not estimates of a treatment effect.</p><table class="table"><tr><th>Period</th><th>Attempts</th><th>Successful</th><th>Failed</th><th>Success / attempt</th></tr>${Object.entries(A.phases).map(([label, p]) => `<tr><td>${esc(label.replace(/_/g, " "))}</td><td>${p.attempts}</td><td>${p.successful_deliveries}</td><td>${p.failed_deliveries}</td><td>${p.success_per_attempt ?? "—"}</td></tr>`).join("")}</table></div>
    <div class="panel table-scroll"><h3>Newcomer participation</h3><table class="table"><tr><th>Member</th><th>Ticks to first successful delivery</th><th>Record reads</th><th>Reads of prearrival records</th></tr>${A.newcomers.map((n) => `<tr><td>${esc(S.manifest.agents[n.agent]?.name || n.agent)}</td><td>${n.ticks_to_first_success ?? "No success before run ended"}</td><td>${n.record_reads}</td><td>${n.prearrival_record_reads}</td></tr>`).join("")}</table></div>
    <div class="panel"><h3>Research question coverage</h3>${Object.entries(A.question_status).map(([q, text]) => `<p><b>${esc(q)}</b> ${esc(text)}</p>`).join("")}</div>
    <div class="panel"><h3>Interpretation</h3>${A.limitations.map((text) => `<p class="small">${esc(text)}</p>`).join("")}</div>`;
}

boot();
