// Research tools read archived evidence. They never send interpretations back to agents.
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (v) => v == null ? "—" : typeof v === "number" ? Number(v.toPrecision(4)).toString() : esc(v);
const colors = ["#2a78d6", "#d56a39", "#168263", "#9260b7", "#aa8210", "#b94777", "#377b9a", "#65742d"];
let hooks, catalog = [], currentRun = null, analysis = null, selected = null, refreshToken = 0, timer;
const fileURL = (path) => "/api/research/files/" + path.split("/").map(encodeURIComponent).join("/");

async function api(path, options = {}) {
  const r = await fetch("/api/research" + path, options);
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `Request failed (${r.status})`); }
  return r.json();
}
function query(params) { return new URLSearchParams(Object.entries(params).filter(([,v]) => v != null && v !== "")).toString(); }
function message(text, error = false) { const el = $("#researchMessage"); el.textContent = text; el.classList.toggle("error", error); }
function guarded(fn) { return async (...args) => { try { await fn(...args); } catch(e) { message(e.message, true); } }; }
function observer() { return {backend: $("#observerBackend").value || null, model: $("#observerModel").value.trim() || null}; }
async function launch(body) {
  const job = await api("/jobs", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  message(`Started ${body.action}. The job continues if this page closes. Saved progress and logs appear below.`);
  await refreshJobs();
  return job;
}

export function mountResearch(callbacks) {
  hooks = callbacks;
  const root = $("#view-research");
  root.innerHTML = `<div class="research-heading"><div><span class="eyebrow">MEMETICS OBSERVATORY</span><h1>Follow ideas as a society develops.</h1><p>Run comparative worlds, discover emerging concepts, and investigate their changing meanings through original evidence.</p></div><button class="pill" id="researchRefresh">Refresh</button></div>
    <div id="researchMessage" role="status" aria-live="polite"></div>
    <div class="research-columns"><section class="panel"><h2>Experiment library</h2><p class="muted">Editable designs change existing world conditions. The observer remains open to developments beyond the motivating questions.</p>
      <label class="research-field">Study<select id="studySelect" aria-label="Study"></select></label><div id="studyDescription"></div>
      <div class="research-controls"><label class="research-field">Execution<select id="studyBackend"><option value="mock">Mock · software verification</option><option value="">Configured live providers</option><option value="claude_cli">Claude CLI</option><option value="anthropic">Anthropic API</option></select></label><label class="research-field">Parallel societies<input id="studyParallel" type="number" min="1" max="64" value="1"></label></div>
      <div class="research-controls"><button class="primary" id="runStudy">Run full experiment</button><button class="pill" id="reportStudy">Refresh report</button></div>
      <p id="studyExecutionNote" class="muted small"></p><div id="studyStatus"></div></section>
    <section class="panel"><h2>Investigate the selected society</h2><p class="muted" id="researchRunName">Select a run in the header.</p>
      <div class="research-controls"><label class="research-field">Observer<select id="observerBackend"><option value="">From run configuration</option><option value="mock">Mock · no semantic judgments</option><option value="claude_cli">Claude CLI</option><option value="anthropic">Anthropic API</option></select></label><label class="research-field">Model override<input id="observerModel" placeholder="From configuration"></label></div>
      <div class="research-controls"><button class="primary" id="observeRun">Observe recorded history</button><button class="pill" id="auditRun">Audit interpretations</button><button class="pill" id="pauseRun">Pause simulation</button></div>
      <details><summary>Continue or extend a recorded society</summary><div class="research-controls"><label class="research-field">New run directory<input id="continueOut" placeholder="e.g. extended_society"></label><label class="research-field">Total days<input id="continueDays" type="number" min="1" placeholder="Original horizon"></label><button class="pill" id="continueRun">Continue and analyze</button></div><p class="muted small">Replays and verifies the committed prefix before continuing. Uses a new recording and the original simulation code.</p></details>
      <label class="research-field">Research question<textarea id="researchQuestion" rows="3" placeholder="How did the ideas around responsibility change, and did different groups interpret them differently?"></textarea></label>
      <details><summary>Focus the inquiry (optional)</summary><div class="research-controls"><label class="research-field">From tick<input id="inquiryStart" type="number" min="0"></label><label class="research-field">Through tick<input id="inquiryEnd" type="number" min="0"></label><label class="research-field">Agent ID<input id="inquiryAgent" placeholder="All agents"></label><label class="switch"><input id="inquiryPublic" type="checkbox"> Exclude private representations</label></div></details>
      <div class="research-controls"><button class="primary" id="askRun">Investigate this society</button><button class="pill" id="askStudy">Compare societies in this study</button></div>
    </section></div>
    <section class="panel"><div class="research-controls"><h2>Cultural histories</h2><label class="research-field">Saved observation<select id="snapshotSelect"><option value="">Current</option></select></label></div><div id="researchCoverage"></div><div id="researchHistories"></div></section>
    <div class="research-columns"><section class="panel"><h2>Saved inquiries</h2><div id="researchInquiries" class="research-scroll"></div></section><section class="panel"><h2>Execution and audit trail</h2><div id="researchJobs" class="research-scroll"></div><div id="researchAudits"></div></section></div>`;
  $("#researchRefresh").onclick = guarded(refreshResearch);
  $("#studySelect").onchange = guarded(refreshStudy);
  $("#studyBackend").onchange = guarded(refreshStudy);
  $("#snapshotSelect").onchange = guarded(() => refreshAnalysis($("#snapshotSelect").value));
  $("#runStudy").onclick = guarded(() => launch({action:"experiment", experiment:$("#studySelect").value, backend:$("#studyBackend").value || null, parallel:+$("#studyParallel").value}));
  $("#reportStudy").onclick = guarded(() => launch({action:"report", experiment:$("#studySelect").value, backend:$("#studyBackend").value || null}));
  $("#observeRun").onclick = guarded(() => launch({action:"observe", run:hooks.run(), ...observer()}));
  $("#auditRun").onclick = guarded(() => launch({action:"audit", run:hooks.run(), ...observer()}));
  $("#pauseRun").onclick = guarded(async () => { await api("/pause?" + query({run:hooks.run()}), {method:"POST"}); message("Pause requested. The simulation will stop after its current tick is committed."); });
  $("#continueRun").onclick = guarded(() => launch({action:"continue", run:hooks.run(), out:$("#continueOut").value.trim(), days:$("#continueDays").value ? +$("#continueDays").value : null, ...observer()}));
  $("#askRun").onclick = guarded(() => launch({action:"inquire", run:hooks.run(), question:$("#researchQuestion").value,
    start:$("#inquiryStart").value ? +$("#inquiryStart").value : null, end:$("#inquiryEnd").value ? +$("#inquiryEnd").value : null,
    agent:$("#inquiryAgent").value || null, public_only:$("#inquiryPublic").checked, ...observer()}));
  $("#askStudy").onclick = guarded(() => launch({action:"synthesize", experiment:$("#studySelect").value, backend:$("#studyBackend").value || null, question:$("#researchQuestion").value}));
  clearInterval(timer);
  timer = setInterval(() => { if (root.classList.contains("active")) guarded(refreshJobs)(); }, 5000);
}

export async function refreshResearch() {
  try {
    if (!catalog.length) {
      catalog = await api("/experiments");
      $("#studySelect").innerHTML = catalog.map(e => `<option value="${esc(e.id)}">${esc(e.title)}</option>`).join("");
    }
    currentRun = hooks.run();
    $("#researchRunName").textContent = currentRun || "Select a run in the header.";
    await Promise.all([refreshStudy(), refreshJobs(), refreshAnalysis()]);
    if (currentRun) {
      const [saved, inquiries, audits] = await Promise.all([api("/snapshots?" + query({run:currentRun})), api("/inquiries?" + query({run:currentRun})), api("/audits?" + query({run:currentRun}))]);
      $("#snapshotSelect").innerHTML = `<option value="">Current</option>` + saved.map(a => `<option value="${esc(a.analysis_id)}">${esc(a.analysis_id)} · ${esc(a.observer.model)} · ${esc(a.status)} · tick ${a.coverage.through_tick}</option>`).join("");
      $("#researchInquiries").innerHTML = inquiries.map(q => `<article><h3>${esc(q.question)}</h3><p>${esc(q.answer)}</p><p class="muted small">${q.plan.supplied_events}/${q.plan.eligible_events} eligible events supplied to the question-specific judge</p><a target="_blank" href="${fileURL(`${currentRun}/inquiries/${q.id}/report.html`)}">Read claims and source evidence ↗</a> · <a href="${fileURL(`${currentRun}/inquiries/${q.id}/report.md`)}">Markdown</a></article>`).join("") || '<p class="muted">Ask any question about the recorded history. Questions and evidence-backed answers will be saved here.</p>';
      $("#researchAudits").innerHTML = audits.map(a => `<details><summary>Audit · ${a.reviewed}/${a.population} annotations</summary><p>${esc(a.limits)}</p><pre>${esc(JSON.stringify(a.counts, null, 2))}</pre><a href="${fileURL(`${currentRun}/audits/${a.id}/results.json`)}">All verdicts and explanations</a></details>`).join("");
    }
  } catch (e) { message(e.message, true); }
}

async function refreshStudy() {
  const entry = catalog.find(e => e.id === $("#studySelect").value);
  if (!entry) return;
  $("#studyDescription").innerHTML = `<div class="research-tags"><span>${entry.runs} societies</span><span>${entry.seeds.length} world seeds</span><span>${entry.replicates} agent replica(s)</span><span>base horizon ${entry.days} days</span></div><ul>${entry.questions.map(q => `<li>${esc(q)}</li>`).join("")}</ul><details><summary>Conditions and resolved overlays</summary><pre>${esc(JSON.stringify(entry.factors, null, 2))}</pre><p class="muted small">Design: ${esc(entry.design)}. Copy and edit this YAML to create another study.</p></details>`;
  const backend = $("#studyBackend").value;
  $("#studyExecutionNote").textContent = backend === "mock" ? "Mock mode exercises the software only. It cannot provide evidence about memes or meaning change." : `This launches ${entry.runs} simulations plus observer and inquiry calls using the selected live providers.`;
  const result = await api("/experiment?" + query({name:entry.id, backend}));
  const counts = {};
  result.runs.forEach(r => counts[r.status] = (counts[r.status] || 0) + 1);
  $("#studyStatus").innerHTML = `<p>${Object.entries(counts).map(([s,n]) => `<span class="tag">${n} ${esc(s)}</span>`).join(" ")}</p>` + (result.pipeline ? `<p class="muted">Pipeline: ${esc(result.pipeline.status)} · ${esc(result.pipeline.stage)}</p>` : "") +
    (result.report ? `<a target="_blank" href="${fileURL(`${result.directory}/reports/${result.report.report_id}/report.html`)}">Open comparison, uncertainty and research report ↗</a><div class="research-table"><table class="table"><tr><th>Condition</th><th>Concept histories</th><th>Judged changes</th><th>World blocks</th></tr>${Object.entries(result.report.summaries).map(([name,v]) => `<tr><td>${esc(name)}</td><td>${fmt(v.n_threads?.mean)}</td><td>${fmt(v.n_changes?.mean)}</td><td>${fmt(v.n_threads?.world_blocks)}</td></tr>`).join("")}</table></div>` : '<p class="muted small">The completed report includes all run statuses, matched comparisons, uncertainty intervals and source-linked inquiries.</p>');
}

let jobSignature = null;
async function refreshJobs() {
  const jobs = await api("/jobs");
  $("#researchJobs").innerHTML = jobs.map(j => `<details><summary><span class="tag">${esc(j.status)}</span> ${esc(j.kind)} · ${esc(j.id)}</summary><p>${esc(j.error || "")}</p><pre>${esc(j.log || "Waiting for worker…")}</pre></details>`).join("") || '<p class="muted">No research jobs have been launched from the interface.</p>';
  const signature = jobs.map(j => `${j.id}:${j.status}`).join("|");
  if (jobSignature != null && jobSignature !== signature && jobs.some(j => j.status === "complete")) {
    await refreshStudy();
    if (hooks.run()) await refreshAnalysis();
    message("Job status changed. Use Refresh to load newly saved inquiries, audits, and observations.");
  }
  jobSignature = signature;
}

async function refreshAnalysis(id = "") {
  const token = ++refreshToken;
  const run = hooks.run();
  if (!run) { $("#researchHistories").innerHTML = '<p class="muted">Select a recorded society to explore its cultural history.</p>'; return; }
  let a;
  try { a = await api("/analysis?" + query({run, analysis_id:id})); }
  catch(e) { $("#researchCoverage").textContent = e.message; $("#researchHistories").innerHTML = ""; return; }
  if (token !== refreshToken) return;
  analysis = a;
  currentRun = run;
  const c = a.coverage;
  $("#researchCoverage").innerHTML = `${a.synthetic ? '<p class="research-notice">MOCK EXECUTION · No semantic judgments were performed.</p>' : ""}<p><b>${esc(a.status)}</b> · ${c.processed_events}/${c.eligible_events} eligible events observed · through tick ${c.through_tick} · ${esc(a.observer.model)}</p><div class="research-controls"><a target="_blank" href="${fileURL(`${run}/analyses/${a.analysis_id}/report.html`)}">Portable report / print PDF ↗</a><a href="${fileURL(`${run}/analyses/${a.analysis_id}/report.md`)}">Markdown</a><a href="${fileURL(`${run}/analyses/${a.analysis_id}/occurrences.csv`)}">Occurrences CSV</a><a href="${fileURL(`${run}/analyses/${a.analysis_id}/analysis.json`)}">Analysis JSON</a></div>`;
  renderHistoryBrowser($("#researchHistories"), a, run);
}

export function renderMemeticsCulture(a, run, callback) {
  $("#cultureHeading").textContent = "Emerging ideas and changing meanings";
  $("#cultureDescription").textContent = "Open observation of the recorded society. Every interpretation links to its source; hearing, using, and endorsing an idea remain distinct.";
  $("#onlyConv").parentElement.hidden = true;
  $("#cultureSummary").innerHTML = [["Concept histories", a.summary.n_threads], ["Public histories", a.summary.n_public_threads], ["Multi-agent histories", a.summary.n_multi_agent_threads], ["Judged changes", a.summary.n_changes]].map(([k,v]) => `<div class="stat"><div class="v">${fmt(v)}</div><div class="l">${k}</div></div>`).join("");
  $("#cards").innerHTML = "";
  $("#memeDetail").style.display = "block";
  renderHistoryBrowser($("#memeDetail"), a, run, callback);
}

function renderHistoryBrowser(root, a, run, callback) {
  if (!a.threads.length) { root.innerHTML = `<p class="muted">${a.synthetic ? "Mock runs do not perform semantic discovery. Select a live observer in Research to interpret recorded evidence." : "This observer identified no concept histories in the processed evidence."}</p>`; return; }
  root.innerHTML = `<div class="history-layout"><aside><label class="research-field">Find an idea<input class="history-search" placeholder="Label, meaning, or expression"></label><div class="history-list"></div></aside><article class="history-detail"></article></div>`;
  const list = $(".history-list", root), detail = $(".history-detail", root);
  function choose(t) {
    selected = t.id;
    $$("button", list).forEach(b => b.classList.toggle("selected", b.dataset.id === t.id));
    const m = a.measurements.threads[t.id];
    detail.innerHTML = `<h2>${esc(t.label)}</h2><p>${esc(t.description)}</p><div class="research-tags"><span>${m.public_users} public users</span><span>${m.private_occurrences} private representations</span><span>${m.senses} interpretations</span><span>observed span ${fmt(m.observed_span_days)} days</span></div>
      <h3>Circulation through time</h3>${lineChart([{name:"Public uses", values:m.daily.map(d => [d.day,d.public_occurrences])}], "Public uses per day")}
      <h3>Changing distribution of interpretations</h3><p class="muted small">Each observed agent has equal weight. Gaps mean no observed uses; colors are observer-distinguished senses.</p>${lineChart(Object.entries(t.senses).map(([s,name]) => ({name, values:m.daily.map(d => [d.day,d.observed_agents ? (d.collective_distribution[s] || 0) : null])})), "Share among observed agents", true)}
      <details open><summary>Individual interpretations over time</summary>${heatmap(t,m)}</details>
      <details><summary>Possible transmission paths (${m.possible_transmission.length})</summary><p class="muted small">Logged exposure preceded first public use. These links are possible sources, not proof of influence or endorsement.</p>${network(m)}<div class="research-table"><table class="table"><tr><th>Source → user</th><th>Route</th><th>Reuse tick</th><th>Evidence</th></tr>${m.possible_transmission.map(e => `<tr><td>${esc(e.source)} → ${esc(e.target)}</td><td>${esc(e.via)}${e.same_exchange ? " · same exchange" : " · later exchange"}</td><td>${e.reuse_tick}</td><td>${evidenceButton(e.source_event)} ${evidenceButton(e.target_event)}</td></tr>`).join("")}</table></div></details>
      <h3>Changes and alternative interpretations</h3>${m.changes.map(c => `<article class="research-change"><p>${esc(c.description)}</p><p class="muted">${esc(c.uncertainty)}</p>${[...c.before,...c.after].map(evidenceButton).join(" ")}</article>`).join("") || '<p class="muted">No explicit change was judged for this history.</p>'}
      <details><summary>Relationships to other ideas</summary>${a.relationships.filter(r => r.source === t.id || r.target === t.id).map(r => `<p>${esc(r.relation)} · ${esc(r.uncertainty)} ${r.evidence_ids.map(evidenceButton).join(" ")}</p>`).join("") || '<p class="muted">No relationships recorded.</p>'}</details>
      <h3>Contextual uses (${t.occurrences.length})</h3><div class="research-scroll">${t.occurrences.map(o => `<article class="research-occurrence"><div class="muted small">Day ${o.day} · ${esc(o.actor)} · ${esc(o.channel)}</div><blockquote>${esc(o.quote)}</blockquote><p>${esc(o.interpretation)}</p><p class="muted small">${esc(o.stance)} · ${esc(o.function)} ${esc(o.uncertainty)}</p>${evidenceButton(o.event_id)} <button class="pill replay-event" data-tick="${o.tick}">Replay this moment</button></article>`).join("")}</div>`;
    $$(".evidence-link", detail).forEach(b => b.onclick = guarded(() => showEvidence(run,b.dataset.event)));
    $$(".replay-event", detail).forEach(b => b.onclick = () => (callback || hooks?.replay)?.(+b.dataset.tick));
  }
  function filter() {
    const q = $(".history-search",root).value.toLowerCase();
    const ts = a.threads.filter(t => JSON.stringify(t).toLowerCase().includes(q));
    list.innerHTML = ts.map(t => `<button class="history-item ${selected === t.id ? "selected" : ""}" data-id="${esc(t.id)}"><b>${esc(t.label)}</b><span>${a.measurements.threads[t.id].public_users} public users · ${t.occurrences.length} observed uses</span></button>`).join("");
    $$("button",list).forEach(b => b.onclick = () => choose(a.threads.find(t => t.id === b.dataset.id)));
    if (!ts.length) detail.innerHTML = '<p class="muted">No histories match this search.</p>';
    else choose(ts.find(t => t.id === selected) || ts[0]);
  }
  $(".history-search",root).oninput = filter;
  filter();
}

const evidenceButton = (id) => `<button class="pill evidence-link" data-event="${esc(id)}">Source ${esc(id)}</button>`;
async function showEvidence(run, id) {
  const rows = await api("/evidence?" + query({run,event_id:id}));
  const e = rows[0];
  if (!e) throw new Error("This source is outside the committed evidence index");
  $("#modalContent").innerHTML = `<h2>Original evidence</h2><p class="muted">${esc(e.id)} · day ${e.day} · tick ${e.tick} · ${esc(e.actor)} · ${esc(e.channel)}</p><blockquote class="research-source">${esc(e.text)}</blockquote><p>Listeners: ${esc(e.listeners.join(", ") || "No recorded listeners")}</p><pre>${esc(JSON.stringify({kind:e.kind,conversation:e.conversation,turn:e.turn,source_ids:e.source_ids}, null, 2))}</pre><button class="pill" id="evidenceReplay">Replay this moment</button>`;
  $("#modal").hidden = false;
  $("#evidenceReplay").onclick = () => { $("#modal").hidden = true; hooks?.replay(e.tick); };
}

function lineChart(series, title, normalized = false) {
  const points = series.flatMap(s => s.values).filter(p => p[1] != null), xmax = Math.max(1,...points.map(p => p[0])), ymax = normalized ? 1 : Math.max(1,...points.map(p => p[1]));
  const X = x => 45 + 650 * x / xmax, Y = y => 150 - 125 * y / ymax;
  const plots = series.map((s,i) => { let drawing = false; return `<path d="${s.values.map(([x,y]) => { if (y == null) { drawing=false; return ""; } const code=`${drawing ? "L" : "M"}${X(x)},${Y(y)}`; drawing=true; return code; }).join(" ")}" fill="none" stroke="${colors[i%colors.length]}" stroke-width="2.5"/>` + s.values.filter(p => p[1] != null).map(([x,y]) => `<circle cx="${X(x)}" cy="${Y(y)}" r="2" fill="${colors[i%colors.length]}"><title>${esc(s.name)} · day ${x}: ${fmt(y)}</title></circle>`).join(""); }).join("");
  return `<div class="research-chart"><svg viewBox="0 0 730 185" role="img" aria-label="${esc(title)}"><path d="M45 20 V150 H700" fill="none" stroke="currentColor" opacity=".3"/>${[0,.5,1].map(f => `<text x="35" y="${Y(ymax*f)+4}" text-anchor="end">${fmt(ymax*f)}</text>`).join("")}<text x="45" y="175">Day 0</text><text x="680" y="175">${xmax}</text>${plots}</svg><div class="research-legend">${series.map((s,i) => `<span><i style="background:${colors[i%colors.length]}"></i>${esc(s.name)}</span>`).join("")}</div></div>`;
}

function heatmap(t,m) {
  const senses = Object.keys(t.senses), users = [...new Set(m.daily.flatMap(d => Object.keys(d.agent_distributions)))].sort();
  return `<div class="research-table"><table class="meaning-matrix"><thead><tr><th>Agent / day</th>${m.daily.map(d => `<th>${d.day}</th>`).join("")}</tr></thead><tbody>${users.map(a => `<tr><th>${esc(a)}</th>${m.daily.map(d => { const dist = d.agent_distributions[a]; if (!dist) return '<td class="unobserved" title="No observed use">·</td>'; const entries = Object.entries(dist).sort((a,b) => b[1]-a[1]), sense=entries[0][0]; return `<td style="background:${colors[senses.indexOf(sense)%colors.length]}" title="${esc(entries.map(([s,v]) => `${t.senses[s]}: ${Math.round(v*100)}%`).join("; "))}">${senses.indexOf(sense)+1}${entries.length>1 ? "+" : ""}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div><p class="muted small">Dominant observed interpretation, numbered in legend order; + indicates multiple senses. Hover for shares. · means unobserved.</p>`;
}

function network(m) {
  const names=[...new Set(m.possible_transmission.flatMap(e => [e.source,e.target]))].sort();
  if (!names.length) return '<p class="muted">No logged exposure-to-reuse paths identified.</p>';
  const positions=Object.fromEntries(names.map((a,i) => [a,[300+210*Math.cos(i*2*Math.PI/names.length),160+115*Math.sin(i*2*Math.PI/names.length)]]));
  const pairs=[...new Set(m.possible_transmission.map(e => `${e.source}|${e.target}`))];
  return `<svg class="research-network" viewBox="0 0 600 320" role="img" aria-label="Possible exposure to reuse network"><defs><marker id="researchArrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10" fill="#8297aa"/></marker></defs>${pairs.map(p => {const [a,b]=p.split("|"),[x1,y1]=positions[a],[x2,y2]=positions[b],len=Math.hypot(x2-x1,y2-y1)||1; return `<line x1="${x1}" y1="${y1}" x2="${x2-14*(x2-x1)/len}" y2="${y2-14*(y2-y1)/len}" stroke="#8297aa" marker-end="url(#researchArrow)"/>`;}).join("")}${names.map(a => {const [x,y]=positions[a];return `<circle cx="${x}" cy="${y}" r="10" fill="#2a78d6"/><text x="${x}" y="${y+26}" text-anchor="middle">${esc(a)}</text>`;}).join("")}</svg>`;
}
