// Culture trends view: meme leaderboard, per-meme detail, ecology — built on GET /api/runs/{id}/trends.
// Observer-side only; follows the replay tick. Falls back to a clearly-labelled fixture if the endpoint is missing.

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// Okabe-Ito (colour-blind safe), assigned stably by meme id hash
const PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999933", "#882255"];
export const memeColor = (id) => { let h = 0; for (const c of String(id)) h = (h * 31 + c.charCodeAt(0)) >>> 0; return PAL[h % PAL.length]; };
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "—" : (+v).toFixed(d));
const pct = (v) => (v == null ? "—" : (100 * v).toFixed(1) + "%");

const C = { runId: null, tick: null, data: null, sel: null, detail: null, timer: null, req: 0, fixture: false, apiFn: null, el: null };

// Public entry: debounced; call on tab show, run load and tick change.
export function renderCultureTrends(el, { runId, tick, api }) {
  C.el = el; C.apiFn = api;
  clearTimeout(C.timer);
  C.timer = setTimeout(() => load(runId, tick), C.data ? 250 : 0);
}

async function load(runId, tick) {
  if (runId !== C.runId) { C.sel = null; C.detail = null; }
  C.runId = runId; C.tick = tick;
  const my = ++C.req;
  let data;
  try { data = await C.apiFn(`/runs/${runId}/trends?tick=${tick}&window=4&top=20`); C.fixture = false; }
  catch (e) { data = fixture(runId, tick); C.fixture = true; }
  if (my !== C.req) return;
  C.data = data;
  if (!C.sel || !data.memes.some((m) => m.id === C.sel)) C.sel = data.memes[0]?.id ?? null;
  draw();
  loadDetail();
}

async function loadDetail() {
  if (!C.sel) { C.detail = null; return; }
  const my = C.req;
  let d = null;
  if (!C.fixture) { try { d = await C.apiFn(`/runs/${C.runId}/trends/meme/${encodeURIComponent(C.sel)}?tick=${C.tick}`); } catch { d = null; } }
  if (my !== C.req) return;
  C.detail = d;
  const box = C.el.querySelector("#ctDetail"); if (box) box.innerHTML = detailHTML();
}

// ------------------------------------------------------------------ layout
function draw() {
  const D = C.data, P = D.population || {}, suf = D.sufficiency || {};
  const alive = P.n_alive_now ?? D.memes.filter((m) => m.stage !== "extinct").length;
  const spreading = D.memes.filter((m) => m.stage === "spreading").length;
  const conv = D.memes.filter((m) => m.judge?.is_convention).length;
  const tile = (k, v, hint = "") => `<div class="ct-tile"><div class="ct-k">${k}</div><div class="ct-v">${v}</div>${hint ? `<div class="ct-h">${hint}</div>` : ""}</div>`;
  const banner = [];
  if (C.fixture) banner.push(`<div class="ct-banner warn">Trends endpoint not available for this run — showing a <b>synthetic fixture</b>, not real data.</div>`);
  if (suf.ok === false) banner.push(`<div class="ct-banner">Limited data (${suf.n_utterances ?? "?"} utterances, ${suf.n_windows ?? "?"} windows): ${esc((suf.notes || []).join("; ") || "treat trends as tentative")}.</div>`);
  C.el.innerHTML = `
    ${banner.join("")}
    <div class="ct-tiles">
      ${tile("Memes alive", alive)}${tile("Spreading", spreading)}${tile("Judged conventions", conv)}
      ${tile("Mean R", fmt(P.mean_R), "R&gt;1 grows, R&lt;1 fades")}
      ${tile("Tick", D.tick ?? C.tick, `${D.window_ticks ?? 4}-tick windows`)}
    </div>
    <div class="ct-grid">
      <div class="panel ct-board"><h3>Leaderboard <span class="muted small">share of all talk, up to the current tick</span></h3>${boardHTML()}</div>
      <div class="panel ct-eco"><h3>Ecology <span class="muted small">memes per window</span></h3>${ecoHTML()}</div>
    </div>
    <div id="ctDetail" class="panel ct-detail">${detailHTML()}</div>`;
  C.el.querySelectorAll("tr[data-id]").forEach((tr) => tr.onclick = () => { C.sel = tr.dataset.id; C.detail = null; draw(); loadDetail(); });
}

function judgeHTML(j) {
  if (!j || !j.provenance || j.provenance === "none") return `<span class="muted small">no real judge yet</span>`;
  if (/mock/i.test(j.provenance)) return `<span class="chip ghost" title="placeholder verdict from the mock backend">mock placeholder</span>`;
  return `<span class="chip" title="${esc(j.gloss || "")}">${esc(j.provenance)}${j.is_convention ? " ✓" : ""}</span>${j.gloss ? `<div class="small muted ct-gloss">${esc(j.gloss)}</div>` : ""}`;
}

function boardHTML() {
  const D = C.data;
  if (!D.memes.length) return `<div class="muted">No candidate expressions yet at this tick.</div>`;
  const rows = D.memes.map((m) => {
    const col = memeColor(m.id), n = m.p_adopt?.n_exposed ?? 0;
    const rcls = m.R == null ? "" : m.R > 1 ? "grow" : "fade";
    const pa = m.p_adopt?.value == null ? "—" : `${pct(m.p_adopt.value)} <span class="muted small">[${pct(m.p_adopt.ci?.[0])}–${pct(m.p_adopt.ci?.[1])}]</span>`;
    return `<tr data-id="${esc(m.id)}" class="${m.id === C.sel ? "sel" : ""}">
      <td><span class="sw" style="background:${col}"></span><b>${esc(m.phrase)}</b></td>
      <td><span class="chip tier">${esc(m.tier ?? m.status ?? "")}</span></td>
      <td><span class="chip st-${esc(m.stage)}">${esc(m.stage)}</span></td>
      <td>${spark(m.share, col)}</td>
      <td><span class="rb ${rcls}" title="mean secondary adopters per adopter">R ${fmt(m.R, 1)}</span></td>
      <td>${(m.speakers_cum || []).at(-1) ?? 0}</td>
      <td title="adopted / exposed = ${m.p_adopt?.n_adopted ?? 0}/${n}">${pa}</td>
      <td>${judgeHTML(m.judge)}</td></tr>`;
  }).join("");
  return `<div class="ct-scroll"><table class="ct-table"><thead><tr><th>Expression</th><th>Tier</th><th>Stage</th><th>Share</th><th>R</th><th>Adopters</th><th>P(adopt | exposed)</th><th>Judge</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function spark(vals = [], col) {
  const w = 110, h = 24, n = vals.length; if (!n) return "";
  const mx = Math.max(...vals, 1e-9), x = (i) => (n === 1 ? w / 2 : (i / (n - 1)) * (w - 4) + 2), y = (v) => h - 3 - (v / mx) * (h - 6);
  const pk = vals.indexOf(Math.max(...vals));
  const d = vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return `<svg class="spark" width="${w}" height="${h}" role="img" aria-label="share over time, peak ${pct(vals[pk])}"><title>peak ${pct(vals[pk])}, now ${pct(vals.at(-1))}</title><path d="${d}" fill="none" stroke="${col}" stroke-width="1.6"/><circle cx="${x(pk)}" cy="${y(pk)}" r="2.5" fill="${col}"/></svg>`;
}

// ------------------------------------------------------------------ charts
// Simple SVG chart: series [{vals, col, kind:"line"|"bar", label}], optional hline, day bounds.
function chart({ series, labels, yLabel, h = 130, hline = null, fmtY = (v) => fmt(v, 2), xLabel = "window start tick" }) {
  const W = 560, L = 44, R = 8, T = 8, B = 26, n = labels.length; if (!n) return `<div class="muted small">No windows yet.</div>`;
  const all = series.flatMap((s) => s.vals.filter((v) => v != null)).concat(hline ?? []);
  const mx = Math.max(...all, 1e-9) * 1.08;
  const bw = (W - L - R) / n, x = (i) => L + bw * (i + 0.5), y = (v) => T + (h - T - B) * (1 - v / mx);
  let g = `<line x1="${L}" x2="${W - R}" y1="${h - B}" y2="${h - B}" class="ax"/><line x1="${L}" x2="${L}" y1="${T}" y2="${h - B}" class="ax"/>`;
  g += `<text x="${L - 4}" y="${T + 8}" class="tl" text-anchor="end">${esc(fmtY(mx / 1.08))}</text><text x="${L - 4}" y="${h - B}" class="tl" text-anchor="end">0</text>`;
  g += `<text x="10" y="${(h - B) / 2 + T}" class="tl" transform="rotate(-90 10 ${(h - B) / 2 + T})" text-anchor="middle">${esc(yLabel)}</text>`;
  for (const db of C.data.day_bounds || []) { const i = C.data.windows.findIndex((w) => w >= db); if (i > 0) g += `<line x1="${L + bw * i}" x2="${L + bw * i}" y1="${T}" y2="${h - B}" class="day"/><text x="${L + bw * i + 2}" y="${T + 8}" class="tl">day</text>`; }
  if (hline != null) g += `<line x1="${L}" x2="${W - R}" y1="${y(hline)}" y2="${y(hline)}" class="hl"/><text x="${W - R}" y="${y(hline) - 2}" class="tl" text-anchor="end">R = 1</text>`;
  for (const s of series) {
    if (s.kind === "bar") s.vals.forEach((v, i) => { if (v) g += `<rect x="${x(i) - bw * 0.35}" y="${y(v)}" width="${bw * 0.7}" height="${h - B - y(v)}" fill="${s.col}" opacity="${s.op ?? 0.45}"><title>${esc(s.label)} @${labels[i]}: ${esc(fmtY(v))}</title></rect>`; });
    else {
      const pts = s.vals.map((v, i) => (v == null ? null : [x(i), y(v), v, i]));
      let d = "", pen = false; for (const p of pts) { if (!p) { pen = false; continue; } d += `${pen ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`; pen = true; }
      g += `<path d="${d}" fill="none" stroke="${s.col}" stroke-width="2"/>`;
      g += pts.filter(Boolean).map((p) => `<circle cx="${p[0]}" cy="${p[1]}" r="2.2" fill="${s.col}"><title>${esc(s.label)} @${labels[p[3]]}: ${esc(fmtY(p[2]))}</title></circle>`).join("");
    }
  }
  const step = Math.max(1, Math.ceil(n / 8));
  labels.forEach((lb, i) => { if (i % step === 0) g += `<text x="${x(i)}" y="${h - B + 12}" class="tl" text-anchor="middle">${esc(lb)}</text>`; });
  g += `<text x="${(W + L) / 2}" y="${h - 2}" class="tl" text-anchor="middle">${esc(xLabel)}</text>`;
  return `<svg class="ct-chart" viewBox="0 0 ${W} ${h}" preserveAspectRatio="xMidYMid meet">${g}</svg>`;
}

function ecoHTML() {
  const D = C.data, P = D.population || {}, lb = D.windows || [];
  const leg = `<div class="ct-legend small"><span class="sw" style="background:#0072B2"></span>alive <span class="sw" style="background:#009E73"></span>births <span class="sw" style="background:#D55E00"></span>deaths</div>`;
  return leg + chart({ labels: lb, yLabel: "memes", h: 150, fmtY: (v) => fmt(v, 0), series: [
    { vals: P.births || [], col: "#009E73", kind: "bar", label: "births", op: 0.6 },
    { vals: (P.deaths || []).map((v) => v || 0), col: "#D55E00", kind: "bar", label: "deaths", op: 0.35 },
    { vals: P.alive || [], col: "#0072B2", kind: "line", label: "alive" }] });
}

function detailHTML() {
  const D = C.data; const m = D?.memes.find((x) => x.id === C.sel);
  if (!m) return `<div class="muted">Select an expression in the leaderboard.</div>`;
  const col = memeColor(m.id), lb = D.windows || [], P = D.population || {};
  const fu = m.first_use || {};
  const det = C.detail;
  const tree = det?.tree ? treeHTML(det.tree, det.adoptions || []) : `<div class="muted small">${C.fixture ? "Transmission tree unavailable in fixture." : "Loading…"}</div>`;
  const ctx = (det?.contexts || []).slice(0, 12).map((c) => `<li><span class="muted small">t${c.tick} ${esc(c.speaker)}</span> ${esc(c.text)}</li>`).join("");
  return `<h3><span class="sw" style="background:${col}"></span>${esc(m.phrase)} <span class="chip st-${esc(m.stage)}">${esc(m.stage)}</span> <span class="muted small">first used t${fu.tick ?? "?"} by ${esc(fu.speaker_name || fu.speaker || "?")}${m.time_to_peak_ticks != null ? ` · peak after ${m.time_to_peak_ticks} ticks` : ""}${m.half_life_ticks != null ? ` · half-life ${m.half_life_ticks} ticks` : ""}</span></h3>
    ${fu.text ? `<p class="small">“${esc(fu.text)}”</p>` : ""}
    <div class="ct-dgrid">
      <div><h4>Share of all talk</h4>${chart({ labels: lb, yLabel: "share", fmtY: pct, series: [{ vals: m.share || [], col, kind: "line", label: "share" }] })}
        <div class="ct-sub">${chart({ labels: lb, yLabel: "utterances", h: 60, fmtY: (v) => fmt(v, 0), xLabel: "talk volume (all utterances)", series: [{ vals: P.talk_volume || [], col: "#888", kind: "bar", label: "talk volume", op: 0.3 }] })}</div></div>
      <div><h4>Adopters</h4><div class="ct-legend small"><span class="sw" style="background:${col}"></span>cumulative speakers · bars: new adopters</div>${chart({ labels: lb, yLabel: "agents", fmtY: (v) => fmt(v, 0), series: [{ vals: m.new_adopters || [], col, kind: "bar", label: "new adopters" }, { vals: m.speakers_cum || [], col, kind: "line", label: "cumulative adopters" }] })}</div>
      <div><h4>Reproduction number R<sub>t</sub></h4>${chart({ labels: lb, yLabel: "R_t", hline: 1, fmtY: (v) => fmt(v, 2), series: [{ vals: m.R_t || [], col, kind: "line", label: "R_t" }] })}</div>
      <div><h4>Transmission tree</h4>${tree}</div>
    </div>
    <h4>Contexts</h4>${ctx ? `<ul class="ct-ctx">${ctx}</ul>` : `<div class="muted small">No contexts.</div>`}`;
}

function treeHTML(tree, adoptions) {
  const name = Object.fromEntries(adoptions.map((a) => [a.agent, a.name || a.agent]));
  const kids = {}; for (const e of tree.edges || []) (kids[e.from] ||= []).push(e);
  const seen = new Set();
  const node = (id, tick, depth) => {
    if (seen.has(id) || depth > 12) return ""; seen.add(id);
    const ch = (kids[id] || []).map((e) => node(e.to, e.tick, depth + 1)).join("");
    return `<li><b>${esc(name[id] || id)}</b>${tick != null ? ` <span class="muted small">t${tick}</span>` : ""}${ch ? `<ul>${ch}</ul>` : ""}</li>`;
  };
  const roots = [tree.root, ...adoptions.filter((a) => !a.source && a.agent !== tree.root).map((a) => a.agent)].filter(Boolean);
  const html = roots.map((r) => node(r, null, 0)).join("");
  return html ? `<ul class="ct-tree">${html}</ul>` : `<div class="muted small">No attributed transmissions yet.</div>`;
}

// ------------------------------------------------------------------ fixture (endpoint missing)
function fixture(runId, tick) {
  const nW = Math.max(1, Math.floor((tick ?? 0) / 4) + 1), windows = [...Array(nW)].map((_, i) => i * 4);
  const mk = (id, phrase, t0, peak) => {
    const share = windows.map((w) => (w < t0 ? 0 : Math.max(0, peak * Math.exp(-((w - t0 - 12) ** 2) / 200))));
    let cum = 0; const na = share.map((s) => (s > 0.005 ? 1 : 0)); const sc = na.map((v) => (cum += v));
    return { id, phrase, tier: "candidate", status: "candidate", stage: "spreading", share, uses: share.map((s) => Math.round(s * 40)), speakers_cum: sc, new_adopters: na,
      R_t: share.map((s) => (s ? 0.6 + s * 8 : null)), R: 1.2, p_adopt: { value: 0.3, n_exposed: 10, n_adopted: 3, ci: [0.11, 0.6] }, peak: { window: 0, share: peak },
      time_to_peak_ticks: 12, half_life_ticks: null, judge: null, first_use: { tick: t0, speaker: "a1", speaker_name: "fixture agent", text: "" } };
  };
  return { run_id: runId, tick, window_ticks: 4, windows, labels: windows.map(String), day_bounds: [],
    sufficiency: { n_utterances: 0, n_windows: nW, ok: false, notes: ["fixture"] },
    population: { alive: windows.map((_, i) => Math.min(i, 5)), births: windows.map((_, i) => (i < 5 ? 1 : 0)), deaths: windows.map(() => 0), turnover: [], talk_volume: windows.map(() => 40), n_alive_now: 2, mean_R: 1.2 },
    memes: [mk("fx1", "fixture phrase A", 0, 0.08), mk("fx2", "fixture phrase B", 8, 0.05)] };
}
