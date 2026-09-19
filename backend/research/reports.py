"""Portable research reports and standalone SVG/CSV exports, all derived from saved evidence."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from backend.research.common import digest, write_json
from backend.research.evidence import Evidence

STYLE = """
:root{font-family:system-ui,sans-serif;color:#202b38;background:#f5f7fa}
body{max-width:1180px;margin:auto;padding:32px}h1,h2,h3{line-height:1.2}
p,li{line-height:1.65}article,.panel{background:white;border:1px solid #dce2e9;border-radius:12px;padding:20px;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e0e5eb;vertical-align:top}
blockquote{border-left:3px solid #368482;margin:12px 0;padding:10px 18px;background:#f0f8f7;white-space:pre-wrap}
.muted{color:#596b7c}.warning{padding:14px;background:#fff1c9;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
button,select,input{font:inherit;padding:8px 12px;border:1px solid #c0cbd6;border-radius:6px}
svg{width:100%;height:auto}a{color:#086b84}summary{cursor:pointer;line-height:1.7}
code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}
@media print{body{padding:0;background:white}button,.controls{display:none}article{break-inside:avoid}}
"""


def esc(value):
    return html.escape(str(value), quote=True)


def anchor(eid):
    return "e-" + digest(eid)[:16]


def chart(series, title, ylabel="Public occurrences"):
    """A standalone, portable SVG figure with explicit axes and source-derived values."""
    w, h, left, top, bottom = 900, 300, 60, 45, 245
    all_points = [p for _, points in series for p in points if p[1] is not None]
    xmax = max((p[0] for p in all_points), default=1)
    ymax = max((p[1] for p in all_points), default=1) or 1
    X = lambda x: left + (w - left - 160) * x / max(1, xmax)
    Y = lambda y: bottom - (bottom - top) * y / ymax
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="20" y="24" font-family="sans-serif" font-size="17">{esc(title)}</text>',
            f'<path d="M{left},{top} V{bottom} H{w-145}" fill="none" stroke="#8899aa"/>']
    for i in range(5):
        value = ymax * i / 4
        body.append(f'<text x="{left-8}" y="{Y(value)+4:.1f}" text-anchor="end" font-size="11">{value:.2g}</text>')
    for i in range(min(8, xmax) + 1):
        day = round(xmax * i / max(1, min(8, xmax)))
        body.append(f'<text x="{X(day):.1f}" y="{bottom+20}" text-anchor="middle" font-size="11">{day}</text>')
    colors = ["#197d85", "#c45a40", "#7262b1", "#3f8a51", "#a17415", "#ba5390"]
    for i, (name, points) in enumerate(series):
        segments, pen = [], False
        for x, y in points:
            if y is None:
                pen = False
                continue
            segments.append(f"{'L' if pen else 'M'}{X(x):.2f},{Y(y):.2f}")
            pen = True
        path = " ".join(segments)
        color = colors[i % len(colors)]
        body.append(f'<path d="{path}" stroke="{color}" stroke-width="2.5" fill="none"/>')
        body.append(f'<text x="{w-135}" y="{50+i*18}" fill="{color}" font-size="11">{esc(name[:22])}</text>')
    body += [f'<text x="{left}" y="{h-8}" font-size="12">Simulated day · {esc(ylabel)}</text>', "</svg>"]
    return "".join(body)


def document(title, body):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{STYLE}</style><body>{body}</body></html>'


def export_run(run, analysis):
    run = Path(run)
    out = run / "analyses" / analysis["analysis_id"]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ev = Evidence(run)
    ids = {o["event_id"] for t in analysis["threads"] for o in t["occurrences"]}
    for o in analysis["observations"]:
        ids.update(o["evidence_ids"])
    for r in analysis["relationships"]:
        ids.update(r["evidence_ids"])
    for c in analysis["changes"]:
        ids.update(c["before"] + c["after"])
    evidence = {e["id"]: {k: v for k, v in e.items() if k != "raw"} for e in ev.events(ids=ids)}
    write_json(out / "evidence.json", evidence)
    title = f"Cultural history: {analysis['run_id']}"
    coverage = analysis["coverage"]
    body = [f"<h1>{esc(title)}</h1><p class='muted'>Analysis {esc(analysis['analysis_id'])} · {esc(analysis['status'])}</p>",
            "<button onclick='window.print()'>Print / save PDF</button>",
            f"<p>Observer processed {coverage['processed_events']} of {coverage['eligible_events']} eligible events "
            f"through tick {coverage['through_tick']}. Descriptions and concept boundaries are revisable observer interpretations.</p>"]
    md = [f"# {title}", "", f"Analysis: {analysis['analysis_id']} ({analysis['status']}).",
          f"Coverage: {coverage['processed_events']}/{coverage['eligible_events']} eligible events.", ""]
    if analysis.get("synthetic"):
        msg = "Mock software verification. No semantic LLM judgments were performed; this is not evidence of cultural outcomes."
        body.append(f"<p class='warning'>{msg}</p>")
        md += [msg, ""]
    if analysis.get("error"):
        body.append(f"<p class='warning'>{esc(analysis['error'])}</p>")
    body.append("<div class='panel'><h2>Measurements</h2><table><tr><th>Measurement</th><th>Value</th><th>Definition</th></tr>")
    for name, value in analysis["summary"].items():
        shown = "Not observed / insufficient evidence" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)
        definition = analysis["measurements"]["definitions"].get(name, "")
        body.append(f"<tr><td>{esc(name.replace('_', ' '))}</td><td>{shown}</td><td>{esc(definition)}</td></tr>")
        md.append(f"- **{name}**: {shown}. {definition}")
    body.append("</table></div>")
    series = [(t["label"], [(d["day"], d["public_occurrences"]) for d in
              analysis["measurements"]["threads"][t["id"]]["daily"]]) for t in analysis["threads"][:6]]
    svg = chart(series, "Observed public use over time")
    (figures / "public_use.svg").write_text(svg, encoding="utf-8")
    body.append("<div class='panel'>" + svg + "<p class='muted'>Overview shows the first six histories in chronological order. Every history is included below.</p></div>")
    md += ["", "![Observed public use](figures/public_use.svg)", "", "## Concept histories", ""]
    if not analysis["threads"]:
        body.append("<p>No concept histories were identified by this observer in the processed evidence.</p>")
    body.append("<label class='controls'>Find a history <input id='find' placeholder='Label or interpretation'></label>")
    for t in analysis["threads"]:
        measurements = analysis["measurements"]["threads"][t["id"]]
        body.append(f"<article class='history' id='{esc(t['id'])}'><h2>{esc(t['label'])}</h2><p>{esc(t['description'])}</p>")
        body.append(f"<p class='muted'>{measurements['public_users']} public users · {measurements['senses']} interpretations · "
                    f"origin evidence: <a href='#{anchor(measurements['origin_evidence'])}'>{esc(measurements['origin_type'])}</a></p>")
        sense_series = [(desc, [(d["day"], d["collective_distribution"].get(s, 0) if d["observed_agents"] else None) for d in measurements["daily"]])
                        for s, desc in t["senses"].items()]
        body.append(chart(sense_series[:6], "Agent-weighted observed interpretations", "Share among observed agents"))
        body.append("<details><summary>Agent interpretations, exposure paths and daily evidence coverage</summary><pre>" +
                    esc(json.dumps({k: measurements[k] for k in ("daily", "uptake", "possible_transmission")}, indent=2)) + "</pre></details>")
        md += [f"### {t['label']}", "", t["description"], ""]
        for o in t["occurrences"]:
            body.append(f"<details><summary>Day {o['day']} · {esc(o['actor'])} · {esc(o['channel'])}: {esc(o['interpretation'])}</summary>"
                        f"<blockquote>{esc(o['quote'])}</blockquote><p>{esc(o['stance'])} · {esc(o['function'])}</p>"
                        f"<p class='muted'>{esc(o['uncertainty'])}</p><a href='#{anchor(o['event_id'])}'>Original evidence</a></details>")
            md += [f"- Day {o['day']}, {o['actor']} ({o['channel']}): {o['interpretation']} "
                   f"[Evidence](report.html#{anchor(o['event_id'])})."]
        body.append("</article>")
    body.append("<div class='panel'><h2>Changes, relationships and other observations</h2>")
    for label, rows in (("Changes", analysis["changes"]), ("Relationships", analysis["relationships"]),
                        ("Observations", analysis["observations"])):
        body.append(f"<h3>{label}</h3>")
        md += ["", f"## {label}", ""]
        for row in rows:
            text = row.get("description") or row.get("relation")
            sources = row.get("evidence_ids") or row.get("before", []) + row.get("after", [])
            links = " ".join(f"<a href='#{anchor(e)}'>{esc(e)}</a>" for e in sources)
            body.append(f"<p>{esc(text)} <span class='muted'>{esc(row.get('uncertainty', ''))}</span><br>{links}</p>")
            md.append(f"- {text} " + " ".join(f"[{e}](report.html#{anchor(e)})" for e in sources))
    body.append("</div><div class='panel'><h2>Source evidence</h2>")
    for eid, e in evidence.items():
        body.append(f"<details id='{anchor(eid)}'><summary>{esc(eid)} · day {e['day']} · {esc(e['actor'])} · {esc(e['channel'])}</summary>"
                    f"<blockquote>{esc(e['text'])}</blockquote></details>")
    body.append("</div><div class='panel'><h2>Methods and provenance</h2>")
    body.append("<details><summary>Realized activity and manipulation checks</summary><pre>" +
                esc(json.dumps(analysis.get("activity", {}), indent=2)) + "</pre></details>")
    for note in analysis["measurements"]["notes"]:
        body.append(f"<p>{esc(note)}</p>")
    body.append("<pre>" + esc(json.dumps({"observer": analysis["observer"], "input": analysis["input"]}, indent=2)) + "</pre></div>")
    body.append("""<script>document.getElementById('find').addEventListener('input',e=>{
      const q=e.target.value.toLowerCase();document.querySelectorAll('.history').forEach(a=>a.hidden=!a.textContent.toLowerCase().includes(q));
    });window.addEventListener('beforeprint',()=>document.querySelectorAll('.history').forEach(a=>a.hidden=false));</script>""")
    (out / "report.html").write_text(document(title, "".join(body)), encoding="utf-8")
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (out / "occurrences.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["thread", "event_id", "actor", "day", "tick", "channel", "quote", "interpretation", "sense", "stance", "function", "uncertainty"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for t in analysis["threads"]:
            writer.writerows(t["occurrences"])
    return out


def export_inquiry(directory, result, evidence):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = [f"<h1>{esc(result['question'])}</h1>", f"<p>{esc(result['answer'])}</p>"]
    if result.get("synthetic"):
        body.append("<p class='warning'>Mock software execution. No semantic judgment was performed.</p>")
    md = ["# " + result["question"], "", result["answer"], ""]
    by_id = {e["id"]: e for e in evidence}
    for claim in result["claims"]:
        body.append("<article><p>" + esc(claim["description"]) + "</p><p class='muted'>" + esc(claim.get("uncertainty", "")) + "</p>")
        md.append("- " + claim["description"])
        for eid in claim["evidence_ids"]:
            body.append(f"<details><summary>{esc(eid)}</summary><blockquote>{esc(by_id[eid]['text'])}</blockquote></details>")
        body.append("</article>")
    body.append("<h2>Computed measurements</h2><pre>" + esc(json.dumps(result.get("measurements"), indent=2)) + "</pre>")
    body.append("<h2>Further questions</h2><ul>" + "".join("<li>" + esc(q) + "</li>" for q in result["followups"]) + "</ul>")
    body.append("<details><summary>Analysis plan and coverage</summary><pre>" + esc(json.dumps(result["plan"], indent=2)) + "</pre></details>")
    write_json(directory / "evidence.json", evidence)
    write_json(directory / "results.json", result)
    (directory / "report.html").write_text(document(result["question"], "".join(body)), encoding="utf-8")
    (directory / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
