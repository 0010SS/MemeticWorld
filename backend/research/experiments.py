"""Experiment library and reproducible comparisons of independently generated societies."""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import yaml

from backend.config import REPO_ROOT
from backend.experiment import design as D
from backend.research.common import digest, read_json, write_json
from backend.research.reports import document, esc, export_inquiry


def catalog():
    return yaml.safe_load((REPO_ROOT / "configs" / "research" / "experiments.yaml").read_text(encoding="utf-8"))["experiments"]


def resolve(name, runs_root=None):
    choices = {e["id"]: e for e in catalog()}
    if name in choices:
        path = choices[name]["design"]
    else:
        path = name
    return D.load_design(path, runs_root=runs_root)


def interval(values, seed=0, samples=2000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [draws[int(0.025 * samples)], draws[min(samples - 1, int(0.975 * samples))]]


def report(design, backend_override=None):
    cells = D.expand(design, backend_override)
    rows = []
    for cell in cells:
        out = read_json(cell.run_dir / "outcomes.json", {})
        status = D.cell_status(cell)
        valid = status == "analyzed" and out.get("status") == "complete" and out.get("observer", {}).get("analysis_version", "").startswith("memetics")
        row = {"cell": cell.cell_id, "seed": cell.seed, "replica": getattr(cell, "replica", 0),
               "levels": cell.levels, "run_dir": str(cell.run_dir), "status": status,
               "analysis_id": out.get("analysis_id"), "synthetic": out.get("synthetic"),
               "observer": out.get("observer"), "metrics": out.get("memetics", {}) if valid else {}}
        rows.append(row)
    # Only combine judgments made under one observer specification.
    observer_ids = {digest(r["observer"]) for r in rows if r["metrics"]}
    if len(observer_ids) > 1:
        raise ValueError("Experiment outputs use different observers; reanalyze consistently before comparison")
    metrics = [o.removeprefix("memetics.") for o in design.outcomes]
    names = list(dict.fromkeys(c.cell_id for c in cells))
    blocks, summaries = {}, {}
    for cell in names:
        summaries[cell] = {}
        mine = [r for r in rows if r["cell"] == cell]
        for metric in metrics:
            by_seed = defaultdict(list)
            for row in mine:
                value = row["metrics"].get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    by_seed[row["seed"]].append(value)
            vals = {s: mean(v) for s, v in by_seed.items() if len(v) == getattr(design, "replicates", 1)}
            blocks[(cell, metric)] = vals
            summaries[cell][metric] = {"mean": mean(vals.values()) if vals else None,
                                       "world_blocks": len(vals), "ci95": interval(list(vals.values())),
                                       "values": vals}
    contrasts = []
    levels = {c.cell_id: c.levels for c in cells}
    for a, b in itertools.combinations(names, 2):
        changed = [f for f in set(levels[a]) | set(levels[b]) if levels[a].get(f) != levels[b].get(f)]
        if len(changed) != 1:
            continue
        for metric in metrics:
            aa, bb = blocks[(a, metric)], blocks[(b, metric)]
            paired = sorted(set(aa) & set(bb))
            differences = [bb[s] - aa[s] for s in paired]
            contrasts.append({"a": a, "b": b, "factor": changed[0], "metric": metric,
                              "world_blocks": len(paired), "difference_b_minus_a": mean(differences) if differences else None,
                              "ci95": interval(differences), "differences": dict(zip(paired, differences))})
    root = D.design_dir(design, backend_override)
    inquiries = []
    for path in sorted((root / "inquiries").glob("*/results.json")):
        q = read_json(path)
        if q:
            inquiries.append({"id": q["id"], "question": q["question"], "answer": q["answer"],
                              "report": str(path.parent / "report.html")})
    payload = {"name": design.name, "questions": getattr(design, "questions", []), "design_sha256": design.sha256,
               "inquiries": inquiries,
               "rows": rows, "summaries": summaries, "contrasts": contrasts,
               "methods": "Agent replicas are averaged within each world-seed block. Intervals bootstrap independent world blocks (2000 draws). Paired contrasts change one factor. Intervals are exploratory and unadjusted for multiple comparisons. Missing replicas are excluded as incomplete blocks; their run statuses remain visible.",
               "synthetic": any(r["synthetic"] for r in rows)}
    rid = "r_" + digest(payload)[:20]
    output = root / "reports" / rid
    write_json(output / "results.json", payload)
    write_json(root / "report.json", {**payload, "report_id": rid})
    body = [f"<h1>{esc(design.name)}</h1>"]
    md = [f"# {design.name}", "", *getattr(design, "questions", []), "", payload["methods"], ""]
    for q in getattr(design, "questions", []):
        body.append(f"<p>{esc(q)}</p>")
    if payload["synthetic"]:
        body.append("<p class='warning'>Mock software execution. These results do not provide evidence about cultural phenomena.</p>")
        md += ["Mock software execution; no cultural inference.", ""]
    body.append("<p>" + esc(payload["methods"]) + "</p><h2>Run coverage</h2><table><tr><th>Condition</th><th>World / replica</th><th>Status</th><th>Recorded analysis</th></tr>")
    for row in rows:
        p = Path(row["run_dir"])
        relative = Path("..") / ".." / p.relative_to(root) / "analyses" / str(row["analysis_id"]) / "report.html"
        link = f"<a href='{esc(relative.as_posix())}'>Evidence and cultural histories</a>" if row["analysis_id"] else "No complete analysis"
        body.append(f"<tr><td>{esc(row['cell'])}</td><td>{row['seed']} / {row['replica']}</td><td>{esc(row['status'])}</td><td>{link}</td></tr>")
    body.append("</table><h2>Research inquiries</h2>")
    for q in inquiries:
        link = (Path("..") / ".." / "inquiries" / q["id"] / "report.html").as_posix()
        body.append(f"<article><h3><a href='{esc(link)}'>{esc(q['question'])}</a></h3><p>{esc(q['answer'])}</p></article>")
        md += [f"\n## {q['question']}", q["answer"], f"[Evidence and interpretation]({link})"]
    body.append("<h2>Condition measurements</h2>")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        figure = comparison_figure(summaries, metric)
        (figures / (metric + ".svg")).write_text(figure, encoding="utf-8")
        body.append(figure)
    body.append("<table><tr><th>Condition</th><th>Measurement</th><th>Mean</th><th>95% interval</th><th>World blocks</th></tr>")
    for cell, results in summaries.items():
        for metric, value in results.items():
            body.append(f"<tr><td>{esc(cell)}</td><td>{esc(metric)}</td><td>{esc(value['mean'])}</td><td>{esc(value['ci95'])}</td><td>{value['world_blocks']}</td></tr>")
            md.append(f"- {cell}, {metric}: {value['mean']}; interval {value['ci95']}; {value['world_blocks']} world blocks.")
    body.append("</table><h2>Matched comparisons</h2><table><tr><th>A → B</th><th>Measurement</th><th>Difference</th><th>95% interval</th><th>Blocks</th></tr>")
    for c in contrasts:
        body.append(f"<tr><td>{esc(c['a'])} → {esc(c['b'])}</td><td>{esc(c['metric'])}</td><td>{esc(c['difference_b_minus_a'])}</td><td>{esc(c['ci95'])}</td><td>{c['world_blocks']}</td></tr>")
    body.append("</table>")
    (output / "report.html").write_text(document(design.name, "".join(body)), encoding="utf-8")
    (output / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (output / "runs.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        fields = ["cell", "seed", "replica", "status", "analysis_id", "synthetic"] + metrics
        writer = csv.DictWriter(fh, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, **r["metrics"]} for r in rows)
    return {**payload, "report_id": rid, "report_dir": str(output)}


def comparison_figure(summaries, metric):
    """World-block means and uncertainty, with missing cells retained as missing."""
    values = [(name, row[metric]) for name, row in summaries.items()]
    maximum = max([v["ci95"][1] if v["ci95"] else v["mean"] for _, v in values if v["mean"] is not None] or [1]) or 1
    width, height = 1000, 80 + len(values) * 32
    X = lambda value: 410 + 480 * value / maximum
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(metric)}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="15" y="25" font-family="sans-serif" font-size="16">{esc(metric)} · mean and 95% world-block interval</text>']
    for i, (name, v) in enumerate(values):
        y = 55 + i * 32
        out.append(f'<text x="15" y="{y+4}" font-size="12">{esc(name)}</text>')
        if v["mean"] is None:
            out.append(f'<text x="410" y="{y+4}" font-size="12">No complete observed blocks</text>')
            continue
        if v["ci95"]:
            a, b = v["ci95"]
            out.append(f'<path d="M{X(a):.1f},{y} H{X(b):.1f}" stroke="#197d85" stroke-width="3"/>')
        out.append(f'<circle cx="{X(v["mean"]):.1f}" cy="{y}" r="4" fill="#197d85"/><text x="910" y="{y+4}" font-size="12">n={v["world_blocks"]}</text>')
    out.append(f'<text x="410" y="{height-8}" font-size="11">0</text><text x="890" y="{height-8}" font-size="11">{maximum:.3g}</text></svg>')
    return "".join(out)


def synthesize(design, question, backend_override=None, cells=None):
    """Interpret across societies using per-society inquiries and namespaced original evidence."""
    from backend.research.inquiries import inquire
    from backend.research.observer import Judge, settings
    from backend.research.common import lock
    cells = cells if cells is not None else D.expand(design, backend_override)
    complete = [c for c in cells if D.cell_status(c) == "analyzed"]
    if not complete:
        raise ValueError("No complete, compatible society analyses are available")
    results, sources, contexts = [], {}, []
    for c in complete:
        result = inquire(c.run_dir, question, backend=backend_override)
        results.append({"run": str(c.run_dir), "inquiry": result["id"], "analysis": result["analysis_id"]})
        namespace = c.run_dir.relative_to(D.design_dir(design, backend_override)).as_posix()
        evidence = read_json(c.run_dir / "inquiries" / result["id"] / "evidence.json", [])
        by_id = {e["id"]: e for e in evidence}
        claims = []
        for claim in result["claims"]:
            ids = []
            for eid in claim["evidence_ids"]:
                nid = namespace + "::" + eid
                sources[nid] = {**by_id[eid], "id": nid, "original_id": eid, "run": namespace}
                ids.append(nid)
            claims.append({**claim, "evidence_ids": ids})
        contexts.append({"run": namespace, "condition": c.levels, "world_seed": c.seed, "replica": c.replica,
                         "answer": result["answer"], "claims": claims, "summary": result["measurements"]["summary"]})
    spec = settings(complete[0].config(), backend_override)
    request = {"question": question, "sources": results, "observer": spec, "design": design.sha256}
    iid = "q_" + digest(request)[:20]
    directory = D.design_dir(design, backend_override) / "inquiries" / iid
    with lock(directory / ".synthesis.lock"):
        if prior := read_json(directory / "results.json"):
            return {**prior, "report_dir": str(directory)}
        judge = Judge(spec, directory / "judge_calls.jsonl")
        try:
            # Bound each synthesis pass, while visiting every society. Intermediate claims must
            # continue to cite original evidence; summaries never become evidence themselves.
            batches, batch, length = [], [], 0
            for context in contexts:
                size = len(json.dumps(context))
                if batch and length + size > 45000:
                    batches.append(batch)
                    batch, length = [], 0
                batch.append(context)
                length += size
            if batch:
                batches.append(batch)
            stages = []
            for batch in batches:
                ids = {eid for c in batch for claim in c["claims"] for eid in claim["evidence_ids"]}
                packet = [{**sources[eid], "text": sources[eid]["text"][:3000]} for eid in sorted(ids)]
                answer = judge.ask("inquire", {"question": question, "societies": batch, "evidence": packet,
                    "instructions": "Compare recurring and divergent meanings. Society-specific IDs and similar labels do not establish conceptual equivalence. Consider counterexamples and missing evidence. Do not infer causality from within-run association."})
                validate_claims(answer, ids)
                stages.append(answer)
            if len(stages) == 1:
                answer = stages[0]
            else:
                ids = {eid for stage in stages for c in stage["claims"] for eid in c["evidence_ids"]}
                answer = judge.ask("inquire", {"question": question, "partial_comparisons": stages,
                    "evidence": [{**sources[eid], "text": sources[eid]["text"][:2000]} for eid in sorted(ids)],
                    "instructions": "Synthesize across all supplied comparisons, retaining disagreement and limitations; cite original evidence IDs."})
                validate_claims(answer, ids)
            answer.update({"id": iid, "question": question, "observer": spec, "synthetic": spec["backend"] == "mock",
                           "plan": {**request, "societies": len(complete), "requested_societies": len(cells),
                                    "passes": len(batches), "evidence_excerpt_chars": 3000,
                                    "scope": "All available compatible complete societies; uncompleted runs remain missing."},
                           "measurements": {c["run"]: c["summary"] for c in contexts},
                           "followups": answer.get("followups", [])})
            write_json(directory / "stages.json", stages)
            export_inquiry(directory, answer, list(sources.values()))
            return {**answer, "report_dir": str(directory)}
        finally:
            judge.close()


def validate_claims(result, allowed):
    if not isinstance(result.get("answer"), str) or not isinstance(result.get("claims"), list):
        raise ValueError("Synthesis requires an answer and claims")
    for claim in result["claims"]:
        if not claim.get("evidence_ids") or not set(claim["evidence_ids"]) <= allowed:
            raise ValueError("Synthesis cites missing original evidence")


def run(design, cells=None, *, backend_override=None, parallel=1, dry_run=False, questions=True):
    from backend.research.common import lock
    cells = cells if cells is not None else D.expand(design, backend_override)
    if not cells:
        raise ValueError("No experiment runs match the selection")
    if not 1 <= parallel <= 64:
        raise ValueError("Parallelism must be between 1 and 64")
    if dry_run:
        return D.run_design(design, cells, parallel, backend_override, dry_run=True)
    root = D.design_dir(design, backend_override)
    with lock(root / ".pipeline.lock"):
        state = {"status": "running", "stage": "simulate_and_observe", "errors": [],
                 "selected_runs": [str(c.run_dir) for c in cells], "questions": design.questions if questions else []}
        write_json(root / "pipeline.json", state)
        try:
            results = D.run_design(design, cells, parallel, backend_override)
            state["execution"] = results
            state["errors"].extend({"run": r.get("log"), "error": "Simulation or observation failed"}
                                   for r in results if r["rc"] != 0)
            state["stage"] = "interpret_questions"
            write_json(root / "pipeline.json", state)
            for question in state["questions"]:
                print("Investigating: " + question, flush=True)
                try:
                    synthesize(design, question, backend_override, cells)
                except Exception as exc:
                    state["errors"].append({"question": question, "error": f"{type(exc).__name__}: {exc}"})
                    write_json(root / "pipeline.json", state)
            result = report(design, backend_override)
            incomplete = [s for s in D.status(design, cells=cells) if s["status"] != "analyzed"]
            state.update(status="incomplete" if incomplete or state["errors"] else "complete", stage="done",
                         incomplete=incomplete, report_dir=result["report_dir"], report_id=result["report_id"])
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(root / "pipeline.json", state)
        return state
