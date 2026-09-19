"""Descriptive RQ2 instruments. This observer never calls or updates an agent."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean


def analyze_commons(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "finished":
        raise ValueError("Commons analysis requires a finished, valid run")
    events = [json.loads(line) for line in (run_dir / "commons_events.jsonl").read_text(encoding="utf-8").splitlines()]
    final = json.loads((run_dir / "commons_final.json").read_text(encoding="utf-8"))
    boundary = manifest["research_design"]["change_tick"]
    deliveries = [e for e in events if e["kind"] == "delivery"]
    reads = [e for e in events if e["kind"] == "record_read"]
    writes = [e for e in events if e["kind"] == "record_written"]
    speech = [e for e in events if e["kind"] == "speech"]
    projects = list(final["projects"].values())

    def ratio(n, d):
        return round(n / d, 4) if d else None

    def performance(rows):
        success = [e for e in rows if e["passed"]]
        return {"attempts": len(rows), "successful_deliveries": len(success),
                "failed_deliveries": len(rows) - len(success),
                "success_per_attempt": ratio(len(success), len(rows)),
                "mean_completion_ticks": round(mean(e["tick"] - e["released"] for e in success), 2) if success else None,
                "on_time_deliveries": sum(e["tick"] <= e["due"] for e in success)}

    phases = {}
    for label, start, end in (("before_boundary", 0, boundary),
                              ("after_boundary", boundary, manifest["ticks"])):
        rows = [e for e in deliveries if start <= e["tick"] < end]
        phases[label] = {"start_tick": start, "end_tick_exclusive": min(end, manifest["ticks"]),
                         "observed_ticks": max(0, min(end, manifest["ticks"]) - start),
                         **performance(rows),
                         "sites": {site: performance([e for e in rows if e["site"] == site])
                                   for site in ("Library", "Quad")},
                         "projects_released": sum(start <= p["released"] < end for p in projects),
                         "record_reads": sum(start <= e["tick"] < end for e in reads)}
    newcomers = []
    for event in (e for e in events if e["kind"] == "membership"):
        aid, joined = event["arrived"], event["tick"]
        successes = [e for e in deliveries if e["actor"] == aid and e["passed"]]
        acquired = min((e["tick"] for e in successes), default=None)
        access = [e for e in reads if e["actor"] == aid]
        newcomers.append({"agent": aid, "joined_tick": joined,
                          "first_success_tick": acquired,
                          "ticks_to_first_success": acquired - joined if acquired is not None else None,
                          "observation_ticks": manifest["ticks"] - joined,
                          "no_success_before_run_end": acquired is None,
                          "successful_deliveries": len(successes),
                          "record_reads": len(access),
                          "prearrival_record_reads": sum(
                              final["records"][e["record"]]["versions"][e["version"] - 1]["tick"] < joined for e in access)})
    read_counts = Counter((e["record"], e["version"]) for e in reads)
    records = [{"record": e["record"], **e["version"],
                "reads": read_counts[(e["record"], e["version"]["version"])],
                "readers": sorted({r["actor"] for r in reads
                                   if r["record"] == e["record"] and r["version"] == e["version"]["version"]})}
               for e in writes]
    summary = {"n_utterances": len(speech), "n_conversations": 0, "n_events": len(events),
               "projects_released": len(projects),
               "projects_completed": sum(p["completed"] is not None for p in projects),
               "projects_unfinished": sum(p["completed"] is None for p in projects),
               "record_versions": len(writes), "record_reads": len(reads),
               "actions_rejected": sum(e["kind"] == "action_rejected" for e in events),
               "actions_pending_at_end": len(final["operations"]),
               **performance(deliveries)}
    result = {"mode": "commons", "run_id": run_dir.name, "analysis_model": None,
              "research_design": manifest["research_design"], "summary": summary,
              "design_executed": {"turnover_observed": any(e["kind"] == "membership" for e in events),
                                  "intervention_boundary_reached": any(e["kind"] == "intervention" for e in events)},
              "phases": phases, "newcomers": newcomers, "records": records,
              "resources_consumed": final["consumed"],
              "project_outcomes": projects,
              "record_access": [{k: e[k] for k in ("event_key", "tick", "actor", "record", "version", "author")}
                                for e in reads],
              "question_status": {
                  "RQ1": "Action and exposure histories recorded; semantic change not evaluated.",
                  "RQ2": "Descriptive inheritance/adaptation measures; replicated treatment comparisons required.",
                  "RQ3": "Different task contexts available; group interpretation not evaluated.",
                  "RQ4": "Histories preserved; matched-state counterfactuals not implemented."},
              "limitations": [
                  "Task success and record access do not establish understanding or semantic change.",
                  "Reading and writing consume time: the records factor measures availability's total effect.",
                  "Before/after summaries alone are not treatment-effect estimates.",
                  "Unfinished projects and newcomers without a success are censored at the recorded run end.",
                  "Mock-backend behavior is scripted software validation, not a research result."],
              "candidates": [], "transmission": {}, "semantics": {}, "probes": {}, "evaluation": {}}
    (run_dir / "analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
