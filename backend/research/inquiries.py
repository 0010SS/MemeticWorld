"""Saved, question-driven investigations over a run's evidence and open observations."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

from backend.research.common import digest, lock, read_json, write_json
from backend.research.evidence import Evidence
from backend.research.metrics import measure
from backend.research.observer import Judge, observe, settings
from backend.research.reports import export_inquiry


def inquire(run_dir, question, *, backend=None, model=None, start=None, end=None,
            actor=None, include_private=True, limit=100, judge_backend=None):
    question = question.strip()
    if not question:
        raise ValueError("A research question is required")
    if not 1 <= int(limit) <= 1000:
        raise ValueError("Evidence limit must be between 1 and 1000")
    if start is not None and end is not None and start > end:
        raise ValueError("Start tick must not exceed end tick")
    run = Path(run_dir).resolve()
    evidence = Evidence(run)
    index = evidence.build()
    analysis = read_json(run / "analysis.json", {})
    if analysis.get("kind") != "memetics" or analysis.get("input", {}).get("fingerprint") != index["fingerprint"]:
        analysis = observe(run, backend, model, judge_backend=judge_backend)
    spec = settings(evidence.cfg, backend, model)
    scope = {"start": start, "end": end, "actor": actor,
             "channels": ["speech", "record", "environment", "initial"] + (["private"] if include_private else [])}
    request = {"question": question, "scope": scope, "limit": limit, "analysis_id": analysis["analysis_id"],
               "observer": spec, "input_fingerprint": index["fingerprint"]}
    iid = "q_" + digest(request)[:20]
    directory = run / "inquiries" / iid
    with lock(directory / ".inquiry.lock"):
        if cached := read_json(directory / "results.json"):
            return cached
        write_json(directory / "question.json", request)
        scoped = {e["id"]: e for e in evidence.events(text_only=True, **scope)}
        threads = []
        for original in analysis["threads"]:
            t = copy.deepcopy(original)
            t["occurrences"] = [o for o in t["occurrences"] if o["event_id"] in scoped]
            t["senses"] = {s: meaning for s, meaning in t["senses"].items()
                           if any(o["sense"] == s for o in t["occurrences"])}
            if t["occurrences"]:
                threads.append(t)
        judge = Judge(spec, directory / "judge_calls.jsonl", backend=judge_backend)
        try:
            plan = judge.ask("plan", {"question": question, "scope": scope,
                                    "threads": [{"id": t["id"], "label": t["label"],
                                                 "interpretations": list(t["senses"].values())} for t in threads]})
            queries = [question] + [str(q) for q in plan.get("searches", [])][:12]
            selected = set(plan.get("thread_ids") or [])
            pools = [evidence.search(q, limit=limit, **scope) for q in queries]
            anchors = []
            for t in threads:
                if t["id"] in selected:
                    anchors.extend(scoped[o["event_id"]] for o in t["occurrences"])
            if anchors:
                pools.insert(0, anchors)
            # Round-robin preserves expanded queries and concept anchors within the budget.
            chosen = {}
            for group in itertools.zip_longest(*pools):
                for e in group:
                    if e is not None:
                        chosen.setdefault(e["id"], e)
                    if len(chosen) >= limit:
                        break
                if len(chosen) >= limit:
                    break
            rows = list(chosen.values())
            selected_ids = {e["id"] for e in rows}
            related = [t for t in threads if t["id"] in selected or any(o["event_id"] in selected_ids for o in t["occurrences"])]
            changes = [c for c in analysis["changes"] if set(c["before"] + c["after"]) <= set(scoped)]
            measurements = measure(related, changes, evidence)
            plan.update({"scope": scope, "searches_used": queries, "eligible_events": len(scoped),
                          "supplied_events": len(rows), "supplied_ids": sorted(selected_ids),
                          "temporal_scope": "Source evidence is filtered by tick. Concept identities were reconstructed using the complete analyzed history; this is retrospective interpretation, not a blinded prediction.",
                         "measurements_scope": "All in-scope annotated occurrences of the retrieved concept histories.",
                         "retrieval": "LLM-expanded searches, source text matching, and linked concept evidence."})
            payload = {"question": question, "plan": plan,
                       "evidence": [{k: v for k, v in e.items() if k != "raw"} for e in rows],
                       "interpretations": [{"id": t["id"], "label": t["label"],
                                            "occurrences": [o for o in t["occurrences"] if o["event_id"] in selected_ids]}
                                           for t in related],
                       "computed_measurements": {"summary": measurements["summary"],
                                                 "definitions": measurements["definitions"],
                                                 "notes": measurements["notes"]}}
            result = judge.ask("inquire", payload)
            if not isinstance(result.get("answer"), str) or not isinstance(result.get("claims"), list):
                raise ValueError("Inquiry response needs an answer and claims")
            for claim in result["claims"]:
                ids = claim.get("evidence_ids")
                if not isinstance(ids, list) or not ids or not set(ids) <= selected_ids:
                    raise ValueError("Inquiry claim cites missing or unsupplied evidence")
            result.update({"id": iid, "question": question, "analysis_id": analysis["analysis_id"],
                           "plan": plan, "observer": spec, "measurements": measurements,
                           "synthetic": spec["backend"] == "mock", "followups": result.get("followups", [])})
            write_json(directory / "analysis_plan.json", plan)
            write_json(directory / "calculations.json", measurements)
            export_inquiry(directory, result, [{k: v for k, v in e.items() if k != "raw"} for e in rows])
            return result
        finally:
            judge.close()
