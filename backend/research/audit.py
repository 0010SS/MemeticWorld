"""Reproducible, independent audit of sampled observer interpretations."""
from __future__ import annotations

import random
from pathlib import Path

from backend.research.common import digest, read_json, write_json, lock
from backend.research.evidence import Evidence
from backend.research.observer import Judge, settings


def audit(run_dir, *, backend=None, model=None, sample=40, seed=0, judge_backend=None):
    run = Path(run_dir)
    analysis = read_json(run / "analysis.json", {})
    if analysis.get("kind") != "memetics":
        raise ValueError("Observe this run before auditing interpretations")
    if sample < 1:
        raise ValueError("Audit sample must be positive")
    evidence = Evidence(run)
    evidence.build()
    items = [{"id": o["id"], "type": "occurrence", "claim": o, "evidence_ids": [o["event_id"]]}
             for t in analysis["threads"] for o in t["occurrences"]]
    items += [{"id": "change_" + digest(c)[:16], "type": "change", "claim": c,
               "evidence_ids": c["before"] + c["after"]} for c in analysis["changes"]]
    sampled = random.Random(seed).sample(items, min(sample, len(items)))
    spec = settings(evidence.cfg, backend, model)
    request = {"analysis_id": analysis["analysis_id"], "sample": sample, "seed": seed,
               "observer": spec, "items": [i["id"] for i in sampled]}
    aid = "audit_" + digest(request)[:20]
    directory = run / "audits" / aid
    with lock(directory / ".audit.lock"):
        if result := read_json(directory / "results.json"):
            return result
        judge = Judge(spec, directory / "judge_calls.jsonl", backend=judge_backend)
        verdicts = []
        try:
            for item in sampled:
                rows = evidence.events(ids=item["evidence_ids"])
                payload = {"item": item, "evidence": [{k: v for k, v in e.items() if k != "raw"} for e in rows],
                           "instructions": "Independently assess the claimed interpretation. Return supported, contested, or insufficient, and explain alternatives. Do not assume the original observer is correct."}
                result = judge.ask("audit", payload)
                if result.get("verdict") not in ("supported", "contested", "insufficient", "mock"):
                    raise ValueError("Invalid audit verdict")
                if not isinstance(result.get("reason"), str):
                    raise ValueError("Audit needs an explanation")
                verdicts.append({"item": item, **result})
            out = {"id": aid, **request, "population": len(items), "reviewed": len(verdicts),
                   "synthetic": spec["backend"] == "mock", "verdicts": verdicts,
                   "counts": {v: sum(x["verdict"] == v for x in verdicts) for v in ("supported", "contested", "insufficient", "mock")},
                   "limits": "Agreement between observers is not ground truth. This audit estimates support for sampled positive judgments; it does not measure missed concepts or discovery recall."}
            write_json(directory / "results.json", out)
            write_json(directory / "evidence.json", evidence.events(ids={eid for item in sampled for eid in item["evidence_ids"]}))
            return out
        finally:
            judge.close()
