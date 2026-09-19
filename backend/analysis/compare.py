"""Cross-condition comparison of analyzed runs (OBSERVER ONLY)."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def summarize(run_dir: Path) -> dict | None:
    run_dir = Path(run_dir)
    p = run_dir / "analysis.json"
    if not p.exists():
        return None
    a = json.load(open(p))
    man = json.load(open(run_dir / "manifest.json"))
    cands = a["candidates"]
    conv = [c for c in cands if c.get("llm", {}).get("is_convention")]
    pool = conv or cands[:3]
    ev = a.get("evaluation", {})
    tr = a.get("transmission", {})
    sem = a.get("semantics", {})
    pop = len(man["agents"])

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(mean(xs), 3) if xs else None
    top = max(pool, key=lambda c: (len(c["speakers"]), c["usage_count"]), default=None)
    return {
        "run_id": run_dir.name, "condition": man["config"].get("run_name"), "seed": man["config"]["seed"],
        "modules": man.get("modules"), "noise": man["config"]["memory"]["encoding_noise"],
        "events": a["summary"]["n_events"], "conversations": a["summary"]["n_conversations"],
        "utterances": a["summary"]["n_utterances"], "candidates": len(cands), "llm_conventions": len(conv),
        "max_adoption": round(max((len(c["speakers"]) / pop for c in pool), default=0), 3),
        "mean_depth": avg([tr[c["id"]]["depth"] for c in pool]),
        "cross_group_edges": sum(1 for c in pool for e in tr[c["id"]]["edges"] if e["cross_group"] and e["confidence"] >= 0.3),
        "mean_coherence": avg([sem[c["id"]].get("coherence") for c in pool]),
        "mean_alignment": avg([ev[c["id"]]["alignment"] for c in pool]),
        "mean_lift": avg([ev[c["id"]].get("lift") for c in pool]),
        # candidates whose latent alignment beats the label-permutation null (expect ~5% by chance)
        "n_aligned_p05": sum(1 for c in cands if (ev[c["id"]].get("permutation") or {}).get("p_value", 1) < 0.05),
        "model": man["config"]["llm"].get("model"),
        "days": man["config"].get("simulation_days"),
        "top_expression": top and top["canonical_form"], "top_users": top and len(top["speakers"]),
        "top_gloss": top and top.get("llm", {}).get("gloss"),
        "llm_calls": man["stats"]["llm"]["calls"], "llm_errors": man["stats"]["llm"]["errors"],
        "reflections": man["stats"].get("reflections"),
    }


def compare(run_dirs) -> list[dict]:
    return [s for s in (summarize(d) for d in run_dirs) if s]


def markdown(rows: list[dict]) -> str:
    cols = ["condition", "seed", "model", "events", "conversations", "utterances", "reflections", "candidates", "llm_conventions",
            "max_adoption", "mean_depth", "cross_group_edges", "mean_coherence", "mean_alignment", "mean_lift",
            "n_aligned_p05", "top_expression"]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return out
