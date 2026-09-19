"""Cross-condition comparison of analyzed runs (OBSERVER ONLY).

v2 rows come from outcomes.json (the fixed, classifier-independent outcome vector); the legacy
columns from analysis.json are kept alongside. Runs analysed with different observer specs
(backend, model, analysis_version) are not comparable -- a different observer can move every
outcome -- so compare() refuses to pool them unless explicitly allowed.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from backend.analysis import outcomes as O


class ObserverMismatch(ValueError):
    pass


def observer_of(run_dir: Path) -> dict | None:
    """The observer spec a run was analysed with (outcomes.json; legacy analysis.json -> version v1)."""
    run_dir = Path(run_dir)
    o = O.load(run_dir)
    if o:
        return o.get("observer")
    p = run_dir / "analysis.json"
    if not p.exists():
        return None
    a = json.load(open(p))
    return a.get("observer") or {"backend": None, "model": a.get("analysis_model"), "analysis_version": "v1"}


def summarize(run_dir: Path) -> dict | None:
    run_dir = Path(run_dir)
    p = run_dir / "analysis.json"
    out = O.load(run_dir)
    if not p.exists() and out is None:
        return None
    man = json.load(open(run_dir / "manifest.json"))
    row = {"run_id": run_dir.name, "condition": man["config"].get("run_name"), "seed": man["config"]["seed"],
           "modules": man.get("modules"), "noise": man["config"]["memory"]["encoding_noise"],
           "model": man["config"]["llm"].get("model"), "days": man["config"].get("simulation_days"),
           "observer": observer_of(run_dir)}
    if man.get("condition"):
        row["cell"] = man["condition"].get("cell")
        row["levels"] = man["condition"].get("levels")
    if out:
        row.update({k: out.get(k) for k in ("n_candidates", "n_emerged", "max_emerged_adoption", "n_grounded")})
        row["validity_inactive"] = sorted(k for k, v in (out.get("validity") or {}).items() if v == "inactive")
    if p.exists():
        row.update(_legacy(json.load(open(p)), man))
    return row


def _legacy(a: dict, man: dict) -> dict:
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
        "events": a["summary"]["n_events"], "conversations": a["summary"]["n_conversations"],
        "utterances": a["summary"]["n_utterances"], "candidates": len(cands), "llm_conventions": len(conv),
        "max_adoption": round(max((len(c["speakers"]) / pop for c in pool), default=0), 3),
        "mean_depth": avg([tr[c["id"]]["depth"] for c in pool]),
        "cross_group_edges": sum(1 for c in pool for e in tr[c["id"]]["edges"] if e["cross_group"] and e["confidence"] >= 0.3),
        "mean_coherence": avg([sem[c["id"]].get("coherence") for c in pool]),
        # legacy retrieval-linked alignment (see evaluation.py)
        "mean_alignment": avg([ev[c["id"]]["alignment"] for c in pool]),
        "mean_lift": avg([ev[c["id"]].get("lift") for c in pool]),
        "n_aligned_p05": sum(1 for c in cands if (ev[c["id"]].get("permutation") or {}).get("p_value", 1) < 0.05),
        "top_expression": top and top["canonical_form"], "top_users": top and len(top["speakers"]),
        "top_gloss": top and top.get("llm", {}).get("gloss"),
        "llm_calls": man["stats"]["llm"]["calls"], "llm_errors": man["stats"]["llm"]["errors"],
        "reflections": man["stats"].get("reflections"),
    }


def compare(run_dirs, allow_mixed_observers: bool = False) -> list[dict]:
    """One row per analysed run. Raises ObserverMismatch when the runs were analysed with different
    observer specs, unless allow_mixed_observers (then every row still carries its `observer`)."""
    rows = [s for s in (summarize(d) for d in run_dirs) if s]
    specs = {}
    for r in rows:
        specs.setdefault(O.spec_key(r.get("observer") or {}), []).append(r["run_id"])
    if len(specs) > 1 and not allow_mixed_observers:
        detail = "; ".join(f"backend={b} model={m} analysis={v}: {', '.join(ids)}" for (b, m, v), ids in specs.items())
        raise ObserverMismatch("runs were analysed with different observer specs and are not comparable "
                               f"({detail}). Re-analyse them with one observer (analysis.observer / analyze "
                               "--backend --model), or pass allow_mixed_observers=True.")
    return rows


def markdown(rows: list[dict]) -> str:
    cols = ["condition", "seed", "model", "n_candidates", "n_emerged", "max_emerged_adoption", "n_grounded",
            "validity_inactive", "events", "conversations", "utterances", "reflections", "llm_conventions",
            "max_adoption", "mean_depth", "cross_group_edges", "mean_coherence", "n_aligned_p05", "top_expression"]
    if any(r.get("cell") for r in rows):
        cols.insert(1, "cell")
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return out
