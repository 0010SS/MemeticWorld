"""outcomes.json: the fixed per-run outcome vector (OBSERVER ONLY; ontology v2 §4).

Counts come from the n-gram candidates, emergence, grounding and the funnel, none of which depend
on the LLM classifier or the probes: candidates added by the LLM discovery pass are left out of
every count, and the classifier's verdict appears only as an annotation (`llm_is_convention`).
The observer spec is recorded so that `compare` can refuse to pool runs analysed differently.

- n_emerged counts candidates that spread by carried adoption (emergence.py) and are not world-provided
  wording: not population vocabulary (in_lexicon) and not verbatim world event text (in_world_text:
  fact texts, viewpoint renderings, referent names). Candidates that spread but are world wording are
  listed under `spread_world` and counted in n_spread_world, never in n_emerged.
- max_emerged_adoption = max over emerged candidates of (1 + carried adopters) / population.
- n_grounded is null (not 0) when grounding is not identifiable in the run (grounding.py): the
  family could not be told apart from the circle, which is missing data, not a negative result.
  `grounding` holds the identifiability diagnostics (and the grounding warning, if any).
"""
from __future__ import annotations

import json
from pathlib import Path

ANALYSIS_VERSION = "v2"


def observer_spec(cfg: dict, llm_backend: str | None = None, llm_model: str | None = None) -> dict:
    """Explicit override > cfg analysis.observer > the agents' llm settings."""
    obs = (cfg.get("analysis") or {}).get("observer") or {}
    llm = cfg.get("llm") or {}
    return {"backend": llm_backend or obs.get("backend") or llm.get("backend"),
            "model": llm_model or obs.get("model") or llm.get("model"),
            "analysis_version": ANALYSIS_VERSION}


def spec_key(spec: dict) -> tuple:
    return (spec.get("backend"), spec.get("model"), spec.get("analysis_version"))


def build(rd, cands: list[dict], emergence: dict, grounding: dict, diag: dict, observer: dict) -> dict:
    core = [c for c in cands if not c.get("features", {}).get("llm_discovered")]
    pop = max(1, len(rd.agents))
    gc = grounding.get("candidates", {})
    emerged, spread_world, grounded = [], [], []
    for c in core:
        e, g = emergence.get(c["id"], {}), gc.get(c["id"], {})
        note = c.get("llm", {}).get("is_convention")
        carried = e.get("n_adopters_carried", 0)
        rec = {"id": c["id"], "phrase": c["canonical_form"], "n_adopters_carried": carried,
               "n_adopters": e.get("n_adopters"), "n_independent": e.get("n_independent"),
               "n_exposed": e.get("n_exposed"), "fisher_p": e.get("fisher_p"),
               "adoption": round((1 + carried) / pop, 3), "in_world_text": e.get("in_world_text"),
               "in_lexicon": e.get("in_lexicon"), "grounded": bool(g.get("grounded")), "llm_is_convention": note}
        if e.get("emerged"):
            emerged.append(rec)
        elif e.get("spread") and e.get("in_world_text") and not e.get("in_lexicon"):
            spread_world.append(rec)
        if g.get("grounded"):
            grounded.append({"id": c["id"], "phrase": c["canonical_form"], "best_family": g["best_family"],
                             "share": g["share"], "null_mean": g.get("null_mean"), "p_value": g["p_value"],
                             "q_value": g["q_value"], "linked_usages": g["linked_usages"],
                             "linked_speakers": g["linked_speakers"], "z_surfaces": g.get("z_surfaces"),
                             "holdout": g.get("holdout"), "emerged": bool(e.get("emerged")),
                             "in_world_text": e.get("in_world_text"), "in_lexicon": e.get("in_lexicon"),
                             "llm_is_convention": note})
    cfg = rd.cfg
    identifiable = grounding.get("identifiable")
    return {"run_id": rd.dir.name, "condition": rd.manifest.get("condition"), "observer": observer,
            "seed": cfg.get("seed"), "world_seed": cfg.get("world_seed") if cfg.get("world_seed") is not None else cfg.get("seed"),
            "code_version": rd.manifest.get("code_version"),
            "n_candidates": len(core), "n_emerged": len(emerged),
            "max_emerged_adoption": max((x["adoption"] for x in emerged), default=0.0),
            "n_spread_world": len(spread_world),
            "n_grounded": None if identifiable is False else len(grounded),
            "n_grounding_tested": sum(1 for c in core if gc.get(c["id"], {}).get("testable")),
            # nested under "grounding" (a hidden key in API demo mode): it describes the family-circle design
            "grounding": {k: grounding.get(k) for k in ("identifiable", "family_determined_by_circle", "mixed_strata_share",
                                                        "strata_by", "n_events", "n_units", "warning")},
            "grounded": grounded, "emerged": emerged, "spread_world": spread_world,
            "funnel": diag["metrics"], "manipulation": diag["manipulation"], "validity": diag["validity"]}


def write(run_dir: Path, outcomes: dict) -> Path:
    p = Path(run_dir) / "outcomes.json"
    json.dump(outcomes, open(p, "w"), indent=1, default=str)
    return p


def load(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "outcomes.json"
    return json.load(open(p)) if p.exists() else None
