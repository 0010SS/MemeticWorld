"""OBSERVER pipeline: utterances -> candidates -> grouping -> adoption ->
transmission graph -> semantics -> lineage -> probes -> emergence / grounding / funnel
(ontology v2 §4) -> status / tier (tiers.py) -> legacy ground-truth evaluation.

Per candidate, analysis.json carries `status` / `flags` (planted | system_wording | world_wording | emerged |
spreading | echo | new), `tier` / `tier_reasons` (candidate -> spreading -> convention; a convention needs a
REAL judge verdict: mock verdicts are placeholders), `wording` (system match) and `lifecycle` (the pre-tier
status: established | spreading | emerging | fading; the UI card's `status` keeps it).

Writes runs/<id>/analysis.json and runs/<id>/outcomes.json. Reads the run directory only; nothing
here can reach an agent (the simulation is finished, and probe answers are only written to
analysis.json). Uses its own LLM log: analysis_llm_calls.jsonl; the convention classifier runs through the
default judge (judge.py; LLM judges log to judge_llm_calls.jsonl). The observer LLM is fixed by
`analysis.observer` (or an explicit override) and recorded, so conditions are never compared under
different observers.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from backend import ga_compat
from backend.analysis import candidates as C
from backend.analysis import outcomes as O
from backend.analysis import tiers as T
from backend.analysis.emergence import analyze_emergence
from backend.analysis.evaluation import base_rates, evaluate, status
from backend.analysis.funnel import diagnose
from backend.analysis.grounding import analyze_grounding
from backend.analysis.judge import get_judge
from backend.analysis.judge import pipeline_spec as pipeline_judge_spec
from backend.analysis.lineage import candidate_lineage, centroid, variant_tree
from backend.analysis.rundata import RunData
from backend.analysis.semantics import analyze_semantics
from backend.analysis.transmission import analyze_transmission
from backend.llm.client import LLMClient, llm_scope, make_backend
from backend.llm.embeddings import make_embedder

LEGACY_NOTE = ("`evaluation`, `latent_type_base_rates` and card.latent_alignment are the pre-v2 retrieval-linked "
               "alignment (kept for the UI and old comparisons); use `grounding` / `emergence` / outcomes.json.")


def _retained(rd) -> dict:
    """(agent, utterance_id) -> text of the memory the agent encoded from hearing it."""
    out = {}
    for r in rd.of("memory_encoded"):
        for uid in r.get("utterance_ids") or []:
            out[(r["agent"], uid)] = r["text"]
    return out


def analyze(run_dir: Path, llm_backend: str | None = None, probes: bool = True, top_probe: int = 5,
            verbose: bool = True, llm_model: str | None = None) -> dict:
    """llm_backend / llm_model override the observer spec; otherwise cfg analysis.observer is used,
    falling back to the agents' llm settings."""
    rd = RunData(run_dir)
    if (rd.cfg.get("analysis") or {}).get("pipeline") == "memetics":
        from backend.research.observer import observe
        return observe(run_dir, backend=llm_backend, model=llm_model)
    if rd.cfg.get("world", {}).get("mode") == "commons":
        from backend.analysis.commons import analyze_commons
        out = analyze_commons(run_dir)
        if verbose:
            print(f"[analyze] commons: {out['summary']['projects_completed']} completed projects; "
                  "semantic change not evaluated")
        return out
    acfg = rd.cfg.get("analysis", {})
    embed = make_embedder(rd.cfg.get("embedding"))
    observer = O.observer_spec(rd.cfg, llm_backend, llm_model)
    llm_cfg = dict(rd.cfg["llm"], backend=observer["backend"], model=observer["model"])
    llm = LLMClient(make_backend(llm_cfg), Path(run_dir) / "analysis_llm_calls.jsonl")
    ga_compat.load(llm, embed)

    ex = C.CandidateExtractor(rd, acfg)
    cands = ex.extract()
    if verbose:
        print(f"[analyze] {len(rd.utterances)} utterances -> {len(cands)} candidates")
    discovered, judge_info = [], None
    if acfg.get("llm_classifier", True) and rd.utterances:
        with llm_scope("analysis:discover"):
            discovered = C.llm_discover(rd, llm)
        known = {v for c in cands for v in c["variants"]}
        low_utts = [(u, " ".join(C.tokens(u["text"]))) for u in rd.utterances]
        for expr in discovered:
            e = " ".join(C.tokens(expr))
            if not e or e in known or ex.wellformed(e.split()):
                continue                      # not a reusable expression (names, frames, function-word edges ...)
            hits = [u for u, t in low_utts if e in t]
            if len({u["speaker"] for u in hits}) >= 2:
                st = {"uses": [u["id"] for u in hits], "speakers": {u["speaker"] for u in hits},
                      "surface": {expr: len(hits)}}
                extra = ex.group([(tuple(e.split()), st, {**ex.score(tuple(e.split()), st), "llm_discovered": True})])
                for c in extra:
                    c["features"]["llm_discovered"] = True
                    cands.append(c)
        cands.sort(key=lambda c: -c["score"])
        for i, c in enumerate(cands):
            c["id"] = f"m{i:02d}"
        # the classifier is the default judge (analysis.judge, else the observer's backend/model; an explicit
        # llm_backend/llm_model override also overrides the judge); LLM judges cache in judge_llm_calls.jsonl
        judge = get_judge(pipeline_judge_spec(rd.cfg, observer, llm_backend is not None or llm_model is not None),
                          run_dir=run_dir)
        judge_info = judge.describe()
        try:
            with llm_scope("analysis:classify"):
                C.llm_classify(cands, llm, judge=judge)
        finally:
            judge.close()

    retained = _retained(rd)
    rates = base_rates(rd)
    trans, sem, lin_var = {}, {}, {}
    for c in cands:
        trans[c["id"]] = analyze_transmission(c, rd, retained)
        sem[c["id"]] = analyze_semantics(c, rd, embed)
        lin_var[c["id"]] = variant_tree(c)
    for c in cands:
        sem[c["id"]]["_centroid"] = centroid(c, embed)
    lineage = candidate_lineage(cands, rd, sem)
    for s in sem.values():
        s.pop("_centroid", None)

    emergence = analyze_emergence(cands, rd)
    probe_out = {}
    if probes and acfg.get("probes", True):
        from backend.analysis.probes import load_probe_agents, probe_candidate
        agents = load_probe_agents(rd, embed)
        ranked = sorted(cands, key=lambda c: (-emergence[c["id"]]["emerged"],
                                              -bool((c.get("llm") or {}).get("is_convention")), -c["score"]))
        fams = sorted({f for f in (rd.cfg.get("latent_events") or {}).get("families") or [] if f != "E0"}) or None
        v2 = not rd.events or any(e.get("generator") == "script_v2" for e in rd.events.values())
        for c in ranked[:top_probe]:
            heard = {"heard": {l for u in c["usages"] for l in u["listeners"]}, "used": set(c["speakers"])}
            probe_out[c["id"]] = probe_candidate(c, agents, llm, heard, rd.cfg["seed"], fams, v2)
            if verbose:
                print(f"[analyze] probed '{c['canonical_form']}'")
    grounding = analyze_grounding(cands, rd, probe_out)
    diag = diagnose(rd)

    from backend.analysis.judge import analysis_verdicts, judgement_verdicts
    vidx = T.VerdictIndex(analysis_verdicts(run_dir, {"candidates": cands, "generated_at": dt.datetime.now().isoformat(
        timespec="seconds")}) + judgement_verdicts(run_dir))
    judge_state = vidx.state()
    pl = rd.planted
    pl_norm = " ".join(C.tokens(pl["phrase"])) if pl and pl.get("phrase") else None
    evals = {}
    for c in cands:
        pr = probe_out.get(c["id"], {}).get("agents")
        evals[c["id"]] = evaluate(c, rd, pr, rates)
        _classify(c, emergence[c["id"]], vidx, pl_norm)
        c["lifecycle"] = status(c, rd)
        t = trans[c["id"]]
        real = (c.get("verdict") or {}).get("real")
        c["card"] = {
            "first_used_by": rd.names.get(c["first_occurrence"]["speaker"]),
            "first_used_label": _label(rd, c["first_occurrence"]["tick"]),
            "users": len(c["speakers"]), "population": len(rd.agents), "uses": c["usage_count"],
            "transmission_depth": t["depth"], "semantic_coherence": sem[c["id"]].get("coherence"),
            "latent_alignment": evals[c["id"]]["alignment"], "status": c["lifecycle"],
            # "convention" only for the convention tier; None when no real judge has judged it (mock = placeholder)
            "is_convention": (c["tier"] == "convention") if real else None,
            "gloss": (c.get("llm") or {}).get("gloss"),
            "tier": c["tier"], "tier_reasons": c["tier_reasons"], "observer_status": c["status"],
            "emerged": emergence[c["id"]]["emerged"], "adopters": emergence[c["id"]].get("n_adopters_carried"),
        }

    conv_cands = [c for c in cands if (c.get("llm") or {}).get("is_convention")] or cands[:3]
    first_use = min((c["usages"][0] for c in conv_cands), key=lambda u: u["tick"], default=None)
    cross = [trans[c["id"]]["first_cross_group"] for c in conv_cands if trans[c["id"]]["first_cross_group"]]
    first_cross = min(cross, key=lambda e: e["first_reuse_timestamp"]) if cross else None
    outcomes = O.build(rd, cands, emergence, grounding, diag, observer,
                       judge={**(judge_info or {}), **judge_state, "real_judge": judge_state["status"] == "real"})
    out = {
        "run_id": rd.dir.name, "analysis_version": O.ANALYSIS_VERSION, "observer": observer,
        "analysis_model": observer["model"], "judge": judge_info,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "summary": {"n_utterances": len(rd.utterances), "n_conversations": len(rd.conversations),
                    "n_events": len(rd.events), "n_candidates": len(cands),
                    # real-judge verdicts only; mock verdicts are placeholders (n_llm_conventions_placeholder)
                    "n_llm_conventions": sum(1 for c in cands if (c.get("llm") or {}).get("is_convention")),
                    "n_llm_conventions_placeholder": sum(1 for c in cands if ((c.get("llm") or {}).get("placeholder") or {}).get("is_convention")),
                    "tiers": T.tier_counts(cands), "n_conventions": T.tier_counts(cands)["convention"],
                    "status_counts": {st: sum(1 for c in cands if c.get("status") == st) for st in T.STATUSES},
                    "judge_status": judge_state["status"], "judge_note": judge_state["note"],
                    "first_meme_use": first_use and {k: first_use[k] for k in ("tick", "utterance_id", "speaker", "text")},
                    "first_cross_group_transmission": first_cross,
                    "llm_discovered": discovered},
        "latent_type_base_rates": rates,
        "candidates": cands, "transmission": trans, "semantics": sem,
        "lineage": {"candidates": lineage, "variants": lin_var},
        "probes": probe_out,
        "emergence": emergence, "grounding": grounding, "funnel": diag,
        "outcomes": {k: outcomes[k] for k in ("n_candidates", "n_emerged", "max_emerged_adoption", "n_grounded",
                                              "n_conventions", "tiers")},
        "evaluation": evals, "legacy": LEGACY_NOTE,
    }
    json.dump(out, open(Path(run_dir) / "analysis.json", "w"), indent=1, default=str)
    O.write(run_dir, outcomes)
    llm.close()
    if (rd.cfg.get("workshop") or {}).get("enabled"):     # v3 block (§5.12): merged into outcomes.json
        from backend.analysis.outcomes_v3 import analyze_v3
        outcomes["v3"] = analyze_v3(run_dir, write=True, verbose=verbose)
    if verbose:
        gc = grounding["candidates"]
        for c in cands[:10]:
            e, g = emergence[c["id"]], gc[c["id"]]
            print(f"  {c['id']} {c['canonical_form']!r:40} {c['tier']:10} {c['status']:14} uses={c['usage_count']:3} users={len(c['speakers'])} "
                  f"carried={e['n_adopters_carried']} echo={e['n_echo_only']} indep={e['n_independent']} "
                  f"world={e['in_world_text']} emerged={e['emerged']} "
                  f"linked={g['linked_usages']} q={g.get('q_value')} grounded={g['grounded']}")
        print(f"[analyze] emerged={outcomes['n_emerged']} grounded={outcomes['n_grounded']} "
              f"conventions={outcomes['n_conventions']} of {outcomes['n_candidates']} candidates; observer={observer}; "
              f"judge: {judge_state['note']}")
    return out


def _classify(c: dict, e: dict, vidx: "T.VerdictIndex", planted_norm: str | None) -> None:
    """Sets c["status"], c["flags"], c["tier"], c["tier_reasons"], c["verdict"], c["planted"], c["control"]."""
    w = c.get("wording") or {}
    system = bool(w.get("system") or e.get("in_lexicon") or e.get("in_system_text"))
    world = bool(e.get("in_world_text") or ((c.get("features") or {}).get("factual_repetition") and not system))
    planted = False
    if planted_norm:
        p = f" {planted_norm} "
        planted = any(f and (f" {f} " in p or p in f" {f} ") for f in T.forms(c))
    st, flags = T.classify_status(e, world=world, planted=planted, system=system)
    real, mock = vidx.lookup(T.forms(c))
    tier, why = T.tier_of(e, system=system, world=world, planted=planted, verdict=(real or {}).get("verdict"),
                          placeholder=(mock or {}).get("verdict"))
    c.update(status=st, flags=flags, tier=tier, tier_reasons=why, planted=planted, control=planted,
             verdict=T.verdict_summary(real or mock))
    c["wording"] = dict(w, system=system, world=world)


def _label(rd, tick):
    tpd = rd.manifest["ticks_per_day"]
    t = dt.datetime.fromisoformat(rd.manifest["start"]) + dt.timedelta(
        days=tick // tpd, minutes=(tick % tpd) * rd.manifest["tick_minutes"])
    return f"Day {tick // tpd + 1}, {t.strftime('%H:%M')}"
