"""OBSERVER pipeline: utterances -> candidates -> grouping -> adoption ->
transmission graph -> semantics -> lineage -> probes -> emergence / grounding / funnel
(ontology v2 §4) -> legacy ground-truth evaluation.

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
            if not e or e in known:
                continue
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
                                              -(c.get("llm", {}).get("is_convention", False)), -c["score"]))
        fams = sorted({f for f in (rd.cfg.get("latent_events") or {}).get("families") or [] if f != "E0"}) or None
        v2 = not rd.events or any(e.get("generator") == "script_v2" for e in rd.events.values())
        for c in ranked[:top_probe]:
            heard = {"heard": {l for u in c["usages"] for l in u["listeners"]}, "used": set(c["speakers"])}
            probe_out[c["id"]] = probe_candidate(c, agents, llm, heard, rd.cfg["seed"], fams, v2)
            if verbose:
                print(f"[analyze] probed '{c['canonical_form']}'")
    grounding = analyze_grounding(cands, rd, probe_out)
    diag = diagnose(rd)

    evals = {}
    for c in cands:
        pr = probe_out.get(c["id"], {}).get("agents")
        evals[c["id"]] = evaluate(c, rd, pr, rates)
        c["status"] = status(c, rd)
        t = trans[c["id"]]
        c["card"] = {
            "first_used_by": rd.names.get(c["first_occurrence"]["speaker"]),
            "first_used_label": _label(rd, c["first_occurrence"]["tick"]),
            "users": len(c["speakers"]), "population": len(rd.agents), "uses": c["usage_count"],
            "transmission_depth": t["depth"], "semantic_coherence": sem[c["id"]].get("coherence"),
            "latent_alignment": evals[c["id"]]["alignment"], "status": c["status"],
            "is_convention": c.get("llm", {}).get("is_convention"),
            "gloss": c.get("llm", {}).get("gloss"),
            "emerged": emergence[c["id"]]["emerged"], "adopters": emergence[c["id"]].get("n_adopters_carried"),
        }

    conv_cands = [c for c in cands if c.get("llm", {}).get("is_convention")] or cands[:3]
    first_use = min((c["usages"][0] for c in conv_cands), key=lambda u: u["tick"], default=None)
    cross = [trans[c["id"]]["first_cross_group"] for c in conv_cands if trans[c["id"]]["first_cross_group"]]
    first_cross = min(cross, key=lambda e: e["first_reuse_timestamp"]) if cross else None
    outcomes = O.build(rd, cands, emergence, grounding, diag, observer)
    out = {
        "run_id": rd.dir.name, "analysis_version": O.ANALYSIS_VERSION, "observer": observer,
        "analysis_model": observer["model"], "judge": judge_info,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "summary": {"n_utterances": len(rd.utterances), "n_conversations": len(rd.conversations),
                    "n_events": len(rd.events), "n_candidates": len(cands),
                    "n_llm_conventions": sum(1 for c in cands if c.get("llm", {}).get("is_convention")),
                    "first_meme_use": first_use and {k: first_use[k] for k in ("tick", "utterance_id", "speaker", "text")},
                    "first_cross_group_transmission": first_cross,
                    "llm_discovered": discovered},
        "latent_type_base_rates": rates,
        "candidates": cands, "transmission": trans, "semantics": sem,
        "lineage": {"candidates": lineage, "variants": lin_var},
        "probes": probe_out,
        "emergence": emergence, "grounding": grounding, "funnel": diag,
        "outcomes": {k: outcomes[k] for k in ("n_candidates", "n_emerged", "max_emerged_adoption", "n_grounded")},
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
            print(f"  {c['id']} {c['canonical_form']!r:40} uses={c['usage_count']:3} users={len(c['speakers'])} "
                  f"carried={e['n_adopters_carried']} echo={e['n_echo_only']} indep={e['n_independent']} "
                  f"world={e['in_world_text']} emerged={e['emerged']} "
                  f"linked={g['linked_usages']} q={g.get('q_value')} grounded={g['grounded']}")
        print(f"[analyze] emerged={outcomes['n_emerged']} grounded={outcomes['n_grounded']} "
              f"of {outcomes['n_candidates']} candidates; observer={observer}")
    return out


def _label(rd, tick):
    tpd = rd.manifest["ticks_per_day"]
    t = dt.datetime.fromisoformat(rd.manifest["start"]) + dt.timedelta(
        days=tick // tpd, minutes=(tick % tpd) * rd.manifest["tick_minutes"])
    return f"Day {tick // tpd + 1}, {t.strftime('%H:%M')}"
