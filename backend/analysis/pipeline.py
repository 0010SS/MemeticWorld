"""OBSERVER pipeline: utterances -> candidates -> grouping -> adoption ->
transmission graph -> semantics -> lineage -> probes -> ground-truth evaluation.

Writes runs/<id>/analysis.json. Reads the run directory only; nothing here can
reach an agent (the simulation is finished, and probe answers are only written
to analysis.json). Uses its own LLM log: analysis_llm_calls.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from backend import ga_compat
from backend.analysis import candidates as C
from backend.analysis.evaluation import base_rates, evaluate, status
from backend.analysis.lineage import candidate_lineage, centroid, variant_tree
from backend.analysis.rundata import RunData
from backend.analysis.semantics import analyze_semantics
from backend.analysis.transmission import analyze_transmission
from backend.llm.client import LLMClient, llm_scope, make_backend
from backend.llm.embeddings import make_embedder


def _retained(rd) -> dict:
    """(agent, utterance_id) -> text of the memory the agent encoded from hearing it."""
    out = {}
    for r in rd.of("memory_encoded"):
        for uid in r.get("utterance_ids") or []:
            out[(r["agent"], uid)] = r["text"]
    return out


def analyze(run_dir: Path, llm_backend: str | None = None, probes: bool = True, top_probe: int = 5,
            verbose: bool = True, llm_model: str | None = None) -> dict:
    rd = RunData(run_dir)
    if rd.cfg.get("world", {}).get("mode") == "commons":
        from backend.analysis.commons import analyze_commons
        out = analyze_commons(run_dir)
        if verbose:
            print(f"[analyze] commons: {out['summary']['projects_completed']} completed projects; "
                  "semantic change not evaluated")
        return out
    acfg = rd.cfg.get("analysis", {})
    embed = make_embedder(rd.cfg.get("embedding"))
    llm_cfg = dict(rd.cfg["llm"])
    if llm_backend:
        llm_cfg["backend"] = llm_backend
    if llm_model:  # hold the observer model fixed across conditions that vary the agents' model
        llm_cfg["model"] = llm_model
    llm = LLMClient(make_backend(llm_cfg), Path(run_dir) / "analysis_llm_calls.jsonl")
    ga_compat.load(llm, embed)

    ex = C.CandidateExtractor(rd, acfg)
    cands = ex.extract()
    if verbose:
        print(f"[analyze] {len(rd.utterances)} utterances -> {len(cands)} candidates")
    discovered = []
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
        with llm_scope("analysis:classify"):
            C.llm_classify(cands, llm)

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

    probe_out = {}
    if probes and acfg.get("probes", True):
        from backend.analysis.probes import load_probe_agents, probe_candidate
        agents = load_probe_agents(rd, embed)
        ranked = sorted(cands, key=lambda c: (-(c.get("llm", {}).get("is_convention", False)), -c["score"]))
        for c in ranked[:top_probe]:
            heard = {"heard": {l for u in c["usages"] for l in u["listeners"]}, "used": set(c["speakers"])}
            probe_out[c["id"]] = probe_candidate(c, agents, llm, heard, rd.cfg["seed"])
            if verbose:
                print(f"[analyze] probed '{c['canonical_form']}'")

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
        }

    conv_cands = [c for c in cands if c.get("llm", {}).get("is_convention")] or cands[:3]
    first_use = min((c["usages"][0] for c in conv_cands), key=lambda u: u["tick"], default=None)
    cross = [trans[c["id"]]["first_cross_group"] for c in conv_cands if trans[c["id"]]["first_cross_group"]]
    first_cross = min(cross, key=lambda e: e["first_reuse_timestamp"]) if cross else None
    out = {
        "run_id": rd.dir.name, "analysis_model": llm_cfg.get("model"), "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "summary": {"n_utterances": len(rd.utterances), "n_conversations": len(rd.conversations),
                    "n_events": len(rd.events), "n_candidates": len(cands),
                    "n_llm_conventions": sum(1 for c in cands if c.get("llm", {}).get("is_convention")),
                    "first_meme_use": first_use and {k: first_use[k] for k in ("tick", "utterance_id", "speaker", "text")},
                    "first_cross_group_transmission": first_cross,
                    "llm_discovered": discovered},
        "latent_type_base_rates": rates,
        "candidates": cands, "transmission": trans, "semantics": sem,
        "lineage": {"candidates": lineage, "variants": lin_var},
        "probes": probe_out, "evaluation": evals,
    }
    json.dump(out, open(Path(run_dir) / "analysis.json", "w"), indent=1, default=str)
    llm.close()
    if verbose:
        for c in cands[:10]:
            print(f"  {c['id']} {c['canonical_form']!r:40} uses={c['usage_count']:3} users={len(c['speakers'])} "
                  f"conv={c.get('llm', {}).get('is_convention')} align={evals[c['id']]['alignment']} "
                  f"best={evals[c['id']]['best_type']}")
    return out


def _label(rd, tick):
    tpd = rd.manifest["ticks_per_day"]
    t = dt.datetime.fromisoformat(rd.manifest["start"]) + dt.timedelta(
        days=tick // tpd, minutes=(tick % tpd) * rd.manifest["tick_minutes"])
    return f"Day {tick // tpd + 1}, {t.strftime('%H:%M')}"
