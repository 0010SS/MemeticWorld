"""The graded probe battery for the injected-meme cohort (memo §3.3, §7 RQ1). OBSERVER ONLY.

The memo is explicit about three things and this module is built around them:
  - "Meaning must be measured by interpretation and behaviour" -> an applicability judgement on matched
    situations, plus an explanation of what the copy would be DOING by saying it;
  - "Final-memory probes give weak evidence about intermediate meanings - use temporal checkpoints; probe
    isolated copies" -> every call runs against a checksummed copy of a checkpoint (runner.py's §5.2
    protocol), never the live run, and the battery is meant to run at several checkpoints;
  - "Context-similarity measures can confuse topic change with meaning change" -> the dependent variable
    here is an answer about a situation the world never produced, not an embedding of what was said.

Per (agent x meme x checkpoint) the battery asks two questions:
  form G  the registry's four gradient situations (literal / near / mid / far) PLUS that meme's foils,
          shuffled, each answered fits / doesn't fit / not sure;
  form E  what the phrase means to this copy and what it would be doing by saying it (pragmatic function),
          answered from its own decayed memory - never from a stored definition (R7).

FOILS ARE NOT OPTIONAL. They share surface features with the gradient but not its underlying structure, so
a copy that says "fits" to everything - the single most likely false positive in this study - shows up as a
high foil rate instead of as broadening. boundary.py refuses to report a cohort whose foil rate is high.

ONE PANEL, ALL MEMES. The panel is sampled once per checkpoint and answers every meme, so agreeableness is
a within-subject constant: a copy that says yes to one meme's far item and no to another's is evidence; two
different copies disagreeing is not. It is also N times cheaper than a panel per meme.

Cost is the binding constraint (6d): see `project_cost`, which is what the script prints before spending.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend import ga_compat
from backend.analysis.battery import registry as REG
from backend.analysis.battery.runner import (ProbeInvalid, Responder, _read_json, _work_agents_sums,
                                             build_agents, checkpoint_time, parse_json, prepare_checkpoint,
                                             tree_checksums, verify_unchanged)
from backend.llm.client import llm_purpose, llm_scope
from backend.simulation.rngs import seed_rng

PDIR = Path(__file__).resolve().parent / "prompts"
FITS = {"fits": 1.0, "doesn't fit": 0.0, "does not fit": 0.0, "doesnt fit": 0.0, "not sure": 0.5}
# the cohort order is fixed so that sampling is deterministic and the comparative table always reads the same
COHORTS = ("seeded", "exposed_adopter", "unexposed_adopter", "non_adopter")


def _render(template: str, inputs: list) -> str:
    ga = ga_compat.load()
    return ga.gs.generate_prompt([str(x) for x in inputs], str(PDIR / template))


# ---------------------------------------------------------------------------------------------- cohorts
def exposure_sites(rd, specs: list[REG.MemeSpec]) -> dict:
    """{agent: "exposed" | "unexposed" | "unknown"} for the RUN, not per meme.

    Exposed means the agent was at a place where a grounded meme's incident happened while it was
    happening. The split is computed once from the union of grounded incident sites and applied to all four
    memes, so "the two halls infer different things from this phrase" (local specialization, memo §2.1) is
    a comparison of the same two populations across four phrases rather than four different splits."""
    sites, days = set(), set()
    for s in specs:
        if s.location:
            sites.add(str(s.location))
            days |= set(s.incident_days)
    if not sites:
        return {}
    tpd = rd.ticks_per_day
    seen: dict[str, set] = {}
    for r in rd.of("move"):
        to = r.get("to")
        loc = to[0] if isinstance(to, (list, tuple)) and to else to
        seen.setdefault(r["agent"], set()).add((int(r["tick"]) // tpd + 1, str(loc)))
    for u in rd.utterances:                      # a conversation is presence too, and remarks log a location
        seen.setdefault(u["speaker"], set()).add((int(u["tick"]) // tpd + 1, str(u.get("location"))))
    out = {}
    for aid in sorted(rd.manifest.get("agents") or {}):
        hits = seen.get(aid, set())
        if not hits:
            out[aid] = "unknown"
        elif any(loc in sites and (not days or d in days) for d, loc in hits):
            out[aid] = "exposed"
        else:
            out[aid] = "unexposed"
    return out


def cohort_map(rd, spec: REG.MemeSpec, sites: dict, tick: int | None = None) -> dict:
    """{agent: cohort} for one meme. Non-adopters are one bucket (they are the baseline); `site` is kept
    separately by `panel_rows` so the unexposed hall can still be read on its own (6b)."""
    seeds = set(REG.seed_agents(rd, spec))
    users = {u["speaker"] for u in REG.find_usages(rd, spec, tick)}
    out = {}
    for aid in sorted(rd.manifest.get("agents") or {}):
        if aid in seeds:
            out[aid] = "seeded"
        elif aid in users:
            out[aid] = "exposed_adopter" if sites.get(aid) == "exposed" else "unexposed_adopter"
        else:
            out[aid] = "non_adopter"
    return out


def cohort_index(rd, specs: list[REG.MemeSpec], tick: int | None = None) -> dict:
    """{"sites": {agent: site}, "per_meme": {meme: {agent: cohort}}} - who counts as what, for every meme."""
    sites = exposure_sites(rd, specs)
    return {"sites": sites, "per_meme": {s.id: cohort_map(rd, s, sites, tick) for s in specs}}


def sample_panel(rd, specs: list[REG.MemeSpec], ckpt: str, *, per_cohort: int, max_agents: int,
                 available: list[str] | None = None, seed: int | None = None) -> tuple[list[str], dict]:
    """(panel, cohorts) - deterministic stratified sample. Draws `per_cohort` from each (meme, cohort) in a
    fixed order, skipping agents already drawn, until `max_agents`. The same seed and run always give the
    same panel; nothing here touches an RNG the simulation used."""
    seed = int(rd.cfg.get("seed", 0)) if seed is None else int(seed)
    cohorts = cohort_index(rd, specs)
    pool = set(available) if available is not None else set(rd.manifest.get("agents") or {})
    panel: list[str] = []
    per_meme = cohorts["per_meme"]
    for s in specs:                              # meme order is config order: a stable, declared order
        for c in COHORTS:
            members = sorted(a for a in pool if per_meme[s.id].get(a) == c and a not in panel)
            if not members:
                continue
            rng = seed_rng(seed, "graded", ckpt, s.id, c)
            take = [members[i] for i in rng.permutation(len(members))[:max(0, per_cohort)]]
            for a in sorted(take):
                if len(panel) < max_agents and a not in panel:
                    panel.append(a)
    panel.sort()
    return panel, cohorts


def panel_rows(panel: list[str], cohorts: dict) -> list[dict]:
    return [{"agent": a, "site": cohorts["sites"].get(a, "unknown"),
             "cohort": {m: c.get(a) for m, c in sorted(cohorts["per_meme"].items())}} for a in panel]


# ------------------------------------------------------------------------------------------------ items
def gradient_items(spec: REG.MemeSpec, foils_per_meme: int) -> list[dict]:
    """The meme's four gradient situations and its foils, as probe items. `kind` separates the two: a foil
    answered "fits" is a false positive, never evidence of breadth."""
    out = [{"id": f"{spec.id}.{lvl}", "meme": spec.id, "kind": "gradient", "level": lvl,
            "text": spec.gradient(lvl)} for lvl in REG.LEVELS if spec.gradient(lvl)]
    out += [{"id": f"{spec.id}.foil{i}", "meme": spec.id, "kind": "foil", "level": "foil", "text": t}
            for i, t in enumerate(spec.foils[:max(0, foils_per_meme)])]
    return out


# --------------------------------------------------------------------------------------------- the asks
class GradedSession:
    """One checkpoint's calls. Scope prefix `graded:` keeps the cache and the cost apart from LB1's."""

    def __init__(self, llm, seed: int, ckpt: str, weekday: str, k: int, raw_meta: dict):
        self.llm, self.seed, self.ckpt = llm, int(seed), ckpt
        self.weekday, self.k, self.raw_meta = weekday, int(k), raw_meta

    def _call(self, purpose: str, rid: str, scope: str, prompt: str, max_tokens: int) -> tuple[dict | None, str | None]:
        try:
            with llm_scope(scope), llm_purpose(purpose, rid):
                text = self.llm.complete(prompt, system=ga_compat.CHAT_SYSTEM, max_tokens=max_tokens, temperature=0)
        except Exception as e:  # noqa: BLE001 - an error is never recorded as an answer
            return None, f"{type(e).__name__}: {e}"[:300]
        out = parse_json(text)
        return out, None if out is not None else ("llm_error" if str(text).startswith("LLM_ERROR") else "unparsed")

    def graded(self, r: Responder, spec: REG.MemeSpec, items: list[dict]) -> list[dict]:
        parts = (self.ckpt, r.rid, spec.id, "G")
        # the focal point is the PHRASE: what we want is this copy's memory of the expression, not of
        # whichever situation we happen to be asking about
        mem, ids, hit = r.memories(spec.phrase, self.seed, parts, self.k, self.raw_meta)
        perm = seed_rng(self.seed, "graded", *parts, "order").permutation(len(items))
        shown = [items[i] for i in perm]
        sit = "\n".join(f"{j + 1}. {it['text']}" for j, it in enumerate(shown))
        prompt = _render("graded_apply_v1.txt", [r.iss, self.weekday, r.first, mem, spec.phrase, sit])
        scope = f"graded:{self.ckpt}:{r.rid}:{spec.id}:G"
        # purpose "probe_apply": this IS format A (applicability), and reusing the purpose keeps the mock
        # backend answering it, so the pipeline has a runnable smoke path without a real model
        out, err = self._call("probe_apply", r.rid, scope, prompt, 400)
        rows = []
        for j, it in enumerate(shown):
            v = str((out or {}).get(str(j + 1), "")).strip().lower().replace("’", "'")
            rows.append({"form": "G", "ckpt": self.ckpt, "agent": r.rid, "meme": spec.id, "cell": spec.cell,
                         "item": it["id"], "kind": it["kind"], "level": it["level"], "position": j + 1,
                         "answer": v or None, "fit": FITS.get(v), "retrieved": ids, "hit": hit,
                         "scope": scope, "error": err, "valid": err is None and v in FITS})
        return rows

    def explain(self, r: Responder, spec: REG.MemeSpec) -> dict:
        parts = (self.ckpt, r.rid, spec.id, "E")
        mem, ids, hit = r.memories(spec.phrase, self.seed, parts, self.k, self.raw_meta)
        prompt = _render("graded_explain_v1.txt", [r.iss, self.weekday, r.first, mem, spec.phrase])
        scope = f"graded:{self.ckpt}:{r.rid}:{spec.id}:E"
        out, err = self._call("probe_note", r.rid, scope, prompt, 300)
        return {"form": "E", "ckpt": self.ckpt, "agent": r.rid, "meme": spec.id, "cell": spec.cell,
                "item": f"{spec.id}.explain", "kind": "explain", "level": "explain",
                "meaning": (out or {}).get("meaning"), "use": (out or {}).get("use"),
                "heard_before": (out or {}).get("heard_before"), "retrieved": ids, "hit": hit,
                "scope": scope, "error": err, "valid": err is None and bool((out or {}).get("meaning"))}


# ----------------------------------------------------------------------------------------------- driver
def project_cost(rd, specs: list[REG.MemeSpec], checkpoints: list[str], bcfg: dict) -> dict:
    """What the battery will actually cost, computed rather than estimated: the number of probe calls, the
    worst case if every agent were probed, and the judge calls the explanation mix will need."""
    n_agents = len(rd.manifest.get("agents") or {})
    panel = min(int(bcfg["sample"]["max_agents"]), n_agents)
    per_agent = len(specs) * (2 if bcfg["explain"] else 1)
    probe = panel * per_agent * len(checkpoints)
    items = sum(len(gradient_items(s, bcfg["foils_per_meme"])) for s in specs)
    judge = min(int(bcfg["max_judge_calls"]), panel * len(specs) * len(checkpoints)) if bcfg["judge_explanations"] else 0
    return {"memes": len(specs), "checkpoints": len(checkpoints), "agents_in_run": n_agents,
            "panel_size": panel, "calls_per_agent_per_checkpoint": per_agent,
            "situations_per_agent_per_checkpoint": items, "probe_calls": probe, "judge_calls": judge,
            "total_llm_calls": probe + judge,
            "all_agents_probe_calls": n_agents * per_agent * len(checkpoints)}


def run_graded(run_dir, ckpt: str = "final", *, llm=None, backend=None, probes_root=None, force: bool = False,
               panel: list[str] | None = None, specs: list[REG.MemeSpec] | None = None,
               bcfg: dict | None = None) -> dict:
    """Probe isolated copies of `ckpt` with the registry battery. Writes
    `<probes_root>/graded/<ckpt>/{responses.jsonl,meta.json}` and nothing else; raises ProbeInvalid if the
    source checkpoint changed while it was being probed."""
    from backend.analysis.rundata import RunData
    run_dir = Path(run_dir)
    rd = RunData(run_dir)
    specs = specs if specs is not None else REG.load_registry(rd.cfg, only_enabled=False)
    if not specs:
        raise ValueError(f"{run_dir}: memes.registry is empty - nothing to probe")
    errs = REG.validate(specs)
    bcfg = bcfg or REG.battery_cfg(rd.cfg)
    probes_root = Path(probes_root) if probes_root else run_dir / "probes"
    out_dir = probes_root / "graded" / ckpt
    out_dir.mkdir(parents=True, exist_ok=True)

    cd = prepare_checkpoint(run_dir, ckpt, out_dir)
    when = checkpoint_time(rd.cfg, rd.manifest, cd)
    if panel is None:
        panel, cohorts = sample_panel(rd, specs, ckpt, per_cohort=bcfg["sample"]["per_cohort"],
                                      max_agents=bcfg["sample"]["max_agents"], available=cd.agent_ids)
    else:
        cohorts = cohort_index(rd, specs)
    spec_key = hashlib.sha256(json.dumps(
        {"memes": [s.raw for s in specs], "b": bcfg, "panel": sorted(panel)}, sort_keys=True, default=str
    ).encode()).hexdigest()[:16]
    prev = _read_json(out_dir / "meta.json")
    if prev and not force and prev.get("digest") == cd.digest and prev.get("spec_key") == spec_key and prev.get("valid"):
        shutil.rmtree(cd.work, ignore_errors=True)
        return dict(prev, cached=True)           # §5.2.8: an identical checkpoint is probed once

    own_llm = llm is None
    if own_llm:
        from backend.llm.client import observer_client
        llm = observer_client("probe", out_dir / "llm_calls.jsonl", rd.cfg, backend=backend)
    from backend.llm.embeddings import make_embedder
    ga_compat.load(None, make_embedder(rd.cfg.get("embedding")))

    sess = GradedSession(llm, int(rd.cfg.get("seed", 0)), ckpt, when.strftime("%A"), bcfg["k"], cd.memory_meta)
    copies = build_agents(rd.cfg, rd.manifest, cd, when, ids=[a for a in panel if a in cd.agent_ids])
    skipped = sorted(set(panel) - set(copies))
    items = {s.id: gradient_items(s, bcfg["foils_per_meme"]) for s in specs}

    def one(aid: str) -> list[dict]:
        a = copies[aid]
        # focal_extra=(): no second focal point, so retrieval is not steered toward any world artefact
        r = Responder(aid, a.iss(), a.profile.first_name, a, focal_extra=())
        rows = []
        for s in specs:
            rows += sess.graded(r, s, items[s.id])
            if bcfg["explain"]:
                rows.append(sess.explain(r, s))
        return rows

    with ThreadPoolExecutor(max_workers=max(1, int(bcfg["workers"]))) as ex:
        results = list(ex.map(one, sorted(copies)))
    records = [x for rs in results for x in rs]
    valid = verify_unchanged(run_dir, cd) and tree_checksums(cd.work / "agents") == _work_agents_sums(cd)
    site = cohorts["sites"]
    for rec in records:
        rec.update({"digest": cd.digest, "site": site.get(rec["agent"], "unknown"),
                    "cohort": cohorts["per_meme"].get(rec["meme"], {}).get(rec["agent"])})
    records.sort(key=lambda x: (x["scope"], x.get("position") or 0))
    with open(out_dir / "responses.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    summary = {"ckpt": ckpt, "run": str(run_dir), "layout": cd.layout, "digest": cd.digest, "spec_key": spec_key,
               "valid": bool(valid), "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
               "memes": [{"id": s.id, "phrase": s.phrase, "cell": s.cell, "repaired_day": s.repaired_day,
                          "n_items": len(items[s.id])} for s in specs],
               "registry_errors": errs, "panel": panel_rows(sorted(copies), cohorts), "skipped_agents": skipped,
               "n_calls": len({r["scope"] for r in records}), "n_rows": len(records),
               "n_errors": len({r["scope"] for r in records if r.get("error")}),
               "hit_rate": _hit_rate(records), "time": when.isoformat(),
               "cost": project_cost(rd, specs, [ckpt], bcfg)}
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    shutil.rmtree(cd.work, ignore_errors=True)
    if own_llm:
        llm.close()
    if not valid:
        raise ProbeInvalid(f"{run_dir}/graded/{ckpt}: source files changed during probing")
    return summary


def _hit_rate(records: list[dict]) -> dict:
    acc: dict[str, list[int]] = {}
    for r in records:
        acc.setdefault(f"{r['form']}:{r['meme']}", []).append(int(bool(r.get("hit"))))
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}


def read_responses(run_dir, ckpt: str, probes_root=None) -> list[dict]:
    p = (Path(probes_root) if probes_root else Path(run_dir) / "probes") / "graded" / ckpt / "responses.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def available_checkpoints(run_dir, probes_root=None) -> list[str]:
    root = (Path(probes_root) if probes_root else Path(run_dir) / "probes") / "graded"
    ks = sorted({p.parent.name for p in root.glob("*/responses.jsonl")}) if root.exists() else []
    return sorted(ks, key=lambda c: (int(c[1:]) if c[1:].isdigit() else 10 ** 9, c))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Graded meme battery on an isolated checkpoint copy (observer).")
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="final")
    ap.add_argument("--backend", default=None, help="override probe.backend (e.g. mock)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    backend = None
    if a.backend:
        from backend.llm.client import make_backend
        backend = make_backend({"backend": a.backend, "model": "haiku"})
    s = run_graded(a.run, a.checkpoint, backend=backend, force=a.force)
    print(json.dumps({k: s[k] for k in ("ckpt", "valid", "n_calls", "n_errors", "cost")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
