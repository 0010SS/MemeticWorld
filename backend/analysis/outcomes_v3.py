"""The outcomes.json `v3` block (OBSERVER ONLY; ontology v3 §5.12, §7.4) and the `analyze_v3(run_dir)` entry.

Per run (node x seed): continuity, inertia, inherited, records, attribution, social share, internalization,
expressions, manipulation, validity, cost. Across the branch nodes of one seed (sibling run dirs
runs/<design>/<node>/s<seed>/, found via the manifest's branch info / directory layout):
  PO1 Δ1 = A1(trunk) − A1(wipe4), A1 = mean over W1 newcomers of memory-only ACC_cur on K1c at C4;
      convergent: CRN-matched all-operator first-attempt accuracy on D4 faulted jobs;
  PO2 Δ2 = O2(keep_shift) − O2(wipe_shift), O2 = mean over present members of memory-only ORR on K1c at C5;
  S1 matched OLD on K1 first attempts D5–D6; S2 W2 ORR at C6 (prior-adjusted); S3 NEW at C6 + latency;
  positive control K1c OLD at C5 keep_noshift − keep_shift; LAG(c) = shift − keep_noshift.
Node names default to the study-1 tree and can be overridden with `analysis.v3.nodes` in the config.
Stubbed (NotImplementedError modules, reported as {"status": "not_run"}): DCF, coding/pragmatics, test-retest,
kappa, synthetic calibration, replicate-branch floor, specialization permutations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from backend.analysis import attribution as AT
from backend.analysis import jobs as J
from backend.analysis import meaning as M
from backend.analysis import record_lineage as RL
from backend.analysis import v3common as V
from backend.analysis.rundata import RunData

V3_VERSION = "v3.0"
DEFAULT_NODES = {"trunk": "trunk", "wipe4": "wipe4", "keep_shift": "keep_shift", "wipe_shift": "wipe_shift",
                 "keep_noshift": "keep_noshift"}
NOT_RUN = {"status": "not_run"}


# ------------------------------------------------------------------------------------------------ cohorts
def cohorts_roles(rd) -> tuple[dict, dict]:
    """({agent: founder|W1|W2|...}, {agent: am_crew|pm_crew|stores}) from the latest checkpoint's
    agent_state.json when present, else from roster_change trace records."""
    coh, role = {}, {}
    ck = rd.dir / "checkpoints"
    if ck.exists():
        for p in sorted(ck.glob("C*/agent_state.json"), key=lambda p: V.ckpt_day(p.parent.name)):
            for aid, st in (V.load_json(p, {}) or {}).items():
                if isinstance(st, dict):
                    if st.get("cohort"):
                        coh[aid] = st["cohort"]
                    if st.get("role"):
                        role[aid] = st["role"]
    rc = rd.of("roster_change")
    arrive_days = sorted({int(r.get("day", 0)) for r in rc if r.get("kind") == "arrive"})
    for r in rc:
        a = r.get("agent")
        if r.get("role"):
            role.setdefault(a, r["role"])
        if r.get("kind") == "arrive":
            coh.setdefault(a, f"W{arrive_days.index(int(r.get('day', 0))) + 1}")
    arrivals = {r.get("agent") for r in rc if r.get("kind") == "arrive"}
    for aid, a in (rd.agents or {}).items():
        if aid in coh or aid in arrivals:
            continue
        if isinstance(a, dict) and (a.get("reserve") or a.get("cohort") == "reserve"):
            continue
        coh[aid] = "founder"
        r = (a or {}).get("coop_role") or (a or {}).get("role") if isinstance(a, dict) else None
        if r and aid not in role:
            role[aid] = r
    return coh, role


def _wave_leavers(rd, wave: int) -> list[str]:
    rc = rd.of("roster_change")
    days = sorted({int(r.get("day", 0)) for r in rc if r.get("kind") == "depart"})
    if len(days) < wave:
        return []
    return sorted(r.get("agent") for r in rc if r.get("kind") == "depart" and int(r.get("day", 0)) == days[wave - 1])


# ------------------------------------------------------------------------------------------------ helpers
def _cell(run_dir, cfg, ck, coh, targets=None):
    rows = V.probe_responses(run_dir, ck)
    return M.cell(rows, cfg, ck, coh, targets) if rows else None


def _mean_agents(cell, agents, key="acc_cur_k1c"):
    if not cell:
        return None
    return V.mean(cell["per_agent"].get(a, {}).get(key) for a in agents if a in cell["per_agent"])


def _in(coh, *labels):
    return sorted(a for a, c in coh.items() if c in labels)


def load_targets(run_dir) -> dict:
    """probes/targets.json (frozen on the trunk at C3) from this run, its parent chain or the design dir."""
    seen, d = 0, Path(run_dir)
    while d is not None and seen < 4:
        t = V.load_json(d / "probes" / "targets.json")
        if t:
            return t
        d, seen = V.parent_run(d), seen + 1
    for base in (Path(run_dir).parent.parent / "trunk" / Path(run_dir).name, Path(run_dir).parent.parent):
        t = V.load_json(base / "probes" / "targets.json")
        if t:
            return t
    return {}


def _target_forms(t: dict, cfg: dict) -> dict:
    """{"X": phrase|None, "W": "F4", "Y": phrase|None} from targets.json (values may be dicts)."""
    out = {}
    for k in ("X", "W", "Y", "X_desc", "Y_desc"):
        v = t.get(k)
        if isinstance(v, dict):
            v = v.get("phrase") or v.get("form") or v.get("text")
        if v:
            out[k] = v
    pc = ((cfg.get("workshop") or {}).get("panel_code") or {})
    if "W" not in out and pc.get("enabled"):
        out["W"] = pc.get("text") or "F4"
    return out


# ------------------------------------------------------------------------------------------------ checks
_AUDIT = [re.compile(p) for p in (r"\bK[0-3]\b", r"\b(LENS|DAMP|BELT|AIR|WARP)\b", r"\bM[12]\b", r"\bj\d\d\.\d\b",
                                  r"\b(trunk|wipe4|keep_shift|wipe_shift|keep_noshift|noshift|placebo)\b",
                                  r"\bregime\b", r"\bmapping\b", r"\blaser_(alpha|beta)\b", r"\bE[1-4]\b")]


def prompt_audit(run_dir, max_examples: int = 5) -> dict:
    """§8.1 step 10 over the simulation llm_calls.jsonl: hidden ids, job ids, arm names, regime/mapping."""
    n = bad = 0
    examples = []
    for r in V.load_jsonl(Path(run_dir) / "llm_calls.jsonl"):
        p = r.get("prompt")
        if not isinstance(p, str):
            continue
        n += 1
        hits = sorted({m.group(0) for rx in _AUDIT for m in rx.finditer(p)})
        if hits:
            bad += 1
            if len(examples) < max_examples:
                examples.append({"scope": r.get("scope"), "purpose": r.get("purpose"), "hits": hits})
    return {"n_prompts": n, "n_flagged": bad, "ok": bad == 0, "examples": examples}


def llm_errors(run_dir) -> int:
    return sum(1 for r in V.load_jsonl(Path(run_dir) / "llm_calls.jsonl")
               if r.get("error") or str(r.get("response", "")).startswith("LLM_ERROR"))


def manipulation(rd) -> dict:
    cfg = rd.cfg
    en = lambda *ks: _enabled(cfg, *ks)
    counts = {
        "jobs": len(rd.of("job_start")), "k1_faults": sum(1 for r in rd.of("job_truth") if (r.get("klass") or r.get("class")) == "K1"),
        "decisions": len(rd.of("job_decision")), "asks": len(rd.of("clarification")),
        "handovers": len(rd.of("handover")), "meetings": len(rd.of("meeting")),
        "reads": len(rd.of("record_read")), "writes": sum(1 for w in rd.of("record_write") if w.get("choice") not in (None, "none")),
        "transitions": len(rd.of("record_transition")),
        "departures": sum(1 for r in rd.of("roster_change") if r.get("kind") == "depart"),
        "arrivals": sum(1 for r in rd.of("roster_change") if r.get("kind") == "arrive"),
        "regime_applied": sorted({(int(r.get("day", 0)), r.get("regime")) for r in rd.of("regime_active")}),
    }
    newcomers = {r.get("agent") for r in rd.of("roster_change") if r.get("kind") == "arrive"}
    counts["talk_to_newcomers"] = sum(1 for u in rd.utterances if set(u.get("listeners") or []) & newcomers)
    mech = {"jobs": en("workshop"), "k1_faults": en("workshop"), "decisions": en("workshop"),
            "asks": en("comm", "clarify"), "handovers": en("comm", "handover"), "meetings": en("comm", "meeting"),
            "reads": en("records"), "writes": en("records"),
            "transitions": en("records") and bool((cfg.get("records") or {}).get("transitions")),
            "departures": en("turnover") and bool((cfg.get("turnover") or {}).get("waves")),
            "arrivals": en("turnover") and bool((cfg.get("turnover") or {}).get("waves"))}
    status = {k: ("off" if not on else "active" if counts.get(k) else "inactive") for k, on in mech.items()}
    return {"counts": counts, "status": status}


def _enabled(cfg, *keys) -> bool:
    d = cfg
    for k in keys:
        d = (d or {}).get(k) or {}
    return bool(d.get("enabled")) if isinstance(d, dict) else bool(d)


def predictive_validity(run_dir, rd, table) -> dict:
    """An agent's modal P-sit K1c choice at C_k matches its own next real K1 first attempt on day k+1."""
    out = {}
    for ck in ("C3", "C5"):
        rows = [r for r in V.probe_responses(run_dir, ck) if r.get("form") == "P-sit" and r.get("type") == "K1c"
                and r.get("action")]
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["agent"], []).append(r["action"])
        d = V.ckpt_day(ck) + 1
        hits = n = 0
        for a, acts in by.items():
            nxt = sorted((j for j in table.values() if j["class"] == "K1" and j["day"] == d and j["operator"] == a
                          and j["first_valid"]), key=lambda j: j["first_tick"] or 0)
            if nxt:
                n += 1
                hits += int(max(set(acts), key=acts.count) == nxt[0]["first_action"])
        out[ck] = {"match": round(hits / n, 4) if n else None, "n": n}
    return out


# ------------------------------------------------------------------------------------------------ per run
def build_v3(run_dir, contrasts: bool = True) -> dict:
    run_dir = Path(run_dir)
    rd = RunData(run_dir)
    cfg = rd.cfg
    coh, role = cohorts_roles(rd)
    table = J.job_table(rd)
    targets_raw = load_targets(run_dir)
    tforms = _target_forms(targets_raw, cfg)
    cells = {ck: _cell(run_dir, cfg, ck, coh, tforms) for ck in ("C3", "C4", "C5", "C6")}
    c0 = V.c0_responses(run_dir)
    c0cell = M.cell(c0, cfg, "C0", None, tforms) if c0 else None
    present = lambda ck: (cells[ck] or {}).get("agents") or []
    w1, w2 = _in(coh, "W1"), _in(coh, "W2")
    newcomers = w1 + w2
    sd = V.shift_day(cfg)

    acc_w1_c4 = _mean_agents(cells["C4"], w1)
    leavers_c3 = _mean_agents(cells["C3"], _wave_leavers(rd, 1))
    continuity = {"acc_mem_w1_c4": acc_w1_c4,
                  "acc_mem_w1crew_c4": _mean_agents(cells["C4"], [a for a in w1 if role.get(a) != "stores"]),
                  "acc_mem_all_c4": _mean_agents(cells["C4"], present("C4")),
                  "first_attempt_acc_d4": J.first_attempt_accuracy(table, [4])["acc"],
                  "retention_ratio": round(acc_w1_c4 / leavers_c3, 4) if acc_w1_c4 is not None and leavers_c3 else None}
    tw = lambda ck, k: ((cells[ck] or {}).get("three_way") or {}).get(k)
    onx = J.old_new_other(table, cfg, [5, 6])
    inertia = {"orr_mem_all_c5": _mean_agents(cells["C5"], present("C5"), "orr"),
               "orr_mem_all_c6": _mean_agents(cells["C6"], present("C6"), "orr"),
               "orr_jobs_d5d6": onx["old"], "jobs_three_way_d5d6": onx,
               "new_mem_all_c6": tw("C6", "new"), "other_mem_all_c5": tw("C5", "other"),
               "hedge_c5": tw("C5", "hedge"), "switch_latency": J.switch_latency(table, cfg, sd)}
    c0_orr = {a: v.get("orr") for a, v in ((c0cell or {}).get("per_agent") or {}).items()}
    orr_w2 = _mean_agents(cells["C6"], w2, "orr")
    orr_w2_c0 = V.mean(c0_orr.get(a) for a in w2)
    inherited = {"orr_mem_w2_c6": orr_w2, "orr_w2_prior_c0": orr_w2_c0,
                 "orr_mem_w2_c6_prior_adjusted": V.sub(orr_w2, orr_w2_c0)}
    records = RL.record_metrics(rd, table)

    def social(ck):
        c = cells[ck]
        if not c:
            return None
        vals = [V.sub(_acc_types(c, a, "P"), _acc_types(c, a, "P-abl")) for a in newcomers if a in c["per_agent"]]
        return V.mean(vals)

    def internal(ck):
        c = cells[ck]
        return None if not c else {"ratio": c["internalization"], "gap": c["situated_gap"]}

    expr = {"targets": targets_raw or {k: v for k, v in tforms.items()},
            "sri": {ck: (cells[ck] or {}).get("sri") for ck in cells if cells[ck]},
            "know": {ck: (cells[ck] or {}).get("know") for ck in cells if cells[ck]},
            "ext": {ck: {c: e["ext"] for c, e in ((cells[ck] or {}).get("ext") or {}).items()} for ck in cells if cells[ck]},
            "breadth": {ck: {c: e["breadth"] for c, e in ((cells[ck] or {}).get("ext") or {}).items()} for ck in cells if cells[ck]},
            "form": {k: M.form_persistence(rd, v, sd or 5, table) for k, v in tforms.items()},
            "answer_persistence": {"C4_C6": M.answer_persistence(V.probe_responses(run_dir, "C4"),
                                                                 V.probe_responses(run_dir, "C6"))},
            "lag": None, "class": None}
    hit = {ck: (cells[ck] or {}).get("retrieval_hit_rate") for ck in cells if cells[ck]}
    block = {
        "version": V3_VERSION, "node": V.node_of(run_dir), "seed": cfg.get("seed"),
        "world_seed": cfg.get("world_seed") if cfg.get("world_seed") is not None else cfg.get("seed"),
        "mapping": V.mapping_of(cfg), "shift_day": sd, "cohorts": coh, "roles": role,
        "continuity": continuity, "inertia": inertia, "inherited": inherited, "records": records,
        "jobs": J.analyze_jobs(rd) if table else None,
        "dcf": dict(NOT_RUN), "attribution": AT.analyze_attribution(rd, tforms, table),
        "social_share": {ck: social(ck) for ck in ("C4", "C6")},
        "internalization": {ck: internal(ck) for ck in ("C4", "C6")},
        "expressions": expr, "pragmatics": dict(NOT_RUN), "specialization": dict(NOT_RUN),
        "manipulation": manipulation(rd),
        "validity": {"predictive_validity": predictive_validity(run_dir, rd, table), "retest_agreement": None,
                     "kappa": None, "retrieval_hit_rate": hit, "prompt_audit": prompt_audit(run_dir),
                     "llm_errors": llm_errors(run_dir), "prefix_digest_match": _prefix_match(rd.manifest)},
        "cost": {"llm_calls": sum(1 for _ in V.load_jsonl(run_dir / "llm_calls.jsonl")),
                 "calls_per_day": None, "wall_seconds": rd.manifest.get("wall_seconds")},
    }
    days = max(1, int(rd.manifest.get("ticks") or V.ticks_per_day(rd)) // V.ticks_per_day(rd))
    block["cost"]["calls_per_day"] = round(block["cost"]["llm_calls"] / days, 2)
    if contrasts:
        block["contrasts"] = seed_contrasts(run_dir, block)
        lag = block["contrasts"].get("lag") or {}
        expr["lag"] = lag or None
        expr["class"] = block["contrasts"].get("class")
    return block


def _acc_types(cell, agent, form):
    """One agent's accuracy on the source-ablated item set, memory-only (P) or ablated (P-abl)."""
    pa = cell["per_agent"].get(agent) or {}
    return pa.get("acc_p_on_abl_items") if form == "P" else pa.get("acc_abl")


def _prefix_match(manifest: dict):
    b = manifest.get("branch") or {}
    if not b:
        return None
    if b.get("parent_prefix_digest") is None or b.get("prefix_digest") is None:
        return None
    return b["parent_prefix_digest"] == b["prefix_digest"]


# ------------------------------------------------------------------------------------------------ across nodes
def seed_contrasts(run_dir, own: dict | None = None) -> dict:
    """Seed-level paired contrasts from sibling node runs of the same seed (missing nodes -> None)."""
    run_dir = Path(run_dir)
    rd = RunData(run_dir)
    names = dict(DEFAULT_NODES, **(((rd.cfg.get("analysis") or {}).get("v3") or {}).get("nodes") or {}))
    me = V.node_of(run_dir)
    cache: dict[str, dict | None] = {}

    def blk(key):
        node = names[key]
        if node in cache:
            return cache[node]
        if node == me and own is not None:
            cache[node] = own
        else:
            d = run_dir if node == me else V.sibling(run_dir, node)
            cache[node] = build_v3(d, contrasts=False) if d else None
        return cache[node]

    def tab(key):
        node = names[key]
        d = run_dir if node == me else V.sibling(run_dir, node)
        return J.job_table(d) if d else None

    g = lambda b, *ks: _dig(b, ks)
    tr, w4, ks, ws, kn = (blk(k) for k in ("trunk", "wipe4", "keep_shift", "wipe_shift", "keep_noshift"))
    out = {"nodes_found": {k: v is not None for k, v in (("trunk", tr), ("wipe4", w4), ("keep_shift", ks),
                                                         ("wipe_shift", ws), ("keep_noshift", kn))}}
    out["po1_continuity"] = {"a1_trunk": g(tr, "continuity", "acc_mem_w1_c4"), "a1_wipe4": g(w4, "continuity", "acc_mem_w1_c4"),
                             "delta": V.sub(g(tr, "continuity", "acc_mem_w1_c4"), g(w4, "continuity", "acc_mem_w1_c4"))}
    ttr, tw4 = (tab("trunk"), tab("wipe4")) if tr and w4 else (None, None)
    out["po1_convergent_first_attempt_d4"] = J.matched_first_attempts(ttr, tw4, [4]) if ttr and tw4 else None
    out["po2_inertia"] = {"o2_keep_shift": g(ks, "inertia", "orr_mem_all_c5"), "o2_wipe_shift": g(ws, "inertia", "orr_mem_all_c5"),
                          "delta": V.sub(g(ks, "inertia", "orr_mem_all_c5"), g(ws, "inertia", "orr_mem_all_c5"))}
    tks, tws = (tab("keep_shift"), tab("wipe_shift")) if ks and ws else (None, None)
    if tks and tws:
        ids = sorted({j for j, x in tks.items() if x["class"] == "K1" and x["day"] in (5, 6) and x["first_valid"]}
                     & {j for j, x in tws.items() if x["class"] == "K1" and x["day"] in (5, 6) and x["first_valid"]})
        cfg = RunData(V.sibling(run_dir, names["keep_shift"]) or run_dir).cfg
        afix = V.regime_fixes(cfg)["A"]
        o_k = V.share(ids, lambda i: tks[i]["first_action"] == afix)
        o_w = V.share(ids, lambda i: tws[i]["first_action"] == afix)
        out["s1_behavioural_inertia"] = {"n": len(ids), "old_keep": o_k, "old_wipe": o_w, "delta": V.sub(o_k, o_w)}
    else:
        out["s1_behavioural_inertia"] = None
    out["s2_inherited_inertia"] = {"delta": V.sub(g(ks, "inherited", "orr_mem_w2_c6_prior_adjusted"),
                                                  g(ws, "inherited", "orr_mem_w2_c6_prior_adjusted"))}
    out["s3_adaptation"] = {"new_c6_keep": g(ks, "inertia", "new_mem_all_c6"), "new_c6_wipe": g(ws, "inertia", "new_mem_all_c6"),
                            "latency_keep": g(ks, "inertia", "switch_latency", "latency"),
                            "latency_wipe": g(ws, "inertia", "switch_latency", "latency")}
    out["positive_control_old_c5"] = V.sub(_dig(kn, ("inertia", "orr_mem_all_c5")), _dig(ks, ("inertia", "orr_mem_all_c5")))
    # LAG(c) = [ΔSRI(c) − ΔKNOW] C4→C6, shift arm minus keep_noshift
    lag, klass = {}, {}
    for arm_key, arm in (("keep_shift", ks), ("wipe_shift", ws)):
        if not arm or not kn:
            continue
        for c in sorted(set((_dig(arm, ("expressions", "sri", "C6")) or {})) | {"X", "W", "Y"}):
            la, ln = _lag1(arm, c), _lag1(kn, c)
            lag.setdefault(arm_key, {})[c] = V.sub(la, ln)
        for c in ("X", "W"):
            d_sri = _dsri(arm, c)
            fp = (_dig(arm, ("expressions", "form")) or {}).get(c) or {}
            klass.setdefault(arm_key, {})[c] = M.classify(
                fp.get("use_ratio"), d_sri, _dsri(kn, c), _dsri(arm, "Y"),
                (_dig(arm, ("expressions", "sri", "C6")) or {}).get(c)) if d_sri is not None else None
    out["lag"], out["class"] = lag, klass
    return out


def _dig(b, ks):
    for k in ks:
        if not isinstance(b, dict):
            return None
        b = b.get(k)
    return b


def _dsri(b, c):
    s4 = (_dig(b, ("expressions", "sri", "C4")) or {}).get(c)
    s6 = (_dig(b, ("expressions", "sri", "C6")) or {}).get(c)
    return V.sub(s6, s4)


def _lag1(b, c):
    dk = V.sub(_dig(b, ("expressions", "know", "C6")), _dig(b, ("expressions", "know", "C4")))
    return V.sub(_dsri(b, c), dk)


# ------------------------------------------------------------------------------------------------ entry
def analyze_v3(run_dir, write: bool = True, verbose: bool = False) -> dict:
    """Compute the v3 block; write meaning.json, lineage.json and merge {"v3": block} into outcomes.json."""
    run_dir = Path(run_dir)
    rd = RunData(run_dir)
    coh, _role = cohorts_roles(rd)
    block = build_v3(run_dir)
    if write:
        M.analyze_meaning(run_dir, coh, _target_forms(load_targets(run_dir), rd.cfg), write=True)
        RL.write_lineage(rd)
        p = run_dir / "outcomes.json"
        cur = V.load_json(p, {}) or {}
        cur["v3"] = block
        json.dump(cur, open(p, "w"), indent=1, default=str)
    if verbose:
        c, i = block["continuity"], block["inertia"]
        print(f"[analyze_v3] node={block['node']} acc_w1_c4={c['acc_mem_w1_c4']} orr_c5={i['orr_mem_all_c5']} "
              f"po1={block['contrasts']['po1_continuity']['delta']} po2={block['contrasts']['po2_inertia']['delta']}")
    return block
