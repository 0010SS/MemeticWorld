"""Observer v3 metrics (ontology v3 §5.5-5.8, §5.12, §7.4) on synthetic traces and probe responses.

Seed 13 -> mapping M1: A-fix = lens, B-fix = dry (K2 -> belt). One synthetic seed family:
runs/<design>/{trunk, wipe4, keep_shift, wipe_shift, keep_noshift}/s13/.
"""
import json
from pathlib import Path

import pytest
import yaml

from backend.analysis import attribution as AT
from backend.analysis import jobs as J
from backend.analysis import meaning as M
from backend.analysis import outcomes_v3 as O3
from backend.analysis import record_lineage as RL
from backend.analysis import v3common as V
from backend.analysis.rundata import RunData

FOUNDERS = {"maya": "am_crew", "dev": "am_crew", "hana": "am_crew", "ethan": "pm_crew", "leo": "pm_crew",
            "jordan": "pm_crew", "priya": "stores", "sofia": "stores"}
RESERVES = ["noor", "tomas", "wei", "amara", "felix"]
W1 = [("hana", "wei", "am_crew"), ("ethan", "amara", "pm_crew"), ("priya", "felix", "stores")]
W2 = [("maya", "noor", "am_crew"), ("leo", "tomas", "pm_crew")]
SHIFT = [{"day": 1, "regime": "A"}, {"day": 5, "regime": "B"}]
NOSHIFT = [{"day": 1, "regime": "A"}]


def _cfg(schedule, transitions):
    return {"seed": 13, "world_seed": 13, "llm": {"backend": "mock", "model": "mock"},
            "workshop": {"enabled": True, "content": "laser_alpha", "panel_code": {"enabled": True, "text": "F4"}},
            "regimes": {"mapping": "auto", "schedule": schedule},
            "records": {"enabled": True, "transitions": transitions},
            "turnover": {"enabled": True, "waves": [{"day": 4}, {"day": 6}]},
            "comm": {"clarify": {"enabled": True}, "handover": {"enabled": False}, "meeting": {"enabled": False}}}


def _roster(days):
    out = []
    for d, wave in ((4, W1), (6, W2)):
        if d > days:
            continue
        for old, new, role in wave:
            out.append({"type": "roster_change", "tick": (d - 1) * 60, "day": d, "agent": old, "kind": "depart", "role": role})
            out.append({"type": "roster_change", "tick": (d - 1) * 60, "day": d, "agent": new, "kind": "arrive",
                        "role": role, "replaces": old})
    return out


def _k1c(agent, actions, form="P"):
    return [{"agent": agent, "item": f"k1c_{i}", "type": "K1c", "form": form, "action": a, "order": i % 2}
            for i, a in enumerate(actions)]


def _acc_rows(agent, n_right, right, wrong="slow", n=8):
    return _k1c(agent, [right] * n_right + [wrong] * (n - n_right))


def _write_run(root: Path, node, days, schedule, transitions, trace, probes, parent=None, at_day=None,
               llm_calls=()):
    d = root / node / "s13"
    d.mkdir(parents=True)
    agents = {a: {"name": a.title(), "coop_role": r} for a, r in FOUNDERS.items()}
    agents.update({a: {"name": a.title(), "reserve": True} for a in RESERVES})
    man = {"run_id": "s13", "node": node, "ticks_per_day": 60, "tick_minutes": 15, "ticks": 60 * days,
           "start": "2026-09-14T07:30:00", "agents": agents, "status": "finished", "wall_seconds": 12.0}
    if parent:
        man["branch"] = {"parent_run": str(root / parent / "s13"), "at_day": at_day, "at_tick": (at_day - 1) * 60,
                         "parent_prefix_digest": "abc", "prefix_digest": "abc"}
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "config.resolved.yaml").write_text(yaml.safe_dump(_cfg(schedule, transitions)))
    recs = _roster(days) + list(trace)
    (d / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "llm_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in llm_calls))
    for ck, rows in probes.items():
        (d / "probes" / ck).mkdir(parents=True)
        (d / "probes" / ck / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def _job(jid, day, klass, cause, regime, agent, action, tick, view=(), retrieved=(), attempt_success=None):
    out = [{"type": "job_start", "tick": tick, "job": jid, "day": day, "operator": agent},
           {"type": "job_truth", "tick": tick, "job": jid, "regime": regime, "mapping": "M1", "klass": klass, "cause": cause},
           {"type": "job_decision", "tick": tick, "agent": agent, "job": jid, "attempt": 1, "action": action,
            "valid": True, "binder_entry_ids": list(view), "retrieved": list(retrieved)}]
    succ = attempt_success if attempt_success is not None else V.fix_of(cause) == action
    out.append({"type": "job_attempt", "tick": tick + 1, "job": jid, "attempt": 1, "operator": agent, "action": action,
                "success": succ})
    out.append({"type": "job_end", "tick": tick + 2, "job": jid, "operator": agent, "delivered": succ})
    return out


REV1 = "F4 or tea edges on the left = lens, wipe it first."
E2 = "Left side not cutting -> dried the sheets on the rack, fine."
REV2 = "Left side not cutting: dry the sheets on the rack first."
REV3 = "Left side not cutting: dry the sheets on the rack first. Wavy lines: belt."


def _prefix_trace():
    """Days 1-4 (shared by every node): talk with 'tea edges', the F4 panel fact, rev1, wei reading rev1."""
    return [
        {"type": "conversation", "tick": 60, "id": "c1", "participants": ["dev", "hana"], "transcript": []},
        {"type": "utterance", "tick": 60, "id": "u1", "speaker": "dev", "text": "those tea edges again, lens for sure",
         "listeners": ["hana"], "conversation_id": "c1", "idx": 0},
        {"type": "viewpoint", "tick": 95, "agent": "leo", "facts": [{"perceived": "The laser's panel showed F4."}]},
        {"type": "utterance", "tick": 100, "id": "u2", "speaker": "leo", "text": "F4 again on the laser",
         "listeners": ["jordan"], "conversation_id": None, "idx": 0},
        {"type": "record_write", "tick": 70, "agent": "hana", "offer": "job", "choice": "front", "rev_id": "r1",
         "entry_id": None, "text": REV1},
        {"type": "record_write", "tick": 75, "agent": "dev", "offer": "job", "choice": "none", "text": None},
        {"type": "record_read", "tick": 190, "agent": "wei", "context": "decision", "entry_ids": [], "rev_ids": ["r1"],
         "new_ids": ["r1"], "view_sha": "x"},
        {"type": "conversation", "tick": 200, "id": "c2", "participants": ["wei", "maya"], "transcript": []},
        {"type": "utterance", "tick": 200, "id": "u3", "speaker": "wei", "text": "tea edges again today",
         "listeners": ["maya"], "conversation_id": "c2", "idx": 0},
        {"type": "regime_active", "tick": 0, "day": 1, "regime": "A", "mapping": "M1"},
    ]


def _d4_jobs(actions):
    out = []
    for i, a in enumerate(actions):
        out += _job(f"j04.{i}", 4, "K1", "LENS", "A", "wei", a, 186 + 4 * i, view=["r1"])
        out.append({"type": "record_read", "tick": 186 + 4 * i, "agent": "wei", "context": "decision",
                    "entry_ids": [], "rev_ids": ["r1"], "new_ids": [], "view_sha": "x"})
    return out


def _shift_trace():
    t = [
        {"type": "memory_encoded", "tick": 244, "node_id": "n1", "agent": "amara", "source_type": "conversation",
         "text": "Maya said drying the sheets fixed the left side"},
        {"type": "record_write", "tick": 250, "agent": "dev", "offer": "job", "choice": "log", "entry_id": "e2",
         "rev_id": None, "text": E2},
        {"type": "record_write", "tick": 262, "agent": "maya", "offer": "job", "choice": "front", "rev_id": "r2",
         "entry_id": None, "text": REV2},
        {"type": "record_write", "tick": 330, "agent": "wei", "offer": "job", "choice": "front", "rev_id": "r3",
         "entry_id": None, "text": REV3},
    ]
    t += _job("j05.0", 5, "K1", "DAMP", "B", "dev", "lens", 246, view=["r1"])
    t += _job("j05.1", 5, "K1", "DAMP", "B", "maya", "slow", 250, view=["r1"])
    t += _job("j05.2", 5, "K1", "DAMP", "B", "wei", "dry", 254, view=["r1", "e2"])
    t += _job("j05.3", 5, "K1", "DAMP", "B", "amara", "dry", 266, view=["r2"], retrieved=["n1"])
    t += _job("j05.4", 5, "K2", "BELT", "B", "leo", "belt", 270, view=["r2"])
    return t


def _members(day):
    act = set(FOUNDERS)
    for d, wave in ((4, W1), (6, W2)):
        if d <= day:
            act -= {o for o, _n, _r in wave}
            act |= {n for _o, n, _r in wave}
    return sorted(act)


@pytest.fixture(scope="module")
def family(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs") / "records_boundaries_v3"
    pre = _prefix_trace()
    c3 = sum((_acc_rows(a, 8, "lens") for a in ("hana", "ethan", "priya")), []) + _acc_rows("dev", 6, "lens")
    c4_trunk = _acc_rows("wei", 8, "lens") + _acc_rows("amara", 6, "lens") + _acc_rows("felix", 4, "lens")
    c4_trunk += _acc_rows("wei", 4, "lens", n=8)[:0] + [
        {"agent": "wei", "item": "k1c_0", "type": "K1c", "form": "P-abl", "action": "slow"},
        {"agent": "wei", "item": "k1c_1", "type": "K1c", "form": "P-abl", "action": "lens"}]
    c4_wipe = _acc_rows("wei", 4, "lens") + _acc_rows("amara", 4, "lens") + _acc_rows("felix", 2, "lens")
    trunk = _write_run(root, "trunk", 4, NOSHIFT, [{"day": 4, "mode": "keep"}], pre + _d4_jobs(["lens"] * 4),
                       {"C3": c3, "C4": c4_trunk},
                       llm_calls=[{"scope": "t0001:02work:dev", "purpose": "job", "prompt": "It is Monday. regime"},
                                  {"scope": "t0002:02work:dev", "purpose": "job", "prompt": "clean prompt"}])
    wipe4 = _write_run(root, "wipe4", 4, NOSHIFT, [{"day": 4, "mode": "wipe"}],
                       pre + _d4_jobs(["lens", "lens", "slow", "slow"]) +
                       [{"type": "record_transition", "tick": 180, "day": 4, "mode": "wipe", "archived_binder_id": "b1"}],
                       {"C4": c4_wipe}, parent="trunk", at_day=4)
    shift_tr = pre + _d4_jobs(["lens"] * 4) + _shift_trace()
    c5 = lambda n_old: sum((_acc_rows(a, n_old, "lens", wrong="dry") for a in _members(5)), [])
    c6 = [dict(r, cue=None) for r in sum((_acc_rows(a, 2, "lens", wrong="dry") for a in _members(6)), [])]
    ks = _write_run(root, "keep_shift", 6, SHIFT, [{"day": 4, "mode": "keep"}, {"day": 5, "mode": "keep"}],
                    shift_tr, {"C5": c5(6), "C6": c6}, parent="trunk", at_day=5)
    ws = _write_run(root, "wipe_shift", 6, SHIFT, [{"day": 4, "mode": "keep"}, {"day": 5, "mode": "wipe"}],
                    shift_tr, {"C5": c5(2)}, parent="trunk", at_day=5)
    kn = _write_run(root, "keep_noshift", 6, NOSHIFT, [{"day": 4, "mode": "keep"}, {"day": 5, "mode": "keep"}],
                    pre + _d4_jobs(["lens"] * 4), {"C5": c5(8)}, parent="trunk", at_day=5)
    return {"root": root, "trunk": trunk, "wipe4": wipe4, "keep_shift": ks, "wipe_shift": ws, "keep_noshift": kn}


# ---------------------------------------------------------------------------------------------- ground truth
def test_gt_and_mapping():
    cfg = _cfg(SHIFT, [])
    assert V.mapping_of(cfg) == "M1" and V.regime_fixes(cfg) == {"A": "lens", "B": "dry"}
    assert V.gt("K1c", "A", cfg) == "lens" and V.gt("K1c", "B", cfg) == "dry"
    assert V.gt("K2", "B", cfg) == "belt" and V.gt("K3", "A", cfg) == "dry" and V.gt("CUE", "A", cfg) is None
    cfg2 = dict(cfg, seed=12, world_seed=12)
    assert V.regime_fixes(cfg2) == {"A": "dry", "B": "lens"}
    assert V.regime_on_day(cfg, 4) == "A" and V.regime_on_day(cfg, 5) == "B" and V.shift_day(cfg) == 5
    assert V.shift_day(_cfg(NOSHIFT, [])) is None


def test_recommended_action_keyword_map():
    assert V.recommended_action(REV1) == ("lens", False)
    assert V.recommended_action(E2) == ("dry", False)
    assert V.recommended_action(REV3)[0] is None and V.recommended_action(REV3)[1] is True
    assert V.recommended_actions("tighten the belt knob") == {"belt"}


# ---------------------------------------------------------------------------------------------- meaning
def test_meaning_cell_metrics():
    cfg = _cfg(SHIFT, [])
    rows = _acc_rows("a", 6, "lens", wrong="dry") + _acc_rows("b", 2, "lens", wrong="slow")
    rows += [{"agent": "a", "item": "k2_0", "type": "K2", "form": "P", "action": "belt"}]
    # N: cue X pulls to lens in both A and B framings; NONCE is neutral (slow)
    rows += [{"agent": "a", "form": "N", "cue": "X", "framing": f, "action": "lens"} for f in ("note", "text", "overheard")]
    rows += [{"agent": "a", "form": "N", "cue": "NONCE", "framing": f, "action": "slow"} for f in ("note", "text", "overheard")]
    # A: X fits K1c items, not sure on K3, doesn't fit K0
    rows += [{"agent": "a", "form": "A", "cue": "X", "fits": [
        {"item": "k1c_0", "type": "K1c", "fit": "fits"}, {"item": "k1c_1", "type": "K1c", "fit": "fits"},
        {"item": "k3_0", "type": "K3", "fit": "not sure"}, {"item": "k0_0", "type": "K0", "fit": "doesn't fit"}]}]
    rows = [x for r in rows for x in V.norm_response(r)]
    c5 = M.cell(rows, cfg, "C5", {"a": "founder", "b": "W1"})
    assert c5["regime"] == "B"
    assert c5["per_agent"]["a"]["orr"] == 0.75 and c5["per_agent"]["a"]["acc_cur_k1c"] == 0.25
    assert c5["per_agent"]["b"]["three_way"]["hedge"] == 0.75
    assert c5["orr"] == 0.5                                  # pooled 8/16
    assert c5["know"] == pytest.approx(0.5 * (2 / 16 - 8 / 16))
    assert c5["acc_by_type"]["K2"] == 1.0
    assert c5["by_cohort"]["W1"]["orr"] == 0.25
    assert c5["sri"]["X"] == pytest.approx(-0.5)            # X implies the A-fix
    e = c5["ext"]["X"]
    assert e["ext"] == {"K0": 0.0, "K1c": 1.0, "K3": 0.5} and e["breadth"] == 3
    c4 = M.cell(rows, cfg, "C4")
    assert c4["per_agent"]["a"]["acc_cur_k1c"] == 0.75       # scored against GT(A) at C4


def test_classify_rules():
    assert M.classify(0, 0.5, 0, 0, 0.3) == "LOSS"
    assert M.classify(0.8, 0.5, 0.1, 0.0, 0.3) == "SEMANTIC_CHANGE"
    assert M.classify(0.8, 0.5, 0.6, 0.0, 0.3) != "SEMANTIC_CHANGE"
    assert M.classify(0.8, 0.1, 0.0, 0.0, -0.2) == "INERTIA"
    assert M.classify(0.3, 0.1, 0.0, 0.0, 0.1, new_expr_sri=0.4) == "REPLACEMENT"


# ---------------------------------------------------------------------------------------------- jobs + records
def test_jobs_behavioural(family):
    t = J.job_table(family["keep_shift"])
    assert t["j05.0"]["gt"] == "dry" and t["j05.0"]["first_correct"] is False and t["j04.0"]["first_correct"] is True
    onx = J.old_new_other(t, RunData(family["keep_shift"]).cfg, [5, 6])
    assert onx == {"old": 0.25, "new": 0.5, "other": 0.25, "hedge": 0.25, "n": 4}
    assert J.switch_latency(t, RunData(family["keep_shift"]).cfg)["latency"] == 4
    assert J.follow_rate(family["keep_shift"], t) == {"follow_rate": 0.875, "n": 8}
    assert J.follow_rate(family["keep_shift"], t, days=[5]) == {"follow_rate": 0.75, "n": 4}  # j05.1 ignored it
    m = J.matched_first_attempts(J.job_table(family["trunk"]), J.job_table(family["wipe4"]), [4])
    assert m["n"] == 4 and m["delta"] == 0.5


def test_record_metrics_and_lineage(family):
    rm = RL.record_metrics(family["keep_shift"])
    assert rm["staleness_hours"] == pytest.approx((262 - 246) * 15 / 60) and rm["staleness_censored"] is False
    assert rm["stale_view_share"] == 0.5
    s1 = "Left side not cutting: dry the sheets on the rack first."
    assert rm["departed_author_share"] == pytest.approx(len(s1) / (len(s1) + len("Wavy lines: belt.")), abs=1e-3)
    assert rm["front_revisions"] == 3 and rm["writes"] == 4 and rm["offers"] == 5
    g = RL.lineage_graph(family["keep_shift"])
    assert {"src": "r1", "dst": "read:wei@190", "kind": "read", "context": "decision", "new": True} in g["edges"]
    assert any(e["kind"] == "view" and e["src"] == "e2" for e in g["edges"])


def test_attribution_and_spread(family):
    ad = AT.adoption(family["keep_shift"])
    assert set(ad["b_fix_first_attempt"]) == {"wei", "amara"}
    assert ad["b_fix_first_attempt"]["amara"]["labels"] == ["RECORD", "TALK"]
    assert ad["shares"]["RECORD"] == 1.0 and ad["shares"]["NONE"] == 0.0
    x = AT.expression_spread(family["keep_shift"], "tea edges")
    assert x["n_read_exposures"] == 5 and x["carried_adopters"] == ["hana", "wei"] and x["emerged"] is True
    w = AT.expression_spread(family["keep_shift"], "F4", world_anchored=True)
    assert w["in_world_text"] is True and w["emerged"] is False and w["world_exposed_producers"] == ["leo"]


# ---------------------------------------------------------------------------------------------- outcomes v3
def test_outcomes_v3_block_and_contrasts(family):
    b = O3.analyze_v3(family["keep_shift"])
    assert b["cohorts"]["wei"] == "W1" and b["cohorts"]["noor"] == "W2" and b["cohorts"]["dev"] == "founder"
    assert "tomas" in b["cohorts"] and "hana" in b["cohorts"]
    assert b["inertia"]["orr_mem_all_c5"] == 0.75 and b["inertia"]["orr_mem_all_c6"] == 0.25
    assert b["continuity"]["acc_mem_w1_c4"] == 0.75           # C4 read from the parent trunk (shared prefix)
    c = b["contrasts"]
    assert c["po1_continuity"]["delta"] == pytest.approx(0.75 - (0.5 + 0.5 + 0.25) / 3, abs=1e-3)
    assert c["po2_inertia"] == {"o2_keep_shift": 0.75, "o2_wipe_shift": 0.25, "delta": 0.5}
    assert c["positive_control_old_c5"] == 0.25
    assert c["po1_convergent_first_attempt_d4"]["delta"] == 0.5
    assert c["s1_behavioural_inertia"]["n"] == 4
    for k in ("continuity", "inertia", "inherited", "records", "dcf", "attribution", "social_share", "internalization",
              "expressions", "pragmatics", "specialization", "manipulation", "validity", "cost"):
        assert k in b
    assert b["manipulation"]["status"]["reads"] == "active" and b["manipulation"]["status"]["asks"] == "inactive"
    assert b["validity"]["prefix_digest_match"] is True
    out = json.loads((family["keep_shift"] / "outcomes.json").read_text())
    assert out["v3"]["node"] == "keep_shift"
    assert (family["keep_shift"] / "meaning.json").exists() and (family["keep_shift"] / "lineage.json").exists()


def test_trunk_retention_social_share_and_audit(family):
    b = O3.build_v3(family["trunk"])
    assert b["continuity"]["retention_ratio"] == 0.75         # W1 at C4 / W1 leavers at C3 (1.0)
    assert b["continuity"]["first_attempt_acc_d4"] == 1.0
    assert b["social_share"]["C4"] == pytest.approx(1.0 - 0.5)  # wei: P on the ablated items vs P-abl
    au = b["validity"]["prompt_audit"]
    assert au["n_prompts"] == 2 and au["n_flagged"] == 1 and au["examples"][0]["hits"] == ["regime"]
    assert b["contrasts"]["po2_inertia"]["delta"] == 0.5      # computed from siblings whatever node runs it


def test_stubs_raise():
    from backend.analysis import coding, counterfactual
    with pytest.raises(NotImplementedError):
        counterfactual.run_dcf("x")
    with pytest.raises(NotImplementedError):
        coding.code_items([], "implication")


def test_observer_not_imported_by_simulation():
    root = Path(__file__).resolve().parents[1] / "backend"
    for sub in ("simulation", "agents", "memory"):
        for p in (root / sub).rglob("*.py"):
            assert "backend.analysis" not in p.read_text(), p
