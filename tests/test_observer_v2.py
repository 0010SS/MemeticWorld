"""Observer v2 (ontology v2 §4): provenance, emergence, grounding, funnel/validity, outcomes, compare."""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from backend.analysis import compare as CMP
from backend.analysis import outcomes as O
from backend.analysis.emergence import emergence_for, fisher_exact, lexicon_flag, world_match
from backend.analysis.funnel import diagnose
from backend.analysis.grounding import analyze_grounding, bh, permutation_test, stratified_permutations, z_surfaces
from backend.analysis.pipeline import analyze
from backend.analysis.rundata import RunData, precedes

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_KEYS = {"run_id", "condition", "observer", "n_candidates", "n_emerged", "max_emerged_adoption", "n_grounded",
                "grounded", "emerged", "funnel", "manipulation", "validity"}


# ------------------------------------------------------------------ statistics

def _fisher_brute(a, b, c, d, alternative):
    r1, r2, c1 = a + b, c + d, a + c
    probs = {x: math.comb(r1, x) * math.comb(r2, c1 - x) for x in range(max(0, c1 - r2), min(r1, c1) + 1)}
    tot = sum(probs.values())
    if alternative == "greater":
        return sum(v for x, v in probs.items() if x >= a) / tot
    return sum(v for v in probs.values() if v <= probs[a]) / tot


def test_fisher_exact_known_values():
    # Fisher's lady tasting tea: one-sided 17/70, two-sided 34/70
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(17 / 70)
    assert fisher_exact(3, 1, 1, 3, "two-sided") == pytest.approx(34 / 70)
    # scipy.stats.fisher_exact([[6, 2], [1, 4]]): greater 0.08624708624708627, two-sided 0.10256410256410256
    assert fisher_exact(6, 2, 1, 4) == pytest.approx(0.08624708624708627)
    assert fisher_exact(6, 2, 1, 4, "two-sided") == pytest.approx(0.10256410256410256)
    assert fisher_exact(0, 5, 5, 0, "less") == pytest.approx(1 / 252)
    for t in [(2, 7, 3, 1), (0, 0, 3, 4), (5, 0, 0, 5), (1, 6, 2, 9)]:
        for alt in ("greater", "two-sided"):
            assert fisher_exact(*t, alt) == pytest.approx(_fisher_brute(*t, alt))


def test_benjamini_hochberg():
    assert bh([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    q = bh([0.001, 0.2, 0.9, 0.04, 0.5])
    assert q == pytest.approx([0.005, 1 / 3, 0.9, 0.1, 0.625])
    assert bh([]) == []
    assert max(bh([0.9, 0.95])) <= 1.0


def test_stratified_null_preserves_strata():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 60)
    strata = rng.integers(0, 6, 60)
    P = stratified_permutations(labels, strata, 200, np.random.default_rng(1))
    assert P.shape == (200, 60)
    for s in np.unique(strata):
        idx = strata == s
        want = np.sort(labels[idx])
        assert all((np.sort(row[idx]) == want).all() for row in P)
    assert len({tuple(r) for r in P}) > 100          # the permutations actually vary


# ------------------------------------------------------------------ grounding on a synthetic world

class FakeRD:
    """Just what analyze_grounding reads: events, provenance, cfg, ticks_per_day (+ manifest circles)."""

    def __init__(self, events, prov, permutations=1000, mode=None, circles=None):
        self.events = events
        self.provenance = prov
        self.cfg = {"seed": 7, "analysis": {"permutations": permutations, "fdr_q": 0.1}}
        if mode:
            self.cfg["latent_events"] = {"assignment": {"mode": mode}}
        self.manifest = {"circles": circles} if circles is not None else {}
        self.ticks_per_day = 60


def _world(per_family=2):
    """2 circles x 3 days, every family equally often in every circle-day."""
    events, k = {}, 0
    for circle in ("c_a", "c_b"):
        for day in range(3):
            for fam in ("E1", "E2", "E3", "E4"):
                for _ in range(per_family):
                    events[f"ev{k:03d}"] = {"id": f"ev{k:03d}", "latent_type": fam, "circle": circle,
                                            "start_tick": day * 60 + k % 40, "holdout": False}
                    k += 1
    return events


def _cand(cid, links, speakers=("a", "b", "c")):
    usages = [{"utterance_id": f"{cid}.u{i}", "speaker": speakers[i % len(speakers)], "tick": 200 + i}
              for i in range(len(links))]
    prov = {u["utterance_id"]: {"referent_event_ids": ls} for u, ls in zip(usages, links)}
    return {"id": cid, "canonical_form": cid, "usages": usages, "first_occurrence": {"tick": 200}}, prov


def test_grounding_flags_family_phrase_not_circle_phrase():
    ev = _world()
    e2 = [e for e, x in ev.items() if x["latent_type"] == "E2"]
    circ_a = [e for e, x in ev.items() if x["circle"] == "c_a"]
    rng = np.random.default_rng(3)
    fam, p1 = _cand("family_phrase", [[e] for e in e2[:8]])
    circ, p2 = _cand("circle_phrase", [[e] for e in circ_a[:24]])
    rand, p3 = _cand("random_phrase", [[e] for e in rng.choice(sorted(ev), 10, replace=False)])
    few, p4 = _cand("two_uses", [[e] for e in e2[:2]])
    g = analyze_grounding([fam, circ, rand, few], FakeRD(ev, {**p1, **p2, **p3, **p4}))
    c = g["candidates"]
    # outside assignment mode `home` the family never depends on the circle: day strata only
    assert g["strata_by"] == "day" and g["n_strata"] == 3 and g["identifiable"] is True
    assert c["family_phrase"]["z_surfaces"] == 8           # no skins: every event is its own surface unit
    assert c["family_phrase"]["grounded"] and c["family_phrase"]["best_family"] == "E2"
    assert c["family_phrase"]["share"] == 1.0 and c["family_phrase"]["p_value"] < 0.01
    assert not c["circle_phrase"]["grounded"] and c["circle_phrase"]["share"] == pytest.approx(0.25)
    assert not c["random_phrase"]["grounded"]
    assert not c["two_uses"]["testable"] and c["two_uses"]["q_value"] is None and not c["two_uses"]["grounded"]


def test_circle_stratification_absorbs_circle_family_confound():
    """Circle A holds mostly E1 and E3 (home assignment with imperfect casting: some instances of other
    families have a circle-A protagonist). A phrase that is only about circle A looks E1-specific
    against a run-wide null, but not against the within-circle null."""
    fams = ["E1"] * 6 + ["E2"] * 2 + ["E3"] * 6 + ["E4"] * 2
    labels = np.array([int(f[1]) - 1 for f in fams] * 3)
    circle = np.array(([0] * 8 + [1] * 8) * 3)
    day = np.repeat([0, 1, 2], 16)
    v = (circle == 0).astype(float)[None, :]          # one usage per circle-A event
    n = v.sum(axis=1)
    strat = permutation_test(v, n, labels, circle * 3 + day, 4, 1000, np.random.default_rng(0))
    flat = permutation_test(v, n, labels, np.zeros_like(labels), 4, 1000, np.random.default_rng(0))
    assert strat["observed"][0] == pytest.approx(0.75)
    assert strat["p"][0] > 0.5
    assert flat["p"][0] < 0.01


def test_grounding_is_fast_enough():
    ev = _world(per_family=3)                          # 72 events
    ids = sorted(ev)
    rng = np.random.default_rng(1)
    cands, prov = [], {}
    for i in range(40):
        c, p = _cand(f"m{i:02d}", [list(rng.choice(ids, rng.integers(1, 4), replace=False)) for _ in range(12)])
        cands.append(c)
        prov.update(p)
    t = time.time()
    g = analyze_grounding(cands, FakeRD(ev, prov, permutations=1000))
    assert time.time() - t < 3.0
    assert g["n_testable"] == 40


def _skinned_world():
    """4 families x 3 train skins x 2 days; skin e1_a recurs 4 times (siblings share their wording)."""
    events, k = {}, 0
    for day in range(2):
        for fam in ("E1", "E2", "E3", "E4"):
            for j in range(3):
                reps = 2 if (fam, j) == ("E1", 0) else 1
                for _ in range(reps):
                    events[f"ev{k:03d}"] = {"id": f"ev{k:03d}", "latent_type": fam, "skin": f"{fam.lower()}_{'abc'[j]}",
                                            "start_tick": day * 60 + k % 50, "holdout": False}
                    k += 1
    return events


def test_same_skin_siblings_are_one_surface_unit():
    """Talk about ONE recurring incident type (a skin) co-links all its same-skin siblings. With labels
    permuted per event, 4 same-family siblings looked like 4 independent draws (p ~ 1/64); per surface
    unit they are one draw, so the phrase is not grounded, and it names only one surface of E1."""
    ev = _skinned_world()
    sib = sorted(e for e, x in ev.items() if x["skin"] == "e1_a")
    assert len(sib) == 4
    one, p1 = _cand("one_incident", [sib] * 6)                         # every usage co-links all 4 siblings
    e1 = sorted(e for e, x in ev.items() if x["latent_type"] == "E1" and x["skin"] != "e1_a")
    many, p2 = _cand("family_phrase", [[e] for e in e1] + [[sib[0]]])   # E1 across its 3 skins
    g = analyze_grounding([one, many], FakeRD(ev, {**p1, **p2}))
    assert g["n_events"] == 26 and g["n_units"] == 12          # 4 families x 3 skins, recurring over 2 days
    c = g["candidates"]
    assert c["one_incident"]["share"] == 1.0 and c["one_incident"]["p_value"] > 0.5
    assert c["one_incident"]["z_surfaces"] == 1 and not c["one_incident"]["grounded"]
    assert c["family_phrase"]["z_surfaces"] == 3 and c["family_phrase"]["grounded"]
    # the event-level null (the old behaviour) calls the one-incident phrase significant
    ids = sorted(ev)
    V = np.array([[1.0 if e in sib else 0.0 for e in ids]]) * 6 / 4
    lab = np.array([int(ev[e]["latent_type"][1]) - 1 for e in ids])
    day = np.array([ev[e]["start_tick"] // 60 for e in ids])
    assert permutation_test(V, np.array([6.0]), lab, day, 4, 1000, np.random.default_rng(0))["p"][0] < 0.05


def test_z_surfaces_votes_one_unit_per_usage():
    ev = {"a1": {"latent_type": "E1", "skin": "e1_a"}, "a2": {"latent_type": "E1", "skin": "e1_a"},
          "b1": {"latent_type": "E1", "skin": "e1_b"}, "x1": {"latent_type": "E2", "skin": "e2_a"}}
    unit = {e: f"{x['latent_type']}|{x['skin']}" for e, x in ev.items()}
    # every usage is mostly about skin a (b only rides along): one surface
    assert z_surfaces([["a1", "a2", "b1"]] * 5, "E1", ev, unit) == ["E1|e1_a"]
    assert z_surfaces([["a1"], ["b1", "x1"], ["x1"]], "E1", ev, unit) == ["E1|e1_a", "E1|e1_b"]


def _home_world(fams_by_circle, free_cast=0.0, days=3, per=8, seed=0):
    """Assignment mode `home`: every event's `circle` field is its family's home circle; P is a member of
    that circle except for a `free_cast` share whose P is drawn from the whole population."""
    rng = np.random.default_rng(seed)
    circles = {"c_a": ["a1", "a2", "a3"], "c_b": ["b1", "b2", "b3"]}
    pop = sum(circles.values(), []) + ["f1", "f2"]
    home = {f: c for c, fs in fams_by_circle.items() for f in fs}
    fams = sorted(home)
    events, k = {}, 0
    for day in range(days):
        for i in range(per):
            fam = fams[(k + day) % len(fams)]
            p = rng.choice(pop) if rng.random() < free_cast else rng.choice(circles[home[fam]])
            events[f"ev{k:03d}"] = {"id": f"ev{k:03d}", "latent_type": fam, "circle": home[fam], "start_tick": day * 60 + i,
                                    "roles": {"P": {"agent": str(p), "name": str(p)}}, "holdout": False}
            k += 1
    return events, circles


def test_home_mode_strata_use_protagonist_circle():
    """In `home` mode the event's `circle` field is a function of its family: with one family per
    circle, strata on it hold a single family each and nothing can be permuted. Strata on the
    protagonist's ACTUAL circle keep the free-cast instances as variation: a family phrase is grounded,
    while a phrase that only tracks circle A (mostly, but not only, E1) is not."""
    ev, circles = _home_world({"c_a": ["E1"], "c_b": ["E2"]}, free_cast=0.4, days=3, per=12, seed=0)
    e1 = sorted(e for e, x in ev.items() if x["latent_type"] == "E1")
    fam, p1 = _cand("family_phrase", [[e] for e in e1])
    in_a = sorted(e for e, x in ev.items() if x["roles"]["P"]["agent"].startswith("a"))
    circ, p2 = _cand("circle_phrase", [[e] for e in in_a])
    g = analyze_grounding([fam, circ], FakeRD(ev, {**p1, **p2}, mode="home", circles=circles))
    assert g["strata_by"] == "circle x day" and g["circle_source"] == "protagonist" and g["identifiable"] is True
    c = g["candidates"]
    assert c["family_phrase"]["grounded"] and c["family_phrase"]["p_value"] < 0.01
    assert 0.5 < c["circle_phrase"]["share"] < 1.0 and c["circle_phrase"]["p_value"] > 0.5
    assert not c["circle_phrase"]["grounded"]
    # without circle membership the event's `circle` field is the only stratifier: not identifiable
    g_old = analyze_grounding([fam], FakeRD(ev, p1, mode="home"))
    assert g_old["circle_source"] == "event_field" and g_old["identifiable"] is False
    assert g_old["candidates"]["family_phrase"]["p_value"] == 1.0


def test_home_mode_family_determined_by_circle_is_not_identifiable():
    """One family per circle and perfect home casting: family = f(circle). Grounding is flagged as not
    identifiable (not 'nothing grounded'), and outcomes report n_grounded as null."""
    ev, circles = _home_world({"c_a": ["E1"], "c_b": ["E2"]}, free_cast=0.0)
    e1 = sorted(e for e, x in ev.items() if x["latent_type"] == "E1")
    fam, p1 = _cand("family_phrase", [[e] for e in e1])
    g = analyze_grounding([fam], FakeRD(ev, p1, mode="home", circles=circles))
    assert g["identifiable"] is False and g["family_determined_by_circle"] is True and g["mixed_strata_share"] == 0
    assert "not identifiable" in g["warning"]
    assert g["candidates"]["family_phrase"]["p_value"] == 1.0 and not g["candidates"]["family_phrase"]["grounded"]
    rd = type("RD", (), {"agents": {"a1": {}}, "cfg": {"seed": 1}, "manifest": {}, "dir": Path("x")})()
    o = O.build(rd, [fam], {}, g, {"metrics": {}, "manipulation": {}, "validity": {}}, {})
    assert o["n_grounded"] is None and o["grounding"]["identifiable"] is False and o["grounding"]["warning"]
    # outside home mode the same world is stratified by day and stays identifiable
    assert analyze_grounding([fam], FakeRD(ev, p1, circles=circles))["identifiable"] is True


# ------------------------------------------------------------------ emergence

POP = ["a", "b", "c", "d", "e", "f", "g", "h"]


def _u(i, tick, speaker, listeners, conv=None):
    return {"utterance_id": f"u{i}", "tick": tick, "speaker": speaker, "listeners": listeners,
            "conversation_id": conv, "idx": 0}


def test_emergence_distinguishes_spread_from_independent_use():
    spread = [_u(0, 1, "a", ["b", "c"], "c1"), _u(1, 5, "b", ["d"], "c2"), _u(2, 9, "c", ["e"], "c3"),
              _u(3, 12, "d", ["a"], "c4")]
    e = emergence_for(spread, POP)
    assert e["originator"] == "a" and e["n_adopters"] == 3 and e["n_independent"] == 0
    assert e["n_adopters_carried"] == 3
    assert e["n_exposed"] == 4 and e["n_unexposed"] == 3     # e heard it but never used it
    assert e["emerged"]
    assert e["fisher_p"] == pytest.approx(fisher_exact(3, 1, 0, 3), abs=1e-5)

    independent = [_u(0, 1, "a", [], None), _u(1, 4, "b", ["h"], None), _u(2, 8, "c", [], None),
                   _u(3, 11, "d", ["g"], None)]
    e = emergence_for(independent, POP)
    assert e["n_adopters"] == 0 and e["n_independent"] == 3 and not e["emerged"]
    assert e["fisher_p"] == 1.0

    # heard and echoed only inside the same conversation: adoption, but not carried to a new one
    echo = [_u(0, 1, "a", ["b"], "c1"), _u(1, 1, "b", ["a"], "c1") | {"idx": 1}]
    e = emergence_for(echo, POP)
    assert e["n_adopters"] == 1 and e["n_adopters_carried"] == 0

    # routine vocabulary never counts as emerged, however it spreads
    assert not emergence_for(spread, POP, in_lexicon=True)["emerged"]
    assert lexicon_flag("dining hall", {"dining", "hall"}, set())
    assert not lexicon_flag("full pickle", {"full"}, set())


def _r(tick, speaker, listeners, ev="ev1"):
    return {"utterance_id": f"t{tick:04d}:remark:{speaker}:{ev}.b0", "tick": tick, "speaker": speaker,
            "listeners": listeners, "conversation_id": None, "idx": 0}


def test_same_tick_remark_does_not_expose_a_conversation_turn():
    """Remarks overheard at tick t reach memory only after tick t's conversations (phase 06), so a
    tick-t conversation turn cannot have been caused by a tick-t remark: b is an independent user."""
    us = [_u(0, 4, "b", ["a"], "d1t0004c0"), _r(4, "a", ["b"])]
    e = emergence_for(us, POP)
    assert e["originator"] == "a" and e["n_adopters"] == 0 and e["n_independent"] == 1
    assert not precedes(us[1], us[0]) and not precedes(us[0], us[1])
    # the same remark one tick earlier does expose b
    e = emergence_for([_u(0, 5, "b", ["a"], "d1t0005c0"), _r(4, "a", ["b"])], POP)
    assert e["n_adopters"] == 1 and e["n_independent"] == 0 and e["n_adopters_carried"] == 1


def test_simultaneous_witness_remarks_are_independent():
    """Three witnesses react to the same beat at the same tick: decided in parallel, none heard the
    others first. Whoever sorts first is the originator; the other two are independent, nothing emerged."""
    pop = ["dev", "maya", "sofia", "leo", "hana", "priya", "ethan", "jordan"]
    rs = [_r(7, s, [x for x in ("dev", "maya", "sofia") if x != s], "ev003") for s in ("sofia", "maya", "dev")]
    e = emergence_for(rs, pop)
    assert e["originator"] == "dev" and e["n_adopters"] == 0 and e["n_adopters_carried"] == 0
    assert e["n_independent"] == 2 and not e["emerged"] and not e["spread"]
    assert precedes(_r(6, "a", []), rs[0]) and not precedes(rs[0], rs[1])


def test_group_chat_echo_is_not_emergence():
    """a says it in a group chat, b and c echo it on the next turns and never use it again: two
    adopters by the old rule (emerged), zero carried adopters now."""
    g = [_u(0, 40, "a", ["b", "c", "d"], "g1"), _u(1, 40, "b", ["a", "c", "d"], "g1") | {"idx": 1},
         _u(2, 40, "c", ["a", "b", "d"], "g1") | {"idx": 2}]
    e = emergence_for(g, POP)
    assert e["n_adopters"] == 2 and e["n_echo_only"] == 2 and e["n_adopters_carried"] == 0
    assert e["n_exposed"] == 3 and not e["emerged"] and e["fisher_p"] == 1.0
    # b carries it into a new conversation the next tick -> one carried adopter; c then echoes b there,
    # which is still an echo (c heard it earlier in g2 itself)
    g2 = g + [_u(3, 41, "b", ["c", "e"], "g2"), _u(4, 41, "c", ["b", "e"], "g2") | {"idx": 1}]
    e = emergence_for(g2, POP)
    assert e["carried_adopters"] == ["b"] and e["n_adopters_carried"] == 1 and not e["emerged"]
    # c brings it up first in yet another conversation later: now two carried adopters -> emerged
    e = emergence_for(g2 + [_u(5, 50, "c", ["f"], "g3")], POP)
    assert e["carried_adopters"] == ["b", "c"] and e["spread"] and e["emerged"]
    # a later use inside the SAME conversation it was heard in is not carried either
    e = emergence_for([_u(0, 40, "a", ["b"], "g1"), _u(1, 42, "b", ["a"], "g1") | {"idx": 5}], POP)
    assert e["n_adopters"] == 1 and e["n_adopters_carried"] == 0


def test_world_wording_never_emerges():
    spread = [_u(0, 1, "a", ["b", "c"], "c1"), _u(1, 5, "b", ["d"], "c2"), _u(2, 9, "c", ["e"], "c3")]
    e = emergence_for(spread, POP, in_world_text=True)
    assert e["spread"] and not e["emerged"] and e["n_adopters_carried"] == 2
    segs = [["ethan", "submitted", "a", "problem", "set", "to", "the", "wrong", "course"], ["the", "night", "shuttle"]]
    assert world_match(["night", "shuttle"], segs) == "verbatim"
    assert world_match("submitted to the wrong".split(), segs) == "gapped"      # 3 tokens skipped
    assert world_match("submitting to the wrong".split(), segs) == "gapped"     # inflection folded
    assert world_match("night shuttles".split(), segs) == "gapped"
    assert world_match("submitted wrong".split(), segs) is None                 # 5 tokens skipped: too far
    assert world_match("full pickle".split(), segs) is None
    assert world_match("shuttle night".split(), segs) is None                   # order matters


def test_outcomes_count_only_non_world_emergence():
    cands = [{"id": f"m{i}", "canonical_form": f"p{i}", "features": {}} for i in range(3)]
    base = {"n_adopters": 3, "n_adopters_carried": 2, "n_independent": 0, "n_exposed": 3, "fisher_p": 0.1,
            "spread": True, "in_lexicon": False}
    em = {"m0": {**base, "in_world_text": False, "emerged": True},
          "m1": {**base, "in_world_text": True, "emerged": False},                     # event wording
          "m2": {**base, "in_world_text": False, "in_lexicon": True, "emerged": False}}  # campus vocabulary
    rd = type("RD", (), {"agents": {a: {} for a in "abcdefgh"}, "cfg": {"seed": 1}, "manifest": {}, "dir": Path("x")})()
    o = O.build(rd, cands, em, {"candidates": {}, "identifiable": True},
                {"metrics": {}, "manipulation": {}, "validity": {}}, {})
    assert o["n_emerged"] == 1 and [x["id"] for x in o["emerged"]] == ["m0"]
    assert o["n_spread_world"] == 1 and o["spread_world"][0]["id"] == "m1"
    assert o["max_emerged_adoption"] == pytest.approx(3 / 8)       # (1 + carried adopters) / population
    assert o["n_grounded"] == 0


# ------------------------------------------------------------------ provenance, funnel, validity on a synthetic run dir

def _write_run(d: Path, cfg_over=None, trace_extra=(), events=None, manifest_extra=None):
    d.mkdir(parents=True, exist_ok=True)
    agents = {a: {"id": a, "name": f"{a.title()} Test", "background": "", "habits": [], "routine": []}
              for a in ("maya", "leo", "dev")}
    man = {"run_id": d.name, "ticks": 120, "ticks_per_day": 60, "tick_minutes": 15, "start": "2026-09-14T07:30:00",
           "agents": agents, "groups": {}, "config": {}, "stats": {"llm": {"calls": 0, "errors": 0}},
           "world": {"graph": {"Dorm": []}, "arenas": {"Dorm": ["Room"]}}, **(manifest_extra or {})}
    cfg = {"seed": 3, "population": "configs/population/homewood8.yaml", "llm": {"backend": "mock", "model": "m"},
           "analysis": {"permutations": 200, "fdr_q": 0.1}}
    for k, v in (cfg_over or {}).items():
        cfg[k] = v
    json.dump(man, open(d / "manifest.json", "w"))
    yaml.safe_dump(cfg, open(d / "config.resolved.yaml", "w"))
    events = events if events is not None else [
        {"id": "ev000", "latent_type": "E1", "start_tick": 2, "holdout": False, "circle": "c_lab", "cast_from_home": True,
         "referents": [{"id": "transit:3", "domain": "transit", "name": "the night shuttle", "reused": False}],
         "roles": {"P": {"agent": "maya", "name": "Maya"}},
         "beats": [{"idx": 0, "tick": 2, "facts": [{"id": "ev000.b0.f0", "text": "Maya waited twenty minutes at the night shuttle stop.",
                                                    "kind": "action", "visibility": "all"},
                                                   {"id": "ev000.b0.f1", "text": "Maya had misread the timetable.",
                                                    "kind": "inner", "visibility": "all"}]},
                   {"idx": 2, "tick": 30, "facts": [{"id": "ev000.b2.f0", "text": "Maya spilled bubble tea on the sofa cushions.",
                                                     "kind": "action", "visibility": "all"}]}]},
        {"id": "ev001", "latent_type": "E2", "start_tick": 50, "holdout": True, "circle": None, "cast_from_home": False,
         "referents": [{"id": "transit:3", "domain": "transit", "name": "the night shuttle", "reused": True}],
         "beats": [{"idx": 0, "tick": 50, "facts": [{"id": "ev001.b0.f0", "text": "Leo left a violin case by the shuttle shelter.",
                                                     "kind": "action", "visibility": "all"}]}]}]
    with open(d / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    trace = [
        {"type": "memory_encoded", "tick": 2, "agent": "maya", "node_id": "maya:m1", "source_type": "perception",
         "text": "x", "originating_event_ids": ["ev000"], "utterance_ids": []},
        {"type": "memory_encoded", "tick": 5, "agent": "leo", "node_id": "leo:m1", "source_type": "conversation",
         "text": "Maya said the night shuttle timetable", "originating_event_ids": ["ev000"], "utterance_ids": ["c1.u0"]},
        {"type": "viewpoint", "tick": 3, "agent": "dev", "originating_event_ids": ["ev000"], "dropped": ["ev000.b0.f1"],
         "facts": [{"id": "ev000.b0.f0", "perceived": "Someone kept glancing at a crumpled timetable near the kiosk."}]},
        {"type": "conversation", "tick": 4, "id": "c1", "participants": ["maya", "leo"], "trigger": {"agent": "maya", "event_ids": ["ev000"]},
         "transcript": []},
        {"type": "utterance", "tick": 4, "id": "c1.u0", "conversation_id": "c1", "idx": 0, "speaker": "maya",
         "listeners": ["leo"], "text": "I waited twenty minutes at the shuttle stop, unreal.", "retrieved": ["maya:m1"]},
        {"type": "utterance", "tick": 20, "id": "t0020:remark:dev:x", "conversation_id": None, "idx": 0, "speaker": "dev",
         "listeners": [], "text": "That crumpled timetable by the kiosk was a mess.", "retrieved": []},
        {"type": "utterance", "tick": 25, "id": "c2.u0", "conversation_id": "c2", "idx": 0, "speaker": "leo",
         "listeners": ["dev"], "text": "Bubble tea on the sofa cushions? Also a violin case somewhere.", "retrieved": ["leo:m1"]},
        {"type": "conversation", "tick": 25, "id": "c2", "participants": ["leo", "dev"], "trigger": {"agent": None, "text": None, "topic": "catchup"},
         "transcript": []},
        *trace_extra,
    ]
    with open(d / "trace.jsonl", "w") as f:
        for r in trace:
            f.write(json.dumps(r) + "\n")
    return d


def test_provenance_links(tmp_path):
    rd = RunData(_write_run(tmp_path / "run"))
    p = rd.provenance
    # shared content tokens with the fact text + the conversation's trigger; perception memory -> direct
    assert p["c1.u0"]["referent_event_ids"] == ["ev000"]
    assert p["c1.u0"]["direct_event_ids"] == ["ev000"] and p["c1.u0"]["carried_event_ids"] == []
    # linked only through dev's viewpoint rendering ("crumpled timetable ... kiosk")
    assert p["t0020:remark:dev:x"]["referent_event_ids"] == ["ev000"]
    # beat 2 (tick 30) is not released yet at tick 25; ev001 has not started: no links
    assert p["c2.u0"]["referent_event_ids"] == []
    assert p["c2.u0"]["carried_event_ids"] == ["ev000"] and p["c2.u0"]["direct_event_ids"] == []


def test_funnel_manipulation_and_validity(tmp_path):
    extra = [{"type": "reminding", "tick": 40, "agent": "leo", "reminded_of": "leo:m1", "node_id": "leo:m9",
              "originating_event_ids": ["ev001"]},
             {"type": "memory_encoded", "tick": 41, "agent": "dev", "node_id": "dev:m3", "source_type": "overheard",
              "text": "y", "originating_event_ids": [], "utterance_ids": ["c2.u0"],
              "wordings": [{"phrase": "bubble tea", "heard_from": "leo", "self_produced": False}]},
             {"type": "memory_link", "tick": 40, "agent": "leo", "from": "leo:m9", "to": "leo:m1", "mechanism": "reminding"}]
    cfg = {"reminding": {"enabled": True}, "priming": {"enabled": True}, "need": {"enabled": False},
           "memory": {"verbatim": {"enabled": True}, "encoding_noise": 0.3},
           "conversation": {"catchup": {"enabled": True}, "group": {"enabled": True}},
           "latent_events": {"referents": {"enabled": True}, "assignment": {"mode": "balanced"}, "link_visibility": 0.5},
           "modules": {"emotion": True, "prestige_bias": False, "conformity": True, "social_reward": True}}
    checks = {"emotion": {"active": True, "counters": {"appraisals": 4}},
              "conformity": {"active": False, "counters": {"modify_retrieval": 0}, "mean_shift": 0.3}}
    rd = RunData(_write_run(tmp_path / "run", cfg, extra, manifest_extra={"manipulation": checks}))
    d = diagnose(rd)
    m, v = d["manipulation"], d["validity"]
    assert m["reminding"] == {"asked": 1, "linked": 1, "same_event": 0, "cross_event": 1, "no_event": 0}
    assert m["verbatim"]["stuck"] == 1 and m["verbatim"]["stuck_other"] == 1
    assert m["memory_links"] == {"reminding": 1}
    assert m["casting"]["cast_from_home_rate"] == 1.0 and m["referents"]["reuse_rate"] == 0.5
    assert m["viewpoint"]["dropped"] == 1 and m["link_visibility"]["inner_public"] == 1
    assert v["reminding"] == "active" and v["verbatim"] == "active" and v["catchup"] == "active"
    assert v["referents"] == "active" and v["assignment"] == "active" and v["link_visibility"] == "active"
    assert v["priming"] == "inactive" and v["group"] == "inactive"      # enabled but never fired
    assert v["need"] == "off" and v["module:prestige_bias"] == "off" and v["module:emotion"] == "active"
    assert v["module:conformity"] == "inactive" and "module:social_reward" not in v   # no counters recorded
    f = d["metrics"]
    assert f["events"] == 2 and f["utterances"] == 3 and f["catchup_conversations"] == 1
    assert f["utt_about_events_pct"] == pytest.approx(66.7)


def test_planted_phrase_is_not_routine_vocabulary(tmp_path):
    from backend.analysis.emergence import vocabulary
    habit = 'Maya has a habit of calling any mess "a full pickle".'
    extra = [{"type": "utterance", "tick": 50, "id": "c9.u0", "conversation_id": "c9", "idx": 0, "speaker": "maya",
              "listeners": ["leo"], "text": "What a full pickle.", "retrieved": []}]
    # Keep this vocabulary fixture independent of new characters' unrelated prose.
    d = _write_run(tmp_path / "run", {"population_size": 8,
                   "controls": {"planted_phrase": {"agent": "maya", "habit": habit}}}, extra,
                   manifest_extra={"population_lexicon": {"tokens": ["dining", "hall", "full", "pickle", "mess"]}})
    rd = RunData(d)
    v = vocabulary(rd)
    assert v["source"] == "manifest" and {"dining", "hall"} <= v["tokens"] and not {"full", "pickle"} & v["tokens"]
    m = diagnose(rd)
    assert m["manipulation"]["planted"]["uses_by_agent"] == 1 and m["validity"]["planted_phrase"] == "active"


def test_provenance_does_not_link_through_campus_vocabulary(tmp_path):
    """Two shared tokens that are both population vocabulary ({dining, hall}) are routine talk, not
    talk about an incident; a line that also names the event's own thing still links."""
    ev = [{"id": "ev000", "latent_type": "E1", "start_tick": 2, "holdout": False,
           "referents": [{"id": "dining:0", "domain": "dining", "name": "the dining hall card reader", "reused": False}],
           "roles": {"P": {"agent": "maya", "name": "Maya"}},
           "beats": [{"idx": 0, "tick": 2, "facts": [{"id": "ev000.b0.f0", "kind": "action", "visibility": "all",
                                                      "text": "The dining hall card reader rejected Maya's card twice."}]}]}]
    lines = {"c5.u0": "Meet you at the dining hall for lunch?",
             "c5.u1": "The dining hall card reader rejected me too.",
             "c5.u2": "Is the hall reader broken again?"}
    extra = [{"type": "utterance", "tick": 10, "id": uid, "conversation_id": "c5", "idx": i, "speaker": ("leo", "dev")[i % 2],
              "listeners": [("dev", "leo")[i % 2]], "text": t, "retrieved": []} for i, (uid, t) in enumerate(lines.items())]
    lex = {"tokens": ["dining", "hall", "lunch", "meet"]}
    rd = RunData(_write_run(tmp_path / "run", trace_extra=extra, events=ev, manifest_extra={"population_lexicon": lex}))
    p = rd.provenance
    assert p["c5.u0"]["referent_event_ids"] == []                  # {dining, hall}: campus words only
    assert p["c5.u1"]["referent_event_ids"] == ["ev000"]           # + reader, rejected
    assert p["c5.u2"]["referent_event_ids"] == ["ev000"]           # {hall, reader}: one campus word + the event's own
    # without the lexicon the logistics line would link through {dining, hall}
    from backend.analysis.provenance import EventText
    assert EventText(rd, lexicon=set()).link(lines["c5.u0"], 10) == ["ev000"]


def test_emergence_world_text_covers_facts_viewpoints_and_referents(tmp_path):
    from backend.analysis.emergence import analyze_emergence
    rd = RunData(_write_run(tmp_path / "run"))
    u = {"utterance_id": "c1.u0", "tick": 4, "speaker": "maya", "listeners": ["leo"], "conversation_id": "c1"}
    cands = [{"id": f"m{i}", "canonical_form": p, "usages": [u]} for i, p in
             enumerate(["the night shuttle", "crumpled timetable", "waiting twenty minutes", "full pickle"])]
    em = analyze_emergence(cands, rd)
    assert em["m0"]["world_match"] == "verbatim"          # referent name
    assert em["m1"]["world_match"] == "verbatim"          # dev's viewpoint rendering
    assert em["m2"]["world_match"] == "gapped"            # "waited twenty minutes" (fact text), inflected
    assert em["m3"]["world_match"] is None and not em["m3"]["in_world_text"]
    assert all(em[f"m{i}"]["in_world_text"] and not em[f"m{i}"]["emerged"] for i in range(3))


def _planted_run(tmp_path, manifest_planted=False):
    """A run whose manifest holds maya's habits as the engine writes them: apply_planted's output."""
    from types import SimpleNamespace

    from backend.agents.profile import apply_planted
    habit = 'Maya has a habit of calling any mess "a full pickle".'
    cfg = {"controls": {"planted_phrase": {"agent": "maya", "habit": habit}}}
    stub = SimpleNamespace(first_name="Maya", habits=["grabs coffee before class"])
    applied = apply_planted({"maya": stub}, cfg)
    assert stub.habits[-1] == 'has a habit of calling any mess "a full pickle"' != habit
    extra = [{"type": "utterance", "tick": 50 + i, "id": f"c9.u{i}", "conversation_id": "c9", "idx": i,
              "speaker": ("maya", "leo")[i % 2], "listeners": [("leo", "maya")[i % 2]], "text": "What a full pickle.",
              "retrieved": []} for i in range(3)]
    d = _write_run(tmp_path / "run", cfg, extra, manifest_extra={"planted": applied} if manifest_planted else None)
    man = json.load(open(d / "manifest.json"))
    man["agents"]["maya"]["habits"] = list(stub.habits)
    json.dump(man, open(d / "manifest.json", "w"))
    return RunData(d)


@pytest.mark.parametrize("manifest_planted", [False, True])
def test_planted_habit_is_not_world_wording(tmp_path, manifest_planted):
    """The engine stores the planted habit normalised (first name and final period dropped); the
    observer must leave exactly that string out of world text, so the positive control is not scored
    as factual repetition (x0.35) and pushed off the candidate list."""
    from backend.analysis.candidates import CandidateExtractor
    rd = _planted_run(tmp_path, manifest_planted)
    assert rd.planted["habit"] == 'has a habit of calling any mess "a full pickle"' and rd.planted["phrase"] == "a full pickle"
    assert "full pickle" not in rd.world_text and "grabs coffee before class" in rd.world_text
    st = {"uses": [u["id"] for u in rd.utterances if "pickle" in u["text"]], "speakers": {"maya", "leo"}}
    sc = CandidateExtractor(rd, {}).score(("full", "pickle"), st)
    assert sc["factual_repetition"] is False


def test_probe_personas_match_the_simulated_ones(tmp_path):
    """Probe agents are rebuilt like the engine builds them (planted habit, generated topology) and the
    manifest's record of each profile wins, so the probed Maya's ISS carries her planted habit."""
    from conftest import run_sim

    from backend.analysis.probes import load_probe_agents, probe_profiles
    habit = 'Maya has a habit of calling any mess "a full pickle".'
    sim = run_sim(tmp_path, {"controls": {"planted_phrase": {"agent": "maya", "habit": habit}},
                             "topology": {"mode": "generated"}}, name="planted")
    rd = RunData(sim.run_dir)
    profs = probe_profiles(rd)
    for aid, prof in profs.items():
        m = rd.agents[aid]
        assert prof.habits == m["habits"], aid
        assert [(r.time, r.location, r.activity) for r in prof.routine] == \
            [(r["time"], r["location"], r["activity"]) for r in m["routine"]], aid
        assert {b: r.familiarity for b, r in prof.relationships.items()} == \
            {b: r["familiarity"] for b, r in m["relationships"].items()}, aid
    # the engine-style rebuild alone (no manifest record) gives the same personas
    rd2 = RunData(sim.run_dir)
    rd2.manifest = {**rd.manifest, "agents": {}}
    alone = probe_profiles(rd2)
    assert all(alone[a].habits == profs[a].habits and alone[a].ga_daily_plan_req() == profs[a].ga_daily_plan_req()
               for a in profs)
    agents = load_probe_agents(rd, sim.embed)
    assert "a full pickle" in agents["maya"].iss()
    assert agents["maya"].scratch.lifestyle == sim.agents["maya"].profile.ga_lifestyle()
    assert all(agents[a].scratch.daily_plan_req == sim.agents[a].profile.ga_daily_plan_req() for a in agents)


# ------------------------------------------------------------------ end to end

def _check_outcomes(run_dir: Path) -> dict:
    o = json.load(open(run_dir / "outcomes.json"))
    assert OUTCOME_KEYS <= set(o)
    assert o["observer"]["analysis_version"] == "v2"
    assert set(o["validity"].values()) <= {"active", "inactive", "off"}
    a = json.load(open(run_dir / "analysis.json"))
    for k in ("emergence", "grounding", "funnel", "evaluation", "observer", "analysis_version"):
        assert k in a
    return o


def test_analyze_mock_simulation_writes_outcomes(tmp_path):
    from conftest import run_sim
    sim = run_sim(tmp_path, {"analysis": {"observer": {"backend": "mock", "model": "observer-x"}}}, name="obs")
    analyze(sim.run_dir, probes=False, verbose=False)
    o = _check_outcomes(sim.run_dir)
    assert o["observer"] == {"backend": "mock", "model": "observer-x", "analysis_version": "v2"}
    # an explicit override beats the config's observer spec
    analyze(sim.run_dir, llm_backend="mock", llm_model="observer-y", probes=False, verbose=False)
    assert _check_outcomes(sim.run_dir)["observer"]["model"] == "observer-y"


V1_RUNS = ["dev_live_c3", "c2_no_events_sonnet_s1"]


@pytest.mark.parametrize("name", V1_RUNS)
def test_analyze_copy_of_v1_run(tmp_path, name):
    src = ROOT / "runs" / name
    if not (src / "trace.jsonl").exists():
        pytest.skip(f"{src} not available")
    dst = tmp_path / name
    dst.mkdir()
    for f in ("manifest.json", "config.resolved.yaml", "trace.jsonl", "events.jsonl", "memory_meta.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    analyze(dst, llm_backend="mock", probes=False, verbose=False)
    o = _check_outcomes(dst)
    assert o["n_candidates"] > 0
    if name == "c2_no_events_sonnet_s1":
        assert o["n_grounded"] == 0 and o["funnel"]["events"] == 0
    assert not (src / "outcomes.json").exists()


def test_compare_refuses_mixed_observers(tmp_path):
    from conftest import run_sim
    sim = run_sim(tmp_path, name="cmp")
    a, b = tmp_path / "a", tmp_path / "b"
    for d, model in ((a, "sonnet"), (b, "haiku")):
        shutil.copytree(sim.run_dir, d)
        analyze(d, llm_backend="mock", llm_model=model, probes=False, verbose=False)
    with pytest.raises(CMP.ObserverMismatch, match="different observer specs"):
        CMP.compare([a, b])
    rows = CMP.compare([a, b], allow_mixed_observers=True)
    assert {r["observer"]["model"] for r in rows} == {"sonnet", "haiku"}
    assert all("n_emerged" in r and "n_grounded" in r for r in rows)
    assert len(CMP.compare([a])) == 1
    assert "n_emerged" in CMP.markdown(rows)


def test_analyze_with_v2_world_script_events(tmp_path):
    """A mock run whose events.jsonl is replaced by a real v2 world script (circles, referents,
    inner facts, holdout skins): every observer block handles the v2 record."""
    from conftest import run_sim
    try:
        from backend.agents.profile import load_population
        from backend.config import load_config
        from backend.simulation import world_script as WS
        from backend.simulation.circles import load_circles
        from backend.simulation.world import Clock
    except ImportError as e:
        pytest.skip(f"world script not available: {e}")
    sim = run_sim(tmp_path, name="v2ev")
    cfg = load_config("configs/baseline.yaml", {"simulation_days": 1, "latent_events": {
        "event_rate": 0.3, "assignment": {"mode": "balanced"}, "referents": {"enabled": True}}})
    profiles = load_population(cfg["population"], 8)[0]
    insts = WS.generate(cfg, profiles, load_circles(cfg["population"], list(profiles)), Clock(cfg))
    with open(sim.run_dir / "events.jsonl", "w") as f:
        for i in insts:
            f.write(json.dumps(i.ground_truth()) + "\n")
    analyze(sim.run_dir, llm_backend="mock", probes=False, verbose=False)
    o = _check_outcomes(sim.run_dir)
    assert o["manipulation"]["casting"]["source"] == "events" and o["manipulation"]["referents"]["n"] == len(insts)
    g = json.load(open(sim.run_dir / "analysis.json"))["grounding"]
    # balanced assignment: the family does not depend on the circle, so day strata only
    assert g["strata_by"] == "day" and g["circle_source"] is None
    assert g["n_units"] <= g["n_events"] == len(insts) and g["identifiable"] in (True, None)
