"""The injected meme cohort: the registry, its matched committed minorities, and the observer's
treatment of a study meme as the dependent variable rather than as a control.

* the registry is validated loudly (a habit that does not quote its phrase, one carrying the observer's
  probe wording, one carrying another meme's phrase);
* `controls.planted_phrase` keeps its exact legacy behaviour and return shape;
* on the real 100-agent population the four minorities are disjoint, deterministic from the seed, and
  matched on degree, weighted degree and routine reach (R6);
* an injected habit is neither world wording nor routine campus vocabulary, for BOTH roles;
* the funnel's per-meme manipulation check sees who said it, to how many hearers, how the cohort's
  airtime splits, and world text carrying a meme's distinctive words;
* the control stays a control (excluded from convention counts) while a study meme flows through the
  tiers and the live view as a first-class candidate, identified by its registry id.
"""
import json

import pytest
import yaml

from backend.agents.profile import (apply_planted, load_population, matching_table, meme_registry,
                                    plan_memes, seed_matching)
from backend.analysis import funnel as F
from backend.analysis.live import clear_cache, live_snapshot
from backend.analysis.tiers import classify_status, tier_counts, tier_of
from tests.test_live_analysis import PLANTED, _conv, _synthetic

TRAY = {"id": "tray_washer", "phrase": "blue tray", "grounding": "grounded", "breadth": "broad",
        "habit": 'has a habit of calling anything that looks fine but is not "blue tray".',
        "seeds": {"k": 2, "strategy": "spread"}, "incident": {"location": "Dining Hall"},
        "probe_gradient": {"far": "a friendship that looks fine but is not"}}
LIST = {"id": "tuesday_list", "phrase": "tuesday list", "grounding": "ungrounded", "breadth": "narrow",
        "habit": 'has a habit of calling the weekly reading queue "tuesday list".',
        "seeds": {"k": 1, "strategy": "spread"}}
POP = "configs/population/homewood100.yaml"


def _cfg(registry, **over):
    return {"seed": 42, "world_seed": 42, "population": POP, "population_size": 100,
            "memes": {"enabled": True, "registry": registry}, **over}


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield
    clear_cache()


# ------------------------------------------------------------------------------------------------ registry
def test_registry_is_validated_loudly():
    assert meme_registry({"memes": {"enabled": False, "registry": [TRAY]}}) == []
    ok = meme_registry(_cfg([TRAY, LIST]))
    assert [e["id"] for e in ok] == ["tray_washer", "tuesday_list"]
    assert ok[0]["site"] == "Dining Hall" and ok[1]["site"] is None and ok[0]["k"] == 2

    def bad(entry, msg):
        with pytest.raises(ValueError, match=msg):
            meme_registry(_cfg([entry]))
    bad({**TRAY, "habit": "has a habit of saying things look fine."}, "quote the phrase")
    bad({**TRAY, "habit": TRAY["habit"] + " a friendship that looks fine but is not"}, "probe_gradient")
    bad({**TRAY, "grounding": "sort of"}, "grounding must be")
    bad({**TRAY, "seeds": {"k": 0}}, "seeds.k")
    with pytest.raises(ValueError, match="contains meme"):          # one meme's phrase inside another's habit
        meme_registry(_cfg([TRAY, {**LIST, "habit": LIST["habit"] + ' like a "blue tray".'}]))
    with pytest.raises(ValueError, match="unique id"):
        meme_registry(_cfg([TRAY, TRAY]))


def test_positive_control_is_unchanged_and_memes_are_applied_on_top():
    profiles, _ = load_population(POP, 12)
    aid = sorted(profiles)[0]
    control = {"agent": aid, "habit": f'{profiles[aid].first_name} has a habit of calling any mess "a full pickle".'}
    before = {a: list(p.habits) for a, p in profiles.items()}
    rec = apply_planted(profiles, {"controls": {"planted_phrase": control}})
    assert rec == {"agent": aid, "habit": 'has a habit of calling any mess "a full pickle"'}   # legacy shape
    assert profiles[aid].habits == before[aid] + [rec["habit"]]

    profiles, _ = load_population(POP, 12)
    cfg = {**_cfg([TRAY, LIST], population_size=12), "controls": {"planted_phrase": control}}
    rec = apply_planted(profiles, cfg)
    assert rec["agent"] == aid and [m["id"] for m in rec["memes"]] == ["tray_washer", "tuesday_list"]
    seeds = {m["id"]: m["seeds"] for m in rec["memes"]}
    assert len(seeds["tray_washer"]) == 2 and len(seeds["tuesday_list"]) == 1
    assert aid not in seeds["tray_washer"] + seeds["tuesday_list"]        # the control agent carries one phrase
    for m in rec["memes"]:
        for a in m["seeds"]:
            assert profiles[a].habits[-1] == m["habits"][a] and m["phrase"] in profiles[a].habits[-1]


# ------------------------------------------------------------------------------------------------ R6
def test_minorities_are_disjoint_deterministic_and_matched():
    reg = [dict(TRAY, id=i, phrase=p, seeds={"k": 5, "strategy": "spread"},
                habit=f'has a habit of calling things "{p}".')
           for i, p in (("a", "phrase a"), ("b", "phrase b"), ("c", "phrase c"), ("d", "phrase d"))]
    cfg = _cfg(reg)
    profiles, _ = load_population(POP, 100)
    plan = plan_memes(profiles, cfg)
    seeds = {m["id"]: m["seeds"] for m in plan}
    assert all(len(v) == 5 for v in seeds.values())
    flat = [a for v in seeds.values() for a in v]
    assert len(set(flat)) == len(flat)                                   # disjoint
    again = plan_memes(load_population(POP, 100)[0], cfg)
    assert {m["id"]: m["seeds"] for m in again} == seeds                 # deterministic from world_seed

    rep = seed_matching(profiles, plan, cfg)
    assert rep["disjoint"] and not rep["overlap"]
    for cov, b in rep["balance"].items():
        assert b["ok"], f"{cov} not matched: |SMD| {b['max_abs_smd']} between {b['pair']}"
    assert rep["population"]["matched"]
    assert "max |SMD| degree" in matching_table(rep)

    # a different seed moves the minorities but not the matching: the design does not depend on luck
    other = seed_matching(profiles, plan_memes(profiles, {**cfg, "world_seed": 7}), cfg)
    assert {m: r["seeds"] for m, r in other["memes"].items()} != seeds
    assert other["population"]["matched"]


def test_unmatched_strategies_are_available_and_visibly_unmatched():
    reg = [dict(TRAY, id=i, phrase=p, habit=f'has a habit of calling things "{p}".',
                seeds={"k": 5, "strategy": s})
           for i, p, s in (("a", "phrase a", "high_degree"), ("b", "phrase b", "random"),
                           ("c", "phrase c", "cluster"), ("d", "phrase d", "random"))]
    cfg = _cfg(reg)
    profiles, _ = load_population(POP, 100)
    rep = seed_matching(profiles, plan_memes(profiles, cfg), cfg)
    assert rep["memes"]["a"]["degree"]["mean"] > rep["memes"]["b"]["degree"]["mean"]
    assert not rep["population"]["matched"]              # and the report says so rather than hiding it


def test_explicit_minorities_must_not_overlap():
    a = {**TRAY, "seeds": {"k": 2, "agents": ["p0001", "p0006"]}}
    b = {**LIST, "seeds": {"k": 2, "agents": ["p0006", "p0009"]}}
    profiles, _ = load_population(POP, 100)
    with pytest.raises(ValueError, match="overlaps another meme"):
        plan_memes(profiles, _cfg([a, b]))
    plan = plan_memes(profiles, _cfg([a, b], memes={"enabled": True, "registry": [a, b], "allow_overlap": True}))
    assert plan[1]["seeds"] == ["p0006", "p0009"]


# ------------------------------------------------------------------------------------------------ a run
def _run(tmp_path, memes, utts, habits=None) -> "Path":
    """The synthetic run of test_live_analysis with a meme registry and seed habits in the manifest."""
    d = _synthetic(tmp_path / "memes")
    cfg = yaml.safe_load(open(d / "config.resolved.yaml"))
    cfg["memes"] = {"enabled": True, "registry": memes}
    yaml.safe_dump(cfg, open(d / "config.resolved.yaml", "w"))
    man = json.load(open(d / "manifest.json"))
    for aid, hs in (habits or {}).items():
        man["agents"][aid]["habits"] = hs
    json.dump(man, open(d / "manifest.json", "w"))
    with open(d / "trace.jsonl", "a") as f:
        for r in utts:
            f.write(json.dumps(r) + "\n")
    return d


SEEDED = {"maya": ['has a habit of calling any mess "a full pickle"',                 # the control too
                   'has a habit of calling anything that looks fine but is not "blue tray"'],
          "leo": ['has a habit of calling anything that looks fine but is not "blue tray"'],
          "hana": ['has a habit of calling the weekly reading queue "tuesday list"'],
          "dev": ["cycles to the reservoir every morning"]}
TRAY_TALK = [*_conv("m1", 60, [("maya", ["leo", "dev"], "Careful, that is a blue tray.")]),
             *_conv("m2", 70, [("dev", ["hana"], "Classic blue tray situation.")]),
             *_conv("m3", 80, [("hana", ["dev"], "Another blue tray.")])]


def test_meme_checks_are_per_meme_and_report_crowding(tmp_path):
    from backend.analysis.rundata import RunData
    d = _run(tmp_path, [TRAY, LIST], TRAY_TALK, SEEDED)
    rd = RunData(d)
    assert [m["id"] for m in rd.memes] == ["tray_washer", "tuesday_list"]
    assert rd.memes[0]["seeds"] == ["leo", "maya"] and rd.memes[0]["seeds_source"] == "manifest_habits"

    chk = F.meme_checks(rd)
    tray, lst = chk["memes"]["tray_washer"], chk["memes"]["tuesday_list"]
    assert tray["injected"] and tray["seed_uses"] == 1 and tray["seed_speakers"] == 1
    assert tray["silent_seeds"] == 1 and tray["distinct_hearers"] == 2
    assert tray["uses"] == 3 and tray["other_uses"] == 2 and tray["other_speakers"] == 2
    assert lst["injected"] is False and lst["uses"] == 0        # a meme its minority never said
    assert chk["not_injected"] == ["tuesday_list"] and chk["crowding"]["max_share"] == 1.0
    assert chk["crowding"]["total_uses"] == 3 and chk["crowding"]["evenness"] == 0.0
    assert tray["cell"] == "groundedxbroad" and lst["cell"] == "ungroundedxnarrow"

    v = F.validity(rd.cfg, F.manipulation(rd))
    assert v["meme:tray_washer"] == "active" and v["meme:tuesday_list"] == "inactive"
    assert "seeds who said it" in F.meme_table(chk)


def test_world_saying_a_memes_words_is_reported_as_contamination(tmp_path):
    from backend.analysis.rundata import RunData
    kettle = {**TRAY, "phrase": "purple kettle",                      # the synthetic run's world text
              "habit": 'has a habit of calling anything that looks fine but is not "purple kettle".'}
    rd = RunData(_run(tmp_path, [kettle], []))
    c = F.meme_checks(rd)["memes"]["tray_washer"]["contamination"]
    assert c["phrase_in_world_text"] and c["distinctive_words_in_world_text"] == ["kettle", "purple"]
    assert F.meme_checks(rd)["contaminated"] == ["tray_washer"]
    clean = F.meme_checks(RunData(_run(tmp_path / "clean", [TRAY], [], SEEDED)))["memes"]["tray_washer"]
    assert not clean["contamination"]["phrase_in_world_text"]
    assert clean["contamination"]["distinctive_words_in_world_text"] == []


def test_injected_habit_is_neither_world_wording_nor_campus_vocabulary(tmp_path):
    from backend.analysis.emergence import vocabulary
    from backend.analysis.rundata import RunData
    rd = RunData(_run(tmp_path, [TRAY, LIST], TRAY_TALK, SEEDED))
    assert "blue tray" not in rd.world_text and "tuesday list" not in rd.world_text
    assert "a full pickle" not in rd.world_text                 # the control is excluded as before
    assert "cycles to the reservoir" in rd.world_text           # an ordinary habit still is world text
    rd.manifest["population_lexicon"]["tokens"] += ["blue", "tray", "tuesday", "list"]
    kept = vocabulary(rd)["tokens"]
    assert not {"tray", "tuesday"} & kept                       # injected-only tokens are not campus vocabulary
    assert {"dining", "hall"} <= kept


# ------------------------------------------------------------------------------------------------ roles
def test_control_and_study_meme_are_scored_differently():
    em = {"spread": True, "n_adopters_carried": 2, "n_adopters": 2, "n_independent": 0}
    verdict = {"is_convention": True}
    ctl_status, ctl_flags = classify_status(em, world=False, planted=True)
    meme_status, meme_flags = classify_status(em, world=False, planted=False, meme=True)
    assert ctl_status == "planted" and "emerged" in ctl_flags
    assert meme_status == "emerged" and meme_flags[-1] == "meme"     # never first: the seven chips stand

    assert tier_of(em, system=False, world=False, planted=True, verdict=verdict) == ("convention", ["planted_control"])
    assert tier_of(em, system=False, world=False, planted=False, verdict=verdict) == ("convention", [])
    # a study meme is NOT exempt from the wording stops: those are how R1/R2 contamination shows up
    assert tier_of(em, system=False, world=True, planted=False)[0] == "candidate"

    rows = [{"tier": "convention", "control": True, "planted": True},
            {"tier": "convention", "control": False, "planted": True, "meme_id": "tray_washer"},
            {"tier": "spreading", "control": False, "planted": False}]
    assert tier_counts(rows) == {"candidate": 0, "spreading": 1, "convention": 1}


def test_study_memes_are_first_class_in_the_live_view(tmp_path):
    d = _run(tmp_path, [TRAY, LIST], TRAY_TALK, SEEDED)
    s = live_snapshot(d, top=3)
    rec = next(e for e in s["expressions"] if e["meme_id"] == "tray_washer")
    assert rec["planted"] and not rec["control"] and "meme" in rec["flags"]
    assert rec["status"] != "planted" and rec["uses"] == 3
    assert rec["bucket"] == "expression"                    # never demoted out of the head of the list
    ctl = next(e for e in s["expressions"] if e["control"])
    assert ctl["status"] == "planted" and ctl["meme_id"] is None

    cohort = s["memes"]
    assert set(cohort["memes"]) == {"tray_washer", "tuesday_list"}
    assert cohort["memes"]["tray_washer"]["expression_id"] == rec["id"]
    assert cohort["memes"]["tuesday_list"]["expression_id"] is None      # never said, still reported
    assert cohort["not_injected"] == ["tuesday_list"]
    assert cohort["crowding"]["total_uses"] == 3
    assert s["planted"]["phrase"] == "a full pickle"        # the control block is untouched


@pytest.mark.xfail(reason="wording.Infrastructure still counts a study meme's own seed habit as system "
                          "wording, which stops every meme at 'candidate'; the control is already "
                          "exempted there and the cohort needs the same exemption (see the contract)",
                   strict=False)
def test_a_study_meme_can_reach_emerged(tmp_path):
    """Two exposure-driven adopters carried "blue tray" into later conversations, so the exposure test
    calls it emerged. It can only be reported as emerged once its own seed habit stops being counted as
    wording the system gave the agents -- the same exemption the positive control already has."""
    s = live_snapshot(_run(tmp_path, [TRAY, LIST], TRAY_TALK, SEEDED), top=8)
    rec = next(e for e in s["expressions"] if e["meme_id"] == "tray_washer")
    assert rec["emergence"]["n_adopters_carried"] == 2 and rec["emergence"]["emerged"]
    assert rec["status"] == "emerged" and rec["tier"] == "spreading"
