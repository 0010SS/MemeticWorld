"""v3 workshop world (ONTOLOGY_V3 §1, build step 2): content hygiene, day-local prefix identity,
regime-independence of the script, K1/K2 slots across regimes, the outcome model and gt, regimes and
cues, arena visibility and the Stockroom arena."""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.simulation import content as C
from backend.simulation import regimes as R
from backend.simulation import structures as ST
from backend.simulation import workshop as W
from backend.simulation.world import ARENAS, Clock, default_arena

ROOT = Path(__file__).resolve().parents[1]
HIDDEN_IDS = re.compile(r"\b(K[0-3]|LENS|DAMP|BELT|AIR|WARP|M1|M2|regime|mapping)\b|j\d\d\.\d")
CAUSE_WORDS = re.compile(r"\b(lens|focus|damp|wet|moist\w*|humid\w*|dry|dried|belt|nozzle|air|warp\w*|"
                         r"flat\w*|pins?|speed|slow\w*|tighten\w*|supplier)\b", re.I)


def cfg(days=7, seed=13, **ws):
    c = {"seed": seed, "world_seed": None, "start_date": "2026-09-14", "day_start": "07:30", "day_end": "22:30",
         "tick_minutes": 15, "simulation_days": days, "workshop": {"enabled": True, **ws},
         "regimes": {"mapping": "auto", "schedule": [{"day": 1, "regime": "A"}, {"day": 5, "regime": "B"}],
                     "cues": [{"day": 4, "time": "16:45", "arena": "Stockroom", "text_key": "new_supplier"}]}}
    return c


PROFILES = {a: None for crew in W.DEFAULT_CREWS.values() for a in crew}


def gen(c):
    return W.generate(c, PROFILES, Clock(c))


def jobs(recs):
    return [r for r in recs if r.kind == "job"]


# ------------------------------------------------------------------------------------ content hygiene
def _pools(pack):
    """(class key, surfaces): K3 counts as one class per mapping."""
    out = []
    for cls, s in pack.SURFACES.items():
        if isinstance(s, dict):
            out += [(f"{cls}:{m}", v) for m, v in s.items()]
        else:
            out.append((cls, s))
    return out


@pytest.mark.parametrize("name", C.PACKS)
def test_surface_counts_and_lengths(name):
    pack = C.load(name)
    for key, surf in _pools(pack):
        assert len(surf) == 8, key
        for t in surf:
            n = len(t.split())
            assert 10 <= n <= 25, (key, n, t)


@pytest.mark.parametrize("name", C.PACKS)
def test_surface_bigram_hygiene(name):
    pack = C.load(name)
    pools = _pools(pack)
    owner: dict = {}
    for key, surf in pools:
        cls = key.split(":")[0]
        count: dict = {}
        for t in surf:
            for bg in ST.content_bigrams(t):
                count[bg] = count.get(bg, 0) + 1
                owner.setdefault(bg, set()).add(cls)
        over = {b: n for b, n in count.items() if n > 2}
        assert not over, (key, over)
    shared = {b: sorted(o) for b, o in owner.items() if len(o) > 1}
    assert not shared, shared


@pytest.mark.parametrize("name", C.PACKS)
def test_surfaces_name_no_cause_fix_label_or_hidden_id(name):
    pack = C.load(name)
    for key, surf in _pools(pack):
        for t in surf:
            assert not CAUSE_WORDS.search(t), (key, t)
            assert not HIDDEN_IDS.search(t), (key, t)
            assert not ST.QUOTED.search(t), (key, t)
            m = ST.NICKNAME.search(t)
            assert not m or m.group(1).split()[0].lower() in ST._NICK_OK, (key, t)
            assert "F4" not in t


@pytest.mark.parametrize("path", sorted((ROOT / "backend/analysis/battery").glob("items_lb1_*.yaml")))
def test_battery_heldout_shares_no_bigram_with_world_surfaces(path):
    doc = yaml.safe_load(path.read_text())
    pack = C.load(doc.get("pack", "laser_alpha"))
    world = set()
    for _, surf in _pools(pack):
        for t in surf:
            world |= ST.content_bigrams(t)
    for it in doc["items"]:
        if it.get("wording") == "heldout" and it.get("text"):
            assert not (ST.content_bigrams(it["text"]) & world), (it["id"], ST.content_bigrams(it["text"]) & world)


def test_all_world_text_is_free_of_hidden_ids():
    for name in C.PACKS:
        pack = C.load(name)
        texts = list(pack.ACTIONS.values()) + list(pack.ACTION_PAST.values()) + list(pack.ODDITIES) + [pack.SUCCESS]
        texts += [v for f in pack.FAIL.values() for v in (f.values() if isinstance(f, dict) else [f])]
        texts += list(C.CUES.values()) + [p for pr in C.PROJECTS for p in pr]
        for t in texts:
            assert not HIDDEN_IDS.search(t), t
    for t in C.CUES.values():
        assert not re.search(r"fault|broke|problem|lens|damp|wet", t, re.I)


# ------------------------------------------------------------------------------------ generation
def test_job_record_shape():
    recs = gen(cfg())
    js = jobs(recs)
    assert len(js) == 7 * 8
    j = js[0].to_dict()
    for k in ("id", "kind", "generator", "day", "slot", "shift", "start_tick", "operator", "project", "menu_order",
              "surface", "code", "oddity", "hidden"):
        assert k in j
    assert j["id"] == "j01.0" and j["start_tick"] == 6 and j["shift"] == "am"
    assert [x.start_tick for x in js if x.day == 4] == [180 + k for k in (6, 10, 14, 18, 24, 28, 32, 36)]
    assert [x.shift for x in js if x.day == 2] == ["am"] * 4 + ["pm"] * 4
    assert set(j["hidden"]["class"]) == {"A", "B"} and len(j["hidden"]["u_attempt"]) == 2
    assert all(sorted(m) == sorted(C.load("laser_alpha").ACTIONS) for m in j["menu_order"])
    assert W.to_jsonl(recs).count("\n") == len(recs)
    back = [W.record_from_dict(r.to_dict()) for r in recs]
    assert [b.to_dict() for b in back] == [r.to_dict() for r in recs]


def test_prefix_identity_for_any_horizon():
    a, b = gen(cfg(days=4)), gen(cfg(days=7))
    da = [r.to_dict() for r in a]
    db = [r.to_dict() for r in b if r.day <= 4]
    assert da == db


def test_script_is_regime_independent():
    c1, c2 = cfg(), cfg()
    c2["regimes"]["schedule"] = [{"day": 1, "regime": "A"}]                      # no-shift arm
    c2["records"] = {"enabled": True, "transitions": [{"day": 5, "mode": "wipe"}]}
    assert [r.to_dict() for r in gen(c1)] == [r.to_dict() for r in gen(c2)]


def test_k1_k2_slots_equal_across_regimes_and_k3_from_k0():
    for seed in (1, 2, 13):
        for j in jobs(gen(cfg(seed=seed))):
            a, b = j.hidden["class"]["A"], j.hidden["class"]["B"]
            assert a in ("K0", "K1", "K2") and b in ("K0", "K1", "K2", "K3")
            if a in ("K1", "K2"):
                assert b == a
            else:
                assert b in ("K0", "K3")


def test_switches_shift_nothing_else():
    base = gen(cfg())
    code = gen(cfg(panel_code={"enabled": True}))
    scr = gen(cfg(causal="scrambled"))
    strip = lambda r, *ks: {k: v for k, v in r.to_dict().items() if k not in ks}  # noqa: E731
    assert [strip(r, "code") for r in base] == [strip(r, "code") for r in code]
    assert any(j.code for j in jobs(code)) and not any(j.code for j in jobs(base))
    assert all(j.hidden["class"]["A"] == "K1" for j in jobs(code) if j.code)
    for x, y in zip(jobs(base), jobs(scr)):
        assert x.hidden["class"] == y.hidden["class"] and x.hidden["u"] == y.hidden["u"]
        assert x.menu_order == y.menu_order and x.operator == y.operator


def test_rates_roughly_as_contract():
    js = [j for s in range(1, 9) for j in jobs(gen(cfg(seed=s, days=7)))]
    n = len(js)
    frac = lambda f: sum(map(f, js)) / n  # noqa: E731
    assert 0.65 < frac(lambda j: j.hidden["fault"]) < 0.85
    assert 0.44 < frac(lambda j: j.hidden["class"]["A"] == "K1") < 0.62
    assert 0.08 < frac(lambda j: j.hidden["class"]["B"] == "K3") < 0.22


def test_operator_rota_and_newcomers():
    js = [j for j in jobs(gen(cfg())) if j.day == 2]
    am = [j.operator for j in js if j.shift == "am"]
    assert set(am) == {"maya", "dev", "hana"} and am[0] == am[3]
    ros = lambda day: {"am": ["maya", "dev", "wei"], "pm": ["ethan", "leo", "jordan"], "stores": ["priya"],  # noqa: E731
                       "newcomers": ["wei"] if day in (4, 5) else []}
    c = cfg()
    js = [j for j in W.generate(c, PROFILES, Clock(c), roster=ros) if j.kind == "job" and j.shift == "am"]
    for d in (4, 5):
        ops = [j.operator for j in js if j.day == d]
        assert ops[0] == "wei" and ops.count("wei") == 2


def test_roster_object_protocol():
    class Ro:
        def members(self, role, day):
            return {"am_crew": ["a1", "a2", "a3"], "pm_crew": ["p1", "p2", "p3"], "stores": ["s1"]}[role]

        def is_newcomer(self, aid, day):
            return aid == "a3" and day == 1
    c = cfg(days=1)
    js = jobs(W.generate(c, {}, Clock(c), roster=Ro()))
    assert js[0].operator == "a3" and {j.operator for j in js[4:]} == {"p1", "p2", "p3"}


def test_cue_and_tally_records():
    recs = gen(cfg())
    cues = [r for r in recs if r.kind == "cue"]
    assert len(cues) == 1 and cues[0].tick == 180 + 37 and cues[0].arena == "Stockroom"
    assert cues[0].text == C.CUES["new_supplier"]
    tallies = [r for r in recs if r.kind == "tally"]
    assert [t.tick for t in tallies] == [60 * d + 40 for d in range(7)]


# ------------------------------------------------------------------------------------ outcome model / gt
def test_outcome_table_and_gt():
    om = W.OutcomeModel(cfg())
    assert om.p("lens", "LENS") == 0.85 and om.p("dry", "LENS") == 0.10 and om.p("rerun", "DAMP") == 0.10
    assert om.p("slow", "LENS") == 0.35 and om.p("slow", "BELT") == 0.20 and om.p("stop", "LENS") == 0.0
    assert om.p("belt", None) == 1.0
    assert W.gt("K1", "A", "M1") == "lens" and W.gt("K1", "B", "M1") == "dry"
    assert W.gt("K1", "A", "M2") == "dry" and W.gt("K1", "B", "M2") == "lens"
    assert W.gt("K2", "A", "M1") == W.gt("K2", "B", "M2") == "belt"
    assert W.gt("K3", "B", "M1") == "dry" and W.gt("K3", "B", "M2") == "lens"
    assert W.gt("K0", "A", "M1") == "rerun"
    assert W.gt("K1", "A", "M1", content="laser_beta") == "air"
    assert om.score("lens", "K1", "M1") == "old" and om.score("dry", "K1", "M1") == "new"
    assert om.score("slow", "K1", "M1") == "other"
    lever = W.OutcomeModel({"seed": 1, "workshop": {"success": {"match": 0.9, "other_fix": 0.05, "rerun": 0.05}}})
    assert lever.p("lens", "LENS") == 0.9 and lever.p("slow", "LENS") == 0.35
    assert W.OutcomeModel({"success": {"match": 0.9}}).p("dry", "DAMP") == 0.9    # bare workshop block


def test_resolve_is_coupled_by_predrawn_uniforms():
    c = cfg()
    om = W.OutcomeModel(c)
    k1 = [j for j in jobs(gen(c)) if j.hidden["class"]["A"] == "K1"]
    for j in k1:
        u = j.hidden["u_attempt"][0]
        for a in ("rerun", "lens", "dry", "belt", "slow"):
            assert om.resolve(j, 1, a, "A") == ("success" if u < om.p(a, j.cause_at("A")) else "fail")
        assert om.resolve(j, 1, "stop", "A") == "defer"
    with pytest.raises(ValueError):
        om.resolve(k1[0], 3, "lens", "A")
    # M1 (seed 13 odd): K1's cause swaps LENS -> DAMP
    assert all(j.cause_at("A") == "LENS" and j.cause_at("B") == "DAMP" for j in k1)


def test_scrambled_causes_are_independent_of_class():
    js = [j for s in range(1, 9) for j in jobs(gen(cfg(seed=s, causal="scrambled")))]
    causes = {j.cause_at("A") for j in js if j.hidden["class"]["A"] == "K2"}
    assert causes == {"LENS", "DAMP", "BELT"}


# ------------------------------------------------------------------------------------ regimes
def test_regimes_mapping_schedule_active():
    assert R.mapping({"seed": 13}) == "M1" and R.mapping({"seed": 14}) == "M2"
    assert R.mapping({"seed": 14, "world_seed": 3}) == "M1"
    assert R.mapping({"seed": 1, "regimes": {"mapping": "M2"}}) == "M2"
    c = cfg()
    assert [R.active(d, c) for d in range(1, 8)] == ["A"] * 4 + ["B"] * 3
    assert R.changes(c) == [{"day": 5, "from": "A", "to": "B"}]
    rev = {"regimes": {"schedule": [{"day": 1, "regime": "A"}, {"day": 5, "regime": "B"}, {"day": 7, "regime": "A"}]}}
    assert R.active(7, rev) == "A" and R.active(6, rev) == "B"
    assert R.active(3, {}) == "A"
    with pytest.raises(ValueError):
        R.schedule({"regimes": {"schedule": [{"day": 2, "regime": "A"}]}})
    g = R.Regimes(c)
    assert g.trace(5) == {"day": 5, "regime": "B", "mapping": "M1", "changed": True}
    assert R.causes("M2", C.load("laser_alpha")) == {"A": "DAMP", "B": "LENS"}


# ------------------------------------------------------------------------------------ beats
def test_runtime_beats_are_clean_world_text():
    c = cfg(panel_code={"enabled": True})
    recs = gen(c)
    om = W.OutcomeModel(c)
    for j in jobs(recs):
        for reg in ("A", "B"):
            b = W.start_beat(j, reg, c, "Dev")
            assert b["location"] == "Research Lab" and b["arena"] == "Makerspace" and b["movers"] == [j.operator]
            assert all(f["visibility"] == "arena" and f["id"].startswith(f"{j.id}.b0.f") for f in b["facts"])
            kinds = [f["kind"] for f in b["facts"]]
            assert ("code" in kinds) == (j.code and j.cls_at(reg) == "K1")
            texts = [f["text"] for f in b["facts"]]
            if W.needs_decision(j, reg):
                act = j.menu_order[0][0]
                out = om.resolve(j, 1, act, reg)
                texts.append(W.outcome_beat(j, 1, j.start_tick + 1, act, out, reg, c, "Dev")["facts"][0]["text"])
            for t in texts:
                assert not HIDDEN_IDS.search(t) and not re.search(r"0\.\d", t), t
    tal = [r for r in recs if r.kind == "tally"][0]
    day1 = [j for j in jobs(recs) if j.day == 1]
    res = [(j, i not in (1, 3)) for i, j in enumerate(day1)]
    txt = W.tally_beat(tal, res, c)["facts"][0]["text"]
    assert txt.startswith("End-of-day tally at the laser: 6 of 8 jobs delivered;") and txt.endswith("were not.")
    assert W.tally_text([(j, True) for j in day1]) == "End-of-day tally at the laser: 8 of 8 jobs delivered."
    cue = [r for r in recs if r.kind == "cue"][0]
    cb = W.cue_beat(cue, c)
    assert cb["arena"] == "Stockroom" and cb["facts"][0]["text"] == C.CUES["new_supplier"]
    assert W.outcome_text("Dev", "lens", True, "K1", "M1").startswith("Dev cleaned the focus lens")
    assert W.symptom_text(jobs(recs)[0].to_dict(), "K2", "M1", c) in C.load("laser_alpha").SURFACES["K2"]


def test_menu_text_uses_world_wording():
    c = cfg()
    j = jobs(gen(c))[0]
    m = W.menu_text(j, 2, c)
    assert [x[0] for x in m] == list("ABCDEF") and [x[1] for x in m] == j.menu_order[1]
    assert {x[2] for x in m} == set(C.load("laser_alpha").ACTIONS.values())


def test_trace_builders():
    c = cfg()
    j = [j for j in jobs(gen(c)) if j.hidden["class"]["A"] == "K1"][0]
    t = W.trace_job_truth(j, "A", c)
    assert t["gt"] == "lens" and t["cause"] == "LENS" and t["mapping"] == "M1"
    assert "hidden" not in W.trace_job_start(j) and "hidden" not in j.public()
    e = W.trace_job_end(j, [{"attempt": 1, "action": "stop"}], False)
    assert e["result"] == "defer"


# ------------------------------------------------------------------------------------ world / perception
def test_stockroom_arena_appended():
    assert ARENAS["Research Lab"][-1] == "Stockroom" and "Makerspace" in ARENAS["Research Lab"]
    assert default_arena("Research Lab") == "Dry Lab"


def _agent(aid, loc, arena):
    return SimpleNamespace(
        id=aid, state=SimpleNamespace(location=loc, arena=arena, in_conversation=None),
        profile=SimpleNamespace(rel=lambda a: SimpleNamespace(familiarity=0.0)),
        ctx=SimpleNamespace(agents={}), mods=SimpleNamespace(modify_attention=lambda ag, f, p: p))


def test_arena_visibility():
    from backend.agents.perception import attention_prob
    pc = {"perception": {"base_attention": 0.8, "other_arena_factor": 0.5, "busy_factor": 0.5,
                         "familiarity_weight": 0.2}}
    beat = {"location": "Research Lab", "arena": "Makerspace"}
    fact = {"visibility": "arena", "salience": 0.6, "involves": ["dev"]}
    assert attention_prob(_agent("dev", "Research Lab", "Stockroom"), fact, beat, pc) == 1.0   # participant
    assert attention_prob(_agent("maya", "Research Lab", "Makerspace"), fact, beat, pc) > 0
    assert attention_prob(_agent("priya", "Research Lab", "Stockroom"), fact, beat, pc) == 0.0
    assert attention_prob(_agent("leo", "Quad", "Lawn"), fact, beat, pc) == 0.0
    allv = dict(fact, visibility="all")
    assert attention_prob(_agent("priya", "Research Lab", "Stockroom"), allv, beat, pc) > 0      # v2 unchanged
