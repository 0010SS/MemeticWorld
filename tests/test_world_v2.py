"""World v2 (ontology v2 §1): pre-generated world script, structure x skin, circles, assignment,
referents, holdout, link visibility, the EventInstance record, and skin hygiene."""
import copy
import hashlib
import json
import re
import zlib
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from backend.agents.profile import load_population
from backend.config import deep_merge, load_config
from backend.simulation import latent_events as LE
from backend.simulation import structures as ST
from backend.simulation import world_script as WS
from backend.simulation.circles import load_circles, validate_circles
from backend.simulation.lexicon import population_lexicon
from backend.simulation.referents import CATEGORIES, HOLDOUT_DOMAINS, TRAIN_DOMAINS
from backend.simulation.rngs import seed_rng, world_seed
from backend.simulation.scheduler import plan_day, routine_position
from backend.simulation.world import Clock

POP = "configs/population/homewood8.yaml"
DOMAINS = TRAIN_DOMAINS + HOLDOUT_DOMAINS


# ------------------------------------------------------------------ fixtures
def fixture_skin(fam: str, dom: str) -> dict:
    """A synthetic skin whose content words are unique to (family, domain), so it passes hygiene."""
    w = lambda k: f"{fam.lower()}{dom}{k}"  # noqa: E731
    r2 = ST.second_role(fam)
    return {"key": f"{fam.lower()}_{dom}", "family": fam, "domain": dom, "holdout": dom in HOLDOUT_DOMAINS,
            "slots": {"X": [f"a {w('xa')} box", f"a {w('xb')} box"]},
            "facts": {"n0": f"{{P}} {w('a')} {w('b')} at {{R}} in the morning with {{X}}.",
                      "n0_private": f"{{P}} had not {w('c')} the {w('d')} before leaving that day.",
                      "n1": f"{{{r2}}} also {w('e')} the {w('f')} later on that afternoon.",
                      "n2": f"{{P}} {w('g')} the {w('h')} near {{R}} again before the evening."}}


FIXTURE_SKINS = [fixture_skin(f, d) for f in ST.FAMILIES for d in DOMAINS]


@pytest.fixture(scope="module")
def pop():
    profiles, _ = load_population(POP, 8)
    return profiles, load_circles(POP, list(profiles))


def make_cfg(**over) -> dict:
    base = {"seed": 7, "world_seed": None, "simulation_days": 3, "llm": {"backend": "mock"},
            "latent_events": {"event_rate": 0.2}}
    return load_config(None, deep_merge(base, over))


def gen(pop, skins=FIXTURE_SKINS, **over):
    cfg = make_cfg(**over)
    profiles, circles = pop
    return WS.generate(cfg, profiles, circles, Clock(cfg), skins=skins), cfg


def records(insts):
    return [i.ground_truth() for i in insts]


# ------------------------------------------------------------------ rngs / plans
def test_seed_rng_matches_engine_scheme():
    ref = np.random.default_rng([zlib.crc32(str(p).encode()) for p in (42, "plan", "maya", 1)])
    assert np.array_equal(seed_rng(42, "plan", "maya", 1).random(5), ref.random(5))
    try:
        from backend.simulation.engine import _seed_rng
    except ImportError:
        return
    assert np.array_equal(seed_rng(3, "timing").random(5), _seed_rng(3, "timing").random(5))


def test_world_seed_fallback():
    assert world_seed({"seed": 5, "world_seed": None}) == 5
    assert world_seed({"seed": 5, "world_seed": 9}) == 9
    assert world_seed({"seed": 5}) == 5


def test_unpinned_locations_are_planned_positions(pop):
    """Every beat is at P's planned position; plans are world (seeded by world_seed, as in the engine)."""
    insts, cfg = gen(pop, seed=3, world_seed=11)
    profiles, _ = pop
    clock = Clock(cfg)
    assert insts

    def planned(aid, tick):
        day = clock.day_of(tick)
        plan = plan_day(SimpleNamespace(cfg=cfg, profile=profiles[aid]), clock, seed_rng(11, "plan", aid, day))
        pos = routine_position(SimpleNamespace(profile=profiles[aid], day_plan=plan), tick % clock.ticks_per_day)
        return pos["location"], pos["arena"]

    for e in insts:
        p = e.roles["P"]["agent"]
        for b in e.beats:
            assert (b["location"], b["arena"]) == planned(p, b["tick"])


def test_pinned_location_is_used(pop):
    skins = copy.deepcopy(FIXTURE_SKINS)
    for s in skins:
        s["locations"] = {"n1": "Gym"}
    insts, _ = gen(pop, skins=skins)
    assert insts and all(e.beats[1]["location"] == "Gym" for e in insts)
    assert all(e.beats[1]["arena"] == "Main Floor" for e in insts)


# ------------------------------------------------------------------ determinism / CRN
def test_same_world_seed_same_script(pop, tmp_path):
    a, _ = gen(pop)
    b, _ = gen(pop)
    assert records(a) == records(b)
    c, _ = gen(pop, world_seed=8)
    assert records(a) != records(c)
    d, _ = gen(pop, seed=7, world_seed=7)
    assert records(a) == records(d)


@pytest.mark.parametrize("schedule", ["balanced", "bernoulli"])
def test_seed_never_changes_the_world_script(pop, schedule):
    """Routine plans are part of the world (seeded by world_seed), so with world_seed fixed a different
    `seed` changes nothing, including where beats happen (who can witness them)."""
    a, _ = gen(pop, seed=1, world_seed=5, simulation_days=4, latent_events={"schedule": schedule})
    b, _ = gen(pop, seed=2, world_seed=5, simulation_days=4, latent_events={"schedule": schedule})
    assert a and records(a) == records(b)


def test_cognitive_and_social_keys_do_not_change_script(pop, tmp_path):
    a, _ = gen(pop)
    b, _ = gen(pop, reminding={"enabled": True}, memory={"verbatim": {"enabled": True}, "encoding_noise": 0.9},
               conversation={"catchup": {"enabled": True}, "group": {"enabled": True}},
               need={"enabled": True}, priming={"enabled": True}, llm={"model": "sonnet"},
               modules={"emotion": True, "conformity": True})
    assert WS.save(a, tmp_path / "a.jsonl") == WS.save(b, tmp_path / "b.jsonl")


@pytest.mark.parametrize("schedule", ["balanced", "bernoulli"])
def test_world_mechanisms_keep_timing_and_families_aligned(pop, schedule):
    base, _ = gen(pop, simulation_days=4, latent_events={"schedule": schedule})
    for over in ({"assignment": {"mode": "home"}}, {"referents": {"enabled": True}}, {"link_visibility": 1.0},
                 {"structure": "scrambled"}, {"structure": "scrambled", "referents": {"enabled": True}}):
        other, _ = gen(pop, simulation_days=4, latent_events={"schedule": schedule, **over})
        ta = {e.start_tick: e for e in base}
        tb = {e.start_tick: e for e in other}
        common = set(ta) & set(tb)
        assert len(common) >= 0.9 * max(len(ta), len(tb)), over
        if schedule == "balanced":
            assert set(ta) == set(tb), over                                   # timing never moves
        assert all(ta[t].holdout == tb[t].holdout for t in common)
        if "structure" not in over:
            # referent reuse picks the reused referent's domain skin; every other instance keeps its skin
            kept = [t for t in common if not any(r["reused"] for r in tb[t].referents)]
            assert all(ta[t].skin == tb[t].skin for t in kept), over
            assert len(kept) < len(common) or "referents" not in over
        assert all(ta[t].latent_type == tb[t].latent_type for t in common), over   # family stream shared


# ------------------------------------------------------------------ shape
def test_shape_invariants(pop):
    insts, cfg = gen(pop, simulation_days=4)
    clock = Clock(cfg)
    tpd = clock.ticks_per_day
    assert len(insts) > 20
    assert [e.id for e in insts] == [f"ev{i:03d}" for i in range(len(insts))]
    active = []
    for e in insts:
        assert e.generator == "script_v2" and e.structure_mode == "real"
        assert [b["tick"] - e.start_tick for b in e.beats] == ST.BEAT_OFFSETS
        assert e.start_tick // tpd == e.beats[-1]["tick"] // tpd            # never crosses a day end
        assert [len(b["facts"]) for b in e.beats] == [2, 1, 1]
        f0, f0p = e.beats[0]["facts"]
        (f1,), (f2,) = e.beats[1]["facts"], e.beats[2]["facts"]
        assert [f["salience"] for f in (f0, f0p, f1, f2)] == [0.55, 0.30, 0.50, 0.55]
        assert [f["kind"] for f in (f0, f0p, f1, f2)] == ["action", "inner", "action", "action"]
        p = e.roles["P"]["agent"]
        assert p and f0p["visibility"] == [p] and f0["visibility"] == f1["visibility"] == f2["visibility"] == "all"
        role2 = "Q" if e.latent_type == "E3" else "S"
        assert set(e.roles) == {"P", role2}
        second = e.roles[role2]["agent"]
        # involvement and movers come from the shape only (identical for every family)
        assert f0["involves"] == [p] and f0p["involves"] == [p] and f2["involves"] == [p]
        assert f1["involves"] == ([second] if second else [])
        assert [b["movers"] for b in e.beats] == [[p], sorted({p, second} - {None}), [p]]
        for b in e.beats:
            assert b["location"] and b["arena"]
            for i, f in enumerate(b["facts"]):
                assert f["id"] == f"{e.id}.b{b['idx']}.f{i}"
                assert "{" not in f["text"] and "}" not in f["text"]
                assert set(f) == {"id", "text", "salience", "visibility", "kind", "involves"}
        assert e.skin.split("_")[0] == e.latent_type.lower()
        assert e.schema == ST.SCHEMAS[e.latent_type]["id"]
        assert e.narrative == " ".join(f["text"] for b in e.beats for f in b["facts"])
        # busy: an agent is in at most one scheduled instance at a time
        people = {a for a in (p, second) if a}
        for end, ags in active:
            if end >= e.start_tick:
                assert not people & ags
        active.append((e.beats[-1]["tick"], people))


def test_second_actor_relatedness(pop):
    profiles, _ = pop
    insts, _ = gen(pop, simulation_days=5)
    for e in insts:
        p = e.roles["P"]["agent"]
        if "Q" in e.roles and e.roles["Q"]["agent"]:
            assert profiles[p].rel(e.roles["Q"]["agent"]).familiarity < WS.RELATED
        if "S" in e.roles and e.roles["S"]["agent"]:
            assert profiles[p].rel(e.roles["S"]["agent"]).familiarity >= WS.RELATED
        for r, v in e.roles.items():
            if v["agent"] is None:
                assert v["name"] in (WS.NPC_S if r == "S" else WS.NPC_Q)
    assert any(e.latent_type == "E3" for e in insts)


def test_event_rate_zero_gives_empty_script(pop):
    insts, _ = gen(pop, skins=[], latent_events={"event_rate": 0.0})
    assert insts == []


def _real_skins_or_skip():
    from backend.simulation import skins as SK
    real = SK.load_skins()
    if SK.load_errors():
        pytest.skip(f"skin modules not loadable: {SK.load_errors()}")
    return real


def test_presence_and_involvement_do_not_depend_on_family(pop):
    """E4 used to name P in n1 (P forced to S and a 'participant' in every E4 instance) and E2 named S in n2
    far more often than E3. Now every family has the same beat shape: P is at every beat, the second actor is
    brought to beat 1 only, involvement is shape-only, and a fact names nobody it does not involve (so the
    viewpoint name guard, which works from `involves`, covers every name)."""
    profiles, _ = pop
    _real_skins_or_skip()
    shape = defaultdict(Counter)
    for structure in ("real", "scrambled"):
        for ws in range(6):
            insts, _ = gen(pop, skins=None, world_seed=ws, simulation_days=3,
                           latent_events={"structure": structure, "assignment": {"mode": "balanced"}})
            for e in insts:
                p = e.roles["P"]["agent"]
                second = next(v["agent"] for r, v in e.roles.items() if r != "P")
                sig = (tuple(p in b["movers"] for b in e.beats),                 # P forced to every beat
                       tuple(second in b["movers"] for b in e.beats) if second else None,
                       [f["involves"] for b in e.beats for f in b["facts"]]
                       == [[p], [p], [second] if second else [], [p]])
                shape[e.latent_type][sig] += 1
                for b in e.beats:
                    for f in b["facts"]:
                        named = {a for a in profiles if re.search(rf"\b{re.escape(profiles[a].first_name)}\b", f["text"])}
                        assert named <= set(f["involves"]), (e.id, f["text"], f["involves"])
    assert set(shape) == set(ST.FAMILIES)
    for fam, sigs in shape.items():
        assert set(sigs) <= {((True, True, True), (False, True, False), True), ((True, True, True), None, True)}, \
            (fam, sigs)


def test_balanced_schedule_counts_strata_blocks_and_holdout(pop):
    for ws, days, rate, frac in ((0, 3, 0.2, 0.25), (3, 5, 0.12, 0.25), (9, 4, 0.3, 0.5)):
        insts, cfg = gen(pop, world_seed=ws, simulation_days=days,
                         latent_events={"event_rate": rate, "holdout_frac": frac})
        clock = Clock(cfg)
        tpd, usable = clock.ticks_per_day, clock.ticks_per_day - ST.SPAN
        n_day = WS.per_day(rate, clock)
        assert n_day == int(np.floor(rate * usable + 0.5)) and len(insts) == n_day * days
        for d in range(days):
            ks = sorted(e.start_tick - d * tpd for e in insts if e.start_tick // tpd == d)
            assert len(ks) == n_day
            for j, k in enumerate(ks):                                   # one start per stratum, jittered
                assert j * usable // n_day <= k < (j + 1) * usable // n_day
        fams = [e.latent_type for e in insts]
        for i in range(0, len(fams) - len(ST.FAMILIES) + 1, len(ST.FAMILIES)):
            assert sorted(fams[i:i + 4]) == ST.FAMILIES                  # every block has every family once
        for f in ST.FAMILIES:
            hs = [e.holdout for e in insts if e.latent_type == f]
            assert int(np.floor(frac * len(hs))) <= sum(hs) <= int(np.ceil(frac * len(hs))), (f, hs)
    # jitter differs between days and seeds (starts are not a fixed grid)
    a, _ = gen(pop, world_seed=1)
    b, _ = gen(pop, world_seed=2)
    assert [e.start_tick for e in a] != [e.start_tick for e in b]


def test_balanced_schedule_family_weights_and_prefix(pop):
    insts, _ = gen(pop, simulation_days=6, latent_events={"family_weights": {"E1": 3.0, "E2": 1.0, "E3": 0.0,
                                                                              "E4": 0.0}})
    c = Counter(e.latent_type for e in insts)
    assert set(c) == {"E1", "E2"}
    fams = [e.latent_type for e in insts]
    assert all(fams[i:i + 4].count("E2") == 1 for i in range(0, len(fams) - 3, 4))   # 3:1 in every block
    # a longer run only appends days: its first days are the shorter run's script
    short, _ = gen(pop, simulation_days=2, latent_events={"referents": {"enabled": True}})
    long, _ = gen(pop, simulation_days=4, latent_events={"referents": {"enabled": True}})
    assert records(short) == records(long)[:len(short)]


def test_bernoulli_schedule_keeps_v20_timing(pop):
    """schedule: bernoulli is the v2.0 per-tick draw: counts vary between seeds, starts never cross a day end."""
    counts = []
    for ws in range(8):
        insts, cfg = gen(pop, world_seed=ws, latent_events={"schedule": "bernoulli"})
        tpd = Clock(cfg).ticks_per_day
        assert all(e.start_tick % tpd + ST.SPAN < tpd for e in insts)
        counts.append(len(insts))
    assert len(set(counts)) > 2
    bal = {len(gen(pop, world_seed=ws)[0]) for ws in range(8)}
    assert len(bal) == 1


@pytest.mark.filterwarnings("ignore")
def test_bad_schedule_and_holdout_raise(pop):
    with pytest.raises(ValueError, match="schedule"):
        gen(pop, latent_events={"schedule": "poisson"})
    with pytest.raises(ValueError, match="holdout_frac"):
        gen(pop, latent_events={"holdout_frac": 1.5})


# ------------------------------------------------------------------ assignment
def test_assignment_none(pop):
    insts, _ = gen(pop)
    assert all(e.circle is None and e.cast_from_home is False for e in insts)


def test_assignment_balanced_has_no_family_circle_correlation(pop):
    table = defaultdict(Counter)
    n = 0
    for ws in range(4):
        insts, _ = gen(pop, world_seed=ws, simulation_days=6,
                       latent_events={"event_rate": 0.3, "assignment": {"mode": "balanced"}})
        for e in insts:
            table[e.latent_type][e.circle] += 1
            n += 1
    assert n > 200
    circles = sorted({c for row in table.values() for c in row})
    assert circles == ["c_dorm", "c_lab"]
    for fam, row in table.items():
        share = row["c_lab"] / sum(row.values())
        assert 0.35 < share < 0.65, (fam, dict(row))


def test_assignment_home_maps_families_and_rotates(pop):
    maps = []
    for ws in (0, 1):
        insts, _ = gen(pop, world_seed=ws, simulation_days=5, latent_events={"assignment": {"mode": "home"}})
        m = {}
        for e in insts:
            assert m.setdefault(e.latent_type, e.circle) == e.circle     # each family -> one circle
        assert set(m.values()) == {"c_dorm", "c_lab"}
        maps.append(m)
    assert all(maps[0][f] != maps[1][f] for f in maps[0] if f in maps[1])   # rotated by world_seed % 2


@pytest.mark.parametrize("strength", [0.0, 0.7, 1.0])
def test_cast_from_home_rate_tracks_strength(pop, strength):
    _, circles = pop
    flags = []
    for ws in range(3):
        insts, _ = gen(pop, world_seed=ws, simulation_days=5,
                       latent_events={"assignment": {"mode": "home", "strength": strength}})
        for e in insts:
            flags.append(e.cast_from_home)
            if e.cast_from_home:
                assert e.roles["P"]["agent"] in circles[e.circle]
    rate = sum(flags) / len(flags)
    assert abs(rate - strength) < 0.12, rate


# ------------------------------------------------------------------ structure controls
def test_structure_scrambled_and_none(pop):
    real, _ = gen(pop, simulation_days=4)
    scr, _ = gen(pop, simulation_days=4, latent_events={"structure": "scrambled"})
    non, _ = gen(pop, simulation_days=4, latent_events={"structure": "none"})
    assert all(e.composed_from is None and e.skin for e in real)
    for insts, mode in ((scr, "scrambled"), (non, "none")):
        assert insts
        for e in insts:
            assert e.structure_mode == mode and e.skin is None and e.scenario is None
            assert [c["node"] for c in e.composed_from] == ST.NODES
            src = [c["skin"] for c in e.composed_from]
            assert src[0] == src[1] and len({src[0], src[2], src[3]}) == 3
            role2 = ST.second_role(e.composed_from[2]["family"])       # role follows the n1 source
            assert role2 in e.roles
            assert all(c["family"] == c["skin"].split("_")[0].upper() for c in e.composed_from)
    assert {e.latent_type for e in scr} <= set(ST.FAMILIES) and len({e.latent_type for e in scr}) > 1
    assert {e.latent_type for e in non} == {"E0"}
    assert all(e.schema == ST.SCHEMAS["E0"]["id"] for e in non)
    # the label carries no structure: it is the same family draw as in the real condition
    by_t = {e.start_tick: e.latent_type for e in real}
    same = [by_t[e.start_tick] == e.latent_type for e in scr if e.start_tick in by_t]
    assert len(same) > 10 and all(same)


def test_composed_nodes_respect_holdout(pop):
    insts, _ = gen(pop, simulation_days=4, latent_events={"structure": "scrambled", "holdout_frac": 0.5})
    for e in insts:
        doms = {c["skin"].split("_", 1)[1] for c in e.composed_from}
        assert all((d in HOLDOUT_DOMAINS) == e.holdout for d in doms)


# ------------------------------------------------------------------ referents
def _repeat_rate(insts, circles):
    member = {m: c for c, ms in circles.items() for m in ms}
    met, hits, n = defaultdict(set), 0, 0
    for e in insts:
        key = member.get(e.roles["P"]["agent"], "*")
        r = e.referents[0]
        n += 1
        hits += r["id"] in met[(key, r["domain"])]
        met[(key, r["domain"])].add(r["id"])
        met[("*", r["domain"])].add(r["id"])
    return hits / n


def test_referent_records(pop):
    insts, _ = gen(pop)
    for e in insts:
        r0 = e.referents[0]
        assert r0["domain"] == e.skin.split("_", 1)[1]
        for r in e.referents:
            dom, idx = r["id"].split(":")
            assert dom == r["domain"] and CATEGORIES[dom][int(idx)] == r["name"] and r["reused"] is False
        assert r0["name"] in e.beats[0]["facts"][0]["text"]
        assert r0["name"] in e.beats[2]["facts"][0]["text"]


def test_referent_reuse_rate_higher_when_enabled(pop):
    _, circles = pop
    off, on = [], []
    for ws in range(4):
        kw = dict(world_seed=ws, simulation_days=3)
        a, _ = gen(pop, **kw, latent_events={"holdout_frac": 0.0})
        b, _ = gen(pop, **kw, latent_events={"holdout_frac": 0.0, "referents": {"enabled": True, "reuse": 0.6}})
        off.append(_repeat_rate(a, circles))
        on.append(_repeat_rate(b, circles))
        assert any(r["reused"] for e in b for r in e.referents)
    assert np.mean(on) > np.mean(off) + 0.1, (on, off)


@pytest.mark.parametrize("structure", ["real", "scrambled"])
def test_referent_reuse_is_domain_first_and_respects_holdout(pop, structure):
    """With reuse decided before the skin, a circle meets the same referent again at a realistic rate even
    at design scale (2 days, event_rate 0.12); the reused referent's domain picks the skin, and holdout
    instances only reuse held-out-domain referents."""
    _, circles = pop
    member = {m: c for c, ms in circles.items() for m in ms}
    shares = []
    for ws in range(1, 7):
        insts, _ = gen(pop, skins=None, world_seed=ws, simulation_days=2,
                       latent_events={"event_rate": 0.12, "structure": structure, "assignment": {"mode": "balanced"},
                                      "referents": {"enabled": True, "reuse": 0.6}})
        met = defaultdict(set)
        for e in insts:
            key = member.get(e.roles["P"]["agent"], "*")
            r0 = e.referents[0]
            n0_skin = e.skin if structure == "real" else e.composed_from[0]["skin"]
            assert n0_skin.split("_", 1)[1] == r0["domain"]
            assert (r0["domain"] in HOLDOUT_DOMAINS) == e.holdout
            if r0["reused"]:
                assert r0["id"] in met[key]                         # met before by P's circle (or population)
            for r in e.referents:
                met[key].add(r["id"])
                met["*"].add(r["id"])
        shares.append(np.mean([e.referents[0]["reused"] for e in insts]))
    assert np.mean(shares) > 0.3, shares


# ------------------------------------------------------------------ holdout / link visibility
def test_holdout_frac(pop):
    for frac in (0.0, 1.0):
        insts, _ = gen(pop, latent_events={"holdout_frac": frac})
        assert all(e.holdout == bool(frac) for e in insts)
    hold = []
    for ws in range(3):
        insts, _ = gen(pop, world_seed=ws, simulation_days=5, latent_events={"holdout_frac": 0.25})
        for e in insts:
            hold.append(e.holdout)
            assert (e.skin.split("_", 1)[1] in HOLDOUT_DOMAINS) == e.holdout
    assert abs(np.mean(hold) - 0.25) < 0.08
    assert any(e.start_tick < Clock(make_cfg()).ticks_per_day and e.holdout
               for e in gen(pop, latent_events={"holdout_frac": 0.5})[0])          # interleaved from day 1


def test_link_visibility(pop):
    closed, _ = gen(pop)
    opened, _ = gen(pop, latent_events={"link_visibility": 1.0})
    half, _ = gen(pop, simulation_days=5, latent_events={"link_visibility": 0.5})
    assert all(e.beats[0]["facts"][1]["visibility"] == [e.roles["P"]["agent"]] for e in closed)
    assert all(e.beats[0]["facts"][1]["visibility"] == "all" for e in opened)
    vis = [e.beats[0]["facts"][1]["visibility"] == "all" for e in half]
    assert 0.3 < np.mean(vis) < 0.7
    assert [e.narrative for e in closed] == [e.narrative for e in opened]


# ------------------------------------------------------------------ record / save / load
def test_ground_truth_fields_and_roundtrip(pop, tmp_path):
    insts, _ = gen(pop, latent_events={"structure": "scrambled", "assignment": {"mode": "balanced"},
                                       "referents": {"enabled": True}})
    path = tmp_path / "world_script.jsonl"
    sha = WS.save(insts, path)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    lines = [json.loads(line) for line in open(path)]
    for rec in lines:
        assert set(LE.RECORD_KEYS) <= set(rec)
        assert rec["generator"] == "script_v2"
        for b in rec["beats"]:
            assert set(b) == {"idx", "tick", "location", "arena", "facts", "movers"}
    back = WS.load(path)
    assert all(isinstance(e, LE.EventInstance) for e in back)
    assert records(back) == records(insts)
    assert WS.save(back, tmp_path / "again.jsonl") == sha
    # script_from loads verbatim
    cfg = make_cfg(latent_events={"script_from": str(path)})
    profiles, circles = pop
    assert records(WS.generate(cfg, profiles, circles, Clock(cfg))) == records(insts)
    tf = WS.trace_fields(insts[0])
    assert {"latent_type", "circle", "cast_from_home", "referents", "structure_mode"} <= set(tf)


def _saved(pop, tmp_path, name="ws.jsonl", meta=True, **over):
    insts, cfg = gen(pop, **over)
    profiles, circles = pop
    path = tmp_path / name
    WS.save(insts, path, meta=WS.script_meta(cfg, profiles, circles, Clock(cfg)) if meta else None)
    return insts, path


def _reuse(pop, path, n_agents=8, **over):
    profiles, _ = load_population(POP, n_agents)
    cfg = make_cfg(**deep_merge({"latent_events": {"script_from": str(path)}}, over))
    return WS.generate(cfg, profiles, load_circles(POP, list(profiles)), Clock(cfg))


def test_script_from_checks_the_run(pop, tmp_path):
    insts, path = _saved(pop, tmp_path, world_seed=4)
    assert WS.meta_path(path).exists()
    assert records(_reuse(pop, path, world_seed=4, seed=99)) == records(insts)       # any seed, same world
    for over, n_agents, msg in (({"world_seed": 4, "simulation_days": 2}, 8, "days"),
                                ({"world_seed": 4, "day_end": "20:30"}, 8, "ticks_per_day"),
                                ({"world_seed": 5}, 8, "world_seed"),
                                ({"world_seed": 4}, 6, "not in this run's population"),
                                ({"world_seed": 4, "routine": {"jitter_minutes": 5}}, 8, "routine")):
        with pytest.raises(ValueError, match=msg):
            _reuse(pop, path, n_agents, **over)


def test_script_from_content_checks_without_meta(pop, tmp_path):
    """A bare script (no sidecar, no run manifest) is still checked against the run's clock, agents, circles."""
    insts, path = _saved(pop, tmp_path, meta=False, world_seed=4)
    assert records(_reuse(pop, path, world_seed=4)) == records(insts)
    with pytest.raises(ValueError, match="outside this run"):
        _reuse(pop, path, world_seed=4, simulation_days=2)
    with pytest.raises(ValueError, match="not in this run's population"):
        _reuse(pop, path, 6, world_seed=4)
    circ, cpath = _saved(pop, tmp_path, name="c.jsonl", meta=False, latent_events={"assignment": {"mode": "balanced"}})
    profiles, circles = pop
    cfg = make_cfg(latent_events={"script_from": str(cpath)})
    with pytest.raises(ValueError, match="circle"):
        WS.generate(cfg, profiles, {"c_lab": circles["c_lab"]}, Clock(cfg))


def test_script_from_uses_the_source_run_manifest(pop, tmp_path):
    insts, cfg = gen(pop, world_seed=4)
    profiles, circles = pop
    run = tmp_path / "run1"
    run.mkdir()
    WS.save(insts, run / "world_script.jsonl")
    clock = Clock(cfg)
    man_cfg = {**cfg, "day_end": 22 * 60 + 30}                # YAML 1.1 reads an unquoted 22:30 as minutes
    (run / "manifest.json").write_text(json.dumps({"config": man_cfg, "ticks_per_day": clock.ticks_per_day,
                                                   "agents": {a: {} for a in profiles}, "circles": circles}))
    assert WS.saved_meta(run / "world_script.jsonl")[1].endswith("manifest.json")
    assert records(_reuse(pop, run / "world_script.jsonl", world_seed=4)) == records(insts)
    with pytest.raises(ValueError, match="world_seed"):
        _reuse(pop, run / "world_script.jsonl", world_seed=3)


@pytest.mark.filterwarnings("ignore")        # the config loader also warns about the removed key
def test_llm_generator_is_rejected(pop):
    with pytest.raises(ValueError, match="llm"):
        gen(pop, latent_events={"generator": "llm"})
    with pytest.raises(ValueError, match="structure"):
        gen(pop, latent_events={"structure": "shuffled"})
    with pytest.raises(ValueError, match="mode"):
        gen(pop, latent_events={"assignment": {"mode": "clustered"}})


def test_missing_skins_raise_clearly(pop):
    with pytest.raises(ValueError, match="E2"):
        gen(pop, skins=[s for s in FIXTURE_SKINS if s["family"] != "E2"])
    only_train = [s for s in FIXTURE_SKINS if not s["holdout"]]
    with pytest.raises(ValueError, match="holdout"):
        gen(pop, skins=only_train)
    assert gen(pop, skins=only_train, latent_events={"holdout_frac": 0.0})[0]


# ------------------------------------------------------------------ latent_events compatibility
def test_latent_events_v1_compat_and_v2_record():
    assert set(LE.LATENT_TYPES) == {"E0", "E1", "E2", "E3", "E4"}
    for name in ("SCENARIOS", "scenarios_for", "instantiate", "assign_home_groups", "llm_scenario", "_fill"):
        assert hasattr(LE, name)
    e = LE.EventInstance(id="ev001", latent_type="E1", scenario="room_mixup", holdout=False, start_tick=3,
                         roles={"P": {"agent": "maya", "name": "Maya"}}, beats=[])
    gt = e.ground_truth()
    assert set(LE.RECORD_KEYS) <= set(gt) and gt["narrative"] == "" and gt["scenario"] == "room_mixup"
    assert LE.EventInstance.from_dict(gt).ground_truth() == gt


def test_v1_instantiate_still_builds_instances(pop):
    profiles, _ = pop
    agents = {aid: SimpleNamespace(profile=p) for aid, p in profiles.items()}
    rng = np.random.default_rng(0)
    for scn in LE.SCENARIOS:
        inst = LE.instantiate(scn, "ev999", 2, rng, agents, set(), 60)
        assert inst is not None and inst.ground_truth()["scenario"] == scn["key"]


# ------------------------------------------------------------------ circles
def test_load_circles(tmp_path):
    profiles, _ = load_population(POP, 8)
    assert load_circles(POP, list(profiles)) == {"c_dorm": ["ethan", "leo", "jordan"], "c_lab": ["maya", "dev", "hana"]}
    # population_size truncation dissolves circles left with < 3 members
    assert load_circles(POP, ["maya", "priya", "ethan", "leo", "jordan"]) == {"c_dorm": ["ethan", "leo", "jordan"]}
    ids = list(profiles)
    for bad, msg in (({"a": ["maya", "dev"]}, ">= 3"), ({"a": ["maya", "dev", "hana"], "b": ["hana", "leo", "ethan"]},
                     "disjoint"), ({"a": ["maya", "dev", "nobody"]}, "unknown")):
        with pytest.raises(ValueError, match=msg):
            validate_circles(bad, ids)
    raw = json.loads(json.dumps({"agents": [{"id": i} for i in ids], "circles": {"x": ["maya", "maya", "dev"]}}))
    p = tmp_path / "pop.yaml"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_circles(p, ids)


# ------------------------------------------------------------------ skin validation
@pytest.fixture(scope="module")
def lexicon():
    profiles, _ = load_population(POP, 8)
    return population_lexicon(profiles)


def test_fixture_skins_are_valid(lexicon):
    assert ST.validate_skins(FIXTURE_SKINS, lexicon) == []


def _mutated(key, **facts):
    skins = copy.deepcopy(FIXTURE_SKINS)
    s = next(s for s in skins if s["key"] == key)
    s["facts"].update(facts)
    return skins


@pytest.mark.parametrize("key,facts,expect", [
    ("e1_dining", {"n1": "{S} was upset because the e1dininge tray fell over later that afternoon."}, "because"),
    ("e2_gym", {"n2": "{P} laughed about the whole e2gymh thing near {R} again before the evening."}, "nickname"),
    ("e4_cafe", {"n2": "{P} called it \"the e4cafeg\" near {R} again before the evening."}, "quoted"),
    ("e1_lab", {"n1": "{Q} also e1labe the e1labf later on that afternoon."}, "placeholders"),
    ("e3_dorm", {"n1": "{S} also e3dorme the e3dormf later on that afternoon."}, "placeholders"),
    ("e2_cafe", {"n0": "{P} e2cafea at {R} and at {R} with {X}."}, "exactly once"),
    ("e4_gym", {"n0_private": "{P} was e4gymc."}, "words after filling"),
    ("e1_quad", {"n0": "{R} e1quada e1quadb {P} in the morning with {X}."}, "person first"),
    ("e2_tech", {"n0": "{P} e2techa e2techb at {R}. Then {P} left with {X}."}, "one sentence"),
    ("e3_clubs", {"n0": "{P} saw {Q} e3clubsa e3clubsb at {R} in the morning with {X}."}, "n0 involves P only"),
    ("e1_transit", {"n0_private": "{P} liked {R} a lot that e1transitc day."}, "n0_private"),
    # causal glue across facts (ontology v2 §1.2 bans "so")
    ("e1_cafe", {"n2": "{P} e1cafeg the e1cafeh so {P} missed {R} again before the evening."}, "'so'"),
    ("e2_lab", {"n2": "{P} ended up with the e2labh near {R} again before the evening."}, "ended up"),
    ("e3_gym", {"n2": "{P} e3gymg the e3gymh near {R}, which is why the evening went badly."}, "which is why"),
    ("e4_dorm", {"n1": "{S} also e4dorme the e4dormf later, and that's why it rained."}, "that's why"),
    ("e1_lab", {"n1": "{S} also e1labe the e1labf later on due to the rain that afternoon."}, "due to"),
    ("e2_dorm", {"n2": "{P} e2dormg the e2dormh near {R}, and that meant a quiet evening."}, "that meant"),
    # a fact names only the people it involves
    ("e4_gym", {"n1": "{S} also e4gyme the e4gymf with {P} later on that afternoon."}, "n1: must not mention {P}"),
    ("e2_cafe", {"n2": "{P} e2cafeg the e2cafeh with {S} near {R} again before the evening."},
     "n2: must not mention {S}"),
    ("e3_quad", {"n2": "{P} e3quadg the e3quadh near {R} while {Q} watched that evening."},
     "n2: must not mention {Q}"),
])
def test_validate_skins_catches_defects(lexicon, key, facts, expect):
    probs = ST.validate_skins(_mutated(key, **facts), lexicon)
    assert any(expect in p for p in probs), probs


def test_validate_skins_shape_rules(lexicon):
    skins = [s for s in FIXTURE_SKINS if s["key"] != "e1_library"]
    assert any("E1" in p and "library" in p for p in ST.validate_skins(skins, lexicon))
    skins = copy.deepcopy(FIXTURE_SKINS)
    skins[0]["holdout"] = True
    skins[1]["slots"] = {"X": ["only one"]}
    skins[2]["locations"] = {"n1": "Moon"}
    probs = ST.validate_skins(skins, lexicon)
    assert any("holdout must be" in p for p in probs)
    assert any("2-4" in p for p in probs)
    assert any("unknown location" in p for p in probs)


def test_validate_skins_pins_are_per_domain(lexicon):
    skins = copy.deepcopy(FIXTURE_SKINS)
    for s in skins:
        if s["domain"] == "gym":
            s["locations"] = {"n0": "Gym"}
    assert ST.validate_skins(skins, lexicon) == []                 # same pin for every family: fine
    next(s for s in skins if s["key"] == "e4_gym")["locations"] = {}
    probs = ST.validate_skins(skins, lexicon)
    assert any("domain 'gym': location pins differ across families" in p for p in probs), probs


def test_validate_skins_trigram_overlap(lexicon):
    """Content phrases with a stopword in the middle ('bowl of cornflakes') escape the bigram check."""
    skins = _mutated("e1_dining", n2="{P} dropped a bowl of cornflakes near {R} again before the evening.")
    s = next(s for s in skins if s["key"] == "e3_gym")
    s["facts"]["n2"] = "{P} found a bowl of cornflakes near {R} again before the evening."
    probs = ST.validate_skins(skins, lexicon)
    assert any("content trigram 'bowl of cornflakes' shared across families" in p for p in probs), probs
    skins = _mutated("e2_dining", n2="{P} found a bowl of cornflakes near {R} again before the evening.")
    s = next(s for s in skins if s["key"] == "e2_quad")
    s["facts"]["n2"] = "{P} lost a bowl of cornflakes near {R} again before the evening."
    probs = ST.validate_skins(skins, lexicon)
    assert any("trigram 'bowl of cornflakes' shared by E2 holdout" in p for p in probs), probs
    assert ST.content_trigrams("{P} took a bowl of {X} to the lab") == {"took a bowl"}   # {X} breaks "bowl of"


def test_validate_skins_bigram_overlap(lexicon):
    # the same content bigram in two families
    skins = _mutated("e1_dining", n2="{P} dropped the sticky clipboard near {R} again before the evening.")
    s = next(s for s in skins if s["key"] == "e3_gym")
    s["facts"]["n2"] = "{P} found the sticky clipboard near {R} again before the evening."
    probs = ST.validate_skins(skins, lexicon)
    assert any("'sticky clipboard' shared across families" in p for p in probs), probs
    # holdout vs train of the same family
    skins = _mutated("e2_dining", n2="{P} found the purple umbrella near {R} again before the evening.")
    s = next(s for s in skins if s["key"] == "e2_quad")
    s["facts"]["n2"] = "{P} lost the purple umbrella near {R} again before the evening."
    probs = ST.validate_skins(skins, lexicon)
    assert any("'purple umbrella' shared by E2 holdout e2_quad and train e2_dining" in p for p in probs), probs
    # the same bigram within one family's train skins is fine; population-lexicon bigrams are exempt
    skins = _mutated("e4_dining", n2="{P} found the purple umbrella near {R} again before the evening.")
    s = next(s for s in skins if s["key"] == "e4_gym")
    s["facts"]["n2"] = "{P} lost the purple umbrella near {R} again before the evening."
    assert ST.validate_skins(skins, lexicon) == []
    lex_bg = next(b for b in lexicon["bigrams"] if ST.content_bigrams(b))
    skins = _mutated("e1_cafe", n2=f"{{P}} mentioned {lex_bg} near {{R}} again before the evening.")
    s = next(s for s in skins if s["key"] == "e2_lab")
    s["facts"]["n2"] = f"{{P}} forgot {lex_bg} near {{R}} again before the evening."
    assert not any(lex_bg in p for p in ST.validate_skins(skins, lexicon))
    # referent-name words and placeholders are not content words
    assert ST.content_bigrams("{P} fixed the salad bar with {X} tape") == set()


# ------------------------------------------------------------------ the real skins
def test_real_skins_are_valid_and_generate(pop, lexicon):
    from pathlib import Path

    from backend.simulation import skins as SK
    missing = [m for m in SK.FAMILY_MODULES if not (Path(SK.__file__).parent / f"{m}.py").exists()]
    if missing:
        pytest.skip(f"skin modules not written yet: {missing}")
    real = SK.load_skins(strict=True)
    assert Counter(s["family"] for s in real) == {f: 11 for f in ST.FAMILIES}
    probs = ST.validate_skins(real, lexicon)
    assert probs == [], "\n".join(probs)
    for structure in ("real", "scrambled"):
        insts, _ = gen(pop, skins=None, simulation_days=2, latent_events={"structure": structure})
        assert insts
        for e in insts:
            for b in e.beats:
                for f in b["facts"]:
                    assert not re.search(r"[{}]", f["text"])
