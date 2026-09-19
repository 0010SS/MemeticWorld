"""Roster and turnover (ontology v3 §3): Latin square, departure, arrival ties, seed memories, shift hook."""
from types import SimpleNamespace

import pytest

from backend.agents.agent import Agent
from backend.agents.profile import load_population
from backend.config import deep_merge, load_config
from backend.llm.client import current_scope
from backend.memory.store import SimMemoryMeta
from backend.simulation import roster as R
from backend.simulation.rngs import seed_rng
from backend.simulation.scheduler import plan_day
from backend.simulation.world import Clock

POP = "configs/population/coop13.yaml"
WAVES = [{"day": 4, "depart": {"am_crew": 1, "pm_crew": 1, "stores": 1}},
         {"day": 6, "depart": {"am_crew": 1, "pm_crew": 1}}]
HIDDEN = ["K0", "K1", "K2", "K3", "LENS", "DAMP", "BELT", "AIR", "WARP", "M1", "M2", "regime", "mapping"]
BANNED = ["invent", "coin", "label", "term", "slang", "meme", "nickname", "shorthand", "rule", "tip ", "procedure",
          "jot"]


def make_cfg(seed=13, enabled=True, turnover=True, **over):
    base = {"population": POP, "seed": 1, "world_seed": seed, "simulation_days": 7,
            "llm": {"backend": "mock"},
            "roster": {"enabled": enabled},
            "turnover": {"enabled": turnover, "waves": WAVES}}
    return load_config(None, deep_merge(base, over))


class Tracer:
    def __init__(self):
        self.recs = []

    def log(self, type_, **kw):
        self.recs.append({"type": type_, **kw})


def make_sim(tmp_path, seed=13, **over):
    cfg = make_cfg(seed, **over)
    profiles, _ = load_population(POP, cfg.get("population_size"), include_reserves=True)
    ctx = SimpleNamespace(agents={}, mods=None, active=None)
    for pid, prof in profiles.items():
        a = Agent(prof, cfg, None, cfg["seed"])
        a.ctx = ctx
        ctx.agents[pid] = a
    scopes = []

    def embed(text):
        scopes.append(current_scope())
        return [0.0] * 4

    sim = SimpleNamespace(cfg=cfg, ctx=ctx, agents=ctx.agents, tracer=Tracer(), embed=embed, meta=SimMemoryMeta(),
                          clock=Clock(cfg), run_dir=tmp_path, pending_obs={}, overrides={}, follow={})
    sim.scopes = scopes
    return sim


# ------------------------------------------------------------------------------------------ population
def test_coop13_population():
    p8, _ = load_population(POP, 8)
    assert list(p8) == ["maya", "priya", "ethan", "leo", "jordan", "dev", "sofia", "hana"]
    assert {a: p.role for a, p in p8.items()} == {"maya": "am_crew", "dev": "am_crew", "hana": "am_crew",
                                                   "ethan": "pm_crew", "leo": "pm_crew", "jordan": "pm_crew",
                                                   "priya": "stores", "sofia": "stores"}
    p13, _ = load_population(POP, 8, include_reserves=True)
    res = [a for a, p in p13.items() if p.reserve]
    assert res == ["noor", "tomas", "wei", "amara", "felix"]
    for r in res:
        assert p13[r].role is None
        assert all(rel.relation_type == "stranger" for rel in p13[r].relationships.values())
    # without reserves, the homewood8 structure is unchanged (same ids, same ties)
    h8, _ = load_population("configs/population/homewood8.yaml", 8)
    for a in h8:
        assert {k: vars(v) for k, v in h8[a].relationships.items()} == \
               {k: vars(v) for k, v in p8[a].relationships.items()}
    assert R.persona_universe(POP) == {a: p.name for a, p in p13.items()}


# ------------------------------------------------------------------------------------------ Latin square
TABLE = {11: ({"maya": "noor", "leo": "tomas", "priya": "wei"}, {"dev": "amara", "jordan": "felix"}),
         12: ({"dev": "tomas", "jordan": "wei", "sofia": "amara"}, {"hana": "felix", "ethan": "noor"}),
         13: ({"hana": "wei", "ethan": "amara", "priya": "felix"}, {"maya": "noor", "leo": "tomas"})}


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_latin_square(seed):
    cfg = make_cfg(seed)
    profiles, _ = load_population(POP, 8, include_reserves=True)
    ro = R.Roster(cfg, profiles, seed, Clock(cfg))
    for day, want in zip((4, 6), TABLE[seed]):
        got = {c["replaces"]: c["agent"] for c in ro.changes_on(day) if c["kind"] == "arrive"}
        assert got == want
    for d in range(1, 8):
        assert len(ro.active_on(d)) == 8
    # D6: each crew has 1 founder, 1 W1, 1 W2; stores 1 founder + 1 W1
    for role in ("am_crew", "pm_crew"):
        assert sorted(ro.cohort[a] for a in ro.members(role, 6)) == ["W1", "W2", "founder"]
    assert sorted(ro.cohort[a] for a in ro.members("stores", 6)) == ["W1", "founder"]
    # identical in every arm: nothing but world_seed and waves matters
    ro2 = R.Roster(deep_merge(cfg, {"seed": 99, "records": {"enabled": True}}), profiles, seed, Clock(cfg))
    assert ro2.schedule == ro.schedule


def test_rota_newcomers_first():
    cfg = make_cfg(13)
    profiles, _ = load_population(POP, 8, include_reserves=True)
    ro = R.Roster(cfg, profiles, 13, Clock(cfg))
    assert ro.rota_order("am_crew", 4)[0] == "wei" and ro.rota_order("am_crew", 5)[0] == "wei"
    assert ro.rota_order("am_crew", 6)[0] == "noor" and ro.rota_order("am_crew", 7)[0] == "noor"
    assert ro.rota_order("am_crew", 3) == ["maya", "dev", "hana"]


def test_disabled_roster_is_a_noop(tmp_path):
    sim = make_sim(tmp_path, enabled=False)
    assert R.apply_day_start(sim, 4) == []
    assert len(R.active_agents(sim)) == 13            # v2 behaviour: every built agent acts
    assert sim.tracer.recs == []


# ------------------------------------------------------------------------------------------ day start
def test_departure_and_arrival(tmp_path):
    sim = make_sim(tmp_path, 13)
    for d in (1, 2, 3):
        R.apply_day_start(sim, d)
    assert sorted(R.active_agents(sim)) == sorted(["maya", "dev", "hana", "ethan", "leo", "jordan", "priya", "sofia"])
    old_hana = sim.ctx.agents["hana"]
    n_hana = len(old_hana.a_mem.all_nodes())
    ch = R.apply_day_start(sim, 4)
    assert R.apply_day_start(sim, 4) == []           # idempotent
    assert {c["agent"] for c in ch if c["kind"] == "depart"} == {"hana", "ethan", "priya"}
    act = R.active_agents(sim)
    assert len(act) == 8 and not {"hana", "ethan", "priya"} & set(act)
    assert {"wei", "amara", "felix"} <= set(act)
    assert sim.ctx.agents["hana"].state.active is False
    assert len(sim.ctx.agents["hana"].a_mem.all_nodes()) == n_hana     # departed memory untouched
    assert (tmp_path / "departed" / "hana.json").exists()
    assert "hana" in sim.ctx.agents                  # kept for name resolution
    # newcomer: fresh agent, profile role set, only its onboarding memories
    wei = sim.ctx.agents["wei"]
    assert wei.profile.role == "am_crew" and wei.state.active
    crew = ["maya", "dev"]
    assert len(wei.a_mem.all_nodes()) == len(crew)
    assert all(s == "t0180:00roster" for s in sim.scopes)
    assert all(m.source_type == "seed" for m in sim.meta.meta.values())
    # symmetric ties
    for m in crew:
        for x, y in (("wei", m), (m, "wei")):
            rel = sim.ctx.agents[x].profile.relationships[y]
            assert (rel.relation_type, rel.familiarity, rel.affinity) == ("co-op crewmate", 0.35, 0.5)
    rel = sim.ctx.agents["wei"].profile.relationships["leo"]
    assert (rel.relation_type, rel.familiarity) == ("co-op member", 0.2)
    assert sim.ctx.agents["leo"].profile.relationships["wei"].familiarity == 0.2
    # existing stronger ties are never downgraded
    assert sim.ctx.agents["maya"].profile.relationships["dev"].relation_type == "labmate"
    # trace
    kinds = [r["type"] for r in sim.tracer.recs]
    assert kinds.count("roster_change") == 6 and kinds.count("onboarding") == 3
    arr = [r for r in sim.tracer.recs if r["type"] == "roster_change" and r["agent"] == "wei"][0]
    assert (arr["day"], arr["kind"], arr["role"], arr["replaces"]) == (4, "arrive", "am_crew", "hana")
    # wave 2: W1 newcomer and W2 newcomer become crewmates
    R.apply_day_start(sim, 5)
    R.apply_day_start(sim, 6)
    assert sim.ctx.agents["noor"].profile.relationships["wei"].relation_type == "co-op crewmate"
    assert sim.ctx.agents["wei"].profile.relationships["noor"].familiarity == 0.35
    assert sorted(R.active_agents(sim)) == sorted(["dev", "wei", "noor", "jordan", "amara", "tomas", "sofia", "felix"])
    # no departed agent is ever handed out to act or perceive
    assert not {"hana", "ethan", "priya", "maya", "leo"} & set(R.active_agents(sim))


def test_departed_agent_is_dropped_from_pending_state(tmp_path):
    sim = make_sim(tmp_path, 13)
    R.apply_day_start(sim, 1)
    sim.pending_obs["hana"] = ["x"]
    sim.overrides["hana"] = {"until": 999}
    R.apply_day_start(sim, 4)
    assert "hana" not in sim.pending_obs and "hana" not in sim.overrides


# ------------------------------------------------------------------------------------------ script records
def test_farewell_and_orientation_records():
    cfg = make_cfg(13)
    profiles, _ = load_population(POP, 8, include_reserves=True)
    clock = Clock(cfg)
    ro = R.Roster(cfg, profiles, 13, clock)
    fw = {f["agent"]: f for f in ro.farewells()}
    assert set(fw) == {"hana", "ethan", "priya", "maya", "leo"}
    h = fw["hana"]
    assert h["day"] == 3 and h["arena"] == "Makerspace"
    # during the last hour of the last shift (AM ends 13:15 -> k 23)
    assert clock.time_of(h["end_tick"]).strftime("%H:%M") == "13:15"
    assert 0 < h["end_tick"] - h["tick"] <= 60 // clock.tick_minutes
    assert h["facts"][0]["text"] == ("Hana mentioned that today was her last shift at the co-op; she is moving to "
                                     "another lab next week.")
    assert fw["priya"]["arena"] == "Stockroom"
    ors = {o["agent"]: o for o in ro.orientations()}
    assert ors["wei"]["guide"] == "maya" and clock.time_of(ors["wei"]["tick"]).strftime("%H:%M") == "08:45"
    assert ors["wei"]["facts"][1]["text"] == "Wei joined the co-op's morning crew today."
    assert ors["noor"]["guide"] == "dev"             # maya departs the same morning
    recs = ro.script_records()
    assert [r["tick"] for r in recs] == sorted(r["tick"] for r in recs)
    for r in recs:
        for f in r.get("facts", []):
            for w in HIDDEN:
                assert w not in f["text"]
            for w in BANNED:
                assert w not in f["text"].lower()


# ------------------------------------------------------------------------------------------ shift hook
def _plan(prof, cfg, day=2, ws=13):
    clock = Clock(cfg)
    return plan_day(SimpleNamespace(cfg=cfg, profile=prof), clock, seed_rng(ws, "plan", prof.id, day))


def test_shift_hook():
    on, off = make_cfg(13), make_cfg(13, enabled=False)
    profiles, _ = load_population(POP, 8, include_reserves=True)
    clock = Clock(on)
    for aid, prof in profiles.items():
        base = _plan(prof, off)
        if prof.role is None:
            assert _plan(prof, on) == base
            continue
        plan = _plan(prof, on)
        wins = R.shift_ticks(on, prof.role, clock)
        loc, arena = R.ROLE_PLACE[prof.role]
        for ks, ke in wins:
            assert any(e["k"] == ks and e["arena"] == arena for e in plan)
            inside = [e for e in plan if ks <= e["k"] < ke]
            assert [e["arena"] for e in inside] == [arena]
            assert any(e["k"] == ke for e in plan)
        # routine entries outside the windows are untouched (same rng draws)
        outside = [e for e in base if not any(ks <= e["k"] <= ke for ks, ke in wins)]
        assert all(e in plan for e in outside)
    # the AM crew gets lunch after the shift; stores between the two windows
    maya = _plan(profiles["maya"], on)
    assert any(e["k"] >= 23 and "lunch" in e["activity"] for e in maya)
    assert [w for w in R.shift_ticks(on, "stores", clock)] == [(5, 18), (22, 41)]
    # a reserve given a role on arrival gets that role's shifts
    profiles["wei"].role = "pm_crew"
    wei = _plan(profiles["wei"], on)
    assert any(e["k"] == 22 and e["arena"] == "Makerspace" for e in wei)


# ------------------------------------------------------------------------------------------ integration shapes
def test_integration_shapes(tmp_path):
    """The call shapes CoopWorld / workshop.generate use: Roster(cfg=, sim=, agents=, world_seed=, clock=),
    roster(day) -> crews, role_of(aid), apply_day_start(sim=, day=)."""
    sim = make_sim(tmp_path, 13)
    founders_only = {a: x for a, x in sim.ctx.agents.items() if not x.profile.reserve}
    ro = R.Roster(cfg=sim.cfg, agents=founders_only, world_seed=13, clock=sim.clock)
    assert ro.reserves == ["noor", "tomas", "wei", "amara", "felix"]      # loaded from the population file
    ro = R.Roster(cfg=sim.cfg, sim=sim)
    assert ro(4) == {"am": ["maya", "dev", "wei"], "pm": ["leo", "jordan", "amara"], "stores": ["sofia", "felix"],
                     "newcomers": ["amara", "felix", "wei"]}
    assert ro(6)["newcomers"] == ["noor", "tomas"]
    assert ro.role_of("maya") == "am_crew"
    ro.apply_day_start(sim=sim, day=4)
    assert ro.role_of("hana") is None and ro.role_of("wei") == "am_crew"
    assert sim.ctx.agents["wei"].profile.coop_role == "am_crew"
    ids = [r["id"] for r in ro.script_records()]
    assert len(ids) == len(set(ids))
    st = ro.state_dict(4)
    assert st["agents"]["wei"] == {"role": "am_crew", "cohort": "W1", "arrival_day": 4, "departure_day": None,
                                   "active": True}
    assert st["agents"]["hana"]["active"] is False and ro.checkpoint_state() == st
    assert ro.active_agents() == sorted(R.active_agents(sim))
