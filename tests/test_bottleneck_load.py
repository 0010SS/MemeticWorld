"""The transmission bottleneck and the perceptual-load model (D74).

Two mechanisms docs/COGNITIVE_GROUNDING.md found we claim but do not run:

* **Forgetting never binds.** 25 forgets in 67,224 archived encodings, because `memory.memory_capacity`
  was 300 and an agent's store peaks around 100-150. Iterated learning attributes compressed, schematic
  forms to a transmission bottleneck; with nothing lost after encoding we had no bottleneck at all.
* **Load is modelled backwards.** Crowding was counted per building rather than per room, and the
  familiarity bonus was added after every multiplier, so the social term escaped load entirely and
  crowding *raised* the share of attention going to familiar faces.

These tests pin the fixed behaviour so neither can quietly stop operating again.
"""
import json

import pytest

from backend.agents.perception import attention_prob
from backend.memory.encoder import forget
from tests.conftest import run_sim

BEAT = {"tick": 5, "location": "Library", "arena": "Study Tables"}
ANON = {"id": "x.f0", "text": "Someone dropped a stack of books.", "salience": 0.5,
        "visibility": "all", "kind": "action", "involves": []}
PRIVATE = {"id": "x.f1", "text": "Maya forwarded the wrong room number.", "salience": 0.3,
           "visibility": ["maya"], "kind": "cause", "involves": []}
ARENA_ONLY = dict(ANON, id="x.f3", visibility="arena")


@pytest.fixture
def places(mock_run):
    """Park every agent out of the way; the test then places the ones it cares about. Restores the
    session fixture's positions so other modules see the run as it ended."""
    before = {a.id: (a.state.location, a.state.arena, a.state.in_conversation) for a in mock_run.agents.values()}
    for a in mock_run.agents.values():
        a.state.location, a.state.arena, a.state.in_conversation = "Gym", "Main Floor", None

    def put(agent_id, arena="Study Tables", location="Library"):
        a = mock_run.agents[agent_id]
        a.state.location, a.state.arena = location, arena
        return a
    yield put
    for a in mock_run.agents.values():
        a.state.location, a.state.arena, a.state.in_conversation = before[a.id]


def _fact_about(agent, other_id, salience=0.5):
    """A fact involving someone the observer has a relationship with (so familiarity > 0)."""
    return dict(ANON, id="x.fam", involves=[other_id], salience=salience)


def _familiar_other(agent, ctx):
    """The person this observer knows best, and how well."""
    best = max((o for o in ctx.agents if o != agent.id), key=lambda o: agent.profile.rel(o).familiarity)
    return best, agent.profile.rel(best).familiarity


# ------------------------------------------------------------------------------------- the load model
def test_participation_and_visibility_gate_before_the_priority(mock_run, places):
    """p = 1 for anyone the fact involves; 0 for a fact they cannot have access to."""
    obs = places("leo")
    assert attention_prob(obs, dict(ANON, involves=["leo"]), BEAT, mock_run.cfg) == 1.0
    assert attention_prob(obs, PRIVATE, BEAT, mock_run.cfg) == 0.0
    places("leo", arena="Stacks")
    assert attention_prob(obs, ARENA_ONLY, BEAT, mock_run.cfg) == 0.0      # same building, wrong room
    assert attention_prob(obs, ANON, BEAT, mock_run.cfg) > 0.0             # building-wide fact still lands


def test_p_attend_rises_with_salience(mock_run, places):
    obs = places("leo")
    ps = [attention_prob(obs, dict(ANON, salience=s), BEAT, mock_run.cfg) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert ps == sorted(ps) and ps[0] < ps[-1]


def test_crowding_is_counted_per_room_not_per_building(mock_run, places):
    """Perceptual load is what is in the observer's own field. Six people in the room load attention;
    the same six two rooms away in the same building do not (until `building_crowd_factor` is set)."""
    obs = places("leo")
    alone = attention_prob(obs, ANON, BEAT, mock_run.cfg)
    crowd = [a for a in mock_run.agents if a != "leo"][:6]
    for aid in crowd:
        places(aid)
    packed = attention_prob(obs, ANON, BEAT, mock_run.cfg)
    assert packed < alone * 0.8                                   # 0.92 ** 5 = 0.66

    for aid in crowd:
        places(aid, arena="Stacks")                               # same building, different room
    assert attention_prob(obs, ANON, BEAT, mock_run.cfg) == pytest.approx(alone)

    cfg = {**mock_run.cfg, "perception": {**mock_run.cfg["perception"], "building_crowd_factor": 0.97}}
    elsewhere = attention_prob(obs, ANON, BEAT, cfg)
    assert packed < elsewhere < alone                             # second-order, and weaker than in-room load


def test_familiarity_is_a_gain_on_the_priority_not_an_escape_hatch(mock_run, places):
    """v2 added `familiarity_weight * fam` after every multiplier, so under load the social term was the
    only channel left and the familiar/stranger ratio grew with the crowd. As a multiplicative gain the
    ratio is constant and familiar faces lose attention under load like everyone else."""
    obs = places("leo")
    other, fam = _familiar_other(obs, mock_run.ctx)
    assert fam >= 0.3, "the mock population should give leo someone it knows"
    known = _fact_about(obs, other)

    def ratio():
        return (attention_prob(obs, known, BEAT, mock_run.cfg)
                / attention_prob(obs, ANON, BEAT, mock_run.cfg))

    quiet_ratio, quiet_p = ratio(), attention_prob(obs, known, BEAT, mock_run.cfg)
    for aid in [a for a in mock_run.agents if a not in ("leo", other)][:6]:
        places(aid)
    assert ratio() == pytest.approx(quiet_ratio)                  # load does not change the social ratio
    assert quiet_ratio == pytest.approx(1 + mock_run.cfg["perception"]["familiarity_weight"] * fam)
    assert attention_prob(obs, known, BEAT, mock_run.cfg) < quiet_p * 0.8    # and it is under load itself


def test_attention_probability_is_clamped_to_one(mock_run, places):
    obs = places("leo")
    other, fam = _familiar_other(obs, mock_run.ctx)
    cfg = {**mock_run.cfg, "perception": {**mock_run.cfg["perception"],
                                          "base_attention": 1.0, "familiarity_weight": 5.0}}
    p = attention_prob(obs, _fact_about(obs, other, salience=1.0), BEAT, cfg)
    assert p == 1.0 and 0.0 <= p <= 1.0


# ------------------------------------------------------------------------------ the memory bottleneck
@pytest.fixture(scope="module")
def bottlenecked(tmp_path_factory):
    """A short mock run whose capacity actually binds (the default 80 does not in half a day)."""
    return run_sim(tmp_path_factory.mktemp("bottleneck"), {"memory": {"memory_capacity": 20}}, name="cap20")


def _forgotten(sim) -> list:
    return [r for r in (json.loads(l) for l in open(sim.run_dir / "trace.jsonl"))
            if r["type"] == "memory_forgotten"]


def _source(sim, node_id: str) -> str | None:
    m = sim.meta.get(node_id)
    return m.source_type if m else None


def test_forgetting_binds_and_the_store_stays_at_capacity(bottlenecked):
    cap = bottlenecked.cfg["memory"]["memory_capacity"]
    dropped = _forgotten(bottlenecked)
    assert len(dropped) > 0, "capacity forgetting never fired: the bottleneck has no mechanism"
    sizes = [len(a.a_mem.all_nodes()) for a in bottlenecked.agents.values()]
    assert sum(1 for s in sizes if s >= cap) >= len(sizes) // 2, "the cap binds for barely anyone"
    # Reflection thoughts are added outside the encode path (`memory/reflection.py` does not call
    # forget), so a store can sit a few nodes over the cap until the next encoding trims it. It may
    # never run away from it.
    assert max(sizes) <= cap + 10, sizes


def test_seed_memories_are_the_floor_of_the_store(bottlenecked):
    """`memory.protect_sources` keeps what the agent was handed about who the people around it are. They
    are old, low-importance and never re-accessed, so importance x recency would drop them first."""
    assert not [r for r in _forgotten(bottlenecked) if _source(bottlenecked, r["node_id"]) == "seed"]
    left = [n for a in bottlenecked.agents.values() for n in a.a_mem.all_nodes()
            if _source(bottlenecked, n.node_id) == "seed"]
    assert left, "no seed memories survived, so the floor is not doing anything"


def test_the_floor_is_the_only_thing_keeping_the_seeds(tmp_path):
    """Squeeze two finished agents down to exactly their number of seed memories. With the floor the
    store that survives IS the seeds; with `protect_sources: []` the v2 ranking takes most of them,
    because they are old, importance 5 and mostly never re-accessed. (On a 3-day mock run at capacity 80
    the unprotected ranking ends with 26% of the seeds left.) Its own run: it evicts as it measures."""
    sim = run_sim(tmp_path, {"memory": {"memory_capacity": 30}}, name="squeeze")
    ranked = sorted(sim.agents.values(),
                    key=lambda a: -sum(1 for n in a.a_mem.all_nodes() if _source(sim, n.node_id) == "seed"))

    def squeeze(agent, protect):
        seeds = {n.node_id for n in agent.a_mem.all_nodes() if _source(sim, n.node_id) == "seed"}
        assert len(seeds) >= 3 and len(agent.a_mem.all_nodes()) > len(seeds)
        agent.cfg = {**agent.cfg, "memory": {**agent.cfg["memory"], "memory_capacity": len(seeds),
                                             "protect_sources": protect}}
        forget(agent)
        return seeds, {n.node_id for n in agent.a_mem.all_nodes()}

    seeds, left = squeeze(ranked[0], ["seed"])
    assert left == seeds, "the floor should leave exactly the seeds standing"
    seeds, left = squeeze(ranked[1], [])
    assert len(seeds & left) < len(seeds), "without the floor, seeds should lose to recent ambient sightings"
