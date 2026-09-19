"""Campus places (world graph + tile map) and campus errands (scheduler.plan_day).

Map: every ARENAS location/arena is a place/arena of frontend/homewood_map.json with enough walkable, reachable
standing spots. World: the graph is reciprocal and connected, the Quad stays the hub, and the original eight places
(and their shortest paths) are unchanged. Errands: deterministic per seed, inserted into free gaps (never over a
routine entry, never interrupting a class / meal / shift), and used by a real 1-day mock run.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.profile import load_population
from backend.config import deep_merge, load_config
from backend.simulation import roster as R
from backend.simulation.rngs import seed_rng
from backend.simulation.scheduler import (ERRANDS, apply_shifts, errand_places, plan_day, protected,
                                          routine_position)
from backend.simulation.world import (ARENAS, HOMEWOOD_LABELS, MAP_POS, PUBLIC_PLACES, WORLD_GRAPH, Clock,
                                      shortest_path)

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "frontend/homewood_map.json"
ORIGINAL_GRAPH = {
    "Dorm": ["Dining Hall", "Quad"],
    "Dining Hall": ["Dorm", "Quad", "Classroom"],
    "Classroom": ["Dining Hall", "Library", "Quad"],
    "Library": ["Classroom", "Quad", "Research Lab", "Cafe"],
    "Research Lab": ["Library", "Quad"],
    "Gym": ["Quad"],
    "Cafe": ["Quad", "Library"],
    "Quad": ["Dorm", "Dining Hall", "Classroom", "Library", "Research Lab", "Gym", "Cafe"],
}
ORIGINAL_ARENAS = {
    "Dorm": ["Room 214", "Room 310", "Room 118", "Room 105", "Room 402", "Grad Apartment", "Lounge"],
    "Dining Hall": ["Main Floor"], "Classroom": ["Lecture Hall", "Seminar Room"],
    "Library": ["Study Tables", "Quiet Floor"], "Research Lab": ["Dry Lab", "Wet Lab", "Makerspace", "Stockroom"],
    "Gym": ["Main Floor"], "Cafe": ["Counter"], "Quad": ["Lawn"],
}
NEW_PLACES = [p for p in ARENAS if p not in ORIGINAL_ARENAS]


# ------------------------------------------------------------------------------------------------ world graph
def test_new_places_are_appended_and_originals_unchanged():
    assert len(NEW_PLACES) == 10
    for loc, arenas in ORIGINAL_ARENAS.items():
        assert ARENAS[loc] == arenas                              # default arena (ARENAS[loc][0]) unchanged
        assert WORLD_GRAPH[loc][:len(ORIGINAL_GRAPH[loc])] == ORIGINAL_GRAPH[loc]
    assert list(ARENAS)[:8] == list(ORIGINAL_ARENAS)
    assert set(WORLD_GRAPH) == set(ARENAS) == set(HOMEWOOD_LABELS) == set(MAP_POS)


def test_graph_reciprocal_connected_quad_hub():
    for a, ns in WORLD_GRAPH.items():
        assert a not in ns and len(ns) == len(set(ns)), a
        for b in ns:
            assert a in WORLD_GRAPH[b], (a, b)
    seen, q = {"Quad"}, deque(["Quad"])
    while q:
        for n in WORLD_GRAPH[q.popleft()]:
            if n not in seen:
                seen.add(n); q.append(n)
    assert seen == set(WORLD_GRAPH)
    deg = {k: len(v) for k, v in WORLD_GRAPH.items()}
    assert deg["Quad"] == max(deg.values()) and sorted(deg.values())[-2] < deg["Quad"]
    for a in WORLD_GRAPH:
        p = shortest_path(a, "Quad")
        assert p[0] == a and p[-1] == "Quad" and all(y in WORLD_GRAPH[x] for x, y in zip(p, p[1:]))


def test_original_shortest_paths_unchanged():
    """New edges are appended, so BFS between two original places takes exactly the old route."""
    def bfs(graph, a, b):
        if a == b:
            return [a]
        prev, q = {a: None}, deque([a])
        while q:
            cur = q.popleft()
            for nb in graph[cur]:
                if nb not in prev:
                    prev[nb] = cur
                    if nb == b:
                        path = [b]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        return path[::-1]
                    q.append(nb)
    for a in ORIGINAL_GRAPH:
        for b in ORIGINAL_GRAPH:
            assert shortest_path(a, b) == bfs(ORIGINAL_GRAPH, a, b), (a, b)


def test_map_pos_normalised_and_distinct():
    pts = list(MAP_POS.values())
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pts)
    assert len(set(pts)) == len(pts)


def test_public_places_known_to_everyone():
    assert set(PUBLIC_PLACES) <= set(ARENAS) and "Apartments" not in PUBLIC_PLACES
    for pop in ("configs/population/homewood8.yaml", "configs/population/coop13.yaml"):
        profiles, _ = load_population(pop, None, include_reserves=True)
        for p in profiles.values():
            assert set(PUBLIC_PLACES) <= set(p.known_locations), p.id
            assert all(k in WORLD_GRAPH for k in p.known_locations)


# ------------------------------------------------------------------------------------------------ tile map
@pytest.fixture(scope="module")
def tilemap():
    d = json.loads(MAP.read_text())
    n = d["width"] * d["height"]
    cost, i = [0] * n, 0
    for k in range(0, len(d["cost"]), 2):
        v, r = d["cost"][k], d["cost"][k + 1]
        cost[i:i + r] = [v] * r
        i += r
    assert i == n
    d["_cost"] = cost
    return d


def test_every_arena_is_on_the_map_with_spots(tilemap):
    W, H, cost, places = tilemap["width"], tilemap["height"], tilemap["_cost"], tilemap["places"]
    for loc, arenas in ARENAS.items():
        assert loc in places, loc
        pl = places[loc]
        assert pl["name"] == loc and pl["label"] == HOMEWOOD_LABELS[loc]
        for a in arenas:
            assert a in pl["arenas"], (loc, a)
            spots = [tuple(s) for s in pl["arenas"][a]["spots"]]
            assert len(spots) >= (4 if loc == "Dorm" else 8), (loc, a, len(spots))
            assert len(set(spots)) == len(spots)
            for x, y in spots:
                assert 0 <= x < W and 0 <= y < H and cost[y * W + x] > 0, (loc, a, (x, y))
            x0, y0, x1, y1 = pl["box"]
            assert all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in spots), (loc, a)


def test_every_spot_reachable_from_the_quad(tilemap):
    W, H, cost, places = tilemap["width"], tilemap["height"], tilemap["_cost"], tilemap["places"]
    start = tuple(places["Quad"]["arenas"]["Lawn"]["spots"][0])
    seen = bytearray(W * H)
    seen[start[1] * W + start[0]] = 1
    q = deque([start])
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < W and 0 <= ny < H and not seen[ny * W + nx] and cost[ny * W + nx] > 0:
                seen[ny * W + nx] = 1
                q.append((nx, ny))
    for loc in ARENAS:
        for a, ar in places[loc]["arenas"].items():
            assert all(seen[y * W + x] for x, y in ar["spots"]), (loc, a)


def test_new_places_drawn_as_distinct_real_buildings(tilemap):
    places = tilemap["places"]
    labels = [HOMEWOOD_LABELS[p] for p in ARENAS]
    assert len(set(labels)) == len(labels)
    for p in NEW_PLACES:
        assert places[p]["kind"] in ("building", "stop")
        if places[p]["kind"] == "building":
            assert places[p]["entrance"], p
    # no two places' footprints overlap (Museum's box includes its garden; Shuttle Stop is a verge)
    boxes = {p: places[p]["boxes"][0] for p in ARENAS if p != "Quad"}
    for a in boxes:
        for b in boxes:
            if a < b:
                A, B = boxes[a], boxes[b]
                assert A[2] < B[0] or B[2] < A[0] or A[3] < B[1] or B[3] < A[1] or {a, b} == {"Gym", "Athletic Center"}, (a, b)


# ------------------------------------------------------------------------------------------------ errands
def _cfg(**routine):
    # merged after loading: the errand knobs are not in configs/default.yaml, so an override would warn
    return deep_merge(load_config("configs/baseline.yaml"), {"routine": routine})


def _plan(cfg, prof, ws, day):
    return plan_day(SimpleNamespace(cfg=cfg, profile=prof), Clock(cfg), seed_rng(ws, "plan", prof.id, day))


def _errands(plan, base):
    """-> [(errand_entry, resume_entry)], checking the base (errand-free) plan is kept entry for entry."""
    it = iter(plan)
    extra = []
    for e in base:
        for x in it:
            if x == e:
                break
            extra.append(x)
        else:
            raise AssertionError(f"routine entry {e} dropped")
    extra += list(it)
    assert len(extra) % 2 == 0
    return list(zip(extra[0::2], extra[1::2]))


def test_errands_deterministic_per_seed():
    cfg = _cfg()
    profiles, _ = load_population(cfg["population"], cfg["population_size"])
    a = {(aid, d): _plan(cfg, p, 7, d) for aid, p in profiles.items() for d in (1, 2)}
    b = {(aid, d): _plan(cfg, p, 7, d) for aid, p in profiles.items() for d in (1, 2)}
    c = {(aid, d): _plan(cfg, p, 8, d) for aid, p in profiles.items() for d in (1, 2)}
    assert a == b and a != c


def test_errands_fill_free_gaps_only():
    cfg, off = _cfg(), _cfg(errands_per_day=0)
    clock = Clock(cfg)
    profiles, _ = load_population(cfg["population"], cfg["population_size"])
    places = set(errand_places(cfg))
    n = 0
    for ws in (1, 2, 3):
        for day in (1, 2, 3):
            for aid, prof in profiles.items():
                base = _plan(off, prof, ws, day)
                plan = _plan(cfg, prof, ws, day)
                pairs = _errands(plan, base)
                assert len(pairs) <= 2
                spans = []
                for go, back in pairs:
                    n += 1
                    assert go["location"] in places and go["arena"] in ARENAS[go["location"]]
                    assert (go["arena"], go["activity"]) in ERRANDS[go["location"]]
                    assert go["k"] < back["k"] <= go["k"] + 3
                    # no routine entry starts during the errand or the moment it ends
                    assert not any(go["k"] <= e["k"] <= back["k"] for e in base)
                    # it interrupts an interruptible entry, which the agent then resumes
                    cur = routine_position(SimpleNamespace(profile=prof, day_plan=base), go["k"])
                    assert cur in base and cur["k"] < go["k"] and not protected(cur)
                    assert {k: back[k] for k in ("location", "arena", "activity")} == \
                           {k: cur[k] for k in ("location", "arena", "activity")}
                    assert go["location"] != cur["location"]
                    assert 2 <= go["k"] and back["k"] <= clock.ticks_per_day
                    spans.append((go["k"], back["k"]))
                spans.sort()
                assert all(b0 > a1 + 1 for (a0, a1), (b0, b1) in zip(spans, spans[1:]))   # errands never overlap
    assert n >= 0.9 * 3 * 3 * len(profiles) * 2          # almost every agent-day finds room for both errands


def test_classes_meals_and_rest_are_protected():
    for act, loc in (("attending the Probability lecture", "Library"), ("eating lunch", "Quad"),
                     ("working a barista shift", "Cafe"), ("winding down", "Dorm"), ("sleeping in", "Dorm"),
                     ("rehearsing with the a cappella group", "Quad"), ("studying", "Classroom"),
                     ("chatting", "Dining Hall"), ("playing chess with Sam", "Dorm"),
                     ("meeting up with the outdoors club", "Quad"), ("tutoring intro economics", "Library")):
        assert protected({"location": loc, "activity": act}), act
    for act, loc in (("studying", "Library"), ("running machine-learning experiments", "Research Lab"),
                     ("hanging out on the lawn", "Quad"), ("reading a book", "Dorm"),
                     ("reading essays for class", "Library"), ("studying for an exam", "Library")):
        assert not protected({"location": loc, "activity": act}), act


def test_errands_config_knobs():
    cfg = _cfg(errands_per_day=4, errand_places=["Shuttle Stop"])
    profiles, _ = load_population(cfg["population"], cfg["population_size"])
    off = _cfg(errands_per_day=0)
    seen = 0
    for aid, prof in profiles.items():
        pairs = _errands(_plan(cfg, prof, 3, 1), _plan(off, prof, 3, 1))
        assert all(go["location"] == "Shuttle Stop" and go["arena"] == "Bench" for go, _ in pairs)
        seen += len(pairs)
    assert seen > len(profiles)
    with pytest.raises(ValueError):
        errand_places({"routine": {"errand_places": ["Moon Base"]}})


def test_errands_respect_shift_windows():
    """v3 roster: errands avoid the role's shift windows, so the roster-on plan is exactly the roster-off plan with
    the shifts laid over it (every errand survives the overlay)."""
    def cfg(enabled):
        return load_config(None, {"population": "configs/population/coop13.yaml", "seed": 1, "world_seed": 13,
                                  "simulation_days": 7, "llm": {"backend": "mock"}, "roster": {"enabled": enabled}})
    on, off = cfg(True), cfg(False)
    clock = Clock(on)
    profiles, _ = load_population(on["population"], None, include_reserves=True)
    no_err = deep_merge(off, {"routine": {"errands_per_day": 0}})
    for day in (1, 2, 3):
        for aid, prof in profiles.items():
            if prof.role is None:
                continue
            wins = R.shift_ticks(on, prof.role, clock)
            base = _plan(off, prof, 13, day)
            pairs = _errands(base, _plan(no_err, prof, 13, day))
            for go, back in pairs:
                assert all(back["k"] < ks or go["k"] > ke for ks, ke in wins), (aid, day, go)
            loc, arena = R.ROLE_PLACE[prof.role]
            shift = {"location": loc, "arena": arena, "activity": R.ROLE_ACTIVITY[prof.role]}
            plan = _plan(on, prof, 13, day)
            assert plan == apply_shifts(base, wins, shift, prof.home)
            for go, back in pairs:
                assert go in plan and back in plan


def test_world_script_plans_the_same_errands():
    """world_script plans through the same plan_day call and stream, so its planned positions include errands."""
    from backend.simulation.world_script import _Plans
    cfg = _cfg()
    clock = Clock(cfg)
    profiles, _ = load_population(cfg["population"], cfg["population_size"])
    plans = _Plans(cfg, profiles, clock)
    for aid, prof in profiles.items():
        plan = plan_day(SimpleNamespace(cfg=cfg, profile=prof), clock, seed_rng(plans.ws, "plan", aid, 1))
        agent = SimpleNamespace(profile=prof, day_plan=plan)
        for k in range(clock.ticks_per_day):
            assert plans.position(aid, k) == routine_position(agent, k)


def test_one_day_mock_run_uses_the_new_places(tmp_path):
    from backend.simulation.engine import Simulation
    cfg = load_config("configs/baseline.yaml", {"simulation_days": 1, "llm": {"backend": "mock", "max_workers": 4}})
    sim = Simulation(cfg, tmp_path / "run", progress=False)
    sim.run()
    visited = set()
    for line in open(tmp_path / "run" / "frames.jsonl"):
        for a in json.loads(line)["agents"].values():
            assert a["arena"] in ARENAS[a["location"]]
            visited.add(a["location"])
    assert len(visited & set(NEW_PLACES)) >= 6, sorted(visited)
