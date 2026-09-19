"""Pre-generated world script (ontology v2 §1.1-1.7). SIMULATOR-ONLY.

The whole sequence of latent-event instances is generated before the simulation from random
streams seeded only by `world_seed` (null -> `seed`), one stream per purpose. It never looks at
agents' decisions, LLM outputs or cognitive mechanisms, so every condition that shares
`world_seed` sees exactly the same events (common random numbers across conditions). Routine
plans are part of the world: they are seeded by `world_seed` too (`seed_rng(world_seed, "plan",
aid, day)`, the same call the engine makes), so `seed` never moves a beat.

Schedule (`latent_events.schedule`):
- `balanced` (default): every day gets the same number of instances, round(event_rate x usable
  ticks per day) (a start in the last SPAN ticks would cross the day end), with start ticks spread
  over the day (one per equal stratum, jittered inside it). Families come in blocks that contain
  every family once (weighted blocks by systematic sampling if `family_weights` differ), and each
  family's instances are held out at a steady `holdout_frac` share (systematic, random phase).
- `bernoulli`: the v2.0 behaviour: an instance starts at each eligible tick with probability
  `event_rate`; family and holdout are independent draws per instance.

Each candidate instance draws a FIXED number of numbers from every stream, whatever the config
chooses to do with them. So switching a world mechanism (assignment, referents, structure,
link_visibility) changes only what that mechanism controls: timing, families and holdout stay
aligned across conditions, and so do skins, except that referent reuse (referents.enabled) picks
the skin of the reused referent's domain (the reuse is decided first, then the family's skin for
that domain).

Every beat happens at P's PLANNED routine position (or the skin's pinned location), resolved here
at generation time; involvement and forced movers come from the shared shape (structures.SHAPE,
structures.BEAT_MOVERS), never from who a skin names, so presence carries no family information.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.simulation import structures as ST
from backend.simulation.circles import circle_of
from backend.simulation.latent_events import EventInstance
from backend.simulation.referents import CATEGORIES
from backend.simulation.rngs import seed_rng, world_seed
from backend.simulation.scheduler import plan_day, routine_position
from backend.simulation.world import ARENAS, WORLD_GRAPH, _minutes, default_arena

GENERATOR = "script_v2"
STRUCTURES = ("real", "scrambled", "none")
ASSIGNMENTS = ("none", "balanced", "home")
SCHEDULES = ("balanced", "bernoulli")
PURPOSES = ("timing", "family", "skin", "cast", "slots", "referent", "holdout", "compose", "link")
N_DRAWS = {"family": 1, "holdout": 1, "skin": 1, "link": 1, "compose": 3, "cast": 4, "slots": 6, "referent": 9}
RELATED = 0.3            # familiarity from which a person counts as related to P ({S}); below: unrelated ({Q})

# NPC fallbacks when no suitable free agent exists (v1 style)
NPC_S = ["a classmate", "a friend from another dorm", "a TA", "a lab technician"]
NPC_Q = ["a student nobody seemed to know", "a student from another department", "a visiting student"]


def _pick(u: float, n: int) -> int:
    return min(int(u * n), n - 1)


def _render(text: str, subst: dict) -> str:
    for k, v in subst.items():
        text = text.replace("{" + k + "}", v)
    return text[0].upper() + text[1:] if text else text


class _Plans:
    """Planned routine positions, built exactly as the engine builds them: the same plan_day call on the
    same stream, `seed_rng(world_seed, "plan", aid, day)` (routines are part of the world)."""

    def __init__(self, cfg: dict, profiles: dict, clock):
        self.cfg, self.profiles, self.clock, self.cache = cfg, profiles, clock, {}
        self.ws = world_seed(cfg)

    def position(self, aid: str, tick: int) -> dict:
        day = self.clock.day_of(tick)
        if (aid, day) not in self.cache:
            prof = self.profiles[aid]
            plan = plan_day(SimpleNamespace(cfg=self.cfg, profile=prof), self.clock,
                            seed_rng(self.ws, "plan", aid, day))
            self.cache[(aid, day)] = SimpleNamespace(profile=prof, day_plan=plan)
        return routine_position(self.cache[(aid, day)], tick % self.clock.ticks_per_day)

    def place(self, aid: str, tick: int, pinned: str | None) -> tuple[str, str]:
        pos = self.position(aid, tick)
        if not pinned or pinned == pos["location"]:
            return pos["location"], pos["arena"]
        return pinned, default_arena(pinned)


def _pools(skins: list[dict], fams: list[str], structure: str, holdout_frac: float, hint: str = "") -> dict:
    """(family|"*", holdout) -> skins sorted by key; raises when a needed pool is empty."""
    pools = defaultdict(list)
    for s in sorted(skins, key=lambda s: s["key"]):
        if s["family"] in fams:
            pools[(s["family"], bool(s["holdout"]))].append(s)
            pools[("*", bool(s["holdout"]))].append(s)
    need = [h for h, on in ((False, holdout_frac < 1), (True, holdout_frac > 0)) if on]
    for h in need:
        if structure == "real":
            for f in fams:
                if not pools[(f, h)]:
                    raise ValueError(f"no {'holdout' if h else 'train'} skins for family {f}{hint}")
        elif len(pools[("*", h)]) < len(ST.GROUPS):
            raise ValueError(f"structure={structure} needs >= {len(ST.GROUPS)} distinct "
                             f"{'holdout' if h else 'train'} skins{hint}")
    return pools


def per_day(rate: float, clock) -> int:
    """Instances per day under the balanced schedule: round(event_rate x usable ticks per day), where a
    start in the last SPAN ticks of a day would cross the day end (at most one start per tick)."""
    usable = max(0, clock.ticks_per_day - ST.SPAN)
    return min(usable, int(np.floor(max(0.0, rate) * usable + 0.5)))


def _family_blocks(fams: list[str], w: np.ndarray, n: int, rng) -> list[str]:
    """n family labels in consecutive blocks of len(fams). Equal weights: each block is a random
    permutation of the families (every family once). Unequal weights: each block is a systematic sample
    of the weights (each family floor/ceil of its expected count), in random order."""
    k, out = len(fams), []
    p = w / w.sum()
    equal = bool(np.allclose(p, p[0]))
    cum = np.cumsum(p)
    while len(out) < n:
        if equal:
            block = [fams[i] for i in rng.permutation(k)]
        else:
            x = (np.arange(k) + rng.random()) / k
            block = [fams[min(int(np.searchsorted(cum, v, side="right")), k - 1)] for v in x]
            block = [block[i] for i in rng.permutation(k)]
        out += block
    return out[:n]


def _candidates(schedule: str, rate: float, holdout_frac: float, fams: list[str], w: np.ndarray, clock, ws: int,
                rng: dict) -> list[tuple]:
    """(start tick, family | None, holdout | None) of every candidate instance, in start order. Under
    `bernoulli` the family and holdout are left to the per-candidate draws (None)."""
    tpd = clock.ticks_per_day
    if schedule == "bernoulli":
        out = []
        for tick in range(clock.total_ticks):
            if rng["timing"].random() >= rate or tick % tpd + ST.SPAN >= tpd:   # no instance crosses a day end
                continue
            out.append((tick, None, None))
        return out
    n_day, usable = per_day(rate, clock), tpd - ST.SPAN
    ticks = []
    for day in range(clock.days):
        jit = rng["timing"].random(n_day)                    # one stratum per instance, jittered inside it
        for j in range(n_day):
            lo, hi = j * usable // n_day, (j + 1) * usable // n_day
            ticks.append(day * tpd + lo + _pick(jit[j], hi - lo))
    fam_seq = _family_blocks(fams, w, len(ticks), seed_rng(ws, "family", "blocks"))
    # holdout: the j-th instance of family f is held out iff floor((j+1)h + o_f) > floor(jh + o_f), a steady
    # share h of every family's instances with a random phase o_f
    phase = {f: float(seed_rng(ws, "holdout", "phase", f).random()) for f in fams}
    seen: dict = defaultdict(int)
    out = []
    for tick, f in zip(ticks, fam_seq):
        j = seen[f]
        seen[f] += 1
        hold = np.floor((j + 1) * holdout_frac + phase[f]) > np.floor(j * holdout_frac + phase[f])
        out.append((tick, f, bool(hold)))
    return out


def generate(cfg: dict, profiles: dict, circles: dict[str, list[str]], clock,
             skins: list[dict] | None = None) -> list[EventInstance]:
    """The run's full list of latent-event instances, in start order (ids ev000, ev001, ...).

    `profiles`: agent id -> AgentProfile (after any topology routine additions: plans must match
    the engine's). `circles`: disjoint circles (circles.load_circles or generated topology).
    `skins`: override of the skin set (tests); default: backend.simulation.skins.
    `latent_events.script_from` short-circuits to load_for_run() of that file, which raises a clear
    error if the saved script does not fit this run (agents, clock, circles, world_seed).
    """
    lc = cfg["latent_events"]
    if lc.get("generator") == "llm":
        raise ValueError("latent_events.generator: llm was removed in ontology v2; instances come from the "
                         "pre-generated world script (structure schemas x skins). Remove the key.")
    if lc.get("script_from"):
        return load_for_run(lc["script_from"], cfg, profiles, circles, clock)
    structure = lc.get("structure", "real")
    if structure not in STRUCTURES:
        raise ValueError(f"latent_events.structure must be one of {STRUCTURES}, got {structure!r}")
    asg = lc.get("assignment") or {}
    mode, strength = asg.get("mode", "none"), float(asg.get("strength", 0.0))
    if mode not in ASSIGNMENTS:
        raise ValueError(f"latent_events.assignment.mode must be one of {ASSIGNMENTS}, got {mode!r}")
    fams = list(lc["families"])
    if not fams or set(fams) - set(ST.FAMILIES):
        raise ValueError(f"latent_events.families must be a non-empty subset of {ST.FAMILIES}, got {fams}")
    rate, holdout_frac = float(lc["event_rate"]), float(lc.get("holdout_frac", 0.0))
    if not 0.0 <= holdout_frac <= 1.0:
        raise ValueError(f"latent_events.holdout_frac must be in [0, 1], got {holdout_frac}")
    schedule = lc.get("schedule") or "balanced"
    if schedule not in SCHEDULES:
        raise ValueError(f"latent_events.schedule must be one of {SCHEDULES}, got {schedule!r}")
    link_vis = float(lc.get("link_visibility", 0.0))
    rc = lc.get("referents") or {}
    ref_on, reuse = bool(rc.get("enabled", False)), float(rc.get("reuse", 0.0))
    hint = ""
    if skins is None:
        from backend.simulation.skins import load_errors, load_skins
        skins = load_skins()
        hint = f" (skin modules that failed to load: {load_errors()})" if load_errors() else ""
    pools = _pools(skins, fams, structure, holdout_frac, hint) if rate > 0 else {}

    ws = world_seed(cfg)
    rng = {p: seed_rng(ws, p) for p in PURPOSES}
    w = np.array([float((lc.get("family_weights") or {}).get(f, 1.0)) for f in fams])
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"latent_events.family_weights must be >= 0 with a positive sum, got {w.tolist()}")
    cum = np.cumsum(w / w.sum())
    circles = {c: [m for m in circles[c] if m in profiles] for c in sorted(circles)}
    member_of, circ_ids = circle_of(circles), sorted(circles)
    # home: families -> circles round-robin, rotated by world_seed (counterbalanced across seeds)
    home = {f: circ_ids[(i + ws) % len(circ_ids)] for i, f in enumerate(sorted(fams))} if circ_ids else {}
    offset = int(seed_rng(ws, "cast", "offset").integers(len(circ_ids))) if circ_ids else 0
    plans = _Plans(cfg, profiles, clock)
    met: dict = defaultdict(lambda: defaultdict(set))    # circle id | "*" -> domain -> referent indices met
    active: list[tuple[int, list[str]]] = []              # (last beat tick, agents) of scheduled instances
    out: list[EventInstance] = []
    n_cand = 0

    starts = _candidates(schedule, rate, holdout_frac, fams, w, clock, ws, rng) if rate > 0 else []
    for tick, fam_b, hold_b in starts:
        u = {p: rng[p].random(n) for p, n in N_DRAWS.items()}             # fixed draws per candidate
        fam = fam_b if fam_b is not None else \
            fams[min(int(np.searchsorted(cum, u["family"][0], side="right")), len(fams) - 1)]
        holdout = hold_b if hold_b is not None else bool(u["holdout"][0] < holdout_frac)
        circle = None
        if mode == "balanced" and circ_ids:
            circle = circ_ids[(offset + n_cand) % len(circ_ids)]
        elif mode == "home" and circ_ids:
            circle = home[fam]
        n_cand += 1

        busy = {a for end, ags in active if end >= tick for a in ags}
        free = sorted(a for a in profiles if a not in busy)
        if not free:
            continue
        members = [a for a in circles.get(circle, []) if a in free]
        from_circle = bool(members) and bool(u["cast"][0] < strength)
        cast_pool = members if from_circle else free
        p = cast_pool[_pick(u["cast"][1], len(cast_pool))]
        prof = profiles[p]
        mem_key = member_of.get(p, "*")

        # referent reuse is decided BEFORE the skin: with prob. `reuse`, one referent P's circle has already
        # met (any domain the instance's pool covers, so the holdout draw is respected) is picked uniformly,
        # and the instance then uses that domain's skin. Every family has one skin per domain, so the reused
        # domain says nothing about the family.
        pool = pools[(fam, holdout)] if structure == "real" else list(pools[("*", holdout)])
        ur0 = u["referent"][:3]
        again = None
        if ref_on and ur0[0] < reuse:
            doms = {s["domain"] for s in pool}
            seen = sorted((d, i) for d, idx in met[mem_key].items() if d in doms for i in idx)
            if seen:
                again = seen[_pick(ur0[1], len(seen))]
        if structure == "real":
            if again:
                same = [s for s in pool if s["domain"] == again[0]]
                skin = same[_pick(u["skin"][0], len(same))]
            else:
                skin = pool[_pick(u["skin"][0], len(pool))]
            src = {g: skin for g in ST.GROUPS}
            label, skin_key, composed = fam, skin["key"], None
        else:
            src = {}
            if again:
                first = [s for s in pool if s["domain"] == again[0]]
                src["n0"] = first[_pick(u["compose"][0], len(first))]
                pool.remove(src["n0"])
            for i, g in enumerate(ST.GROUPS):
                if g not in src:
                    src[g] = pool.pop(_pick(u["compose"][i], len(pool)))
            label, skin_key = (fam if structure == "scrambled" else "E0"), None
            composed = [{"node": n, "family": src[_group(n)]["family"], "skin": src[_group(n)]["key"]}
                        for n in ST.NODES]

        role2 = ST.second_role(src["n1"]["family"])
        others = [a for a in free if a != p]
        if role2 == "S":
            cands = [a for a in others if prof.rel(a).familiarity >= RELATED]
            if from_circle:
                cands = [a for a in cands if a in circles[circle]] or cands
        else:
            cands = [a for a in others if prof.rel(a).familiarity < RELATED]
        if cands:
            second = cands[_pick(u["cast"][2], len(cands))]
            roles = {"P": {"agent": p, "name": prof.first_name},
                     role2: {"agent": second, "name": profiles[second].first_name}}
        else:
            npcs = NPC_S if role2 == "S" else NPC_Q
            second = None
            roles = {"P": {"agent": p, "name": prof.first_name},
                     role2: {"agent": None, "name": npcs[_pick(u["cast"][3], len(npcs))]}}

        # referents: one per distinct domain whose text uses {R} (n0 always does, and comes first)
        ref_by_dom, referents = {}, []
        for g in ST.GROUPS:
            sk, texts = src[g], [src[g]["facts"][n] for n in ST.NODES if _group(n) == g]
            d = sk["domain"]
            if d in ref_by_dom or not any("{R}" in t for t in texts):
                continue
            ur, names = u["referent"][3 * len(referents):3 * len(referents) + 3], CATEGORIES[d]
            if g == "n0" and again:
                idx, reused = again[1], True
            else:
                seen = sorted(met[mem_key][d]) if ref_on and g != "n0" else []
                if seen and ur[0] < reuse:
                    idx, reused = seen[_pick(ur[1], len(seen))], True
                else:
                    idx, reused = _pick(ur[2], len(names)), False
            ref_by_dom[d] = names[idx]
            referents.append({"id": f"{d}:{idx}", "domain": d, "name": names[idx], "reused": reused})

        # slot fillers, one set per source skin (shared by all nodes that come from it)
        fills = {}
        for gi, g in enumerate(ST.GROUPS):
            sk = src[g]
            if sk["key"] not in fills:
                fills[sk["key"]] = {name: opts[_pick(u["slots"][2 * gi + si], len(opts))]
                                    for si, name in enumerate(ST.SLOT_NAMES)
                                    if (opts := (sk.get("slots") or {}).get(name))}

        # the shared shape: every beat at P's planned position (or the skin's pin); involvement and movers
        # from structures.SHAPE / BEAT_MOVERS only
        eid = f"ev{len(out):03d}"
        t = [tick + off for off in ST.BEAT_OFFSETS]
        who = {"P": p, "second": second}
        places = [plans.place(p, t[b], _pin(src[g], g)) for b, g in enumerate(ST.GROUPS)]
        beats = [{"idx": b, "tick": t[b], "location": places[b][0], "arena": places[b][1], "facts": [],
                  "movers": sorted({who[r] for r in ST.BEAT_MOVERS[b] if who[r]})}
                 for b in range(len(ST.BEAT_OFFSETS))]
        for node in ST.NODES:
            sh, sk = ST.SHAPE[node], src[_group(node)]
            subst = {"P": prof.first_name, "S": roles[role2]["name"], "Q": roles[role2]["name"],
                     "R": ref_by_dom.get(sk["domain"], ""), **fills[sk["key"]]}
            vis = [p] if sh["private"] and u["link"][0] >= link_vis else "all"
            beat = beats[sh["beat"]]
            beat["facts"].append({"id": f"{eid}.b{sh['beat']}.f{len(beat['facts'])}",
                                  "text": _render(sk["facts"][node], subst), "salience": sh["salience"],
                                  "visibility": vis, "kind": sh["kind"],
                                  "involves": sorted({who[r] for r in sh["involves"] if who[r]})})

        out.append(EventInstance(
            id=eid, latent_type=label, scenario=skin_key, holdout=holdout, start_tick=tick, roles=roles,
            beats=beats, generator=GENERATOR, structure_mode=structure, schema=ST.SCHEMAS[label]["id"],
            skin=skin_key, composed_from=composed, circle=circle, cast_from_home=from_circle, referents=referents,
            narrative=" ".join(f["text"] for b in beats for f in b["facts"])))
        active.append((t[-1], [a for a in (p, second) if a]))
        for r in referents:
            idx = int(r["id"].rsplit(":", 1)[1])
            met[mem_key][r["domain"]].add(idx)
            met["*"][r["domain"]].add(idx)
    return out


def _group(node: str) -> str:
    return "n0" if node == "n0_private" else node


def _pin(skin: dict, group: str) -> str | None:
    return (skin.get("locations") or {}).get(group)


# ----------------------------------------------------------------------------- save / load / check
def script_meta(cfg: dict, profiles: dict, circles: dict, clock) -> dict:
    """What a world script depends on besides its generator: written next to a saved script
    (save(meta=...)) and compared on reuse (`latent_events.script_from`)."""
    return {"generator": GENERATOR, "world_seed": world_seed(cfg), "ticks_per_day": clock.ticks_per_day,
            "days": clock.days, "tick_minutes": clock.tick_minutes,
            "day_start": int(clock.day_start.total_seconds() // 60),      # minutes after midnight
            "day_end": int(clock.day_end.total_seconds() // 60), "agents": sorted(profiles),
            "circles": {c: sorted(m for m in circles[c] if m in profiles) for c in sorted(circles)},
            "routine": cfg.get("routine")}


def meta_path(path) -> Path:
    """world_script.jsonl -> world_script.meta.json"""
    return Path(path).with_suffix(".meta.json")


def save(instances: list[EventInstance], path, meta: dict | None = None) -> str:
    """Write one §1.7 record per line; returns the sha256 of the file (for the manifest). `meta`
    (script_meta()) goes to a sidecar meta_path(path), so a later `script_from` can check the fit."""
    text = "".join(json.dumps(i.ground_truth()) + "\n" for i in instances)
    Path(path).write_text(text)
    if meta is not None:
        meta_path(path).write_text(json.dumps(meta, indent=1, sort_keys=True))
    return hashlib.sha256(text.encode()).hexdigest()


def load(path) -> list[EventInstance]:
    """Read a saved world script verbatim (no check against a run: see load_for_run)."""
    with open(path) as f:
        return [EventInstance.from_dict(json.loads(line)) for line in f if line.strip()]


def saved_meta(path) -> tuple[dict | None, str | None]:
    """(meta, where it came from) for a saved script: its sidecar, else the manifest.json of the run
    directory it sits in (engine runs), else (None, None)."""
    side = meta_path(path)
    if side.exists():
        return json.loads(side.read_text()), str(side)
    man = Path(path).parent / "manifest.json"
    if not man.exists():
        return None, None
    m = json.loads(man.read_text())
    c = m.get("config") or {}
    if "seed" not in c:
        return None, None
    return {"world_seed": world_seed(c), "ticks_per_day": m.get("ticks_per_day"),
            "days": c.get("simulation_days"), "tick_minutes": c.get("tick_minutes"),
            "day_start": _minutes(c["day_start"]) if c.get("day_start") is not None else None,
            "day_end": _minutes(c["day_end"]) if c.get("day_end") is not None else None,
            "agents": sorted(m.get("agents") or {}),
            "circles": {k: sorted(v) for k, v in sorted((m.get("circles") or {}).items())},
            "routine": c.get("routine")}, str(man)


def check_script(instances: list[EventInstance], cfg: dict, profiles: dict, circles: dict, clock,
                 meta: dict | None = None) -> list[str]:
    """Why a (loaded) script does not fit this run; empty when it fits. Content checks always run (every
    agent it names exists, every beat is inside the clock and inside one day, circles and places are
    known); `meta` (what the script was generated for) must match the run where it is recorded."""
    probs = []
    if meta:
        now = script_meta(cfg, profiles, circles, clock)
        for k in ("world_seed", "ticks_per_day", "days", "tick_minutes", "day_start", "day_end", "agents",
                  "circles", "routine"):
            if k in meta and meta[k] is not None and meta[k] != now[k]:
                probs.append(f"{k}: script {meta[k]!r} != run {now[k]!r}")
    tpd, total, known = clock.ticks_per_day, clock.total_ticks, set(profiles)
    for e in instances:
        where = f"instance {e.id}"
        people = [r.get("agent") for r in (e.roles or {}).values()]
        for b in e.beats:
            people += list(b.get("movers") or [])
            for f in b.get("facts") or []:
                people += list(f.get("involves") or [])
                if isinstance(f.get("visibility"), list):
                    people += f["visibility"]
            if not 0 <= int(b["tick"]) < total:
                probs.append(f"{where}: beat at tick {b['tick']} is outside this run's {total} ticks")
            if b.get("location") not in WORLD_GRAPH or b.get("arena") not in ARENAS.get(b.get("location"), []):
                probs.append(f"{where}: unknown place {b.get('location')!r}/{b.get('arena')!r}")
        missing = sorted({a for a in people if a is not None and a not in known})
        if missing:
            probs.append(f"{where}: agents {missing} are not in this run's population")
        ticks = [int(b["tick"]) for b in e.beats] or [e.start_tick]
        if e.start_tick // tpd != max(ticks) // tpd or min(ticks) < e.start_tick:
            probs.append(f"{where}: beats {ticks} (start {e.start_tick}) cross a day end at {tpd} ticks per day")
        if e.circle is not None and e.circle not in circles:
            probs.append(f"{where}: circle {e.circle!r} is not one of this run's circles {sorted(circles)}")
    return probs


def load_for_run(path, cfg: dict, profiles: dict, circles: dict, clock) -> list[EventInstance]:
    """load() a saved script for reuse in this run (`latent_events.script_from`), raising a clear
    ValueError, before any LLM call, if it does not fit (see check_script)."""
    instances = load(path)
    meta, src = saved_meta(path)
    probs = check_script(instances, cfg, profiles, circles, clock, meta)
    if probs:
        more = f" (+{len(probs) - 8} more)" if len(probs) > 8 else ""
        raise ValueError(f"latent_events.script_from={path} does not fit this run"
                         + (f" (recorded in {src})" if src else "") + ":\n  " + "\n  ".join(probs[:8]) + more
                         + "\nGenerate the script with this run's population, clock, circles and world_seed, "
                           "or set them to match the script.")
    return instances


def trace_fields(inst: EventInstance) -> dict:
    """Hidden per-instance fields for the `world_event_start` trace record (manipulation checks)."""
    gt = inst.ground_truth()
    return {k: gt[k] for k in ("latent_type", "structure_mode", "schema", "skin", "composed_from", "holdout",
                               "roles", "circle", "cast_from_home", "referents", "narrative")}
