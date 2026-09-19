"""Roster and turnover (ontology v3 §3). WORLD layer, SIMULATOR-ONLY.

Who is an active co-op member on which day, in which role, is fully deterministic: a Latin square over
`world_seed` (§3.2), no randomness, identical in every arm of a seed. The roster never looks at decisions.

- Persona universe: every persona of the population file (founders + reserves) is loaded once at t0, so
  name resolution, the population lexicon and the encoder's name guards are identical in every arm (§3.1).
  `ctx.agents` keeps every persona; `ctx.active` (a set of ids) is who acts and perceives.
- Departure (§3.3): at the first tick of the wave day the member becomes inactive (`state.active = False`):
  no movement, perception, talk, group seat, hook or frame entry. Their memory stays frozen in the
  previous day's checkpoint (`departed/<id>.json` points to it).
- Arrival (§3.4): a fresh `Agent` from the reserve persona (empty memory, streams `seed_rng(seed, id, name)`),
  symmetric onboarding ties to crewmates / other members, and one no-LLM relationship memory per crewmate
  pair, created under the llm scope `t{T:04d}:00roster` (never "seed").

Public API: `Roster`, `get_roster(sim)`, `apply_day_start(sim, day)`, `active_agents(sim)`,
`persona_universe(path)`, `rotation(...)`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROLES = ("am_crew", "pm_crew", "stores")
ROLE_SHIFT = {"am_crew": "am", "pm_crew": "pm", "stores": "stores"}
ROLE_PLACE = {"am_crew": ("Research Lab", "Makerspace"), "pm_crew": ("Research Lab", "Makerspace"),
              "stores": ("Research Lab", "Stockroom")}
ROLE_ACTIVITY = {"am_crew": "working the morning shift at the laser cutter",
                 "pm_crew": "working the afternoon shift at the laser cutter",
                 "stores": "working the stores desk in the stockroom"}
ROLE_PHRASE = {"am_crew": "morning crew", "pm_crew": "afternoon crew", "stores": "stores team"}
DEFAULT_SHIFTS = {"am": "08:45-13:15", "pm": "13:00-17:45", "stores": ["08:45-12:00", "13:00-17:45"]}
ROTATE_BASE = 11                     # seed 11 is row 0 of the Latin square (§3.2)
NEWCOMER_DAYS = 2                    # newcomers lead the operator round-robin on their first two days (§1.6)
FAREWELL_BEFORE_END_MIN = 30         # the farewell fact falls inside the last hour of the last shift (§3.3)
ORIENTATION_TEXT = ("the laser runs a morning and an afternoon shift, campus groups send in jobs, the binder hangs "
                    "next to the laser, and the stockroom keeps the sheets")
PRONOUNS = {"she": ("she", "her", "is"), "he": ("he", "his", "is"), "they": ("they", "their", "are")}


# ---------------------------------------------------------------------------------------------- helpers
def _mins(hhmm: str) -> int:
    h, m = map(int, str(hhmm).split(":"))
    return h * 60 + m


def _k(clock, hhmm: str) -> int:
    """Tick within the day of a wall-clock time (same rounding as scheduler.plan_day)."""
    start = int(clock.day_start.total_seconds() // 60)
    return max(0, round((_mins(hhmm) - start) / clock.tick_minutes))


def shift_windows(cfg: dict, role: str) -> list[tuple[str, str]]:
    """[(start "HH:MM", end "HH:MM"), ...] of a role's shift, from `roster.shifts` (contract defaults)."""
    shifts = {**DEFAULT_SHIFTS, **((cfg.get("roster") or {}).get("shifts") or {})}
    spec = shifts[ROLE_SHIFT[role]]
    spec = [spec] if isinstance(spec, str) else list(spec)
    return [tuple(s.split("-")) for s in spec]


def shift_ticks(cfg: dict, role: str, clock) -> list[tuple[int, int]]:
    """[(k_start, k_end), ...] within the day (k_end exclusive)."""
    return [(_k(clock, a), _k(clock, b)) for a, b in shift_windows(cfg, role)]


def _pron(profile) -> tuple[str, str, str]:
    return PRONOUNS.get(str((profile.demographics or {}).get("pronoun", "they")), PRONOUNS["they"])


def persona_universe(path) -> dict[str, str]:
    """{id: full name} of every persona in a population file, founders and reserves (§3.1)."""
    from backend.agents.profile import load_population
    profiles, _ = load_population(path, None, include_reserves=True)
    return {aid: p.name for aid, p in profiles.items()}


def rotation(world_seed: int, founders: dict[str, list[str]], reserves: list[str], waves: list[dict],
             rotate="by_world_seed") -> list[dict]:
    """The deterministic change schedule (§3.2): a Latin square over the seed, no randomness.

    Role ri's departure queue is its founders rotated by (s - 11 + ri) mod n_role; reserves are consumed in
    order from r = (s - 11) mod n_reserves. Within a wave roles go am_crew, pm_crew, stores. For the study's two
    waves this reproduces the §3.2 table exactly. Returns [{day, kind: depart|arrive, agent, role[, replaces]}]."""
    if rotate in (None, "none", False):
        off = 0
    elif rotate == "by_world_seed":
        off = int(world_seed) - ROTATE_BASE
    else:
        off = int(rotate)
    queues = {}
    for ri, role in enumerate(ROLES):
        f = list(founders.get(role, []))
        if f:
            o = (off + ri) % len(f)
            queues[role] = f[o:] + f[:o]
        else:
            queues[role] = []
    nres = len(reserves)
    r = off % nres if nres else 0
    out = []
    for w in sorted(waves, key=lambda w: int(w["day"])):
        day = int(w["day"])
        if day < 2:
            raise ValueError(f"turnover wave on day {day}: departures need a last shift on an earlier day")
        dep = w.get("depart") or {}
        unknown = set(dep) - set(ROLES)
        if unknown:
            raise ValueError(f"turnover wave: unknown roles {sorted(unknown)}")
        for role in ROLES:
            for _ in range(int(dep.get(role, 0))):
                if not queues[role]:
                    raise ValueError(f"turnover: no founder left to depart from {role} on day {day}")
                if len(out) // 2 >= nres:
                    raise ValueError("turnover: reserves exhausted")
                gone = queues[role].pop(0)
                new = reserves[r % nres]
                r += 1
                out.append({"day": day, "kind": "depart", "agent": gone, "role": role})
                out.append({"day": day, "kind": "arrive", "agent": new, "role": role, "replaces": gone})
    return out


# ---------------------------------------------------------------------------------------------- roster
class Roster:
    """Deterministic roster of one run. Pure schedule (no agents needed) + `apply_day_start` mutation."""

    def __init__(self, cfg: dict, profiles: dict | None = None, world_seed: int | None = None, clock=None,
                 sim=None, agents: dict | None = None):
        """`profiles` {id: AgentProfile} of the whole persona universe; or taken from `agents`/`sim.ctx.agents`
        (reserves are loaded from `cfg.population` when the engine did not build them). `world_seed` defaults to
        rngs.world_seed(cfg); `clock` to sim.clock or Clock(cfg)."""
        from backend.simulation.rngs import world_seed as _ws
        self.cfg = cfg
        self.rc = cfg.get("roster") or {}
        self.tc = cfg.get("turnover") or {}
        self.enabled = bool(self.rc.get("enabled"))
        if profiles is None:
            agents = agents if agents is not None else (sim.ctx.agents if sim is not None else {})
            profiles = _with_reserves(cfg, {aid: a.profile for aid, a in agents.items()})
        if clock is None:
            clock = getattr(sim, "clock", None)
            if clock is None and "day_start" in cfg:
                from backend.simulation.world import Clock
                clock = Clock(cfg)
        self.clock = clock
        self.world_seed = int(_ws(cfg) if world_seed is None else world_seed)
        self.profiles = profiles
        self.current_day: int | None = None
        self.founders = {r: [a for a, p in profiles.items() if getattr(p, "role", None) == r and
                             not getattr(p, "reserve", False)] for r in ROLES}
        self.reserves = [a for a, p in profiles.items() if getattr(p, "reserve", False)]
        if self.tc.get("reassign"):
            raise NotImplementedError("turnover.reassign (RQ3 extension) is not implemented in the v3 MVP")
        waves = (self.tc.get("waves") or []) if (self.enabled and self.tc.get("enabled")) else []
        self.schedule = rotation(self.world_seed, self.founders, self.reserves, waves,
                                 self.tc.get("rotate", "by_world_seed"))
        self.arrival_day = {a: 1 for r in ROLES for a in self.founders[r]}
        self.departure_day: dict[str, int] = {}
        self.base_role = {a: r for r in ROLES for a in self.founders[r]}
        self.cohort = {a: "founder" for a in self.arrival_day}
        wave_days = sorted({c["day"] for c in self.schedule})
        for c in self.schedule:
            if c["kind"] == "arrive":
                self.arrival_day[c["agent"]] = c["day"]
                self.base_role[c["agent"]] = c["role"]
                self.cohort[c["agent"]] = f"W{wave_days.index(c['day']) + 1}"
            else:
                self.departure_day[c["agent"]] = c["day"]
        self.applied: set[int] = set()
        self._initialised = False

    # ---- pure schedule queries (usable by the world script before the run) ---------------------
    def is_active(self, aid: str, day: int) -> bool:
        a = self.arrival_day.get(aid)
        return a is not None and a <= day < self.departure_day.get(aid, 10 ** 9)

    def role_on(self, aid: str, day: int) -> str | None:
        return self.base_role.get(aid) if self.is_active(aid, day) else None

    def role_of(self, aid: str, day: int | None = None) -> str | None:
        """Role on `day` (default: the day last applied by apply_day_start; before that, the base role)."""
        day = self.current_day if day is None else day
        return self.base_role.get(aid) if day is None else self.role_on(aid, day)

    def newcomers(self, day: int) -> list[str]:
        return [a for a in self.active_on(day) if self.is_newcomer(a, day)]

    def __call__(self, day: int) -> dict:
        """workshop.generate's roster callable: {"am", "pm", "stores", "newcomers"} for a day."""
        return {"am": self.members("am_crew", day), "pm": self.members("pm_crew", day),
                "stores": self.members("stores", day), "newcomers": self.newcomers(day)}

    def active_on(self, day: int) -> list[str]:
        """Active member ids on a day, sorted."""
        return sorted(a for a in self.arrival_day if self.is_active(a, day))

    def members(self, role: str, day: int) -> list[str]:
        """Active members of a role on a day: founders first (file order), then newcomers by arrival."""
        order = self.founders[role] + [c["agent"] for c in self.schedule if c["kind"] == "arrive" and c["role"] == role]
        return [a for a in order if self.is_active(a, day)]

    def is_newcomer(self, aid: str, day: int) -> bool:
        a = self.arrival_day.get(aid, 1)
        return self.cohort.get(aid) != "founder" and a <= day < a + NEWCOMER_DAYS

    def rota_order(self, role: str, day: int) -> list[str]:
        """Operator round-robin base order (§1.6): newcomers on their first two days first, then the rest.
        The world script applies its per-day `operator` offset on top of this order."""
        m = self.members(role, day)
        return [a for a in m if self.is_newcomer(a, day)] + [a for a in m if not self.is_newcomer(a, day)]

    def changes_on(self, day: int) -> list[dict]:
        return [dict(c) for c in self.schedule if c["day"] == day]

    def _T(self, day: int) -> int:
        return self.clock.ticks_per_day * (day - 1)

    def farewells(self) -> list[dict]:
        """Farewell facts (§3.3): released in the departing member's shift arena during the last hour of the
        last shift (the day before the wave); `end_tick` is when that shift ends (farewell RecordWrite offer)."""
        if not self.tc.get("farewell", True):
            return []
        out = []
        for c in self.schedule:
            if c["kind"] != "depart":
                continue
            aid, role, last = c["agent"], c["role"], c["day"] - 1
            prof = self.profiles[aid]
            subj, pos, be = _pron(prof)
            s_end = shift_windows(self.cfg, role)[-1][1]
            k_end = _k(self.clock, s_end)
            k_rel = max(0, k_end - max(1, round(FAREWELL_BEFORE_END_MIN / self.clock.tick_minutes)))
            loc, arena = ROLE_PLACE[role]
            text = (f"{prof.first_name} mentioned that today was {pos} last shift at the co-op; {subj} {be} "
                    f"moving to another lab next week.")
            out.append({"kind": "farewell", "id": f"farewell.d{last}.{aid}", "day": last,
                        "tick": self._T(last) + k_rel, "end_tick": self._T(last) + k_end,
                        "offer_tick": self._T(last) + k_end, "agent": aid, "role": role, "location": loc,
                        "arena": arena, "involves": [aid], "movers": [aid], "salience": 0.5,
                        "facts": [{"text": text, "salience": 0.5, "involves": [aid], "visibility": "arena"}]})
        return out

    def orientations(self) -> list[dict]:
        """Orientation facts (§3.4) at the start of the newcomer's role shift on the arrival day (08:45 for the
        morning crew and stores; the afternoon crew's shift starts at 13:00). The guide is the first incumbent
        of the role (founders first) who did not arrive that day. Free of fault or fix content."""
        out = []
        for c in self.schedule:
            if c["kind"] != "arrive":
                continue
            aid, role, day = c["agent"], c["role"], c["day"]
            incumbents = [m for m in self.members(role, day) if self.arrival_day.get(m) < day]
            guide = incumbents[0] if incumbents else None
            new = self.profiles[aid].first_name
            loc, arena = ROLE_PLACE[role]
            facts = []
            if guide:
                facts.append({"text": f"{self.profiles[guide].first_name} showed {new} around the co-op: {ORIENTATION_TEXT}.",
                              "salience": 0.5, "involves": [guide, aid], "visibility": "arena"})
            else:
                facts.append({"text": f"{new} was shown around the co-op: {ORIENTATION_TEXT}.",
                              "salience": 0.5, "involves": [aid], "visibility": "arena"})
            facts.append({"text": f"{new} joined the co-op's {ROLE_PHRASE[role]} today.", "salience": 0.4,
                          "involves": [aid], "visibility": "arena"})
            k0 = _k(self.clock, shift_windows(self.cfg, role)[0][0])
            inv = [guide, aid] if guide else [aid]
            out.append({"kind": "orientation", "id": f"orientation.d{day}.{aid}", "day": day,
                        "tick": self._T(day) + k0, "agent": aid, "role": role, "guide": guide, "location": loc,
                        "arena": arena, "involves": inv, "movers": inv, "salience": 0.5, "facts": facts})
        return out

    def script_records(self) -> list[dict]:
        """World-script records (§1.8 kinds `roster`, `farewell`, `orientation`), sorted by tick. The world
        script / CoopWorld turns the `facts` into arena-visibility world facts at `tick`."""
        if not self.enabled:
            return []
        recs = [{"kind": "roster", "id": f"roster.d{c['day']}.{c['kind']}.{c['agent']}", "tick": self._T(c["day"]),
                 **{("change" if k == "kind" else k): v for k, v in c.items()}} for c in self.schedule]
        recs += self.farewells() + self.orientations()
        return sorted(recs, key=lambda r: (r["tick"], r["kind"], r.get("agent") or ""))

    def agent_info(self, aid: str, day: int | None = None) -> dict:
        """For checkpoints' agent_state.json (§5.1): role, cohort, arrival day, departure day, active."""
        day = self.current_day if day is None else day
        return {"role": self.base_role.get(aid), "cohort": self.cohort.get(aid, "reserve"),
                "arrival_day": self.arrival_day.get(aid), "departure_day": self.departure_day.get(aid),
                "active": self.is_active(aid, day or 1)}

    def active_agents(self, day: int | None = None) -> list[str]:
        """Active ids on `day` (default: the day last applied). Checkpoint duck-typing entry point."""
        return self.active_on(day or self.current_day or 1)

    def state_dict(self, day: int | None = None) -> dict:
        """roster.json for checkpoints (deterministic, sorted); `day` defaults to the day last applied."""
        day = day or self.current_day or 1
        return {"day": day, "active": self.active_on(day),
                "roles": {r: self.members(r, day) for r in ROLES},
                "agents": {a: self.agent_info(a, day) for a in sorted(self.profiles)},
                "schedule": self.schedule}

    checkpoint_state = state_dict

    # ---- mutation at the day start (§4.7 phase 0a) --------------------------------------------
    def init_active(self, sim) -> None:
        """Mark founders active and reserves inactive (idempotent); sets `ctx.active`."""
        ctx = sim.ctx
        if self._initialised and getattr(ctx, "active", None) is not None:
            return
        act = set(self.active_on(1))
        for aid, a in ctx.agents.items():
            a.state.active = aid in act
        ctx.active = act
        self._initialised = True

    def apply_day_start(self, sim, day: int) -> list[dict]:
        """Apply the day's departures and arrivals (departures first). Idempotent per day. Returns the
        applied changes. Must run BEFORE the day's plan_day loop (so newcomers get a shift plan and departed
        members none)."""
        if not self.enabled:
            return []
        self.init_active(sim)
        if day in self.applied:
            return []
        self.applied.add(day)
        self.current_day = day
        changes = self.changes_on(day)
        if not changes:
            return []
        from backend.llm.client import llm_scope
        with llm_scope(f"t{self._T(day):04d}:00roster"):         # trace ids and any embed calls are tick-scoped
            for c in [c for c in changes if c["kind"] == "depart"]:
                self._depart(sim, day, c)
            for c in [c for c in changes if c["kind"] == "arrive"]:
                self._arrive(sim, day, c)
        return changes

    def _log(self, sim, type_, **kw):
        tr = getattr(sim, "tracer", None)
        if tr is not None:
            tr.log(type_, **kw)

    def _depart(self, sim, day, c):
        ctx, aid = sim.ctx, c["agent"]
        a = ctx.agents[aid]
        a.state.active = False
        a.state.in_conversation = None
        a.state.forced_by = None
        a.state.path = []
        ctx.active.discard(aid)
        for attr in ("pending_obs", "overrides", "follow"):
            d = getattr(sim, attr, None)
            if isinstance(d, dict):
                d.pop(aid, None)
        rd = getattr(sim, "run_dir", None)
        if rd is not None:
            p = Path(rd) / "departed"
            p.mkdir(parents=True, exist_ok=True)
            (p / f"{aid}.json").write_text(json.dumps(
                {"agent": aid, "role": c["role"], "departed_day": day, "last_day": day - 1,
                 "checkpoint": f"checkpoints/C{day - 1}"}, sort_keys=True, indent=1))
        self._log(sim, "roster_change", day=day, agent=aid, kind="depart", role=c["role"])

    def _arrive(self, sim, day, c):
        from backend.agents.agent import Agent
        from backend.agents.profile import Relationship
        ctx, aid, role = sim.ctx, c["agent"], c["role"]
        prof = ctx.agents[aid].profile if aid in ctx.agents else self.profiles[aid]
        prof.role = role
        old = ctx.agents.get(aid)
        a = Agent(prof, sim.cfg, getattr(ctx, "mods", None), sim.cfg["seed"])     # fresh: empty memory
        a.ctx = ctx
        a.state.active = True
        ctx.agents[aid] = a
        ctx.active.add(aid)
        if isinstance(getattr(sim, "agents", None), dict) and sim.agents is not ctx.agents:
            sim.agents[aid] = a
        fam_c = float(self.tc.get("crew_familiarity", 0.35))
        fam_m = float(self.tc.get("member_familiarity", 0.2))
        crew = [m for m in self.members(role, day) if m != aid]
        ties = []
        for m in sorted(ctx.active - {aid}):
            same = m in crew
            rt, fam = ("co-op crewmate", fam_c) if same else ("co-op member", fam_m)
            other = ctx.agents[m].profile
            for x, y in ((prof, other), (other, prof)):
                cur = x.relationships.get(y.id)
                if cur is None or cur.relation_type == "stranger" or cur.familiarity < fam:
                    x.relationships[y.id] = Relationship(rt, fam, 0.5)
            ties.append({"with": m, "relation_type": rt, "familiarity": fam, "affinity": 0.5})
        self._log(sim, "roster_change", day=day, agent=aid, kind="arrive", role=role, replaces=c.get("replaces"))
        self._log(sim, "onboarding", agent=aid, day=day, ties=ties)
        self._seed_pairs(sim, day, aid, role, crew)

    def _seed_pairs(self, sim, day, aid, role, crew):
        """One no-LLM relationship memory per crewmate pair, in both memories (scope t{T}:00roster)."""
        embed, meta, clock = getattr(sim, "embed", None), getattr(sim, "meta", None), self.clock
        if embed is None or clock is None:
            return
        from backend.memory.store import MemoryMeta
        T = self._T(day)
        t0 = clock.time_of(T)
        ctx = sim.ctx
        new = ctx.agents[aid]
        for m in crew:                                   # runs inside llm_scope t{T}:00roster (apply_day_start)
            mate = ctx.agents[m]
            text = (f"{new.profile.first_name} and {mate.profile.first_name} met at co-op orientation; they are "
                    f"on the {ROLE_PHRASE[role]} together.")
            for x, y in ((new, mate), (mate, new)):
                x.set_time(t0)
                node = x.a_mem.add("thought", t0, x.name, "knows", y.name, text,
                                   {x.name.lower(), y.name.lower()}, 5, embed(text), [])
                if meta is not None:
                    meta.set(node.node_id, MemoryMeta(agent_id=x.id, source_type="seed", salience=0.3))
                self._log(sim, "memory_encoded", agent=x.id, node_id=node.node_id, kind="thought", text=text,
                          importance=5, salience=0.3, source_type="seed", observation_id=None,
                          observation=None, encoding_ops=[], prompt=None, originating_event_ids=[],
                          speakers=[], utterance_ids=[])


# ---------------------------------------------------------------------------------------------- facade
def _with_reserves(cfg: dict, profiles: dict) -> dict:
    """Add the population file's reserves when the engine did not build them (so the roster can arrive them)."""
    if any(getattr(p, "reserve", False) for p in profiles.values()) or not cfg.get("population"):
        return profiles
    from backend.agents.profile import load_population
    allp, _ = load_population(cfg["population"], None, include_reserves=True)
    return {**profiles, **{a: p for a, p in allp.items() if getattr(p, "reserve", False) and a not in profiles}}


def get_roster(sim) -> Roster:
    """The run's Roster (cached on `sim.roster`), built from `ctx.agents` profiles (all personas)."""
    r = getattr(sim, "roster", None)
    if r is None:
        r = Roster(sim.cfg, sim=sim)
        sim.roster = r
    return r


def apply_day_start(sim, day: int) -> list[dict]:
    """§4.7 phase 0a (roster part): departures, arrivals, ties, seed memories. No-op when roster is off."""
    if not (sim.cfg.get("roster") or {}).get("enabled"):
        return []
    return get_roster(sim).apply_day_start(sim, day)


def active_agents(sim) -> dict:
    """{id: Agent} of the active members, sorted by id. Every persona when the roster is off (v2 behaviour)
    or before the roster initialised. Use it everywhere agents act or perceive."""
    agents = sim.ctx.agents
    act = getattr(sim.ctx, "active", None)
    if not (sim.cfg.get("roster") or {}).get("enabled") or act is None:
        return {aid: agents[aid] for aid in sorted(agents) if getattr(agents[aid].state, "active", True)}
    return {aid: agents[aid] for aid in sorted(act)}
