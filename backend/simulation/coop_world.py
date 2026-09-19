"""CoopWorld: the v3 co-op facade the engine calls (docs/ONTOLOGY_V3.md §4.7, §8.1 step 5). WORLD layer.

The engine owns the tick loop; CoopWorld owns everything the co-op adds to it. Engine hooks, in tick order:

    init    sim.coop = CoopWorld(sim)          after agents/modules exist (sets sim.roster, coop.binder)
    0a      coop.day_start(tick)               first tick of a day, BEFORE the day-plan loop: roster
                                               (depart / arrive), hidden `regime_active`, binder transitions,
                                               the day's handover / meeting schedule
    1       beats += coop.world(tick)          (ev, beat) pairs in v2 beat shape: job start / outcome beats,
                                               cue, orientation, farewell, transition fact, tally, and
                                               fact-less mover beats (handover 13:00-13:30, meeting 18:30-19:15)
    3+      obs = coop.perception_hook(tick, obs)
                                               buffers job facts (workshop.encode: per_job) and adds the job
                                               episodes and binder-read observations queued for this tick
    4       engine _cognition                  for obs with `coop_episode=True` render with
                                               work.render_pending (only unrendered facts) instead of
                                               viewpoint.render; skip the react decision if `no_react=True`
    4b      coop.after_cognition(tick)         job decisions, parallel per operator (work.decide_job)
    4c      req = coop.apply(tick, results)    outcomes (next tick's beats), job ends -> episode queue,
                                               binder writes (sorted), says-aloud remarks merged into
                                               `results` as REACT decisions; returns ConvRequests
    5/6     forced_talks = req.dyads + talks   clarify / handover as v2 forced talks (trigger["topic"]);
            groups += req.groups               meeting as v2 run_group_conversation(topic="meeting")
    8b      coop.day_end(tick)                 last tick of a day: day digest; checkpoint (checkpoints.*)

Nothing here reads observer output. Hidden values (regime, mapping, class, cause, uniforms) are only written to
hidden trace records (`job_truth`, `regime_active`) and to `checkpoint_state()`; never to a prompt or a fact.
Co-op modules are imported lazily, so v2 runs never load them.
"""
from __future__ import annotations

import importlib
import json
from collections import defaultdict
from dataclasses import dataclass, field

from backend.agents.perception import AgentObservation

MAKERSPACE = ("Research Lab", "Makerspace")
COOP_KINDS = ("job", "tally", "cue", "farewell", "orientation", "roster")


def _m(name: str):
    return importlib.import_module(name)


def _get(rec, key, default=None):
    return rec.get(key, default) if isinstance(rec, dict) else getattr(rec, key, default)


def _tick_of(rec) -> int:
    t = _get(rec, "start_tick")
    return int(t if t is not None else _get(rec, "tick"))


@dataclass
class JobState:
    job: object                      # workshop.JobInstance
    regime: str                      # HIDDEN
    decide_at: int | None = None
    outcome_at: int | None = None
    pending: dict | None = None      # {attempt, action, outcome} released as a beat at outcome_at
    attempts: list = field(default_factory=list)   # [{attempt, action, outcome, tick}]
    questions: int = 0
    reasons: list = field(default_factory=list)
    obs: dict = field(default_factory=lambda: defaultdict(list))   # agent -> buffered job observations
    beats: int = 1
    done: bool = False
    ended: bool = False
    delivered: bool = False

    @property
    def id(self) -> str:
        return self.job.id

    @property
    def operator(self) -> str:
        return self.job.operator


@dataclass
class ConvRequests:
    """Conversations for phase 6. dyads: (init, target, opening, trigger) in the engine's forced-talk format
    (trigger["topic"] in clarify | handover); groups: (sorted participants, topic, max_utterances)."""
    dyads: list = field(default_factory=list)
    groups: list = field(default_factory=list)


class _Ev:
    """Event handle for the engine's beat loop: `id` keys perception streams and observation event_ids."""

    def __init__(self, eid: str, kind: str):
        self.id, self.kind, self.latent_type = eid, kind, "coop"


class CoopWorld:
    def __init__(self, sim, script: list | None = None):
        self.sim, self.cfg, self.clock = sim, sim.cfg, sim.clock
        self.tpd = sim.clock.ticks_per_day
        cfg = self.cfg
        self.workshop_on = bool((cfg.get("workshop") or {}).get("enabled"))
        self.records_on = bool((cfg.get("records") or {}).get("enabled"))
        self.roster_on = bool((cfg.get("roster") or {}).get("enabled"))
        self.per_job = (cfg.get("workshop") or {}).get("encode", "per_job") == "per_job"
        RG = _m("backend.simulation.regimes")
        self.mapping = RG.mapping(cfg)                                    # HIDDEN
        self.regime: str | None = None                                    # HIDDEN
        self.roster = _m("backend.simulation.roster").get_roster(sim) if self.roster_on else None
        self.binder = _m("backend.simulation.records").Binder.for_sim(cfg, sim.clock) if self.records_on else None
        if self.roster is not None:
            self.roster.init_active(sim)
        _m("backend.agents.coop_talk").register_group_topics()
        self.om = _m("backend.simulation.workshop").OutcomeModel(cfg) if self.workshop_on else None
        self.jobs: dict[str, JobState] = {}
        self.by_tick: dict[int, list] = defaultdict(list)
        self.pending: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self.offers: dict[int, list] = defaultdict(list)                 # tick -> [(agent, offer, remembered, job)]
        self.movers: dict[int, dict[str, tuple]] = defaultdict(dict)
        self.convs: dict[int, ConvRequests] = defaultdict(ConvRequests)
        self.decisions: dict[str, dict] = {}
        self.outcomes: list[dict] = []                                    # HIDDEN (checkpoint world_state)
        self.counts: dict[str, int] = defaultdict(int)
        self.script = self._script(script)
        for r in self.script:
            self.by_tick[_tick_of(r)].append(r)

    # ------------------------------------------------------------------------------------ script
    def _crews(self, day: int) -> dict:
        R = self.roster
        return {"am": R.rota_order("am_crew", day), "pm": R.rota_order("pm_crew", day),
                "stores": R.members("stores", day), "newcomers": [a for a in R.active_on(day) if R.is_newcomer(a, day)]}

    def _script(self, script) -> list:
        """Co-op world-script records: given, or sim.coop_script, or the co-op records inside sim.world, or
        generated here (workshop.generate + roster.script_records). Deterministic from config and world_seed."""
        recs = script if script is not None else getattr(self.sim, "coop_script", None)
        if recs is None:
            recs = [r for r in (getattr(self.sim, "world", None) or []) if _get(r, "kind") in COOP_KINDS]
        recs = list(recs or [])
        kinds = {_get(r, "kind") for r in recs}
        if self.workshop_on and "job" not in kinds:
            W = _m("backend.simulation.workshop")
            profiles = {aid: a.profile for aid, a in self.sim.agents.items()}
            recs += W.generate(self.cfg, profiles, self.clock, roster=self._crews if self.roster else None)
        if self.roster is not None and not kinds & {"farewell", "orientation"}:
            recs += [r for r in self.roster.script_records() if r["kind"] in ("farewell", "orientation")]
        return sorted(recs, key=lambda r: (_tick_of(r), str(_get(r, "kind")), str(_get(r, "id", _get(r, "agent")))))

    def script_jsonl(self) -> str:
        """The co-op records as JSON lines (sorted keys), for the integrator to append to world_script.jsonl."""
        return "".join(json.dumps(r.to_dict() if hasattr(r, "to_dict") else dict(r), sort_keys=True, default=str)
                       + "\n" for r in self.script)

    # ----------------------------------------------------------------------------------- helpers
    def active_ids(self) -> list[str]:
        if self.roster is not None:
            return sorted(_m("backend.simulation.roster").active_agents(self.sim))
        return sorted(a for a, x in self.sim.agents.items() if getattr(x.state, "active", True))

    def _first(self, aid: str) -> str:
        return self.sim.agents[aid].profile.first_name

    def _tick(self, day: int, hhmm: str) -> int:
        return _m("backend.simulation.regimes").tick_of(day, hhmm, self.clock)

    def _stores_on(self, day: int) -> str | None:
        """The rostered stores member at the tally: the stores team in turn, one per day."""
        st = self.roster.members("stores", day) if self.roster else \
            [a for a in ("priya", "sofia") if a in self.active_ids()]
        return st[(day - 1) % len(st)] if st else None

    def _pm_crew(self, day: int) -> list[str]:
        if self.roster:
            return self.roster.members("pm_crew", day)
        return [a for a in ("ethan", "leo", "jordan") if a in self.active_ids()]

    # ----------------------------------------------------------------------------- phase 0a
    def day_start(self, tick: int):
        day = self.clock.day_of(tick)
        if self.roster is not None:
            _m("backend.simulation.roster").apply_day_start(self.sim, day)
        if self.workshop_on:
            RG = _m("backend.simulation.regimes")
            self.regime = RG.active(day, self.cfg)
            self.sim.tracer.log("regime_active", **RG.Regimes(self.cfg).trace(day))       # HIDDEN
        if self.binder is not None:
            R = _m("backend.simulation.records")
            for f in R.apply_transitions(self.binder, self.cfg, self.sim.tracer, day, tick):
                self.counts["transitions"] += 1
                t = self._tick(day, f["time"])
                self.by_tick[t].append(
                    {"kind": "transition", "id": f"transition{day:02d}", "tick": t, "day": day,
                     "location": MAKERSPACE[0], "arena": MAKERSPACE[1],
                     "facts": [{"text": f["text"], "salience": f["salience"], "involves": [], "visibility": "arena"}]})
        self._schedule_talk(day)

    def _schedule_talk(self, day: int):
        CT = _m("backend.agents.coop_talk")
        act = set(self.active_ids())
        if self.workshop_on and CT.enabled(self.cfg, "handover"):
            jobs = [r.to_dict() for r in self.script if _get(r, "kind") == "job" and _get(r, "day") == day]
            pair = CT.handover_pair(jobs)
            if pair and set(pair) <= act:
                a, b = pair
                for t in range(self._tick(day, "13:00"), self._tick(day, "13:30") + 1):
                    self.movers[t].update({a: MAKERSPACE, b: MAKERSPACE})
                t = self._tick(day, str(CT.comm_cfg(self.cfg, "handover")["time"]))
                self.convs[t].dyads.append(CT.handover_talk(self.sim.agents[a], self.sim.agents[b]))
                self.sim.tracer.log("handover", **CT.handover_record(day, a, b))
                self.counts["handovers"] += 1
        if CT.is_meeting_day(self.cfg, day):
            parts = CT.meeting_participants(self.cfg, act)
            mc = CT.comm_cfg(self.cfg, "meeting")
            t0 = self._tick(day, str(mc["time"]))
            if len(parts) >= 2:
                for t in range(t0, min(self._tick(day, "19:15"), day * self.tpd - 1) + 1):
                    self.movers[t].update({x: MAKERSPACE for x in parts})
                self.convs[t0].groups.append((sorted(parts), "meeting", int(mc["max_utterances"])))
                self.sim.tracer.log("meeting", **CT.meeting_record(day, parts))
                self.counts["meetings"] += 1

    # ------------------------------------------------------------------------------- phase 1
    def world(self, tick: int) -> list[tuple]:
        W = _m("backend.simulation.workshop") if self.workshop_on else None
        out = []
        for rec in self.by_tick.get(tick, []):
            kind = _get(rec, "kind")
            if kind == "job" and W:
                out.append(self._job_start(rec, tick))
            elif kind == "tally" and W:
                out.append(self._tally(rec, tick))
            elif kind == "cue" and W:
                out.append((_Ev(rec.id, "cue"), W.cue_beat(rec, self.cfg)))
                self.sim.tracer.log("cue_event", **W.trace_cue(rec))
            elif kind in ("farewell", "orientation", "transition"):
                out.append(self._fact_beat(rec, tick))
                if kind == "farewell" and self.binder is not None and \
                        _m("backend.simulation.records").offer_enabled(self.cfg, "farewell"):
                    self.offers[int(rec.get("end_tick", tick))].append((rec["agent"], "farewell", None, None))
        for js in sorted(self.jobs.values(), key=lambda j: j.id):
            if js.outcome_at == tick and js.pending is not None:
                out.append(self._outcome(js, tick))
        by_place = defaultdict(list)
        for aid, place in sorted(self.movers.pop(tick, {}).items()):
            by_place[place].append(aid)
        for (loc, arena), ids in sorted(by_place.items()):
            out.append((_Ev(f"coop.movers.{arena}", "movers"),
                        {"idx": 0, "tick": tick, "location": loc, "arena": arena, "facts": [], "movers": ids}))
        return out

    def _job_start(self, job, tick):
        W = _m("backend.simulation.workshop")
        regime = self.regime or _m("backend.simulation.regimes").active(self.clock.day_of(tick), self.cfg)
        js = JobState(job=job, regime=regime)
        self.jobs[job.id] = js
        beat = W.start_beat(job, regime, self.cfg, self._first(job.operator))
        self.sim.tracer.log("job_start", **W.trace_job_start(job))
        self.sim.tracer.log("job_truth", **W.trace_job_truth(job, regime, self.cfg))            # HIDDEN
        self.counts["jobs"] += 1
        if W.needs_decision(job, regime):
            self.counts["faulted"] += 1
            js.decide_at = tick
        else:
            js.done, js.delivered = True, True                  # K0: clean cut, ends in this tick's 4c
        return _Ev(job.id, "job"), beat

    def _outcome(self, js: JobState, tick):
        W = _m("backend.simulation.workshop")
        p = js.pending
        beat = W.outcome_beat(js.job, js.beats, tick, p["action"], p["outcome"], js.regime, self.cfg,
                              self._first(js.operator))
        js.beats += 1
        self.sim.tracer.log("job_attempt", **W.trace_job_attempt(js.job, p["attempt"], p["action"], p["outcome"], tick))
        js.pending, js.outcome_at = None, None
        if p["outcome"] in ("success", "defer") or len(js.attempts) >= int(self.om.max_attempts):
            js.done, js.delivered = True, p["outcome"] == "success"
        else:
            js.decide_at = tick                                 # the next attempt is decided in this tick's 4b
        return _Ev(js.id, "job"), beat

    def _tally(self, rec, tick):
        W = _m("backend.simulation.workshop")
        day = rec.day
        results = [(js.job, js.delivered) for js in sorted(self.jobs.values(), key=lambda j: j.job.slot)
                   if js.job.day == day]
        stores = self._stores_on(day)
        self.sim.tracer.log("tally", **W.trace_tally(rec, results), stores=stores)
        if stores and self.binder is not None:
            self._read(stores, tick, day, "tally")
            if _m("backend.simulation.records").offer_enabled(self.cfg, "tally"):
                self.offers[tick].append((stores, "tally", [W.tally_text(results)], None))
        return _Ev(rec.id, "tally"), W.tally_beat(rec, results, self.cfg, movers=[stores] if stores else [])

    def _fact_beat(self, rec: dict, tick):
        eid = rec.get("id") or f"{rec['kind']}.{rec.get('agent', '')}.d{int(rec.get('day', 0)):02d}"
        facts = [{"id": f"{eid}.b0.f{i}", "text": f["text"], "salience": float(f.get("salience", 0.5)),
                  "visibility": f.get("visibility", "arena"), "kind": rec["kind"],
                  "involves": [a for a in f.get("involves", []) if a in self.sim.agents]}
                 for i, f in enumerate(rec.get("facts") or [])]
        return _Ev(eid, rec["kind"]), {"idx": 0, "tick": tick, "location": rec.get("location", MAKERSPACE[0]),
                                       "arena": rec.get("arena", MAKERSPACE[1]), "facts": facts,
                                       "movers": [a for a in rec.get("movers", []) if a in self.sim.agents]}

    # ----------------------------------------------------------------------------- phase 3+
    def perception_hook(self, tick: int, obs_by_agent: dict) -> dict:
        out: dict[str, list] = {}
        for aid in sorted(obs_by_agent):
            for o in obs_by_agent[aid]:
                js = self.jobs.get(o.event_ids[0]) if o.event_ids else None
                if js is not None:
                    js.obs[aid].append(o)
                    if self.per_job:
                        continue                                  # encoded as one episode at job end
                out.setdefault(aid, []).append(o)
        act = set(self.active_ids())
        for aid, lst in sorted(self.pending.pop(tick, {}).items()):
            if aid in act:
                out.setdefault(aid, []).extend(lst)
        return out

    # ------------------------------------------------------------------------------ binder
    def _read(self, aid: str, tick: int, day: int, context: str, query: str | None = None):
        R = _m("backend.simulation.records")
        a = self.sim.agents[aid]
        v, obs = R.read(self.binder, self.cfg, self.sim.tracer, aid, tick, day, context, location=a.state.location,
                        arena=a.state.arena, query=query, embed=self.sim.embed)
        self.counts["reads"] += 1
        if obs is not None:
            obs.no_react = True
            self.pending[tick + 1][aid].append(obs)             # encoded next tick (§2.3)
        return v

    # ------------------------------------------------------------------------------ phase 4b
    def after_cognition(self, tick: int, obs_by_agent: dict | None = None) -> dict:
        self.decisions = {}
        due = [js for js in sorted(self.jobs.values(), key=lambda j: j.id) if js.decide_at == tick and not js.done]
        if not due:
            return {}
        WK = _m("backend.agents.work")
        R = _m("backend.simulation.records")
        day = self.clock.day_of(tick)
        views = {}
        for js in due:                                   # world-side reads, sequential (receipts, trace order)
            mode = WK.consult_mode(self.cfg, day) if self.binder is not None else "never"
            focal = self._symptom_focal(js)
            if mode == "always":
                views[js.id] = self._read(js.operator, tick, day, "decision", query=focal)
            elif mode == "on_choice":
                views[js.id] = R.render_for(self.binder, self.cfg, day, query=focal, embed=self.sim.embed)
            else:
                views[js.id] = None

        def job(js):
            a = self.sim.agents[js.operator]
            for o in js.obs.get(js.operator, []):
                if self.per_job:
                    WK.render_pending(a, o)                  # the operator's own facts, rendered per beat
                else:
                    for f in o.facts:                        # per_beat: already rendered in phase 4
                        f["rendered"] = True
            facts = [f for o in js.obs.get(js.operator, []) for f in o.facts]
            present = [self.sim.agents[x] for x in self.active_ids() if x != js.operator and
                       (self.sim.agents[x].state.location, self.sim.agents[x].state.arena) ==
                       (a.state.location, a.state.arena)]
            jv = WK.make_job_view(js.job.to_dict(), len(js.attempts) + 1, facts, present,
                                  questions_asked=js.questions, focal=self._symptom_focal(js))
            return WK.decide_job(a, jv, binder_view=views[js.id], rng=a.stream("work"))
        res = self.sim._parallel([(f"t{tick:04d}:02work:{js.operator}", (lambda js=js: job(js))) for js in due])
        for js, d in zip(due, res):
            self.decisions[js.id] = d
            if d.get("binder_chosen") and self.binder is not None:          # on_choice: the look is a read
                self._read(js.operator, tick, day, "decision", query=self._symptom_focal(js))
        return self.decisions

    def _symptom_focal(self, js: JobState) -> str | None:
        for o in js.obs.get(js.operator, []):
            for f in o.facts:
                if f.get("kind") == "symptom":
                    return f["text"]
        return None

    # ------------------------------------------------------------------------------ phase 4c
    def apply(self, tick: int, results: dict | None = None) -> ConvRequests:
        CT = _m("backend.agents.coop_talk")
        WK = _m("backend.agents.work")
        req = self.convs.pop(tick, ConvRequests())
        results = results if results is not None else {}
        for jid in sorted(self.decisions):
            js, d = self.jobs[jid], self.decisions[jid]
            op = self.sim.agents[js.operator]
            attempt = len(js.attempts) + 1
            if d.get("action") == "ask" and d.get("ask"):
                js.questions += 1
                js.decide_at = tick + 1                   # attempt 1 is decided at t0+1, without G
                tgt = self.sim.agents[d["ask"]["target"]]
                req.dyads.insert(0, CT.clarify_talk(op, tgt, d["ask"]["question"] or "", jid))
                self.sim.tracer.log("clarification", **CT.clarification_record(jid, attempt, op.id, tgt.id,
                                                                               d["ask"]["question"]))
                self.counts["asks"] += 1
            else:
                action = d.get("action") if d.get("action") in self.om.actions else "stop"
                outcome = self.om.resolve(js.job, attempt, action, js.regime)
                js.attempts.append({"attempt": attempt, "action": action, "outcome": outcome, "tick": tick})
                js.pending, js.outcome_at, js.decide_at = {"attempt": attempt, "action": action,
                                                           "outcome": outcome}, tick + 1, None
                if d.get("reason"):
                    js.reasons.append(d["reason"])
                self.counts["decisions"] += 1
            r = WK.remark_of(d)
            if r is not None:                             # says_aloud -> v2 REACT remark path
                r["observation"] = AgentObservation(
                    id=f"t{tick:04d}:work:{jid}.a{attempt}:{op.id}", agent_id=op.id, tick=tick,
                    location=op.state.location, arena=op.state.arena, source_type="perception", facts=[],
                    event_ids=[jid])
                results.setdefault(op.id, {"decisions": []})["decisions"].append(r)
        self.decisions = {}
        for js in sorted(self.jobs.values(), key=lambda j: j.id):
            if js.done and not js.ended:
                self._end_job(js, tick)
        self._writes(tick)
        return req

    def _end_job(self, js: JobState, tick: int):
        W = _m("backend.simulation.workshop")
        WK = _m("backend.agents.work")
        js.ended = True
        atts = [{k: x[k] for k in ("attempt", "action", "outcome")} for x in js.attempts]
        self.sim.tracer.log("job_end", **W.trace_job_end(js.job, atts, js.delivered))
        self.outcomes.append({"job": js.id, "day": js.job.day, "tick": tick, "delivered": js.delivered,
                              "attempts": atts, "regime": js.regime})
        if self.per_job:
            act = set(self.active_ids())
            for aid in sorted(js.obs):
                if aid not in act:
                    continue
                a = self.sim.agents[aid]
                facts = [f for o in js.obs[aid] for f in o.facts]
                ep = WK.job_episode(a, js.id, facts, tick=tick + 1,
                                    reasons=js.reasons if aid == js.operator else (),
                                    obs_id=f"t{tick + 1:04d}:obs:{js.id}.ep:{aid}")
                if ep is None:
                    continue
                ep.coop_episode = True
                ep.no_react = aid == js.operator          # react decisions are for the other perceivers (§4.2)
                self.pending[tick + 1][aid].append(ep)
        if js.attempts and self.binder is not None and \
                _m("backend.simulation.records").offer_enabled(self.cfg, "job"):
            own = [f["text"] for o in js.obs.get(js.operator, []) for f in o.facts]
            self.offers[tick].append((js.operator, "job", own, js.id))

    def _writes(self, tick: int):
        offers = sorted(self.offers.pop(tick, []), key=lambda o: (o[0], o[1]))
        if not offers or self.binder is None:
            return
        R = _m("backend.simulation.records")
        RW = _m("backend.agents.record_write")
        act = set(self.active_ids())
        offers = [o for o in offers if o[0] in act]
        view = R.render_for(self.binder, self.cfg, self.clock.day_of(tick)).text   # every writer sees this view

        def job(o):
            aid, offer, remembered, jid = o
            a = self.sim.agents[aid]
            return RW.decide_write(a, offer, view, remembered, tick=tick, job_id=jid, rng=a.stream("record"))
        res = self.sim._parallel([(f"t{tick:04d}:03record:{o[0]}", (lambda o=o: job(o))) for o in offers])
        for applied in R.apply_writes(self.binder, self.cfg, self.sim.tracer, res):      # sorted (tick, agent)
            if applied["choice"] in ("log", "front"):
                self.counts["writes"] += 1
                RW.remember_write(self.sim.agents[applied["agent"]], applied)

    # ------------------------------------------------------------------------------ phase 8b
    def day_end(self, tick: int):
        if (tick + 1) % self.tpd:
            return None
        day = (tick + 1) // self.tpd
        self.sim.tracer.log("coop_day", day=day, counts=dict(sorted(self.counts.items())))
        CK = _m("backend.experiment.checkpoint")
        if not CK.enabled_for(self.cfg, day):
            return None
        extra = {"roster": self.roster.state_dict(day)} if self.roster is not None else None
        return CK.write_checkpoint(self.sim, day, extra=extra)

    # ------------------------------------------------------------------------ world-side views
    def checkpoint_state(self) -> dict:
        """HIDDEN: the co-op's world position for checkpoints/world_state.json (found as sim.coop)."""
        return {"mapping": self.mapping, "regime": self.regime, "outcomes": list(self.outcomes),
                "counts": dict(sorted(self.counts.items()))}

    def manipulation_checks(self) -> dict:
        """§5.11 item 14: every mechanism that is on, with its count; `inactive` if it never fired."""
        CT = _m("backend.agents.coop_talk")
        on = {"jobs": self.workshop_on, "faulted": self.workshop_on, "decisions": self.workshop_on,
              "asks": self.workshop_on and CT.enabled(self.cfg, "clarify"),
              "handovers": self.workshop_on and CT.enabled(self.cfg, "handover"),
              "meetings": CT.enabled(self.cfg, "meeting"), "reads": self.records_on, "writes": self.records_on,
              "transitions": bool((self.cfg.get("records") or {}).get("transitions")) and self.records_on}
        return {k: {"n": self.counts.get(k, 0), "status": "active" if self.counts.get(k) else "inactive"}
                for k, v in on.items() if v}

    def frame_state(self) -> dict:
        """Agent-visible binder (as displayed; no job ids) for the frame record / UI."""
        return {"binder": self.binder.frame()} if self.binder is not None else {}
