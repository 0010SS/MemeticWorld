"""Simulation engine: WORLD layer + orchestration of AGENT cognition.

Tick phases (deterministic given seed + recorded LLM outputs):
  0 modules.on_tick
  1 latent event scheduling (world RNG)
  2 movement (routine / event-forced / invitation / reaction re-plan)
  3 partial perception sampling (+ ambient perception of co-located people)
  4 [parallel per agent] memory encoding + reaction decision
  5 apply decisions (remarks -> exposures; TALK -> conversations; MOVE -> re-plan)
  6 conversation gating, GA decide_to_talk [parallel], conversations [parallel],
    overheard-speech encoding [parallel per listener]
  7 [parallel] reflection for agents whose importance trigger fired
  8 frame snapshot; flush trace
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from backend import ga_compat
from backend.agents.agent import Agent
from backend.agents.conversation import decide_to_talk, run_conversation, talk_gate
from backend.agents.perception import AgentObservation, observe
from backend.agents.planner import decide_reaction
from backend.agents.profile import load_population
from backend.llm.client import LLMClient, llm_scope, make_backend
from backend.llm.embeddings import make_embedder
from backend.memory.encoder import add_simple_event, encode
from backend.memory.reflection import reflect, should_reflect
from backend.memory.store import MemoryMeta, SimMemoryMeta
from backend.modules.base import build_modules
from backend.simulation import latent_events as LE
from backend.simulation.commons import CommonsWorld
from backend.simulation.scheduler import next_entry, plan_day, routine_position
from backend.simulation.world import (ARENAS, HOMEWOOD_LABELS, MAP_POS, WORLD_GRAPH, Clock,
                                      default_arena, shortest_path)
from backend.tracing.logger import TraceLogger


class SimContext:
    """Shared handles. `agents` is the population; `meta` is SIMULATOR-ONLY."""

    def __init__(self, cfg, llm, embed, meta, tracer, clock):
        self.cfg, self.llm, self.embed, self.meta, self.tracer, self.clock = cfg, llm, embed, meta, tracer, clock
        self.agents: dict[str, Agent] = {}
        self.mods = None


def _seed_rng(*parts) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(str(p).encode()) for p in parts])


class Simulation:
    def __init__(self, cfg: dict, run_dir: Path, replay_from: Path | None = None, progress=True):
        self.cfg = cfg
        mode_name = cfg.get("world", {}).get("mode", "latent_events")
        if mode_name not in ("latent_events", "commons"):
            raise ValueError(f"Unknown world mode: {mode_name}")
        if mode_name == "commons" and any(cfg.get("modules", {}).values()):
            raise ValueError("Commons experiments currently require optional cognition modules to be disabled")
        self.run_dir = Path(run_dir)
        if any((self.run_dir / name).exists() for name in ("manifest.json", "trace.jsonl", "llm_calls.jsonl")):
            raise FileExistsError(f"Run directory already contains a recording: {self.run_dir}. Choose a new output directory.")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress = progress
        yaml.safe_dump(cfg, open(self.run_dir / "config.resolved.yaml", "w"), sort_keys=False)
        self.clock = Clock(cfg)
        backend = make_backend(cfg["llm"])
        replay_from = replay_from or cfg["llm"].get("replay_from")
        mode = "replay" if replay_from else "record"
        self.llm = LLMClient(backend, self.run_dir / "llm_calls.jsonl", mode=mode,
                             replay_path=Path(replay_from) if replay_from else None,
                             fallback=None if mode_name == "commons" or (mode == "replay" and cfg["llm"]["backend"] != "mock") else backend,
                             raise_on_error=mode_name == "commons")
        self.embed = make_embedder(cfg.get("embedding"))
        ga_compat.load(self.llm, self.embed)
        self.meta = SimMemoryMeta()
        self.tracer = TraceLogger(self.run_dir)
        self.ctx = SimContext(cfg, self.llm, self.embed, self.meta, self.tracer, self.clock)
        profiles, self.groups = load_population(cfg["population"], cfg.get("population_size"))
        self.rng = np.random.default_rng(cfg["seed"])
        for pid, prof in profiles.items():
            a = Agent(prof, cfg, None, cfg["seed"])
            a.ctx = self.ctx
            self.ctx.agents[pid] = a
        self.ctx.mods = build_modules(cfg, self.ctx)
        for a in self.ctx.agents.values():
            a.mods = self.ctx.mods
        self.agents = self.ctx.agents
        self.commons = (CommonsWorld({**cfg.get("commons", {}), "seed": cfg["seed"]},
                                    {aid: a.name for aid, a in self.agents.items()}, self.clock.ticks_per_day)
                        if mode_name == "commons" else None)
        self.events: list[LE.EventInstance] = []
        self.pending_obs: dict[str, list[AgentObservation]] = {}
        self.overrides: dict[str, dict] = {}      # agent -> {"until": tick, pos...}
        self.follow: dict[str, tuple[str, int]] = {}
        self.ev_counter = 0
        self.frames_fh = open(self.run_dir / "frames.jsonl", "w")
        self.events_fh = open(self.run_dir / "events.jsonl", "w")
        self.pool = ThreadPoolExecutor(max_workers=int(cfg["llm"].get("max_workers", 4)))
        self.stats = {"conversations": 0, "utterances": 0, "events": 0, "reflections": 0}

    # ------------------------------------------------------------------ utils
    def _parallel(self, jobs: list[tuple[str, callable]]):
        """Run (scope, fn) jobs concurrently; results returned in input order."""
        def wrap(scope, fn):
            with llm_scope(scope):
                return fn()
        futs = [self.pool.submit(wrap, s, f) for s, f in jobs]
        return [f.result() for f in futs]

    def _seed_memories(self):
        """Initial memories (GA seeds personas with their description; we add relationship knowledge)."""
        t0 = self.clock.time_of(0) - dt.timedelta(hours=10)
        with llm_scope("seed"):
            for a in self.agents.values():
                a.set_time(t0)
                for oid, other in sorted(self.agents.items()):
                    if oid == a.id or a.profile.rel(oid).relation_type == "stranger":
                        continue
                    text = a.relationship_line(other)
                    node = a.a_mem.add("thought", t0, a.name, "knows", other.name, text,
                                       {a.name.lower(), other.name.lower()}, 5, self.embed(text), [])
                    self.meta.set(node.node_id, MemoryMeta(agent_id=a.id, source_type="seed", salience=0.3))
                    self.tracer.log("memory_encoded", agent=a.id, node_id=node.node_id, kind="thought", text=text,
                                    importance=5, salience=0.3, source_type="seed", observation_id=None,
                                    observation=None, encoding_ops=[], prompt=None, originating_event_ids=[],
                                    speakers=[], utterance_ids=[])
        self.tracer.flush()

    def write_manifest(self, status="running", extra=None):
        man = {
            "run_id": self.run_dir.name, "status": status, "config": self.cfg,
            "ticks": self.clock.total_ticks, "ticks_per_day": self.clock.ticks_per_day,
            "tick_minutes": self.clock.tick_minutes,
            "start": self.clock.time_of(0).isoformat(),
            "world": {"mode": self.cfg.get("world", {}).get("mode", "latent_events"),
                      "graph": WORLD_GRAPH, "arenas": ARENAS, "labels": HOMEWOOD_LABELS, "map_pos": MAP_POS},
            "agents": {aid: a.profile.to_public_dict() for aid, a in self.agents.items()},
            "groups": self.groups,
            "latent_types": LE.LATENT_TYPES,
            "modules": [m.name for m in self.ctx.mods.modules],
            "ga_upstream": "joonspk-research/generative_agents@fe05a71",
            "stats": {**self.stats, "llm": self.llm.stats},
        }
        if self.commons:
            man.pop("latent_types")
            man["research_design"] = {
                "primary_question": "RQ2: How do shared records affect continuity and adaptation?",
                "records_enabled": self.commons.records_enabled,
                "change_enabled": self.commons.change_enabled,
                "change_tick": self.commons.change_tick, "turnover_tick": self.commons.turnover_tick,
                "schema_version": self.commons.schema_version,
                "semantic_change_claim": False,
                "active_agents": self.commons.active_ids(),
            }
        if extra:
            man.update(extra)
        json.dump(man, open(self.run_dir / "manifest.json", "w"), indent=1, default=str)

    # ------------------------------------------------------------------ world
    def _maybe_start_event(self, tick: int):
        lc = self.cfg["latent_events"]
        if self.rng.random() >= float(lc["event_rate"]):
            return
        fams = [f for f in lc["families"]]
        w = np.array([float(lc["family_weights"].get(f, 1.0)) for f in fams])
        fam = fams[int(self.rng.choice(len(fams), p=w / w.sum()))]
        hf = lc.get("holdout_from_day")
        holdout = hf is not None and self.clock.day_of(tick) >= int(hf)
        scn = None
        if lc.get("generator") == "llm" and not holdout:
            with llm_scope(f"t{tick:04d}:00worldgen"):
                scn = LE.llm_scenario(fam, self.rng, self.llm)
        if scn is None:
            pool = LE.scenarios_for(fam, holdout)
            scn = pool[int(self.rng.integers(len(pool)))]
        busy = {a for e in self.events if e.beats[-1]["tick"] >= tick for a in
                [r["agent"] for r in e.roles.values() if r["agent"]]}
        busy |= {a for a, ag in self.agents.items() if ag.state.in_conversation}
        eid = f"ev{self.ev_counter:03d}"
        inst = LE.instantiate(scn, eid, tick, self.rng, self.agents, busy, self.clock.ticks_per_day)
        if inst is None:
            return
        self.ev_counter += 1
        self.events.append(inst)
        self.stats["events"] += 1
        gt = inst.ground_truth()
        self.events_fh.write(json.dumps(gt) + "\n")
        self.events_fh.flush()
        self.tracer.log("world_event_start", event_id=inst.id, latent_type=inst.latent_type,
                        scenario=inst.scenario, holdout=inst.holdout, roles=inst.roles, narrative=gt["narrative"])

    def _beats_now(self, tick):
        out = []
        for e in self.events:
            for b in e.beats:
                if b["tick"] == tick:
                    out.append((e, b))
        return out

    def _move(self, tick: int, beats):
        k = tick % self.clock.ticks_per_day
        forced = {}
        for e, b in beats:
            p = e.roles["P"]["agent"]
            if b["location"] == "current":
                pos = routine_position(self.agents[p], k)
                if p in self.overrides:
                    pos = self.overrides[p]
                b["location"], b["arena"] = pos["location"], b.get("arena") or pos["arena"]
                b["forced"] = False
            elif b["location"] == "current:Q":
                q = e.roles["Q"]["agent"]
                src = self.agents[q] if q else self.agents[p]
                pos = routine_position(src, k)
                b["location"], b["arena"] = pos["location"], pos["arena"]
                b["forced"] = False
            else:
                if not b.get("arena"):
                    rp = routine_position(self.agents[p], k)
                    b["arena"] = rp["arena"] if rp["location"] == b["location"] else default_arena(b["location"])
                b["forced"] = True
            for a in b["movers"]:
                if b["forced"] or a == p:
                    forced[a] = {"location": b["location"], "arena": b["arena"], "event": e.id,
                                 "activity": None}
        for aid, a in self.agents.items():
            prev = (a.state.location, a.state.arena)
            pos = routine_position(a, k)
            activity = pos["activity"]
            forced_by = None
            ov = self.overrides.get(aid)
            if ov and ov["until"] >= tick:
                pos, activity = ov, ov["activity"]
            elif ov:
                self.overrides.pop(aid)
            fl = self.follow.get(aid)
            if fl and fl[1] >= tick:
                leader = self.agents[fl[0]]
                lp = routine_position(leader, k)
                pos = lp
                activity = f"hanging out with {leader.profile.first_name}"
            elif fl:
                self.follow.pop(aid)
            if aid in forced:
                f = forced[aid]
                if (f["location"], f["arena"]) != (pos["location"], pos["arena"]):
                    activity = "dealing with something unexpected"
                pos = f
                forced_by = f["event"]
            a.state.location, a.state.arena, a.state.activity = pos["location"], pos["arena"], activity
            a.state.forced_by = forced_by
            a.state.path = shortest_path(prev[0], a.state.location)
            a.state.in_conversation = None
            a.state.pre_chat_activity = None
            if prev != (a.state.location, a.state.arena):
                self.tracer.log("move", agent=aid, frm=list(prev), to=[a.state.location, a.state.arena],
                                path=a.state.path, activity=activity, forced_by_event=forced_by)
            a.sync_scratch()

    # ----------------------------------------------------------- main phases
    def _perceive(self, tick, beats):
        obs_by_agent: dict[str, list[AgentObservation]] = {}
        for e, b in beats:
            self.tracer.log("event_beat", event_id=e.id, latent_type=e.latent_type, beat=b["idx"],
                            location=b["location"], arena=b["arena"], facts=b["facts"])
            for aid in sorted(self.agents):
                a = self.agents[aid]
                if a.state.location != b["location"]:
                    continue
                oid = f"t{tick:04d}:obs:{e.id}.b{b['idx']}:{aid}"
                o = observe(a, b, e.id, self.cfg, a.rng, oid)
                if o is None:
                    continue
                obs_by_agent.setdefault(aid, []).append(o)
                self.tracer.log("observation", agent=aid, observation_id=oid, source_type="perception",
                                event_id=e.id, facts=[{k: f[k] for k in ("id", "text", "salience", "p_attend")}
                                                      for f in o.facts], originating_event_ids=[e.id])
        # ambient perception of co-located people's routine activities (no LLM)
        with llm_scope(f"t{tick:04d}:01ambient"):
            for aid in sorted(self.agents):
                a = self.agents[aid]
                if a.state.activity == "sleeping":
                    continue
                for oid in sorted(self.agents):
                    o = self.agents[oid]
                    if oid == aid or (o.state.location, o.state.arena) != (a.state.location, a.state.arena):
                        continue
                    if o.state.activity == "sleeping":
                        continue
                    text = f"{o.name} is {o.state.activity} at the {o.state.location}."
                    recent = [n.description for n in a.a_mem.seq_event[:6]]
                    if text in recent:
                        continue
                    if a.rng.random() < 0.5:
                        add_simple_event(a, text, 2 if a.profile.rel(oid).familiarity > 0.3 else 1, [oid])
        return obs_by_agent

    def _cognition(self, tick, obs_by_agent):
        rc = self.cfg["reaction"]

        def job(aid):
            a = self.agents[aid]
            out = {"decisions": []}
            for o in obs_by_agent[aid]:
                a.mods.on_observation(a, o.facts)
                encode(a, o, a.rng)
                if rc["enabled"] and max(f["salience"] for f in o.facts) >= rc["min_salience"]:
                    present = [x for x in self.agents.values() if x.id != aid and
                               (x.state.location, x.state.arena) == (a.state.location, a.state.arena)]
                    d = decide_reaction(a, o, present, a.rng)
                    d["observation"] = o
                    out["decisions"].append(d)
            return out
        ids = sorted(obs_by_agent)
        results = self._parallel([(f"t{tick:04d}:02agent:{aid}", (lambda aid=aid: job(aid))) for aid in ids])
        return dict(zip(ids, results))

    def _apply_decisions(self, tick, results):
        talks = []
        remark_obs: dict[str, list[AgentObservation]] = {}
        k = tick % self.clock.ticks_per_day
        with llm_scope(f"t{tick:04d}:03apply"):
            for aid in sorted(results):
                a = self.agents[aid]
                for d in results[aid]["decisions"]:
                    obs = d["observation"]
                    if d["action"] == "REACT" and d.get("utterance"):
                        uid = f"t{tick:04d}:remark:{aid}:{obs.id.split(':')[-2]}"
                        listeners = []
                        for x in sorted(self.agents.values(), key=lambda z: z.id):
                            if x.id == aid or x.state.location != a.state.location:
                                continue
                            p = 0.9 if x.state.arena == a.state.arena else 0.2
                            if a.rng.random() < p:
                                listeners.append(x.id)
                        ev = self.meta.events_of(d["retrieved"])
                        for e in obs.event_ids:
                            if e not in ev:
                                ev.append(e)
                        u = {"id": uid, "conversation_id": None, "idx": 0, "speaker": aid, "text": d["utterance"],
                             "listeners": listeners, "location": a.state.location, "arena": a.state.arena,
                             "retrieved": d["retrieved"], "retrieved_event_ids": ev, "source": "reaction_remark"}
                        self.tracer.log("utterance", **u, context=obs.text())
                        self.tracer.log("exposure", utterance_id=uid, speaker_id=aid, listener_ids=listeners,
                                        utterance=d["utterance"], conversation_id=None,
                                        location=a.state.location, arena=a.state.arena)
                        self.stats["utterances"] += 1
                        self.tick_utts.append(u)
                        for lid in listeners:
                            remark_obs.setdefault(lid, []).append(AgentObservation(
                                id=f"{uid}.obs.{lid}", agent_id=lid, tick=tick, location=a.state.location,
                                arena=a.state.arena, source_type="overheard",
                                facts=[{"id": uid, "text": f'{a.profile.first_name}: "{d["utterance"]}"',
                                        "salience": 0.5, "involves": [aid]}],
                                event_ids=ev, speakers=[aid], utterance_ids=[uid], partner=a.name))
                    elif d["action"] == "TALK" and d.get("target_id"):
                        talks.append((aid, d["target_id"], d.get("utterance"),
                                      {"agent": aid, "text": obs.text(), "event_ids": obs.event_ids}))
                    elif d["action"] == "MOVE":
                        nxt = next_entry(a, k)
                        until = (nxt["k"] - 1 + (tick - k)) if nxt else tick + 4
                        self.overrides[aid] = {"location": d["target"], "arena": default_arena(d["target"]),
                                               "activity": f"heading to the {d['target']}", "until": until}
                        self.tracer.log("replan", agent=aid, reason="reaction", to=d["target"], until=until)
        return talks, remark_obs

    def _conversations(self, tick, forced_talks, extra_obs):
        now = self.clock.time_of(tick)
        busy = set()
        pairs = []
        for init_id, tgt_id, opening, trig in forced_talks:
            a, b = self.agents[init_id], self.agents[tgt_id]
            if init_id in busy or tgt_id in busy or (a.state.location, a.state.arena) != (b.state.location, b.state.arena):
                continue
            busy |= {init_id, tgt_id}
            pairs.append((init_id, tgt_id, opening, trig))
        # stochastic gate -> GA decide_to_talk
        cands = []
        order = sorted(self.agents)
        order = [order[i] for i in self.rng.permutation(len(order))]
        claimed = set(busy)
        for aid in order:
            a = self.agents[aid]
            if aid in claimed or a.state.activity == "sleeping":
                continue
            others = [o for o in sorted(self.agents) if o != aid and o not in claimed and
                      (self.agents[o].state.location, self.agents[o].state.arena) == (a.state.location, a.state.arena)]
            if not others:
                continue
            probs = [talk_gate(a, self.agents[o], now, self.rng) for o in others]
            for o, p in zip(others, probs):
                if p > 0 and self.rng.random() < p:
                    cands.append((aid, o))
                    claimed |= {aid, o}
                    break
        if cands:
            res = self._parallel([(f"t{tick:04d}:04dtt:{i}:{p}",
                                   (lambda p=p, q=q: decide_to_talk(self.agents[p], self.agents[q], self.agents[p].rng)[0]))
                                  for i, (p, q) in enumerate(cands)])
            for (p, q), yes in zip(cands, res):
                if yes and p not in busy and q not in busy:
                    busy |= {p, q}
                    pairs.append((p, q, None, None))
        if not pairs:
            return extra_obs
        convs = []
        for i, (p, q, opening, trig) in enumerate(pairs):
            cid = f"d{self.clock.day_of(tick)}t{tick:04d}c{i}"
            self.agents[p].state.in_conversation = cid
            self.agents[q].state.in_conversation = cid
            self.agents[p].state.pre_chat_activity = self.agents[p].state.activity
            self.agents[q].state.pre_chat_activity = self.agents[q].state.activity
            self.agents[p].state.activity = f"chatting with {self.agents[q].profile.first_name}"
            self.agents[q].state.activity = f"chatting with {self.agents[p].profile.first_name}"
            self.agents[p].sync_scratch()
            self.agents[q].sync_scratch()
            convs.append((cid, p, q, opening, trig))
        in_conv = {x for _, p, q, _, _ in convs for x in (p, q)}

        def conv_job(cid, p, q, opening, trig):
            a, b = self.agents[p], self.agents[q]
            by = [x for x in self.agents.values() if x.id not in in_conv and
                  (x.state.location, x.state.arena) == (a.state.location, a.state.arena)]
            return run_conversation(cid, a, b, sorted(by, key=lambda z: z.id), _seed_rng(self.cfg["seed"], cid),
                                    opening=opening, trigger=trig)
        results = self._parallel([(f"t{tick:04d}:05conv:{c[0]}", (lambda c=c: conv_job(*c))) for c in convs])
        for conv in results:
            self.stats["conversations"] += 1
            self.stats["utterances"] += len(conv["utterances"])
            self.tick_utts.extend(conv["utterances"])
            for o in conv.pop("overheard"):
                extra_obs.setdefault(o.agent_id, []).append(o)
            # invitations (reactive re-planning)
            p, q = conv["participants"]
            a, b = self.agents[p], self.agents[q]
            if conv["utterances"] and a.profile.rel(q).affinity >= 0.6 and self.rng.random() < self.cfg["conversation"]["invite_prob"]:
                self.follow[q] = (p, tick + 3)
                self.tracer.log("invitation", agent=p, target=q, until=tick + 3, conversation_id=conv["id"])
        return extra_obs

    def _encode_extra(self, tick, extra_obs):
        if not extra_obs:
            return

        def job(aid):
            a = self.agents[aid]
            for o in extra_obs[aid]:
                self.tracer.log("observation", agent=aid, observation_id=o.id, source_type=o.source_type,
                                facts=o.facts, originating_event_ids=o.event_ids)
                a.mods.on_observation(a, o.facts)
                encode(a, o, a.rng)
        ids = sorted(extra_obs)
        self._parallel([(f"t{tick:04d}:06heard:{aid}", (lambda aid=aid: job(aid))) for aid in ids])

    def _reflect(self, tick):
        ids = [aid for aid in sorted(self.agents) if should_reflect(self.agents[aid])]
        if not ids:
            return
        res = self._parallel([(f"t{tick:04d}:07reflect:{aid}", (lambda aid=aid: reflect(self.agents[aid], self.agents[aid].rng)))
                              for aid in ids])
        self.stats["reflections"] += sum(len(r) for r in res)

    def _frame(self, tick, beats):
        fr = {"tick": tick, "time": self.clock.time_of(tick).isoformat(), "day": self.clock.day_of(tick),
              "label": self.clock.label(tick),
              "agents": {aid: {**a.snapshot(), "modules": self.ctx.mods.frame_state(aid)}
                         for aid, a in self.agents.items()},
              "utterances": [{"id": u["id"], "speaker": u["speaker"], "text": u["text"], "listeners": u["listeners"],
                              "conversation_id": u["conversation_id"]} for u in self.tick_utts],
              "beats": [{"event_id": e.id, "latent_type": e.latent_type, "location": b["location"],
                         "arena": b["arena"], "facts": [f["text"] for f in b["facts"]]} for e, b in beats]}
        if self.commons:
            fr["commons"] = self.commons.snapshot()
            fr["agents"] = {aid: a for aid, a in fr["agents"].items() if self.commons.residents[aid].active}
        self.frames_fh.write(json.dumps(fr) + "\n")
        self.frames_fh.flush()

    # ------------------------------------------------------------------- run
    def run(self):
        if self.commons:
            from backend.simulation.commons_runtime import CommonsRuntime
            try:
                return CommonsRuntime(self).run()
            except Exception as exc:
                self.pool.shutdown(wait=True)
                self.tracer.log("run_failed", error_type=type(exc).__name__, message=str(exc))
                self.tracer.close()
                self.frames_fh.close()
                self.events_fh.close()
                self.llm.close()
                self.write_manifest("failed", {"failure": {"type": type(exc).__name__, "message": str(exc)}})
                raise
        t_start = time.time()
        self.write_manifest("running")
        self._seed_memories()
        tpd = self.clock.ticks_per_day
        for tick in range(self.clock.total_ticks):
            now = self.clock.time_of(tick)
            self.tracer.tick, self.tracer.time = tick, now.isoformat()
            self.tick_utts = []
            if tick % tpd == 0:
                day = self.clock.day_of(tick)
                for aid in sorted(self.agents):
                    a = self.agents[aid]
                    a.day_plan = plan_day(a, self.clock, _seed_rng(self.cfg["seed"], "plan", aid, day))
                    a.state.talks_today = 0
                    self.tracer.log("day_plan", agent=aid, day=day, plan=a.day_plan)
            for a in self.agents.values():
                a.set_time(now)
            self.ctx.mods.on_tick(tick, self.agents)
            with llm_scope(f"t{tick:04d}:00world"):
                self._maybe_start_event(tick)
                beats = self._beats_now(tick)
                self._move(tick, beats)
                obs = self._perceive(tick, beats)
            results = self._cognition(tick, obs)
            talks, remark_obs = self._apply_decisions(tick, results)
            extra = self._conversations(tick, talks, remark_obs)
            self._encode_extra(tick, extra)
            self._reflect(tick)
            with llm_scope(f"t{tick:04d}:99frame"):
                self._frame(tick, beats)
            self.tracer.flush()
            if self.progress and (tick % 4 == 0 or tick == self.clock.total_ticks - 1):
                s = self.llm.stats
                print(f"[{self.clock.label(tick)}] tick {tick + 1}/{self.clock.total_ticks} "
                      f"events={self.stats['events']} convs={self.stats['conversations']} "
                      f"utts={self.stats['utterances']} refl={self.stats['reflections']} "
                      f"llm={s['calls']} err={s['errors']} {time.time() - t_start:.0f}s", flush=True)
            if tick % 8 == 0:
                self.write_manifest("running")
        self.finish(time.time() - t_start)

    def finish(self, seconds: float):
        out = self.run_dir / "agents_final"
        for aid, a in self.agents.items():
            a.a_mem.save_ga(out / aid / "associative_memory")
        json.dump({nid: asdict(m) for nid, m in self.meta.meta.items()},
                  open(self.run_dir / "memory_meta.json", "w"))
        self.tracer.close()
        self.frames_fh.close()
        self.events_fh.close()
        self.llm.close()
        self.pool.shutdown()
        self.write_manifest("finished", {"wall_seconds": round(seconds, 1),
                                         "trace_sha256": trace_digest(self.run_dir)})


def trace_digest(run_dir: Path) -> str:
    """Hash of the trace excluding wall-clock fields (for replay verification)."""
    h = hashlib.sha256()
    with open(Path(run_dir) / "trace.jsonl") as f:
        for line in f:
            h.update(line.encode())
    return h.hexdigest()


def new_run_dir(cfg: dict, root: Path | None = None) -> Path:
    root = root or (ga_compat.REPO_ROOT / "runs")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(root) / f"{stamp}_{cfg.get('run_name', 'run')}_s{cfg['seed']}"
