"""Simulation engine: WORLD layer + orchestration of AGENT cognition.

Tick phases (deterministic given seed + recorded LLM outputs):
  0 modules.on_tick
  1 latent event scheduling (world RNG)
  2 movement (routine / event-forced / invitation / reaction re-plan); whoever talked last tick and is
    still with a partner there stays busy with that conversation through perception
  3 partial perception sampling (+ ambient perception of co-located people)
  4 [parallel per agent] viewpoint rendering + memory encoding + reminding + reaction decision
  5 apply decisions (remarks -> exposures; TALK -> conversations; MOVE -> re-plan)
  6 group talk at venues, evening catch-ups, conversation gating, GA decide_to_talk [parallel],
    conversations [parallel], overheard-speech encoding [parallel per listener]

Random numbers: routine plans and the world script come from world_seed streams; everything social
and cognitive from `seed` streams, one per purpose (rngs.seed_rng), keyed by tick, agent, beat or the
participants of a conversation, never by the position of something in a per-tick list.
  7 [parallel] reflection for agents whose importance trigger fired
  8 frame snapshot; flush trace
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from collections import defaultdict
import time
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
from backend.agents.viewpoint import render as render_viewpoint
from backend.agents.profile import load_population
from backend.llm.client import LLMClient, llm_scope, make_backend
from backend.llm.embeddings import make_embedder
from backend.memory.encoder import add_simple_event, encode
from backend.memory.reflection import reflect, should_reflect
from backend.memory.reminding import maybe_remind
from backend.simulation.lexicon import population_lexicon
from backend.memory.store import MemoryMeta, SimMemoryMeta
from backend.modules.base import build_modules
from backend.simulation import latent_events as LE
from backend.simulation.rngs import seed_rng, world_seed
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


def _mins(v) -> int:
    h, m = map(int, str(v).split(":"))
    return h * 60 + m


_seed_rng = seed_rng       # the engine's streams use the one scheme in backend.simulation.rngs


def forced_activity(routine: dict, place: dict) -> str:
    """Activity of a beat mover (agreement c): the world moves people, it does not narrate them. Keep what
    the agent was going to do there anyway (same building, awake), else a neutral, event-independent
    'stopping by the <location>'. Never a gloss that tells the agent or onlookers that something happened."""
    if routine["location"] == place["location"] and routine["activity"] != "sleeping":
        return routine["activity"]
    return f"stopping by the {place['location']}"


def _manip(mods) -> dict:
    f = getattr(mods, "manipulation_checks", None)
    return f() if f else {}


# everything a run executes: our code and configs plus the vendored GA code and prompt templates
CODE_PATHS = ("backend", "configs", "third_party")
GA_TEMPLATES = ga_compat.GA_ROOT / "persona" / "prompt_template"


def _code_version() -> dict:
    """git sha + dirty flag + prompt-file hashes, so every run records exactly what produced it.
    `dirty` covers CODE_PATHS (tracked changes and untracked files); `prompt_hashes` covers MemeWorld's
    prompts (by file name) and every upstream GA template (as `ga/<path under prompt_template>`), which
    decide_to_talk, chat utterances, poignancy, relationship summaries and reflection load."""
    root = ga_compat.REPO_ROOT
    out = {"git_sha": None, "dirty": None, "prompt_hashes": {}}
    try:
        out["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                        timeout=10).stdout.strip() or None
        out["dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=root,
                                           capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:  # noqa: BLE001 - not a git checkout
        pass
    for f in sorted((root / "backend" / "prompts").glob("*.txt")):
        out["prompt_hashes"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    for f in sorted(GA_TEMPLATES.rglob("*.txt")):
        rel = f.relative_to(GA_TEMPLATES).as_posix()
        out["prompt_hashes"][f"ga/{rel}"] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


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
        self.world_seed = world_seed(cfg)
        self.topology = None
        if (cfg.get("topology") or {}).get("mode") == "generated":
            from backend.agents import topology as TOPO
            topo = TOPO.generate(profiles, cfg, _seed_rng(self.world_seed, "topology"))
            TOPO.apply(profiles, topo)
            self.circles = topo["circles"]
            self.topology = topo.get("metrics")
        else:
            from backend.simulation.circles import load_circles
            self.circles = load_circles(cfg["population"], list(profiles))
        if self.topology is None:
            from backend.agents import topology as TOPO
            self.topology = TOPO.metrics(profiles, self.circles, mode="file") if hasattr(TOPO, "metrics") else None
        else:
            self.groups = dict(self.circles)              # generated ties replace the file's groups
        # campus vocabulary BEFORE the planted habit, so the planted wording never counts as ordinary vocabulary
        self.lexicon = population_lexicon(profiles)
        from backend.agents.profile import apply_planted
        apply_planted(profiles, cfg)                      # positive-control cell only (controls.planted_phrase)
        self.rng = np.random.default_rng(cfg["seed"])     # legacy; per-purpose streams are used below
        for pid, prof in profiles.items():
            a = Agent(prof, cfg, None, cfg["seed"])
            a.ctx = self.ctx
            self.ctx.agents[pid] = a
        self.ctx.mods = build_modules(cfg, self.ctx)
        for a in self.ctx.agents.values():
            a.mods = self.ctx.mods
        self.agents = self.ctx.agents
        self.ctx.lexicon = self.lexicon
        self.ctx.circles = self.circles
        # WORLD SCRIPT (v2 §1.1): every event is generated before the run from world_seed streams only,
        # so all conditions sharing world_seed see identical events (common random numbers).
        from backend.simulation import world_script as WS
        lc = cfg["latent_events"]
        self.world = WS.load(lc["script_from"]) if lc.get("script_from") else WS.generate(cfg, profiles, self.circles, self.clock)
        self.world_sha = WS.save(self.world, self.run_dir / "world_script.jsonl")
        self.world_by_tick: dict[int, list] = defaultdict(list)
        for inst in self.world:
            self.world_by_tick[inst.start_tick].append(inst)
        self.catchups: set = set()   # (day, pair) already caught up (D47)
        # (day, window, venue) -> agents whose presence already had its one group-talk draw in that window
        self.group_seen: dict[tuple, set] = {}
        self.code_version = _code_version()
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
            "circles": self.circles,
            "assignment": self.cfg["latent_events"].get("assignment"),
            "condition": self.cfg.get("_condition"),
            "code_version": self.code_version,
            "world_script_sha256": self.world_sha,
            "population_lexicon": self.lexicon,
            "topology": self.topology,
            "manipulation": _manip(self.ctx.mods),
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
    def _release_events(self, tick: int):
        """Release pre-generated world-script instances that start at this tick."""
        for inst in self.world_by_tick.get(tick, []):
            self.events.append(inst)
            self.stats["events"] += 1
            gt = inst.ground_truth()
            self.events_fh.write(json.dumps(gt) + "\n")
            self.events_fh.flush()
            from backend.simulation import world_script as WS
            self.tracer.log("world_event_start", event_id=inst.id, **WS.trace_fields(inst))

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
            for a in b["movers"]:
                if a in self.agents:
                    forced[a] = {"location": b["location"], "arena": b["arena"], "event": e.id, "activity": None}
        was = {aid: ((a.state.location, a.state.arena), a.state.in_conversation) for aid, a in self.agents.items()}
        for aid, a in self.agents.items():
            prev = was[aid][0]
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
                activity = forced_activity({**pos, "activity": activity}, f)
                pos = f
                forced_by = f["event"]
            a.state.location, a.state.arena, a.state.activity = pos["location"], pos["arena"], activity
            a.state.forced_by = forced_by
            a.state.path = shortest_path(prev[0], a.state.location)
            a.state.pre_chat_activity = None
            if prev != (a.state.location, a.state.arena):
                self.tracer.log("move", agent=aid, frm=list(prev), to=[a.state.location, a.state.arena],
                                path=a.state.path, activity=activity, forced_by_event=forced_by)
            a.sync_scratch()
        # A conversation fills the rest of its tick, so whoever talked last tick and is still where they
        # talked, with a partner from that conversation, is still busy with it while perceiving this tick
        # (busy_factor, 'distracted' vantage). The conversation phase clears it (_conversations).
        for aid, a in self.agents.items():
            (here, cid) = was[aid]
            here_now = (a.state.location, a.state.arena)
            a.state.in_conversation = cid if cid and here_now == here and any(
                o != aid and was[o][1] == cid and (x.state.location, x.state.arena) == here_now
                for o, x in self.agents.items()) else None

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
                # one attention stream per (agent, beat): what an agent notices of this beat never depends
                # on which earlier beats it happened to be near, or on another fact's visibility
                prng = seed_rng(a.seed, aid, "perceive", e.id, b["idx"])
                o = observe(a, b, e.id, self.cfg, prng, oid)
                if o is None:
                    continue
                obs_by_agent.setdefault(aid, []).append(o)
                self.tracer.log("observation", agent=aid, observation_id=oid, source_type="perception",
                                event_id=e.id, facts=[{k: f[k] for k in ("id", "text", "salience", "p_attend")}
                                                      for f in o.facts], originating_event_ids=[e.id])
        # ambient perception of co-located people's routine activities (no LLM); one stream per agent and
        # tick, so a different position earlier in the day does not shift later ambient draws
        with llm_scope(f"t{tick:04d}:01ambient"):
            for aid in sorted(self.agents):
                a = self.agents[aid]
                if a.state.activity == "sleeping":
                    continue
                arng = seed_rng(a.seed, aid, "ambient", tick)
                for oid in sorted(self.agents):
                    o = self.agents[oid]
                    if oid == aid or (o.state.location, o.state.arena) != (a.state.location, a.state.arena):
                        continue
                    if o.state.activity == "sleeping":
                        continue
                    act, loc = o.state.activity, o.state.location
                    text = f"{o.name} is {act}." if act.endswith(f"the {loc}") else f"{o.name} is {act} at the {loc}."
                    recent = [n.description for n in a.a_mem.seq_event[:6]]
                    if text in recent:
                        continue
                    if arng.random() < 0.5:
                        add_simple_event(a, text, 2 if a.profile.rel(oid).familiarity > 0.3 else 1, [oid])
        return obs_by_agent

    def _cognition(self, tick, obs_by_agent):
        rc = self.cfg["reaction"]

        def job(aid):
            a = self.agents[aid]
            out = {"decisions": []}
            for o in obs_by_agent[aid]:
                render_viewpoint(a, o)                 # the agent's own version of what it noticed (D42)
                if not o.facts:
                    continue
                a.mods.on_observation(a, o.facts)
                node = encode(a, o, a.stream("encode"))
                maybe_remind(a, o, node, a.stream("remind"))  # may link this to an earlier experience (D48)
                if self.cfg.get("need", {}).get("enabled"):
                    from backend.memory import need as NEED
                    NEED.note(a, o, node)
                if rc["enabled"] and max(f["salience"] for f in o.facts) >= rc["min_salience"]:
                    present = [x for x in self.agents.values() if x.id != aid and
                               (x.state.location, x.state.arena) == (a.state.location, a.state.arena)]
                    d = decide_reaction(a, o, present, a.stream("react"))
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
                            if a.stream("remark").random() < p:
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
        seed, day = self.cfg["seed"], self.clock.day_of(tick)
        for a in self.agents.values():
            a.state.in_conversation = None      # last tick's conversation (busy while perceiving) is over
        # One stream per social purpose and tick (D57): switching catch-ups or group talk on never shifts
        # the dyadic gate's draws. Conversations and invitations are seeded by who takes part, never by
        # their position in this tick's list, so one extra conversation does not reseed the others.
        crng = seed_rng(seed, "catchup", tick)
        grng = seed_rng(seed, "group", tick)
        trng = seed_rng(seed, "talk", tick)
        busy = set()
        pairs = []
        for init_id, tgt_id, opening, trig in forced_talks:
            a, b = self.agents[init_id], self.agents[tgt_id]
            if init_id in busy or tgt_id in busy or (a.state.location, a.state.arena) != (b.state.location, b.state.arena):
                continue
            busy |= {init_id, tgt_id}
            pairs.append((init_id, tgt_id, opening, trig))
        # multi-party talk at shared venues (v2 §3), before catch-ups: people sharing a table talk together,
        # and close friends who are not at one catch up. `prob` is per window, not per tick: the first time
        # at least min_participants free, awake agents are at the venue who have not had a draw in this
        # (day, window, venue), they get ONE draw together, whatever its outcome. Agents who arrive later
        # (e.g. another circle's staggered dinner, D62) form a new eligible set with its own draw.
        groups_now = []
        gc = self.cfg["conversation"].get("group") or {}
        if gc.get("enabled"):
            mins = now.hour * 60 + now.minute
            for wi, w in enumerate(gc.get("windows", [])):
                lo, hi = (_mins(x) for x in w.split("-"))
                if not lo <= mins <= hi:
                    continue
                for loc, arena in gc.get("venues", []):
                    seen = self.group_seen.setdefault((day, wi, loc, arena), set())
                    here = [aid for aid in sorted(self.agents) if aid not in busy and aid not in seen
                            and self.agents[aid].state.activity != "sleeping"
                            and (self.agents[aid].state.location, self.agents[aid].state.arena) == (loc, arena)]
                    if len(here) < int(gc.get("min_participants", 3)):
                        continue
                    seen |= set(here)
                    if grng.random() >= float(gc.get("prob", 0.5)):
                        continue
                    here = [here[i] for i in grng.permutation(len(here))][: int(gc.get("max_participants", 5))]
                    busy |= set(here)
                    groups_now.append(sorted(here))
        # daily catch-up between close friends in the evening (D47): a recurring shared context in which
        # what happened lately naturally comes up again. No topic is imposed beyond "catching up".
        cu = self.cfg["conversation"].get("catchup") or {}
        if cu.get("enabled") and (now.hour * 60 + now.minute) >= _mins(cu.get("after", "17:00")):
            for aid in sorted(self.agents):
                a = self.agents[aid]
                if aid in busy or a.state.activity == "sleeping":
                    continue
                for oid in sorted(self.agents):
                    b = self.agents[oid]
                    key = (day, tuple(sorted((aid, oid))))
                    if (oid == aid or oid in busy or key in self.catchups or b.state.activity == "sleeping"
                            or (a.state.location, a.state.arena) != (b.state.location, b.state.arena)
                            or a.profile.rel(oid).familiarity < float(cu.get("min_familiarity", 0.6))):
                        continue
                    if crng.random() < float(cu.get("prob", 0.8)):
                        self.catchups.add(key)
                        busy |= {aid, oid}
                        pairs.append((aid, oid, None, {"agent": None, "text": None, "topic": "catchup"}))
                        break
        # stochastic gate -> GA decide_to_talk
        cands = []
        order = sorted(self.agents)
        order = [order[i] for i in trng.permutation(len(order))]
        claimed = set(busy)
        for aid in order:
            a = self.agents[aid]
            if aid in claimed or a.state.activity == "sleeping":
                continue
            others = [o for o in sorted(self.agents) if o != aid and o not in claimed and
                      (self.agents[o].state.location, self.agents[o].state.arena) == (a.state.location, a.state.arena)]
            if not others:
                continue
            probs = [talk_gate(a, self.agents[o], now, trng) for o in others]
            for o, p in zip(others, probs):
                if p > 0 and trng.random() < p:
                    cands.append((aid, o))
                    claimed |= {aid, o}
                    break
        if cands:
            res = self._parallel([(f"t{tick:04d}:04dtt:{p}:{q}",
                                   (lambda p=p, q=q: decide_to_talk(self.agents[p], self.agents[q], self.agents[p].stream("talk"))[0]))
                                  for p, q in cands])
            for (p, q), yes in zip(cands, res):
                if yes and p not in busy and q not in busy:
                    busy |= {p, q}
                    pairs.append((p, q, None, None))
        if not pairs and not groups_now:
            return extra_obs
        convs = []
        for i, (p, q, opening, trig) in enumerate(pairs):
            cid = f"d{day}t{tick:04d}c{i}"       # display id only; nothing is seeded from it
            self.agents[p].state.in_conversation = cid
            self.agents[q].state.in_conversation = cid
            self.agents[p].state.pre_chat_activity = self.agents[p].state.activity
            self.agents[q].state.pre_chat_activity = self.agents[q].state.activity
            self.agents[p].state.activity = f"chatting with {self.agents[q].profile.first_name}"
            self.agents[q].state.activity = f"chatting with {self.agents[p].profile.first_name}"
            self.agents[p].sync_scratch()
            self.agents[q].sync_scratch()
            convs.append((cid, p, q, opening, trig))
        gconvs = []
        for j, parts in enumerate(groups_now):
            cid = f"d{day}t{tick:04d}g{j}"
            for x in parts:
                ag = self.agents[x]
                ag.state.in_conversation = cid
                ag.state.pre_chat_activity = ag.state.activity
                ag.state.activity = "chatting with " + ", ".join(self.agents[y].profile.first_name for y in parts if y != x)
                ag.sync_scratch()
            gconvs.append((cid, parts))

        def bystanders(parts, where):
            # everyone else in the room, including people busy in another conversation there: they overhear
            # at overhear_prob x busy_factor (conversation.py, group_conversation.py)
            return [x for x in sorted(self.agents.values(), key=lambda z: z.id)
                    if x.id not in parts and (x.state.location, x.state.arena) == where]

        def conv_job(cid, p, q, opening, trig):
            a, b = self.agents[p], self.agents[q]
            return run_conversation(cid, a, b, bystanders({p, q}, (a.state.location, a.state.arena)),
                                    seed_rng(seed, "conv", tick, *sorted((p, q))), opening=opening, trigger=trig)

        def group_job(cid, parts):
            from backend.agents.group_conversation import run_group_conversation
            a = self.agents[parts[0]]
            return run_group_conversation(cid, [self.agents[x] for x in parts],
                                          bystanders(set(parts), (a.state.location, a.state.arena)),
                                          seed_rng(seed, "gconv", tick, *parts), topic="meal")
        results = self._parallel([(f"t{tick:04d}:05conv:{c[0]}", (lambda c=c: conv_job(*c))) for c in convs]
                                 + [(f"t{tick:04d}:05conv:{g[0]}", (lambda g=g: group_job(*g))) for g in gconvs])
        for conv in results:
            self.stats["conversations"] += 1
            self.stats["utterances"] += len(conv["utterances"])
            self.tick_utts.extend(conv["utterances"])
            for o in conv.pop("overheard"):
                extra_obs.setdefault(o.agent_id, []).append(o)
            # invitations (reactive re-planning); dyads only
            if len(conv["participants"]) != 2:
                continue
            p, q = conv["participants"]
            a, b = self.agents[p], self.agents[q]
            if conv["utterances"] and a.profile.rel(q).affinity >= 0.6 and \
                    seed_rng(seed, "invite", tick, p, q).random() < self.cfg["conversation"]["invite_prob"]:
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
                node = encode(a, o, a.stream("encode"))
                maybe_remind(a, o, node, a.stream("remind"))
                if self.cfg.get("need", {}).get("enabled"):
                    from backend.memory import need as NEED
                    NEED.note(a, o, node)
        ids = sorted(extra_obs)
        self._parallel([(f"t{tick:04d}:06heard:{aid}", (lambda aid=aid: job(aid))) for aid in ids])

    def _reflect(self, tick):
        ids = [aid for aid in sorted(self.agents) if should_reflect(self.agents[aid])]
        if not ids:
            return
        res = self._parallel([(f"t{tick:04d}:07reflect:{aid}", (lambda aid=aid: reflect(self.agents[aid], self.agents[aid].stream("reflect"))))
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
                    # routine plans are part of the WORLD (agreement a): seeded by world_seed, exactly as
                    # world_script plans them, so holding world_seed fixed holds every beat's place fixed
                    a.day_plan = plan_day(a, self.clock, seed_rng(self.world_seed, "plan", aid, day))
                    a.state.talks_today = 0
                    self.tracer.log("day_plan", agent=aid, day=day, plan=a.day_plan)
            for a in self.agents.values():
                a.set_time(now)
            self.ctx.mods.on_tick(tick, self.agents)
            with llm_scope(f"t{tick:04d}:00world"):
                self._release_events(tick)
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
        # sorted keys: the sidecar is filled from parallel jobs, so insertion order follows thread timing
        with open(self.run_dir / "memory_meta.json", "w") as fh:
            json.dump({nid: asdict(m) for nid, m in self.meta.meta.items()}, fh, sort_keys=True)
        self.tracer.close()
        self.frames_fh.close()
        self.events_fh.close()
        self.llm.close()
        self.pool.shutdown()
        self.write_manifest("finished", {"wall_seconds": round(seconds, 1),
                                         "manipulation": _manip(self.ctx.mods),
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
