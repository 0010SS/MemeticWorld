"""The simulation loop: observe -> retrieve memory -> decide -> move/talk -> store memory.

One `step()` is one tick of campus time:
  1. fire world events at their locations
  2. each agent perceives ONLY its own location (people, their visible activity, events)
  3. agents with something social going on ask the LLM for a decision; alone/unchanged agents
     follow their schedule on autopilot (saves most of the LLM budget)
  4. resolve moves, then pair up TALKs into short conversations (one LLM call per turn, each
     speaker sees only their own memories), then REACTs
  5. embed and persist the tick's memories, then log a `tick` snapshot for replay

Experimental conditions:
  full              speech is remembered verbatim (the channel phrases travel through)
  no_speech_memory  control: agents remember *that* they talked, never *what* was said
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass, field

from app.db import database as db
from app.llm.client import LLMClient
from app.llm import prompts
from app.memory.store import Memory, MemoryStore
from app.simulation.agent import Agent, Decision, Reply
from app.simulation.campus import GRAPH, LOCATIONS, resolve_location, world_layout
from app.simulation.events import TEMPLATES, EventScheduler, WorldEvent
from app.simulation.personas import PERSONAS
from app.simulation.scheduler import DAY_START, Clock, scheduled_slot, next_hop

CONDITIONS = {
    "full": "Agents remember what others said, word for word.",
    "no_speech_memory": "Control: agents remember that they talked, but not what was said.",
}


@dataclass
class SimConfig:
    days: int = 2
    tick_minutes: int = 15
    condition: str = "full"
    seed: int = 7
    max_turns: int = 4            # utterances per conversation, including the opener
    overhear_prob: float = 0.5    # chance a bystander at the same location overhears a conversation
    idle_decide_prob: float = 0.5 # chance an unchanged social scene still gets an LLM decision
    max_linger_ticks: int = 2     # ticks an agent may ignore its schedule before walking there anyway
    retrieval_k: int = 5
    tick_delay: float = 0.0       # seconds to pause between ticks (makes mock runs watchable live)
    agent_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; expected one of {list(CONDITIONS)}")


@dataclass
class Plan:
    """One agent's perception and choice for the current tick."""
    agent: Agent
    others: list[Agent]
    here_events: list[WorldEvent]
    scheduled: tuple[str, str]
    decision: Decision | None = None
    decision_id: int | None = None
    from_llm: bool = False
    query: str = ""
    memories: list[Memory] = field(default_factory=list)

    @property
    def off_schedule(self) -> bool:
        return self.scheduled[0] != self.agent.current_location


def _auto(action: str, target: str | None = None, activity: str | None = None) -> Decision:
    return Decision.model_construct(action=action, target=target, activity=activity, utterance=None,
                                    reason=None, importance=2)


def _mem_pairs(memories: list[Memory]) -> list[tuple[str, str]]:
    return [(m.sim_time, m.text) for m in memories]


class Simulation:
    def __init__(self, conn, run_id: int, cfg: SimConfig, llm: LLMClient):
        self.conn = conn
        self.run_id = run_id
        self.cfg = cfg
        self.llm = llm
        self.clock = Clock(cfg.tick_minutes)
        self.rng = random.Random(cfg.seed)
        personas = [p for p in PERSONAS if cfg.agent_ids is None or p["id"] in cfg.agent_ids]
        self.agents = [Agent.from_persona(p) for p in personas]
        self.by_id = {a.id: a for a in self.agents}
        self.memory = MemoryStore(conn, run_id, llm, cfg.tick_minutes)
        self.events = EventScheduler(self.clock, random.Random(cfg.seed + 1))
        self.tick = 0
        self.total_ticks = self.clock.total_ticks(cfg.days)
        self.stop_requested = False

    @property
    def speech_memory(self) -> bool:
        return self.cfg.condition == "full"

    def log(self, type: str, **kwargs) -> int:
        return db.insert_event(self.conn, self.run_id, self.tick, self.clock.label(self.tick), type, **kwargs)

    def remember(self, agent: Agent, text: str, importance: float, source_type: str,
                 source_event_id: int | None = None) -> None:
        self.memory.add(agent.id, self.tick, self.clock.label(self.tick), text, importance, source_type,
                        source_event_id)

    def agents_at(self, location: str) -> list[Agent]:
        return [a for a in self.agents if a.current_location == location]

    # --- main loop ------------------------------------------------------------------------

    async def run(self) -> str:
        db.update_run(self.conn, self.run_id, status="running")
        while self.tick < self.total_ticks and not self.stop_requested:
            await self.step()
            db.update_run(self.conn, self.run_id, current_tick=self.tick, stats=self.llm.stats)
            if self.cfg.tick_delay:
                await asyncio.sleep(self.cfg.tick_delay)
        status = "stopped" if self.stop_requested else "finished"
        self.log("run_end", data={"status": status})
        db.update_run(self.conn, self.run_id, status=status, stats=self.llm.stats)
        return status

    async def step(self) -> None:
        day, minute = self.clock.day_and_minute(self.tick)
        if self.clock.is_day_start(self.tick):
            self._start_day(day)

        present = {loc: [(a.id, a.name) for a in self.agents_at(loc)] for loc in LOCATIONS}
        for event in self.events.spawn(self.tick, present):
            event.event_id = self.log("world_event", agent_id=event.subject_id, location=event.location,
                                      text=event.text, data={"key": event.key, "topic": event.topic,
                                                             "until_tick": event.end_tick})
        active = self.events.active(self.tick)

        plans = [self._perceive(agent, active, minute) for agent in self.agents]
        llm_plans = [p for p in plans if p.from_llm]
        if llm_plans:
            queries = await self.llm.embed([p.query for p in llm_plans])
            for plan, query in zip(llm_plans, queries):
                plan.memories = self.memory.retrieve(plan.agent.id, query, self.tick, self.cfg.retrieval_k)
            await asyncio.gather(*(self._decide(p) for p in llm_plans))

        await self._resolve(plans, active)
        await self.memory.flush()
        self.log("tick", data={"day": day, "minute": minute,
                               "agents": {a.id: a.snapshot() for a in self.agents}})
        self.tick += 1

    def _start_day(self, day: int) -> None:
        for agent in self.agents:
            agent.current_location, agent.current_activity = scheduled_slot(agent.schedule, DAY_START)
            agent.linger_ticks, agent.last_seen = 0, frozenset()
        self.events.plan_day(self.tick)
        self.log("day_start", data={"day": day})

    # --- perception -----------------------------------------------------------------------

    def _perceive(self, agent: Agent, active: list[WorldEvent], minute: int) -> Plan:
        location = agent.current_location
        others = [o for o in self.agents_at(location) if o is not agent]
        here_events = [e for e in active if e.location == location]
        plan = Plan(agent, others, here_events, scheduled_slot(agent.schedule, minute))

        new_events = [e for e in here_events if agent.id not in e.seen_by]
        for event in new_events:
            event.seen_by.add(agent.id)
            self.remember(agent, f"At the {location}: {event.text}", 6, "event", event.event_id)
        for other in others:
            if other.id not in agent.last_seen:
                self.remember(agent, f"I saw {other.name} at the {location} ({other.current_activity}).", 2, "sighting")
        seen = frozenset(o.id for o in others)
        changed = seen != agent.last_seen or bool(new_events)
        agent.last_seen = seen
        agent.linger_ticks = agent.linger_ticks + 1 if plan.off_schedule else 0

        scheduled_location, scheduled_activity = plan.scheduled
        if not others and not new_events:
            plan.decision = (_auto("MOVE", scheduled_location) if plan.off_schedule
                             else _auto("CONTINUE_ACTIVITY", activity=scheduled_activity))
        elif plan.off_schedule and agent.linger_ticks > self.cfg.max_linger_ticks:
            plan.decision = _auto("MOVE", scheduled_location)
        elif not changed and not plan.off_schedule and self.rng.random() >= self.cfg.idle_decide_prob:
            plan.decision = _auto("CONTINUE_ACTIVITY", activity=scheduled_activity)
        else:
            plan.from_llm = True
            plan.query = " ".join([f"{location}.", *(o.name for o in others), *(e.text for e in here_events),
                                   agent.current_activity])
        return plan

    def _observation(self, agent: Agent, others: list[Agent], here_events: list[WorldEvent],
                     scheduled: tuple[str, str], memories: list[Memory] | None) -> str:
        location = agent.current_location
        return prompts.render_observation(
            time_label=self.clock.label(self.tick), location=location,
            location_description=LOCATIONS[location]["description"],
            activity=agent.current_activity or "nothing in particular",
            scheduled=scheduled if scheduled[0] != location else None,
            nearby=[(o.name, agent.relationship_to(o), o.current_activity or "hanging around") for o in others],
            happenings=[e.text for e in here_events],
            memories=None if memories is None else _mem_pairs(memories),
        )

    # --- decisions ------------------------------------------------------------------------

    async def _decide(self, plan: Plan) -> None:
        agent = plan.agent
        observation = self._observation(agent, plan.others, plan.here_events, plan.scheduled, plan.memories)
        user = prompts.render_decision(observation, GRAPH[agent.current_location], [o.name for o in plan.others])
        messages = [{"role": "system", "content": agent.system_prompt()}, {"role": "user", "content": user}]
        mock_ctx = {"kind": "decide", "traits": agent.traits, "nearby": [o.name for o in plan.others],
                    "topics": [e.topic for e in plan.here_events], "activity": agent.current_activity,
                    "scheduled_location": plan.scheduled[0] if plan.off_schedule else None,
                    "memories": [m.text for m in plan.memories]}
        decision, raw, error = await self.llm.chat_json(messages, Decision, mock_ctx)
        plan.decision_id = db.insert_decision(
            self.conn, self.run_id, self.tick, agent.id, "decide", f"{messages[0]['content']}\n\n---\n\n{user}",
            [m.id for m in plan.memories], raw, decision.model_dump() if decision else None, error)
        if decision is None:  # model failed twice; fall back to the schedule
            decision = (_auto("MOVE", plan.scheduled[0]) if plan.off_schedule
                        else _auto("CONTINUE_ACTIVITY", activity=plan.scheduled[1]))
        plan.decision = decision
        if decision.reason:
            agent.current_goal = decision.reason

    # --- resolution -----------------------------------------------------------------------

    def _find_person(self, name: str | None, speaker: Agent) -> Agent | None:
        if not name:
            return None
        wanted = name.strip().lower()
        for other in self.agents_at(speaker.current_location):
            if other is not speaker and (other.name.lower() == wanted or wanted.startswith(other.name.lower())):
                return other
        return None

    async def _resolve(self, plans: list[Plan], active: list[WorldEvent]) -> None:
        moved: set[str] = set()
        for plan in plans:
            agent, decision = plan.agent, plan.decision
            if plan.from_llm:
                self.log("action", agent_id=agent.id, location=agent.current_location, text=decision.reason,
                         data={"action": decision.action, "target": decision.target,
                               "activity": decision.activity, "decision_id": plan.decision_id})
            if decision.action == "MOVE" and self._move(plan):
                moved.add(agent.id)

        # Pair TALKs into conversations in random order; a busy or absent target turns it into a REACT.
        busy: set[str] = set()
        conversations: list[tuple[Plan, Agent]] = []
        reacts: list[Plan] = []
        for plan in self.rng.sample(plans, len(plans)):
            agent, decision = plan.agent, plan.decision
            if agent.id in moved or decision.action not in ("TALK", "REACT"):
                continue
            target = self._find_person(decision.target, agent) if decision.action == "TALK" else None
            if target and target.id not in busy and target.id not in moved and agent.id not in busy:
                busy.update({agent.id, target.id})
                conversations.append((plan, target))
            else:
                reacts.append(plan)

        for plan in plans:
            agent, decision = plan.agent, plan.decision
            if agent.id not in moved and agent.id not in busy and decision.action in ("CONTINUE_ACTIVITY", "IDLE"):
                agent.current_activity = decision.activity or (
                    "taking a break" if decision.action == "IDLE" else plan.scheduled[1])

        jobs = []
        for plan, target in conversations:
            bystanders = [o for o in self.agents_at(plan.agent.current_location)
                          if o is not plan.agent and o is not target]
            overhearers = [o for o in bystanders if self.rng.random() < self.cfg.overhear_prob]
            jobs.append(self._converse(plan, target, overhearers, active))
        await asyncio.gather(*jobs)

        for plan in reacts:
            if plan.agent.id in busy:
                continue
            agent = plan.agent
            audience = [o for o in self.agents_at(agent.current_location) if o is not agent]
            self._utterance(agent, audience, [], plan.decision.utterance, f"{self.tick}:{agent.id}:react", 0,
                            "react", plan.decision_id, [m.id for m in plan.memories], plan.decision.importance)
            if plan.decision.activity:
                agent.current_activity = plan.decision.activity

    def _move(self, plan: Plan) -> bool:
        agent, decision = plan.agent, plan.decision
        origin = agent.current_location
        destination = resolve_location(decision.target) or plan.scheduled[0]
        hop = next_hop(GRAPH, origin, destination)
        if hop is None:
            return False
        agent.current_location = hop
        if hop != destination:
            agent.current_activity = f"walking to the {destination}"
        elif destination == plan.scheduled[0]:
            agent.current_activity = plan.scheduled[1]
        else:
            agent.current_activity = decision.activity or f"hanging around the {destination}"
        self.log("move", agent_id=agent.id, location=hop,
                 data={"from": origin, "to": hop, "destination": destination, "auto": not plan.from_llm})
        return True

    # --- speech ---------------------------------------------------------------------------

    def _utterance(self, speaker: Agent, audience: list[Agent], overhearers: list[Agent], text: str,
                   conversation_id: str, turn: int, kind: str, decision_id: int | None,
                   memory_ids: list[int | None], importance: int = 4) -> int:
        location = speaker.current_location
        event_id = self.log("utterance", agent_id=speaker.id, location=location, text=text, data={
            "conversation_id": conversation_id, "turn": turn, "kind": kind,
            "audience": [a.id for a in audience], "overheard_by": [o.id for o in overhearers],
            "decision_id": decision_id, "memory_ids": [m for m in memory_ids if m is not None],
        })
        if not self.speech_memory:
            return event_id
        names = " and ".join(a.name for a in audience)
        if kind == "react":
            self.remember(speaker, f'I said out loud at the {location}: "{text}"', importance, "said", event_id)
            for listener in audience:
                self.remember(listener, f'{speaker.name} said out loud at the {location}: "{text}"', 4, "heard", event_id)
            return event_id
        self.remember(speaker, f'I said to {names} at the {location}: "{text}"', importance, "said", event_id)
        for listener in audience:
            self.remember(listener, f'{speaker.name} said to me at the {location}: "{text}"', 5, "heard", event_id)
        for bystander in overhearers:
            self.remember(bystander, f'I overheard {speaker.name} say to {names} at the {location}: "{text}"', 3,
                          "overheard", event_id)
        return event_id

    async def _converse(self, plan: Plan, target: Agent, overhearers: list[Agent], active: list[WorldEvent]) -> None:
        opener, location = plan.agent, plan.agent.current_location
        conversation_id = f"{self.tick}:{opener.id}:{target.id}"
        history: list[tuple[Agent, str]] = [(opener, plan.decision.utterance)]
        self._utterance(opener, [target], overhearers, plan.decision.utterance, conversation_id, 0, "talk",
                        plan.decision_id, [m.id for m in plan.memories], plan.decision.importance)
        opener.current_activity = f"talking with {target.name}"
        target.current_activity = f"talking with {opener.name}"

        speaker, listener = target, opener
        here_events = [e for e in active if e.location == location]
        for turn in range(1, self.cfg.max_turns):
            reply, decision_id, memories = await self._reply(speaker, listener, history, here_events)
            if reply is None or not reply.utterance:
                break
            self._utterance(speaker, [listener], overhearers, reply.utterance, conversation_id, turn, "talk",
                            decision_id, [m.id for m in memories])
            history.append((speaker, reply.utterance))
            if reply.end_conversation:
                break
            speaker, listener = listener, speaker

        opener.bump_closeness(target.id, 0.03)
        target.bump_closeness(opener.id, 0.03)
        if not self.speech_memory:  # control condition: remember the fact of the conversation only
            self.remember(opener, f"I talked with {target.name} at the {location}.", 4, "conversation")
            self.remember(target, f"I talked with {opener.name} at the {location}.", 4, "conversation")
            for bystander in overhearers:
                self.remember(bystander, f"I noticed {opener.name} and {target.name} talking at the {location}.", 2,
                              "conversation")

    async def _reply(self, speaker: Agent, partner: Agent, history: list[tuple[Agent, str]],
                     here_events: list[WorldEvent]) -> tuple[Reply | None, int, list[Memory]]:
        last_text = history[-1][1]
        query = (await self.llm.embed([f"{partner.name}: {last_text}"]))[0]
        memories = self.memory.retrieve(speaker.id, query, self.tick, self.cfg.retrieval_k)
        others = [o for o in self.agents_at(speaker.current_location) if o is not speaker]
        observation = self._observation(speaker, others, here_events,
                                        scheduled_slot(speaker.schedule, self.clock.day_and_minute(self.tick)[1]),
                                        None)
        user = prompts.render_reply(observation, partner.name, speaker.relationship_to(partner), _mem_pairs(memories),
                                    [("You" if who is speaker else who.name, text) for who, text in history])
        messages = [{"role": "system", "content": speaker.system_prompt()}, {"role": "user", "content": user}]
        mock_ctx = {"kind": "reply", "traits": speaker.traits, "partner": partner.name,
                    "history": [[who.name, text] for who, text in history], "memories": [m.text for m in memories]}
        reply, raw, error = await self.llm.chat_json(messages, Reply, mock_ctx)
        decision_id = db.insert_decision(
            self.conn, self.run_id, self.tick, speaker.id, "reply", f"{messages[0]['content']}\n\n---\n\n{user}",
            [m.id for m in memories], raw, reply.model_dump() if reply else None, error)
        return reply, decision_id, memories


def run_config_snapshot(cfg: SimConfig) -> dict:
    """What gets stored with a run so it can be replayed and analyzed later."""
    personas = [p for p in PERSONAS if cfg.agent_ids is None or p["id"] in cfg.agent_ids]
    return {"sim": asdict(cfg), "agents": personas, "world": world_layout(),
            "event_templates": [asdict(t) for t in TEMPLATES]}
