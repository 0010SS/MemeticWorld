"""Reaction decision after perceiving something salient (structured output).

observe -> retrieve -> (reflect handled by the engine) -> decide -> act.
Routine behaviour is deterministic (scheduler.py); this LLM decision is only
invoked when an agent notices a salient event.
"""
from __future__ import annotations

from backend import ga_compat
from backend.agents.ga_prompts import as_json
from backend.llm.client import llm_purpose
from backend.memory.retrieval import merged_nodes, retrieve
from backend.simulation import reference
from backend.simulation.world import WORLD_GRAPH

ga = ga_compat.load()
PROMPT = str(ga_compat.REPO_ROOT / "backend" / "prompts" / "react_v1.txt")
ACTIONS = {"CONTINUE", "REACT", "TALK", "MOVE"}


def decide_reaction(agent, obs, present: list, rng) -> dict:
    ctx = agent.ctx
    focal = obs.text()
    res = retrieve(agent, [focal], rng=rng)
    nodes = merged_nodes(res)
    mem = "".join(f"- {n.description}\n" for n in nodes) or "- (nothing in particular)\n"
    people = ", ".join(f"{p.name}" for p in present) or "nobody"
    lines = [agent.relationship_line(p) for p in present]
    lines = agent.mods.modify_prompt(agent, "react", lines)
    # Where the agent is, and the places it may move to, in the run's reference mode: canonical mode renders
    # "Dining Hall (Main Floor)" and the old sorted key list; situated mode offers descriptions, which
    # `reference.resolve` maps back to the canonical id below (either form is accepted).
    prompt = ga.gs.generate_prompt(
        [agent.iss(), agent.scratch.curr_time.strftime("%A %H:%M"), agent.name,
         reference.describe(agent.state.location, agent.state.arena, agent=agent, cfg=agent.cfg),
         agent.state.activity, people,
         "\n".join(lines), focal, mem, reference.options(sorted(WORLD_GRAPH), agent=agent, cfg=agent.cfg)],
        PROMPT)
    with llm_purpose("react_decision", agent.id):
        raw = ctx.llm.complete(prompt, max_tokens=200, temperature=0.8)
    d = as_json(raw) or {}
    action = str(d.get("action", "CONTINUE")).upper().split()[0].strip("|") if d.get("action") else "CONTINUE"
    if action not in ACTIONS:
        action = "CONTINUE"
    target = d.get("target")
    utt = d.get("utterance")
    if isinstance(utt, str) and utt.strip().lower() in ("", "null", "none"):
        utt = None
    decision = {"action": action, "target": target if isinstance(target, str) else None,
                "utterance": utt.strip().strip('"') if isinstance(utt, str) else None,
                "reason": d.get("reason")}
    if action == "TALK":
        names = {p.name.lower(): p for p in present} | {p.profile.first_name.lower(): p for p in present}
        t = names.get((decision["target"] or "").lower().strip())
        decision["target_id"] = t.id if t else None
        if not t:
            decision["action"] = "REACT" if decision["utterance"] else "CONTINUE"
    if action == "MOVE":
        # The answer may come back as the canonical id or as the situated description it was offered; both
        # resolve to the canonical id, which is what the engine moves the agent to. `target_said` keeps the
        # agent's own wording for the trace.
        canonical = reference.resolve(decision["target"], agent=agent, cfg=agent.cfg)
        if canonical is not None and canonical != decision["target"]:
            decision["target_said"] = decision["target"]     # the agent's own wording, for the trace
        decision["target"] = canonical
        if canonical is None:
            decision["action"] = "CONTINUE"
    ctx.tracer.log("decision", agent=agent.id, kind="react", observation_id=obs.id, prompt=prompt, response=raw,
                   decision=dict(decision), retrieved=[n.node_id for n in nodes],
                   retrieval_scores={k: v for r in res.values() for k, v in r.scores.items()},
                   retrieved_event_ids=ctx.meta.events_of([n.node_id for n in nodes]))
    decision["retrieved"] = [n.node_id for n in nodes]
    return decision
