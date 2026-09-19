"""Generative-Agents reflection, constrained.

Trigger: upstream importance counter (Scratch.importance_trigger_curr is
decremented by each new memory's importance; reflect when <= 0).
Focal points: upstream generate_focal_pt prompt, plus one constrained
pattern-seeking question (rotating) from the experiment spec. Insights:
upstream insight_and_evidence prompt over *stochastically* retrieved memories.
Nothing here asks the agent to name, label or coin anything.
"""
from __future__ import annotations

from backend.agents import ga_prompts
from backend.memory.retrieval import retrieve
from backend.memory.store import MemoryMeta, record_link

PATTERN_QUESTIONS = [
    "What patterns has {name} noticed recently?",
    "What has {name} learned about the people around {pron}?",
    "Have similar situations happened to {name} or others before?",
]


def should_reflect(agent) -> bool:
    return (agent.cfg["reflection"]["enabled"] and agent.scratch.importance_trigger_curr <= 0
            and len(agent.a_mem.id_to_node) > 3)


def reflect(agent, rng) -> list:
    ctx = agent.ctx
    rc = agent.cfg["reflection"]
    nodes = sorted([n for n in agent.a_mem.all_nodes() if "idle" not in n.embedding_key],
                   key=lambda n: (n.last_accessed, n.node_id))
    n_recent = max(5, min(30, agent.scratch.importance_ele_n))
    statements = "".join(n.embedding_key + "\n" for n in nodes[-n_recent:])
    focal = ga_prompts.focal_points(agent, statements, int(rc["n_focal_points"])) or []
    q = PATTERN_QUESTIONS[int(rng.integers(len(PATTERN_QUESTIONS)))]
    focal.append(q.format(name=agent.profile.first_name, pron="them"))
    created = []
    for fp in focal:
        res = retrieve(agent, [fp], k=8, rng=rng)[fp]
        if len(res.nodes) < 2:
            continue
        thoughts = ga_prompts.insights(agent, res.nodes, int(rc["n_insights"])) or {}
        for thought, evidence in thoughts.items():
            imp = ga_prompts.poignancy(agent, thought, "thought")
            emb = ctx.embed(thought)
            kws = {agent.name.lower()} | {w.lower() for w in thought.split() if len(w) > 5}
            node = agent.a_mem.add("thought", agent.scratch.curr_time, agent.name, "reflects", fp[:60],
                                   thought, set(sorted(kws)[:8]), imp, emb, evidence or [n.node_id for n in res.nodes])
            ev_ids = ctx.meta.events_of(evidence or [n.node_id for n in res.nodes])
            ctx.meta.set(node.node_id, MemoryMeta(agent_id=agent.id, source_type="reflection", salience=0.5,
                                                  originating_event_ids=ev_ids,
                                                  source_ids=list(evidence)))
            ctx.tracer.log("reflection", agent=agent.id, node_id=node.node_id, text=thought, focal_point=fp,
                           evidence=evidence, retrieved=[n.node_id for n in res.nodes],
                           retrieval_scores=res.scores, importance=imp, originating_event_ids=ev_ids)
            for eid in dict.fromkeys(evidence or []):     # a reflection citing evidence links it (v2 §2.3)
                record_link(agent, node.node_id, eid, "reflection", fp[:80])
            created.append(node)
    agent.scratch.importance_trigger_curr = agent.scratch.importance_trigger_max
    agent.scratch.importance_ele_n = 0
    return created
