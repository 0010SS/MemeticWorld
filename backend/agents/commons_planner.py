"""Choose a grounded intention from local observations and personal memory."""
from __future__ import annotations

import json

from backend.agents.ga_prompts import as_json
from backend.ga_compat import REPO_ROOT
from backend.llm.client import llm_purpose
from backend.memory.retrieval import merged_nodes, retrieve


INSTRUCTIONS = (REPO_ROOT / "backend/prompts/commons_action_v1.txt").read_text(encoding="utf-8")


def decide_commons(agent, view: dict, notices: list[str]) -> dict:
    focal = ("Choose useful work and coordinate with the cooperative. "
             + " ".join(notices) + " " + json.dumps(view["requests"][-4:]))
    results = retrieve(agent, [focal], rng=agent.rng)
    nodes = merged_nodes(results)
    # Keep recent feedback available even when a broad task query retrieves old memories.
    recent = sorted(agent.a_mem.all_nodes(), key=lambda n: (n.created, int(n.node_id.split(":m")[-1]))) [-8:]
    nodes = list({n.node_id: n for n in nodes + recent}.values())
    nearby = [agent.ctx.agents[p["id"]] for p in view["people"]]
    payload = {"identity": agent.iss(), "time": agent.scratch.curr_time.isoformat(),
               "view": view, "observations": notices,
               "relationships": [agent.relationship_line(p) for p in nearby],
               "memories": [n.description for n in nodes]}
    prompt = INSTRUCTIONS + "\n\nLOCAL_CONTEXT_JSON\n" + json.dumps(payload, ensure_ascii=False)
    with llm_purpose("commons_action", agent.id):
        raw = agent.ctx.llm.complete(prompt, max_tokens=900, temperature=0.7)
    parsed = as_json(raw)
    intention = parsed if isinstance(parsed, dict) else {"action": "INVALID", "reason": "Invalid JSON response"}
    retrieved = [n.node_id for n in nodes]
    agent.ctx.tracer.log("decision", agent=agent.id, kind="commons", prompt=prompt, response=raw,
                         decision=intention, retrieved=retrieved,
                         retrieval_scores={k: v for r in results.values() for k, v in r.scores.items()},
                         retrieved_event_ids=[])
    return {"intention": intention, "retrieved": retrieved}
