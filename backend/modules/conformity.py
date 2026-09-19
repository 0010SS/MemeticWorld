"""Conformity bias: memories corroborated by several *distinct* peers get more
retrieval weight (ordinary social learning; no meme variables).

For each candidate memory m, count the distinct speakers of the agent's own
conversation memories whose embedding is similar to m (cos >= sim_threshold).
score(m) += strength * log(1 + max(0, n_distinct - 1)).
Speaker attribution comes from the agent's own chat memories (who the agent
remembers talking with: the node's object, "A, B, C" for a group conversation),
not from the observer/analyzer.
"""
from __future__ import annotations

import math

import numpy as np

from backend.modules.base import AgentModifier


class Conformity(AgentModifier):
    name = "conformity"

    def modify_retrieval(self, agent, nodes, scores, focal):
        chats = agent.a_mem.seq_chat
        if not chats:
            return scores
        th = float(self.params.get("sim_threshold", 0.35))
        k = float(self.params.get("strength", 0.6))
        emb = agent.a_mem.embeddings
        C = np.array([emb[c.embedding_key] for c in chats])
        partners = [set(str(c.object).split(", ")) for c in chats]
        out = dict(scores)
        boosted = 0
        for n in nodes:
            v = np.array(emb[n.embedding_key])
            sims = C @ v
            peers = set().union(*(partners[i] for i in np.nonzero(sims >= th)[0]))
            if len(peers) > 1:
                out[n.node_id] = out[n.node_id] + k * math.log(len(peers))
                boosted += 1
        if boosted:
            self.count("nodes_boosted", boosted)
        return out
