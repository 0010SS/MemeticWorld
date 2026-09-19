"""Memory stream = upstream GA `AssociativeMemory`, extended with forgetting.

Each node is an upstream `ConceptNode` (id, created, description, poignancy,
embedding_key, ...). Simulator-only metadata (salience, source type,
originating event ids, speakers) lives in a *separate* sidecar dict owned by the
simulator (`SimMemoryMeta`), never in the node and never in any prompt.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from backend import ga_compat


def _clean(text: str) -> str:
    # upstream add_event() rewrites descriptions containing "(" -- avoid that mangling
    return " ".join(text.replace("(", "- ").replace(")", "").split())


class MemoryStream(ga_compat.load().AssociativeMemory):
    def __init__(self, agent_id: str):  # noqa: D401 - bypass upstream file loading
        self.agent_id = agent_id
        self.id_to_node = dict()
        self.seq_event, self.seq_thought, self.seq_chat = [], [], []
        self.kw_to_event, self.kw_to_thought, self.kw_to_chat = dict(), dict(), dict()
        self.kw_strength_event, self.kw_strength_thought = dict(), dict()
        self.embeddings = dict()
        self._next = 0

    # upstream derives node ids from len(id_to_node); removal would then collide,
    # so we keep a monotonic counter and re-key the node right after insertion.
    def _rekey(self, node):
        self._next += 1
        new_id = f"{self.agent_id}:m{self._next}"
        del self.id_to_node[node.node_id]
        node.node_id = new_id
        self.id_to_node[new_id] = node
        return node

    def add(self, kind: str, created: dt.datetime, s: str, p: str, o: str, text: str,
            keywords: set, poignancy: float, embedding: list, filling=None):
        text = _clean(text)
        pair = (text, embedding)
        # upstream uses len(id_to_node)+1 for the id; make sure it is free
        placeholder = f"node_{len(self.id_to_node) + 1}"
        assert placeholder not in self.id_to_node
        adder = {"event": self.add_event, "thought": self.add_thought, "chat": self.add_chat}[kind]
        node = adder(created, None, s, p, o, text, keywords, poignancy, pair, filling or [])
        return self._rekey(node)

    def all_nodes(self):
        return self.seq_event + self.seq_thought + self.seq_chat

    def remove(self, node_id: str):
        node = self.id_to_node.pop(node_id, None)
        if node is None:
            return
        for seq in (self.seq_event, self.seq_thought, self.seq_chat):
            if node in seq:
                seq.remove(node)
        for idx in (self.kw_to_event, self.kw_to_thought, self.kw_to_chat):
            for kw in list(idx):
                if node in idx[kw]:
                    idx[kw].remove(node)
                    if not idx[kw]:
                        del idx[kw]
        if not any(n.embedding_key == node.embedding_key for n in self.id_to_node.values()):
            self.embeddings.pop(node.embedding_key, None)

    def save_ga(self, folder: Path):
        """Persist in upstream GA format (nodes.json keyed node_1..N)."""
        folder.mkdir(parents=True, exist_ok=True)
        nodes = sorted(self.id_to_node.values(), key=lambda n: int(n.node_id.split(":m")[-1]))
        out = {}
        for i, n in enumerate(nodes, 1):
            out[f"node_{i}"] = {
                "memeworld_id": n.node_id, "node_count": i, "type_count": n.type_count,
                "type": n.type, "depth": n.depth,
                "created": n.created.strftime("%Y-%m-%d %H:%M:%S"), "expiration": None,
                "last_accessed": n.last_accessed.strftime("%Y-%m-%d %H:%M:%S"),
                "subject": n.subject, "predicate": n.predicate, "object": n.object,
                "description": n.description, "embedding_key": n.embedding_key,
                "poignancy": n.poignancy, "keywords": sorted(n.keywords), "filling": n.filling,
            }
        # byte-reproducible files: keyword dicts follow set iteration order (PYTHONHASHSEED), so sort keys
        with open(folder / "nodes.json", "w") as fh:
            json.dump(out, fh, indent=1)
        with open(folder / "kw_strength.json", "w") as fh:
            json.dump({"kw_strength_event": self.kw_strength_event,
                       "kw_strength_thought": self.kw_strength_thought}, fh, sort_keys=True)
        with open(folder / "embeddings.json", "w") as fh:
            json.dump(self.embeddings, fh, sort_keys=True)

    @classmethod
    def load_ga(cls, agent_id: str, folder: Path) -> "MemoryStream":
        # temporary ids live in a separate namespace so they can never collide with the saved ids
        # (forgetting leaves gaps, e.g. saved "ethan:m41" vs. the 41st re-added node)
        ms = cls(agent_id + "~load")
        nodes = json.load(open(folder / "nodes.json"))
        emb = json.load(open(folder / "embeddings.json"))
        for key in sorted(nodes, key=lambda k: int(k.split("_")[1])):
            d = nodes[key]
            created = dt.datetime.strptime(d["created"], "%Y-%m-%d %H:%M:%S")
            node = ms.add(d["type"], created, d["subject"], d["predicate"], d["object"],
                          d["description"], set(d["keywords"]), d["poignancy"],
                          emb.get(d["embedding_key"], []), d["filling"])
            ms.id_to_node.pop(node.node_id)
            node.node_id = d.get("memeworld_id") or f"{agent_id}:m{int(key.split('_')[1])}"
            ms.id_to_node[node.node_id] = node
            node.last_accessed = dt.datetime.strptime(d.get("last_accessed", d["created"]), "%Y-%m-%d %H:%M:%S")
        ms.agent_id = agent_id
        ms._next = max([int(n.split(":m")[-1]) for n in ms.id_to_node] or [0])
        assert len(ms.id_to_node) == len(nodes) == len(ms.all_nodes())
        return ms


@dataclass
class MemoryMeta:
    """Simulator-side metadata for one memory node. NEVER shown to agents."""
    agent_id: str
    source_type: str                     # perception | conversation | overheard | reflection | reminding | seed | ambient
    salience: float = 0.5
    originating_event_ids: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)   # observation / utterance / memory ids
    links: list[dict] = field(default_factory=list)       # LINK: [{"from", "to", "mechanism", "reason", ...}]
    wordings: list[dict] = field(default_factory=list)    # WORDING: [{"phrase", "heard_from", "utterance_id", "tick", ...}]


class SimMemoryMeta:
    """Global sidecar: node_id -> MemoryMeta (owned by the engine, not by agents)."""

    def __init__(self):
        self.meta: dict[str, MemoryMeta] = {}

    def set(self, node_id: str, m: MemoryMeta):
        self.meta[node_id] = m

    def get(self, node_id: str) -> MemoryMeta | None:
        return self.meta.get(node_id)

    def events_of(self, node_ids) -> list[str]:
        out = []
        for nid in node_ids:
            m = self.meta.get(nid)
            if m:
                for e in m.originating_event_ids:
                    if e not in out:
                        out.append(e)
        return out


def record_link(agent, frm: str, to: str, mechanism: str, reason: str = "", *, holder: str | None = None,
                from_event_ids: list | None = None, **extra) -> dict:
    """LINK (ontology v2 §2.3): the agent connected two of its experiences (reminding, lens association,
    merge, reflection evidence). Stored on the `holder` node's sidecar meta (default: the `from` node) and
    traced as `memory_link` with both ends' originating events, so the observer can tell same-event from
    cross-event links without joining records. Simulator-side only; agents never see it."""
    ctx = agent.ctx
    link = {"from": frm, "to": to, "mechanism": mechanism, "reason": reason, **extra}
    m = ctx.meta.get(holder or frm)
    if m is not None:
        m.links.append(link)
    ctx.tracer.log("memory_link", agent=agent.id, **link,
                   from_event_ids=list(from_event_ids) if from_event_ids is not None else ctx.meta.events_of([frm]),
                   to_event_ids=ctx.meta.events_of([to]))
    return link


def recency_score(node, now: dt.datetime, decay_rate: float) -> float:
    hours = max(0.0, (now - node.last_accessed).total_seconds() / 3600.0)
    return math.exp(-decay_rate * hours)
