"""Per-agent memory stream, persisted to SQLite with embeddings.

Memories are queued with `add()` during a tick and embedded + saved in one batch by `flush()`
at the end of the tick, so an agent never retrieves something from the tick it is in.

Speech memories keep the exact words in quotes (`Maya said to me at the Quad: "..."`).
That verbatim quote is the only channel through which a phrase can travel between agents,
so never paraphrase it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.llm.client import LLMClient
from app.memory.retrieval import score_memories, top_k_indices


@dataclass
class Memory:
    agent_id: str
    tick: int
    sim_time: str
    text: str
    importance: float
    source_type: str  # event | sighting | heard | overheard | said | conversation
    source_event_id: int | None = None
    id: int | None = None


class _AgentMemories:
    def __init__(self) -> None:
        self.items: list[Memory] = []
        self.embeddings: np.ndarray | None = None
        self.ticks = np.zeros(0)
        self.importance = np.zeros(0)


class MemoryStore:
    def __init__(self, conn, run_id: int, llm: LLMClient, tick_minutes: int):
        self.conn = conn
        self.run_id = run_id
        self.llm = llm
        self.tick_minutes = tick_minutes
        self.by_agent: dict[str, _AgentMemories] = {}
        self.pending: list[Memory] = []

    def add(self, agent_id: str, tick: int, sim_time: str, text: str, importance: float,
            source_type: str, source_event_id: int | None = None) -> None:
        importance = float(min(max(importance, 1), 10))
        self.pending.append(Memory(agent_id, tick, sim_time, text, importance, source_type, source_event_id))

    async def flush(self) -> int:
        if not self.pending:
            return 0
        batch, self.pending = self.pending, []
        vectors = await self.llm.embed([m.text for m in batch])
        self.conn.execute("BEGIN")
        try:
            for memory, vector in zip(batch, vectors):
                cur = self.conn.execute(
                    "INSERT INTO memories (run_id, agent_id, tick, sim_time, text, importance, source_type,"
                    " source_event_id, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.run_id, memory.agent_id, memory.tick, memory.sim_time, memory.text, memory.importance,
                     memory.source_type, memory.source_event_id, vector.astype(np.float32).tobytes()))
                memory.id = int(cur.lastrowid)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        for agent_id in {m.agent_id for m in batch}:
            rows = [i for i, m in enumerate(batch) if m.agent_id == agent_id]
            self._append(agent_id, [batch[i] for i in rows], vectors[rows].astype(np.float32))
        return len(batch)

    def _append(self, agent_id: str, memories: list[Memory], vectors: np.ndarray) -> None:
        mem = self.by_agent.setdefault(agent_id, _AgentMemories())
        mem.items += memories
        mem.embeddings = vectors if mem.embeddings is None else np.vstack([mem.embeddings, vectors])
        mem.ticks = np.append(mem.ticks, [m.tick for m in memories])
        mem.importance = np.append(mem.importance, [m.importance for m in memories])

    def retrieve(self, agent_id: str, query_vector: np.ndarray, now_tick: int, k: int = 5) -> list[Memory]:
        mem = self.by_agent.get(agent_id)
        if mem is None or mem.embeddings is None:
            return []
        scores = score_memories(mem.embeddings, mem.ticks, mem.importance, query_vector, now_tick, self.tick_minutes)
        indices = top_k_indices(scores, [m.text for m in mem.items], k)
        return sorted((mem.items[i] for i in indices), key=lambda m: m.tick)

    def count(self, agent_id: str) -> int:
        mem = self.by_agent.get(agent_id)
        return len(mem.items) if mem else 0
