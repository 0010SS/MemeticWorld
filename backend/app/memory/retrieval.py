"""Memory scoring: score = 0.6 * semantic similarity + 0.3 * recency + 0.1 * importance.

Recency decays exponentially in *awake* campus hours (ticks), so yesterday evening's memory
is ~14 hours old the next morning, not 24.
"""
from __future__ import annotations

import numpy as np

W_SIMILARITY, W_RECENCY, W_IMPORTANCE = 0.6, 0.3, 0.1
RECENCY_HALF_LIFE_HOURS = 8.0


def score_memories(embeddings: np.ndarray, ticks: np.ndarray, importance: np.ndarray, query: np.ndarray,
                   now_tick: int, tick_minutes: int) -> np.ndarray:
    similarity = np.clip(embeddings @ query, 0.0, 1.0)
    hours_ago = np.maximum(now_tick - ticks, 0) * tick_minutes / 60.0
    recency = 0.5 ** (hours_ago / RECENCY_HALF_LIFE_HOURS)
    return W_SIMILARITY * similarity + W_RECENCY * recency + W_IMPORTANCE * (importance / 10.0)


def top_k_indices(scores: np.ndarray, texts: list[str], k: int) -> list[int]:
    """Best-scoring indices, skipping exact-duplicate texts (keeps the highest-scoring copy)."""
    chosen: list[int] = []
    seen: set[str] = set()
    for i in np.argsort(-scores):
        if texts[i] in seen:
            continue
        seen.add(texts[i])
        chosen.append(int(i))
        if len(chosen) == k:
            break
    return chosen
