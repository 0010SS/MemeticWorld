import numpy as np

from app.memory.retrieval import score_memories, top_k_indices
from app.simulation.campus import GRAPH, resolve_location
from app.simulation.personas import PERSONAS
from app.simulation.scheduler import Clock, next_hop, scheduled_slot


def test_graph_is_symmetric_and_connected():
    for node, neighbors in GRAPH.items():
        for n in neighbors:
            assert node in GRAPH[n]
    for a in GRAPH:
        for b in GRAPH:
            if a != b:
                assert next_hop(GRAPH, a, b) is not None


def test_next_hop_and_resolve_location():
    assert next_hop(GRAPH, "Dorm", "Dorm") is None
    assert next_hop(GRAPH, "Dorm", "Quad") == "Quad"
    assert resolve_location("the library") == "Library"
    assert resolve_location("research lab") == "Research Lab"
    assert resolve_location("Mars") is None


def test_clock_and_schedules():
    clock = Clock(15)
    assert clock.ticks_per_day == 56
    assert clock.label(0) == "Day 1, 8:00 AM"
    assert clock.label(56) == "Day 2, 8:00 AM"
    maya = next(p for p in PERSONAS if p["id"] == "maya")
    assert scheduled_slot(maya["schedule"], 12 * 60 + 45)[0] == "Dining Hall"
    for persona in PERSONAS:
        for _, location, _ in persona["schedule"]:
            assert location in GRAPH, f"{persona['id']} has unknown location {location}"


def test_retrieval_prefers_relevant_recent_important():
    emb = np.eye(3, dtype=np.float32)
    query = np.array([1, 0, 0], dtype=np.float32)
    scores = score_memories(emb, np.array([10, 10, 0]), np.array([5, 5, 5]), query, now_tick=10, tick_minutes=15)
    assert scores.argmax() == 0
    assert top_k_indices(np.array([0.9, 0.8, 0.7]), ["a", "a", "b"], 2) == [0, 2]  # dedupes identical text
