"""End-to-end with the mock LLM: simulate, log, remember, analyze."""
import asyncio

from app.analysis import pipeline
from app.db import database as db
from app.simulation import runner
from app.simulation.world import SimConfig


def _run(conn, condition):
    run_id = runner.create_run(conn, SimConfig(days=1, condition=condition, seed=3))
    assert asyncio.run(runner.execute_run(conn, run_id)) == "finished"
    return run_id


def test_full_run_logs_everything_and_detects(conn):
    run_id = _run(conn, "full")
    run = db.get_run(conn, run_id)
    assert run["status"] == "finished" and run["current_tick"] == run["total_ticks"] == 56
    types = {r["type"] for r in db.fetch_events(conn, run_id, limit=100000)}
    assert {"tick", "move", "utterance", "world_event", "day_start", "run_end"} <= types
    sources = {m["source_type"] for m in db.fetch_memories(conn, run_id, limit=100000)}
    assert {"heard", "said", "event", "sighting"} <= sources
    ticks = db.fetch_events(conn, run_id, types=["tick"], limit=1000)
    assert len(ticks) == 56 and len(ticks[0]["data"]["agents"]) == 8
    corpus, memes, _ = pipeline.analyze_run(conn, run_id)
    assert corpus.utterances and isinstance(memes, list)


def test_control_run_never_stores_speech(conn):
    run_id = _run(conn, "no_speech_memory")
    memories = db.fetch_memories(conn, run_id, limit=100000)
    assert not any(m["source_type"] in ("heard", "said", "overheard") for m in memories)
    assert not any('"' in m["text"] for m in memories)  # no quoted words anywhere
    assert any(m["source_type"] == "conversation" for m in memories)


def test_agents_only_see_their_own_location(conn):
    run_id = _run(conn, "full")
    ticks = {r["tick"]: r["data"]["agents"] for r in db.fetch_events(conn, run_id, types=["tick"], limit=1000)}
    rows = conn.execute("SELECT tick, agent_id, prompt FROM decisions WHERE run_id = ? AND kind = 'decide'",
                        (run_id,)).fetchall()
    assert rows
    for row in rows[:200]:
        before = ticks.get(row["tick"] - 1) if row["tick"] % 56 else None
        if not before:
            continue
        nearby = row["prompt"].split("Nearby:")[1].split("\n\n")[0]
        here = before[row["agent_id"]]["location"]
        for other, state in before.items():
            name = other.capitalize()
            if other != row["agent_id"] and state["location"] != here:
                assert f"- {name} (" not in nearby, f"{row['agent_id']} saw {name} at another location"
