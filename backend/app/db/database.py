"""SQLite persistence. One file holds every run; the `events` table is the replay log.

All access goes through plain functions here. The API server and the CLI can both write
to the same file (WAL mode), so a run started from the CLI can be watched in the UI.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    condition TEXT NOT NULL,          -- 'full' | 'no_speech_memory'
    seed INTEGER NOT NULL,
    days INTEGER NOT NULL,
    status TEXT NOT NULL,             -- 'pending' | 'running' | 'finished' | 'stopped' | 'failed'
    current_tick INTEGER NOT NULL DEFAULT 0,
    total_ticks INTEGER NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,        -- agents, world and sim settings snapshot
    stats_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL          -- heartbeat: bumped every tick while running
);

-- Everything that happens, in order. The frontend replays this table.
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    sim_time TEXT NOT NULL,
    type TEXT NOT NULL,               -- tick | day_start | world_event | move | utterance | action | run_end
    agent_id TEXT,
    location TEXT,
    text TEXT,
    data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(run_id, type);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    tick INTEGER NOT NULL,
    sim_time TEXT NOT NULL,
    text TEXT NOT NULL,
    importance REAL NOT NULL,
    source_type TEXT NOT NULL,        -- event | sighting | heard | overheard | said | conversation
    source_event_id INTEGER,          -- utterance/world_event this memory came from (provenance)
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(run_id, agent_id, tick);

-- Every LLM decision/reply: what the agent saw, what it retrieved, what it answered.
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    kind TEXT NOT NULL,               -- decide | reply
    prompt TEXT NOT NULL,
    retrieved_memory_ids TEXT NOT NULL DEFAULT '[]',
    raw_output TEXT,
    parsed_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decisions(run_id, agent_id, tick);

CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    key TEXT PRIMARY KEY,
    vector BLOB NOT NULL
);
"""

_conn: sqlite3.Connection | None = None


def connect(path: str | None = None) -> sqlite3.Connection:
    db_path = Path(path or settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(SCHEMA)
    return conn


def get_conn() -> sqlite3.Connection:
    """Process-wide connection. The API only touches it from the event loop thread."""
    global _conn
    if _conn is None:
        _conn = connect()
    return _conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- runs -------------------------------------------------------------------------------

def create_run(conn, *, name: str, condition: str, seed: int, days: int, total_ticks: int,
               model: str, config: dict) -> int:
    cur = conn.execute(
        "INSERT INTO runs (name, condition, seed, days, status, total_ticks, model, config_json, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)",
        (name, condition, seed, days, total_ticks, model, json.dumps(config), now_iso(), now_iso()),
    )
    return int(cur.lastrowid)


def update_run(conn, run_id: int, **fields: Any) -> None:
    if "stats" in fields:
        fields["stats_json"] = json.dumps(fields.pop("stats"))
    fields["updated_at"] = now_iso()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id))


def _run_row(row: sqlite3.Row, with_config: bool) -> dict:
    run = dict(row)
    run["stats"] = json.loads(run.pop("stats_json") or "{}")
    config = json.loads(run.pop("config_json"))
    if with_config:
        run["config"] = config
    return run


def get_run(conn, run_id: int, with_config: bool = True) -> dict | None:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _run_row(row, with_config) if row else None


def list_runs(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return [_run_row(r, with_config=False) for r in rows]


def delete_run(conn, run_id: int) -> None:
    for table in ("events", "memories", "decisions"):
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))


# --- events -----------------------------------------------------------------------------

def insert_event(conn, run_id: int, tick: int, sim_time: str, type: str, *, agent_id: str | None = None,
                 location: str | None = None, text: str | None = None, data: dict | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO events (run_id, tick, sim_time, type, agent_id, location, text, data_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, tick, sim_time, type, agent_id, location, text, json.dumps(data or {})),
    )
    return int(cur.lastrowid)


def _event_row(row: sqlite3.Row) -> dict:
    event = dict(row)
    event["data"] = json.loads(event.pop("data_json") or "{}")
    return event


def fetch_events(conn, run_id: int, after_id: int = 0, limit: int = 5000,
                 types: list[str] | None = None) -> list[dict]:
    sql = "SELECT * FROM events WHERE run_id = ? AND id > ?"
    params: list[Any] = [run_id, after_id]
    if types:
        sql += f" AND type IN ({','.join('?' * len(types))})"
        params += types
    sql += " ORDER BY id LIMIT ?"
    params.append(limit)
    return [_event_row(r) for r in conn.execute(sql, params).fetchall()]


# --- memories & decisions ---------------------------------------------------------------

def fetch_memories(conn, run_id: int, agent_id: str | None = None, max_tick: int | None = None,
                   ids: list[int] | None = None, limit: int = 500) -> list[dict]:
    sql = "SELECT id, run_id, agent_id, tick, sim_time, text, importance, source_type, source_event_id" \
          " FROM memories WHERE run_id = ?"
    params: list[Any] = [run_id]
    if agent_id:
        sql += " AND agent_id = ?"
        params.append(agent_id)
    if max_tick is not None:
        sql += " AND tick <= ?"
        params.append(max_tick)
    if ids is not None:
        if not ids:
            return []
        sql += f" AND id IN ({','.join('?' * len(ids))})"
        params += ids
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def insert_decision(conn, run_id: int, tick: int, agent_id: str, kind: str, prompt: str,
                    retrieved_memory_ids: list[int], raw_output: str | None, parsed: dict | None,
                    error: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO decisions (run_id, tick, agent_id, kind, prompt, retrieved_memory_ids, raw_output,"
        " parsed_json, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, tick, agent_id, kind, prompt, json.dumps(retrieved_memory_ids), raw_output,
         json.dumps(parsed) if parsed is not None else None, error),
    )
    return int(cur.lastrowid)


def fetch_decisions(conn, run_id: int, agent_id: str, max_tick: int | None = None, limit: int = 20) -> list[dict]:
    sql = "SELECT * FROM decisions WHERE run_id = ? AND agent_id = ?"
    params: list[Any] = [run_id, agent_id]
    if max_tick is not None:
        sql += " AND tick <= ?"
        params.append(max_tick)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    out = []
    for r in conn.execute(sql, params).fetchall():
        d = dict(r)
        d["retrieved_memory_ids"] = json.loads(d["retrieved_memory_ids"])
        d["parsed"] = json.loads(d.pop("parsed_json")) if d.get("parsed_json") else None
        out.append(d)
    return out
