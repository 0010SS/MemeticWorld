"""Rebuildable evidence index over committed simulation history.

World truth remains available for audit but is never included in semantic judge packets.
Trace order is an archive position, not proof of causality between concurrent conversations.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml

from backend.research.common import digest, jsonl, lock, read_json, write_json

SCHEMA = 4
PRIVATE = {"memory_encoded", "memory_merged", "reflection", "reminding", "viewpoint",
           "observation", "job_decision", "record_write_decision", "record_read"}
HIDDEN = {"job_truth", "regime_active", "world_event_start", "event_beat"}
PUBLIC = {"utterance": "speech", "record_write": "record"}
TEXT_FIELDS = ("text", "thought", "insight", "description", "reason", "utterance", "narrative")


def event_text(row):
    parts = [str(row[k]) for k in TEXT_FIELDS if isinstance(row.get(k), str) and row[k].strip()]
    for fact in row.get("facts") or []:
        if isinstance(fact, dict):
            s = fact.get("perceived") or fact.get("text")
            if s:
                parts.append(str(s))
    return "\n".join(dict.fromkeys(parts))


def precedes(a, b):
    if a["tick"] != b["tick"]:
        return a["tick"] < b["tick"]
    return bool(a.get("conversation") and a.get("conversation") == b.get("conversation")
                and a.get("turn") is not None and b.get("turn") is not None and a["turn"] < b["turn"])


class Evidence:
    def __init__(self, run_dir):
        self.run = Path(run_dir).resolve()
        self.manifest = read_json(self.run / "manifest.json", {})
        self.cfg = yaml.safe_load((self.run / "config.resolved.yaml").read_text(encoding="utf-8"))
        self.tpd = int(self.manifest.get("ticks_per_day", 60))
        self.path = self.run / "index" / "evidence.sqlite"

    def build(self):
        with lock(self.run / "index" / ".index.lock"):
            return self._build()

    def _build(self):
        committed = list(jsonl(self.run / "digests.jsonl"))
        if committed:
            cutoff = max(int(r["tick"]) for r in committed)
        elif self.manifest.get("status") == "finished":
            cutoff = int(self.manifest.get("ticks", 0)) - 1
        else:
            last = self.manifest.get("last_complete_tick")
            cutoff = int(last) if last is not None else -1
        trace_path = self.run / "trace.jsonl"
        source = {"schema": SCHEMA, "cutoff": cutoff, "digests": committed,
                  "run_status": self.manifest.get("status"),
                  "profiles": digest(self.manifest.get("agents", {})),
                  "config": digest(self.cfg),
                  "size": trace_path.stat().st_size if trace_path.exists() else 0,
                  "mtime": trace_path.stat().st_mtime_ns if trace_path.exists() else 0}
        old = read_json(self.run / "index" / "manifest.json", {})
        if old.get("source") == source and self.path.exists():
            return old
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        try:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, tick INTEGER, seq INTEGER, kind TEXT, actor TEXT,
                    channel TEXT, conversation TEXT, text TEXT, data TEXT);
                CREATE TABLE IF NOT EXISTS exposures (
                    source TEXT, actor TEXT, tick INTEGER, via TEXT,
                    PRIMARY KEY(source, actor, tick, via));
                CREATE TABLE IF NOT EXISTS frames (tick INTEGER PRIMARY KEY, data TEXT);
                CREATE INDEX IF NOT EXISTS event_time ON events(tick);
                CREATE INDEX IF NOT EXISTS event_actor ON events(actor);
                DELETE FROM events; DELETE FROM exposures; DELETE FROM frames;
            """)
            n, corpus_hash, frame_hash = 0, [], []
            record_ids, reads, exposure_rows = {}, [], []
            for seq, row in enumerate(jsonl(trace_path)):
                tick = int(row.get("tick", 0))
                if tick > cutoff:
                    continue
                kind = row.get("type", "unknown")
                eid = str(row.get("id") or f"trace:{seq}")
                channel = PUBLIC.get(kind, "private" if kind in PRIVATE else "environment")
                if kind == "shared_background" or (kind == "memory_encoded" and row.get("source_type") == "seed"):
                    channel = "initial"
                if kind in HIDDEN:
                    channel = "hidden"
                actor = row.get("speaker") or row.get("agent") or row.get("operator") or row.get("actor")
                text = event_text(row) if channel != "hidden" else ""
                if kind == "record_write" and row.get("choice") == "none":
                    text = ""
                if kind == "exposure":
                    text = ""  # receipt for another event, not a second cultural expression
                event = {"id": eid, "tick": tick, "day": tick // self.tpd + 1,
                         "seq": seq, "kind": kind, "actor": actor, "channel": channel, "text": text,
                         "time": row.get("time"), "conversation": row.get("conversation_id"),
                         "turn": row.get("idx"), "listeners": row.get("listeners") or [],
                         "source_type": row.get("source_type"), "record_id": row.get("rev_id") or row.get("entry_id"),
                          "source_ids": row.get("utterance_ids") or row.get("originating_event_ids") or [],
                          "node_id": row.get("node_id"), "observation_id": row.get("observation_id"),
                          "memory_refs": row.get("retrieved") or row.get("evidence") or [],
                         "raw": row}
                db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                           (eid, tick, seq, kind, actor, channel, event["conversation"], text,
                            json.dumps(event, ensure_ascii=False)))
                if event["record_id"]:
                    record_ids[event["record_id"]] = eid
                if kind == "record_read":
                    reads.append(row)
                if kind == "utterance":
                    exposure_rows += [(eid, a, tick, "heard") for a in event["listeners"] if a != actor]
                if kind == "exposure":
                    exposure_rows += [(row.get("utterance_id"), a, tick, "heard")
                                      for a in row.get("listener_ids") or []]
                corpus_hash.append(digest(row))
                n += 1
            for row in reads:
                for rid in (row.get("rev_ids") or []) + (row.get("entry_ids") or []):
                    if rid in record_ids:
                        exposure_rows.append((record_ids[rid], row["agent"], int(row["tick"]), "read"))
            db.executemany("INSERT OR IGNORE INTO exposures VALUES (?,?,?,?)",
                           [r for r in exposure_rows if r[0] and r[1]])
            arrivals = {}
            for frame in jsonl(self.run / "frames.jsonl"):
                if int(frame["tick"]) <= cutoff:
                    frame_hash.append(digest(frame))
                    for aid, state in frame.get("agents", {}).items():
                        if state.get("active", True) and state.get("activity") != "departed":
                            arrivals.setdefault(aid, int(frame["tick"]))
                    db.execute("INSERT OR REPLACE INTO frames VALUES (?,?)",
                               (int(frame["tick"]), json.dumps(frame, ensure_ascii=False)))
            # Initial persona material is origin evidence; it is separate from produced culture.
            for aid, profile in sorted(self.manifest.get("agents", {}).items()):
                if cutoff < 0:
                    continue
                if arrivals and aid not in arrivals:
                    continue
                tick = arrivals.get(aid, 0)
                tick = -1 if tick == 0 else tick
                text = "\n".join(str(profile.get(k, "")) for k in ("background", "habits"))
                e = {"id": f"initial:{aid}", "tick": tick, "day": tick // self.tpd + 1, "seq": -1, "kind": "initial",
                     "actor": aid, "channel": "initial", "text": text, "listeners": [],
                     "conversation": None, "turn": None, "raw": {}}
                db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                            (e["id"], tick, -1, "initial", aid, "initial", None, text, json.dumps(e)))
                corpus_hash.append(digest(e))
            db.commit()
            meta = {"source": source, "fingerprint": digest([corpus_hash, frame_hash, source["config"], source["run_status"]]), "events": n,
                    "through_tick": cutoff, "run_status": self.manifest.get("status"),
                    "schema": SCHEMA}
            write_json(self.run / "index" / "manifest.json", meta)
            return meta
        finally:
            db.close()

    def events(self, *, ids=None, start=None, end=None, actor=None, channels=None, text_only=False):
        terms, args = [], []
        for col, val, op in (("tick", start, ">="), ("tick", end, "<="), ("actor", actor, "=")):
            if val is not None:
                terms.append(f"{col}{op}?")
                args.append(val)
        for col, vals in (("id", ids), ("channel", channels)):
            if vals is not None:
                vals = list(vals)
                if not vals:
                    return []
                terms.append(f"{col} IN ({','.join('?' for _ in vals)})")
                args.extend(vals)
        if text_only:
            terms.append("text<>''")
        query = "SELECT data FROM events" + (" WHERE " + " AND ".join(terms) if terms else "") + " ORDER BY tick,seq,id"
        with sqlite3.connect(self.path) as db:
            return [json.loads(r[0]) for r in db.execute(query, args)]

    def search(self, question, limit=80, **scope):
        terms = set(re.findall(r"\w+", question.lower()))
        events = self.events(text_only=True, **scope)
        ranked = sorted(events, key=lambda e: (-len(terms & set(re.findall(r"\w+", e["text"].lower()))), e["tick"], e["seq"]))
        return ranked[:limit]

    def exposures(self):
        with sqlite3.connect(self.path) as db:
            return [{"source": s, "actor": a, "tick": t, "via": v}
                    for s, a, t, v in db.execute("SELECT source,actor,tick,via FROM exposures ORDER BY tick,source,actor")]

    def activity(self):
        with sqlite3.connect(self.path) as db:
            counts = dict(db.execute("SELECT kind,COUNT(*) FROM events GROUP BY kind ORDER BY kind"))
        return {"event_counts": counts, "active_population_by_day": self.populations(),
                "recorded_manipulations": self.manifest.get("manipulation", {}),
                "agent_model_calls": self.manifest.get("stats", {}).get("llm", {}),
                "interpretation": "Configured opportunities are not proof that mechanisms operated. Inspect realized events and exposure receipts."}

    def populations(self):
        days = {}
        with sqlite3.connect(self.path) as db:
            for tick, raw in db.execute("SELECT tick,data FROM frames ORDER BY tick"):
                frame = json.loads(raw)
                active = {a for a, state in frame.get("agents", {}).items()
                          if state.get("active", True) and state.get("activity") != "departed"}
                days.setdefault(tick // self.tpd + 1, set()).update(active)
        return {d: sorted(agents) for d, agents in days.items()}


def packets(events, max_chars=18000, overlap=3):
    """Cover every textual event; large events are split into exact, citable substrings."""
    packet, length, fresh = [], 0, []
    for original in events:
        text = original["text"]
        for offset in range(0, len(text), max(500, max_chars // 2)):
            e = {k: v for k, v in original.items() if k != "raw"}
            e["text"] = text[offset:offset + max(500, max_chars // 2)]
            size = len(json.dumps(e, ensure_ascii=False))
            if fresh and length + size > max_chars:
                yield packet, fresh
                packet = packet[-overlap:] if overlap else []
                while packet and sum(len(json.dumps(p, ensure_ascii=False)) for p in packet) + size > max_chars:
                    packet.pop(0)
                length, fresh = sum(len(json.dumps(p)) for p in packet), []
            packet.append(e)
            fresh.append(e["id"])
            length += size
    if fresh:
        yield packet, fresh
