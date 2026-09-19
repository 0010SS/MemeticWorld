"""Read-only access to a finished (or running) run directory for the OBSERVER layer."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import yaml


class RunData:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.manifest = json.load(open(self.dir / "manifest.json"))
        self.cfg = yaml.safe_load(open(self.dir / "config.resolved.yaml"))

    @cached_property
    def trace(self) -> list[dict]:
        with open(self.dir / "trace.jsonl") as f:
            return [json.loads(l) for l in f]

    def of(self, *types) -> list[dict]:
        t = set(types)
        return [r for r in self.trace if r["type"] in t]

    @cached_property
    def utterances(self) -> list[dict]:
        us = self.of("utterance")
        us.sort(key=lambda u: (u["tick"], u["id"]))
        return us

    @cached_property
    def utt_by_id(self) -> dict:
        return {u["id"]: u for u in self.utterances}

    @cached_property
    def conversations(self) -> dict:
        return {c["id"]: c for c in self.of("conversation")}

    @cached_property
    def events(self) -> dict:
        out = {}
        p = self.dir / "events.jsonl"
        if p.exists():
            for l in open(p):
                e = json.loads(l)
                out[e["id"]] = e
        return out

    @cached_property
    def agents(self) -> dict:
        return self.manifest["agents"]

    @cached_property
    def groups(self) -> dict:
        return self.manifest.get("groups", {})

    def groups_of(self, agent_id) -> set:
        return {g for g, m in self.groups.items() if agent_id in m}

    @cached_property
    def names(self) -> dict:
        return {aid: a["name"] for aid, a in self.agents.items()}

    @cached_property
    def world_text(self) -> str:
        """All surface text the WORLD produced (event facts, routines, profile text)."""
        parts = []
        for e in self.events.values():
            parts.append(e.get("narrative", ""))
        for a in self.agents.values():
            parts += [a.get("background", "")] + [r["activity"] for r in a.get("routine", [])]
            parts += a.get("habits", [])
        return " ".join(parts).lower()

    def conversation_context(self, u: dict, window: int = 1) -> str:
        c = self.conversations.get(u.get("conversation_id") or "")
        if not c:
            return u.get("context", "") + "\n" + u["text"]
        tr = c.get("transcript", [])
        i = u["idx"]
        lo, hi = max(0, i - window), min(len(tr), i + window + 1)
        return "\n".join(f"{s}: {t}" for s, t in tr[lo:hi])
