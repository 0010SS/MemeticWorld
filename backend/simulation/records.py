"""The co-op binder (ontology v3 §2): WORLD state only.

One binder hangs next to the laser cutter. It has a front page (whole-text rewrites; every revision is kept)
and an append-only, dated, signed log. The world adds nothing but author and time stamps: every word in it was
written by an agent. World-side read receipts record who first read which entry or revision when; they are never
shown to agents.

Transitions (`records.transitions`, the manipulation) are applied in the day-start hook:
  keep - the binder is unchanged;
  wipe - the current binder is moved to `archived` (still world state and still in checkpoints, never readable
         by agents) and a fresh empty binder replaces it.
Every arm gets exactly one matched world fact on a transition day (TRANSITION_FACT).

Rendering (`view`) has one format in every arm: author first name and day/time are always shown; an empty binder
always renders EMPTY_TEXT, whatever the reason it is empty; no job ids, class/cause ids or other hidden fields
ever appear in it.

Layering: this module holds and renders world state and writes its own trace records (`record_read`,
`record_write`, `record_transition`). It never touches agent memory; building the reader's observation is a pure
function of the rendered view, and the author's self memory lives in backend/agents/record_write.py.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Callable

HEADER = "The co-op binder next to the laser cutter."
EMPTY_TEXT = "There is nothing written in the binder yet."
RECORD_SALIENCE = 0.5
TRANSITION_FACT = {
    "keep": "Someone put a new cover on the co-op binder by the laser cutter; all the old pages are still in it.",
    "wipe": ("Someone replaced the co-op binder by the laser cutter with a new, empty one; the old pages were "
             "boxed up and archived."),
}
TRANSITION_TIME = "08:45"
OFFER_KEYS = {"job": "faulted_job", "tally": "tally", "farewell": "farewell"}   # offer -> records.write_after item

DEFAULTS = {"enabled": False, "display": "current", "history_from_day": None, "consult": "always",
            "consult_from_day": None, "view_log_n": 5, "front_max_chars": 600, "log_max_chars": 200,
            "revisable": True, "retrieval": "recent", "encode_on_read": "new_only",
            "write_after": ["faulted_job", "tally", "farewell"], "authority": "members", "replies": False,
            "transitions": []}


def rcfg(cfg: dict | None) -> dict:
    """The `records` block with contract defaults (§8.2) for any missing key."""
    return {**DEFAULTS, **((cfg or {}).get("records") or {})}


def enabled(cfg: dict | None) -> bool:
    return bool(rcfg(cfg)["enabled"])


def display_mode(cfg: dict | None, day: int) -> str:
    """What agents see on `day`: none | current | history (history only from `history_from_day`, if set)."""
    rc = rcfg(cfg)
    if not rc["enabled"]:
        return "none"
    mode = rc["display"]
    if mode == "history" and rc["history_from_day"] is not None and day < int(rc["history_from_day"]):
        return "current"
    return mode


def consult_mode(cfg: dict | None, day: int) -> str:
    """How the operator's job decision uses the binder on `day`: always | on_choice | never. Before
    `consult_from_day` (if set) the base behaviour `always` applies."""
    rc = rcfg(cfg)
    if not rc["enabled"] or display_mode(cfg, day) == "none":
        return "never"
    if rc["consult_from_day"] is not None and day < int(rc["consult_from_day"]):
        return "always"
    return rc["consult"]


def offer_enabled(cfg: dict | None, offer: str) -> bool:
    """offer: job | tally | farewell (mapped onto `records.write_after`: faulted_job | tally | farewell)."""
    rc = rcfg(cfg)
    return bool(rc["enabled"]) and OFFER_KEYS[offer] in (rc["write_after"] or [])


def truncate_words(text: str, limit: int) -> tuple[str, bool]:
    """Cut `text` to at most `limit` characters at a word boundary. -> (text, truncated)"""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text, False
    cut = text[:limit + 1]
    sp = cut.rfind(" ")
    out = (cut[:sp] if sp > 0 else text[:limit]).rstrip(" ,;:-")
    return out, True


@dataclass
class Revision:
    rev_id: str
    author: str                     # agent id (world side)
    author_name: str                # first name as signed, captured at write time
    tick: int
    text: str
    revision_of: str | None = None


@dataclass
class Entry:
    entry_id: str
    author: str
    author_name: str
    tick: int
    text: str
    context: str = "job"            # job | tally | farewell
    job_id: str | None = None       # hidden; world side only, never rendered


@dataclass
class View:
    """One rendering of the binder: the exact text an agent sees, and which items it showed."""
    mode: str
    text: str
    rev_ids: list[str] = field(default_factory=list)       # current front first, then superseded versions shown
    entry_ids: list[str] = field(default_factory=list)     # newest first, as displayed
    lines: dict[str, str] = field(default_factory=dict)    # item id -> its displayed form (one line)
    authors: dict[str, str] = field(default_factory=dict)  # item id -> author agent id (world side)

    @property
    def ids(self) -> list[str]:
        return list(self.rev_ids) + list(self.entry_ids)

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


def _default_time_of(tick: int) -> dt.datetime:          # tests only; the engine passes clock.time_of
    return dt.datetime(2026, 9, 14, 7, 0) + dt.timedelta(minutes=15 * tick)


class Binder:
    """The binder by the laser cutter. The object is the persistent world entity; a wipe archives its current
    pages into `archived` and gives it a fresh `binder_id` (so references held by the engine stay valid)."""

    def __init__(self, binder_id: str = "binder-1", created_tick: int = 0,
                 time_of: Callable[[int], dt.datetime] | None = None, authority: str = "members"):
        self.binder_id = binder_id
        self.created_tick = created_tick
        self.front: list[Revision] = []
        self.log: list[Entry] = []
        self.archived: list[dict] = []           # [{binder_id, created_tick, archived_tick, front, log}]
        self.receipts: dict[str, dict[str, int]] = {}   # agent -> {entry_or_rev_id: first_read_tick}
        self.time_of = time_of or _default_time_of
        self.authority = authority
        self._seq = {"rev": 0, "entry": 0, "binder": int(binder_id.rsplit("-", 1)[-1]) if binder_id[-1:].isdigit() else 1}

    @classmethod
    def for_sim(cls, cfg: dict, clock) -> "Binder":
        rc = rcfg(cfg)
        if rc["replies"]:
            raise NotImplementedError("records.replies (repair through the record) is deferred (§8.4)")
        return cls(time_of=clock.time_of, authority=rc["authority"])

    # ------------------------------------------------------------------ writing
    def append(self, author: str, author_name: str, tick: int, text: str, context: str = "job",
               job_id: str | None = None, max_chars: int = 200) -> tuple[Entry, bool]:
        text, truncated = truncate_words(text, max_chars)
        self._seq["entry"] += 1
        e = Entry(f"e{self._seq['entry']}", author, author_name, tick, text, context, job_id)
        self.log.append(e)
        self._mark(author, [e.entry_id], tick)     # the author knows what they wrote: never "new" to them
        return e, truncated

    def rewrite(self, author: str, author_name: str, tick: int, text: str,
                max_chars: int = 600) -> tuple[Revision, bool]:
        """Replace the whole front page. A rewrite made from an older view still becomes the newest revision;
        every earlier revision is kept."""
        text, truncated = truncate_words(text, max_chars)
        self._seq["rev"] += 1
        prev = self.front[-1].rev_id if self.front else None
        r = Revision(f"r{self._seq['rev']}", author, author_name, tick, text, prev)
        self.front.append(r)
        self._mark(author, [r.rev_id], tick)
        return r, truncated

    # ------------------------------------------------------------------ transitions
    def transition(self, mode: str, tick: int) -> str | None:
        """keep: unchanged. wipe: archive the current pages, start a fresh empty binder. -> archived binder id"""
        if mode == "keep":
            return None
        if mode != "wipe":
            raise ValueError(f"unknown binder transition {mode!r}")
        old = self.binder_id
        self.archived.append({"binder_id": old, "created_tick": self.created_tick, "archived_tick": tick,
                              "front": [asdict(r) for r in self.front], "log": [asdict(e) for e in self.log]})
        self._seq["binder"] += 1
        self.binder_id = f"binder-{self._seq['binder']}"
        self.created_tick = tick
        self.front, self.log = [], []
        return old

    # ------------------------------------------------------------------ receipts
    def new_for(self, agent: str, ids=None) -> list[str]:
        """Items (default: everything currently readable) that `agent` has not read before, in the given order."""
        if ids is None:
            ids = [r.rev_id for r in reversed(self.front)] + [e.entry_id for e in reversed(self.log)]
        seen = self.receipts.get(agent, {})
        return [i for i in ids if i not in seen]

    def _mark(self, agent: str, ids, tick: int):
        seen = self.receipts.setdefault(agent, {})
        for i in ids:
            seen.setdefault(i, tick)

    mark_read = _mark

    # ------------------------------------------------------------------ rendering
    def _when(self, tick: int) -> str:
        return self.time_of(tick).strftime("%a %H:%M")

    def is_empty(self) -> bool:
        return not self.front and not self.log

    def render(self, mode: str = "current", n: int = 5, *, history: int = 2, retrieval: str = "recent",
               query: str | None = None, embed: Callable | None = None) -> View:
        """The binder as a reader sees it. mode: none | current | history."""
        if mode == "none":
            return View(mode, "")
        if mode not in ("current", "history"):
            raise ValueError(f"unknown binder display {mode!r}")
        if self.is_empty():
            return View(mode, f"{HEADER}\n{EMPTY_TEXT}")
        v = View(mode, "")
        out = [HEADER]
        if self.front:
            cur = self.front[-1]
            head = ("Front page (kept by the shop manager)" if self.authority == "manager_signed"
                    else f"Front page (last rewritten {self._when(cur.tick)} by {cur.author_name})")
            out += [f"{head}:", f'  "{cur.text}"']
            v.rev_ids.append(cur.rev_id)
            v.lines[cur.rev_id] = f'{head}: "{cur.text}"'
            v.authors[cur.rev_id] = cur.author
            if mode == "history":
                for older, newer in list(zip(self.front[:-1], self.front[1:]))[::-1][:history]:
                    h = f"Earlier front page ({self._when(older.tick)}, {older.author_name}), replaced {self._when(newer.tick)}"
                    out += [f"{h}:", f'  "{older.text}"']
                    v.rev_ids.append(older.rev_id)
                    v.lines[older.rev_id] = f'{h}: "{older.text}"'
                    v.authors[older.rev_id] = older.author
        else:
            out.append("Front page: nothing written on it yet.")
        notes = self._select_log(n, retrieval, query, embed)
        if notes:
            out.append("Log, newest first:")
            for e in notes:
                line = f"[{self._when(e.tick)}, {e.author_name}] {e.text}"
                out.append(f"  {line}")
                v.entry_ids.append(e.entry_id)
                v.lines[e.entry_id] = line
                v.authors[e.entry_id] = e.author
        else:
            out.append("Log: no notes yet.")
        v.text = "\n".join(out)
        return v

    def view(self, mode: str = "current", n: int = 5, **kw) -> str:
        return self.render(mode, n, **kw).text

    def _select_log(self, n: int, retrieval: str, query, embed) -> list[Entry]:
        order = list(reversed(self.log))                     # newest first (ties: later entries first)
        if n <= 0:
            return []
        if retrieval == "recent" or not query or embed is None:
            return order[:n]
        if retrieval != "relevant":
            raise ValueError(f"unknown binder retrieval {retrieval!r}")
        from backend.llm.embeddings import cos
        q = embed(query)
        rank = {e.entry_id: i for i, e in enumerate(order)}
        best = sorted(order, key=lambda e: (-cos(q, embed(e.text)), rank[e.entry_id]))[:n]
        return sorted(best, key=lambda e: rank[e.entry_id])    # displayed newest first

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        """binder.json for checkpoints: all revisions, archived binders, receipts (deterministic key order)."""
        return {"binder_id": self.binder_id, "created_tick": self.created_tick, "authority": self.authority,
                "front": [asdict(r) for r in self.front], "log": [asdict(e) for e in self.log],
                "archived": copy.deepcopy(self.archived),
                "receipts": {a: dict(sorted(r.items())) for a, r in sorted(self.receipts.items())},
                "seq": dict(self._seq)}

    @classmethod
    def from_dict(cls, d: dict, time_of=None) -> "Binder":
        b = cls(d["binder_id"], d["created_tick"], time_of=time_of, authority=d.get("authority", "members"))
        b.front = [Revision(**r) for r in d["front"]]
        b.log = [Entry(**e) for e in d["log"]]
        b.archived = copy.deepcopy(d.get("archived", []))
        b.receipts = {a: dict(r) for a, r in d.get("receipts", {}).items()}
        b._seq = dict(d.get("seq", b._seq))
        return b

    def frame(self, n: int = 5) -> dict:
        """Compact public view for the frame record / UI (no job ids)."""
        cur = self.front[-1] if self.front else None
        return {"binder_id": self.binder_id,
                "front": ({"author": cur.author_name, "when": self._when(cur.tick), "text": cur.text,
                           "revisions": len(self.front)} if cur else None),
                "log": [{"author": e.author_name, "when": self._when(e.tick), "text": e.text}
                        for e in list(reversed(self.log))[:n]],
                "archived": len(self.archived)}


# ---------------------------------------------------------------------------------------- engine-facing hooks
def render_for(binder: Binder, cfg: dict, day: int, *, query: str | None = None, embed=None) -> View:
    """The view configured for `day` (display mode, view_log_n, retrieval)."""
    rc = rcfg(cfg)
    return binder.render(display_mode(cfg, day), int(rc["view_log_n"]), retrieval=rc["retrieval"],
                         query=query, embed=embed)


def read(binder: Binder, cfg: dict, tracer, agent_id: str, tick: int, day: int, context: str, *,
         location: str = "Makerspace", arena: str = "", query: str | None = None, embed=None):
    """An agent reads the binder (context: decision | tally). Marks world-side receipts, traces `record_read`,
    and returns (view, observation-or-None). The observation (source_type "record") holds only the items this
    agent had not read before, formatted as displayed; it is None when nothing is new, when
    `records.encode_on_read` is never, or when the display is none."""
    rc = rcfg(cfg)
    v = render_for(binder, cfg, day, query=query, embed=embed)
    new = binder.new_for(agent_id, v.ids)
    binder.mark_read(agent_id, v.ids, tick)
    tracer.log("record_read", agent=agent_id, tick=tick, context=context, view_mode=v.mode,
               entry_ids=list(v.entry_ids), rev_ids=list(v.rev_ids), new_ids=list(new), view_sha=v.sha)
    obs = None
    if new and rc["encode_on_read"] == "new_only" and v.mode != "none":
        obs = read_observation(v, new, agent_id, tick, context, location, arena)
    return v, obs


def read_observation(v: View, new_ids, agent_id: str, tick: int, context: str, location: str, arena: str):
    """AgentObservation(source_type="record") with one fact per new item, formatted exactly as displayed.
    `speaker` is the author (verbatim's exclude_self skips one's own entries); `record_id` goes into
    MemoryMeta.record_ids. Facts involve nobody, so no name generalisation rewrites a signature."""
    from backend.agents.perception import AgentObservation
    oid = f"rec:{tick}:{agent_id}:{context}"
    facts = [{"id": f"{oid}.{i}", "text": v.lines[i], "world_text": v.lines[i], "salience": RECORD_SALIENCE,
              "involves": [], "speaker": v.authors.get(i), "record_id": i, "kind": "record"} for i in new_ids]
    return AgentObservation(id=oid, agent_id=agent_id, tick=tick, location=location, arena=arena,
                            source_type="record", facts=facts, event_ids=[])


def transitions_on(cfg: dict, day: int) -> list[str]:
    return [str(t["mode"]) for t in (rcfg(cfg)["transitions"] or []) if int(t["day"]) == int(day)]


def apply_transitions(binder: Binder, cfg: dict, tracer, day: int, tick: int) -> list[dict]:
    """Day-start hook (phase 0a, after roster changes). Applies every transition scheduled for `day`, traces
    `record_transition`, and returns the matched world facts the world must release at TRANSITION_TIME in the
    Makerspace (salience RECORD_SALIENCE): [{"mode", "text", "time", "salience"}]."""
    out = []
    if not enabled(cfg):
        return out
    for mode in transitions_on(cfg, day):
        archived = binder.transition(mode, tick)
        tracer.log("record_transition", day=day, mode=mode, archived_binder_id=archived)
        out.append({"mode": mode, "text": TRANSITION_FACT[mode], "time": TRANSITION_TIME,
                    "salience": RECORD_SALIENCE, "place": "Makerspace"})
    return out


def apply_writes(binder: Binder, cfg: dict, tracer, decisions: list[dict]) -> list[dict]:
    """Phase 4c: apply RecordWrite decisions in sorted (tick, agent) order, after all of the tick's decisions.
    decision: {agent, author_name, tick, offer: job|tally|farewell, choice: none|log|front, text, job_id?,
    prompt?, response?}. Traces `record_write` for every decision (including none) and returns the applied
    decisions with `entry_id` / `rev_id` / `text` / `truncated` filled in."""
    rc = rcfg(cfg)
    applied = []
    for d in sorted(decisions, key=lambda d: (int(d["tick"]), str(d["agent"]))):
        choice, text = d.get("choice", "none"), d.get("text")
        if choice == "front" and not rc["revisable"]:
            choice = "log"
        rec = {"agent": d["agent"], "tick": d["tick"], "offer": d.get("offer", "job"), "choice": choice,
               "entry_id": None, "rev_id": None, "text": None, "truncated": False}
        if choice == "log" and text:
            e, tr = binder.append(d["agent"], d["author_name"], d["tick"], text, context=rec["offer"],
                                  job_id=d.get("job_id"), max_chars=int(rc["log_max_chars"]))
            rec.update(entry_id=e.entry_id, text=e.text, truncated=tr)
        elif choice == "front" and text:
            r, tr = binder.rewrite(d["agent"], d["author_name"], d["tick"], text, max_chars=int(rc["front_max_chars"]))
            rec.update(rev_id=r.rev_id, text=r.text, truncated=tr)
        else:
            rec["choice"] = "none"
        tracer.log("record_write", **rec, prompt=d.get("prompt"), response=d.get("response"))
        applied.append(rec)
    return applied
