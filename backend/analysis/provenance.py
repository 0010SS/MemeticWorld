"""Typed provenance links per utterance (OBSERVER ONLY; ontology v2 §4).

Three kinds of link from an utterance to hidden world events:

- referent_event_ids: what the utterance is *about*, judged from its words. An event counts when it
  had started by the utterance's tick and the utterance shares >= 2 content tokens with the event's
  released text: the world's fact texts, every viewpoint rendering of them (what agents actually
  perceived), and its referent names. Content token = not a stopword, not a person's name, not a
  number, >= 3 letters, wordfreq Zipf < 5. At least one shared token must lie OUTSIDE the population
  lexicon (routine activities, places, rooms, profile words; `emergence.vocabulary`): campus
  vocabulary alone ("meet you at the dining hall") is routine talk, not talk about an incident, while
  lexicon words still count towards the 2 when the utterance also names something of the event's own
  ("the lecture hall projector"). The conversation's (or remark's) trigger events count too.
  Grounding uses these links only.
- direct_event_ids / carried_event_ids: how the speaker came to have it in mind, from the source types
  of the memories retrieved for the utterance. Perception is direct; conversation, overheard,
  reminding and reflection memories are carried (the event reached the speaker through talk or
  through their own linking). The two sets may overlap.

Retrieval-based links (the legacy `retrieved_event_ids`) over-attribute: a speaker who retrieved a
memory of an event need not talk about it (D26). Lexical links under-attribute paraphrase, but they
are symmetric across families and conditions, which is what grounding needs.
"""
from __future__ import annotations

import re

from backend.analysis.candidates import STOP, tokens, zipf

MIN_SHARED = 2
ZIPF_MAX = 5.0
DIRECT = {"perception"}
CARRIED = {"conversation", "overheard", "reminding", "reflection"}
_REMARK = re.compile(r":remark:[^:]+:(?P<ev>[^:]+?)\.b\d+$")


def content_tokens(text: str, names: set) -> set:
    out = set()
    for t in tokens(text or ""):
        if t.endswith("'s"):
            t = t[:-2]
        if len(t) < 3 or t.isdigit() or t in STOP or t in names:
            continue
        if zipf(t) < ZIPF_MAX:
            out.add(t)
    return out


def _fact_event(fid: str) -> str:
    return fid.rsplit(".b", 1)[0]


class EventText:
    """Released content tokens per event over time: (tick, tokens) segments from fact texts (at the
    beat's tick), viewpoint renderings (at the rendering's tick) and referent names (at start).

    `lexicon`: population-lexicon tokens that cannot carry a link on their own (default: the run's,
    `rd.lexicon_tokens`; none when rd has no lexicon)."""

    def __init__(self, rd, lexicon: set | None = None):
        names = rd.name_tokens
        if lexicon is None:
            try:
                lexicon = rd.lexicon_tokens
            except AttributeError:        # bare stand-ins without a manifest (tests, scripts)
                lexicon = set()
        self.lexicon = set(lexicon)
        self.start = {eid: int(e.get("start_tick", 0)) for eid, e in rd.events.items()}
        segs: dict[str, list] = {eid: [] for eid in rd.events}
        for eid, e in rd.events.items():
            ref = " ".join(r.get("name", "") for r in e.get("referents") or [])
            if ref:
                segs[eid].append((self.start[eid], content_tokens(ref, names)))
            for b in e.get("beats", []):
                tick = int(b.get("tick", self.start[eid]))
                for f in b.get("facts", []):
                    segs[eid].append((tick, content_tokens(f.get("text", ""), names)))
        for r in rd.of("viewpoint"):
            evs = [x for x in r.get("originating_event_ids") or [] if x in segs]
            for f in r.get("facts", []):
                eid = _fact_event(f.get("id", ""))
                eid = eid if eid in segs else (evs[0] if len(evs) == 1 else None)
                if eid and f.get("perceived"):
                    segs[eid].append((int(r["tick"]), content_tokens(f["perceived"], names)))
        self.segs = {eid: sorted(s, key=lambda x: x[0]) for eid, s in segs.items()}
        self.names = names
        self._cache: dict = {}

    def tokens_at(self, eid: str, tick: int) -> set:
        key = (eid, tick)
        if key not in self._cache:
            out = set()
            for t, s in self.segs[eid]:
                if t > tick:
                    break
                out |= s
            self._cache[key] = out
        return self._cache[key]

    def link(self, text: str, tick: int, min_shared: int = MIN_SHARED) -> list[str]:
        """Events started by `tick` whose released text shares >= min_shared content tokens with text,
        at least one of them outside the population lexicon."""
        ct = content_tokens(text, self.names)
        if len(ct) < min_shared or not ct - self.lexicon:
            return []
        out = []
        for eid in self.segs:
            if self.start[eid] > tick:
                continue
            shared = ct & self.tokens_at(eid, tick)
            if len(shared) >= min_shared and shared - self.lexicon:
                out.append(eid)
        return out


def trigger_event_ids(u: dict, rd) -> list[str]:
    """The private trigger of a reaction remark or a reaction-initiated conversation."""
    out = list(u.get("trigger_event_ids") or [])
    if not u.get("conversation_id"):
        m = _REMARK.search(u.get("id", ""))
        if m:
            out.append(m.group("ev"))
    else:
        trig = (rd.conversations.get(u["conversation_id"]) or {}).get("trigger") or {}
        out += trig.get("event_ids") or []
    return [e for i, e in enumerate(out) if e in rd.events and e not in out[:i]]


def compute(rd) -> dict:
    """utterance_id -> {"referent_event_ids", "direct_event_ids", "carried_event_ids"} (sorted lists)."""
    et = EventText(rd)
    nodes = rd.nodes
    out = {}
    for u in rd.utterances:
        tick = int(u["tick"])
        trig = trigger_event_ids(u, rd)
        ref = set(et.link(u["text"], tick)) | set(trig)
        direct, carried = set(), set()
        for nid in u.get("retrieved") or []:
            n = nodes.get(nid)
            if not n:
                continue
            evs = {e for e in n["event_ids"] if e in et.start and et.start[e] <= tick}
            if n["source_type"] in DIRECT:
                direct |= evs
            elif n["source_type"] in CARRIED:
                carried |= evs
        conv = rd.conversations.get(u.get("conversation_id") or "") or {}
        trig_agent = (conv.get("trigger") or {}).get("agent")
        if not u.get("conversation_id") or (trig_agent == u["speaker"] and int(u.get("idx") or 0) < 2):
            direct |= set(trig)          # the speaker perceived the trigger just now
        out[u["id"]] = {"referent_event_ids": sorted(ref), "direct_event_ids": sorted(direct),
                        "carried_event_ids": sorted(carried)}
    return out
