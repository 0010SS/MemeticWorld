"""Live, LLM-free meme analysis of a run directory up to tick t (OBSERVER ONLY; never imported by
simulation code, never written back into a run).

Works while a run is still writing: trace.jsonl / events.jsonl are read incrementally (only complete
lines; a torn last line is left for the next call), manifest.json / config.resolved.yaml are re-read
when their mtime changes (a half-written manifest keeps the previous copy). Results are cached per
(run, file signatures, tick, params), so scrubbing a replay forward is cheap.

A snapshot at tick t sees exactly the trace records with tick <= t, and world text released by t:
event facts come from `event_beat` trace records (v2 incidents and v3 co-op jobs, cues, tallies,
orientation and farewell facts), merged with events.jsonl metadata (referent names), plus viewpoint
renderings and the binder's fixed rendering strings.

Expression statuses (ontology v2 §4 emergence semantics, per expression, at tick t; tiers.py):
- exposure is causal (`rundata.precedes`); adopters used it after hearing it; a *carried* adopter used
  it in a different exchange at a strictly later tick than an exposure (emergence._carried);
  independent users used it with no prior exposure;
- "planted": matches controls.planted_phrase (the positive control);
- "system_wording": text the system put into agents' heads (wording.Infrastructure: relationship lines,
  routines, seed / ambient memories, memory frames, conversation-context lines, profile text, NPC roles,
  place names, co-op roster / menu text) or population lexicon;
- "world_wording": world-provided event text (fact texts incl. co-op surfaces, viewpoint renderings, referent
  names, cue/tally texts, binder rendering) or verbatim world text;
- "emerged": >= 2 carried adopters and more carried adopters than independent users;
- "spreading": >= 1 carried (exposure-driven, cross-exchange) adopter;
- "echo": others repeated it only inside the exchange where they heard it;
- "new": used only by its originator and/or independent users so far.
`status` is the first applicable of planted > system_wording > world_wording > emerged > spreading > echo > new;
`flags` lists every applicable chip (a planted phrase can also be "emerged").

Injected expressions come in two roles and are never conflated. `controls.planted_phrase` is the positive
CONTROL: one agent, status "planted", left out of convention counts. `memes.registry` is the injected
COHORT (the 2x2), each meme seeded into its own committed minority: those are the DEPENDENT VARIABLE, so
they keep the dynamic status, count in `tier_counts`, carry a `meme` chip and their registry `meme_id`,
and are always shown even when they fall out of the top N or were said only twice. Both roles share one
thing: their habit line is not campus vocabulary and not world wording, so neither is discounted as
routine wording. The `memes` block reports the whole cohort per registry id (including a meme nobody has
said), with the manipulation check and the crowding split.

- "ordinary" is a FLAG, not a status: the actor model's own register (register.Background -- several
  speakers use it in other, independent runs too). It blocks "emerged" and the convention tier.

Tiers (tiers.py): `tier` is "candidate" (frequency only) -> "spreading" (>= 1 carried adopter, not system/world
wording, not model register) -> "convention" (emerged AND a REAL judge verdict is_convention=true AND not the
planted control).
`bucket` (candidates.bucket_of) orders the pool: "expression" (could be culture) > "personal" (one speaker or
one conversation) > "ordinary" (model register / stock formula) > "wording" (system or world). Everything
stays in the pool and in every count; only the ORDER changes, so the head of `expressions` -- what the UI
calls culture -- can hold only candidates for culture. `bucket_counts` reports the split.
Only well-formed expressions are extracted (wording.WordClasses). Judge verdicts come from
analysis_judgements/*.json and analysis.json; a verdict is used at tick t only if it was judged on data up to
t (a live judge_run at tick <= t; analysis.json verdicts only at the latest tick). Mock verdicts are placeholders
and never make a convention; `judge` in the snapshot says whether a real judge has run.

Hidden-derived metrics (v3 ground truth: job class/cause, regime, mapping) are dicts carrying
`"hidden": true` and are listed in `hidden_fields`; `strip_hidden()` removes them for demo mode.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import threading
import time
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
from functools import cached_property
from pathlib import Path

import yaml

from backend.analysis.candidates import BUCKET_RANK, BUCKETS, CandidateExtractor, bucket_of, tokens
from backend.analysis.emergence import _carried, _norm, emergence_for, lexicon_flag, vocabulary, world_match, \
    world_lemma_match
from backend.analysis.rundata import RunData, chron_key, precedes, utterance_key
from backend.analysis.tiers import STATUSES, TIERS, VerdictIndex, classify_status, forms, tier_counts, tier_of, \
    verdict_summary

LIVE_VERSION = "live-v2"
DEFAULTS = {"min_speakers": 1, "min_uses": 2, "pool": 60, "trend_window": 12, "transmission_top": 10}
HIDDEN_FIELDS = ["v3.first_attempt_accuracy", "v3.old_regime_response", "v3.k1_vs_k2", "v3.regime"]
TIMELINE_HIDDEN_FIELDS = ["series[].v3_hidden"]
CACHE_SIZE = 64
_BINDER_TEXT = ("The co-op binder next to the laser cutter.", "There is nothing written in the binder yet.",
                "Front page: nothing written on it yet.", "Log, newest first:")


class LiveError(RuntimeError):
    pass


def _tick(r: dict) -> int:
    try:
        return int(r.get("tick"))
    except (TypeError, ValueError):
        return 0


def _mtime(p: Path):
    try:
        return p.stat().st_mtime_ns
    except FileNotFoundError:
        return None


# ------------------------------------------------------------------------------------------------ incremental IO
class _Tail:
    """Incremental reader of an append-only JSONL file: parses only complete lines; resets when the file
    is replaced, truncated or rewritten (its first bytes change)."""
    HEAD = 512

    def __init__(self, path: Path):
        self.path = Path(path)
        self.generation = -1
        self._reset()

    def _reset(self):
        self.records: list[dict] = []
        self.offset = 0
        self.ino = None
        self.head = b""
        self.sig = None
        self.bad_lines = 0
        self.pending_bytes = 0
        self.max_tick = -1
        self.generation += 1

    def refresh(self):
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            if self.sig is not None:
                self._reset()
            return None
        sig = (st.st_ino, st.st_size, st.st_mtime_ns)
        if sig == self.sig:
            return (self.generation, sig)
        with open(self.path, "rb") as f:
            head = f.read(self.HEAD)
            if self.ino is not None and (st.st_ino != self.ino or st.st_size < self.offset
                                         or head[:len(self.head)] != self.head):
                self._reset()
            f.seek(self.offset)
            data = f.read()
        cut = data.rfind(b"\n") + 1
        for line in data[:cut].split(b"\n"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.bad_lines += 1
                continue
            if isinstance(r, dict):
                self.records.append(r)
                self.max_tick = max(self.max_tick, _tick(r))
        self.offset += cut
        self.pending_bytes = len(data) - cut
        self.ino = st.st_ino
        if len(head) > len(self.head):
            self.head = head
        self.sig = sig
        return (self.generation, sig)


class _RunCache:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.trace = _Tail(self.dir / "trace.jsonl")
        self.events = _Tail(self.dir / "events.jsonl")
        self.lock = threading.RLock()
        self.manifest = None
        self.cfg = None
        self._meta_sig = None
        self._vocab = None
        self._ref_gen = None
        self.ref_links: dict[str, list[str]] = {}
        self.results: OrderedDict = OrderedDict()
        self._vsig = None
        self.vindex = VerdictIndex([])

    def refresh(self) -> tuple:
        ms = self._meta()
        ts = self.trace.refresh()
        es = self.events.refresh()
        if self._ref_gen != (self.trace.generation, self.events.generation, ms):
            self.ref_links = {}           # links are stable per utterance only within one file generation
            self._ref_gen = (self.trace.generation, self.events.generation, ms)
        vs = self._verdicts()
        return (ts, es, ms, vs)

    def _verdicts(self):
        """Judge verdicts (analysis_judgements/*.json + analysis.json), reloaded when those files change."""
        d = self.dir / "analysis_judgements"
        files = tuple(sorted((p.name, _mtime(p)) for p in d.glob("*.json"))) if d.exists() else ()
        sig = (files, _mtime(self.dir / "analysis.json"))
        if sig != self._vsig:
            from backend.analysis.judge import analysis_verdicts, judgement_verdicts
            an = None
            if sig[1] is not None:
                try:
                    an = json.loads((self.dir / "analysis.json").read_text())
                except (json.JSONDecodeError, OSError):
                    an = None
            try:
                rows = judgement_verdicts(self.dir) + analysis_verdicts(self.dir, an)
            except Exception:   # noqa: BLE001 - a malformed verdict file never breaks the live view
                rows = []
            self.vindex, self._vsig = VerdictIndex(rows), sig
        return sig

    def _meta(self):
        mp, cp = self.dir / "manifest.json", self.dir / "config.resolved.yaml"
        sig = (_mtime(mp), _mtime(cp))
        if sig == self._meta_sig and self.manifest is not None:
            return sig
        if sig[0] is None:
            raise LiveError(f"{self.dir} is not a run directory (no manifest.json)")
        err = None
        for _ in range(3):
            try:
                man = json.loads(mp.read_text())
                cfg = yaml.safe_load(cp.read_text()) if cp.exists() else man.get("config")
                break
            except (json.JSONDecodeError, yaml.YAMLError, OSError) as e:   # mid-write: retry, else keep the old copy
                err = e
                time.sleep(0.05)
        else:
            if self.manifest is None:
                raise LiveError(f"cannot read manifest/config of {self.dir}: {err}")
            return self._meta_sig
        self.manifest, self.cfg = man, cfg or {}
        self._meta_sig, self._vocab = sig, None
        return sig

    def vocabulary(self, rd) -> dict:
        if self._vocab is None:
            self._vocab = vocabulary(rd)
        return self._vocab

    def memo(self, key, fn):
        if key in self.results:
            self.results.move_to_end(key)
            return self.results[key]
        v = fn()
        self.results[key] = v
        while len(self.results) > CACHE_SIZE:
            self.results.popitem(last=False)
        return v


_CACHES: dict[str, _RunCache] = {}
_CACHES_LOCK = threading.Lock()


def _cache(run_dir) -> _RunCache:
    key = str(Path(run_dir).resolve())
    with _CACHES_LOCK:
        rc = _CACHES.get(key)
        if rc is None:
            rc = _CACHES[key] = _RunCache(Path(run_dir))
        return rc


def clear_cache(run_dir=None) -> None:
    with _CACHES_LOCK:
        if run_dir is None:
            _CACHES.clear()
        else:
            _CACHES.pop(str(Path(run_dir).resolve()), None)


# ------------------------------------------------------------------------------------------------ prefix view
class _PrefixRun(RunData):
    """RunData restricted to trace records with tick <= t (cached per snapshot)."""

    def __init__(self, rc: _RunCache, tick: int):
        self.dir = rc.dir
        self.manifest = rc.manifest
        self.cfg = rc.cfg
        self.tick = int(tick)
        self._rc = rc

    @cached_property
    def trace(self) -> list[dict]:
        t = self.tick
        return [r for r in self._rc.trace.records if _tick(r) <= t]

    @cached_property
    def memory_meta(self) -> dict:
        return {}                 # the final memory state would leak the future

    @cached_property
    def lexicon_tokens(self) -> set:
        return set(self._rc.vocabulary(self)["tokens"])

    @cached_property
    def events(self) -> dict:
        """Events released by t, with only the beats (facts) released by t. `event_beat` trace records are
        the source of released facts (v2 incidents and v3 co-op facts alike); events.jsonl adds metadata."""
        t = self.tick
        base = {}
        for e in self._rc.events.records:
            if e.get("id") is not None and int(e.get("start_tick") or 0) <= t:
                base[str(e["id"])] = e
        beats: dict[str, list] = {}
        for r in self.of("event_beat"):
            beats.setdefault(str(r.get("event_id")), []).append(
                {"idx": r.get("beat"), "tick": _tick(r), "location": r.get("location"), "arena": r.get("arena"),
                 "facts": list(r.get("facts") or [])})
        out = {}
        for eid in sorted(set(base) | set(beats), key=lambda x: (min([b["tick"] for b in beats.get(x, [])] or
                                                                    [int((base.get(x) or {}).get("start_tick") or 0)]), x)):
            src = base.get(eid)
            e = dict(src) if src else {"id": eid, "generator": "event_beat"}
            if eid in beats:
                e["beats"] = sorted(beats[eid], key=lambda b: (b["tick"], b["idx"] or 0))
            elif src and src.get("beats"):
                st = int(src.get("start_tick") or 0)
                e["beats"] = [b for b in src["beats"] if int(b.get("tick", st)) <= t]
            e.setdefault("start_tick", min([b["tick"] for b in e.get("beats") or []] or [0]))
            texts = [f.get("text", "") for b in e.get("beats") or [] for f in b.get("facts", [])]
            if not texts and not e.get("referents") and not (src and not src.get("beats") and src.get("narrative")):
                continue              # facts-less pseudo events (e.g. forced movers)
            if e.get("beats") is not None and (texts or (src and src.get("beats"))):
                e["narrative"] = " ".join(texts)       # released wording only
            out[eid] = e
        return out


def _label(manifest: dict, tick: int) -> str:
    tpd = int(manifest.get("ticks_per_day") or 60)
    try:
        start = dt.datetime.fromisoformat(manifest["start"])
        t = start + dt.timedelta(days=tick // tpd, minutes=(tick % tpd) * int(manifest.get("tick_minutes") or 15))
        return f"Day {tick // tpd + 1}, {t.strftime('%H:%M')}"
    except (KeyError, ValueError, TypeError):
        return f"Day {tick // tpd + 1}, tick {tick % tpd}"


def _params(cfg: dict, top: int, trend_window=None) -> dict:
    live = dict(((cfg.get("analysis") or {}).get("live")) or {})
    p = {k: live.get(k, v) for k, v in DEFAULTS.items()}
    if trend_window is not None:
        p["trend_window"] = int(trend_window)
    p["top"] = int(top)
    p["pool"] = max(int(p["pool"]), int(top))       # the pool (and so every count) does not depend on top
    return p


def roster_at(rd, tick: int) -> dict:
    """Active agents, cohorts and roles at `tick` from public roster_change records (v3 turnover);
    every non-reserve agent is a founder. Without turnover: everyone, cohort founder."""
    agents = rd.agents or {}
    founders = sorted(a for a, v in agents.items() if not (isinstance(v, dict) and v.get("reserve")))
    changes = sorted((r for r in rd.of("roster_change") if _tick(r) <= tick), key=lambda r: (_tick(r), r.get("agent") or ""))
    arrive_days = sorted({int(r.get("day") or 0) for r in changes if r.get("kind") == "arrive"})
    cohort = {a: "founder" for a in founders}
    role = {a: (agents.get(a) or {}).get("role") or (agents.get(a) or {}).get("coop_role") for a in founders}
    active = set(founders)
    for r in changes:
        a = r.get("agent")
        if r.get("kind") == "arrive":
            active.add(a)
            cohort[a] = f"W{arrive_days.index(int(r.get('day') or 0)) + 1}"
            role[a] = r.get("role") or role.get(a)
        elif r.get("kind") == "depart":
            active.discard(a)
    by_cohort = defaultdict(int)
    for a in active:
        by_cohort[cohort.get(a, "founder")] += 1
    return {"active": sorted(active), "departed": sorted({r.get("agent") for r in changes if r.get("kind") == "depart"}),
            "cohorts": dict(sorted(cohort.items())), "roles": {a: r for a, r in sorted(role.items()) if r},
            "active_by_cohort": dict(sorted(by_cohort.items())),
            "changes": [{k: r.get(k) for k in ("tick", "day", "agent", "kind", "role", "replaces")} for r in changes]}


def adopters_detail(usages: list[dict], em: dict) -> list[dict]:
    """Per agent that used or heard the expression: role (originator | independent | carried | echo |
    exposed), first exposure, first use, first carried use, and the source exposure of the adopting use
    (the latest causally preceding exposure; for carried adopters, one from another exchange)."""
    orig = em.get("originator")
    uses, heard = defaultdict(list), defaultdict(list)
    for u in usages:
        uses[u["speaker"]].append(u)
        for l in u.get("listeners") or []:
            if l != u["speaker"]:
                heard[l].append(u)
    ind, car, ado = set(em.get("independents") or []), set(em.get("carried_adopters") or []), set(em.get("adopters") or [])
    rows = []
    for a in set(uses) | set(heard):
        mine, exp = uses.get(a, []), heard.get(a, [])
        role = ("originator" if a == orig else "independent" if a in ind else "carried" if a in car
                else "echo" if a in ado else "exposed")
        first = mine[0] if mine else None
        cu = next((y for y in mine if _carried(y, exp)), None) if role == "carried" else None
        use = cu or first
        src = None
        if role in ("carried", "echo") and use:
            prior = [x for x in exp if precedes(x, use)]
            if cu:
                prior = [x for x in prior if utterance_key(x) != utterance_key(use) and x["tick"] < use["tick"]] or prior
            if prior:
                x = max(prior, key=chron_key)
                src = {"agent": x["speaker"], "tick": x["tick"], "utterance_id": x["utterance_id"],
                       "conversation_id": x.get("conversation_id")}
        rows.append({"agent": a, "role": role, "first_exposure_tick": min((x["tick"] for x in exp), default=None),
                     "first_use_tick": first["tick"] if first else None,
                     "first_use_utterance_id": first["utterance_id"] if first else None,
                     "carried_tick": cu["tick"] if cu else None, "carried_utterance_id": cu["utterance_id"] if cu else None,
                     "source": src})
    big = 10 ** 9
    rows.sort(key=lambda r: (r["first_use_tick"] if r["first_use_tick"] is not None else big,
                             r["first_exposure_tick"] if r["first_exposure_tick"] is not None else big, r["agent"]))
    return rows


def transmission_tree(rows: list[dict], names: dict | None = None) -> dict:
    """{"root": originator node, "other_roots": independent users' nodes, "edges": n}. A node:
    {agent, name, role, tick, children}; an adopter hangs under the speaker it last heard it from."""
    names = names or {}
    kids = defaultdict(list)
    by = {r["agent"]: r for r in rows}
    for r in rows:
        if r["source"] and r["role"] in ("carried", "echo"):
            kids[r["source"]["agent"]].append(r["agent"])

    def node(a, seen):
        r = by.get(a, {"role": "unknown", "first_use_tick": None, "carried_tick": None})
        kid = [node(k, seen | {a}) for k in sorted(kids.get(a, []), key=lambda k: (by[k]["carried_tick"] or by[k]["first_use_tick"] or 0, k))
               if k not in seen and k != a]
        return {"agent": a, "name": names.get(a, a), "role": r["role"], "tick": r.get("first_use_tick"),
                "carried_tick": r.get("carried_tick"), "children": kid}
    orig = next((r["agent"] for r in rows if r["role"] == "originator"), None)
    return {"root": node(orig, frozenset()) if orig else None,
            "other_roots": [node(r["agent"], frozenset()) for r in rows if r["role"] == "independent"],
            "edges": sum(len(v) for v in kids.values())}


# ------------------------------------------------------------------------------------------------ per (run, tick)
class _Snap:
    def __init__(self, rc: _RunCache, tick: int, params: dict):
        self.rc, self.t, self.p = rc, int(tick), params
        self.rd = _PrefixRun(rc, self.t)
        self.man = rc.manifest
        self.tpd = int(self.man.get("ticks_per_day") or 60)

    @cached_property
    def roster(self) -> dict:
        return roster_at(self.rd, self.t)

    @cached_property
    def population(self) -> list[str]:
        """Everyone who was active at some point by t (plus anyone who spoke or listened)."""
        pop = set(self.roster["cohorts"])
        for u in self.rd.utterances:
            pop.add(u["speaker"])
            pop |= set(u.get("listeners") or [])
        return sorted(a for a in pop if a)

    @cached_property
    def vocab(self) -> set:
        return set(self.rc.vocabulary(self.rd)["tokens"])

    @cached_property
    def planted(self) -> dict | None:
        try:
            pl = self.rd.planted
        except Exception:   # noqa: BLE001 - a malformed control never breaks the live view
            pl = None
        if pl and pl.get("phrase"):
            return dict(pl, norm=" ".join(tokens(pl["phrase"])))
        return None

    @cached_property
    def memes(self) -> list[dict]:
        """The injected STUDY memes (memes.registry), each with its normalised surface form. Unlike the
        positive control these are the dependent variable: they are tracked as first-class expressions,
        never excluded, and each keeps its registry id so four memes never blur into one another."""
        try:
            out = []
            for m in self.rd.memes:
                norm = " ".join(tokens(m["phrase"]))
                if norm:
                    out.append(dict(m, norm=norm))
            return out
        except Exception:   # noqa: BLE001 - a malformed registry never breaks the live view
            return []

    @cached_property
    def world_segments(self) -> list[tuple[int, list[str], list[str]]]:
        """(release tick, tokens, folded tokens) of every world-provided wording released by t."""
        rd, parts = self.rd, []
        for e in rd.events.values():
            st = int(e.get("start_tick") or 0)
            facts = [(int(b.get("tick", st)), f.get("text", "")) for b in e.get("beats") or [] for f in b.get("facts", [])]
            parts += facts or [(st, e.get("narrative") or "")]
            parts += [(st, r.get("name", "")) for r in e.get("referents") or []]
        for r in rd.of("viewpoint"):
            parts += [(_tick(r), f.get("perceived") or "") for f in r.get("facts", [])]
        for r in rd.of("cue_event", "tally"):
            parts.append((_tick(r), r.get("text") or ""))
        for r in rd.of("coop_fact"):              # the public twin of co-op event_beats (newer engines)
            parts += [(_tick(r), f.get("text") or "") for f in r.get("facts") or []]
        parts += [(-1, s) for s in _BINDER_TEXT]
        out = []
        for tk, text in parts:
            toks = tokens(text) if text else []
            if toks:
                out.append((tk, toks, [_norm(x) for x in toks]))
        out.sort(key=lambda s: s[0])
        return out

    def world_tick(self, toks: list[str]) -> tuple[int | None, str | None]:
        """(first release tick of a world wording containing the phrase, "verbatim"|"gapped"|"lemmas").
        The third kind is a PARAPHRASE of one event fact ("soaked by the sprinklers", "backpack situation"):
        without it one incident fills a third of the list under near-synonyms."""
        if not toks:
            return None, None
        segs = self.world_segments
        for tk, seg, normed in segs:
            if world_match(toks, [seg], normed=[normed]):
                kind = world_match(toks, [s for _, s, _ in segs], normed=[n for _, _, n in segs])
                return tk, kind
        names = self.rd.name_tokens
        for tk, seg, _normed in segs:
            if world_lemma_match(toks, [seg], names=names):
                return tk, "lemmas"
        return None, None

    @staticmethod
    def _matches(c: dict, norm: str) -> bool:
        p = f" {norm} "
        return any(f and (f" {f} " in p or p in f" {f} ")
                   for f in {c.get("canonical_form"), *(c.get("variants") or [])})

    def is_planted(self, c: dict) -> bool:
        """The positive CONTROL only. A study meme is not a control and must not be marked as one."""
        pl = self.planted
        return bool(pl and pl["norm"] and self._matches(c, pl["norm"]))

    def meme_of(self, c: dict) -> str | None:
        """The registry id of the study meme this expression is, or None."""
        return next((m["id"] for m in self.memes if self._matches(c, m["norm"])), None)

    # ---------------------------------------------------------------- expressions
    @cached_property
    def extractor(self) -> CandidateExtractor:
        return CandidateExtractor(self.rd, {"min_speakers": self.p["min_speakers"], "min_uses": self.p["min_uses"]})

    @cached_property
    def pool(self) -> list[dict]:
        """Expression records (with private _usages/_adopters) ranked by the extractor's score."""
        if not self.rd.utterances:
            return []
        ex = self.extractor
        cands = ex.extract(int(self.p["pool"]))
        # an injected phrase is in the pool whenever it was said at all, even below min_uses: the
        # control has to be visible to prove detection works, and a study meme IS the measurement --
        # "said twice and died" is a result, not a reason to drop it off the list
        for inj in ([self.planted] if self.planted else []) + self.memes:
            if any(self._matches(c, inj["norm"]) for c in cands):
                continue
            e = inj["norm"]
            hits = [u for u in self.rd.utterances if e and f" {e} " in f" {' '.join(tokens(u['text']))} "]
            if hits:
                g = tuple(e.split())
                st = {"uses": [u["id"] for u in hits], "speakers": {u["speaker"] for u in hits},
                      "surface": {inj["phrase"]: len(hits)}}
                cands += ex.group([(g, st, ex.score(g, st))])
        recs = [self.record(c) for c in cands]
        self.mark_incident_talk(recs)
        # bucket first, score second: system / world wording, the actor model's own register and one
        # person's tic rank BELOW everything that could be culture, so the head of the list the UI shows
        # as culture holds only candidates for culture (they stay in the pool, and keep their counts).
        recs.sort(key=lambda r: (BUCKET_RANK.get(r["bucket"], 9), -r["score"]))
        for i, r in enumerate(recs):
            r["id"] = f"x{i:02d}"
        return recs

    INCIDENT_SHARE = 0.75
    INCIDENT_KEEP = 1
    INCIDENT_KEEP_SURPRISE = 5.0

    def mark_incident_talk(self, recs: list[dict]) -> None:
        """One incident, retold in a dozen wordings, is one topic of conversation, not a dozen expressions.
        Expressions whose uses (>= INCIDENT_SHARE of them) all link to the SAME referent event are collapsed:
        the best-scoring one stays as the incident's label, the rest move to the wording bucket as
        `incident_talk`. Meta-marked, nickname and planted expressions are never collapsed -- "inbox
        apocalypse" is a coinage *about* an incident, "grabbed it by accident" is the incident."""
        links = self.ref_links()
        by_event: dict = defaultdict(list)
        for r in recs:
            q = r["quality"] or {}
            if r["planted"] or q.get("meta_marked") or q.get("quoted") or q.get("novel") \
                    or (q.get("surprise") or 0) >= self.INCIDENT_KEEP_SURPRISE:
                continue        # a coinage ABOUT the incident ("inbox apocalypse") is not the incident
            ev: Counter = Counter()
            for u in r["_usages"]:
                for e in links.get(u["utterance_id"]) or ():
                    ev[e] += 1
            if not ev:
                continue
            top, k = ev.most_common(1)[0]
            if k >= self.INCIDENT_SHARE * len(r["_usages"]):
                by_event[top].append(r)
        for eid, rs in by_event.items():
            rs.sort(key=lambda r: -r["score"])
            for r in rs[self.INCIDENT_KEEP:]:
                r["quality"]["incident_event_id"] = eid
                r["bucket"] = "wording"
                if "incident_talk" not in r["bucket_reasons"]:
                    r["bucket_reasons"] = [*r["bucket_reasons"], "incident_talk"]
                r["quality"]["reasons"] = r["bucket_reasons"]
                r["quality"]["bucket"] = "wording"

    def record(self, c: dict) -> dict:
        rd, t, W = self.rd, self.t, int(self.p["trend_window"])
        usages = sorted((dict(u, idx=(rd.utt_by_id.get(u["utterance_id"]) or {}).get("idx", u.get("idx")) or 0)
                         for u in c["usages"]), key=chron_key)
        toks = tokens(c["canonical_form"])
        wt, wm = self.world_tick(toks)
        in_lex = lexicon_flag(" ".join(toks), self.vocab, rd.name_tokens)
        factual = bool((c.get("features") or {}).get("factual_repetition"))
        control = self.is_planted(c)
        meme_id = None if control else self.meme_of(c)
        # `planted` means INJECTED (either role) wherever the pipeline has to know the wording did not
        # arise on its own: it keeps the expression out of the incident-talk collapse and in the
        # ranked pool. `control` alone means the positive control, which convention counts exclude.
        planted = control or bool(meme_id)
        wsys = c.get("wording") or self.extractor.infra.wording(toks)
        smatch = wsys.get("system_match", wsys.get("match"))
        system = bool(wsys.get("system")) or in_lex
        world = wt is not None or (factual and not system)
        reg = dict(c.get("register") or {})
        if not reg:
            loc = self.extractor.bg.localness([c["canonical_form"], *(c.get("variants") or [])])
            reg = {"ordinary": loc["ordinary"], "background_runs": loc["background_runs"], "stock": loc["stock"]}
        # the control is exempt from the model-register stop; a study meme is NOT. If the actor model
        # says the phrase in other, independent runs too, the cohort is measuring the model's own
        # register rather than this campus's transmission, and that has to show.
        ordinary = bool(reg.get("ordinary")) and not control
        em = emergence_for(usages, self.population, in_lexicon=in_lex, in_world_text=wt is not None,
                           in_system_text=system, in_register=ordinary)
        status, flags = classify_status(em, world=world, planted=control, system=system, ordinary=ordinary,
                                        meme=bool(meme_id))
        real, mock = self.verdict(c)
        tier, why = tier_of(em, system=system, world=world, planted=control, verdict=(real or {}).get("verdict"),
                            placeholder=(mock or {}).get("verdict"), ordinary=ordinary)
        n_convs = len({u.get("conversation_id") for u in usages if u.get("conversation_id")})
        # one speaker, or one exchange -- unless the speakers themselves marked the wording (quoted it, or
        # said they are keeping it), which is the strongest coinage evidence there is
        marked = (reg.get("meta_marked") or 0) or (reg.get("quoted") or 0)
        personal = len({u["speaker"] for u in usages}) < 2 or (n_convs < 2 and not marked)
        bucket, breasons = bucket_of(system=system, world=world, ordinary=ordinary, personal=personal,
                                     planted=planted)
        curve, seen = [], set()
        for u in usages:
            if u["speaker"] not in seen:
                seen.add(u["speaker"])
                curve.append([u["tick"], len(seen)])
        first = usages[0]
        fu = rd.utt_by_id.get(first["utterance_id"]) or {}
        recent = sum(1 for u in usages if u["tick"] > t - W)
        prev = sum(1 for u in usages if t - 2 * W < u["tick"] <= t - W)
        detail = adopters_detail(usages, em)
        return {
            "phrase": c["canonical_form"], "display": c.get("display_form") or c["canonical_form"],
            "variants": list(c.get("variants") or []), "uses": len(usages),
            "speakers": sorted({u["speaker"] for u in usages}), "n_speakers": len({u["speaker"] for u in usages}),
            "first_use": {"tick": first["tick"], "time_label": _label(self.man, first["tick"]), "speaker": first["speaker"],
                          "speaker_name": rd.names.get(first["speaker"], first["speaker"]),
                          "utterance_id": first["utterance_id"], "text": fu.get("text", first.get("text"))},
            "last_use_tick": usages[-1]["tick"], "adoption_curve": curve,
            "trend": {"window_ticks": W, "recent_uses": recent, "previous_uses": prev,
                      "direction": "new" if prev == 0 and recent == len(usages) else
                      "up" if recent > prev else "down" if recent < prev else "flat"},
            "status": status, "flags": flags, "planted": planted, "control": control, "meme_id": meme_id,
            "tier": tier, "tier_reasons": why, "verdict": verdict_summary(real or mock),
            "bucket": bucket, "bucket_reasons": breasons,
            "quality": {"bucket": bucket, "reasons": breasons, "n_conversations": n_convs,
                        "ordinary_register": ordinary, "background_runs": reg.get("background_runs", 0),
                        "stock_phrase": bool(reg.get("stock")), "meta_marked": reg.get("meta_marked", 0),
                        "quoted": reg.get("quoted", 0),
                        "surprise": reg.get("surprise"), "novel": reg.get("novel"),
                        "system_use_share": (c.get("wording") or {}).get("system_use_share")},
            "recited_share": c.get("recited_share", 0.0),
            "emergence": {k: em.get(k) for k in ("originator", "n_exposed", "n_adopters", "n_adopters_carried",
                                                 "n_echo_only", "n_independent", "adopters", "carried_adopters",
                                                 "independents", "fisher_p", "spread", "emerged")},
            "world_wording": {"any": world or system, "world": world, "system": system,
                              "system_match": smatch, "system_tick": (smatch or {}).get("tick", -1 if in_lex else None),
                              "in_world_text": wt is not None, "world_match": wm, "world_tick": wt,
                              "in_lexicon": in_lex, "factual_repetition": factual},
            # the run's own evidence of transmission is what the ontology calls culture, so it ranks:
            # an expression exposed agents carried into another conversation beats one merely said often
            "score": round(c.get("score", 0.0) * (1 + 0.35 * min(3, em.get("n_adopters_carried") or 0)
                                                  + 0.1 * min(3, em.get("n_adopters") or 0)), 3),
            "base_score": c.get("score", 0.0),
            "_usages": usages, "_adopters": detail,
        }

    def meme_rows(self, pool: list[dict]) -> dict | None:
        """The injected cohort at tick t: one row per REGISTRY entry -- including a meme nobody has said
        yet, which no expression pool can contain and which is exactly the row worth seeing early.

        Each row joins the manipulation check (did its own minority say it, to how many hearers) to where
        the phrase has got to (status, tier, exposure-driven adopters), keyed by the registry id so the
        four cells of the 2x2 never blur together. `crowding` says whether one minority is taking the
        airtime of the others."""
        if not self.memes:
            return None
        from backend.analysis.funnel import meme_checks
        try:
            checks = meme_checks(self.rd)
        except Exception:   # noqa: BLE001 - the cohort view never breaks the live view
            return None
        rows = {}
        for m in self.memes:
            chk = dict((checks.get("memes") or {}).get(m["id"]) or {})
            rec = next((r for r in pool if r["meme_id"] == m["id"]), None)
            rows[m["id"]] = {
                "id": m["id"], "phrase": m["phrase"], "cell": f"{m['grounding']}x{m['breadth']}",
                "grounding": m["grounding"], "breadth": m["breadth"], "site": m.get("site"),
                **{k: chk.get(k) for k in ("n_seeds", "seeds", "injected", "seed_uses", "seed_speakers",
                                           "silent_seeds", "seed_conversations", "distinct_hearers",
                                           "distinct_non_seed_hearers", "uses", "other_uses", "other_speakers",
                                           "first_use_day", "first_other_use_day", "last_use_day",
                                           "uses_by_day", "contamination")},
                "expression_id": (rec or {}).get("id"), "status": (rec or {}).get("status"),
                "tier": (rec or {}).get("tier"), "tier_reasons": (rec or {}).get("tier_reasons"),
                "flags": (rec or {}).get("flags"), "n_speakers": (rec or {}).get("n_speakers", 0),
                "trend": (rec or {}).get("trend"),
                "emergence": (rec or {}).get("emergence"),
                "world_wording": (rec or {}).get("world_wording"),
            }
        return {"memes": rows, "crowding": checks.get("crowding"),
                "not_injected": checks.get("not_injected"), "contaminated": checks.get("contaminated")}

    def usable(self, row: dict) -> bool:
        """A verdict judged on data up to tick <= t (analysis.json verdicts: on the whole run, i.e. the latest tick)."""
        tk = row.get("tick")
        if tk is None:
            return self.t >= self.rc.trace.max_tick
        try:
            return int(tk) <= self.t
        except (TypeError, ValueError):
            return False

    def verdict(self, c: dict) -> tuple[dict | None, dict | None]:
        return self.rc.vindex.lookup(forms(c), self.usable)

    # ---------------------------------------------------------------- transmission
    def edges(self, recs: list[dict]) -> list[dict]:
        out = []
        for r in recs:
            for a in r["_adopters"]:
                s = a["source"]
                if not s or a["role"] not in ("carried", "echo"):
                    continue
                out.append({"expression": r["phrase"], "expression_id": r["id"], "from": s["agent"], "to": a["agent"],
                            "heard_tick": s["tick"], "heard_utterance_id": s["utterance_id"],
                            "heard_conversation_id": s["conversation_id"],
                            "used_tick": a["carried_tick"] if a["role"] == "carried" else a["first_use_tick"],
                            "used_utterance_id": a["carried_utterance_id"] if a["role"] == "carried" else a["first_use_utterance_id"],
                            "kind": a["role"]})
        out.sort(key=lambda e: (e["used_tick"], e["expression_id"], e["to"]))
        return out

    # ---------------------------------------------------------------- funnel
    def ref_links(self) -> dict:
        """utterance id -> referent event ids (provenance.EventText + triggers), computed once per
        utterance: an utterance at tick tau links only through wording released by tau, so later ticks
        never change it."""
        from backend.analysis.provenance import EventText, trigger_event_ids
        rc, rd = self.rc, self.rd
        missing = [u for u in rd.utterances if u["id"] not in rc.ref_links]
        if missing:
            et = EventText(rd)
            for u in missing:
                ref = set(et.link(u["text"], int(u["tick"]))) | set(trigger_event_ids(u, rd))
                rc.ref_links[u["id"]] = sorted(ref)
        return rc.ref_links

    @cached_property
    def mediators(self) -> dict:
        """Tick lists of every mediator occurrence up to t (the funnel's raw material)."""
        rd = self.rd
        evs = rd.events
        released = {eid: int(e.get("start_tick") or 0) for eid, e in evs.items()}
        witnessed = {}
        for o in rd.of("observation"):
            if o.get("source_type") != "perception":
                continue
            for e in o.get("originating_event_ids") or ([o["event_id"]] if o.get("event_id") else []):
                if e in released:
                    witnessed[e] = min(witnessed.get(e, math.inf), _tick(o))
        refs = self.ref_links()
        discussed = {}
        for u in rd.utterances:
            for e in refs.get(u["id"], []):
                if e in released:
                    discussed[e] = min(discussed.get(e, math.inf), int(u["tick"]))
        links = rd.of("memory_link")
        cross = [_tick(r) for r in links if r.get("from_event_ids") and r.get("to_event_ids")
                 and not set(r["from_event_ids"]) & set(r["to_event_ids"])]
        stuck_self, stuck_other = [], []
        seen_nodes = set()
        for r in rd.of("wording"):
            seen_nodes.add(r.get("node_id"))
            (stuck_self if r.get("self_produced") or r.get("heard_from") == r.get("agent") else stuck_other).append(_tick(r))
        for m in rd.of("memory_encoded"):
            if m.get("node_id") in seen_nodes:
                continue
            for w in m.get("wordings") or []:
                (stuck_self if w.get("self_produced") or w.get("heard_from") == m.get("agent") else stuck_other).append(_tick(m))
            for ph in (m.get("lens") or {}).get("verbatim") or []:
                own = " ".join(rd.utt_by_id[u]["text"].lower() for u in m.get("utterance_ids") or []
                               if u in rd.utt_by_id and rd.utt_by_id[u]["speaker"] == m.get("agent"))
                (stuck_self if str(ph).lower() in own else stuck_other).append(_tick(m))
        return {"events_released": sorted(released.values()), "events_witnessed": sorted(witnessed.values()),
                "events_discussed": sorted(discussed.values()), "memory_links": sorted(_tick(r) for r in links),
                "cross_incident_links": sorted(cross),
                "reminding_linked": sorted(_tick(r) for r in rd.of("reminding") if r.get("reminded_of")),
                "stuck_self": sorted(stuck_self), "stuck_other": sorted(stuck_other)}

    def reuse_ticks(self, recs: list[dict]) -> list[int]:
        """First carried use tick of every (expression, carried adopter) pair."""
        return sorted(a["carried_tick"] for r in recs for a in r["_adopters"] if a["role"] == "carried" and a["carried_tick"] is not None)

    def funnel(self) -> dict:
        med = dict(self.mediators, reuse_after_exposure=self.reuse_ticks(self.pool))
        cum = {k: len(v) for k, v in med.items()}
        days = self.t // self.tpd + 1
        by_day = []
        for d in range(1, days + 1):
            lo, hi = (d - 1) * self.tpd, d * self.tpd
            by_day.append({"day": d, **{k: sum(1 for x in v if lo <= x < hi) for k, v in med.items()}})
        return {"cumulative": cum, "by_day": by_day,
                "definitions": {
                    "events_released": "world events (v2 incidents, v3 jobs/cues/tallies/...) with facts released by t",
                    "events_witnessed": "released events perceived by at least one agent",
                    "events_discussed": "released events some utterance refers to (provenance referent links)",
                    "memory_links": "memory_link records (reminding, association, merge, reflection)",
                    "cross_incident_links": "memory links between memories of disjoint event sets",
                    "reminding_linked": "reminding attempts that linked to an earlier memory",
                    "stuck_self": "verbatim wordings kept from one's own utterances",
                    "stuck_other": "verbatim wordings kept from someone else's utterance or writing",
                    "reuse_after_exposure": "(expression, agent) pairs with a carried use after exposure"}}

    # ---------------------------------------------------------------- v3
    def is_v3(self) -> bool:
        cfg = self.rd.cfg or {}
        return bool((cfg.get("workshop") or {}).get("enabled")) or bool(self.rc.trace.records and any(
            r.get("type") in ("job_start", "record_write", "roster_change") for r in self.rd.trace))

    def v3(self) -> dict:
        from backend.analysis import jobs as J
        from backend.analysis import v3common as V
        rd, t, cfg = self.rd, self.t, self.rd.cfg or {}
        started = {r.get("job"): _tick(r) for r in rd.of("job_start") if r.get("job")}
        try:
            table = {j: r for j, r in J.job_table(rd).items() if j in started}
        except Exception:   # noqa: BLE001 - partial or foreign job traces never break the live view
            table = {}
        days = sorted({r["day"] for r in table.values() if r.get("day")})
        ends = {r.get("job"): r for r in rd.of("job_end") if r.get("job")}
        jobs = {"started": len(started), "decided": sum(1 for r in table.values() if r.get("first_action")),
                "ended": len(ends), "delivered": sum(1 for r in ends.values() if r.get("delivered")),
                "not_delivered": sum(1 for r in ends.values() if r.get("delivered") is False),
                "deferred": sum(1 for r in table.values() if r.get("deferred")),
                "asks": len(rd.of("clarification")),
                "by_day": {d: {"started": sum(1 for j, r in table.items() if r.get("day") == d),
                               "delivered": sum(1 for j, e in ends.items() if (table.get(j) or {}).get("day") == d and e.get("delivered"))}
                           for d in days}}
        sd = V.shift_day(cfg)
        reg = [r for r in rd.of("regime_active")]
        cur = (max(reg, key=_tick).get("regime") if reg else V.regime_on_day(cfg, t // self.tpd + 1))
        post = [d for d in days if sd and d >= sd]
        writes = rd.of("record_write")
        made = [w for w in writes if w.get("choice") not in (None, "none")]
        reads = rd.of("record_read")
        trans = rd.of("record_transition")
        last_wipe = max((_tick(r) for r in trans if r.get("mode") == "wipe"), default=None)
        cur_items = [w for w in made if last_wipe is None or _tick(w) >= last_wipe]
        fronts = [w for w in cur_items if w.get("choice") == "front"]
        last_change = max([_tick(w) for w in made] + ([last_wipe] if last_wipe is not None else []), default=None)
        departed = set(self.roster["departed"])
        chars = sum(len(w.get("text") or "") for w in cur_items)
        front_now = max(fronts, key=_tick) if fronts else None
        binder = {"offers": len(writes), "writes": len(made),
                  "log_entries": sum(1 for w in made if w.get("choice") == "log"),
                  "front_rewrites": sum(1 for w in made if w.get("choice") == "front"),
                  "reads": len(reads), "readers": len({r.get("agent") for r in reads}),
                  "new_items_read": sum(len(r.get("new_ids") or []) for r in reads),
                  "transitions": [{"tick": _tick(r), "day": r.get("day"), "mode": r.get("mode")} for r in trans],
                  "current_items": len(cur_items), "last_write_tick": max((_tick(w) for w in made), default=None),
                  "staleness_ticks": (t - last_change) if last_change is not None else None,
                  "front_age_ticks": (t - _tick(front_now)) if front_now else None,
                  "front_author": front_now.get("agent") if front_now else None,
                  "departed_author_share": round(sum(len(w.get("text") or "") for w in cur_items
                                                     if w.get("agent") in departed) / chars, 4) if chars else None,
                  "writes_by_day": {d: sum(1 for w in made if _tick(w) // self.tpd + 1 == d)
                                    for d in range(1, t // self.tpd + 2)}}
        return {
            "jobs": jobs, "binder": binder,
            "roster": {k: self.roster[k] for k in ("active", "departed", "cohorts", "roles", "active_by_cohort", "changes")},
            "checkpoints": [{"id": r.get("id"), "tick": _tick(r)} for r in rd.of("checkpoint")],
            "first_attempt_accuracy": {"hidden": True, "by_day": {d: J.first_attempt_accuracy(table, [d]) for d in days},
                                       "all": J.first_attempt_accuracy(table)},
            "old_regime_response": {"hidden": True, "shift_day": sd, "applicable": bool(post),
                                    "by_day": {d: J.old_new_other(table, cfg, [d]) for d in post},
                                    "post_shift": J.old_new_other(table, cfg, post) if post else None,
                                    "switch_latency": J.switch_latency(table, cfg, sd) if post else None},
            "k1_vs_k2": {"hidden": True, **{c: J.first_attempt_accuracy(table, None, (c,)) for c in ("K1", "K2", "K3")}},
            "regime": {"hidden": True, "current": cur, "mapping": V.mapping_of(cfg), "shift_day": sd,
                       "changed": bool(sd and t // self.tpd + 1 >= sd)},
        }

    # ---------------------------------------------------------------- snapshot
    def snapshot(self, latest: int, status: str | None, top: int) -> dict:
        rd, top = self.rd, int(top)
        pool = self.pool
        shown = pool[:top]
        # the control (one row, the proof detection works) and EVERY study meme are always shown: a
        # meme that fell out of the top N is the most informative row in the table, not the least
        shown = shown + [r for r in pool[top:] if r["control"]][:1] + [r for r in pool[top:] if r["meme_id"]]
        counts = {s: 0 for s in STATUSES}
        for r in pool:
            counts[r["status"]] += 1
        pl = self.planted
        js = self.rc.vindex.state()
        out = {
            "run_id": self.rc.dir.name, "live_version": LIVE_VERSION, "tick": self.t, "latest_tick": latest,
            "day": self.t // self.tpd + 1, "time_label": _label(self.man, self.t), "ticks_per_day": self.tpd,
            "run_status": status, "tick_complete": status == "finished" or self.t < latest,
            "n_utterances": len(rd.utterances), "n_conversations": len(rd.conversations),
            "n_events": len(rd.events), "n_agents_active": len(self.roster["active"]),
            "expressions": [_public(r) for r in shown], "n_expressions": len(pool), "status_counts": counts,
            "bucket_counts": {b: sum(1 for r in pool if r["bucket"] == b) for b in BUCKETS},
            "n_culture_candidates": sum(1 for r in pool if r["bucket"] == "expression"),
            "tier_counts": tier_counts(pool), "n_conventions": tier_counts(pool)["convention"],
            "judge": {**js, "real_judge": js["status"] == "real"},
            "transmission": {"expressions": [r["id"] for r in shown[: int(self.p["transmission_top"])]],
                             "edges": self.edges(shown[: int(self.p["transmission_top"])])},
            "funnel": self.funnel(),
            "planted": ({"agent": pl["agent"], "phrase": pl["phrase"],
                         "uses": sum(r["uses"] for r in pool if r["control"]),
                         "status": next((r["status"] for r in pool if r["control"]), None)} if pl else None),
            "memes": self.meme_rows(pool),
            "params": {"top": top, **{k: self.p[k] for k in ("pool", "min_speakers", "min_uses", "trend_window", "transmission_top")}},
            "hidden_fields": list(HIDDEN_FIELDS),
        }
        if self.is_v3():
            out["v3"] = self.v3()
        return out


def _public(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def _resolve(rc: _RunCache, tick) -> tuple[int, int]:
    latest = rc.trace.max_tick
    if tick is None:
        return max(latest, 0), latest
    t = int(tick)
    return (max(0, min(t, latest)) if latest >= 0 else max(t, 0)), latest


# ------------------------------------------------------------------------------------------------ public API
def _snap(rc: _RunCache, sig, t: int, p: dict) -> "_Snap":
    key = tuple(sorted((k, v) for k, v in p.items() if k != "top"))
    return rc.memo(("snap", sig, t, key), lambda: _Snap(rc, t, p))


def live_snapshot(run_dir, tick: int | None = None, top: int = 20, *, trend_window: int | None = None) -> dict:
    """LLM-free analysis of `run_dir` up to `tick` (None -> the latest tick written). JSON-serializable.

    Keys: run_id, live_version, tick (clamped to the latest written tick), latest_tick, day, time_label,
    ticks_per_day, run_status, tick_complete, n_utterances, n_conversations, n_events, n_agents_active,
    expressions (top N records: status, flags, tier, tier_reasons, verdict, world_wording{world, system, ...}),
    n_expressions, status_counts, tier_counts, n_conventions, judge {status: real|mock_only|none, note},
    transmission {expressions, edges}, funnel {cumulative, by_day, definitions}, planted, memes (the
    injected cohort per registry id, or None), params, hidden_fields, and v3 (co-op runs only)."""
    rc = _cache(run_dir)
    with rc.lock:
        sig = rc.refresh()
        t, latest = _resolve(rc, tick)
        p = _params(rc.cfg, top, trend_window)
        status = rc.manifest.get("status")
        key = ("snapshot", sig, t, tuple(sorted(p.items())))
        return copy.deepcopy(rc.memo(key, lambda: _snap(rc, sig, t, p).snapshot(latest, status, p["top"])))


def snapshot_context(run_dir, tick: int | None = None, top: int = 30) -> "_Snap":
    """The cached per-tick analysis object (report.py uses its prefix RunData and helpers). Read-only."""
    rc = _cache(run_dir)
    with rc.lock:
        sig = rc.refresh()
        t, _latest = _resolve(rc, tick)
        sn = _snap(rc, sig, t, _params(rc.cfg, top))
        sn.pool, sn.mediators          # materialize the cached parts under the run lock
        return sn


def expression_pool(run_dir, tick: int | None = None, top: int = 30) -> tuple[int, list[dict]]:
    """(tick, full expression records incl. private `_usages` / `_adopters`) for the report and judges."""
    rc = _cache(run_dir)
    with rc.lock:
        sn = snapshot_context(run_dir, tick, top)
        return sn.t, copy.deepcopy(sn.pool)


def live_timeline(run_dir, step: int | None = None, *, top: int = 20) -> dict:
    """Headline numbers per tick bucket (0, step, 2*step, ..., latest) for charts.

    Expressions are the pool found at the latest tick, re-evaluated on their uses up to each bucket
    (an expression counts once it meets min_uses/min_speakers; world wording counts from its release
    tick). series[i]: {tick, day, n_utterances, n_conversations, n_events, n_agents_active, n_expressions,
    status_counts, tier_counts, funnel (cumulative counts), v3 (public, co-op runs), v3_hidden ({"hidden": true, ...})}."""
    rc = _cache(run_dir)
    with rc.lock:
        sig = rc.refresh()
        latest = max(rc.trace.max_tick, 0)
        tpd = int(rc.manifest.get("ticks_per_day") or 60)
        step = max(1, int(step)) if step else max(1, tpd // 4)
        p = _params(rc.cfg, top)
        key = ("timeline", sig, step, tuple(sorted(p.items())))
        return copy.deepcopy(rc.memo(key, lambda: _timeline(rc, latest, step, p)))


def _timeline(rc: _RunCache, latest: int, step: int, p: dict) -> dict:
    sn = _Snap(rc, latest, p)
    rd = sn.rd
    ticks = list(range(0, latest + 1, step))
    if not ticks or ticks[-1] != latest:
        ticks.append(latest)
    ut = sorted(int(u["tick"]) for u in rd.utterances)
    ct = sorted(_tick(c) for c in rd.conversations.values())
    et = sorted(int(e.get("start_tick") or 0) for e in rd.events.values())
    med = sn.mediators
    pool = sn.pool
    v3 = sn.is_v3()
    if v3:
        from backend.analysis import jobs as J
        started = sorted(_tick(r) for r in rd.of("job_start"))
        delivered = sorted(_tick(r) for r in rd.of("job_end") if r.get("delivered"))
        writes = sorted(_tick(w) for w in rd.of("record_write") if w.get("choice") not in (None, "none"))
        reads = sorted(_tick(r) for r in rd.of("record_read"))
        try:
            table = J.job_table(rd)
        except Exception:   # noqa: BLE001
            table = {}
    series = []
    for b in ticks:
        counts = {s: 0 for s in STATUSES}
        tiers = {t: 0 for t in TIERS}
        memes: dict[str, dict] = {}
        reuse = 0
        n_expr = 0
        for r in pool:
            if r["meme_id"]:      # the cohort's own curve: who is still saying each meme at tick b
                us = [u for u in r["_usages"] if u["tick"] <= b]
                em = emergence_for(us, sn.population)
                memes[r["meme_id"]] = {"uses": len(us), "speakers": len({u["speaker"] for u in us}),
                                       "carried_adopters": em.get("n_adopters_carried") or 0}
            us = [u for u in r["_usages"] if u["tick"] <= b]
            if len(us) < int(p["min_uses"]) or len({u["speaker"] for u in us}) < int(p["min_speakers"]):
                continue
            n_expr += 1
            ww = r["world_wording"]
            in_world = ww["world_tick"] is not None and ww["world_tick"] <= b
            system = bool(ww.get("system")) and (ww.get("system_tick") is None or ww["system_tick"] <= b)
            world = in_world or (ww["factual_repetition"] and not system)
            em = emergence_for(us, sn.population, in_lexicon=ww["in_lexicon"], in_world_text=in_world,
                               in_system_text=system)
            st, _ = classify_status(em, world=world, planted=r["control"], system=system, meme=bool(r["meme_id"]))
            counts[st] += 1
            real, mock = rc.vindex.lookup(forms(r), lambda row, b=b: (row.get("tick") is None and b >= latest) or
                                          (row.get("tick") is not None and int(row["tick"]) <= b))
            tr, _ = tier_of(em, system=system, world=world, planted=r["control"], verdict=(real or {}).get("verdict"),
                            placeholder=(mock or {}).get("verdict"))
            if not r["control"]:      # the control is not evidence for the thing it tests for; a meme is
                tiers[tr] += 1
            reuse += em.get("n_adopters_carried") or 0
        row = {"tick": b, "day": b // sn.tpd + 1, "n_utterances": bisect_right(ut, b), "n_conversations": bisect_right(ct, b),
               "n_events": bisect_right(et, b), "n_agents_active": len(roster_at(rd, b)["active"]),
               "n_expressions": n_expr, "status_counts": counts, "tier_counts": tiers,
               "funnel": {**{k: bisect_right(v, b) for k, v in med.items()}, "reuse_after_exposure": reuse}}
        if memes:
            row["memes"] = memes
        if v3:
            row["v3"] = {"jobs_started": bisect_right(started, b), "jobs_delivered": bisect_right(delivered, b),
                         "binder_writes": bisect_right(writes, b), "binder_reads": bisect_right(reads, b)}
            decided = {j: r for j, r in table.items() if r.get("first_tick") is not None and r["first_tick"] <= b}
            row["v3_hidden"] = {"hidden": True, "first_attempt_accuracy": J.first_attempt_accuracy(decided)}
        series.append(row)
    return {"run_id": rc.dir.name, "live_version": LIVE_VERSION, "step": step, "latest_tick": latest,
            "ticks_per_day": sn.tpd, "run_status": rc.manifest.get("status"), "series": series,
            "hidden_fields": list(TIMELINE_HIDDEN_FIELDS) if v3 else []}


def strip_hidden(obj):
    """A deep copy without hidden-derived metrics: every dict value marked {"hidden": true} is removed
    (lists are filtered the same way). The API calls this in demo mode."""
    if isinstance(obj, dict):
        return {k: strip_hidden(v) for k, v in obj.items() if not (isinstance(v, dict) and v.get("hidden") is True)}
    if isinstance(obj, list):
        return [strip_hidden(v) for v in obj if not (isinstance(v, dict) and v.get("hidden") is True)]
    return copy.deepcopy(obj)
