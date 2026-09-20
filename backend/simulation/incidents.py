"""Registry-driven origin incidents: the GROUNDED arm of the injected meme cohort. WORLD layer.

An incident is the *witnessable basis* of one coined expression: a recurring, located condition that some
people run into and most do not. It is declared entirely by `memes.registry[*].incident`; there is one code
path and N incidents, and adding, removing or swapping a meme is a config edit. An entry whose `incident` is
absent or null (the UNGROUNDED arm) simply contributes no condition - it falls out of the loop that builds
`self.conditions`, which is what makes the grounded/ungrounded contrast a property of the config rather than
of a branch in the code.

What the world does and does not do here:

* An incident is NOT a one-shot beat. On each configured day, an agent who is at the configured place during
  one of the configured windows (a meal, a lab shift) runs into the condition with `prob_per_meal`, drawn from
  that agent's own seeded substream, `seed_rng(agent.seed, agent.id, "incident", meme, day, occasion)`. The
  draw is KEYED by the occasion rather than taken from a running `Agent.stream()` cursor, so an agent who ate
  somewhere else yesterday gets exactly the same draw today: presence history can never shift an encounter
  (common random numbers, v2 §1.1).
* Each encounter is one fact that `involves` the person it happened to, so they perceive it with p = 1 and the
  people around them perceive it at arena visibility under ordinary attention. Which surface sentence a witness
  gets is also drawn from their own substream, so two people at the same table carry away different partial
  views of the same condition before the viewpoint renderer (D42) and the lossy encoder (D44) have even run.
  Nothing here is protected: an incident memory decays, competes and is forgotten like any other experience.
* `repaired_day` removes the basis. The repair is PERCEPTIBLE - the same machinery releases
  `repaired_surfaces` for a matched number of days - but never EVALUATED: no narrator, no "the problem is
  fixed", no causal gloss. D39 is the standing decision here: the world states what a person could see, and
  says nothing about what it means.

R1/R2 (the two contamination rules) are enforced here, not promised. `IncidentWorld` refuses to construct if
any incident surface emits any registry phrase or any of its distinctive words, so a contaminated config fails
before the first LLM call rather than after a paid run. `audit()` is the same check as a function, for the
other builders' text and for the test that scans the whole world's vocabulary.

Nothing in this module reads observer output, and `probe_gradient` / `foils` are carried through untouched and
never rendered into a fact (R7).
"""
from __future__ import annotations

import re
from collections import defaultdict

from backend.simulation.rngs import seed_rng
from backend.simulation.world import ARENAS, WORLD_GRAPH, _minutes

GENERATOR = "incidents_v1"
GROUNDINGS = ("grounded", "ungrounded")
BREADTHS = ("broad", "narrow")

# The run's meal windows (the fixed strings of this build). Only a default: an incident that happens on a lab
# shift rather than at a meal declares its own `windows`.
MEAL_WINDOWS = ("11:30-13:30", "17:30-19:00")

DEFAULT_SALIENCE = 0.6      # >= reaction.min_salience (0.5), so running into the condition can prompt a remark
# Uncapped by default. A per-tick cap on encounters looks like harmless cost control and is a confound: it
# bites hardest where most people are, so two memes with the same `prob_per_meal` end up with different
# realised exposure purely because one site is busier (measured on homewood100: the cap dropped 75 encounters
# in 2 days and left the quiet wet lab ahead of the crowded dining hall, 55 to 42). `prob_per_meal` has to be
# the only thing that sets exposure, or R6 is gone. The knob stays for runs that need it; using it with
# differently-sized sites is a confound the analysis must then carry.
DEFAULT_MAX_PER_TICK = None
FAULT, REPAIRED = "fault", "repaired"

# A hyphen SPLITS: the distinctive words of "blue-tray" are "blue" and "tray" separately, because an agent
# handed either of them can build the compound, and the world writes the compound with a space as readily as
# with a hyphen.
_TOK = re.compile(r"[A-Za-z][A-Za-z']*")
# Words that carry none of a coinage's identity, so they are never what makes a phrase distinctive. Deliberately
# small: anything outside this list counts as distinctive, because under-flagging is the failure that ruins the
# experiment and over-flagging only costs a config author one explicit exemption.
_PHRASE_STOP = frozenset("""
a an the this that these those some any each every no all both of in on at to for from by with without
and or but so as is are was were be been being it its their his her your my our one two
""".split())


# --------------------------------------------------------------------------------- R1 / R2 word machinery
def content_words(phrase: str) -> tuple[str, ...]:
    """The tokens of a coined phrase that carry its identity ("the Tuesday list" -> tuesday, list)."""
    seen, out = set(), []
    for t in _TOK.findall(str(phrase or "")):
        w = t.lower().strip("'-")
        if len(w) > 1 and w not in _PHRASE_STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return tuple(out)


def _word_re(word: str) -> re.Pattern:
    """`word` and its forward inflections. Matching inflections matters: a world that says "trays" has handed
    the agent "tray" for every purpose an independent coinage needs."""
    alts = [re.escape(word) + r"(?:s|es|ed|ing|'s)?"]
    if word.endswith("y"):
        alts.append(re.escape(word[:-1]) + "ies")
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.I)


def _phrase_re(phrase: str) -> re.Pattern:
    """The phrase itself, indifferent to hyphen vs space vs case ("blue-tray" == "Blue tray")."""
    words = [re.escape(t) for t in _TOK.findall(str(phrase or ""))]
    return re.compile(r"\b" + r"[\s\-]+".join(words) + r"\b", re.I) if words else re.compile(r"(?!x)x")


def audit(entries: list[dict], texts) -> list[dict]:
    """R1 + R2 as a machine check: every (meme, word, text) collision in `texts`.

    `texts` is an iterable of strings or of (where, text) pairs. The check reads the registry, so it keeps
    working when the prior check swaps a phrase out. A hit on `kind: "phrase"` is R1 (the world said the
    expression); a hit on `kind: "word"` is R2 (the world handed over a word the expression is built from, so
    an agent could coin it independently and transmission stops being distinguishable from rediscovery).
    """
    out = []
    for text in texts:
        where, body = text if isinstance(text, (tuple, list)) else (None, text)
        body = str(body or "")
        if not body:
            continue
        for e in entries:
            if e["phrase_re"].search(body):
                out.append({"meme": e["id"], "kind": "phrase", "word": e["phrase"], "where": where, "text": body})
            for w, rx in e["word_res"].items():
                if rx.search(body):
                    out.append({"meme": e["id"], "kind": "word", "word": w, "where": where, "text": body})
    return out


# ------------------------------------------------------------------------------------------- the registry
def _str(entry: dict, key: str, where: str, required: bool = True) -> str:
    v = entry.get(key)
    if v is None and not required:
        return ""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{where}: `{key}` must be a non-empty string, got {v!r}")
    return v.strip()


def _windows(raw, where: str) -> list[tuple[int, int]]:
    out = []
    for w in (raw if raw is not None else list(MEAL_WINDOWS)):
        parts = str(w).split("-")
        if len(parts) != 2:
            raise ValueError(f"{where}: window {w!r} must read 'HH:MM-HH:MM'")
        lo, hi = (_minutes(p.strip()) for p in parts)
        if hi <= lo:
            raise ValueError(f"{where}: window {w!r} ends before it starts")
        out.append((lo, hi))
    if not out:
        raise ValueError(f"{where}: `windows` must not be empty")
    return sorted(out)


def _surfaces(raw, where: str, key: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}: `{key}` must be a non-empty list of surface sentences")
    out = []
    for s in raw:
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"{where}: every entry of `{key}` must be a non-empty string")
        if "{P}" not in s:
            raise ValueError(f"{where}: surface {s!r} must contain {{P}}. An incident is something that happens "
                             f"TO someone: the fact names the person so they perceive it as a participant and "
                             f"the people around them perceive it as a bystander's partial view.")
        out.append(s.strip())
    return out


def _incident(raw: dict, where: str) -> dict:
    loc = _str(raw, "location", where)
    if loc not in WORLD_GRAPH:
        raise ValueError(f"{where}: unknown location {loc!r}")
    arena = _str(raw, "arena", where)
    if arena not in ARENAS[loc]:
        raise ValueError(f"{where}: {arena!r} is not a room of {loc!r} ({ARENAS[loc]})")
    days = [int(d) for d in (raw.get("days") or [])]
    if not days or any(d < 1 for d in days):
        raise ValueError(f"{where}: `days` must be a non-empty list of day numbers (1-based)")
    days = sorted(set(days))
    prob = float(raw.get("prob_per_meal", 0.0))
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"{where}: `prob_per_meal` must be in [0, 1], got {prob}")
    rep = raw.get("repaired_day")
    rep = None if rep is None else int(rep)
    if rep is not None and rep <= max(days):
        raise ValueError(f"{where}: `repaired_day` ({rep}) must come after the last incident day ({max(days)})")
    # Matched exposure by default: the repaired condition is perceptible for as many days as the fault was, so
    # "nobody saw the repair" can never be an artefact of giving it a shorter window than the thing it removes.
    rep_days = raw.get("repaired_days")
    if rep is None:
        rep_days = []
    elif rep_days is None:
        rep_days = list(range(rep, rep + len(days)))
    rep_days = sorted({int(d) for d in rep_days})
    sal = float(raw.get("salience", DEFAULT_SALIENCE))
    if not 0.0 <= sal <= 1.0:
        raise ValueError(f"{where}: `salience` must be in [0, 1], got {sal}")
    cap = raw.get("max_per_tick", DEFAULT_MAX_PER_TICK)
    cap = None if cap is None else int(cap)
    if cap is not None and cap < 1:
        raise ValueError(f"{where}: `max_per_tick` must be >= 1 or null (uncapped), got {cap}")
    return {"location": loc, "arena": arena, "days": days, "repaired_day": rep, "repaired_days": rep_days,
            "prob_per_meal": prob, "windows": _windows(raw.get("windows"), where), "salience": sal,
            "max_per_tick": cap, "surfaces": _surfaces(raw.get("surfaces"), where, "surfaces"),
            "repaired_surfaces": _surfaces(raw.get("repaired_surfaces"), where, "repaired_surfaces")
            if rep is not None else []}


def _one(raw: dict, i: int) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"memes.registry[{i}]: every entry must be a mapping")
    mid = _str(raw, "id", f"memes.registry[{i}]")
    where = f"memes.registry[{mid}]"
    phrase = " ".join(_str(raw, "phrase", where).split())
    grounding, breadth = _str(raw, "grounding", where), _str(raw, "breadth", where)
    if grounding not in GROUNDINGS:
        raise ValueError(f"{where}: `grounding` must be one of {GROUNDINGS}, got {grounding!r}")
    if breadth not in BREADTHS:
        raise ValueError(f"{where}: `breadth` must be one of {BREADTHS}, got {breadth!r}")
    inc = raw.get("incident")
    # The 2x2 is the design, so the config states it once and the code holds it: "grounded" MEANS there is a
    # witnessed origin event and "ungrounded" MEANS there is none. A mismatch here would silently move a cell.
    if grounding == "grounded" and not inc:
        raise ValueError(f"{where}: grounding: grounded needs an `incident` block (that is what grounds it)")
    if grounding == "ungrounded" and inc:
        raise ValueError(f"{where}: grounding: ungrounded must not declare an `incident` "
                         f"(the ungrounded arm has no basis to remove - that is what makes it the comparison)")
    derived = content_words(phrase)
    declared = raw.get("distinctive_words")
    if declared is None:
        words = list(derived)
        exempt: list[str] = []
    else:
        if not isinstance(declared, list) or not declared:
            raise ValueError(f"{where}: `distinctive_words`, when given, must be a non-empty list")
        words = sorted({str(w).strip().lower() for w in declared if str(w).strip()})
        # Narrowing the ban below the phrase's own content words is legitimate ("blue-tray" needs the world to
        # be able to say "tray"), but it is the one loosening of R2 available, so it is recorded and reported
        # rather than silently applied.
        exempt = sorted(set(derived) - set(words))
    return {"id": mid, "phrase": phrase, "grounding": grounding, "breadth": breadth, "cell": f"{grounding}/{breadth}",
            "habit": _str(raw, "habit", where, required=False), "seeds": dict(raw.get("seeds") or {}),
            "incident": _incident(inc, f"{where}.incident") if inc else None,
            "probe_gradient": dict(raw.get("probe_gradient") or {}),   # OBSERVER ONLY (R7); never rendered
            "foils": list(raw.get("foils") or []),
            "content_words": list(derived), "distinctive_words": words, "exempt_words": exempt,
            "phrase_re": _phrase_re(phrase), "word_res": {w: _word_re(w) for w in words}}


def registry(cfg: dict | None) -> list[dict]:
    """The validated meme registry, in config order; `[]` when `memes.enabled` is false or absent.

    THE central abstraction of this build: no meme is ever named in code, and adding, removing or swapping one
    is a config edit.

    The shared fields (id, phrase, grounding, breadth, habit, seeds) are validated by
    `backend.agents.profile.meme_registry`, which the injection side already owns: one authority for the
    invariants both sides depend on, so a phrase swap can never leave the two halves of the build disagreeing
    about what the phrase is. This function adds the WORLD-side layer on top - the incident block, and the
    R1/R2 word sets derived from the phrase.
    """
    mc = (cfg or {}).get("memes") or {}
    if not mc.get("enabled"):
        return []
    raw = mc.get("registry")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("memes.registry must be a list of meme entries")
    try:
        from backend.agents.profile import meme_registry
    except ImportError:                                   # pragma: no cover - the injection side is not in yet
        meme_registry = None
    if meme_registry is not None:
        meme_registry(cfg)                                # raises on any shared-field violation
    out = [_one(e, i) for i, e in enumerate(raw)]
    seen: dict[str, int] = {}
    for i, e in enumerate(out):
        if e["id"] in seen:
            raise ValueError(f"memes.registry: duplicate id {e['id']!r} (entries {seen[e['id']]} and {i})")
        seen[e["id"]] = i
    return out


# ------------------------------------------------------------------------- what this config can say out loud
_PROMPT_MARK = "<commentblockmarker>###</commentblockmarker>"   # upstream: the header above it is never sent
_OBSERVER_PROMPTS = ("probe_", "judge_")                        # observer-side; never reaches an agent


def _prompt_reachable(name: str, cfg: dict) -> bool:
    if name.startswith(_OBSERVER_PROMPTS):
        return False
    if name.startswith("commons_"):
        return (cfg.get("world") or {}).get("mode") == "commons"
    if name.startswith("job_"):
        return bool((cfg.get("workshop") or {}).get("enabled"))
    if name.startswith("record_"):
        return bool((cfg.get("records") or {}).get("enabled"))
    return True


def world_vocabulary(cfg: dict, profiles: dict | None = None, all_sources: bool = False) -> list[tuple[str, str]]:
    """(where, text) for every string THIS config can put in front of an agent.

    Reachability is part of the answer, not a detail: `referents.CATEGORIES` holds "the Blue Line shuttle" and
    the skins hold "sun-warmed", but with `latent_events.event_rate: 0` - which is what the meme runs use - no
    agent can ever be shown either. Refusing a phrase over a string the run cannot emit would be as wrong as
    missing one it can. `all_sources=True` ignores the gates, for a report that wants the whole picture.

    Persona identity text is deliberately NOT scanned. That is where the seeded minority's habit line lives,
    and the habit line is the one place a registry phrase is allowed to appear (R1).
    """
    from backend.simulation import reference
    from backend.simulation.world import ARENA_DESCRIPTIONS, ARENAS, PLACE_DESCRIPTIONS
    out: list[tuple[str, str]] = []
    situated = reference.mode(cfg) == reference.SITUATED
    if all_sources or situated:
        out += [(f"world.PLACE_DESCRIPTIONS[{k}]", v) for k, v in PLACE_DESCRIPTIONS.items()]
        out += [(f"world.ARENA_DESCRIPTIONS[{loc}][{a}]", t)
                for loc, d in ARENA_DESCRIPTIONS.items() for a, t in d.items()]
    if all_sources or not situated:
        out += [(f"world.ARENAS[{loc}]", f"{loc} {a}") for loc, arenas in ARENAS.items() for a in arenas]
    from backend.simulation import scheduler as SCH
    rc = cfg.get("routine") or {}
    if all_sources or float(rc.get("deviation_prob") or 0) > 0:
        out += [(f"scheduler.DETOURS[{k}]", v) for k, v in SCH.DETOURS.items()]
    if all_sources or int(rc.get("errands_per_day") or 0) > 0:
        places = SCH.errand_places(cfg) if not all_sources else list(SCH.ERRANDS)
        out += [(f"scheduler.ERRANDS[{p}]", a) for p in places for _, a in SCH.errand_options(p)]
    if all_sources or situated:
        out += [("scheduler.SITUATED_ACTIVITIES", v) for v in SCH.SITUATED_ACTIVITIES.values()]
    lc = cfg.get("latent_events") or {}
    if all_sources or float(lc.get("event_rate") or 0) > 0:
        from backend.simulation.skins import load_skins
        for s in load_skins():
            out += [(f"skins[{s['key']}].{n}", t) for n, t in s["facts"].items()]
            out += [(f"skins[{s['key']}].slots[{slot}]", o)
                    for slot, opts in (s.get("slots") or {}).items() for o in opts]
        if all_sources or (lc.get("referents") or {}).get("enabled"):
            from backend.simulation.referents import CATEGORIES
            out += [(f"referents.CATEGORIES[{d}]", n) for d, names in CATEGORIES.items() for n in names]
    from backend import ga_compat
    for p in sorted((ga_compat.REPO_ROOT / "backend" / "prompts").glob("*.txt")):
        if all_sources or _prompt_reachable(p.name, cfg):
            out.append((f"prompts/{p.name}", p.read_text(encoding="utf-8").split(_PROMPT_MARK, 1)[-1]))
    # Routine ACTIVITY wording is world text: it is written into day plans, into `move` activities, and into
    # what everyone in the room perceives someone else doing ("Priya is running assays at ...").
    for aid, prof in sorted((profiles or {}).items()):
        out += [(f"profile[{aid}].routine", r.activity or "") for r in (prof.routine or [])]
    return [(w, t) for w, t in out if t]


def preflight(cfg: dict, profiles: dict | None = None) -> list[dict]:
    """R1/R2 over the whole world this config can speak, before the run starts. Empty means clean."""
    return audit(registry(cfg), world_vocabulary(cfg, profiles))


def cohort_summary(entries: list[dict]) -> dict:
    """The 2x2 as the config actually fills it, for the manifest and for a quick eyeball of R6."""
    cells: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        cells[e["cell"]].append(e["id"])
    return {c: sorted(cells[c]) for c in sorted(cells)}


# ------------------------------------------------------------------------------------------ the mechanism
class _Ev:
    """Event handle for the engine's beat loop: `id` keys the per-(agent, beat) perception stream and becomes
    the observation's `event_ids`, so every memory of an incident is traceable back to its meme."""

    def __init__(self, eid: str, meme: str, state: str):
        self.id, self.meme, self.state = eid, meme, state
        self.kind, self.latent_type = "incident", "incident"


class Condition:
    """One registry entry's incident: where it is, when it is there, and what a witness can perceive."""

    def __init__(self, entry: dict):
        self.meme, self.entry = entry["id"], entry
        inc = entry["incident"]
        for k in ("location", "arena", "days", "repaired_day", "repaired_days", "prob_per_meal", "windows",
                  "salience", "max_per_tick", "surfaces", "repaired_surfaces"):
            setattr(self, k, inc[k])
        self.place = (self.location, self.arena)

    def state_on(self, day: int) -> str | None:
        if day in self.days:
            return FAULT
        return REPAIRED if day in self.repaired_days else None

    def occasion(self, minutes: int) -> int | None:
        """Index of the window `minutes` falls in (one draw per agent per occasion), else None."""
        for i, (lo, hi) in enumerate(self.windows):
            if lo <= minutes <= hi:
                return i
        return None

    def script_record(self, clock) -> dict:
        """The world-script line for this condition. Encounters are not pre-drawn, because who is at the
        dining hall at 12:15 is settled by routines and by the agents' own re-planning; what IS pre-generated
        and fixed by `world_seed` is the condition itself - where, when and what can be seen."""
        first = min(self.days + (self.repaired_days or []))
        return {"kind": "incident", "generator": GENERATOR, "id": f"inc.{self.meme}", "meme": self.meme,
                "grounding": self.entry["grounding"], "breadth": self.entry["breadth"],
                "location": self.location, "arena": self.arena, "days": list(self.days),
                "repaired_day": self.repaired_day, "repaired_days": list(self.repaired_days),
                "windows": [f"{lo // 60:02d}:{lo % 60:02d}-{hi // 60:02d}:{hi % 60:02d}"
                            for lo, hi in self.windows],
                "prob_per_meal": self.prob_per_meal, "salience": self.salience,
                "max_per_tick": self.max_per_tick, "surfaces": list(self.surfaces),
                "repaired_surfaces": list(self.repaired_surfaces),
                "start_tick": (first - 1) * clock.ticks_per_day, "tick": (first - 1) * clock.ticks_per_day}


def _render(text: str, first: str) -> str:
    out = text.replace("{P}", first)
    return out[0].upper() + out[1:] if out else out


class IncidentWorld:
    """The engine-facing facade: one `world(tick)` hook, in the same (event, beat) shape as everything else.

    Constructed only when `memes.enabled`, so with the mechanism off the engine never reaches this module and
    an existing run reproduces byte for byte (R5).
    """

    def __init__(self, sim):
        self.sim, self.cfg, self.clock = sim, sim.cfg, sim.clock
        self.agents = sim.agents
        self.entries = registry(self.cfg)
        self.conditions = [Condition(e) for e in self.entries if e["incident"]]
        self._check_r2()
        self.drawn: set = set()                                   # (agent, meme, day, occasion) already drawn
        self.counts: dict[str, int] = defaultdict(int)
        self.encounters: dict[str, dict[str, int]] = {e["id"]: {FAULT: 0, REPAIRED: 0} for e in self.entries}
        self.witnesses: dict[str, dict[str, set]] = {e["id"]: {FAULT: set(), REPAIRED: set()}
                                                     for e in self.entries}

    # ------------------------------------------------------------------------------- R1 / R2 at build time
    def _check_r2(self):
        """Refuse to run a contaminated cohort, over BOTH the incident surfaces this module generates and
        everything else this config can say (`world_vocabulary`). A paid run must not discover at tick 40 that
        the world has been handing agents the words a phrase is built from.

        The surfaces are checked ACROSS memes, not only within one: an incident that happens to use another
        registry phrase's words hands that meme's coinage to everyone who eats there.
        """
        texts = []
        for c in self.conditions:
            for key in ("surfaces", "repaired_surfaces"):
                for i, s in enumerate(getattr(c, key)):
                    texts.append((f"memes.registry[{c.meme}].incident.{key}[{i}]", s))
        profiles = {aid: a.profile for aid, a in self.agents.items()}
        texts += world_vocabulary(self.cfg, profiles)
        hits = audit(self.entries, texts)
        if not hits:
            return
        seen, lines = set(), []
        for h in hits:                                    # one line per (meme, word, source), not per hit
            key = (h["meme"], h["word"], h["where"])
            if key in seen:
                continue
            seen.add(key)
            what = "the phrase itself" if h["kind"] == "phrase" else "a word it is built from"
            lines.append(f"  meme {h['meme']!r}: the world says {h['word']!r} ({what}) in {h['where']}\n"
                         f"    {h['text'][:160]!r}")
        more = f"\n  (+{len(lines) - 10} more sources)" if len(lines) > 10 else ""
        raise ValueError(
            "meme contamination (R1/R2): the world must never say a registry phrase or supply the words it is "
            "built from. An agent handed those words can coin the expression by itself, and transmission stops "
            "being distinguishable from rediscovery - which is the whole measurement.\n"
            + "\n".join(lines[:10]) + more +
            "\nFix it by swapping the phrase for one the world does not already speak, by rewording the source, "
            "or - if the word is the referent the world cannot avoid naming rather than what the coinage adds "
            "(\"tray\" in \"blue-tray\") - by narrowing that meme's `distinctive_words`, which is recorded in "
            "the run's manipulation checks as an exemption.")

    # ------------------------------------------------------------------------------------------ world hook
    def script_jsonl(self) -> str:
        import json
        return "".join(json.dumps(c.script_record(self.clock), sort_keys=True) + "\n" for c in self.conditions)

    def world(self, tick: int) -> list[tuple]:
        """(event, beat) pairs for the conditions people are running into at this tick.

        Called AFTER movement: an incident is not a scheduled scene, it is a state of the world that whoever
        turns up encounters, and nobody is ever moved to it (a forced mover would make co-witnessing a world
        decision instead of a consequence of the agents' own days).
        """
        if not self.conditions:
            return []
        day = self.clock.day_of(tick)
        t = self.clock.time_of(tick)
        minutes = t.hour * 60 + t.minute
        out = []
        for c in self.conditions:
            state = c.state_on(day)
            if state is None:
                continue
            occ = c.occasion(minutes)
            if occ is None:
                continue
            beat = self._encounters(c, state, day, occ, tick)
            if beat is not None:
                out.append(beat)
        return out

    def _encounters(self, c: Condition, state: str, day: int, occ: int, tick: int):
        surfaces = c.surfaces if state == FAULT else c.repaired_surfaces
        if not surfaces:
            return None
        fired = []
        for aid in sorted(self.agents):
            a = self.agents[aid]
            if not getattr(a.state, "active", True) or a.state.activity == "sleeping":
                continue
            if (a.state.location, a.state.arena) != c.place:
                continue
            key = (aid, c.meme, day, occ)
            if key in self.drawn:
                continue                       # one draw per person per meal, not one per tick spent eating
            self.drawn.add(key)
            rng = seed_rng(a.seed, aid, "incident", c.meme, day, occ)
            u_hit, u_surface, u_rank = rng.random(3)
            if u_hit >= c.prob_per_meal:
                continue
            s = surfaces[min(int(u_surface * len(surfaces)), len(surfaces) - 1)]
            fired.append((float(u_rank), aid, s))
        if not fired:
            return None
        if c.max_per_tick is not None and len(fired) > c.max_per_tick:
            fired.sort()                       # u_rank is an independent uniform: a uniform random subset,
            self.counts["capped"] += len(fired) - c.max_per_tick    # not the alphabetically-first agents
            fired = fired[:c.max_per_tick]
        fired.sort(key=lambda x: x[1])         # fact order by agent id, so the beat is reproducible
        eid = f"inc.{c.meme}.d{day:02d}.t{tick:04d}"
        facts = []
        for i, (_u, aid, s) in enumerate(fired):
            facts.append({"id": f"{eid}.b0.f{i}", "text": _render(s, self.agents[aid].profile.first_name),
                          "salience": c.salience, "visibility": "arena", "kind": f"incident_{state}",
                          "involves": [aid]})
            self.counts[f"encounters_{state}"] += 1
            self.encounters[c.meme][state] += 1
            self.witnesses[c.meme][state].add(aid)
            # `fact` keys the one fact this person is the participant of, so the observer can tell an
            # encounter apart from the same beat's facts that happened to the people next to them.
            self.sim.tracer.log("incident_encounter", meme=c.meme, incident=eid, fact=facts[-1]["id"],
                                agent=aid, day=day, occasion=occ, state=state, location=c.location,
                                arena=c.arena, salience=c.salience, text=facts[-1]["text"])
        return _Ev(eid, c.meme, state), {"idx": 0, "tick": tick, "location": c.location, "arena": c.arena,
                                         "facts": facts, "movers": []}

    # ----------------------------------------------------------------------------------- world-side views
    def exposure(self, meme: str) -> dict:
        """Who the fault reached and who it did not, over the agents the run actually has."""
        exposed = sorted(self.witnesses.get(meme, {}).get(FAULT, ()))
        return {"exposed": exposed, "n_exposed": len(exposed),
                "n_unexposed": max(0, len(self.agents) - len(exposed))}

    def manipulation_checks(self) -> dict:
        """Per meme: the 2x2 cell, whether it has a basis, how often that basis was actually run into, by how
        many distinct people, the exposed/unexposed split, and the R2 exemptions its config declared."""
        out = {}
        for e in self.entries:
            enc = self.encounters[e["id"]]
            w = self.witnesses[e["id"]]
            inc = e["incident"]
            out[e["id"]] = {
                "cell": e["cell"], "phrase": e["phrase"], "incident": bool(inc),
                "place": f"{inc['location']} / {inc['arena']}" if inc else None,
                "days": list(inc["days"]) if inc else [], "repaired_day": inc["repaired_day"] if inc else None,
                "encounters": dict(enc), "witnesses": {k: len(v) for k, v in w.items()},
                **self.exposure(e["id"]),
                "distinctive_words": list(e["distinctive_words"]), "exempt_words": list(e["exempt_words"]),
                "status": "active" if (enc[FAULT] or enc[REPAIRED]) else "inactive",
            }
        return {"cohort": cohort_summary(self.entries), "capped": self.counts.get("capped", 0), "memes": out}
