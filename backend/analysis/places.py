"""Who calls a place what (OBSERVER ONLY, LLM-free, no name lexicon).

The world keeps canonical names for its own purposes -- tile ids, the map, the trace, this analysis --
but once the world stops putting those names into prompts (see `backend.simulation.reference`), the only
way to know what a place is *called* is to read it off what agents say. This module does that without
being told the names in advance: it finds referring expressions in speech and decides, from
circumstantial evidence, which canonical place each one denotes.

Method (per run, per canonical location in `manifest["world"]["graph"]`):

1. CANDIDATE SPANS. From every utterance: quoted spans, capitalised runs and acronyms. A span is kept
   only when the sentence uses it as a *place* -- a locative frame right before it ("at / to / from / by /
   near / in the / over at / back to ..."; bare "in" is excluded, since English puts fields of study in it
   as readily as rooms), a naming frame around it ("we call it X", "known as X"), or a venue HEAD NOUN
   inside it. Head nouns are the last token of each of the world's own location and arena names ("Dining
   Hall" -> hall, "Research Lab" -> lab), so that vocabulary comes from the map, never from a hand-written
   list of names -- and "Biomedical Engineering" does not become a building. Dropped as well: spans whose
   every token is an agent name, single capitalised words the corpus also writes in lower case
   (sentence-initial ordinary words), and premodifiers of an ordinary word ("the Neuroscience class").
2. DESCRIPTIONS. Determiner + optional modifiers + a venue head noun ("the dining hall", "the hall"),
   plus a small closed set of deictics and generic venue nouns ("here", "this place", "the cafeteria").
   In a situated run the world's own descriptor is matched first and masked, so "the dining hall beside
   the freshman residences" is one world description and not an agent echoing "the dining hall". A
   description whose head IS the world's canonical label is kept as an expression of kind "canonical" --
   that is the world doing the naming, and the point is to watch it compete. Every other description
   names nothing and is counted as `unnamed`.
3. ATTRIBUTION. Evidence is aggregated per expression over all its mentions and scored against every
   place. An expression is attributed only on IDENTIFYING evidence: it is the world's own label, a
   memory or an utterance glosses it against a description of the place, it carries the place's venue
   head noun, or someone introduced it in a naming frame while at or heading to the place. Standing
   somewhere while speaking is not identifying -- location, movement, listeners, meal context (weighted
   by how much eating actually happens at that place in this run) and cross-speaker agreement only choose
   between candidate places and grade confidence. Location evidence is further scaled by LIFT, how much
   more of the expression's use happens at the place than the run's own base rate of talk there, so a
   busy place cannot claim every word said inside it. Each feature saturates (n/(n+1)), so one loud
   speaker cannot outvote the rest. The expression goes to the best-scoring place when that score clears
   `min_score`; a close runner-up is reported in `ambiguous_with`, and anything unclaimed is listed in
   `unattributed` rather than guessed at.

Every expression carries a `kind`:
  "name"                 an agent-side name -- what the naming experiment is about;
  "canonical"            the world's own label for the place, echoed back (world-supplied wording);
  "situated_descriptor"  the neutral description a situated world hands to prompts;
  "description"          a common-noun description built from venue words;
  "deictic"              "here" / "this place" / a generic venue noun, resolved by where the speaker is.
`unnamed` counts references that named nothing: description + situated_descriptor + deictic. The world's
canonical label is NOT unnamed -- it is a name, just not the agents'; it is reported separately as
`world_label_uses`, because a run where it dominates is a run where the world did the naming.

Limitations: attribution is circumstantial, not semantic -- a coined name that no memory or utterance ever
glosses, that carries no venue noun and that nobody introduces in a naming frame is left unattributed
rather than guessed at; a generic word the map supplies no head noun for ("cafeteria" when the map says
"Dining Hall") is caught only by the closed deictic set; surface forms are not merged across misspellings
or morphology. Nothing here is evidence of understanding, agreement, or successful reference -- only of
wording.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_VERSION = 1
MIN_SCORE = 3.0
AMBIGUOUS_RATIO = 0.8          # a runner-up within this fraction of the winner is reported as ambiguous
WEIGHTS = {"canonical": 5.0, "gloss": 4.0, "speech_gloss": 2.5, "venue": 2.0, "naming": 2.0,
           "here": 2.0, "next": 2.0, "listener": 1.0, "listener_next": 0.7, "meal": 1.5, "agreement": 1.0}

# Closed function-word sets. These are English grammar, not a place lexicon: they say *that* a span is
# used as a place, never *which* place it is. Bare "in" is deliberately absent -- English puts fields of
# study in it ("in Computer Science") as readily as rooms, and "in the ..." is covered explicitly.
LOCATIVE = re.compile(r"(?:(?:\bat|\bto|\bfrom|\bby|\bnear|\binto|\binside|\boutside|\baround|\bnext\s+to"
                      r"|\bover\s+(?:at|to|by)|\bback\s+(?:at|to)|\bdown\s+(?:at|to)|\bup\s+(?:at|to))"
                      r"\s+(?:the\s+|a\s+|an\s+|old\s+|new\s+)?|\bin\s+(?:the|a|an)\s+(?:old\s+|new\s+)?)$", re.I)
NAMING = re.compile(r"\b(?:call|calls|called|calling|know\s+it\s+as|known\s+as|refer(?:red|s)?\s+to\s+it\s+as"
                    r"|nickname[ds]?|we\s+say|they\s+say|goes\s+by)\b", re.I)
MEAL = re.compile(r"\b(?:lunch|lunches|dinner|dinners|breakfast|brunch|supper|meal|meals|eat|eats|eating|ate|food"
                  r"|dining|snack|hungry|grab\s+a\s+bite|cafeteria|canteen)\b", re.I)
DEICTIC = re.compile(r"\b(?:over\s+here|in\s+here|right\s+here|here|this\s+place|that\s+place|this\s+spot"
                     r"|the\s+place|the\s+cafeteria|the\s+canteen|the\s+dining\s+room|the\s+food\s+court)\b", re.I)
DETERMINER = r"(?:the|this|that|a|an|our|their|its|main|new|old|campus)"
_HERE_IDIOM = re.compile(r"here(?:'s|’s|\s+is|\s+are|\s+you\s+go|\s+we\s+go)\b", re.I)
# Only real quotation marks: an apostrophe opens no quotation, it makes "I'm ... I'd" one giant span.
_QUOTED = re.compile(r"[\"“]([^\"”]{2,48})[\"”]")
# No "." inside a token: it would bridge "Biomedical Engineering. Like" into one span across a sentence.
_CAPS = re.compile(r"\b[A-Z][A-Za-z0-9&'’\-]*(?:\s+(?:of|the|de|at)\s+[A-Z][A-Za-z0-9&'’\-]*"
                   r"|\s+[A-Z][A-Za-z0-9&'’\-]*){0,3}")
_WORD = re.compile(r"[A-Za-z0-9'’\-]+")
_TRIM = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9'’]+$")
_SKIP_TOKENS = frozenset("""a an and the or but if then so of to in on at for with by from as is are was were be i im
hey hi hello yeah yep yes no oh ok okay well thanks thank sure right maybe monday tuesday wednesday thursday friday
saturday sunday january february march april may june july august september october november december morning
afternoon evening night today tomorrow yesterday""".split())
# A place name can stand alone as the object of a locative ("at the Dining Hall tomorrow"); a premodifier
# cannot ("in the Neuroscience class", "from the Public Health department"). A capitalised span followed by
# an ordinary content word is the modifier of that word, not the thing referred to -- so a span is kept only
# when what follows it is punctuation, a number, or one of these function / deictic words.
_FOLLOWER_OK = frozenset("""and or but so if when while then than because for with without about after before during
around at in on to from by near until till since as of is are was were be been am i we you they he she it my our
your their his her its this that these those the a an s re ll ve d t not no today tomorrow tonight yesterday later
soon now again too instead maybe probably right first next last there here where who what which how why""".split())


def _norm(text: str) -> str:
    """Case-folded, accent-stripped, whitespace-collapsed surface form."""
    folded = unicodedata.normalize("NFKD", str(text).casefold())
    return " ".join("".join(c for c in folded if not unicodedata.combining(c)).split())


def _key(span: str) -> str:
    """The form two mentions must share to be one expression: normalised, leading article dropped."""
    n = _norm(_TRIM.sub("", span))
    for article in ("the ", "a ", "an "):
        if n.startswith(article):
            return n[len(article):]
    return n


def _sat(n: float) -> float:
    """Saturating count: one speaker is evidence, ten speakers are not ten times the evidence."""
    return n / (n + 1.0) if n > 0 else 0.0


def entropy(counts) -> float | None:
    """Shannon entropy in bits over expression uses; None when nothing was said."""
    vals = [c for c in counts if c > 0]
    total = sum(vals)
    if not total:
        return None
    return round(-sum((c / total) * math.log2(c / total) for c in vals), 4) + 0.0    # never -0.0


# ------------------------------------------------------------------------------ the world's vocabulary
def situated_descriptors(places: list[str], cfg: dict | None = None) -> dict:
    """{place: the neutral description the world handed to this run's prompts}, read from
    `backend.simulation.reference`; {} for a run whose world still names places.

    The run's OWN config decides: a canonical-label run analysed on a build that has the situated
    machinery is still a canonical-label run. Duck-typed on purpose -- whether that module is absent,
    exposes a mapping, or exposes a function, the observer keeps working; anything it cannot read is
    simply not a descriptor."""
    try:
        from backend.simulation import reference as mod
    except Exception:                                  # pragma: no cover - absent in canonical-label builds
        return {}
    situated = getattr(mod, "SITUATED", "situated")
    mode = getattr(mod, "mode", None)
    if callable(mode):
        try:
            if mode(cfg or {}) != situated:
                return {}
        except Exception:
            return {}
    elif str((((cfg or {}).get("world") or {}) if isinstance((cfg or {}).get("world"), dict) else {})
             .get("reference_mode") or "canonical").strip().lower() != str(situated):
        return {}
    out = {}
    for place in places:
        found = None
        for attr in ("PLACE_DESCRIPTIONS", "DESCRIPTIONS", "DESCRIPTORS"):
            table = getattr(mod, attr, None)
            if isinstance(table, dict) and isinstance(table.get(place), str) and table[place].strip():
                found = table[place].strip()
                break
        for attr in ("place", "describe", "description", "describe_place", "descriptor"):
            if found:
                break
            fn = getattr(mod, attr, None)
            if not callable(fn):
                continue
            for kwargs in ({"cfg": cfg}, {}):
                try:
                    value = fn(place, **kwargs)
                except Exception:
                    continue
                if isinstance(value, str) and value.strip():
                    found = value.strip()
                    break
        if found and _norm(found) != _norm(place):
            out[place] = found
    return out


def relative_descriptors() -> set:
    """World-supplied descriptions that are relative to the hearer ("your room", "the lab where you
    work"), so they denote a different place for every agent and are resolved by where the speaker is.
    Empty on a build without the situated machinery."""
    try:
        from backend.simulation import reference as mod
    except Exception:                                  # pragma: no cover - absent in canonical-label builds
        return set()
    out = {str(getattr(mod, a)) for a in ("HOME_ARENA", "_FOCUS_DEFAULT") if isinstance(getattr(mod, a, None), str)}
    for _pat, desc in getattr(mod, "_FOCUS_RULES", ()) or ():
        if isinstance(desc, str):
            out.add(desc)
    return {_norm(d) for d in out if d.strip()}


def _world(rd) -> tuple[list[str], dict, dict, dict]:
    """(places, canonical surface keys, name tokens, venue HEAD nouns) from the run's own map.

    Head nouns are the last token of each location and arena name ("Dining Hall" -> hall, "Research Lab"
    -> lab): English puts the venue noun last, so heads are what makes a span look like a place at all
    ("Hopkins Cafe"), while the full token set says *which* place ("Engineering Hall" vs "Science Hall").
    Taking every token as a venue word would make "Biomedical Engineering" a building."""
    world = rd.manifest.get("world") or {}
    graph, arenas = world.get("graph") or {}, world.get("arenas") or {}
    places = list(graph) or sorted(arenas)
    canonical, named, heads = {}, {}, {}
    for place in places:
        forms = [place] + [a for a in (arenas.get(place) or []) if isinstance(a, str)]
        canonical[place] = {_key(f) for f in forms}
        named[place] = {t for f in forms for t in _WORD.findall(_norm(f))
                        if len(t) > 2 and not t.isdigit() and t not in _SKIP_TOKENS}
        heads[place] = {toks[-1] for f in forms if (toks := _WORD.findall(_norm(f)))
                        if len(toks[-1]) > 2 and not toks[-1].isdigit() and toks[-1] not in _SKIP_TOKENS}
    return places, canonical, named, heads


def _meal_affinity(rd, places: list[str]) -> dict:
    """How much eating happens at each place in THIS run: the share of scheduled activities there whose
    wording is about a meal, rescaled so the most meal-heavy place is 1.0. Read from world-side schedule
    text (routines, day plans, moves), never from agent speech."""
    hit, total = Counter(), Counter()
    rows = [(r.get("location"), r.get("activity")) for a in rd.agents.values() for r in (a.get("routine") or [])]
    for row in rd.of("day_plan"):
        rows += [(s.get("location"), s.get("activity")) for s in (row.get("plan") or []) if isinstance(s, dict)]
    for row in rd.of("move"):
        to = row.get("to")
        rows.append(((to[0] if isinstance(to, list) and to else to), row.get("activity")))
    for place, activity in rows:
        if place not in places:
            continue
        total[place] += 1
        if activity and MEAL.search(str(activity)):
            hit[place] += 1
    raw = {p: hit[p] / total[p] for p in places if total[p]}
    top = max(raw.values(), default=0.0)
    return {p: (v / top if top else 0.0) for p, v in raw.items()}


def _timelines(rd) -> dict:
    """agent -> sorted [(tick, place)] from moves and utterances: where they were, where they went next."""
    out = defaultdict(list)
    for row in rd.of("move"):
        to = row.get("to")
        place = to[0] if isinstance(to, list) and to else to
        if isinstance(place, str) and row.get("agent"):
            out[row["agent"]].append((int(row.get("tick") or 0), place))
    for u in rd.utterances:
        if isinstance(u.get("location"), str):
            out[u["speaker"]].append((int(u["tick"]), u["location"]))
    return {agent: sorted(set(rows)) for agent, rows in out.items()}


def _where(timeline: list, tick: int) -> str | None:
    seen = None
    for t, place in timeline:
        if t > tick:
            break
        seen = place
    return seen


def _next_place(timeline: list, tick: int, current: str | None) -> str | None:
    for t, place in timeline:
        if t > tick and place != current:
            return place
    return None


def _lift(mentions: list, place: str, base: float) -> float:
    """How concentrated an expression's use is at `place`, over the run's base rate of talk there, capped
    at 1.0 (twice the base rate is as much as location alone can say). 1.0 when the base rate is unknown."""
    if base <= 0:
        return 1.0
    here = sum(1 for m in mentions if m["here"] == place) / len(mentions)
    return round(min(1.0, (here / base) / 2.0), 3)


# ------------------------------------------------------------------------------ surface extraction
def _lowercase_words(utterances: list[dict]) -> set:
    """Words the corpus also writes in lower case; one capitalised one of these is sentence case."""
    seen = Counter()
    for u in utterances:
        for m in _WORD.finditer(u.get("text") or ""):
            if m.group(0)[:1].islower():
                seen[m.group(0).casefold()] += 1
    return {w for w, n in seen.items() if n >= 2}


def _sentence_before(text: str, start: int) -> str:
    """The text from the start of the current sentence up to `start`."""
    cut = max(text.rfind(ch, 0, start) for ch in ".!?\n")
    return text[cut + 1:start]


def _premodifier(text: str, end: int) -> bool:
    """True when the span ending at `end` modifies the ordinary word after it ("Neuroscience class")."""
    after = text[end:end + 40].lstrip()
    if not after or not after[:1].isalpha():
        return False
    word = _WORD.match(after)
    return bool(word) and word.group(0)[:1].islower() and word.group(0).casefold() not in _FOLLOWER_OK


def _spans(text: str, name_tokens: set, common: set, head_nouns: set) -> list[tuple]:
    """[(span, flags)] -- capitalised or quoted spans the sentence uses as a place."""
    out, seen = [], set()
    for m in list(_QUOTED.finditer(text)) + list(_CAPS.finditer(text)):
        quoted = m.re is _QUOTED
        span = _TRIM.sub("", m.group(1) if quoted else m.group(0))
        if not span or span in seen:
            continue
        toks = [t.casefold() for t in _WORD.findall(span)]
        if not toks or all(t in _SKIP_TOKENS for t in toks) or all(t in name_tokens for t in toks):
            continue
        if len(toks) == 1 and toks[0].isdigit():
            continue
        if not quoted and _premodifier(text, m.end()):
            continue
        sentence = _sentence_before(text, m.start())
        flags = set()
        if LOCATIVE.search(sentence[-30:]):
            flags.add("locative")
        if NAMING.search(sentence[-60 if quoted else -46:]):
            flags.add("naming")
        if head_nouns & set(toks):
            flags.add("venue")
        if not flags:
            continue
        if len(toks) == 1 and not sentence.strip() and toks[0] in common:
            continue                                   # sentence-initial ordinary word, not a name
        seen.add(span)
        out.append((span, sorted(flags)))
    return out


def _descriptions(text: str, venue_re: re.Pattern, venue_by_place: dict, canonical: dict) -> list[tuple]:
    """[(surface, head_key, candidate_places, kind)] for common-noun place references and deictics.

    A match whose head is capitalised is left to `_spans` (it is being used as a name); a match whose head
    is the world's canonical label for a place is kind "canonical"; anything else is "description"."""
    out = []
    for m in venue_re.finditer(text):
        head = re.sub(r"(?i)^" + DETERMINER + r"\s+", "", m.group(0)).strip()
        if head[:1].isupper():
            continue
        key, toks = _key(head), set(_WORD.findall(_norm(head)))
        cands = sorted((p for p in venue_by_place if venue_by_place[p] & toks),
                       key=lambda p: (-len(venue_by_place[p] & toks), p))
        if not cands:
            continue
        kind = "canonical" if any(key in canonical[p] for p in cands) else "description"
        out.append((" ".join(m.group(0).split()), key, cands, kind))
    for m in DEICTIC.finditer(text):
        if _HERE_IDIOM.match(text[m.start():]):
            continue
        out.append((_norm(m.group(0)), None, [], "deictic"))
    return out


def _glosses(rd, tick: int | None) -> list[tuple]:
    """[(agent, normalised text, token set)] over the seed and ambient memories that GLOSS something --
    memories with a naming frame in them ("I have called the dining hall beside AMR III X").

    A gloss is metalinguistic: it ties an expression to a *description* of a place, which is how the
    observer learns what an expression denotes without being told any names in advance. A memory that
    merely mentions a place ("Jordan is eating lunch at the Dining Hall") is not a gloss, or every
    department someone studies in a lab would become a name for that lab."""
    cap = tick if tick is not None else 1 << 30
    out = []
    for row in rd.of("memory_encoded"):
        if row.get("source_type") not in ("seed", "ambient") or int(row.get("tick") or 0) > cap:
            continue
        text = row.get("text") or ""
        if not row.get("agent") or not NAMING.search(text):
            continue
        normalised = _norm(text)
        out.append((row["agent"], normalised, set(_WORD.findall(normalised))))
    return out


# ------------------------------------------------------------------------------ the resolver
def resolve_place_references(run_dir, tick: int | None = None, *, rd=None, min_score: float = MIN_SCORE) -> dict:
    """What agents call each canonical place, inferred from their own speech. See the module docstring."""
    if rd is None:
        from backend.analysis.rundata import RunData
        rd = RunData(Path(run_dir))
    places, canonical, venue_by_place, head_by_place = _world(rd)
    place_set = set(places)
    utts = [u for u in rd.utterances if tick is None or int(u["tick"]) <= int(tick)]
    tpd = max(1, rd.ticks_per_day)
    days = list(range(1, max((int(u["tick"]) // tpd for u in utts), default=0) + 2))
    descriptors = situated_descriptors(places, getattr(rd, "cfg", None))
    descriptor_keys = {_key(v): p for p, v in descriptors.items()}
    relative = relative_descriptors() if descriptors else set()
    # A place is glossed in a memory when every token of its name -- or of its situated description --
    # is there, minus the tokens of the expression being glossed (so "Hopkins Cafe" does not gloss "Cafe").
    describes = {p: [{t for t in _WORD.findall(_norm(p))}] + ([{t for t in _WORD.findall(_norm(descriptors[p]))}]
                                                              if descriptors.get(p) else []) for p in places}
    affinity = _meal_affinity(rd, places)
    timelines = _timelines(rd)
    common = _lowercase_words(utts)
    head_nouns = {t for toks in head_by_place.values() for t in toks}
    venue_re = re.compile(r"(?i)\b" + DETERMINER + r"\s+(?:[A-Za-z]+\s+){0,2}?(?:"
                          + "|".join(sorted(map(re.escape, head_nouns), key=len, reverse=True)) + r")\b") \
        if head_nouns else re.compile(r"(?!x)x")
    relative_re = (re.compile(r"(?i)\b(?:" + "|".join(sorted(map(re.escape, relative), key=len, reverse=True)) + r")\b")
                   if relative else None)
    # "the dining hall beside the freshman residences" contains "the dining hall": match the world's whole
    # descriptor first and blank it out, or a situated run would read as agents echoing the canonical label.
    descriptor_re = (re.compile(r"(?i)\b(?:" + "|".join(sorted((re.escape(v) for v in descriptors.values()),
                                                              key=len, reverse=True)) + r")")
                     if descriptors else None)
    by_descriptor = {_norm(v): p for p, v in descriptors.items()}
    glosses = _glosses(rd, tick)
    base = Counter(u.get("location") for u in utts if u.get("location") in place_set)
    base_rate = {p: base[p] / len(utts) for p in places} if utts else {p: 0.0 for p in places}

    # -- pass 1: every mention, with the circumstances it was made in --------------------------------
    mentions: dict[str, list] = defaultdict(list)
    unnamed: list[dict] = []
    for u in utts:
        text, speaker, t = u.get("text") or "", u["speaker"], int(u["tick"])
        here = u.get("location") if u.get("location") in place_set else _where(timelines.get(speaker, []), t)
        nxt = _next_place(timelines.get(speaker, []), t, here)
        listeners = [l for l in (u.get("listeners") or []) if l != speaker]
        l_here = {_where(timelines.get(l, []), t) for l in listeners} & place_set
        l_next = {_next_place(timelines.get(l, []), t, _where(timelines.get(l, []), t)) for l in listeners} & place_set
        described = set()                              # places this utterance also describes in words
        ctx = {"u": u, "speaker": speaker, "tick": t, "here": here, "next": nxt, "l_here": l_here,
               "l_next": l_next, "meal": bool(MEAL.search(text)), "described": described}
        said = {}                                      # one utterance referring twice is one use
        plain = text
        if descriptor_re is not None:
            for _m in descriptor_re.finditer(text):    # the world's own neutral description, repeated back
                target = by_descriptor.get(_norm(_m.group(0)))
                if target in place_set:
                    described.add(target)
                    unnamed.append({"place": target, "kind": "situated_descriptor",
                                    "surface": _norm(_m.group(0)), **ctx})
                    plain = plain[:_m.start()] + " " * (_m.end() - _m.start()) + plain[_m.end():]
        if relative_re is not None and here in place_set:
            for _m in relative_re.finditer(plain):     # "your room", "the lab where you work": world wording
                unnamed.append({"place": here, "kind": "situated_descriptor", "surface": _norm(_m.group(0)), **ctx})
                break
        for surface, key, cands, kind in _descriptions(plain, venue_re, venue_by_place, canonical):
            # a description with several candidate places ("the hall") goes to the one the speaker is at
            # or heading to, else the closest token match; a deictic goes to where the speaker is.
            place = (next((p for p in cands if p in (here, nxt)), None) or (cands[0] if cands else None)
                     if cands else here)
            if place in place_set and kind != "deictic":
                described.add(place)                   # this utterance says out loud which place it means
            if kind == "canonical":                    # the world's own label, echoed in lower case
                said.setdefault(key, {"span": surface, "flags": ["description"], **ctx})
                continue
            if place in place_set and not any(r["place"] == place and r["kind"] == kind and r["u"] is u
                                              for r in unnamed[-8:]):
                unnamed.append({"place": place, "kind": kind, "surface": surface, **ctx})
        for span, flags in _spans(plain, rd.name_tokens, common, head_nouns):
            said.setdefault(_key(span), {"span": span, "flags": flags, **ctx})
        for key, mention in said.items():
            mentions[key].append(mention)

    # -- pass 2: score each expression against every place --------------------------------------------
    resolved: dict[str, list] = defaultdict(list)
    unattributed = []
    for key, ms in mentions.items():
        speakers = {m["speaker"] for m in ms}
        word = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)")
        key_tokens = set(_WORD.findall(key))
        glossed = defaultdict(set)
        for agent, memory, memory_tokens in glosses:
            if key not in memory or not word.search(memory):
                continue
            rest = memory_tokens - key_tokens
            for place in places:
                if any(described <= rest for described in describes[place]):
                    glossed[place].add(agent)
        named_by = _sat(len({m["speaker"] for m in ms if "naming" in m["flags"]}))
        scores = {}
        for place in places:
            # Being somewhere is not evidence on its own: nearly every utterance happens somewhere. Location
            # evidence is scaled by lift -- how much more of this expression's use happens at the place than
            # the run's own base rate of talk there -- so a busy place cannot claim every word said in it.
            lift = _lift(ms, place, base_rate.get(place, 0.0))
            feats = {
                "canonical": 1.0 if key in canonical[place] else 0.0,
                "gloss": _sat(len(glossed.get(place, ()))),
                "speech_gloss": _sat(len({m["speaker"] for m in ms if place in m["described"]})),
                "venue": (len(venue_by_place[place] & key_tokens)
                          / max(1, len(venue_by_place[place]))) if venue_by_place[place] else 0.0,
                "naming": named_by,
                "here": lift * _sat(len({m["speaker"] for m in ms if m["here"] == place})),
                "next": lift * _sat(len({m["speaker"] for m in ms if m["next"] == place})),
                "listener": lift * _sat(sum(1 for m in ms if place in m["l_here"])),
                "listener_next": lift * _sat(sum(1 for m in ms if place in m["l_next"])),
                "meal": affinity.get(place, 0.0) * sum(1 for m in ms if m["meal"]) / len(ms),
                "agreement": _sat(len(speakers) - 1),
            }
            # Standing somewhere while you speak does not make what you said a name for it. An expression
            # is attributed only on IDENTIFYING evidence -- it is the world's own label, a memory or an
            # utterance glosses it against a description of the place, it carries the place's venue noun,
            # or someone introduced it in a naming frame while at or heading to the place. Location, meal
            # context and agreement then choose between places and grade confidence; alone they attribute
            # nothing, or every subject studied in a building would become a name for the building.
            identifying = (feats["canonical"] or feats["gloss"] or feats["speech_gloss"] or feats["venue"]
                           or (feats["naming"] and (feats["here"] or feats["next"])))
            if not identifying:
                continue
            scores[place] = (round(sum(WEIGHTS[k] * v for k, v in feats.items()), 3), feats)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[0]))
        if not ranked or ranked[0][1][0] < min_score:
            unattributed.append({"expression": Counter(m["span"] for m in ms).most_common(1)[0][0],
                                 "uses": len(ms), "speakers": len(speakers),
                                 "closest_place": ranked[0][0] if ranked else None,
                                 "score": ranked[0][1][0] if ranked else 0.0})
            continue
        place, (score, feats) = ranked[0]
        kind = ("canonical" if key in canonical[place]
                else "situated_descriptor" if descriptor_keys.get(key) == place else "name")
        resolved[place].append({"key": key, "kind": kind, "score": score, "feats": feats, "mentions": ms,
                                "glossed": glossed.get(place, set()),
                                "ambiguous_with": [p for p, (s, _f) in ranked[1:3] if s >= AMBIGUOUS_RATIO * score]})

    # -- pass 3: report ------------------------------------------------------------------------------
    unnamed_by_place = defaultdict(list)
    for row in unnamed:
        unnamed_by_place[row["place"]].append(row)
    out_places = []
    for place in places:
        exprs, rows = resolved.get(place, []), unnamed_by_place.get(place, [])
        if not exprs and not rows:
            continue
        per_day = Counter(int(r["tick"]) // tpd + 1 for r in rows)
        for e in exprs:
            per_day.update(int(m["tick"]) // tpd + 1 for m in e["mentions"])
        expr_rows = []
        for e in sorted(exprs, key=lambda e: (-len(e["mentions"]), e["key"])):
            ms = e["mentions"]
            uses_by_day = Counter(int(m["tick"]) // tpd + 1 for m in ms)
            first = min(ms, key=lambda m: (m["tick"], m["u"].get("id") or ""))
            seeded_uses = sum(1 for m in ms if m["speaker"] in e["glossed"])
            expr_rows.append({
                "expression": Counter(m["span"] for m in ms).most_common(1)[0][0],
                "kind": e["kind"], "uses": len(ms), "speakers": len({m["speaker"] for m in ms}),
                "speaker_ids": sorted({m["speaker"] for m in ms}),
                "first_tick": first["tick"], "ticks": sorted(m["tick"] for m in ms),
                "first_use": {"tick": first["tick"], "speaker": first["speaker"],
                              "utterance_id": first["u"].get("id"), "text": (first["u"].get("text") or "")[:300]},
                "uses_by_day": [uses_by_day.get(d, 0) for d in days],
                "share_by_day": [round(uses_by_day.get(d, 0) / per_day[d], 4) if per_day.get(d) else None
                                 for d in days],
                "seeded_group_share": {
                    "seeded_agents": len(e["glossed"]),
                    "seeded_speakers": len({m["speaker"] for m in ms} & e["glossed"]),
                    "uses_by_seeded": seeded_uses, "uses_by_unseeded": len(ms) - seeded_uses,
                    "share": round(seeded_uses / len(ms), 4) if ms else None},
                "score": e["score"], "ambiguous_with": e["ambiguous_with"],
                "evidence": {k: round(v, 3) for k, v in e["feats"].items() if v},
            })
        names = [r for r in expr_rows if r["kind"] == "name"]
        refs = sum(r["uses"] for r in expr_rows) + len(rows)
        out_places.append({
            "place": place, "canonical_label": place, "situated_descriptor": descriptors.get(place),
            "references": refs,
            "named": sum(r["uses"] for r in expr_rows if r["kind"] in ("name", "canonical")),
            "agent_named": sum(r["uses"] for r in names),
            "world_label_uses": sum(r["uses"] for r in expr_rows if r["kind"] == "canonical"),
            "unnamed": len(rows) + sum(r["uses"] for r in expr_rows if r["kind"] == "situated_descriptor"),
            "unnamed_breakdown": {
                "description": sum(1 for r in rows if r["kind"] == "description"),
                "deictic": sum(1 for r in rows if r["kind"] == "deictic"),
                "situated_descriptor": sum(1 for r in rows if r["kind"] == "situated_descriptor")
                + sum(r["uses"] for r in expr_rows if r["kind"] == "situated_descriptor")},
            "agent_name_share": round(sum(r["uses"] for r in names) / refs, 4) if refs else None,
            "expressions": expr_rows,
            "naming_entropy_by_day": [entropy([r["uses_by_day"][d - 1] for r in expr_rows]) for d in days],
            "leader": max(names, key=lambda r: r["uses"])["expression"] if names else None,
            "days": days,
        })
    out_places.sort(key=lambda p: (-p["references"], p["place"]))
    return {
        "analysis": "place_reference_resolution", "schema_version": SCHEMA_VERSION,
        "run_directory": str(getattr(rd, "dir", run_dir)), "tick": tick, "ticks_per_day": tpd, "days": days,
        "situated": bool(descriptors), "n_utterances": len(utts), "min_score": min_score,
        "places": out_places, "unattributed": sorted(unattributed, key=lambda r: (-r["uses"], r["expression"]))[:20],
        "definitions": {
            "kind": "name = an agent-side name; canonical = the world's own label echoed back; "
                    "situated_descriptor / description / deictic = a reference that names nothing.",
            "unnamed": "references made with a generic description, the world's situated descriptor, or a deictic.",
            "world_label_uses": "references that reuse the world's canonical label: a name, but not the agents'.",
            "share_by_day": "uses of this expression / all references to this place that day.",
            "seeded_group_share": "uses by agents whose seed or ambient memory glosses this expression against "
                                  "this place, over all of its uses.",
            "naming_entropy_by_day": "Shannon entropy in bits over expression uses that day; 0 = one form has won.",
        },
        "limitations": [
            "Attribution is circumstantial (location, movement, meal context, memory glosses, agreement), not semantic.",
            "A name used away from the place and glossed in no memory is left unattributed, not assigned.",
            "Generic words the map supplies no token for are caught only by a small closed deictic set.",
            "Surface forms are not merged across misspellings or morphology.",
            "Counts are wording, not evidence of understanding, agreement, or successful reference.",
        ],
    }


# ------------------------------------------------------------------------------ trends competition block
def competition_series(resolved: dict, windows: list[int], window: int, *, top: int = 6) -> list[dict]:
    """The `competition` block for `trends.compute`: rival expressions for one place, binned into the same
    windows as the rest of the trends payload, with naming entropy per window.

    Only places where something actually competes are returned: two or more expressions, or one expression
    against unnamed references. A place everyone merely describes is not a naming contest."""
    n = len(windows)
    if n <= 0:
        return []
    def wi(t):
        return min(max(int(t) // max(1, window), 0), n - 1)
    out = []
    for place in resolved.get("places") or []:
        exprs = [e for e in place["expressions"] if e["uses"]]
        if len(exprs) < 2 and not (exprs and place["unnamed"]):
            continue
        rivals = []
        for e in sorted(exprs, key=lambda e: (-e["uses"], e["expression"]))[:top]:
            uses = [0] * n
            for t in e["ticks"]:
                uses[wi(t)] += 1
            rivals.append({"expression": e["expression"], "kind": e["kind"], "uses": e["uses"],
                           "speakers": e["speakers"], "first_tick": e["first_tick"], "uses_by_window": uses,
                           "seeded_share": e["seeded_group_share"]["share"]})
        totals = [sum(r["uses_by_window"][i] for r in rivals) for i in range(n)]
        for r in rivals:
            r["share_by_window"] = [round(r["uses_by_window"][i] / totals[i], 4) if totals[i] else None
                                    for i in range(n)]
        names = [r for r in rivals if r["kind"] == "name"]
        out.append({
            "referent": place["place"], "label": place["canonical_label"],
            "situated_descriptor": place["situated_descriptor"],
            "n_references": place["references"], "unnamed": place["unnamed"],
            "world_label_uses": place["world_label_uses"], "agent_name_share": place["agent_name_share"],
            "rivals": rivals,
            "entropy_by_window": [entropy([r["uses_by_window"][i] for r in rivals]) for i in range(n)],
            "entropy_now": entropy([r["uses"] for r in rivals]),
            "leader": max(names, key=lambda r: r["uses"])["expression"] if names else None,
        })
    out.sort(key=lambda r: (-r["n_references"], r["referent"]))
    return out
