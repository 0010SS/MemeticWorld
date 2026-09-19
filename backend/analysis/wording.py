"""What counts as a reusable expression, and which wording came from the SYSTEM rather than the agents
(OBSERVER ONLY; never imported by simulation code).

Three pieces, shared by the pipeline (candidates.py) and the live view (live.py):

1. Word classes and **well-formedness** (`WordClasses.reject`). A candidate must be a plausible lexical unit
   that people could reuse:
   - no person names (first or last names of every agent / persona in the manifest or the run's population
     file, possessives included) EXCEPT inside a nickname construction: "the Leo thing", "pulling a Maya",
     "classic Priya", "going full Dev", hyphen or verbed forms ("Leo-proof", "Maya-style", "Leo'd");
     a name that is also a common English word (Zipf >= 4.5: Miles, Park, Kim ...) only counts as a name
     where the speaker capitalized it;
   - it must not start or end with a function word, auxiliary, pronoun, conjunction, reporting / cognition
     verb, or time word, must not contain a reporting frame ("remembers that", "thinking about how",
     "said that") and must not end in an auxiliary frame ("is preparing", "are friends");
   - no bare relationship / role / place nouns (roommates, classmates, labmates, acquaintances, "a student");
   - no numbers or clock times; no span across sentence punctuation, quote marks or *stage directions*;
   - unigrams only if rare (Zipf < 3.6, judged on the base of a contraction/possessive) and not a name/place.

2. The **infrastructure corpus** (`Infrastructure`): everything the system puts into agents' heads or the
   world, split into
   - "system": relationship lines, routine / day-plan activities, seed and ambient memories (ambient
     sightings), memory-encoding / reminding / need / priming frames, the situational lines of conversation
     contexts, profile text (background, habits except the planted control, interests, demographics), place
     names, NPC role phrases, co-op roster/onboarding/menu text, and the population lexicon (token level);
   - "world": event facts, referent names, viewpoint renderings (the live view adds co-op cue/tally/binder text).
   A candidate that is a substring of a system segment ("verbatim"), matches one after person names are
   replaced by a slot ("template": "<n> <n> are roommates"), or after light inflection folding ("folded"), is
   `system_wording`: it can never be a convention.

3. **Recitation** (`Recitation`): an occurrence is *recited* when it sits inside a run of >= RECITE_K tokens
   copied verbatim from the speaker's own earlier memory / reflection text (the memory was encoded at an
   earlier tick, or was retrieved for that utterance). Phrases used mostly in recited text are down-ranked.
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from backend.analysis.candidates import STOP, _TOK, zipf

# ------------------------------------------------------------------------------------------------ word classes
FUNCTION = set("""a an the this that these those each every either neither both some any no all another such what
which whose whoever whatever whichever i me my mine myself you your yours yourself yourselves he him his himself she
her hers herself it its itself we us our ours ourselves they them their theirs themselves one ones oneself someone
somebody something anyone anybody anything everyone everybody everything nobody nothing none other others of to in
on at for with by from as into onto upon over under above below after before during since until till through
throughout across along among amongst around about against between beyond beside besides behind near toward towards
via per within without off out up down inside outside past and or but nor so yet if then than because although
though while whereas unless whether once when whenever where wherever why how am is are was were be been being have
has had having do does did doing done will would shall should can could may might must ought gonna wanna gotta not
no oh ok okay yeah yep yes hey hi hello well like um uh wow just really very also too still even again always never
ever maybe right sure totally honestly actually literally basically seriously anyway anyways kinda sorta else
there here instead rather anymore ago almost already quite enough soon later""".split())
AUX = set("am is are was were be been being have has had do does did will would shall should can could may might must "
          "gets got get becomes became become".split())
REPORT = set("""remember remembers remembered remembering recall recalls recalled recalling think thinks thinking thought
thoughts say says saying said tell tells telling told hear hears hearing heard overhear overhears overhearing overheard
see sees seeing saw seen notice notices noticed noticing realize realizes realized realizing realise realised
wonder wonders wondered wondering know knows knowing knew known mention mentions mentioned mentioning believe
believes believed believing feel feels feeling felt guess guesses guessed guessing suppose supposed supposes reckon
reckons reckoned ask asks asked asking explain explains explained explaining admit admits admitted keep keeps kept
keeping seem seems seemed seeming remind reminds reminded reminding imagine imagined figured""".split())
COMPLEMENTIZERS = {"that", "how", "what", "about", "if", "whether", "why", "when", "who", "which", "where", "of"}
RELATION = set("""roommate roommates classmate classmates labmate labmates clubmate clubmates teammate teammates crewmate
crewmates dormmate dormmates suitemate suitemates hallmate hallmates housemate housemates flatmate flatmates
acquaintance acquaintances friend friends stranger strangers coworker coworkers co-worker co-workers colleague colleagues
neighbor neighbors neighbour neighbours partner partners""".split())
ROLE = set("""student students undergrad undergrads undergraduate undergraduates freshman freshmen sophomore sophomores
ta tas technician technicians newcomer newcomers member members""".split())
TIME = set("""monday tuesday wednesday thursday friday saturday sunday mondays tuesdays wednesdays thursdays fridays
saturdays sundays am pm a.m p.m o'clock noon midnight tonight tomorrow yesterday today mon tue tues wed thu thur
thurs fri""".split())
NUM = re.compile(r"^(\d+([.,:]\d+)*(st|nd|rd|th|s|am|pm|ish|k)?|\d+[-/]\d+)$")
HOURS = set("one two three four five six seven eight nine ten eleven twelve".split())
MINUTES = {"oh", "o", "fifteen", "thirty", "forty", "forty-five", "fortyfive", "twenty", "ten", "five", "o'clock"}
# nickname constructions (a person's name used as a coined word)
NICK_VERBS = set("pull pulls pulled pulling do does did doing done".split())
NICK_PRE = {"classic", "peak", "full", "pure", "total", "such"}
NICK_DET = {"the", "a"}
AMBIGUOUS_ZIPF = 4.5          # names that are also common words count as names only when capitalized
UNIGRAM_MAX_ZIPF = 3.6
RECITE_K = 8
SLOT = "<n>"
# fallback copies of the simulation's fixed strings (the observer prefers importing them)
_REL_PHRASES = {"roommate": "roommates", "friend": "friends", "classmate": "classmates", "labmate": "labmates",
                "clubmate": "in the same club", "acquaintance": "acquaintances", "stranger": "strangers"}
_FAMILIARITY = ("know each other very well", "know each other fairly well", "know each other a little",
                "barely know each other")
_AFFINITY = ("get along well", "get along fine", "don't click much")
_NPC = ["a classmate", "a friend from another dorm", "a TA", "a lab technician", "a student nobody seemed to know",
        "a student from another department", "a visiting student", "a student", "someone"]
# memory-encoding, reminding, need, priming and conversation-context frames (N = a person-name slot)
_FRAMES = [
    "N N saw", "N N heard in a conversation", "N N overheard", "N N did", "N N experienced",
    "N N read in the co-op binder", "N N remembers", "N N remembers that", "N N heard", "N N noticed",
    "N N realized", "N N was reminded of", "it reminded N N of an earlier time", "what felt alike",
    "it brought to mind something N already remembered", "where these mention N they describe what N did or what "
    "happened to N", "N remembers them as their own experience", "N still has on their mind",
    "things N has heard people say lately", "N N just noticed", "N N is initiating a conversation with N N",
    "in the middle of", "started a conversation with N N", "are catching up on how things have been going lately",
    "N is not completely sure about some of the details", "N misremembers one minor detail", "it seems that",
    "what has been happening lately", "what has happened lately that stood out", "N N is already",
    # the co-op binder's fixed rendering (backend/simulation/records.py)
    "front page kept by the shop manager", "front page last rewritten", "earlier front page", "replaced",
    "front page nothing written on it yet", "log newest first", "log no notes yet",
]
_CONTEXT_FRAME_ONLY = [re.compile(r"^(.*? still has on their mind):"), re.compile(r"^(Things .*? has heard people say lately):"),
                       re.compile(r"^(.*? just noticed):")]
_SPEECH_LINE = re.compile(r"^[A-Z][\w'\-]*( [A-Z][\w'\-]*)?: ")
_BREAK = re.compile(r"[.!?;:,\"“”‘’()\[\]{}…—–]|\s-+\s|(^|\s)'|'(\s|$)")
_STAGE = re.compile(r"\*[^*\n]{1,80}\*")
_ELONGATED = re.compile(r"(.)\1{2,}")


def norm_token(t: str) -> str:
    return t.lower().strip("'-")


def lemma_candidates(t: str) -> list[str]:
    """The word and its likely uninflected forms ("deadlines" -> deadline, "texted" -> text, "prepping" -> prep,
    "flagged" -> flag, "replies" -> reply), so rarity is judged on the word, not on one inflection."""
    out = [t]
    for suf, reps in (("ies", ("y",)), ("es", ("", "e")), ("s", ("",)), ("ed", ("", "e")), ("ing", ("", "e")),
                      ("er", ("", "e")), ("est", ("", "e")), ("ly", ("",))):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            stem = t[: -len(suf)]
            out += [stem + r for r in reps]
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
                out.append(stem[:-1])                      # prepping -> prep, flagged -> flag
    return out


@lru_cache(maxsize=65536)
def lemma_zipf(t: str) -> float:
    return max(zipf(c) for c in lemma_candidates(t))


def fold(t: str) -> str:
    from backend.analysis.emergence import _norm
    return _norm(t)


def segment(text: str) -> tuple[list[str], list[str], list[bool], list[bool]]:
    """(raw tokens, normalized tokens, break_before, stage) of an utterance. break_before[i]: sentence or clause
    punctuation, a quote mark or a dash lies between token i-1 and token i; stage[i]: token i is inside an
    *asterisk stage direction*."""
    stage_spans = [(m.start(), m.end()) for m in _STAGE.finditer(text or "")]
    raw, toks, brk, stage = [], [], [], []
    prev = 0
    for m in _TOK.finditer(text or ""):
        gap = text[prev:m.start()]
        raw.append(m.group())
        toks.append(norm_token(m.group()))
        brk.append(bool(raw[:-1]) and bool(_BREAK.search(gap)))
        stage.append(any(a <= m.start() < b for a, b in stage_spans))
        prev = m.end()
    return raw, toks, brk, stage


# ------------------------------------------------------------------------------------------------ names
def _population_names(cfg: dict) -> set[str]:
    """First and last name tokens of every persona in the run's population file (founders and reserves)."""
    p = (cfg or {}).get("population")
    if not p:
        return set()
    return set(_population_file_names(str(p)))


@lru_cache(maxsize=16)
def _population_file_names(p: str) -> tuple:
    import yaml
    path = Path(p)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return ()
    out = set()
    for a in list(data.get("agents") or []) + list(data.get("reserves") or []):
        if isinstance(a, dict) and a.get("name"):
            out |= {norm_token(t) for t in str(a["name"]).split()}
    return tuple(sorted(x for x in out if x))


def run_names(rd) -> set[str]:
    """Name tokens of every agent / persona of the run: manifest agents (incl. reserves), the population file,
    event role holders."""
    names = set()
    for a in (rd.agents or {}).values():
        if isinstance(a, dict) and a.get("name"):
            names |= {norm_token(t) for t in str(a["name"]).split()}
    names |= _population_names(rd.cfg or {})
    try:
        for e in rd.events.values():
            for r in (e.get("roles") or {}).values():
                if isinstance(r, dict) and r.get("name"):
                    names |= {norm_token(t) for t in str(r["name"]).split()}
    except Exception:   # noqa: BLE001 - events are optional
        pass
    return {n for n in names if n and not n.isdigit()}


def run_full_names(rd) -> set[tuple]:
    """(first, last) token pairs of every agent / persona: a full name is a name whatever its case."""
    out = set()
    names = [str(a.get("name")) for a in (rd.agents or {}).values() if isinstance(a, dict) and a.get("name")]
    p = (rd.cfg or {}).get("population")
    if p:
        names += list(_population_file_fullnames(str(p)))
    for n in names:
        t = [norm_token(x) for x in n.split()]
        out |= {(t[i], t[i + 1]) for i in range(len(t) - 1)}
    return out


@lru_cache(maxsize=16)
def _population_file_fullnames(p: str) -> tuple:
    import yaml
    path = Path(p)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return ()
    return tuple(str(a["name"]) for a in list(data.get("agents") or []) + list(data.get("reserves") or [])
                 if isinstance(a, dict) and a.get("name"))


def run_places(rd) -> set[str]:
    locs = set()
    w = (rd.manifest or {}).get("world") or {}
    for l in w.get("graph") or {}:
        locs |= {norm_token(t) for t in _TOK.findall(l)}
    for arenas in (w.get("arenas") or {}).values():
        for ar in arenas or []:
            locs |= {norm_token(t) for t in _TOK.findall(str(ar))}
    return {l for l in locs if l}


def run_roles(rd) -> set[str]:
    """Role nouns the system assigns (demographics.role, co-op roles, onboarding relation types)."""
    out = set()
    for a in (rd.agents or {}).values():
        if not isinstance(a, dict):
            continue
        for v in ((a.get("demographics") or {}).get("role"), a.get("role"), a.get("coop_role")):
            if v:
                out |= {norm_token(t) for t in re.split(r"[\s_]+", str(v))}
    return {t for t in out if t and zipf(t) < 5.5 and t not in FUNCTION}


# ------------------------------------------------------------------------------------------------ well-formedness
class WordClasses:
    """Per-token classes and the well-formedness rule for one run."""

    def __init__(self, names: set, places: set = frozenset(), roles: set = frozenset(), full_names: set = frozenset()):
        self.names = set(names)
        self.full_names = set(full_names)
        self.ambiguous = {n for n in self.names if zipf(n) >= AMBIGUOUS_ZIPF}
        self.places = set(places)
        self.roles = set(roles) | ROLE

    def is_name(self, tok: str, raw: str | None = None) -> bool:
        if tok not in self.names:
            return False
        if tok in self.ambiguous:
            return bool(raw) and raw.lstrip("'-")[:1].isupper()
        return True

    def base(self, tok: str) -> str:
        """The word under a contraction or possessive ("dinner's" -> dinner)."""
        if "'" in tok:
            return tok.split("'", 1)[0]
        return tok

    def token_class(self, tok: str, raw: str | None = None) -> str:
        """num | name | nick | func | report | time | rel | place | content."""
        if len(tok) < 2:
            return "num" if tok.isdigit() else "func"
        squeezed = _ELONGATED.sub(r"\1", tok)
        if squeezed != tok and (squeezed in FUNCTION or squeezed in STOP):
            return "func"                                       # "nooo", "sooo", "ohhh"
        if NUM.match(tok) or (tok[0].isdigit() and not any(c.isalpha() for c in tok)):
            return "num"
        if tok in TIME:
            return "time"
        if "-" in tok:
            parts = [p for p in tok.split("-") if p]
            if len(parts) >= 2 and parts[0] in HOURS and "-".join(parts[1:]) in MINUTES:
                return "time"                                   # "nine-thirty", "ten-fifteen"
            rparts = [p for p in (raw or tok).strip("'-").split("-") if p]
            if len(rparts) != len(parts):
                rparts = parts
            isn = [self.is_name(p, r) for p, r in zip(parts, rparts)]
            if any(isn):
                return "nick" if any(not x and p.isalpha() for x, p in zip(isn, parts)) else "name"
            if all(p in STOP or p in FUNCTION for p in parts):
                return "func"
            return "content"
        if "'" in tok:
            b, suf = tok.split("'", 1)
            braw = (raw or tok).lstrip("'-").split("'", 1)[0]
            if tok.endswith("n't") or suf in ("ve", "ll", "re", "m"):
                return "func"
            if suf == "d":
                return "nick" if self.is_name(b, braw) else "func"
            if suf in ("s", ""):
                if self.is_name(b, braw):
                    return "name"
                if b in FUNCTION or b in STOP:
                    return "func"
                return self.token_class(b, braw)
            return "func"
        if self.is_name(tok, raw):
            return "name"
        if tok in REPORT:
            return "report"
        if tok in FUNCTION or tok in STOP:
            return "func"
        sing = tok[:-1] if tok.endswith("s") and len(tok) > 3 else tok
        if tok in RELATION or tok in self.roles or sing in self.roles:
            return "rel"
        if tok in self.places or sing in self.places:
            return "place"
        return "content"

    def classes(self, toks, raws=None) -> list[str]:
        raws = raws or [None] * len(toks)
        cls = [self.token_class(t, r) for t, r in zip(toks, raws)]
        for i in range(len(toks) - 1):
            if (toks[i], toks[i + 1]) in self.full_names:        # "jordan kim", "miles carter" in any case
                cls[i] = cls[i + 1] = "name"
        return cls

    @staticmethod
    def nickname(toks, cls) -> bool:
        """A person's name used as a coined word: the NAME X / a NAME X, pull(ing)/do(ing) a NAME, classic /
        peak / full / pure / total NAME, going full NAME, such a NAME (the NAME is a single name token)."""
        n = len(toks)
        if cls.count("name") != 1:
            return False
        if n == 3 and toks[0] in NICK_DET and cls[1] == "name" and toks[2] not in FUNCTION \
                and cls[2] not in ("name", "report", "num", "time", "rel") and len(toks[2]) > 2:
            return True
        if n == 3 and toks[0] in NICK_VERBS and toks[1] == "a" and cls[2] == "name":
            return True
        if n == 3 and toks[:2] in (["going", "full"], ["such", "a"]) and cls[2] == "name":
            return True
        if n == 2 and toks[0] in NICK_PRE and cls[1] == "name":
            return True
        return False

    def reject(self, toks, cls=None, raws=None) -> str | None:
        """Why this n-gram is not a plausible reusable expression, or None when it is well-formed."""
        toks = list(toks)
        cls = cls or self.classes(toks, raws)
        n = len(toks)
        if not n:
            return "empty"
        if "num" in cls:
            return "number_or_time"
        if "name" in cls:
            return None if self.nickname(toks, cls) else "person_name"
        if cls[0] in ("func", "report", "time") or cls[-1] in ("func", "report", "time"):
            return "function_word_edge"
        for i in range(n - 1):
            if cls[i] == "report" and toks[i + 1] in COMPLEMENTIZERS:
                return "reporting_frame"
        if n >= 2 and toks[-2] in AUX and (toks[-1].endswith("ing") or cls[-1] == "rel"):
            return "clause_fragment"
        if n >= 2 and toks[-1].endswith("'s"):
            return "dangling_possessive"                        # "job on the co-op's" [laser]
        content = [c for c in cls if c != "func"]
        if all(c in ("rel", "place", "time", "report") for c in content):
            return "bare_role_or_place"
        if n == 1:
            t = toks[0]
            if cls[0] == "nick":
                return None
            b = self.base(t)
            if len(b) < 4:
                return "short_unigram"
            if lemma_zipf(b) >= UNIGRAM_MAX_ZIPF:
                return "common_unigram"
        return None


# ------------------------------------------------------------------------------------------------ infrastructure
class Infrastructure:
    """System (and extra world) wording of a run, as token segments with the tick they reached an agent.

    `rd` may be a prefix view (live.py): every trace-derived segment is then limited to tick <= t."""

    def __init__(self, rd, wc: WordClasses | None = None, lexicon: set | None = None):
        self.rd = rd
        self.wc = wc or WordClasses(run_names(rd), run_places(rd), run_roles(rd), run_full_names(rd))
        self.lexicon = set(lexicon) if lexicon is not None else self._lexicon()
        self.segments: list[tuple[int, str, str]] = []      # (tick, kind, text)
        self._build()
        seen = {}
        for tk, kind, text in self.segments:
            key = (kind, text)
            if key not in seen or tk < seen[key]:
                seen[key] = tk
        self.segments = sorted(((tk, k, t) for (k, t), tk in seen.items()), key=lambda s: (s[0], s[1], s[2]))
        self._rows = []
        for tk, kind, text in self.segments:
            raw, toks, _b, _s = segment(text)
            if not toks:
                continue
            slotted = [SLOT if (t == "n" and r == "N") or self.wc.is_name(t, r) else t for t, r in zip(toks, raw)]
            self._rows.append((tk, kind, text, " ".join(toks), " ".join(slotted), " ".join(fold(t) for t in toks)))
        self._big = {m: " | ".join(r[i] for r in self._rows) for m, i in (("verbatim", 3), ("template", 4), ("folded", 5))}
        for m in self._big:
            self._big[m] = f" {self._big[m]} "

    def _lexicon(self) -> set:
        try:
            return set(self.rd.lexicon_tokens)
        except Exception:   # noqa: BLE001 - population file moved: no lexicon
            return set()

    # ---------------------------------------------------------------- corpus
    def _add(self, tick, kind, text):
        if text and str(text).strip():
            self.segments.append((int(tick), kind, " ".join(str(text).split())))

    def _build(self):
        rd = self.rd
        pl = getattr(rd, "planted", None) or {}
        from backend.analysis.rundata import _is_planted
        # fixed strings of the agent / world code
        for f in _FRAMES:
            self._add(-1, "frame", f)
        try:
            from backend.agents.agent import REL_PHRASES
        except Exception:   # noqa: BLE001 - agents code unavailable: local copy
            REL_PHRASES = _REL_PHRASES
        for rel in set(REL_PHRASES.values()) | {"co-op crewmates", "co-op members"}:
            for fam in _FAMILIARITY:
                for aff in _AFFINITY:
                    self._add(-1, "relationship", f"N N and N N are {rel}; they {fam} and {aff}.")
        npc = list(_NPC)
        try:
            from backend.simulation import world_script as WS
            npc += list(getattr(WS, "NPC_S", [])) + list(getattr(WS, "NPC_Q", []))
        except Exception:   # noqa: BLE001
            pass
        for p in npc:
            self._add(-1, "npc", p)
        w = (rd.manifest or {}).get("world") or {}
        for loc, arenas in (w.get("arenas") or {}).items():
            self._add(-1, "place", loc)
            for ar in arenas or []:
                self._add(-1, "place", f"{loc} {ar}")
        for loc in w.get("graph") or {}:
            self._add(-1, "place", loc)
        # profiles as simulated (manifest), minus the planted control's habit
        for aid, a in (rd.agents or {}).items():
            if not isinstance(a, dict):
                continue
            self._add(-1, "profile", a.get("background"))
            for h in a.get("habits") or []:
                if not (aid == pl.get("agent") and _is_planted(h, pl)):
                    self._add(-1, "profile", h)
            for r in a.get("routine") or []:
                if isinstance(r, dict):
                    self._add(-1, "routine", r.get("activity"))
            for v in (a.get("demographics") or {}).values():
                if isinstance(v, str) and not v.isdigit():
                    self._add(-1, "profile", v)
            for k in ("topics", "hobbies", "clubs"):
                for x in (a.get("interests") or {}).get(k) or []:
                    self._add(-1, "profile", x)
            pers = a.get("personality") or {}
            for x in (pers.get("traits") or []) + [pers.get("communication_style")]:
                self._add(-1, "profile", x)
            for b, rel in (a.get("relationships") or {}).items():
                if isinstance(rel, dict) and rel.get("relation_type"):
                    self._add(-1, "relationship", str(REL_PHRASES.get(rel["relation_type"], rel["relation_type"])))
        # trace: seed / ambient memories, day plans, conversation-context lines, memory frames, onboarding
        for r in rd.of("memory_encoded"):
            st, tk = r.get("source_type"), _tick(r)
            if st in ("seed", "ambient"):
                self._add(tk, "seed" if st == "seed" else "ambient", r.get("text"))
                if st == "ambient":
                    self._add(tk, "ambient", r.get("observation"))
            else:
                fr = _memory_frame(r.get("text"), r.get("observation"), self.wc)
                if fr:
                    self._add(tk, "frame", fr)
        for r in rd.of("day_plan", "replan"):
            for p in r.get("plan") or []:
                if isinstance(p, dict):
                    self._add(_tick(r), "routine", p.get("activity"))
        for u in rd.of("utterance"):
            if not u.get("conversation_id"):
                continue          # a reaction remark's context is its perception (event wording, not scaffolding)
            for line in str(u.get("context") or "").split("\n"):
                line = line.strip()
                if not line:
                    continue
                cut = next((m.group(1) for rx in _CONTEXT_FRAME_ONLY for m in [rx.match(line)] if m), None)
                if cut:
                    self._add(_tick(u), "frame", cut)
                elif not _SPEECH_LINE.match(line) and '"' not in line and "“" not in line:
                    self._add(_tick(u), "context", line)
        for r in rd.of("onboarding"):
            for t in r.get("ties") or []:
                if isinstance(t, dict):
                    self._add(_tick(r), "relationship", t.get("relation_type"))
        for r in rd.of("job_start"):
            self._add(_tick(r), "coop", r.get("project"))
        cfg = rd.cfg or {}
        if (cfg.get("workshop") or {}).get("enabled"):
            try:
                from backend.analysis.battery.targets import world_text
                for t in world_text(cfg):
                    self._add(-1, "coop", t)
            except Exception:   # noqa: BLE001 - optional v3 content pack
                pass

    # ---------------------------------------------------------------- matching
    def match(self, toks) -> dict | None:
        """{"kind": segment kind, "match": verbatim|template|folded|lexicon, "tick", "text"} of the earliest system
        segment containing the phrase, or None. Person names are slotted in both ("template")."""
        toks = [t for t in toks if t]
        if not toks:
            return None
        raw = " ".join(toks)
        slotted = " ".join(SLOT if self.wc.is_name(t, t.capitalize()) else t for t in toks)
        folded = " ".join(fold(t) for t in toks)
        probes = [("verbatim", 3, raw), ("template", 4, slotted)]
        if len(toks) > 1:              # a single word is system wording only verbatim ("clicked" is not "click much")
            probes.append(("folded", 5, folded))
        for mode, idx, s in probes:
            if f" {s} " not in self._big[mode]:
                continue
            for tk, kind, text, *cols in self._rows:
                if f" {s} " in f" {cols[idx - 3]} ":
                    return {"kind": kind, "match": mode, "tick": tk, "text": text[:160]}
        return None

    def lexicon_match(self, toks) -> bool:
        """Every content word is population vocabulary (routines, places, profiles): emergence.lexicon_flag."""
        if not self.lexicon:
            return False
        content = [t for t in toks if t not in STOP and t not in FUNCTION and not t.isdigit() and t not in self.wc.names]
        return bool(content) and all(t in self.lexicon for t in content)

    def wording(self, toks) -> dict:
        m = self.match(toks)
        lex = self.lexicon_match(toks)
        if m is None and lex:
            m = {"kind": "lexicon", "match": "lexicon", "tick": -1, "text": None}
        return {"system": m is not None, "match": m, "in_lexicon": lex}


def _tick(r) -> int:
    try:
        return int(r.get("tick"))
    except (TypeError, ValueError):
        return 0


def _memory_frame(text, obs, wc: WordClasses) -> str | None:
    """The fixed prefix a memory text wraps around its observation ("Nora Ellis remembers that" + "Nora: ..."),
    names slotted, when the observation starts within the first 6 tokens of the memory text."""
    if not text or not obs:
        return None
    raw, toks, _b, _s = segment(text)
    oraw, otoks, _ob, _os = segment(str(obs).split("\n")[0])
    if len(otoks) < 3:
        return None
    for i in range(1, min(7, len(toks) - 2)):
        if toks[i:i + 3] == otoks[:3]:
            head = [SLOT if wc.is_name(t, r) else t for t, r in zip(toks[:i + 1], raw[:i + 1])]
            if all(h == SLOT for h in head):
                return None
            return " ".join("N" if h == SLOT else h for h in head)
    return None


# ------------------------------------------------------------------------------------------------ recitation
class Recitation:
    """Which utterance tokens repeat the speaker's own earlier memory or reflection text verbatim."""

    def __init__(self, rd, k: int = RECITE_K):
        self.k = k
        self.grams: dict[str, dict[tuple, int]] = defaultdict(dict)
        self.node_grams: dict[str, set] = {}
        for r in rd.of("memory_encoded", "reflection", "reminding", "memory_merged"):
            text = r.get("text") if r.get("type") != "memory_merged" else r.get("into_text")
            a = r.get("agent")
            if not text or not a:
                continue
            toks = segment(str(text))[1]
            gs = {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}
            if not gs:
                continue
            tk = _tick(r)
            d = self.grams[a]
            for g in gs:
                if g not in d or tk < d[g]:
                    d[g] = tk
            if r.get("node_id"):
                self.node_grams.setdefault(r["node_id"], set()).update(gs)

    def mask(self, u: dict, toks: list[str]) -> list[bool]:
        k, n = self.k, len(toks)
        out = [False] * n
        d = self.grams.get(u.get("speaker"))
        if not d or n < k:
            return out
        tk = _tick(u)
        retrieved = None
        for i in range(n - k + 1):
            g = tuple(toks[i:i + k])
            t0 = d.get(g)
            if t0 is None:
                continue
            ok = t0 < tk
            if not ok and t0 == tk:
                if retrieved is None:
                    retrieved = set()
                    for nid in u.get("retrieved") or []:
                        retrieved |= self.node_grams.get(nid, set())
                ok = g in retrieved
            if ok:
                for j in range(i, i + k):
                    out[j] = True
        return out
