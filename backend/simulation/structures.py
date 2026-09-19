"""Structure schemas (hidden families) and the shared instance shape (ontology v2 §1.2).

SIMULATOR-ONLY. A family (E1-E4) is an abstract causal structure; a skin (skins/eN.py) is one
concrete wording of it. All families share ONE shape -- the same nodes, beats, offsets, salience,
visibility, involvement, beat places and forced movers -- so the shape carries no information about
the family; only the content of the facts does. Involvement and movers come from the SHAPE, never
from who a skin happens to name, and `validate_skins` makes the text agree with the shape (a fact
names only the people it involves). It also enforces the skin format and the lexical hygiene that
keeps surface wording from giving the family away (or handing agents a ready-made label).
"""
from __future__ import annotations

import itertools
import re
from collections import defaultdict

from backend.simulation.referents import CATEGORIES, HOLDOUT_DOMAINS, TRAIN_DOMAINS
from backend.simulation.world import WORLD_GRAPH

FAMILIES = ["E1", "E2", "E3", "E4"]          # structured families (E0 = the unstructured control label)
NODES = ["n0", "n0_private", "n1", "n2"]
GROUPS = ["n0", "n1", "n2"]                   # composition units: n0 carries n0_private with it

# node -> beat, offset (ticks after start), salience, kind, private (visible to P only by default), and
# involves: the roles that take part in the fact ("second" = {S}, or {Q} in E3). Involvement drives
# participation in perception (attention 1.0, first-hand memory), so it must not vary by family.
SHAPE = {
    "n0": {"beat": 0, "offset": 0, "salience": 0.55, "kind": "action", "private": False, "involves": ("P",)},
    "n0_private": {"beat": 0, "offset": 0, "salience": 0.30, "kind": "inner", "private": True, "involves": ("P",)},
    "n1": {"beat": 1, "offset": 2, "salience": 0.50, "kind": "action", "private": False, "involves": ("second",)},
    "n2": {"beat": 2, "offset": 4, "salience": 0.55, "kind": "action", "private": False, "involves": ("P",)},
}
BEAT_OFFSETS = [0, 2, 4]
SPAN = BEAT_OFFSETS[-1]                       # an instance occupies ticks start .. start + SPAN
# beat -> roles the engine forces to the beat's place. Every beat happens at P's planned position (or the
# skin's pinned location); at beat 1 the second actor (an agent, not an NPC) is brought to P, in EVERY family,
# so neither presence nor forced movement depends on the family.
BEAT_MOVERS = [("P",), ("P", "second"), ("P",)]

SCHEMAS = {
    "E0": {"id": "e0_plain", "second": None,
           "nodes": {"n0": "a mishap", "n1": "an unconnected mishap", "n2": "another unconnected mishap"}},
    "E1": {"id": "e1_cascade", "second": "S",
           "nodes": {"n0": "small slip by P", "n1": "a consequence of that slip hits S",
                     "n2": "a further, otherwise unrelated failure follows"}},
    "E2": {"id": "e2_cancelling", "second": "S",
           "nodes": {"n0": "P makes a mistake", "n1": "S (or the world) independently makes a second mistake",
                     "n2": "the two cancel out and nothing goes wrong"}},
    "E3": {"id": "e3_coincidence", "second": "Q",
           "nodes": {"n0": "P does an unusual thing",
                     "n1": "an unrelated Q independently does the same unusual thing",
                     "n2": "P's outcome: the thing is noticed again or talked about; no causal link"}},
    "E4": {"id": "e4_beneficial", "second": "S",
           "nodes": {"n0": "P fails at something", "n1": "the failure puts P in contact with S or an opportunity",
                     "n2": "P ends up better off because of it"}},
}


def second_role(family: str) -> str:
    """{Q} (an unrelated person) in E3 -- the unrelatedness IS its structure; {S} elsewhere."""
    return "Q" if family == "E3" else "S"


# ----------------------------------------------------------------------------- skin validation
SKIN_KEYS = {"key", "family", "domain", "holdout", "slots", "locations", "facts"}
SLOT_NAMES = ("X", "Y")
PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
MIN_WORDS, MAX_WORDS = 8, 25
FILL_NAME = "Jamie"                           # one-word stand-in for people when counting words

# narrator causal glue / linking words that would state the structure instead of letting it be inferred.
# Checked (case-insensitive) in public facts and the slot options they use; n0_private is P's own reason.
BANNED_PUBLIC = [
    # labels of the structure itself
    r"\b(?:un)?luckily\b", r"\bfortunately\b", r"\bturned out\b", r"\bcoincid\w*", r"\bsame as\b", r"\bjust like\b",
    # causal connectives across facts (ontology v2 §1.2: "because of", "which meant", "thanks to", "as a result",
    # "so") and their equivalents
    r"\bbecause\b", r"\bas a result\b", r"\bthanks to\b", r"\bso\b", r"\b(?:which|that|this|it) meant\b",
    r"\b(?:which|that|this) is (?:why|how)\b", r"\b(?:that|this|it)(?:'|’)?s (?:why|how)\b", r"\bthats (?:why|how)\b",
    r"\btherefore\b", r"\bhence\b", r"\bthus\b", r"\bconsequently\b", r"\b(?:as|in) (?:a )?consequence\b",
    r"\b(?:end|ends|ended|ending) up\b", r"\bas it (?:happened|happens|turned out)\b", r"\bit so happened\b",
    r"\bdue to\b", r"\bowing to\b", r"\bon account of\b", r"\b(?:led|leads|lead|leading) to\b",
    r"\bresult(?:s|ed|ing)? in\b", r"\bin turn\b", r"\bfor (?:that|this) reason\b", r"\bwhich (?:caused|made)\b",
]
# "the (whole) backpack thing / reply-all saga": ready-made labels for an incident
_NICK_HEAD = r"(?:thing|incident|mix-?up|saga|fiasco|debacle|episode|affair)"
NICKNAME = re.compile(rf"\bthe\s+((?:[\w'-]+\s+){{1,3}}){_NICK_HEAD}\b", re.I)
_NICK_OK = {"same", "only", "other", "next", "first", "last", "one", "right", "wrong", "whole", "entire",
            "main", "real", "best", "worst"}
QUOTED = re.compile(r"[\"“”]|(?:^|\s)'[^']+'(?=[\s.,!?;:]|$)")

STOPWORDS = set("""a an the and or but if then so of to in on at for with by from as is are was were be been being it
its this that these those i you he she they we me him her them my your his their our us do does did done have has had
not no yes just very really about into over after before up down out there here what which who whom when where why how
all any some can could would should will shall may might must also too than oh ok okay well like so actually kind sort
lot bit maybe right sure get got getting go going gone went come came know knew think thought mean said say says tell
told make made see saw look looked want wanted need needed feel felt one two thing things something anything nothing
everything someone anyone everyone somebody today day time now still even again always never ever much many more most
other another same own way back good great new old last next first little big whole without while during through
across around near behind until onto off between under along instead almost already later once twice soon only
each every both either neither whose because while since though although yet himself herself themselves itself
""".split())
_WORD = re.compile(r"\{[^{}]*\}|[a-z0-9][a-z0-9'\-]*")
REFERENT_WORDS = {w for names in CATEGORIES.values() for n in names for w in re.findall(r"[a-z0-9][a-z0-9'\-]*",
                                                                                         n.lower())}


def _fill(text: str, subst: dict) -> str:
    for k, v in subst.items():
        text = text.replace("{" + k + "}", v)
    return text


def _norm(tok: str) -> str:
    tok = tok.strip("'-")
    return tok[:-2] if tok.endswith("'s") else tok


def _tokens(text: str) -> tuple[list[str], list[bool]]:
    """Normalised tokens of a text and, per token, whether it is a content word (not a stopword,
    placeholder, referent-name word or single letter)."""
    toks = [t if t.startswith("{") else _norm(t) for t in _WORD.findall(text.lower())]
    return toks, [not (t.startswith("{") or len(t) < 2 or t in STOPWORDS or t in REFERENT_WORDS) for t in toks]


def content_bigrams(text: str, lexicon_bigrams=frozenset()) -> set[str]:
    """Adjacent word pairs where both are content words: not stopwords, placeholders, referent-name words
    or single letters, and not a population-lexicon bigram. Placeholders break adjacency."""
    toks, ok = _tokens(text)
    out = set()
    for i in range(len(toks) - 1):
        if ok[i] and ok[i + 1]:
            bg = f"{toks[i]} {toks[i + 1]}"
            if bg not in lexicon_bigrams:
                out.add(bg)
    return out


def content_trigrams(text: str, lexicon_bigrams=frozenset()) -> set[str]:
    """Content phrases the bigram check cannot see because a stopword sits in the middle: content word,
    ONE stopword, content word ("stack of flyers", "ladle in hand"). Placeholders break adjacency. A
    trigram both of whose word pairs are population-lexicon bigrams is shared campus vocabulary."""
    toks, ok = _tokens(text)
    out = set()
    for i in range(len(toks) - 2):
        a, m, b = toks[i:i + 3]
        if ok[i] and ok[i + 2] and m in STOPWORDS:
            if not (f"{a} {m}" in lexicon_bigrams and f"{m} {b}" in lexicon_bigrams):
                out.add(f"{a} {m} {b}")
    return out


def _lexicon_bigrams(lexicon) -> frozenset:
    if not lexicon:
        return frozenset()
    if isinstance(lexicon, dict):
        return frozenset(lexicon.get("bigrams", []))
    return frozenset(lexicon)


def _skin_texts(skin: dict) -> list[str]:
    """Everything a skin can put into facts: the four templates plus every slot option."""
    texts = [t for t in (skin.get("facts") or {}).values() if isinstance(t, str)]
    for opts in (skin.get("slots") or {}).values():
        texts += [o for o in (opts or []) if isinstance(o, str)]
    return texts


def _variants(skin: dict, node: str) -> list[str]:
    """Every filled version of one fact: each referent of the skin's domain x each slot option."""
    text = skin["facts"][node]
    slots = {k: v for k, v in (skin.get("slots") or {}).items() if "{" + k + "}" in text}
    refs = CATEGORIES.get(skin.get("domain"), ["the thing"]) if "{R}" in text else [""]
    base = {"P": FILL_NAME, "S": FILL_NAME, "Q": FILL_NAME}
    out = []
    for r in refs:
        for combo in itertools.product(*slots.values()):
            out.append(_fill(text, {**base, "R": r, **dict(zip(slots, combo))}))
    return out


def _check_skin(skin: dict) -> list[str]:
    key = skin.get("key", "?")
    probs = []
    extra = set(skin) - SKIN_KEYS
    if extra:
        probs.append(f"{key}: unknown keys {sorted(extra)}")
    fam, dom = skin.get("family"), skin.get("domain")
    if fam not in FAMILIES:
        return probs + [f"{key}: family must be one of {FAMILIES}, got {fam!r}"]
    if dom not in CATEGORIES:
        return probs + [f"{key}: unknown domain {dom!r}"]
    if key != f"{fam.lower()}_{dom}":
        probs.append(f"{key}: key must be '{fam.lower()}_{dom}'")
    if skin.get("holdout") is not (dom in HOLDOUT_DOMAINS):
        probs.append(f"{key}: holdout must be {dom in HOLDOUT_DOMAINS} for domain {dom!r}")
    facts = skin.get("facts")
    if not isinstance(facts, dict) or set(facts) != set(NODES) or not all(isinstance(t, str) for t in facts.values()):
        return probs + [f"{key}: facts must have exactly the string nodes {NODES}"]
    # slots
    slots = skin.get("slots") or {}
    if not isinstance(slots, dict):
        return probs + [f"{key}: slots must be a dict"]
    slots_ok = True
    for s, opts in slots.items():
        if s not in SLOT_NAMES:
            probs.append(f"{key}: slot {{{s}}} not allowed (only {{X}}, {{Y}})")
        if not isinstance(opts, list) or not 2 <= len(opts) <= 4 or not all(isinstance(o, str) and o for o in opts):
            probs.append(f"{key}: slot {{{s}}} needs 2-4 non-empty string options")
            slots_ok = False
    # locations
    locs = skin.get("locations") or {}
    for node, loc in (locs.items() if isinstance(locs, dict) else []):
        if node not in GROUPS:
            probs.append(f"{key}: locations may pin only {GROUPS}, got {node!r}")
        if loc not in WORLD_GRAPH:
            probs.append(f"{key}: unknown location {loc!r}")
    # placeholders and roles
    role2, other = second_role(fam), ("S" if fam == "E3" else "Q")
    allowed = {"P", "R", role2} | set(SLOT_NAMES)
    used = set()
    for node, text in facts.items():
        ph = PLACEHOLDER.findall(text)
        used |= set(ph)
        bad = sorted(set(ph) - allowed)
        if bad:
            probs.append(f"{key}.{node}: placeholders {bad} not allowed"
                         + (f" ({fam} uses {{{role2}}}, not {{{other}}})" if other in bad else ""))
        if text.count("{") != text.count("}") or len(ph) != text.count("{"):
            probs.append(f"{key}.{node}: unbalanced braces")
        if text.lstrip().startswith("{R}"):
            probs.append(f"{key}.{node}: starts with {{R}}; put a person first")
        if node in ("n0", "n0_private", "n2") and "{P}" not in text:
            probs.append(f"{key}.{node}: must mention {{P}}")
    if facts["n0"].count("{R}") != 1:
        probs.append(f"{key}.n0: {{R}} must appear exactly once")
    if "{R}" in facts["n0_private"]:
        probs.append(f"{key}.n0_private: {{R}} may appear only in n0 (once) and n1/n2")
    if "{" + role2 + "}" not in facts["n1"]:
        probs.append(f"{key}.n1: must mention the second actor {{{role2}}}")
    # a fact names exactly the people it involves (SHAPE), so involvement, presence and the viewpoint name
    # guard never depend on which family's wording happens to name whom
    for node in ("n0", "n0_private", "n2"):
        if "{" + role2 + "}" in facts[node]:
            probs.append(f"{key}.{node}: must not mention {{{role2}}} ({node} involves P only)")
    if "{P}" in facts["n1"]:
        probs.append(f"{key}.n1: must not mention {{P}} (n1 involves the second actor only; P is there as a bystander)")
    for s in used & set(SLOT_NAMES):
        if s not in slots:
            probs.append(f"{key}: {{{s}}} used but has no slot options")
    for s in set(slots) - used:
        probs.append(f"{key}: slot {{{s}}} defined but never used")
    if not slots_ok or not (used & set(SLOT_NAMES)) <= set(slots):
        return probs
    # sentence form and length (after filling every referent / slot option)
    for node in NODES:
        for v in _variants(skin, node):
            n = len(v.split())
            if not MIN_WORDS <= n <= MAX_WORDS:
                probs.append(f"{key}.{node}: {n} words after filling (need {MIN_WORDS}-{MAX_WORDS}): {v!r}")
                break
            if not v.rstrip().endswith((".", "!", "?")) or re.search(r"[.!?]\s+[A-Z]", v):
                probs.append(f"{key}.{node}: must be exactly one sentence: {v!r}")
                break
    # causal glue in public facts; labels anywhere
    public = [facts[n] for n in ("n0", "n1", "n2")]
    public += [o for s, opts in slots.items() if any("{" + s + "}" in t for t in public) for o in opts]
    for text in public:
        for pat in BANNED_PUBLIC:
            m = re.search(pat, text, re.I)
            if m:
                probs.append(f"{key}: banned causal/label wording {m.group(0)!r} in public text {text!r}")
    for text in _skin_texts(skin):
        for m in NICKNAME.finditer(text):
            inner = [w.lower() for w in m.group(1).split()]
            if any(w not in _NICK_OK for w in inner) and not any(w in STOPWORDS and w not in _NICK_OK for w in inner):
                probs.append(f"{key}: nickname-like phrase {m.group(0)!r}")
        if QUOTED.search(text):
            probs.append(f"{key}: quoted text (possible label) in {text!r}")
    return probs


def validate_skins(skins: list[dict], lexicon=None) -> list[str]:
    """Problems with a skin set (empty list = valid). `lexicon`: population_lexicon() output (or an
    iterable of "w1 w2" bigrams) whose bigrams are shared campus vocabulary, not skin content.

    Shape: 11 skins per family present (one per train domain, one per holdout domain), key/holdout
    consistency, only the allowed placeholders ({S} in E1/E2/E4, {Q} in E3), {R} exactly once in n0,
    each fact names only the people it involves ({P} in n0/n0_private/n2, the second actor in n1),
    location pins identical across families for each domain, 8-25 words per fact after filling.
    Hygiene: no causal glue in public facts, no nickname-like phrases or quoted labels, and no content
    bigram or content trigram (content word, stopword, content word) shared across families or between
    a family's holdout and train skins.
    """
    probs = []
    keys = [s.get("key") for s in skins]
    probs += [f"duplicate skin key {k!r}" for k in sorted({k for k in keys if keys.count(k) > 1})]
    by_fam, by_dom = defaultdict(list), defaultdict(list)
    for s in skins:
        probs += _check_skin(s)
        by_fam[s.get("family")].append(s)
        if s.get("family") in FAMILIES:
            by_dom[s.get("domain")].append(s)
    for fam in sorted(f for f in by_fam if f in FAMILIES):
        doms = [s.get("domain") for s in by_fam[fam]]
        for d in TRAIN_DOMAINS + HOLDOUT_DOMAINS:
            if doms.count(d) != 1:
                probs.append(f"{fam}: needs exactly one skin for domain {d!r}, has {doms.count(d)}")
    # a pin moves the beat (and its movers), so a per-family pin difference would be a shape leak
    for d in sorted(by_dom, key=str):
        pins = {s.get("key"): dict(s.get("locations") or {}) for s in by_dom[d]}
        if len({tuple(sorted(p.items())) for p in pins.values()}) > 1:
            probs.append(f"domain {d!r}: location pins differ across families: "
                         + ", ".join(f"{k} {pins[k]}" for k in sorted(pins, key=str)))
    lex = _lexicon_bigrams(lexicon)
    fam_of = {s.get("key"): s.get("family") for s in skins}
    hold = {s.get("key"): bool(s.get("holdout")) for s in skins}
    for what, fn in (("bigram", content_bigrams), ("trigram", content_trigrams)):
        users = defaultdict(set)
        for s in skins:
            if s.get("family") in FAMILIES:
                for t in _skin_texts(s):
                    for g in fn(t, lex):
                        users[g].add(s.get("key"))
        for g in sorted(users):
            ks = sorted(users[g])
            if len({fam_of[k] for k in ks}) > 1:
                probs.append(f"content {what} {g!r} shared across families: {', '.join(ks)}")
            for fam in sorted({fam_of[k] for k in ks}):
                h = [k for k in ks if fam_of[k] == fam and hold[k]]
                t = [k for k in ks if fam_of[k] == fam and not hold[k]]
                if h and t:
                    probs.append(f"content {what} {g!r} shared by {fam} holdout {', '.join(h)} and train {', '.join(t)}")
    return probs
