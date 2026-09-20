"""How the world refers to a place when it talks to an agent.

The principle (D75): **the world gives agents percepts and situated, relative descriptions, never canonical
names.** Naming is the agents' business. A run of the seeded naming experiment cannot measure whose name wins
while every prompt hands the agent the world's own label for the referent: in the first live 100-agent run 72%
of chat prompts contained the literal string "Dining Hall" and only 10% contained the agent's seeded name for
it, so agents copied the label the system had just given them.

Internal ids, `WORLD_GRAPH`/`ARENAS` keys, the map, the trace, the frames and the observer all keep the
canonical names. Only agent-facing text goes through here.

`world.reference_mode`:

  canonical  (default)  what v2 did: the canonical key, verbatim. Existing runs reproduce byte for byte.
  situated              a description from `world.PLACE_DESCRIPTIONS` / `world.ARENA_DESCRIPTIONS`,
                        personalised to the agent where the agent's own profile makes it relative.

Personalisation is derived from the profile, never hard-coded per place:

  * the agent's home arena           -> "your room"
  * the place its routine spends the most time in, other than home
                                     -> "the lab where you work" / "the building where you have class" /
                                        "the building where you teach" / ... , chosen from that routine's own
                                        activity wording and the rooms it uses there
  * everywhere else                  -> the generic description

Descriptions are lower case and carry their own article ("the dining hall beside the freshman residences"),
so `phrase()` adds "the " only in canonical mode and the two modes render the same sentences.
"""
from __future__ import annotations

import re

from backend.simulation.world import ARENA_DESCRIPTIONS, PLACE_DESCRIPTIONS, WORLD_GRAPH

CANONICAL = "canonical"
SITUATED = "situated"
MODES = (CANONICAL, SITUATED)

# The agent's own `home` tile. `home` is this schema's campus ANCHOR, not necessarily a residence (27 of the
# homewood100 students are anchored at the library, every dining worker at their own serving floor), so which
# of the two it is comes from the generic room description's own article: "a single bedroom" / "a graduate
# flat upstairs" is one room among identical ones, i.e. a room of your own; "the serving floor" / "the shared
# work tables" is the building's one shared room, i.e. the spot you are always at. No per-place special case.
HOME_ARENA = "your room"
HOME_SPOT = "your usual spot"

# How an agent thinks of the place its own day revolves around, read off that routine's own activity wording
# and the rooms it uses there. Nothing here mentions a place, so a new place on the map needs no new rule and
# no place gets a hand-written personal description. Strongest evidence first: the rooms say what the place
# physically is, and only then does the activity wording decide between teaching, class, work and study.
_FOCUS_TEACH = re.compile(r"\bteach|\btutor|lesson plan|lecturing|student consultation|office hours", re.I)
_FOCUS_CLASS = re.compile(r"\bclass\b|\bclasses\b|lecture|seminar|attending a", re.I)
_FOCUS_LAB = re.compile(r"experiment|\blab\b|specimen|fieldwork", re.I)
_FOCUS_WORK = re.compile(r"\bshift\b|handover|duties|rounds|work messages|colleagues|serving|maintenance"
                         r"|working as|\bworker\b", re.I)
_FOCUS_STUDY = re.compile(r"coursework|revising|reviewing notes|study|studying|reading|scholarly|writing", re.I)
_LAB_ROOM = re.compile(r"\blab\b|makerspace|bench", re.I)
_FOCUS_DEFAULT = "the place where you spend most of your day"


def _focus_phrase(text: str, rooms) -> str:
    if any(_LAB_ROOM.search(r) for r in rooms):    # the rooms are the strongest evidence of what a place is
        return "the lab where you work"
    if _FOCUS_TEACH.search(text):
        return "the building where you teach"
    if _FOCUS_CLASS.search(text):
        return "the building where you have class"
    if _FOCUS_LAB.search(text):
        return "the lab where you work"
    if _FOCUS_WORK.search(text):
        return "the building where you work"
    if _FOCUS_STUDY.search(text):
        return "the building where you study"
    return _FOCUS_DEFAULT

# Routine entries carry a start time only; the last one of the day is given this much weight so that a single
# long evening block does not silently outrank a whole morning.
_LAST_ENTRY_MINUTES = 60


def mode(cfg=None) -> str:
    """The configured reference mode. Anything unset or unrecognised is `canonical`, so a config that
    predates this key behaves exactly as it did."""
    if not isinstance(cfg, dict):
        return CANONICAL
    m = str(((cfg.get("world") or {}) if isinstance(cfg.get("world"), dict) else {})
            .get("reference_mode") or CANONICAL).strip().lower()
    return m if m in MODES else CANONICAL


# --- profile access ---------------------------------------------------------------------------------------

def _profile(agent):
    """`agent` may be an Agent, a bare AgentProfile, or None."""
    if agent is None:
        return None
    p = getattr(agent, "profile", None)
    if p is not None and hasattr(p, "routine"):
        return p
    return agent if hasattr(agent, "routine") and hasattr(agent, "home") else None


def _minutes(value) -> int:
    """'07:30' -> 450. YAML 1.1 already parses an unquoted 07:30 to 450 minutes."""
    if isinstance(value, int):
        return value
    try:
        h, m = str(value).split(":")
        return int(h) * 60 + int(m)
    except ValueError:
        return 0


def _focus(prof) -> tuple[str | None, str]:
    """(location, personalised description) for the place this profile's routine spends the most time in.

    Derived from the routine and cached on the profile. Nothing is excluded: `home` is a campus anchor, so
    for a dining worker anchored at their own serving floor that place really is where the day happens, and
    for a student who only wakes up in their room it is not."""
    cached = getattr(prof, "_reference_focus", None)
    if cached is not None:
        return cached
    routine = list(prof.routine or [])
    spent: dict[str, int] = {}
    words: dict[str, list[str]] = {}
    rooms: dict[str, list[str]] = {}
    for i, entry in enumerate(routine):
        start = _minutes(entry.time)
        end = _minutes(routine[i + 1].time) if i + 1 < len(routine) else start + _LAST_ENTRY_MINUTES
        spent[entry.location] = spent.get(entry.location, 0) + max(0, end - start)
        words.setdefault(entry.location, []).append(entry.activity or "")
        if entry.arena:
            rooms.setdefault(entry.location, []).append(entry.arena)
    if not spent:
        out = (None, _FOCUS_DEFAULT)
    else:
        loc = max(sorted(spent), key=lambda k: spent[k])
        out = (loc, _focus_phrase(" ".join(words.get(loc, [])), rooms.get(loc, [])))
    try:
        prof._reference_focus = out
    except (AttributeError, TypeError):  # pragma: no cover - frozen profile
        pass
    return out


# --- the descriptions -------------------------------------------------------------------------------------

def place(location: str, agent=None, cfg=None, capitalize: bool = False) -> str:
    """Situated description of a place, with no room. Canonical mode returns the key unchanged."""
    if mode(cfg) == CANONICAL or location not in PLACE_DESCRIPTIONS:
        return _cap(location, capitalize)
    prof = _profile(agent)
    if prof is not None:
        focus_loc, focus_desc = _focus(prof)
        if focus_loc == location:
            return _cap(focus_desc, capitalize)
    return _cap(PLACE_DESCRIPTIONS[location], capitalize)


def room(location: str, arena: str, agent=None, cfg=None, capitalize: bool = False) -> str:
    """Situated description of one room of a place. Canonical mode returns the arena key unchanged."""
    if mode(cfg) == CANONICAL:
        return _cap(arena, capitalize)
    desc = ARENA_DESCRIPTIONS.get(location, {}).get(arena, arena)
    prof = _profile(agent)
    if prof is not None:
        home = prof.home or {}
        if (home.get("location"), home.get("arena")) == (location, arena):
            desc = HOME_ARENA if re.match(r"an? ", desc) else HOME_SPOT
    return _cap(desc, capitalize)


def describe(location: str, arena: str | None = None, agent=None, cfg=None, capitalize: bool = False) -> str:
    """The world's agent-facing reference to (location[, arena]).

    canonical: "Dining Hall" / "Dining Hall (Main Floor)" -- byte for byte what v2 rendered.
    situated:  "the dining hall beside the freshman residences (the serving floor)".
    """
    out = place(location, agent=agent, cfg=cfg)
    if arena:
        out = f"{out} ({room(location, arena, agent=agent, cfg=cfg)})"
    return _cap(out, capitalize)


def phrase(location: str, arena: str | None = None, agent=None, cfg=None, capitalize: bool = False) -> str:
    """`describe()` for the slot that used to be written "the {location}". Canonical mode supplies the
    article ("the Dining Hall"); a situated description already carries its own."""
    out = describe(location, arena, agent=agent, cfg=cfg)
    if mode(cfg) == CANONICAL:
        out = f"the {out}"
    return _cap(out, capitalize)


def inside(location: str, arena: str, agent=None, cfg=None, capitalize: bool = False) -> str:
    """The "<arena> in <sector>" slot (upstream's iterative chat prompt, and group_conversation's `where`).
    Canonical mode is exactly `f"{arena} in {location}"`; situated mode reads
    "the serving floor in the dining hall beside the freshman residences"."""
    return _cap(f"{room(location, arena, agent=agent, cfg=cfg)} in "
                f"{place(location, agent=agent, cfg=cfg)}", capitalize)


def _cap(text: str, capitalize: bool) -> str:
    return (text[:1].upper() + text[1:]) if capitalize and text else text


# --- the MOVE option list ---------------------------------------------------------------------------------

def options(locations=None, agent=None, cfg=None) -> str:
    """The places an agent may be offered, as one comma-joined list. Canonical mode is the old
    `", ".join(sorted(WORLD_GRAPH))`."""
    locs = sorted(WORLD_GRAPH) if locations is None else list(locations)
    return ", ".join(place(x, agent=agent, cfg=cfg) for x in locs)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()).strip().replace("  ", " ")


def resolve(answer, agent=None, cfg=None, locations=None) -> str | None:
    """Map a model's answer back to a canonical place id. Accepts either form -- the canonical name or the
    situated description it was offered (including this agent's personalised one) -- so an answer is never
    thrown away just because the two modes word the option differently."""
    text = _norm(answer)
    if not text:
        return None
    locs = sorted(WORLD_GRAPH) if locations is None else list(locations)
    table: list[tuple[str, str]] = []
    for loc in locs:
        table.append((_norm(loc), loc))
        if mode(cfg) != CANONICAL:
            table.append((_norm(place(loc, agent=agent, cfg=cfg)), loc))
            table.append((_norm(PLACE_DESCRIPTIONS.get(loc, loc)), loc))
    for key, loc in table:
        if key and key == text:
            return loc
    # the model wrapped the option in a sentence ("MOVE to the dining hall beside ..."): longest key wins,
    # so "Research Lab" never loses to a shorter name that happens to be a substring of it.
    hits = [(len(key), loc) for key, loc in table if key and key in text]
    return max(hits)[1] if hits else None
