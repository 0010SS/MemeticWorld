"""Population lexicon: the campus vocabulary every agent is handed by its world and profile
(routine activities, places, rooms, classes, clubs, habits, backgrounds).

Used (a) agent-side by verbatim stickiness, so shared campus vocabulary does not count as
distinctive wording, and (b) observer-side as the vocabulary baseline for emergence (a phrase made
of this vocabulary spreads because of shared routines, not because it was coined and passed on).
Pure function of the population; written to the manifest.
"""
from __future__ import annotations

import re

from backend.simulation.world import ARENAS, WORLD_GRAPH

_TOK = re.compile(r"[a-z][a-z'\-]*")


def _toks(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def population_lexicon(profiles: dict) -> dict:
    """-> {"tokens": sorted words, "bigrams": sorted "w1 w2" strings}."""
    texts = list(WORLD_GRAPH) + [a for arenas in ARENAS.values() for a in arenas]
    for p in profiles.values():
        texts += [p.background] + list(p.habits) + [r.activity for r in p.routine]
        texts += [str(v) for v in (p.demographics or {}).values()]
        for key in ("topics", "hobbies", "clubs"):
            texts += [str(x) for x in (p.interests or {}).get(key, [])]
    tokens, bigrams = set(), set()
    for t in texts:
        w = _toks(t)
        tokens |= set(w)
        bigrams |= {f"{a} {b}" for a, b in zip(w, w[1:])}
    return {"tokens": sorted(tokens), "bigrams": sorted(bigrams)}
