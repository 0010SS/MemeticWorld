"""NONCE and NONE cues for the comprehension format N (ONTOLOGY_V3 §5.3). OBSERVER ONLY.

NONCE is a pseudoword sequence matched to a target expression in number of words and syllables per word, built
from plain CV(C) syllables so it has Zipf frequency 0 (checked against a small word list and the embedder's
stop words). NONE is the no-cue carrier.
"""
from __future__ import annotations

import re

from backend.simulation.rngs import seed_rng

NONE_TEXT = "Heads up about the laser today."
CARRIER = "{c} again on the laser today."

_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "gr", "pl", "sk", "tr"]
_VOWELS = ["a", "e", "i", "o", "u"]
_CODAS = ["", "", "", "n", "r", "s", "m", "l"]
# a few very common short English words the generator must never produce
_REAL = set("""a an as at be by do go he if in is it me my no of on or so to up us we am are art bad bed big bit
bus but can cat cut dog dot fan far fed fun gas gun hat hot let lot man mat men met mud net not pan pen pet pin
pot ran red rod run sad sat set sit son sun tan ten tin top van vet win bam dim dun fin gum hum kin lab lap lid
lit log mob nod nun pal pit pun rim rot rug rum sip sob tab tar tub zen zip bar car den ban bin""".split())


def syllables(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", word.lower())))


def make_nonce(target: str | None, seed, n_words: int = 2) -> str:
    """A pseudoword expression matched to `target` (words, syllables), deterministic in (seed, target)."""
    words = re.findall(r"[A-Za-z0-9']+", target or "") or ["xx"] * n_words
    rng = seed_rng(seed, "nonce", target or "")
    out = []
    for w in words:
        for _ in range(50):
            k = syllables(w) if re.search(r"[A-Za-z]", w) else 1
            s = "".join(_ONSETS[int(rng.integers(len(_ONSETS)))] + _VOWELS[int(rng.integers(len(_VOWELS)))]
                        for _ in range(k))
            s += _CODAS[int(rng.integers(len(_CODAS)))]
            if s not in _REAL and syllables(s) == k:
                break
        out.append(s)
    return " ".join(out)


def carrier(cue: str | None) -> str:
    """The note text for a cue; None -> the NONE carrier."""
    if not cue:
        return NONE_TEXT
    c = cue.strip()
    return CARRIER.format(c=c[:1].upper() + c[1:])
