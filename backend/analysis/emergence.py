"""Exposure-conditioned adoption per candidate (OBSERVER ONLY; ontology v2 §4).

A convention *emerges* when people start using an expression after hearing it, and carry it into new
exchanges, more than people who use it without having heard it. Per candidate:

- originator: the first speaker;
- exposure is causal (`rundata.precedes`): usage x exposes listener a for a use y only when x's tick is
  strictly earlier, or x is an earlier turn of the same conversation at the same tick. Remarks and
  conversation openings of one tick are decided in parallel, and parallel conversations do not hear each
  other, so witnesses reacting to the same beat at the same tick are independent users, not adopters;
- independent: an agent with at least one use that no exposure causally precedes (it came up with the
  expression itself);
- exposed: every other agent that heard a usage (all its uses, if any, come after an exposure);
- n_adopters: exposed agents that used it at all (includes echoes: repeating it right after hearing it
  inside the same conversation). Reported only;
- n_adopters_carried: exposed agents with a *carried* use: a use in a different exchange (conversation,
  or its own remark) from an exposure, at a strictly later tick than that exposure, and not preceded by
  an exposure inside its own exchange (so it is not an echo of what was just said there);
- one-sided Fisher exact test on [[carried, exposed - carried], [independents, unexposed - independents]];
- in_lexicon: every content word is population vocabulary (routines, places, profiles): such a
  phrase spreads because of shared routines, not because it was coined and passed on;
- in_world_text: the phrase is world-produced event wording: its tokens occur in order inside one
  world text segment (a fact text, a viewpoint rendering, a referent name; the narrative for records
  without beats) with at most WORLD_MAX_GAP extra tokens in between ("submitted to the wrong" <-
  "Ethan submitted a problem set to the wrong course"; inflections are folded, so "submitting to the
  wrong" matches too). Every witness perceived that wording, so its repetition is not transmission.
  `world_match` says "verbatim" or "gapped".

- in_system_text: the phrase is SYSTEM wording (wording.Infrastructure: relationship lines, routines, seed and
  ambient memories, memory frames, profile text ...; candidates.py flags it as c["wording"]["system"]).

spread = n_adopters_carried >= 2 and n_adopters_carried > n_independent;
emerged = spread and not in_lexicon and not in_world_text and not in_system_text.
"""
from __future__ import annotations

import math
import re

from backend.analysis.candidates import STOP, tokens
from backend.analysis.rundata import chron_key, precedes, utterance_key


def _pmf_table(r1: int, r2: int, c1: int):
    """Hypergeometric pmf of the top-left cell given the margins: x -> C(r1,x) C(r2,c1-x) / C(n,c1)."""
    n = r1 + r2
    lo, hi = max(0, c1 - r2), min(r1, c1)
    tot = math.comb(n, c1)
    return {x: math.comb(r1, x) * math.comb(r2, c1 - x) / tot for x in range(lo, hi + 1)}


def fisher_exact(a: int, b: int, c: int, d: int, alternative: str = "greater") -> float:
    """Fisher's exact test for [[a, b], [c, d]] (scipy.stats.fisher_exact semantics).
    'greater': odds ratio > 1 (row 1 has the higher rate in column 1); 'less'; 'two-sided' sums every
    table no more likely than the observed one (relative tolerance 1e-7, as scipy)."""
    pmf = _pmf_table(a + b, c + d, a + c)
    if alternative == "greater":
        p = sum(v for x, v in pmf.items() if x >= a)
    elif alternative == "less":
        p = sum(v for x, v in pmf.items() if x <= a)
    elif alternative == "two-sided":
        p0 = pmf[a]
        p = sum(v for v in pmf.values() if v <= p0 * (1 + 1e-7))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return min(1.0, p)


_WORD = re.compile(r"[a-z][a-z'\-]*")


def vocabulary(rd) -> dict:
    """Population lexicon tokens (manifest `population_lexicon`, else recomputed the way the engine does:
    population file + generated topology, before planting). Tokens that come only from a planted-phrase
    habit (controls.planted_phrase) are removed, so the positive control is not discounted as routine
    vocabulary."""
    man = rd.manifest.get("population_lexicon")
    fresh = None

    def from_file():                  # as the engine computes it: after topology, before planting
        from backend.analysis.rundata import simulated_profiles
        from backend.simulation.lexicon import population_lexicon
        return set(population_lexicon(simulated_profiles(rd.cfg, planted=False))["tokens"])
    if man:
        vocab, source = set(man.get("tokens", [])), "manifest"
    else:
        try:
            fresh = vocab = from_file()
            source = "population_file"
        except Exception:                 # population file moved: fall back to the manifest's profiles
            vocab, source = _manifest_vocab(rd), "manifest_agents"
    habit = ((rd.cfg.get("controls") or {}).get("planted_phrase") or {}).get("habit")
    if habit:
        quoted = " ".join(re.findall(r"[\"“]([^\"”]+)[\"”]", habit)) or habit
        if fresh is None:
            try:
                fresh = from_file()
            except Exception:
                fresh = set()
        vocab -= {t for t in _WORD.findall(quoted.lower()) if t not in fresh}
    return {"tokens": vocab, "source": source}


def _manifest_vocab(rd) -> set:
    from backend.simulation.world import ARENAS, WORLD_GRAPH
    texts = list(WORLD_GRAPH) + [a for ar in ARENAS.values() for a in ar]
    for a in rd.agents.values():
        texts += [a.get("background", "")] + list(a.get("habits", [])) + [r["activity"] for r in a.get("routine", [])]
        texts += [str(v) for v in (a.get("demographics") or {}).values()]
        for k in ("topics", "hobbies", "clubs"):
            texts += [str(x) for x in (a.get("interests") or {}).get(k, [])]
    return {w for t in texts for w in _WORD.findall(t.lower())}


WORLD_MAX_GAP = 3


def world_segments(rd) -> list[list[str]]:
    """Token lists of every world-produced wording an agent could perceive: fact texts, viewpoint
    renderings, referent names (narratives only for records without beats)."""
    parts = []
    for e in rd.events.values():
        facts = [f.get("text", "") for b in e.get("beats", []) for f in b.get("facts", [])]
        parts += facts or [e.get("narrative") or ""]
        parts += [r.get("name", "") for r in e.get("referents") or []]
    for r in rd.of("viewpoint"):
        parts += [f.get("perceived") or "" for f in r.get("facts", [])]
    return [t for t in (tokens(p) for p in parts if p) if t]


def _norm(t: str) -> str:
    """Light inflection folding for world matching only: submitted / submitting -> submit, notes -> not(e)."""
    for suf in ("ing", "ed", "s"):
        if t.endswith(suf) and not t.endswith("ss") and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            if len(t) >= 4 and t[-1] == t[-2] and t[-1] not in "lsz":
                t = t[:-1]
            break
    return t[:-1] if len(t) > 3 and t.endswith("e") else t


def world_match(phrase: list[str], segments: list[list[str]], max_gap: int = WORLD_MAX_GAP,
                normed: list[list[str]] | None = None) -> str | None:
    """"verbatim" when the phrase tokens occur contiguously in one segment; "gapped" when they occur in
    order (inflections folded) with at most max_gap extra tokens; else None. `normed`: the segments'
    folded tokens, precomputed by the caller."""
    if not phrase:
        return None
    raw = f" {' '.join(phrase)} "
    if any(raw in f" {' '.join(seg)} " for seg in segments):
        return "verbatim"
    ph = [_norm(t) for t in phrase]
    k, need = len(ph), set(ph)
    for sn in normed if normed is not None else ([_norm(t) for t in seg] for seg in segments):
        if not need <= set(sn):
            continue
        for i, t in enumerate(sn):
            if t != ph[0]:
                continue
            j, pos = 1, i
            for q in range(i + 1, len(sn)):
                if j == k:
                    break
                if sn[q] == ph[j]:
                    j, pos = j + 1, q
            if j == k and pos - i + 1 - k <= max_gap:
                return "gapped"
    return None


def lexicon_flag(phrase: str, vocab: set, names: set) -> bool:
    content = [t for t in phrase.split() if t not in STOP and not t.isdigit() and t not in names]
    return all(t in vocab for t in content)


def emergence_for(usages: list[dict], population: list[str], *, in_lexicon: bool = False,
                  in_world_text: bool = False, in_system_text: bool = False) -> dict:
    """usages: [{"utterance_id", "tick", "speaker", "listeners", "conversation_id", "idx"?}]."""
    us = sorted(usages, key=chron_key)
    if not us:
        return {"emerged": False, "spread": False, "n_adopters": 0, "n_adopters_carried": 0, "n_echo_only": 0,
                "n_independent": 0, "n_exposed": 0, "in_lexicon": in_lexicon, "in_world_text": in_world_text,
                "in_system_text": in_system_text}
    orig = us[0]["speaker"]
    uses: dict[str, list] = {}
    heard: dict[str, list] = {}
    for u in us:
        uses.setdefault(u["speaker"], []).append(u)
        for l in u.get("listeners") or []:
            if l != u["speaker"]:
                heard.setdefault(l, []).append(u)
    others = [a for a in population if a != orig]
    independents, exposed, adopters, carried = [], [], [], []
    for a in others:
        mine, exp = uses.get(a, []), heard.get(a, [])
        if any(not any(precedes(x, y) for x in exp) for y in mine):
            independents.append(a)       # some use of its own that nothing it heard could have caused
            continue
        if not exp:
            continue
        exposed.append(a)
        if mine:
            adopters.append(a)
            if any(_carried(y, exp) for y in mine):
                carried.append(a)
    n_exp, n_unexp = len(exposed), len(others) - len(exposed)
    na, nc, ni = len(adopters), len(carried), len(independents)
    p = fisher_exact(nc, n_exp - nc, ni, n_unexp - ni) if others else 1.0
    spread = nc >= 2 and nc > ni
    first_exp = {a: min(x["tick"] for x in heard[a]) for a in sorted(heard) if a != orig}
    return {"originator": orig, "n_population": len(population), "n_exposed": n_exp, "n_unexposed": n_unexp,
            "n_adopters": na, "n_adopters_carried": nc, "n_echo_only": na - nc, "n_independent": ni,
            "adopter_rate": round(na / n_exp, 3) if n_exp else None,
            "carried_rate": round(nc / n_exp, 3) if n_exp else None,
            "independent_rate": round(ni / n_unexp, 3) if n_unexp else None,
            "fisher_p": round(p, 5), "in_lexicon": in_lexicon, "in_world_text": in_world_text,
            "in_system_text": in_system_text,
            "spread": spread, "emerged": spread and not in_lexicon and not in_world_text and not in_system_text,
            "adopters": sorted(adopters), "carried_adopters": sorted(carried), "independents": sorted(independents),
            "first_exposure_tick": first_exp}


def _carried(y: dict, exposures: list[dict]) -> bool:
    """Use y carries the expression into a new exchange: some exposure in ANOTHER exchange at a strictly
    earlier tick, and no exposure earlier inside y's own exchange (that would make y an echo)."""
    ky = utterance_key(y)
    if any(utterance_key(x) == ky and precedes(x, y) for x in exposures):
        return False
    return any(utterance_key(x) != ky and x["tick"] < y["tick"] for x in exposures)


def analyze_emergence(cands: list[dict], rd) -> dict:
    """candidate id -> emergence record."""
    vocab = vocabulary(rd)
    segs = world_segments(rd)
    normed = [[_norm(t) for t in seg] for seg in segs]
    names = rd.name_tokens
    pop = sorted(rd.agents)
    out = {}
    for c in cands:
        usages = [dict(u, idx=(rd.utt_by_id.get(u["utterance_id"]) or {}).get("idx", 0)) for u in c["usages"]]
        toks = tokens(c["canonical_form"])
        wm = world_match(toks, segs, normed=normed)
        system = bool((c.get("wording") or {}).get("system") or (c.get("features") or {}).get("system_wording"))
        out[c["id"]] = emergence_for(usages, pop, in_lexicon=lexicon_flag(" ".join(toks), vocab["tokens"], names),
                                     in_world_text=wm is not None, in_system_text=system)
        out[c["id"]].update(phrase=c["canonical_form"], world_match=wm)
    return out
