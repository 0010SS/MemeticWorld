"""The injected-meme registry, observer side (memo §2.1, §7 RQ1). OBSERVER ONLY.

One config block declares the whole cohort; nothing about a meme is hardcoded anywhere in the analysis
layer, so adding, removing or swapping a meme is a config edit:

    memes:
      enabled: false
      registry:
        - id: tray_washer
          phrase: "blue-tray"
          grounding: grounded | ungrounded      # a witnessed origin event, or seeded only
          breadth: broad | narrow               # applies to a class of situations, or to one thing
          habit: "..."                          # the seeded minority's habit line (the ONLY carrier, R1)
          seeds: {k: 5, strategy: spread}
          incident: {location, arena, days: [...], repaired_day: 4, prob_per_meal: 0.6}   # grounded only
          probe_gradient: {literal, near, mid, far}    # OBSERVER ONLY (R7): never reaches an agent's world
          foils: ["...", "..."]                        # surface features shared, structure not
      battery: {enabled: false, checkpoints: [...], sample: {...}}   # see graded.py

The 2x2 is (grounding x breadth); `cell()` is the factor cell an entry belongs to. The registry is plain
YAML, so simulation-side code reads `cfg["memes"]` directly rather than importing this module (nothing
under backend/simulation, backend/agents or backend/memory may import backend.analysis.battery).

Usage matching is deliberately MORPHOLOGICAL, not semantic: "blue-tray", "blue tray", "blue trays" are one
expression whose `variant` is recorded, so formal variation (memo §2.1) stays measurable, while paraphrase
is left to the candidate miner rather than smuggled in here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LEVELS = ("literal", "near", "mid", "far")
GROUNDING = ("grounded", "ungrounded")
BREADTH = ("broad", "narrow")
# a leading determiner is optional: "the Tuesday list" is the same expression as "Tuesday list"
_DET = r"(?:the|a|an|that|this)"
_TOK = re.compile(r"[A-Za-z0-9']+")
_INFLECT = r"(?:'s|s|es|ed|ing)?"
# distinctive words are the words R2 forbids the world from emitting; function words carry no coinage risk
STOPWORDS = frozenset("the a an and or but of to in on at for with by from as is are was were be it its "
                      "that this these those my your his her their our not no so if then than too very".split())


@dataclass(frozen=True)
class MemeSpec:
    id: str
    phrase: str
    grounding: str
    breadth: str
    habit: str = ""
    seeds: dict = field(default_factory=dict)
    incident: dict | None = None
    probe_gradient: dict = field(default_factory=dict)
    foils: tuple = ()
    raw: dict = field(default_factory=dict)

    @property
    def cell(self) -> str:
        """The 2x2 cell: the unit of the comparative claim."""
        return f"{self.grounding}_{self.breadth}"

    @property
    def repaired_day(self) -> int | None:
        """The day the meme's basis is removed. None for ungrounded memes: there is nothing to repair,
        which is exactly what makes them the comparison."""
        return None if not self.incident else _int(self.incident.get("repaired_day"))

    @property
    def incident_days(self) -> list[int]:
        return sorted({_int(d) for d in ((self.incident or {}).get("days") or []) if _int(d)})

    @property
    def location(self) -> str | None:
        return (self.incident or {}).get("location")

    @property
    def seed_k(self) -> int:
        """The declared minority size. An explicit `agents` list is the size, whether or not k is given:
        R6 is about how many agents actually carry the phrase, not about what k says."""
        agents = (self.seeds or {}).get("agents") or []
        return len(agents) if agents else (_int((self.seeds or {}).get("k")) or 0)

    @property
    def seed_ids(self) -> tuple:
        return tuple(str(a) for a in ((self.seeds or {}).get("agents") or []))

    @property
    def distinctive_words(self) -> tuple:
        """The content words the world must never emit (R2). A compositional phrase can be coined
        independently the moment the world supplies its parts, and transmission then becomes
        indistinguishable from rediscovery.

        The entry's own `distinctive_words` list is honoured and EXTENDED, never replaced: a near-synonym
        the config author thought of belongs on the list, and so does every word of the phrase itself, which
        is not something an author should have to remember to repeat."""
        words = {w.lower() for w in _TOK.findall(self.phrase)}
        words |= {str(w).lower() for w in (self.raw.get("distinctive_words") or [])}
        return tuple(sorted(w for w in words if w and w not in STOPWORDS))

    @property
    def pattern(self) -> re.Pattern:
        return usage_pattern(self.phrase)

    def gradient(self, level: str) -> str | None:
        return (self.probe_gradient or {}).get(level)


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def usage_pattern(phrase: str) -> re.Pattern:
    """Word-boundary regex for a phrase and its morphological variants: hyphen/space/run-together between
    tokens, an optional leading determiner, an optional inflection on the last token."""
    toks = [t.lower() for t in _TOK.findall(phrase)]
    if not toks:
        return re.compile(r"(?!x)x")            # matches nothing
    lead = ""
    if toks[0] in ("the", "a", "an") and len(toks) > 1:
        lead, toks = rf"(?:{_DET}\s+)?", toks[1:]
    body = r"[\s\-]*".join(re.escape(t) for t in toks)
    return re.compile(rf"\b{lead}{body}{_INFLECT}\b", re.I)


def parse_entry(raw: dict) -> MemeSpec:
    d = dict(raw or {})
    return MemeSpec(id=str(d.get("id") or "").strip(), phrase=str(d.get("phrase") or "").strip(),
                    grounding=str(d.get("grounding") or "").strip().lower(),
                    breadth=str(d.get("breadth") or "").strip().lower(),
                    habit=str(d.get("habit") or ""), seeds=dict(d.get("seeds") or {}),
                    incident=(dict(d["incident"]) if d.get("incident") else None),
                    probe_gradient=dict(d.get("probe_gradient") or {}),
                    foils=tuple(str(f) for f in (d.get("foils") or [])), raw=d)


def memes_block(cfg_or_run) -> dict:
    """The `memes:` block of a config dict, a run directory or a RunData. Absent block -> disabled (R5)."""
    cfg = cfg_or_run
    if isinstance(cfg_or_run, (str, Path)):
        import yaml
        p = Path(cfg_or_run)
        p = p if p.is_file() else p / "config.resolved.yaml"
        cfg = yaml.safe_load(open(p)) if p.exists() else {}
    elif hasattr(cfg_or_run, "cfg"):
        cfg = cfg_or_run.cfg
    return dict((cfg or {}).get("memes") or {})


def load_registry(cfg_or_run, *, only_enabled: bool = True) -> list[MemeSpec]:
    """Every registry entry, in config order. `only_enabled=False` reads the entries even when the
    mechanism is off, which is what the preflight checks and the cost projection want."""
    blk = memes_block(cfg_or_run)
    if only_enabled and not blk.get("enabled", False):
        return []
    return [parse_entry(e) for e in (blk.get("registry") or [])]


def battery_cfg(cfg_or_run) -> dict:
    """memes.battery with its defaults. Everything defaults off / small: a battery is expensive (6d)."""
    b = dict(memes_block(cfg_or_run).get("battery") or {})
    s = dict(b.get("sample") or {})
    return {"enabled": bool(b.get("enabled", False)),
            "checkpoints": list(b.get("checkpoints") or []),
            "sample": {"per_cohort": _int(s.get("per_cohort")) or 8, "max_agents": _int(s.get("max_agents")) or 24},
            "foils_per_meme": _int(b.get("foils_per_meme")) or 2,
            "explain": bool(b.get("explain", True)),
            "judge_explanations": bool(b.get("judge_explanations", True)),
            "max_judge_calls": _int(b.get("max_judge_calls")) or 600,
            "k": _int(b.get("k")) or 6, "workers": _int(b.get("workers")) or 8}


def by_id(specs: list[MemeSpec]) -> dict:
    return {s.id: s for s in specs}


def cells(specs: list[MemeSpec]) -> dict:
    out: dict[str, list[str]] = {}
    for s in specs:
        out.setdefault(s.cell, []).append(s.id)
    return {k: sorted(v) for k, v in sorted(out.items())}


# ------------------------------------------------------------------------------------------ validation
def validate(specs: list[MemeSpec]) -> list[str]:
    """Design problems that invalidate the run, as messages. R6 (matched on everything but the factors)
    is checked here because a cohort whose minorities differ in size is not a 2x2 at all."""
    errs = []
    seen = set()
    for s in specs:
        if not s.id or not s.phrase:
            errs.append(f"entry {s.raw!r}: id and phrase are required")
            continue
        if s.id in seen:
            errs.append(f"{s.id}: duplicate id")
        seen.add(s.id)
        if s.grounding not in GROUNDING:
            errs.append(f"{s.id}: grounding {s.grounding!r} not in {GROUNDING}")
        if s.breadth not in BREADTH:
            errs.append(f"{s.id}: breadth {s.breadth!r} not in {BREADTH}")
        if s.grounding == "grounded" and not s.incident:
            errs.append(f"{s.id}: grounded but has no incident")
        if s.grounding == "ungrounded" and s.incident:
            errs.append(f"{s.id}: ungrounded but declares an incident")
        missing = [l for l in LEVELS if not s.gradient(l)]
        if missing:
            errs.append(f"{s.id}: probe_gradient missing {missing}")
        if not s.foils:
            errs.append(f"{s.id}: no foils - an applies-rate without a foil rate cannot distinguish "
                        "broadening from an agreeable model")
        if s.habit and not s.pattern.search(s.habit):
            errs.append(f"{s.id}: the habit line does not contain the phrase, so nothing carries it")
        for lvl in LEVELS:
            if s.gradient(lvl) and s.pattern.search(s.gradient(lvl)):
                errs.append(f"{s.id}: probe_gradient.{lvl} contains the phrase itself (the probe would "
                            "answer its own question)")
        for i, f in enumerate(s.foils):
            if s.pattern.search(f):
                errs.append(f"{s.id}: foil {i} contains the phrase itself")
    ks = {s.seed_k for s in specs if s.seed_k}
    if len(ks) > 1:
        errs.append(f"minority sizes differ across memes ({sorted(ks)}): spread differences would be "
                    "attributable to k rather than to grounding or breadth (R6)")
    # an agent seeded with two phrases carries both into the same scarce conversational airtime, so the
    # two memes compete inside one speaker rather than across the population
    overlap: dict[str, list[str]] = {}
    for s in specs:
        for a in s.seed_ids:
            overlap.setdefault(a, []).append(s.id)
    doubled = {a: ms for a, ms in sorted(overlap.items()) if len(ms) > 1}
    if doubled:
        errs.append(f"agents seeded with more than one meme {doubled}: their memes compete for the same "
                    "speaker's airtime, which confounds the cohort comparison")
    return errs


def world_contamination(specs: list[MemeSpec], world_text: str) -> dict:
    """R1/R2 machine check: the phrase itself, and each of its distinctive words, must never occur in
    world-produced surface text. Returns {meme_id: {"phrase": n, "words": {word: n}}}; all-zero is clean."""
    low = (world_text or "").lower()
    out = {}
    for s in specs:
        # inflection-tolerant: the world saying "trays" hands an agent the word just as surely as "tray"
        words = {w: len(re.findall(rf"\b{re.escape(w)}{_INFLECT}\b", low)) for w in s.distinctive_words}
        out[s.id] = {"phrase": len(s.pattern.findall(low)), "words": {w: n for w, n in words.items() if n}}
    return out


# --------------------------------------------------------------------------------------------- usages
def find_usages(rd, spec: MemeSpec, tick: int | None = None) -> list[dict]:
    """In-simulation uses of a registry meme, shaped exactly like a candidate's usages so that
    trends.meme_series, transmission.analyze_transmission and lineage.variant_tree all take them
    unchanged. `variant` is the surface form actually said (formal variation)."""
    pat = spec.pattern
    out = []
    for u in rd.utterances:
        if tick is not None and int(u.get("tick") or 0) > tick:
            continue
        m = pat.search(u.get("text") or "")
        if not m:
            continue
        out.append({"utterance_id": u.get("id") or u.get("utterance_id"), "tick": int(u["tick"]),
                    "idx": int(u.get("idx") or 0), "speaker": u["speaker"],
                    "listeners": [l for l in (u.get("listeners") or []) if l != u["speaker"]],
                    "conversation_id": u.get("conversation_id"), "text": u.get("text") or "",
                    "context": rd.conversation_context(u) if u.get("conversation_id") else (u.get("text") or ""),
                    "variant": m.group(0).strip().lower(), "location": u.get("location"), "arena": u.get("arena")})
    return out


def seed_agents(rd, spec: MemeSpec) -> list[str]:
    """The committed minority as the run actually seeded it, in order of authority: the manifest's record
    (the engine writes what it applied), a `meme_seed` trace record, the habit line in the manifest's
    profiles, and only last the config's own `seeds.agents`. What the engine did beats what the config
    asked for, so the observer and the engine cannot disagree about who was seeded."""
    man = getattr(rd, "manifest", None) or {}
    for rec in _manifest_meme_records(man, spec.id):
        ids = [str(a) for a in (rec.get("seeds") or rec.get("agents") or [])]
        if ids:
            return sorted(set(ids))
    ids = [str(r["agent"]) for r in rd.of("meme_seed") if str(r.get("meme") or r.get("meme_id")) == spec.id]
    if ids:
        return sorted(set(ids))
    pat = spec.pattern
    ids = sorted({aid for aid, a in (man.get("agents") or {}).items()
                  if any(pat.search(h) for h in (a.get("habits") or []))})
    return ids or sorted({str(a) for a in ((spec.seeds or {}).get("agents") or [])})


def _manifest_meme_records(man: dict, meme_id: str) -> list[dict]:
    """Every place a manifest may record what was seeded, most authoritative first. The engine's own
    layout has moved once already (profile.apply_planted's `memes` list, the incident world's
    `manipulation.memes`), so the observer reads all of them rather than one."""
    out = []
    top = (man.get("memes") or {})
    if isinstance(top, dict) and isinstance(top.get(meme_id), dict):
        out.append(top[meme_id])
    for holder in (man.get("planted"), man.get("manipulation")):
        lst = (holder or {}).get("memes") if isinstance(holder, dict) else None
        if isinstance(lst, list):
            out += [m for m in lst if isinstance(m, dict) and str(m.get("id")) == meme_id]
        elif isinstance(lst, dict):
            inner = lst.get("memes") if isinstance(lst.get("memes"), dict) else lst
            if isinstance(inner, dict) and isinstance(inner.get(meme_id), dict):
                out.append(inner[meme_id])
    return out


def seed_graph_stats(rd, spec: MemeSpec, seeds: list[str] | None = None) -> dict:
    """The seeds' position in the social graph, per meme, so R6 can be reported rather than asserted:
    if one minority is better connected, its spread says nothing about grounding or breadth."""
    seeds = seeds if seeds is not None else seed_agents(rd, spec)
    rel = {aid: len((a.get("relationships") or {})) for aid, a in ((rd.manifest.get("agents") or {})).items()}
    heard: dict[str, set] = {}
    for u in rd.utterances:                      # realised degree: distinct interlocutors over the run
        spk, ls = u["speaker"], [l for l in (u.get("listeners") or []) if l != u["speaker"]]
        heard.setdefault(spk, set()).update(ls)
        for l in ls:
            heard.setdefault(l, set()).add(spk)
    deg = [len(heard.get(a, ())) for a in seeds]
    prof = [rel.get(a, 0) for a in seeds]
    return {"meme": spec.id, "cell": spec.cell, "k": len(seeds), "seeds": seeds,
            "profile_degree": {"values": sorted(prof), "mean": _mean(prof)},
            "realised_degree": {"values": sorted(deg), "mean": _mean(deg)}}


def _mean(xs) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None
