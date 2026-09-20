#!/usr/bin/env python3
"""Count dining-name wording in a run's manifest.json and trace.jsonl, using stdlib only.

Usage: python scripts/analyze_dining_names.py runs/RUN --out dining_names.json

This is lexical accounting, not evidence of adoption, confusion, or understanding.
Quoted names count. An utterance without a target alias may still refer to the hall.

Schema 2 keeps every schema-1 key unchanged and adds the seeded-name dynamics the experiment asks for:

- `seeded_cohorts` / `by_seeded_cohort`: who was seeded with which name, read out of the run's own
  tick-0 seed memories rather than from a config file, with each cohort's counts per day;
- `crossovers`: the first time an agent uses the name it was NOT seeded with, together with the earlier
  utterance that could have exposed it (same rule as rundata.precedes: a strictly earlier tick, or an
  earlier turn of the same conversation);
- `dining_references`: how many references to the hall use neither seeded name -- counted with a
  DESCRIPTION detector (dining hall / cafeteria / canteen, and a deictic spoken at a dining location),
  which is the denominator the seeded-name shares are otherwise missing;
- `convergence`: Shannon entropy in bits over the two names, per day and per cohort. 1 bit = the two
  names are used equally often; 0 bits = one of them has won; null = no name was used at all.

The cross-run, lexicon-free version of the same question -- what do agents call a place when nobody tells
the observer the names in advance -- is `backend.analysis.places.resolve_place_references`. This script
stays stdlib-only and alias-anchored on purpose: it is the experiment's own instrument for two names that
were deliberately planted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sys
import unicodedata


FFC = re.compile(r"(?<!\w)(?:ffc|fresh\s+food\s+cafe)(?!\w)")
HOPKINS = re.compile(r"(?<!\w)hopkins\s+cafe(?!\w)")
LABELS = ("ffc_only", "hopkins_only", "both", "no_target_alias")
ALIASES = ("ffc", "hopkins_cafe")
# A reference to the hall that names nothing: a common-noun description, or a deictic said while standing
# in a dining location. Deliberately narrow -- a missed description makes the "neither name" share an
# UNDER-estimate, never an over-estimate.
DESCRIPTION = re.compile(r"(?<!\w)(?:the\s+|a\s+|our\s+)?(?:dining\s+(?:hall|room|commons)|cafeteria|canteen"
                         r"|food\s+(?:hall|court)|meal\s+hall)(?!\w)")
DEICTIC = re.compile(r"(?<!\w)(?:over\s+here|in\s+here|right\s+here|here|this\s+place|this\s+spot)(?!\w)")
HERE_IDIOM = re.compile(r"here(?:'s|\s+is|\s+are|\s+you\s+go|\s+we\s+go)(?!\w)")
DINING_PLACE = re.compile(r"(?<!\w)(?:dining|cafeteria|canteen)(?!\w)")


def fold(text: str) -> str:
    """Case-folded, accent-stripped text, so Cafe and Café are one string."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).casefold())
                   if not unicodedata.combining(c))


def lexical_counts(text: str) -> dict:
    """Count nonoverlapping occurrences, folding case and Cafe/Café accents."""
    normalized = fold(text)
    ffc = len(FFC.findall(normalized))
    hopkins = len(HOPKINS.findall(normalized))
    label = "both" if ffc and hopkins else "ffc_only" if ffc else "hopkins_only" if hopkins else "no_target_alias"
    return {"ffc": ffc, "hopkins_cafe": hopkins, "label": label}


def entropy(counts) -> float | None:
    """Shannon entropy in bits over name uses: 1 bit = the two names are used equally, 0 = one has won,
    None = no name was used, which is not the same as convergence."""
    values = [c for c in counts if c > 0]
    total = sum(values)
    if not total:
        return None
    return round(-sum((c / total) * math.log2(c / total) for c in values), 4) + 0.0


def dining_reference(text: str, location, dining_places: set) -> str | None:
    """How an utterance refers to the hall: by a seeded "alias", by a common-noun "description", by a
    "deictic" said while standing in a dining location, or not at all (None)."""
    normalized = fold(text)
    if FFC.search(normalized) or HOPKINS.search(normalized):
        return "alias"
    if DESCRIPTION.search(normalized):
        return "description"
    if location in dining_places:
        for m in DEICTIC.finditer(normalized):
            if not HERE_IDIOM.match(normalized[m.start():]):
                return "deictic"
    return None


def seeded_names(rows) -> dict:
    """agent -> the seeded name(s) in its own seed memories, read from the run rather than from a config.

    The seeded memory says "I have called the dining hall beside AMR III FFC when arranging meals", so the
    cohort is recoverable from the trace alone; a run with no seed memories yields no cohorts."""
    out: dict[str, set] = {}
    for row in rows:
        if row.get("type") != "memory_encoded" or row.get("source_type") != "seed":
            continue
        agent, normalized = row.get("agent"), fold(row.get("text") or "")
        if not isinstance(agent, str):
            continue
        names = out.setdefault(agent, set())
        if FFC.search(normalized):
            names.add("ffc")
        if HOPKINS.search(normalized):
            names.add("hopkins_cafe")
    return {a: names for a, names in out.items() if names}


def cohort_label(names) -> str:
    if not names:
        return "unseeded"
    return "seeded_both" if len(names) > 1 else f"seeded_{sorted(names)[0]}"


def heard_before(utterance: dict, earlier: list) -> dict | None:
    """The latest earlier utterance carrying the same name that this speaker could have heard.

    "Earlier" is rundata.precedes: a strictly earlier tick, or an earlier turn of the SAME conversation.
    Everything else within one tick is simultaneous and cannot have been heard."""
    agent, tick = utterance["speaker"], utterance["tick"]
    conversation, idx = utterance.get("conversation_id"), utterance.get("idx") or 0
    best = None
    for other in earlier:
        if other["speaker"] == agent or agent not in (other.get("listeners") or []):
            continue
        if other["tick"] < tick or (other["tick"] == tick and conversation
                                    and other.get("conversation_id") == conversation
                                    and (other.get("idx") or 0) < idx):
            if best is None or (other["tick"], other.get("idx") or 0) >= (best["tick"], best.get("idx") or 0):
                best = other
    if best is None:
        return None
    return {"tick": best["tick"], "speaker": best["speaker"], "utterance_id": best.get("id"),
            "text": (best.get("text") or "")[:300]}


@dataclass
class Counts:
    population_ids: set[str] = field(default_factory=set)
    speaker_ids: set[str] = field(default_factory=set)
    alias_speaker_ids: set[str] = field(default_factory=set)
    ffc_speaker_ids: set[str] = field(default_factory=set)
    hopkins_speaker_ids: set[str] = field(default_factory=set)
    utterances: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LABELS, 0))
    mentions: dict[str, int] = field(default_factory=lambda: {"ffc": 0, "hopkins_cafe": 0})

    def add(self, speaker: str, counts: dict) -> None:
        self.speaker_ids.add(speaker)
        self.utterances[counts["label"]] += 1
        for alias in self.mentions:
            self.mentions[alias] += counts[alias]
        if counts["ffc"]:
            self.ffc_speaker_ids.add(speaker)
        if counts["hopkins_cafe"]:
            self.hopkins_speaker_ids.add(speaker)
        if counts["label"] != "no_target_alias":
            self.alias_speaker_ids.add(speaker)

    def report(self) -> dict:
        total = sum(self.utterances.values())
        denominator = total - self.utterances["no_target_alias"]
        return {
            "population_agents": len(self.population_ids),
            "total_utterances": total,
            "utterances_by_alias": dict(self.utterances),
            "lexical_mentions": dict(self.mentions),
            "alias_bearing_utterance_denominator": denominator,
            "fractions_of_alias_bearing_utterances": {
                k: self.utterances[k] / denominator if denominator else None
                for k in LABELS if k != "no_target_alias"
            },
            "unique_speakers": {
                "any_utterance": len(self.speaker_ids),
                "any_target_alias": len(self.alias_speaker_ids),
                "ffc": len(self.ffc_speaker_ids),
                "hopkins_cafe": len(self.hopkins_speaker_ids),
                "both_aliases_across_run": len(self.ffc_speaker_ids & self.hopkins_speaker_ids),
            },
            "population_agents_with_no_logged_utterances": len(self.population_ids - self.speaker_ids),
            "population_agents_with_no_logged_target_alias": len(self.population_ids - self.alias_speaker_ids),
            "observed_naming_opportunities": None,
        }


def cohort(profile: dict) -> tuple[str, str]:
    demographics = profile.get("demographics") or {}
    return tuple(str(demographics.get(key) or "unknown") for key in ("category", "year"))


def analyze(run_dir: str | Path) -> dict:
    """Read only logged utterance rows; do not count exposures or repeated transcripts."""
    run_dir = Path(run_dir)
    with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    agents = manifest.get("agents")
    if not isinstance(agents, dict) or any(not isinstance(p, dict) for p in agents.values()):
        raise ValueError("manifest.json must contain an agents object mapping IDs to profiles")

    overall = Counts(population_ids=set(agents))
    by_category: dict[str, Counts] = {}
    by_year: dict[str, Counts] = {}
    by_cohort: dict[tuple[str, str], Counts] = {}
    agent_cohorts = {aid: cohort(profile) for aid, profile in agents.items()}
    for aid, (category, year) in agent_cohorts.items():
        by_category.setdefault(category, Counts()).population_ids.add(aid)
        by_year.setdefault(year, Counts()).population_ids.add(aid)
        by_cohort.setdefault((category, year), Counts()).population_ids.add(aid)

    unknown_speakers: set[str] = set()
    said: list[dict] = []
    trace: list[dict] = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"trace.jsonl line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"trace.jsonl line {line_number}: expected an object")
            if row.get("type") == "memory_encoded":
                trace.append(row)
            if row.get("type") != "utterance":
                continue
            speaker, text = row.get("speaker"), row.get("text")
            if not isinstance(speaker, str) or not speaker or not isinstance(text, str):
                raise ValueError(f"trace.jsonl line {line_number}: utterance needs speaker and text strings")
            if speaker not in agent_cohorts:
                unknown_speakers.add(speaker)
            category, year = agent_cohorts.get(speaker, ("unknown", "unknown"))
            counts = lexical_counts(text)
            for group in (
                overall,
                by_category.setdefault(category, Counts()),
                by_year.setdefault(year, Counts()),
                by_cohort.setdefault((category, year), Counts()),
            ):
                group.add(speaker, counts)
            said.append({"speaker": speaker, "text": text, "tick": int(row.get("tick") or 0),
                         "id": row.get("id"), "conversation_id": row.get("conversation_id"),
                         "idx": row.get("idx"), "listeners": row.get("listeners") or [],
                         "location": row.get("location"), "counts": counts})

    seeded = seeded_names(trace)
    dynamics = seeded_dynamics(said, seeded, set(agents), manifest)
    return {
        "analysis": "dining_name_lexical_mentions",
        "schema_version": 2,
        "run_directory": str(run_dir),
        "aliases": {"ffc": ["FFC", "Fresh Food Cafe", "Fresh Food Café"],
                    "hopkins_cafe": ["Hopkins Cafe", "Hopkins Café"]},
        "definitions": {
            "unit": "One trace row of type utterance; exposures and conversation transcripts are excluded.",
            "lexical_mentions": "Occurrences of each alias, including repetition and quoted or explanatory mentions.",
            "alias_bearing_utterance_denominator": "Utterances containing either alias, counted once even when both occur.",
            "fractions_of_alias_bearing_utterances": "Mutually exclusive ffc_only, hopkins_only, and both; null when denominator is zero.",
            "no_target_alias": "Neither target alias occurs; this does not mean no reference to the dining hall.",
            "cohorts": "demographics.category and demographics.year from the manifest; missing values are unknown.",
            "seeded_cohorts": "Which name an agent's own tick-0 seed memories gave it; agents with no such "
                              "memory are unseeded.",
            "dining_references": "Utterances that refer to the hall at all: by a seeded name, by a common-noun "
                                 "description, or by a deictic spoken in a dining location.",
            "crossovers": "The first time an agent uses the name it was not seeded with, with the earlier "
                          "utterance it could have heard it in (a strictly earlier tick, or an earlier turn "
                          "of the same conversation).",
            "convergence": "Shannon entropy in bits over the two names: 1 = used equally, 0 = one has won, "
                           "null = neither name was used.",
        },
        "limitations": [
            "These are lexical mentions, not validated dining-hall references or preferred-name choices.",
            "Quoted or explanatory mentions are included and are not evidence of adoption.",
            "No adoption, confusion, communication repair, understanding, or causal influence is inferred.",
            "Observed naming opportunities are unavailable from these counts; null is not zero.",
            "Agents with no logged target alias may lack an opportunity to mention the hall.",
            "Unique-speaker counts for the two aliases can overlap.",
            "A crossover's exposure is an utterance the agent COULD have heard, not a demonstrated cause.",
            "The description detector is narrow, so the share of references using neither name is a lower bound.",
            "Entropy over two planted names says how mixed the wording is, not whether anyone agreed on it.",
        ],
        "speakers_missing_from_manifest": sorted(unknown_speakers),
        "overall": overall.report(),
        "by_category": {key: value.report() for key, value in sorted(by_category.items())},
        "by_year": {key: value.report() for key, value in sorted(by_year.items())},
        "by_category_and_year": [
            {"category": category, "year": year, **value.report()}
            for (category, year), value in sorted(by_cohort.items())
        ],
        **dynamics,
    }


def seeded_dynamics(said: list[dict], seeded: dict, population: set, manifest: dict) -> dict:
    """The seeded-name blocks of schema 2: cohorts, per-day shares, crossovers, reference denominator and
    convergence. `said` is every logged utterance in trace order, `seeded` is agent -> seeded name(s)."""
    ticks_per_day = int(manifest.get("ticks_per_day") or 0)
    day_of = (lambda tick: tick // ticks_per_day + 1) if ticks_per_day > 0 else (lambda tick: 1)
    days = sorted({day_of(u["tick"]) for u in said}) or [1]
    members: dict[str, set] = {}
    for agent in population | set(seeded):
        members.setdefault(cohort_label(seeded.get(agent)), set()).add(agent)
    dining_places = {name for name in (manifest.get("world") or {}).get("graph") or {}
                     if DINING_PLACE.search(fold(name))}

    per_day: dict[tuple, dict] = {}
    references = {"alias": 0, "description": 0, "deictic": 0}
    for u in said:
        label, day = cohort_label(seeded.get(u["speaker"])), day_of(u["tick"])
        kind = dining_reference(u["text"], u.get("location"), dining_places)
        if kind:
            references[kind] += 1
        for key in ((label, day), ("all", day)):
            cell = per_day.setdefault(key, {"utterances": 0, "ffc": 0, "hopkins_cafe": 0,
                                            "alias_bearing": 0, "dining_references": 0, "neither_name": 0})
            cell["utterances"] += 1
            cell["ffc"] += 1 if u["counts"]["ffc"] else 0
            cell["hopkins_cafe"] += 1 if u["counts"]["hopkins_cafe"] else 0
            cell["alias_bearing"] += 1 if u["counts"]["label"] != "no_target_alias" else 0
            cell["dining_references"] += 1 if kind else 0
            cell["neither_name"] += 1 if kind and kind != "alias" else 0

    cached: dict[str, list] = {}

    def series(label: str) -> list:
        if label in cached:
            return cached[label]
        rows = cached.setdefault(label, [])
        for day in days:
            cell = per_day.get((label, day)) or {"utterances": 0, "ffc": 0, "hopkins_cafe": 0,
                                                 "alias_bearing": 0, "dining_references": 0, "neither_name": 0}
            bearing = cell["alias_bearing"]
            rows.append({"day": day, **cell,
                         "share_ffc": round(cell["ffc"] / bearing, 4) if bearing else None,
                         "share_hopkins_cafe": round(cell["hopkins_cafe"] / bearing, 4) if bearing else None,
                         "entropy_bits": entropy([cell["ffc"], cell["hopkins_cafe"]])})
        return rows

    crossings, unseeded_first = [], []
    first_use: dict[tuple, dict] = {}
    for index, u in enumerate(said):
        for alias, pattern in (("ffc", FFC), ("hopkins_cafe", HOPKINS)):
            if not pattern.search(fold(u["text"])) or (u["speaker"], alias) in first_use:
                continue
            first_use[(u["speaker"], alias)] = u
            names = seeded.get(u["speaker"]) or set()
            record = {"agent": u["speaker"], "alias": alias, "seeded_with": sorted(names),
                      "tick": u["tick"], "day": day_of(u["tick"]), "utterance_id": u["id"],
                      "text": u["text"][:300],
                      "exposed_by": heard_before(u, [x for x in said[:index]
                                                     if pattern.search(fold(x["text"]))])}
            if names and alias not in names:
                crossings.append(record)              # a seeded agent reaching for the other name
            elif not names:
                unseeded_first.append(record)         # nobody seeded this agent; not a crossover
    total_references = sum(references.values())
    return {
        "days": days,
        "seeded_cohorts": {
            "source": "tick-0 seed memories in trace.jsonl",
            "sizes": {label: len(agents) for label, agents in sorted(members.items())},
            "agents_with_a_seeded_name": len(seeded),
        },
        "by_seeded_cohort": {label: {"population_agents": len(members.get(label, ())), "by_day": series(label),
                                     "entropy_bits": entropy([sum(r["ffc"] for r in series(label)),
                                                              sum(r["hopkins_cafe"] for r in series(label))])}
                             for label in sorted(members)},
        "by_day": series("all"),
        "crossovers": crossings,
        "first_uses_by_unseeded_agents": unseeded_first,
        "dining_references": {
            "total": total_references, "with_seeded_name": references["alias"],
            "description_only": references["description"],
            "deictic_in_a_dining_location_only": references["deictic"],
            "dining_locations": sorted(dining_places),
            "share_using_neither_seeded_name":
                round((total_references - references["alias"]) / total_references, 4) if total_references else None,
        },
        "convergence": {
            "entropy_bits_by_day": [row["entropy_bits"] for row in series("all")],
            "entropy_bits_overall": entropy([sum(row["ffc"] for row in series("all")),
                                             sum(row["hopkins_cafe"] for row in series("all"))]),
            "by_cohort_by_day": {label: [row["entropy_bits"] for row in series(label)] for label in sorted(members)},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing manifest.json and trace.jsonl")
    parser.add_argument("--out", type=Path, help="Write JSON here; otherwise print it to stdout")
    args = parser.parse_args(argv)
    if args.out and args.out.resolve() in {
        (args.run_dir / "manifest.json").resolve(), (args.run_dir / "trace.jsonl").resolve()
    }:
        parser.error("--out must not overwrite an input file")
    try:
        result = json.dumps(analyze(args.run_dir), indent=2, ensure_ascii=False) + "\n"
        if args.out:
            args.out.write_text(result, encoding="utf-8")
        else:
            sys.stdout.write(result)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"dining-name analysis failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
