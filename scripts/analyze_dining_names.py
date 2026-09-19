#!/usr/bin/env python3
"""Count dining-name wording in a run's manifest.json and trace.jsonl, using stdlib only.

Usage: python scripts/analyze_dining_names.py runs/RUN --out dining_names.json

This is lexical accounting, not evidence of adoption, confusion, or understanding.
Quoted names count. An utterance without a target alias may still refer to the hall.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
import unicodedata


FFC = re.compile(r"(?<!\w)(?:ffc|fresh\s+food\s+cafe)(?!\w)")
HOPKINS = re.compile(r"(?<!\w)hopkins\s+cafe(?!\w)")
LABELS = ("ffc_only", "hopkins_only", "both", "no_target_alias")


def lexical_counts(text: str) -> dict:
    """Count nonoverlapping occurrences, folding case and Cafe/Café accents."""
    normalized = "".join(
        c for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    )
    ffc = len(FFC.findall(normalized))
    hopkins = len(HOPKINS.findall(normalized))
    label = "both" if ffc and hopkins else "ffc_only" if ffc else "hopkins_only" if hopkins else "no_target_alias"
    return {"ffc": ffc, "hopkins_cafe": hopkins, "label": label}


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

    return {
        "analysis": "dining_name_lexical_mentions",
        "schema_version": 1,
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
        },
        "limitations": [
            "These are lexical mentions, not validated dining-hall references or preferred-name choices.",
            "Quoted or explanatory mentions are included and are not evidence of adoption.",
            "No adoption, confusion, communication repair, understanding, or causal influence is inferred.",
            "Observed naming opportunities are unavailable from these counts; null is not zero.",
            "Agents with no logged target alias may lack an opportunity to mention the hall.",
            "Unique-speaker counts for the two aliases can overlap.",
        ],
        "speakers_missing_from_manifest": sorted(unknown_speakers),
        "overall": overall.report(),
        "by_category": {key: value.report() for key, value in sorted(by_category.items())},
        "by_year": {key: value.report() for key, value in sorted(by_year.items())},
        "by_category_and_year": [
            {"category": category, "year": year, **value.report()}
            for (category, year), value in sorted(by_cohort.items())
        ],
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
