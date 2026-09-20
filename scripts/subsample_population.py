#!/usr/bin/env python3
"""Cut a smaller, socially connected population out of a larger one, deterministically.

No network, LLM calls, or simulation runs. Reads a population file
(``agents`` / ``relationships`` / ``groups`` / ``circles``) plus the matching per-agent seed-memory
files, and writes the same three files restricted to a chosen subset.

Why not simply take the first N agents: the relationship graph is built over the whole source
population, so a prefix keeps only the few edges whose *both* endpoints happen to land inside it
(100 of homewood500 keeps 60 of 1,404 edges) and inherits whatever cohort mix the file order
happens to give. This script instead picks the subset.

Method (deterministic; ``--seed`` only breaks exact ties):

1. **Naming label.** Each agent's label comes from its primary seed memory: the first ``--split``
   label that occurs in the text, else the residual pool ``(other)``. A text carrying two declared
   labels is an error, because a silent misclassification would corrupt the split.
2. **Quotas.** Agents are bucketed by ``(demographics.category, demographics.year, label)``. Each
   declared label's target count is spread over its buckets by largest remainder, in proportion to
   the source bucket sizes; the residual pool gets ``N - sum(targets)``. So the naming split is
   exact and the category/year mix stays proportional to the source.
3. **Snowball.** Growth starts from the densest group cluster (a ``groups:`` entry, ranked by
   internal relationship edges per possible member pair) and then repeatedly admits the candidate
   with the most ties *into the set already chosen*, tie-broken by shared group memberships, then
   source degree, then a seeded per-agent key. Only candidates whose bucket still has quota are
   admissible, so growth cannot drift off the split. Because a greedy snowball is sensitive to
   where it starts, the growth is repeated once per candidate seed group in density order and the
   best-scoring result is kept (``--seed-groups 1`` keeps only the densest).
4. **Split repair.** Swap until every label's count equals its target, preferring a replacement
   from the same category and year so the demographic mix survives. This is a safety net: when a
   label follows the cohort (as in homewood500, where only first-years carry "Hopkins Cafe") step 2
   already pins the split and this pass reports zero swaps.
5. **Tie maximisation.** Best-improvement local search: repeatedly swap one selected agent for an
   unselected agent of the *same bucket* whenever that strictly increases the number of retained
   edges. Quotas and the split are invariant under such a swap.

The subset file keeps only relationships with both endpoints inside it, groups restricted to
selected members (dropped below two members), and the source circles restricted the same way
(dropped below ``backend.simulation.circles.MIN_MEMBERS``, so the result still validates).

Connectivity is bounded by the source graph and is reported, not assumed: a cohort that the source
never ties to anyone outside its own department cannot be connected to the rest by any subset.

    .venv/bin/python scripts/subsample_population.py --n 100 --split "Hopkins Cafe=20" --seed 42 \\
        --out configs/population/homewood100.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.simulation.circles import MIN_MEMBERS as MIN_CIRCLE_MEMBERS  # noqa: E402

OTHER = "(other)"          # residual pool: agents matching no declared --split label
MIN_GROUP_MEMBERS = 2      # backend.agents.profile.load_population drops smaller groups anyway
MIN_SEED_MEMBERS = 3       # a snowball seed must be a cluster, not a single tied pair
MAX_REPAIR_SWAPS = 1000    # loop guard; a repair needs at most one swap per mismatched agent


def parse_split(text: str) -> dict[str, int]:
    """'Hopkins Cafe=20' or 'Hopkins Cafe=20,FFC=80' -> {label: target count}, order preserved."""
    targets: dict[str, int] = {}
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        label, sep, count = part.rpartition("=")
        label = label.strip()
        if not sep or not label:
            raise ValueError(f"--split entry {part!r} must read 'LABEL=COUNT'")
        if label == OTHER:
            raise ValueError(f"--split label {OTHER!r} is reserved for the residual pool")
        if label in targets:
            raise ValueError(f"--split names {label!r} twice")
        try:
            targets[label] = int(count)
        except ValueError:
            raise ValueError(f"--split entry {part!r} needs an integer count") from None
        if targets[label] < 0:
            raise ValueError(f"--split count for {label!r} must not be negative")
    return targets


def label_agents(memories: dict[str, list[str]], ids: list[str], labels: list[str]) -> dict[str, str]:
    """Agent id -> naming label, read from the primary seed memories (never the both-names file)."""
    out = {}
    for aid in ids:
        text = " ".join(memories.get(aid) or [])
        found = [label for label in labels if label in text]
        if len(found) > 1:
            raise ValueError(f"agent {aid!r}: seed memory carries several --split labels {found}; "
                             "classify against the primary seed file, not a both-names control")
        out[aid] = found[0] if found else OTHER
    return out


def largest_remainder(total: int, sizes: dict) -> dict:
    """Split `total` over keys in proportion to `sizes`; remainders go to the largest first."""
    base = sum(sizes.values())
    if base == 0:
        return {key: 0 for key in sizes}
    exact = {key: total * value / base for key, value in sizes.items()}
    out = {key: math.floor(value) for key, value in exact.items()}
    spare = total - sum(out.values())
    for key in sorted(sizes, key=lambda k: (-(exact[k] - out[k]), str(k)))[:spare]:
        out[key] += 1
    return out


def bucket_quotas(buckets: dict, targets: dict[str, int], n: int) -> dict:
    """Per-(category, year, label) quotas that hit every label target exactly and keep the mix."""
    declared = sum(targets.values())
    if declared > n:
        raise ValueError(f"--split asks for {declared} agents but --n is {n}")
    by_label = defaultdict(dict)
    for key, ids in buckets.items():
        by_label[key[2]][key] = len(ids)
    for label in targets:
        if label not in by_label:
            raise ValueError(f"no agent's seed memory contains the --split label {label!r}")
    if declared < n and OTHER not in by_label:
        raise ValueError(f"--split accounts for {declared} of {n} agents, but every source agent carries a "
                         "declared label, so the residual pool is empty; raise the counts or add the "
                         "remaining label(s) to --split")
    quotas = {}
    for label, sizes in by_label.items():
        wanted = targets.get(label, n - declared if label == OTHER else 0)
        if wanted > sum(sizes.values()):
            raise ValueError(f"only {sum(sizes.values())} source agents carry {label!r}, need {wanted}")
        quotas.update(largest_remainder(wanted, sizes))
    if sum(quotas.values()) != n:
        raise ValueError(f"quotas sum to {sum(quotas.values())}, expected {n}")
    return quotas


def build_graph(data: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """-> (relationship adjacency, agent -> group ids). Undirected; an agent may be in many groups."""
    adjacency = defaultdict(set)
    for agent in data["agents"]:
        adjacency[agent["id"]] = set()
    for edge in data.get("relationships") or []:
        adjacency[edge["a"]].add(edge["b"])
        adjacency[edge["b"]].add(edge["a"])
    groups_of = defaultdict(set)
    for gid, gmembers in (data.get("groups") or {}).items():
        for member in gmembers:
            groups_of[member].add(gid)
    return dict(adjacency), dict(groups_of)


def internal_edges(adjacency: dict, ids) -> int:
    inside = set(ids)
    return sum(len(adjacency[a] & inside) for a in inside) // 2


def seed_order(data: dict, adjacency: dict) -> list[str]:
    """Group ids by internal edge density (edges per possible member pair), densest first.

    A pair is not a cluster: any two tied agents score 1.0, so groups below MIN_SEED_MEMBERS are
    ranked last rather than winning the seed slot on a single edge.
    """
    density = {}
    for gid, gmembers in data["groups"].items():
        size = len(set(gmembers))
        pairs = size * (size - 1) / 2
        density[gid] = (size >= MIN_SEED_MEMBERS,
                        internal_edges(adjacency, gmembers) / pairs if pairs else 0.0, size)
    return sorted(density, key=lambda gid: (not density[gid][0], -density[gid][1], -density[gid][2], gid))


def grow(seed_group, members, quotas, adjacency, groups_of, tiebreak, ids):
    """Snowball from one group: admit the candidate with the most ties into the set, quotas permitting.

    Returns (selected ids, number of admissions that had no tie into the set at all -- each of those
    starts a new component, which happens only when a quota can no longer be filled from the fringe).
    """
    bucket_of = {aid: key for key, bucket in members.items() for aid in bucket}
    selected: set[str] = set()
    used: Counter = Counter()
    group_hits: Counter = Counter()
    unconnected = 0

    def admit(aid):
        nonlocal unconnected
        if not adjacency[aid] & selected and selected:
            unconnected += 1
        selected.add(aid)
        used[bucket_of[aid]] += 1
        group_hits.update(groups_of.get(aid, ()))

    def admissible(aid):
        return aid not in selected and used[bucket_of[aid]] < quotas.get(bucket_of[aid], 0)

    total = sum(quotas.values())
    inside = set(seed_group)
    for aid in sorted(seed_group, key=lambda a: (-len(adjacency[a] & inside), tiebreak[a])):
        if len(selected) < total and admissible(aid):
            admit(aid)
    while len(selected) < total:
        candidates = [aid for aid in ids if admissible(aid)]
        if not candidates:
            raise ValueError("ran out of admissible agents; quotas exceed the source population")
        admit(min(candidates, key=lambda a: (-len(adjacency[a] & selected),
                                             -sum(group_hits[g] for g in groups_of.get(a, ())),
                                             -len(adjacency[a]), tiebreak[a])))
    return selected, unconnected


def repair_split(selected: set, labels: dict, targets: dict, demographics: dict,
                 adjacency: dict, tiebreak: dict, ids: list[str]) -> int:
    """Swap until every label count matches its target, preferring same category and year.

    Normally a no-op: `bucket_quotas` already pins the split when a label follows the cohort. It
    matters for source populations whose cohorts mix labels, where the snowball can finish inside
    its cohort quotas and still miss the naming targets.
    """
    def cohort(aid):
        return (demographics[aid].get("category"), demographics[aid].get("year"))

    swaps = 0
    while swaps < MAX_REPAIR_SWAPS:
        counts = Counter(labels[aid] for aid in selected)
        deltas = {label: counts.get(label, 0) - target for label, target in targets.items()}
        over = max((label for label in deltas if deltas[label] > 0), default=None,
                   key=lambda label: (deltas[label], label))
        under = min((label for label in deltas if deltas[label] < 0), default=None,
                    key=lambda label: (deltas[label], label))
        if over is None or under is None:
            return swaps
        drop = [aid for aid in selected if labels[aid] == over]
        take = [aid for aid in ids if aid not in selected and labels[aid] == under]
        if not take:
            raise ValueError(f"cannot reach the split: no unselected agent carries {under!r}")
        pairs = [(out, into) for out in drop for into in take]
        same = [pair for pair in pairs if cohort(pair[0]) == cohort(pair[1])]
        same = same or [pair for pair in pairs if cohort(pair[0])[0] == cohort(pair[1])[0]] or pairs
        out, into = min(same, key=lambda pair: (
            len(adjacency[pair[0]] & (selected - {pair[0]})) - len(adjacency[pair[1]] & (selected - {pair[0]})),
            tiebreak[pair[0]], tiebreak[pair[1]]))
        selected.discard(out)
        selected.add(into)
        swaps += 1
    raise ValueError("split repair did not converge")


def maximise_ties(selected: set, members: dict, adjacency: dict, tiebreak: dict) -> int:
    """Best-improvement swaps inside a bucket; quotas, mix and split are invariant. -> swaps applied."""
    bucket_of = {aid: key for key, bucket in members.items() for aid in bucket}
    swaps = 0
    while True:
        best, best_gain = None, 0
        for out in sorted(selected, key=lambda a: tiebreak[a]):
            rest = selected - {out}
            held = len(adjacency[out] & rest)
            for into in members[bucket_of[out]]:
                if into in selected:
                    continue
                gain = len(adjacency[into] & rest) - held
                if gain > best_gain:
                    best, best_gain = (out, into), gain
        if best is None:
            return swaps
        selected.discard(best[0])
        selected.add(best[1])
        swaps += 1


def components(selected: set, adjacency: dict) -> list[list[str]]:
    """Connected components of the retained relationship graph, largest first."""
    seen: set[str] = set()
    found = []
    for start in sorted(selected):
        if start in seen:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            seen.add(node)
            stack.extend(adjacency[node] & selected - component)
        found.append(sorted(component))
    return sorted(found, key=lambda c: (-len(c), c[0]))


def select(data: dict, memories: dict, n: int, targets: dict[str, int], seed: int,
           seed_groups: int) -> tuple[list[str], dict]:
    """-> (selected ids in source order, provenance of the selection)."""
    ids = [agent["id"] for agent in data["agents"]]
    if len(ids) != len(set(ids)):
        raise ValueError("source population contains duplicate agent ids")
    if n > len(ids):
        raise ValueError(f"--n {n} exceeds the {len(ids)} agents in the source population")
    demographics = {agent["id"]: agent.get("demographics") or {} for agent in data["agents"]}
    missing = [aid for aid in ids if aid not in memories]
    if missing:
        raise ValueError(f"{len(missing)} source agents have no seed memory, e.g. {missing[:3]}")
    labels = label_agents(memories, ids, list(targets))
    members = defaultdict(list)
    for aid in ids:
        members[(demographics[aid].get("category"), demographics[aid].get("year"), labels[aid])].append(aid)
    quotas = bucket_quotas(members, targets, n)
    # Per-label targets including the residual pool, so the repair pass sees the whole picture.
    label_targets: Counter = Counter()
    for key, quota in quotas.items():
        label_targets[key[2]] += quota
    adjacency, groups_of = build_graph(data)
    tiebreak = {aid: hashlib.blake2b(f"{seed}:{aid}".encode(), digest_size=8).hexdigest() for aid in ids}

    candidates = seed_order(data, adjacency)[: seed_groups or None]
    if not candidates:
        raise ValueError("the source population has no groups to snowball from")
    best = None
    for rank, gid in enumerate(candidates):
        grown, unconnected = grow(data["groups"][gid], members, quotas, adjacency, groups_of, tiebreak, ids)
        repairs = repair_split(grown, labels, label_targets, demographics, adjacency, tiebreak, ids)
        swaps = maximise_ties(grown, members, adjacency, tiebreak)
        score = (-internal_edges(adjacency, grown), rank, gid)
        if best is None or score < best[0]:
            best = (score, gid, grown, {"split_repair_swaps": repairs, "tie_maximising_swaps": swaps,
                                        "unconnected_additions": unconnected})
    _, gid, selected, provenance = best
    return [aid for aid in ids if aid in selected], {"seed_group": gid, "labels": labels,
                                                     "quotas": quotas, "adjacency": adjacency, **provenance}


def subset_population(data: dict, selected: list[str]) -> dict:
    """The same file shape, restricted: relationships with both ends inside, groups and circles pruned."""
    inside = set(selected)
    agents = [agent for agent in data["agents"] if agent["id"] in inside]
    relationships = [edge for edge in (data.get("relationships") or [])
                     if edge["a"] in inside and edge["b"] in inside]
    groups = {gid: [m for m in gmembers if m in inside]
              for gid, gmembers in (data.get("groups") or {}).items()}
    groups = {gid: gmembers for gid, gmembers in sorted(groups.items())
              if len(gmembers) >= MIN_GROUP_MEMBERS}
    circles = {cid: [m for m in cmembers if m in inside]
               for cid, cmembers in (data.get("circles") or {}).items()}
    circles = {cid: cmembers for cid, cmembers in sorted(circles.items())
               if len(cmembers) >= MIN_CIRCLE_MEMBERS}
    return {"agents": agents, "relationships": relationships, "groups": groups, "circles": circles}


def summarise(data: dict, subset: dict, selected: list[str], provenance: dict, args) -> dict:
    """Counts only. Nothing here describes conversation, adoption, or any cultural outcome."""
    adjacency = provenance["adjacency"]
    labels = provenance["labels"]
    ids = [agent["id"] for agent in data["agents"]]
    demographics = {agent["id"]: agent.get("demographics") or {} for agent in data["agents"]}
    n = len(selected)
    prefix = set(ids[:n])
    kept = len(subset["relationships"])
    prefix_edges = internal_edges(adjacency, prefix)
    parts = components(set(selected), adjacency)
    source_edges = len(data.get("relationships") or [])
    return {
        "schema_version": 1,
        "selection": "deterministic quota-constrained snowball, split repair, tie-maximising swaps",
        "source_population": str(args.population),
        "source_memories": str(args.memories),
        "seed": args.seed,
        "seed_group": provenance["seed_group"],
        "seed_groups_tried": args.seed_groups or len(data["groups"]),
        "split_repair_swaps": provenance["split_repair_swaps"],
        "tie_maximising_swaps": provenance["tie_maximising_swaps"],
        "unconnected_additions": provenance["unconnected_additions"],
        "source_agents": len(ids),
        "selected_agents": n,
        "naming_split": dict(sorted(Counter(labels[aid] for aid in selected).items())),
        "naming_split_source": dict(sorted(Counter(labels[aid] for aid in ids).items())),
        "categories": dict(sorted(Counter(demographics[aid].get("category") for aid in selected).items())),
        "categories_source_share": {key: round(value / len(ids), 4) for key, value in
                                    sorted(Counter(demographics[aid].get("category") for aid in ids).items())},
        "years": dict(sorted(Counter(demographics[aid].get("year") for aid in selected).items())),
        "relationships": {"source": source_edges, "retained": kept,
                          "retained_fraction": round(kept / source_edges, 4) if source_edges else None,
                          "first_n_prefix_baseline": prefix_edges},
        "mean_degree": {"subset": round(2 * kept / n, 3) if n else None,
                        "source": round(2 * source_edges / len(ids), 3),
                        "first_n_prefix_baseline": round(2 * prefix_edges / n, 3) if n else None},
        "components": {"count": len(parts), "sizes": [len(part) for part in parts],
                       "largest": len(parts[0]) if parts else 0},
        "groups": {"source": len(data.get("groups") or {}), "kept": len(subset["groups"]),
                   "dropped_below_2_members": len(data.get("groups") or {}) - len(subset["groups"])},
        "circles": {"source": len(data.get("circles") or {}), "kept": len(subset["circles"])},
        "limitations": [
            "Counts describe a synthetic population, not measured campus behaviour.",
            "Retained ties and components bound who could meet through the designed structure; "
            "they do not predict contact, conversation, or naming outcomes.",
            "A cohort the source never ties outside its own department cannot be connected by any subset.",
        ],
    }


def write_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=Path, default=Path("configs/population/homewood500.yaml"))
    parser.add_argument("--memories", type=Path,
                        default=Path("configs/population/homewood500_initial_memories.yaml"),
                        help="primary seed memories; the naming label is read from these")
    parser.add_argument("--both-names-memories", default="configs/population/homewood500_both_names_memories.yaml",
                        help="optional recognition-control seeds; pass '' to skip")
    parser.add_argument("--n", type=int, required=True, help="agents to select")
    parser.add_argument("--split", default="", help='exact naming targets, e.g. "Hopkins Cafe=20"')
    parser.add_argument("--seed", type=int, default=42, help="breaks exact ties only")
    parser.add_argument("--seed-groups", type=int, default=0,
                        help="candidate snowball seeds to try, densest first (0 = every group)")
    parser.add_argument("--out", type=Path, required=True, help="population YAML to write")
    parser.add_argument("--out-memories", type=Path, help="default: <out>_initial_memories.yaml")
    parser.add_argument("--out-both-names", type=Path, help="default: <out>_both_names_memories.yaml")
    parser.add_argument("--summary", type=Path, help="also write the printed summary here as JSON")
    args = parser.parse_args(argv)

    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    stem = args.out.with_suffix("")
    args.out_memories = args.out_memories or Path(f"{stem}_initial_memories.yaml")
    args.out_both_names = args.out_both_names or Path(f"{stem}_both_names_memories.yaml")
    outputs = {resolve(args.out), resolve(args.out_memories), resolve(args.out_both_names)}
    if outputs & {resolve(args.population), resolve(args.memories)}:
        parser.error("--out paths must not overwrite an input file")

    try:
        targets = parse_split(args.split)
        data = yaml.safe_load(resolve(args.population).read_text(encoding="utf-8"))
        memories = yaml.safe_load(resolve(args.memories).read_text(encoding="utf-8")) or {}
        selected, provenance = select(data, memories, args.n, targets, args.seed, args.seed_groups)
        subset = subset_population(data, selected)
        write_yaml(resolve(args.out), subset)
        write_yaml(resolve(args.out_memories), {aid: memories[aid] for aid in selected})
        if args.both_names_memories:
            both = yaml.safe_load(resolve(args.both_names_memories).read_text(encoding="utf-8")) or {}
            absent = [aid for aid in selected if aid not in both]
            if absent:
                raise ValueError(f"{len(absent)} selected agents are missing from the both-names file")
            write_yaml(resolve(args.out_both_names), {aid: both[aid] for aid in selected})
        summary = summarise(data, subset, selected, provenance, args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"subsampling failed: {exc}\n")
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.summary:
        resolve(args.summary).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
