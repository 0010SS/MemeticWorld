"""Agent profile ontology (close to Generative Agents' persona description).

Deliberately contains NO cultural state: no memes, meanings, event labels, etc.
`tests/test_invariants.py` enforces that.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, median, pstdev

import yaml

from backend.ga_compat import REPO_ROOT

FORBIDDEN_FIELDS = {"known_memes", "meme_dictionary", "meme_meanings", "meme_preference",
                    "meme_fitness", "current_meme", "current_culture",
                    "latent_event_understanding", "latent_event_id", "latent_type"}

PERSONAL_LABELS = {
    "goal": "Personal goal",
    "values": "Values",
    "strengths": "Strengths",
    "blind_spots": "Blind spots",
    "stress_response": "Response to stress",
    "coping_strategy": "Coping strategy",
    "social_energy": "Social energy",
    "trust_style": "How trust develops",
    "conflict_style": "Approach to conflict",
    "humor_style": "Sense of humor",
    "pet_peeves": "Pet peeves",
    "small_joys": "Small joys",
}
PERSONAL_LIST_FIELDS = {"values", "strengths", "blind_spots", "pet_peeves", "small_joys"}


@dataclass
class Relationship:
    relation_type: str = "stranger"
    familiarity: float = 0.05
    affinity: float = 0.5


@dataclass
class RoutineEntry:
    time: str
    location: str
    activity: str
    arena: str | None = None


@dataclass
class AgentProfile:
    id: str
    name: str
    demographics: dict
    background: str
    personality: dict
    interests: dict
    habits: list[str]
    routine: list[RoutineEntry]
    home: dict
    sprite: str = "Abigail_Chen"
    relationships: dict[str, Relationship] = field(default_factory=dict)
    known_locations: list[str] = field(default_factory=list)
    personal: dict = field(default_factory=dict)
    friend_groups: list[dict] = field(default_factory=list)
    # v3 roster (ontology v3 §3): the co-op role (am_crew | pm_crew | stores; None = not a member, or a reserve
    # before arrival: the roster sets it on arrival) and whether the persona is a reserve newcomer.
    role: str | None = None
    reserve: bool = False

    @property
    def coop_role(self) -> str | None:
        return self.role

    @property
    def first_name(self) -> str:
        return self.name.split()[0]

    def rel(self, other_id: str) -> Relationship:
        return self.relationships.get(other_id, Relationship())

    # --- GA "identity stable set" fields -------------------------------------
    def ga_innate(self) -> str:
        return ", ".join(self.personality.get("traits", []))

    def ga_learned(self) -> str:
        d = self.demographics
        it = self.interests
        if d.get("category") in {"faculty", "staff"}:
            role = d.get("role") or f"{d['category']} member"
            department = f" in {d['department']}" if d.get("department") else ""
            parts = [f"{self.first_name} is a {d.get('age')}-year-old {role}{department}. {self.background}"]
        else:
            parts = [f"{self.first_name} is a {d.get('age')}-year-old {d.get('year')} studying {d.get('major')}"
                     f" ({d.get('role')}). {self.background}"]
        if it.get("topics"):
            parts.append(f"{self.first_name} is interested in {', '.join(it['topics'])}.")
        if it.get("hobbies"):
            parts.append(f"Hobbies: {', '.join(it['hobbies'])}.")
        if it.get("clubs"):
            parts.append(f"Member of: {', '.join(it['clubs'])}.")
        style = self.personality.get("communication_style")
        if style:
            parts.append(f"Communication style: {style}.")
        for key, label in PERSONAL_LABELS.items():
            value = self.personal.get(key)
            if value:
                value = ", ".join(value) if isinstance(value, list) else value
                parts.append(f"{label}: {value.rstrip('.')}.")
        for group in self.friend_groups:
            parts.append(f"{self.first_name}'s {group['name']}: {', '.join(group['members'])}.")
        return " ".join(parts)

    def ga_lifestyle(self) -> str:
        return f"{self.first_name} " + "; ".join(self.habits) + "."

    def ga_daily_plan_req(self, cfg: dict | None = None) -> str:
        """GA `daily_plan_req`, part of every ISS. In situated mode the plan says where the agent goes the way
        the agent would think of it ("the dining hall beside the freshman residences"), not by the world's
        canonical label: the ISS reaches almost every prompt, so a name here is a name everywhere."""
        from backend.simulation import reference
        return ", ".join(f"{r.activity} at {reference.phrase(r.location, agent=self, cfg=cfg)} around {r.time}"
                         for r in self.routine[:6])

    def to_public_dict(self) -> dict:
        d = asdict(self)
        d["relationships"] = {k: asdict(v) for k, v in self.relationships.items()}
        return d


# ------------------------------------------------------------------------------------------------
# Injected expressions: the positive control, and the meme registry (the 2x2 cohort).
#
# Two roles share one mechanism (a verbal habit appended to a profile) and must never share an
# interpretation:
#   - CONTROL (`controls.planted_phrase`): one agent, one phrase, there to prove the detection
#     pipeline can see a convention when one exists. It is excluded from convention counts.
#   - STUDY (`memes.registry`): a cohort of memes, each seeded into its own committed minority of k
#     agents (Centola's critical mass; Ashery et al., Sci. Adv. 2025). These are the DEPENDENT
#     VARIABLE: they are tracked, never excluded.
# Nothing here knows a phrase. Adding, removing or swapping a meme is a config edit.

MEME_GROUNDING = ("grounded", "ungrounded")
MEME_BREADTH = ("broad", "narrow")
MEME_STRATEGIES = ("spread", "random", "cluster", "high_degree")
MEME_DEFAULT_STRATEGY = "spread"
#: registry keys that belong to the OBSERVER (R7: they must never reach an agent). Read here only to
#: refuse a habit line that has swallowed one of them.
MEME_OBSERVER_KEYS = ("probe_gradient", "foils")
_QUOTED_PHRASE = re.compile(r"[\"“]([^\"”]+)[\"”]")
#: |SMD| between any two minorities on a matching covariate. 0.1 is the conventional "well matched"
#: line in the matching literature, 0.25 the outer limit of "acceptable" (Stuart 2010).
MEME_SMD_OK = 0.25
#: how many agents of a degree stratum the `spread` allocation may choose between per meme. 1 would be
#: pure random sampling; measured over 30 seeds on the 100-agent population, 1 leaves every seed
#: unmatched on some covariate, 2 leaves 4 of 30 and 3 leaves 1 of 30. Above 3 the seeds are barely
#: sampled any more, they are optimised, which buys little and costs the protection randomisation gives
#: against covariates nobody thought to measure.
MEME_CANDIDATES_PER_SEED = 3
#: random allocations the balance check compares itself against (R6: "how well matched" needs a scale)
MEME_BENCHMARK_DRAWS = 200


def _norm_habit(habit: str, prof: AgentProfile) -> str:
    """The habit as it is appended to a profile. Written as a full sentence ("Maya has a habit of ...");
    a leading first name and the final period are dropped so it reads naturally inside `ga_lifestyle()`
    ("Maya grabs coffee ...; has a habit of ..."). `{name}` is filled in with the agent's first name, so
    one registry line can seed k different people."""
    h = " ".join(str(habit or "").split()).replace("{name}", prof.first_name)
    h = h.rstrip(".").strip()
    return h[len(prof.first_name) + 1:] if h.startswith(prof.first_name + " ") else h


def meme_registry(cfg: dict) -> list[dict]:
    """Validated `memes.registry` entries in config order when `memes.enabled`; [] otherwise.

    Each entry is normalised to {"id", "phrase", "grounding", "breadth", "habit", "k", "strategy",
    "agents" (an explicit seed list, or None), "site" (the incident's location, None when ungrounded)}.
    The incident itself is staged elsewhere; `site` is read only to report who is exposed to it.

    Validation is deliberately loud. A habit that does not quote its own phrase would make every
    downstream matcher disagree about what the phrase is; a habit carrying the observer's probe wording
    would hand the agents the gloss the experiment is supposed to measure (R7); a habit carrying another
    meme's phrase would cross-contaminate two cells of the 2x2.
    """
    block = cfg.get("memes") or {}
    if not isinstance(block, dict):
        raise ValueError("memes must be a mapping")
    if not block.get("enabled"):
        return []
    raw = block.get("registry")
    if not isinstance(raw, list) or not raw:
        raise ValueError("memes.enabled is true but memes.registry is empty")
    out: list[dict] = []
    for e in raw:
        if not isinstance(e, dict):
            raise ValueError(f"memes.registry: each entry must be a mapping, got {e!r}")
        mid = str(e.get("id") or "").strip()
        if not mid or any(m["id"] == mid for m in out):
            raise ValueError(f"memes.registry: each entry needs a unique id, got {mid!r}")
        phrase = " ".join(str(e.get("phrase") or "").split())
        habit = " ".join(str(e.get("habit") or "").split())
        if not phrase or not habit:
            raise ValueError(f"meme {mid!r}: phrase and habit are both required")
        if phrase.lower() not in [q.strip().lower() for q in _QUOTED_PHRASE.findall(habit)]:
            raise ValueError(f"meme {mid!r}: the habit must quote the phrase verbatim ({phrase!r} not quoted in habit)")
        grounding, breadth = str(e.get("grounding") or ""), str(e.get("breadth") or "")
        if grounding not in MEME_GROUNDING or breadth not in MEME_BREADTH:
            raise ValueError(f"meme {mid!r}: grounding must be one of {MEME_GROUNDING} and breadth one of {MEME_BREADTH}")
        low = habit.lower()
        for key in MEME_OBSERVER_KEYS:                     # R7: the observer's gloss never reaches an agent
            for text in _strings(e.get(key)):
                if len(text) > 12 and text.lower() in low:
                    raise ValueError(f"meme {mid!r}: the habit line carries {key} wording ({text!r}); "
                                     "probe gradients and foils are observer-only")
        seeds = e.get("seeds") or {}
        if not isinstance(seeds, dict):
            raise ValueError(f"meme {mid!r}: seeds must be a mapping")
        agents = seeds.get("agents")
        if agents is not None and (not isinstance(agents, list) or not all(isinstance(a, str) for a in agents)):
            raise ValueError(f"meme {mid!r}: seeds.agents must be a list of agent ids")
        k = int(seeds.get("k", len(agents) if agents else 0) or 0)
        if k < 1:
            raise ValueError(f"meme {mid!r}: seeds.k must be >= 1 (the committed minority)")
        if agents and len(agents) != k:
            raise ValueError(f"meme {mid!r}: seeds.agents has {len(agents)} ids but seeds.k is {k}")
        strategy = str(seeds.get("strategy") or MEME_DEFAULT_STRATEGY)
        if strategy not in MEME_STRATEGIES:
            raise ValueError(f"meme {mid!r}: seeds.strategy must be one of {MEME_STRATEGIES}")
        incident = e.get("incident") or {}
        out.append({"id": mid, "phrase": phrase, "grounding": grounding, "breadth": breadth, "habit": habit,
                    "k": k, "strategy": strategy, "agents": list(agents) if agents else None,
                    "site": (incident.get("location") if isinstance(incident, dict) else None)})
    for a in out:                                          # one meme's phrase inside another's habit is contamination
        for b in out:
            if a is not b and a["phrase"].lower() in b["habit"].lower():
                raise ValueError(f"meme {b['id']!r}: its habit line contains meme {a['id']!r}'s phrase")
    return out


def _strings(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


# ---------------------------------------------------------------- matching covariates

def tie_degrees(profiles: dict[str, AgentProfile]) -> dict[str, tuple[int, float]]:
    """agent -> (degree, familiarity-weighted degree) over real ties. `load_population` fills every
    unnamed pair with a default `stranger` Relationship, so a tie is any relationship that is not one."""
    out = {}
    for aid, p in profiles.items():
        ties = [r for r in p.relationships.values() if r.relation_type != "stranger"]
        out[aid] = (len(ties), round(sum(r.familiarity for r in ties), 6))
    return out


def _reach(prof: AgentProfile) -> int:
    """How much of campus the agent's routine touches: a proxy for how many different people it can say
    anything in front of, and therefore for a seed's chance of getting its phrase heard."""
    return len({r.location for r in prof.routine})


def _components(profiles: dict[str, AgentProfile]) -> dict[str, str]:
    """Connected components of the tie graph, named by their lowest member id."""
    adj = {a: {b for b, r in p.relationships.items() if r.relation_type != "stranger" and b in profiles}
           for a, p in profiles.items()}
    out, seen = {}, set()
    for a in sorted(profiles):
        if a in seen:
            continue
        comp, stack = [], [a]
        seen.add(a)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in sorted(adj[x] - seen):
                seen.add(y)
                stack.append(y)
        for x in comp:
            out[x] = f"component:{min(comp)}"
    return out


def seed_clusters(profiles: dict[str, AgentProfile], cfg: dict) -> tuple[dict[str, str], str]:
    """(agent -> cluster id, where the clusters came from). The population file's friend circles when it
    has them (the same partition the simulator concentrates recurrence in), else tie-graph components."""
    try:
        from backend.simulation.circles import load_circles
        circles = load_circles(cfg["population"], list(profiles))
        if circles:
            return {a: c for c, ms in circles.items() for a in ms if a in profiles}, "circles"
    except Exception:                          # noqa: BLE001 - no circles section, or the file moved
        pass
    return _components(profiles), "components"


# ---------------------------------------------------------------- committed-minority selection

def select_seeds(profiles: dict[str, AgentProfile], entries: list[dict], cfg: dict,
                 exclude: set[str] | None = None) -> dict[str, list[str]]:
    """meme id -> its committed minority, deterministic from `world_seed` (R4).

    Seeds are DISJOINT across memes unless `memes.allow_overlap` is set: an agent carrying two phrases
    would be a confound that the 2x2 has no way to model.

    Strategies, per meme (`seeds.strategy`):
      spread       - the matched default (R6). The eligible agents, ranked by familiarity-weighted tie
                     degree, are cut into K contiguous strata (K = the largest minority) and every
                     participating meme draws exactly one agent from each stratum, so the minorities hold
                     one agent per degree band BY CONSTRUCTION rather than by luck. Within a stratum the
                     drawn agents are ordered by routine reach and dealt out by a cyclic rotation (a
                     Latin square), which balances that second covariate across memes as well; which meme
                     sits in which rotation slot is randomised once from the run seed.
      high_degree  - the k best-connected remaining agents (a deliberately advantaged minority).
      cluster      - k agents inside one friend circle, grown outward from a random member.
      random       - k agents drawn uniformly.
    `seeds.agents` pins a minority explicitly and skips selection altogether.

    Explicit lists are honoured first, then the joint `spread` allocation, then the other strategies in
    registry order: a strategy that deliberately takes a particular part of the graph must not be able
    to pull the matched cohort off its strata.
    """
    from backend.simulation.rngs import seed_rng, world_seed
    allow_overlap = bool((cfg.get("memes") or {}).get("allow_overlap"))
    rng = seed_rng(world_seed(cfg), "meme_seeds")
    deg = tie_degrees(profiles)
    pool = sorted(a for a, p in profiles.items() if not p.reserve and a not in (exclude or set()))
    used: set[str] = set()
    chosen: dict[str, list[str]] = {}

    def take(mid: str, ids: list[str]) -> None:
        chosen[mid] = list(ids)
        if not allow_overlap:
            used.update(ids)

    for e in entries:                                      # 1. explicit minorities
        if e["agents"]:
            unknown = [a for a in e["agents"] if a not in profiles]
            if unknown:
                raise ValueError(f"meme {e['id']!r}: seeds.agents names agents outside the population: {unknown}")
            clash = sorted(set(e["agents"]) & used)
            if clash:
                raise ValueError(f"meme {e['id']!r}: seeds.agents overlaps another meme's minority ({clash}); "
                                 "set memes.allow_overlap to accept an agent carrying two phrases")
            take(e["id"], e["agents"])
    spread = [e for e in entries if not e["agents"] and e["strategy"] == "spread"]
    if spread:
        avail = [a for a in pool if a not in used]
        need = sum(e["k"] for e in spread)
        if len(avail) < need:
            raise ValueError(f"memes: {need} seeds needed for the matched cohort but only {len(avail)} agents are free")
        for mid, ids in _spread(avail, spread, deg, profiles, rng).items():
            take(mid, ids)
    for e in entries:                                      # 3. the deliberately unmatched strategies
        if e["id"] in chosen:
            continue
        avail = [a for a in pool if a not in used]
        if len(avail) < e["k"]:
            raise ValueError(f"meme {e['id']!r}: {e['k']} seeds needed but only {len(avail)} agents are free")
        if e["strategy"] == "high_degree":
            ids = sorted(avail, key=lambda a: (-deg[a][1], -deg[a][0], a))[: e["k"]]
        elif e["strategy"] == "cluster":
            ids = _cluster(avail, e["k"], profiles, cfg, rng)
        else:
            ids = sorted(str(x) for x in rng.choice(avail, size=e["k"], replace=False))
        take(e["id"], ids)
    return {e["id"]: chosen[e["id"]] for e in entries}


def _spread(avail: list[str], entries: list[dict], deg: dict, profiles: dict, rng) -> dict[str, list[str]]:
    """Stratification (which agents) + minimization (which meme gets which of them).

    A cyclic rotation inside a stratum only balances the second covariate when the number of strata is a
    multiple of the number of memes; with k=5 and four memes one meme systematically gets the top slot
    twice. So the stratum's drawn agents are instead dealt out by MINIMIZATION (Pocock & Simon 1975, the
    covariate-adaptive allocation used in trials for exactly this): of the ways to hand this stratum's
    agents to the participating memes, take the one leaving the smallest spread between the memes'
    running totals, summed over all three covariates in standardized units. Which agents a stratum
    contributes stays random; only who gets which of them is chosen. Ties go by a meme order randomised
    once from the run seed."""
    from itertools import permutations
    cov = {a: (float(deg[a][0]), deg[a][1], float(_reach(profiles[a]))) for a in avail}
    sd = [pstdev(v) or 1.0 for v in zip(*cov.values())] if len(cov) > 1 else [1.0, 1.0, 1.0]
    z = {a: tuple(v / s for v, s in zip(cov[a], sd)) for a in avail}
    ranked = sorted(avail, key=lambda a: (-deg[a][1], -deg[a][0], a))
    K, n = max(e["k"] for e in entries), len(ranked)
    strata = [ranked[(s * n) // K: ((s + 1) * n) // K] for s in range(K)]
    # a meme wanting fewer than K seeds takes its strata evenly across the degree range, never the top k
    wants = {e["id"]: {(i * K) // e["k"] for i in range(e["k"])} for e in entries}
    order = [entries[int(i)]["id"] for i in rng.permutation(len(entries))]
    out: dict[str, list[tuple[int, str]]] = {e["id"]: [] for e in entries}
    run = {e["id"]: [0.0, 0.0, 0.0] for e in entries}
    cand: dict[int, list[str]] = {}
    for s, block in enumerate(strata):
        part = [mid for mid in order if s in wants[mid]]
        if not part:
            continue
        if len(block) < len(part):
            raise ValueError(f"memes: degree stratum {s} holds {len(block)} agents, too few for {len(part)} memes")
        n_cand = min(len(block), MEME_CANDIDATES_PER_SEED * len(part))
        cand[s] = sorted(str(x) for x in rng.choice(block, size=n_cand, replace=False))
        drawn = sorted(cand[s], key=lambda a: (-sum(z[a]), a))[: len(part)]
        best, pick = None, None
        for perm in (permutations(drawn) if len(drawn) <= 6 else [tuple(drawn)]):
            trial = dict(run)
            for mid, a in zip(part, perm):
                trial[mid] = [r + x for r, x in zip(run[mid], z[a])]
            score = round(sum(max(t[c] for t in trial.values()) - min(t[c] for t in trial.values())
                              for c in range(3)), 9)
            if best is None or score < best:
                best, pick = score, perm
        for mid, a in zip(part, pick):
            out[mid].append((s, a))
            run[mid] = [r + x for r, x in zip(run[mid], z[a])]
    _rebalance(out, z, order, cand)
    return {mid: sorted(a for _s, a in ids) for mid, ids in out.items()}


def _imbalance(out: dict, z: dict) -> tuple[float, float]:
    """How unevenly the covariates are spread over the minorities, as (worst covariate, all covariates):
    the variance of the group totals, one term per covariate, in standardized units. Zero means every
    minority carries the same total degree, weighted degree and routine reach. The worst covariate leads,
    because that is what the balance check judges; the sum only breaks ties."""
    if len(out) < 2:
        return 0.0, 0.0
    tot = {m: [sum(z[a][c] for _s, a in ids) for c in range(3)] for m, ids in out.items()}
    var = [pstdev([t[c] for t in tot.values()]) ** 2 for c in range(3)]
    return round(max(var), 9), round(sum(var), 9)


def _rebalance(out: dict, z: dict, order: list[str], cand: dict[int, list[str]]) -> None:
    """Local search over the legal allocations while it lowers the imbalance. Two moves, both of which
    keep the stratification exact (each meme still holds one agent per degree band):
      - swap two memes' agents inside one stratum;
      - replace a meme's agent with an unused candidate drawn from the same stratum.

    Dealing the strata one after another is greedy: an early stratum can force a later one into a bad
    hand. Deterministic: strata, meme pairs and candidates are visited in a fixed order, and the first
    improving move is taken."""
    strata = sorted(cand)
    for _pass in range(64):
        best = _imbalance(out, z)
        moved = False
        for s in strata:
            at = {m: k for m, ids in out.items() for k, (t, _a) in enumerate(ids) if t == s}
            for i, m in enumerate(order):
                if m not in at:
                    continue
                for n in order[i + 1:]:
                    if n not in at:
                        continue
                    out[m][at[m]], out[n][at[n]] = out[n][at[n]], out[m][at[m]]
                    now = _imbalance(out, z)
                    if now < best:
                        best, moved = now, True
                    else:
                        out[m][at[m]], out[n][at[n]] = out[n][at[n]], out[m][at[m]]
            used = {a for ids in out.values() for _t, a in ids}
            for m in order:
                if m not in at:
                    continue
                keep = out[m][at[m]]
                for c in cand[s]:
                    if c in used:
                        continue
                    out[m][at[m]] = (s, c)
                    now = _imbalance(out, z)
                    if now < best:
                        best, moved, keep = now, True, (s, c)
                        used = {a for ids in out.values() for _t, a in ids}
                out[m][at[m]] = keep
        if not moved:
            return


def _cluster(avail: list[str], k: int, profiles: dict, cfg: dict, rng) -> list[str]:
    """k agents inside one friend circle: start from a random member of a circle big enough to hold the
    minority and grow outward along ties, so the seeds are socially adjacent rather than scattered."""
    clusters, _src = seed_clusters(profiles, cfg)
    free = set(avail)
    groups = {}
    for a in avail:
        groups.setdefault(clusters.get(a, f"free:{a}"), []).append(a)
    big = sorted((g for g, ms in groups.items() if len(ms) >= k), key=str)
    members = sorted(groups[big[int(rng.integers(len(big)))]]) if big else sorted(avail)
    start = members[int(rng.integers(len(members)))]
    picked, frontier = [start], [start]
    while len(picked) < k and frontier:
        x = frontier.pop(0)
        for y in sorted(b for b, r in profiles[x].relationships.items()
                        if r.relation_type != "stranger" and b in free and b not in picked):
            if len(picked) >= k:
                break
            picked.append(y)
            frontier.append(y)
    for a in members + sorted(avail):                     # the circle ran out of ties: fill from it, then anywhere
        if len(picked) >= k:
            break
        if a not in picked:
            picked.append(a)
    return sorted(picked[:k])


def plan_memes(profiles: dict[str, AgentProfile], cfg: dict, exclude: set[str] | None = None) -> list[dict]:
    """The registry with its committed minorities resolved: each entry plus "seeds" (agent ids) and
    "habits" ({agent: the exact line appended to that agent's profile}). Pure: it does not touch the
    profiles, so the observer can recompute the same plan offline from the same config and seed."""
    entries = meme_registry(cfg)
    if not entries:
        return []
    seeds = select_seeds(profiles, entries, cfg, exclude)
    return [dict(e, seeds=seeds[e["id"]],
                 habits={a: _norm_habit(e["habit"], profiles[a]) for a in seeds[e["id"]]}) for e in entries]


def apply_control(profiles: dict[str, AgentProfile], cfg: dict) -> dict | None:
    """The positive control alone (`controls.planted_phrase`): one agent, one habit, {"agent", "habit"}
    or None. Kept separate from the cohort so the observer can recover the control's exact applied
    wording from a stub profile without also having to rebuild the population's tie graph."""
    pp = (cfg.get("controls") or {}).get("planted_phrase")
    if not pp:
        return None
    aid = pp.get("agent")
    if aid not in profiles:
        raise ValueError(f"controls.planted_phrase.agent {aid!r} is not in the loaded population")
    if not " ".join(str(pp.get("habit") or "").split()):
        raise ValueError("controls.planted_phrase.habit is empty")
    habit = _norm_habit(pp["habit"], profiles[aid])
    profiles[aid].habits = list(profiles[aid].habits) + [habit]
    return {"agent": aid, "habit": habit}


def apply_planted(profiles: dict[str, AgentProfile], cfg: dict) -> dict | None:
    """Apply every injected expression to the profiles, before the agents are built. Returns what was
    applied (for the manifest), or None when nothing was.

    Two roles, one mechanism (see the section header):
      - `controls.planted_phrase` (positive control): one agent, one habit. Unchanged, and reported in
        exactly the {"agent", "habit"} shape existing configs, tests and manifests expect.
      - `memes.registry` (study memes, `memes.enabled`): each meme's habit line is appended to every
        agent of its own committed minority. The phrase appears ONLY here -- R1: the world never says it.
    The two are disambiguated by which config block declared them, and reported under separate keys, so
    a run with no registry returns the legacy record unchanged: {"agent", "habit"} (+ "memes": [...]).
    """
    applied = apply_control(profiles, cfg) or {}
    # the control agent is kept out of every study minority: one agent carrying two injected phrases is
    # a confound the 2x2 cannot model
    plan = plan_memes(profiles, cfg, exclude={applied["agent"]} if applied else None)
    for m in plan:
        for aid, habit in m["habits"].items():
            profiles[aid].habits = list(profiles[aid].habits) + [habit]
    if plan:
        applied["memes"] = [{k: m[k] for k in ("id", "phrase", "grounding", "breadth", "habit", "strategy",
                                               "site", "seeds", "habits")} for m in plan]
    return applied or None


# ---------------------------------------------------------------- R6 evidence

def _smd(a: list[float], b: list[float]) -> float | None:
    """Standardized mean difference between two groups on one covariate (the matching literature's
    balance measure): the difference in means over the pooled SD. None when neither group varies."""
    if not a or not b:
        return None
    sa, sb = (pstdev(a) if len(a) > 1 else 0.0), (pstdev(b) if len(b) > 1 else 0.0)
    pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    if pooled == 0:
        return 0.0 if fmean(a) == fmean(b) else None
    return (fmean(a) - fmean(b)) / pooled


def _dist(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": round(fmean(xs), 2) if xs else None,
            "sd": round(pstdev(xs), 2) if len(xs) > 1 else 0.0,
            "median": round(median(xs), 1) if xs else None,
            "min": round(min(xs), 1) if xs else None, "max": round(max(xs), 1) if xs else None}


def seed_matching(profiles: dict[str, AgentProfile], plan: list[dict], cfg: dict) -> dict:
    """R6 evidence: the committed minorities side by side.

    If one meme's seeds are better connected than another's, a difference in spread is attributable to
    the graph rather than to grounding or breadth and the 2x2 is dead. Reported per meme: the degree
    distribution, cluster membership, exposure to each meme's incident site, and routine reach; across
    memes: the largest |SMD| between any two minorities on each covariate (`balance`), and whether the
    minorities are disjoint.
    """
    deg = tie_degrees(profiles)
    clusters, csrc = seed_clusters(profiles, cfg)
    sites = sorted({m["site"] for m in plan if m["site"]})
    cov = {"degree": lambda a: float(deg[a][0]), "weighted_degree": lambda a: deg[a][1],
           "routine_reach": lambda a: float(_reach(profiles[a]))}
    rows = {}
    for m in plan:
        ids = m["seeds"]
        routes = {a: {r.location for r in profiles[a].routine} for a in ids}
        nbrs = {a: {b for b, r in profiles[a].relationships.items()
                    if r.relation_type != "stranger" and b in profiles} for a in ids}
        rows[m["id"]] = {
            "id": m["id"], "grounding": m["grounding"], "breadth": m["breadth"], "strategy": m["strategy"],
            "k": len(ids), "seeds": list(ids), "site": m["site"],
            # R6 also asks the seed habits to be comparable: a longer, more emphatic line is itself a
            # manipulation. Reported, not enforced -- the wording is the config's to write.
            "habit_words": len(m["habit"].split()), "phrase_words": len(m["phrase"].split()),
            **{name: _dist([f(a) for a in ids]) for name, f in cov.items()},
            "clusters": {c: sum(1 for a in ids if clusters.get(a) == c)
                         for c in sorted({clusters.get(a, "-") for a in ids})},
            "n_clusters": len({clusters.get(a, f"free:{a}") for a in ids}),
            # how much of the graph the minority can reach first-hand, and how much of its talking it
            # would be doing to itself: two minorities of equal degree still differ if one is a clique
            "reach_1hop": len(set().union(*nbrs.values()) - set(ids)) if ids else 0,
            "internal_ties": sum(1 for i, a in enumerate(ids) for b in ids[i + 1:] if b in nbrs[a]),
            "site_exposure": {s: sum(1 for a in ids if s in routes[a]) for s in sites},
            "at_own_site": (sum(1 for a in ids if m["site"] in routes[a]) if m["site"] else None),
            "unexposed": sum(1 for a in ids if not (routes[a] & set(sites))) if sites else len(ids),
        }
    balance = {}
    for name, f in cov.items():
        worst, pair = None, None
        for i, x in enumerate(plan):
            for y in plan[i + 1:]:
                d = _smd([f(a) for a in x["seeds"]], [f(a) for a in y["seeds"]])
                if d is not None and (worst is None or abs(d) > abs(worst)):
                    worst, pair = d, (x["id"], y["id"])
        balance[name] = {"max_abs_smd": (round(abs(worst), 3) if worst is not None else None), "pair": pair,
                         "ok": (abs(worst) <= MEME_SMD_OK if worst is not None else None)}
    seen: dict[str, list[str]] = {}
    for m in plan:
        for a in m["seeds"]:
            seen.setdefault(a, []).append(m["id"])
    pop = [float(deg[a][0]) for a in profiles]
    hw = [r["habit_words"] for r in rows.values()]
    balance["habit_words"] = {"max_abs_smd": None, "pair": None, "spread": (max(hw) - min(hw)) if hw else 0,
                              "ok": (max(hw) - min(hw) <= max(3, 0.25 * min(hw))) if hw else None}
    return {"memes": rows, "balance": balance, "cluster_source": csrc, "sites": sites,
            "vs_random": _benchmark(profiles, plan, cov),
            "disjoint": all(len(v) == 1 for v in seen.values()),
            "overlap": {a: v for a, v in sorted(seen.items()) if len(v) > 1},
            "population": {"n": len(profiles), "degree": _dist(pop), "smd_threshold": MEME_SMD_OK,
                           "matched": all(b["ok"] is not False for b in balance.values())}}


def _benchmark(profiles: dict, plan: list[dict], cov: dict) -> dict:
    """"Well matched" needs a scale, so the observed imbalance is compared with the imbalance of plain
    random minorities of the same sizes: the share of MEME_BENCHMARK_DRAWS random allocations whose
    worst |SMD| is no better than this one. 0.99 means only 1% of random draws would match this well.
    The benchmark's own RNG is seeded from the meme ids, so it is reproducible and cannot be confused
    with the run's seed."""
    from backend.simulation.rngs import seed_rng
    ids = sorted(a for a, p in profiles.items() if not p.reserve)
    sizes = [len(m["seeds"]) for m in plan]
    if len(plan) < 2 or sum(sizes) > len(ids):
        return {}

    def worst(groups: list[list[str]]) -> float:
        out = 0.0
        for f in cov.values():
            for i, g in enumerate(groups):
                for h in groups[i + 1:]:
                    d = _smd([f(a) for a in g], [f(a) for a in h])
                    out = max(out, abs(d)) if d is not None else out
        return out
    rng = seed_rng("meme_balance_benchmark", *[m["id"] for m in plan])
    obs = worst([m["seeds"] for m in plan])
    beaten = 0
    for _ in range(MEME_BENCHMARK_DRAWS):
        pick = [str(x) for x in rng.choice(ids, size=sum(sizes), replace=False)]
        groups, at = [], 0
        for k in sizes:
            groups.append(pick[at: at + k])
            at += k
        beaten += worst(groups) >= obs
    return {"draws": MEME_BENCHMARK_DRAWS, "observed_max_abs_smd": round(obs, 3),
            "better_than_random_share": round(beaten / MEME_BENCHMARK_DRAWS, 3)}


def matching_table(report: dict) -> str:
    """The R6 table as plain text: one column per meme, one row per covariate."""
    rows = report["memes"]
    ids = list(rows)
    if not ids:
        return "(no memes)"
    w = max(22, *(len(i) + 2 for i in ids))
    out = [f"{'':22}" + "".join(f"{i:>{w}}" for i in ids)]

    def line(label, fn):
        out.append(f"{label:22}" + "".join(f"{str(fn(rows[i])):>{w}}" for i in ids))
    line("cell", lambda r: f"{r['grounding'][:5]}x{r['breadth']}")
    line("strategy", lambda r: r["strategy"])
    line("k", lambda r: r["k"])
    for cv in ("degree", "weighted_degree", "routine_reach"):
        line(f"{cv} mean(sd)", lambda r, c=cv: f"{r[c]['mean']}({r[c]['sd']})")
        line(f"{cv} min-max", lambda r, c=cv: f"{r[c]['min']}-{r[c]['max']}")
    line("distinct clusters", lambda r: r["n_clusters"])
    line("1-hop reach", lambda r: r["reach_1hop"])
    line("ties inside minority", lambda r: r["internal_ties"])
    line("habit line words", lambda r: r["habit_words"])
    for s in report["sites"]:
        line(f"routine at {s[:11]}", lambda r, s=s: r["site_exposure"].get(s, 0))
    line("at no incident site", lambda r: r["unexposed"])
    out.append("")
    for cv, b in report["balance"].items():
        if b["max_abs_smd"] is None:
            out.append(f"spread of  {cv:16} {b.get('spread')} words ({'ok' if b['ok'] else 'NOT COMPARABLE'})")
            continue
        out.append(f"max |SMD| {cv:16} {b['max_abs_smd']} {b['pair'] or ''} "
                   f"({'ok' if b['ok'] else 'NOT MATCHED'} at {report['population']['smd_threshold']})")
    vr = report.get("vs_random") or {}
    if vr:
        out.append(f"worst |SMD| {vr['observed_max_abs_smd']}: better than {100 * vr['better_than_random_share']:.0f}% "
                   f"of {vr['draws']} random allocations of the same sizes")
    out.append(f"disjoint minorities: {report['disjoint']}   clusters from: {report['cluster_source']}   "
               f"population degree mean {report['population']['degree']['mean']}")
    return "\n".join(out)


def _personal_properties(agent: dict) -> dict:
    personal = agent.get("personal", {})
    if not isinstance(personal, dict):
        raise ValueError(f"Agent {agent['id']!r}: personal must be a mapping")
    for key, value in personal.items():
        if key not in PERSONAL_LABELS:
            raise ValueError(f"Agent {agent['id']!r}: unknown personal property {key!r}")
        valid = (isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)
                 if key in PERSONAL_LIST_FIELDS else isinstance(value, str) and bool(value.strip()))
        if not valid:
            expected = "a list of nonempty strings" if key in PERSONAL_LIST_FIELDS else "a nonempty string"
            raise ValueError(f"Agent {agent['id']!r}: personal.{key} must be {expected}")
    return personal


def _friend_group_definitions(data: dict) -> dict:
    """Validate against the complete file before limiting the run's population."""
    all_ids = [a["id"] for a in data["agents"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Population contains duplicate agent IDs")
    definitions = data.get("friend_groups", {})
    if not isinstance(definitions, dict):
        raise ValueError("friend_groups must be a mapping")
    for gid, group in definitions.items():
        if not isinstance(gid, str) or not gid.strip() or not isinstance(group, dict):
            raise ValueError("Each friend group needs a string ID and a mapping")
        if not isinstance(group.get("name"), str) or not group["name"].strip():
            raise ValueError(f"Friend group {gid!r} needs a nonempty display name")
        members = group.get("members")
        if (not isinstance(members, list) or len(members) < 2
                or not all(isinstance(m, str) for m in members)):
            raise ValueError(f"Friend group {gid!r} needs at least two agent IDs")
        if len(set(members)) != len(members):
            raise ValueError(f"Friend group {gid!r} has duplicate members")
        unknown = set(members) - set(all_ids)
        if unknown:
            raise ValueError(f"Friend group {gid!r} has unknown members: {sorted(unknown)}")
    for agent in data["agents"]:
        declared = agent.get("friend_groups", [])
        if not isinstance(declared, list) or not all(isinstance(gid, str) for gid in declared):
            raise ValueError(f"Agent {agent['id']!r}: friend_groups must be a list of group IDs")
        if len(set(declared)) != len(declared):
            raise ValueError(f"Agent {agent['id']!r}: duplicate friend group IDs")
        unknown = set(declared) - set(definitions)
        if unknown:
            raise ValueError(f"Agent {agent['id']!r}: unknown friend groups: {sorted(unknown)}")
        expected = {gid for gid, group in definitions.items() if agent["id"] in group["members"]}
        if set(declared) != expected:
            raise ValueError(f"Agent {agent['id']!r}: friend_groups do not match group memberships")
    return definitions


def load_population(path: str | Path, n: int | None = None, include_reserves: bool = False):
    """-> (profiles, groups). `n` takes the first n of `agents:` (the founders). `include_reserves` also loads
    the file's `reserves:` personas (v3 §3.1: the full persona universe; `reserve=True`, no role until they
    arrive). Without reserves the result is exactly the v2 one (plus `role` from `coop_role`)."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open(encoding="utf-8-sig") as source:
        data = yaml.safe_load(source)
    friend_definitions = _friend_group_definitions(data)
    raw = [dict(a, _reserve=False) for a in data["agents"][: n or None]]
    if include_reserves:
        raw += [dict(a, _reserve=True) for a in (data.get("reserves") or [])]
    ids = {a["id"] for a in raw}
    from backend.simulation.world import PUBLIC_PLACES, WORLD_GRAPH
    profiles = {}
    for a in raw:
        routine = [RoutineEntry(**r) for r in a["routine"]]
        known = sorted({r.location for r in routine} | {"Quad", "Dining Hall", "Cafe", "Library"} | set(PUBLIC_PLACES))
        known = [k for k in known if k in WORLD_GRAPH]
        prof = AgentProfile(id=a["id"], name=a["name"], demographics=a["demographics"],
                            background=a["background"], personality=a["personality"],
                            interests=a["interests"], habits=a["habits"], routine=routine,
                            home=a["home"], sprite=a.get("sprite", "Abigail_Chen"),
                            known_locations=known, personal=_personal_properties(a),
                            role=None if a["_reserve"] else a.get("coop_role", a.get("role")),
                            reserve=bool(a["_reserve"]))
        profiles[prof.id] = prof
    for r in data.get("relationships", []):
        if r["a"] in ids and r["b"] in ids:
            rel = Relationship(r["type"], float(r["familiarity"]), float(r["affinity"]))
            profiles[r["a"]].relationships[r["b"]] = rel
            profiles[r["b"]].relationships[r["a"]] = Relationship(**asdict(rel))
    for a in profiles.values():
        for b in profiles:
            if b != a.id and b not in a.relationships:
                a.relationships[b] = Relationship()
    groups = {g: [m for m in members if m in ids] for g, members in (data.get("groups") or {}).items()}
    groups = {g: m for g, m in groups.items() if len(m) >= 2}
    for gid, group in friend_definitions.items():
        members = [m for m in group["members"] if m in ids]
        if len(members) < 2:
            continue
        if gid in groups and set(groups[gid]) != set(members):
            raise ValueError(f"Friend group {gid!r} conflicts with the ordinary group of the same ID")
        groups[gid] = members
        for aid in members:
            profiles[aid].friend_groups.append({"id": gid, "name": group["name"],
                                                "members": [profiles[m].name for m in members if m != aid]})
    return profiles, groups
