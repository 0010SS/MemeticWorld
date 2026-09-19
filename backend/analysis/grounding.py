"""Does a candidate track the hidden structure beyond chance? (OBSERVER ONLY; ontology v2 §4)

- Each usage linked to events (provenance `referent_event_ids` only) gets a fractional family
  distribution: 1/|links| per linked event, split by the events' latent_type.
- Statistic: the weighted share of the post-hoc best family z,
  share = max_f sum_u w_u(f) / n_linked_usages. (The legacy any()-hit precision counted a usage as a
  hit when *any* link had type z, so multi-linked usages pushed it towards 1.)
- Surface units: instances that share one surface wording (a v2 skin, a v1 scenario template) carry
  the same family by construction and share their fact wording, so lexical links co-link an
  utterance about one incident with all its same-skin siblings. The null therefore permutes family
  labels over surface UNITS, not events: link weights are summed per unit and each unit gets one
  label draw (composed scrambled/none instances have no skin and stay one unit each). An
  event-level null would count k siblings as k independent label draws and make talk about one
  recurring incident type look family-specific (in the real arm only).
- Strata follow the world design; a unit sits in the stratum of its first instance.
  * assignment mode `home` (families cast into their home circles): protagonist's circle x day. The
    circle is the one P actually belongs to (manifest `circles`, else the population file; "" for the
    free pool), which carries the social confound (who witnessed it, who talks to whom). The event's
    `circle` field is NOT used when membership is known: in `home` mode it is a function of the family
    itself, so permuting within it would re-encode the label (no power at all).
  * modes `none` / `balanced`: day only. The family is drawn independently of the circle there, so a
    circle-specific phrase already meets the right null, and circle strata would only cost power.
  * v1 runs: day only (a warning when their families were clustered in home groups).
- Identifiability: when every stratum holds units of a single family (for example `home` mode where
  each circle only ever gets its own family), the permutation null equals the observed labelling:
  grounding cannot separate family-specific from circle-specific talk. The run is flagged
  `identifiable: false` (with `family_determined_by_circle`), nothing is grounded, and outcomes report
  n_grounded as null (not 0). A low share of units in mixed strata gets a low-power warning.
- p = (1 + #{null >= observed}) / (1 + permutations); Benjamini-Hochberg q across the testable
  candidates (n-gram candidates with linked_usages >= 3 and linked_speakers >= 2; the filter does
  not look at labels).
- grounded = q < analysis.fdr_q and linked_usages >= 3 and linked_speakers >= 2 and the run is
  identifiable and z_surfaces >= 2: the usages linked to z must be about at least two distinct
  surface units of z (each usage votes for the z unit it is most linked to), so one recurring
  incident type never counts as grounded.

The statistic is linear in the per-unit link weights, so every candidate x permutation is one
matrix product per family (numpy), which keeps 40 candidates x 1000 permutations well under a second.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

MIN_LINKED_USAGES = 3
MIN_LINKED_SPEAKERS = 2
MIN_SURFACES = 2
LOW_POWER_SHARE = 0.5


def stratified_permutations(labels: np.ndarray, strata: np.ndarray, n: int, rng) -> np.ndarray:
    """(n, E) label arrays, each a permutation of `labels` within every stratum."""
    labels = np.asarray(labels)
    out = np.empty((n, len(labels)), dtype=labels.dtype)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        order = np.argsort(rng.random((n, len(idx))), axis=1)
        out[:, idx] = labels[idx][order]
    return out


def best_share(V: np.ndarray, L: np.ndarray, n_fam: int) -> tuple[np.ndarray, np.ndarray]:
    """V: (C, E) link weight per event; L: (P, E) family index per event.
    -> (max over families of the weight share (C, P), argmax family (C, P)). Rows of V sum to n_linked,
    callers divide."""
    per = np.stack([V @ (L == f).T.astype(float) for f in range(n_fam)], axis=-1)  # (C, P, F)
    return per.max(axis=-1), per.argmax(axis=-1)


def bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values (monotone, capped at 1), in input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return [float(x) for x in out]


def link_weights(usage_links: list[list[str]], ev_index: dict) -> tuple[np.ndarray, int]:
    """Per-event weight: each linked usage spreads 1 over its linked events."""
    v = np.zeros(len(ev_index))
    n = 0
    for links in usage_links:
        ls = [ev_index[e] for e in links if e in ev_index]
        if ls:
            n += 1
            for i in ls:
                v[i] += 1.0 / len(ls)
    return v, n


def permutation_test(V: np.ndarray, n_linked: np.ndarray, labels: np.ndarray, strata: np.ndarray, n_fam: int,
                     n_perm: int, rng) -> dict:
    """Observed best-family share, null summary and p-value per row of V."""
    denom = np.maximum(n_linked, 1)[:, None]
    obs, z = best_share(V, labels[None, :], n_fam)
    obs = obs[:, 0] / denom[:, 0]
    L = stratified_permutations(labels, strata, n_perm, rng)
    null = best_share(V, L, n_fam)[0] / denom
    p = (1 + (null >= obs[:, None] - 1e-9).sum(axis=1)) / (n_perm + 1)
    return {"observed": obs, "best": z[:, 0], "null_mean": null.mean(axis=1),
            "null_p95": np.quantile(null, 0.95, axis=1), "p": p}


def surface_unit(eid: str, ev: dict) -> str:
    """Events sharing one surface wording (v2 skin, v1 scenario template) form one unit; composed
    instances (skin null) are their own unit. The label is part of the key, so a unit's label is constant."""
    return f"{ev.get('latent_type')}|{ev.get('skin') or ev.get('scenario') or eid}"


def circle_membership(rd) -> dict | None:
    """agent id -> circle id: the run's manifest `circles` (written by the v2 engine, including generated
    topology), else the population file's `circles:`; None when neither is available."""
    circles = (getattr(rd, "manifest", None) or {}).get("circles")
    if not isinstance(circles, dict):
        circles = None
        pop = (getattr(rd, "cfg", None) or {}).get("population")
        if pop and (rd.cfg.get("topology") or {}).get("mode") != "generated":
            try:
                from backend.simulation.circles import load_circles
                circles = load_circles(pop, list(getattr(rd, "agents", None) or []))
            except Exception:      # population file moved or has no circles section
                circles = None
    if circles is None:
        return None
    return {m: c for c, ms in circles.items() for m in ms or []}


def strata_of(events: dict, ids: list[str], tpd: int, member_of: dict | None = None,
              by_circle: bool = True) -> tuple[np.ndarray, list, str | None]:
    """Per-event stratum index, the stratum keys (circle, day), and the circle source.

    by_circle: circle = the protagonist's actual circle ("" = free pool) when membership is known, else
    the event's `circle` field (v2 records without membership), else "" (v1). Otherwise day only."""
    def p_agent(ev):
        return (((ev.get("roles") or {}).get("P")) or {}).get("agent")
    if by_circle and member_of:
        circ, source = [member_of.get(p_agent(events[e]), "") for e in ids], "protagonist"
    elif by_circle and any(events[e].get("circle") for e in ids):
        circ, source = [events[e].get("circle") or "" for e in ids], "event_field"
    else:
        circ, source = [""] * len(ids), None
    keys = [(c, int(events[e].get("start_tick", 0)) // tpd) for c, e in zip(circ, ids)]
    uniq = sorted(set(keys))
    return np.array([uniq.index(k) for k in keys], dtype=int), uniq, source


def unit_design(events: dict, ids: list[str], labels: np.ndarray, strata: np.ndarray):
    """-> (M (E, U) event-to-unit indicator, unit ids, unit labels, unit strata (first instance's))."""
    unit_of = [surface_unit(e, events[e]) for e in ids]
    order = sorted(range(len(ids)), key=lambda i: (int(events[ids[i]].get("start_tick", 0)), ids[i]))
    units, first = [], {}
    for i in order:
        if unit_of[i] not in first:
            first[unit_of[i]] = i
            units.append(unit_of[i])
    ui = {u: k for k, u in enumerate(units)}
    M = np.zeros((len(ids), len(units)))
    for i, u in enumerate(unit_of):
        M[i, ui[u]] = 1.0
    return M, units, labels[[first[u] for u in units]], strata[[first[u] for u in units]], unit_of


def identifiability(unit_labels: np.ndarray, unit_strata: np.ndarray, labels: np.ndarray, strata: np.ndarray,
                    uniq: list, n_fam: int) -> dict:
    """Can the within-strata null separate family from stratum? mixed_strata_share = share of units in
    strata that hold >= 2 families (0: every permutation reproduces the observed labelling);
    family_determined_by_circle: over events, every circle (>= 2 of them) holds a single family."""
    if n_fam < 2 or len(unit_labels) == 0:
        return {"identifiable": None, "mixed_strata_share": None, "family_determined_by_circle": None}
    per: dict = {}
    for lab, st in zip(unit_labels, unit_strata):
        per.setdefault(int(st), set()).add(int(lab))
    mixed = sum(1 for st in unit_strata if len(per[int(st)]) >= 2) / len(unit_labels)
    by_circle: dict = {}
    for lab, st in zip(labels, strata):
        by_circle.setdefault(uniq[int(st)][0], set()).add(int(lab))
    det = len(by_circle) >= 2 and all(len(f) == 1 for f in by_circle.values())
    return {"identifiable": mixed > 0, "mixed_strata_share": round(mixed, 3), "family_determined_by_circle": det}


def z_surfaces(links: list[list[str]], z: str, events: dict, unit_of: dict) -> list[str]:
    """Distinct surface units of family z the usages are about: each usage linked to z votes for the z
    unit it has most links to (ties: the first unit id)."""
    votes = set()
    for ls in links:
        cnt = Counter(unit_of[e] for e in ls if e in unit_of and events[e]["latent_type"] == z)
        if cnt:
            top = max(cnt.values())
            votes.add(min(u for u, k in cnt.items() if k == top))
    return sorted(votes)


def _probe_accuracy(probe: dict | None, z: str | None) -> dict | None:
    """Probe match accuracy vs chance for agents who heard or used the expression vs those who didn't."""
    if not probe or not z:
        return None
    fams = [o.get("family") for o in probe.get("options", [])]
    if z not in fams:
        return None
    res = {"chance": round(1 / len(fams), 3)}
    for grp, pick in (("exposed", True), ("unexposed", False)):
        ans = [a for a in probe.get("agents", {}).values()
               if bool(a.get("heard_before") or a.get("used")) == pick and a.get("match_family")]
        res[grp] = {"n": len(ans),
                    "accuracy": round(sum(a["match_family"] == z for a in ans) / len(ans), 3) if ans else None}
    return res


def analyze_grounding(cands: list[dict], rd, probes: dict | None = None) -> dict:
    acfg = rd.cfg.get("analysis") or {}
    n_perm = int(acfg.get("permutations", 1000))
    fdr_q = float(acfg.get("fdr_q", 0.1))
    events = {k: e for k, e in rd.events.items() if e.get("latent_type")}
    ids = sorted(events)
    fams = sorted({events[e]["latent_type"] for e in ids})
    fi = {f: i for i, f in enumerate(fams)}
    ev_index = {e: i for i, e in enumerate(ids)}
    labels = np.array([fi[events[e]["latent_type"]] for e in ids], dtype=int)
    le = rd.cfg.get("latent_events") or {}
    home = (le.get("assignment") or {}).get("mode") == "home"
    strata, uniq, csrc = strata_of(events, ids, rd.ticks_per_day, circle_membership(rd) if home else None, by_circle=home)
    M, units, u_labels, u_strata, unit_list = unit_design(events, ids, labels, strata)
    unit_of = dict(zip(ids, unit_list))
    ident = identifiability(u_labels, u_strata, labels, strata, uniq, len(fams))
    prov = rd.provenance
    head = {"method": "best-family weighted share; labels permuted over surface units within "
                      + ("protagonist-circle x day" if home else "day") + " strata",
            "families": fams, "permutations": n_perm, "fdr_q": fdr_q, "n_strata": len(uniq),
            "strata_by": "circle x day" if any(c for c, _ in uniq) else "day", "circle_source": csrc,
            "n_events": len(ids), "n_units": len(units), "min_surfaces": MIN_SURFACES, **ident}
    warnings = []
    if head["strata_by"] == "day" and (home or float(le.get("clustering") or 0) > 0):
        warnings.append("families were clustered in home groups but events carry no circle: grounding cannot "
                        "separate family-specific from group-specific talk in this run")
    if ident["identifiable"] is False:
        why = ("the family is a deterministic function of the protagonist's circle" if ident["family_determined_by_circle"]
               else "every circle x day stratum holds a single family")
        warnings.append(f"grounding is not identifiable: {why}, so the permutation null equals the observed "
                        "labelling and family-specific talk cannot be told from circle-specific talk")
    elif ident["mixed_strata_share"] is not None and ident["mixed_strata_share"] < LOW_POWER_SHARE:
        warnings.append(f"only {ident['mixed_strata_share']:.0%} of surface units sit in strata holding >= 2 families: "
                        "grounding has little power in this run")
    if warnings:
        head["warning"] = "; ".join(warnings)
    rows, out, all_links = [], {}, {}
    for c in cands:
        links = [prov.get(u["utterance_id"], {}).get("referent_event_ids", []) for u in c["usages"]]
        links = [[e for e in ls if e in ev_index] for ls in links]
        all_links[c["id"]] = links
        v, n = link_weights(links, ev_index)
        linked = [u for u, ls in zip(c["usages"], links) if ls]
        per_usage = []
        for u, ls in zip(c["usages"], links):
            if ls:
                fc = Counter(events[e]["latent_type"] for e in ls)
                per_usage.append({"utterance_id": u["utterance_id"], "referent_event_ids": ls,
                                  "families": {f: round(k / len(ls), 3) for f, k in fc.items()}})
        mass = {f: float(v[labels == i].sum()) for f, i in fi.items()}
        out[c["id"]] = {"phrase": c["canonical_form"], "linked_usages": n,
                        "linked_speakers": len({u["speaker"] for u in linked}), "usages": per_usage,
                        "family_distribution": {f: round(m, 3) for f, m in mass.items() if m > 0}}
        rows.append((c, v, n))
    if fams and rows:
        V = np.stack([v for _, v, _ in rows]) @ M              # (C, U): link weight per surface unit
        nl = np.array([n for _, _, n in rows], dtype=float)
        rng = np.random.default_rng([int(rd.cfg.get("seed", 0)), 0x6D656D65])
        t = permutation_test(V, nl, u_labels, u_strata, len(fams), n_perm, rng)
    # LLM-discovered candidates get a p-value but stay out of the FDR family, so q (and n_grounded)
    # do not depend on the observer LLM
    testable = [out[c["id"]]["linked_usages"] >= MIN_LINKED_USAGES and out[c["id"]]["linked_speakers"] >= MIN_LINKED_SPEAKERS
                and not c.get("features", {}).get("llm_discovered") for c, _, _ in rows]
    qs = iter(bh([float(t["p"][i]) for i, ok in enumerate(testable) if ok])) if fams and rows else iter([])
    for i, (c, v, n) in enumerate(rows):
        g = out[c["id"]]
        if not fams or n == 0:
            g.update({"best_family": None, "share": None, "p_value": None, "q_value": None, "testable": False,
                      "z_surfaces": 0, "grounded": False, "holdout": None, "probe_accuracy": None})
            continue
        z = fams[int(t["best"][i])]
        q = next(qs) if testable[i] else None
        surf = z_surfaces(all_links[c["id"]], z, events, unit_of)
        g.update({"best_family": z, "share": round(float(t["observed"][i]), 3),
                  "null_mean": round(float(t["null_mean"][i]), 3), "null_p95": round(float(t["null_p95"][i]), 3),
                  "p_value": round(float(t["p"][i]), 4), "q_value": None if q is None else round(q, 4),
                  "testable": testable[i], "z_surfaces": len(surf), "z_surface_units": surf,
                  "grounded": bool(testable[i] and q is not None and q < fdr_q and len(surf) >= MIN_SURFACES
                                   and ident["identifiable"] is not False),
                  "holdout": _holdout(c, z, events, prov),
                  "probe_accuracy": _probe_accuracy((probes or {}).get(c["id"]), z)})
    return {**head, "n_testable": sum(testable), "candidates": out}


def _holdout(cand: dict, z: str, events: dict, prov: dict) -> dict:
    """Generalization to held-out surfaces: held-out instances of z (from the first use on) that a
    usage refers to."""
    t0 = cand["first_occurrence"]["tick"]
    hold = {e for e, ev in events.items() if ev.get("holdout") and ev["latent_type"] == z and ev.get("start_tick", 0) >= t0}
    hit_usages = [u for u in cand["usages"] if hold & set(prov.get(u["utterance_id"], {}).get("referent_event_ids", []))]
    linked = hold & {e for u in cand["usages"] for e in prov.get(u["utterance_id"], {}).get("referent_event_ids", [])}
    return {"holdout_instances": len(hold), "linked_instances": len(linked),
            "rate": round(len(linked) / len(hold), 3) if hold else None, "usages_linked": len(hit_usages)}
