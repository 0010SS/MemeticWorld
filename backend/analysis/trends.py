"""Meme trends over time windows (OBSERVER ONLY, LLM-free): share, uses, adopters, R_t, p_adopt, life-cycle
stage, half-life, and population turnover. Built on live.snapshot_context (the expression pool at tick t).

Definitions (per window of `window` ticks, windows start at 0, window, 2*window, ... <= t):
- share = uses of the meme in the window / all utterances in the window;
- adopter = an agent's first use; its *source* is the speaker of the latest use it heard strictly earlier
  in a different conversation (secondary adoption); R_t[w] = mean #secondary adopters per adopter who
  adopted in window w (null if none adopted in w); R = the same over all adopters (right-censored);
- p_adopt = exposed agents (heard it before any own use, or never used it) who later adopted in another
  conversation / all exposed, with a Wilson 95% interval;
- stage: extinct (no use in the last 8 windows) > declining (max share of the last 4 windows < 50% of peak)
  > established (>= max(3, active/3) speakers and used on >= 2 days or in >= 8 windows) > spreading
  (new adopters in the last 4 windows or last R_t > 1) > emerging;
- half_life_ticks = ticks from the peak window to the first later window with share < peak/2 (null if never).

- competition = rival expressions for ONE referent over the same windows, from places.resolve_place_references
  (which infers what agents call each place from their own speech, with no name lexicon): per referent the
  rivals, their uses and share per window, naming entropy per window (0 bits = one form has won), how many
  references named nothing, and how many reused the world's own label. The competition panel of the Culture
  UI renders this block when it is non-empty; it stays empty, and "competition" stays in `deferred`, when
  nothing competes or the run has no world map to resolve against.

- registry = the INJECTED meme cohort (battery.registry), listed separately from the mined pool because
  these are declared rather than discovered: per meme its adoption curve (with the seeded minority excluded
  from the adopters, since they were handed the phrase), unique users, formal variants, the seeds' degree
  distribution (so R6 - matched minorities - is reported rather than asserted), and survival after the day
  its basis was repaired. Empty for any run with no `memes` block.

Deferred (keys present, empty/null): neutral-copying baseline, group divergence, copy fidelity, meaning drift.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path

EXTINCT_WINDOWS = 8
RECENT_WINDOWS = 4
TOP_TURNOVER = 10
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_LOCK = threading.Lock()


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n <= 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _conv(u: dict):
    return u.get("conversation_id") or u.get("utterance_id")


def adoptions(usages: list[dict], seeds=()) -> list[dict]:
    """[{agent, adopt_tick, exposure_tick, source, conversation_id}] in adoption order.

    `seeds` is the committed minority of an injected meme: they were given the phrase, so their first use
    is not an adoption and counting it would inflate both the adoption curve and R_t. They remain sources
    for everyone else. Default () leaves discovered candidates scored exactly as before."""
    seeds = set(seeds)
    us = sorted(usages, key=lambda u: (u["tick"], u.get("idx", 0)))
    first, heard = {}, defaultdict(list)
    for u in us:
        if u["speaker"] not in seeds:
            first.setdefault(u["speaker"], u)
        for l in u.get("listeners") or []:
            if l != u["speaker"]:
                heard[l].append(u)
    out = []
    for a, u in first.items():
        prior = [x for x in heard.get(a, []) if x["tick"] < u["tick"] and _conv(x) != _conv(u)]
        src = prior[-1]["speaker"] if prior else None
        exp = min((x["tick"] for x in heard.get(a, [])), default=None)
        out.append({"agent": a, "adopt_tick": u["tick"], "exposure_tick": exp, "source": src,
                    "conversation_id": _conv(u)})
    return out


def exposure_stats(usages: list[dict], seeds=()) -> tuple[int, int]:
    """(n_exposed, n_adopted): exposed before any own use (or never used); adopted = later secondary adoption.
    A seed is never "exposed": it was handed the phrase, so it cannot be at risk of adopting it."""
    seeds = set(seeds)
    ad = {r["agent"]: r for r in adoptions(usages, seeds)}
    exposed = set()
    for u in usages:
        for l in u.get("listeners") or []:
            if l == u["speaker"] or l in seeds:
                continue
            if l not in ad or u["tick"] < ad[l]["adopt_tick"]:
                exposed.add(l)
    n_ad = sum(1 for a in exposed if a in ad and ad[a]["source"] is not None)
    return len(exposed), n_ad


def meme_series(usages: list[dict], n_windows: int, window: int, totals: list[int], tpd: int,
                n_active: int, t: int, seeds=()) -> dict:
    W = n_windows
    wi = lambda tick: min(max(int(tick) // window, 0), W - 1)
    seen, uses = set(), [0] * W
    for u in usages:
        if u.get("utterance_id") in seen:
            continue
        seen.add(u.get("utterance_id"))
        uses[wi(u["tick"])] += 1
    share = [round(uses[i] / totals[i], 4) if totals[i] else 0.0 for i in range(W)]
    ad = adoptions(usages, seeds)
    kids = defaultdict(int)
    for r in ad:
        if r["source"]:
            kids[r["source"]] += 1
    new_ad, rsum = [0] * W, [0] * W
    for r in ad:
        w = wi(r["adopt_tick"])
        new_ad[w] += 1
        rsum[w] += kids.get(r["agent"], 0)
    R_t = [round(rsum[i] / new_ad[i], 3) if new_ad[i] else None for i in range(W)]
    R = round(sum(kids.values()) / len(ad), 3) if ad else None
    cum, speakers_cum = 0, []
    for i in range(W):
        cum += new_ad[i]
        speakers_cum.append(cum)
    n_exp, n_ado = exposure_stats(usages, seeds)
    pk = max(range(W), key=lambda i: (share[i], -i)) if W else 0
    peak = share[pk] if W else 0.0
    half = None
    for i in range(pk + 1, W):
        if share[i] < peak / 2:
            half = (i - pk) * window
            break
    first_tick = min((u["tick"] for u in usages), default=0)
    active_w = [i for i in range(W) if uses[i]]
    last_w = active_w[-1] if active_w else -1
    days = {u["tick"] // tpd for u in usages}
    n_speakers = len({u["speaker"] for u in usages})
    if last_w < 0 or W - 1 - last_w >= EXTINCT_WINDOWS:
        stage = "extinct"
    elif W > RECENT_WINDOWS and pk < W - RECENT_WINDOWS and max(share[-RECENT_WINDOWS:]) < 0.5 * peak:
        stage = "declining"
    elif n_speakers >= max(3, math.ceil(n_active / 3)) and (len(days) >= 2 or len(active_w) >= 8):
        stage = "established"
    elif sum(new_ad[-RECENT_WINDOWS:]) > 0 or (next((x for x in reversed(R_t) if x is not None), 0) or 0) > 1:
        stage = "spreading"
    else:
        stage = "emerging"
    return {"share": share, "uses": uses, "speakers_cum": speakers_cum, "new_adopters": new_ad, "R_t": R_t, "R": R,
            "p_adopt": {"value": round(n_ado / n_exp, 4) if n_exp else None, "n_exposed": n_exp, "n_adopted": n_ado,
                        "ci": wilson(n_ado, n_exp)},
            "peak": {"window": pk, "share": peak}, "time_to_peak_ticks": max(0, pk * window - first_tick) if usages else None,
            "half_life_ticks": half, "stage": stage, "_adoptions": ad, "_last_w": last_w, "_first_w": wi(first_tick)}


def _label(man: dict, tick: int, tpd: int) -> str:
    mins = int(man.get("tick_minutes") or 15)
    try:
        h, m = (int(x) for x in str(man.get("start", "")).split("T")[1].split(":")[:2])
    except (IndexError, ValueError):
        h, m = 8, 0
    tot = h * 60 + m + (tick % tpd) * mins
    return f"D{tick // tpd + 1} {tot // 60 % 24:02d}:{tot % 60:02d}"


def compute(utterances: list[dict], memes: list[dict], *, tick: int, window: int, tpd: int, n_active: int,
            man: dict | None = None, competition: list | None = None) -> dict:
    """Pure core. memes: [{id, phrase, usages, ...extra fields copied through}].

    `competition` is the rival-names-per-referent block (places.competition_series); it is passed in
    rather than computed here so this core stays pure and LLM-free."""
    window = max(1, int(window))
    W = tick // window + 1
    totals = [0] * W
    for u in utterances:
        if u["tick"] <= tick:
            totals[min(u["tick"] // window, W - 1)] += 1
    out, series = [], []
    for m in memes:
        us = [u for u in m["usages"] if u["tick"] <= tick]
        if not us:
            continue
        s = meme_series(us, W, window, totals, tpd, n_active, tick)
        series.append(s)
        row = {k: v for k, v in m.items() if k != "usages"}
        row.update({k: v for k, v in s.items() if not k.startswith("_")})
        out.append(row)
    alive, births, deaths, turnover, prev_top = [], [], [], [], None
    for w in range(W):
        alive.append(sum(1 for s in series if s["_first_w"] <= w and any(s["uses"][max(0, w - EXTINCT_WINDOWS + 1):w + 1])))
        births.append(sum(1 for s in series if s["_first_w"] == w))
        deaths.append(sum(1 for s in series if s["_last_w"] >= 0 and s["_last_w"] + EXTINCT_WINDOWS == w))
        ranked = sorted((i for i, s in enumerate(series) if s["uses"][w]), key=lambda i: -series[i]["uses"][w])
        top = set(ranked[:TOP_TURNOVER])
        turnover.append(None if prev_top is None or not prev_top else round(len(top - prev_top) / max(len(prev_top), 1), 3))
        prev_top = top if top else prev_top
    Rs = [s["R"] for s in series if s["R"] is not None]
    n_utt = sum(totals)
    notes = []
    if n_utt < 50:
        notes.append(f"only {n_utt} utterances: shares are noisy")
    if W < 2 * RECENT_WINDOWS:
        notes.append(f"only {W} windows: stages and half-lives are unreliable")
    if not Rs:
        notes.append("no adopters yet: R undefined")
    man = man or {}
    comp = list(competition or [])
    return {"tick": tick, "window_ticks": window, "windows": [w * window for w in range(W)],
            "labels": [_label(man, w * window, tpd) for w in range(W)],
            "day_bounds": [d * tpd for d in range(1, tick // tpd + 1)],
            "sufficiency": {"n_utterances": n_utt, "n_windows": W, "ok": not notes, "notes": notes},
            "population": {"alive": alive, "births": births, "deaths": deaths, "turnover": turnover,
                           "talk_volume": totals, "n_alive_now": alive[-1] if alive else 0,
                           "mean_R": round(sum(Rs) / len(Rs), 3) if Rs else None},
            "memes": out,
            "competition": comp, "baseline": None, "divergence": None, "fidelity": None, "drift": None,
            "deferred": [k for k in ("competition", "baseline", "divergence", "fidelity", "drift") if not comp or k != "competition"]}


# --------------------------------------------------------------------- injected-meme cohort (registry)
def survival_after_repair(usages: list[dict], repaired_day: int | None, tpd: int, seeds=(),
                          last_tick: int | None = None) -> dict:
    """Use before vs after the day the meme's basis was removed. Rates are per ELAPSED DAY of each window,
    not per day the phrase happened to be said: dividing by the days it was used would score a meme said
    once in a ten-day window exactly as highly as one said daily, which is the opposite of survival.

    An ungrounded meme has no repaired_day: `after` is null by construction, which is exactly what makes it
    the comparison for the grounded ones rather than a missing measurement. A grounded meme that stopped
    dead scores 0.0, which is a measurement and must not be confused with null."""
    empty = {"repaired_day": repaired_day, "uses_before": 0, "uses_after": 0, "rate_before_per_day": None,
             "rate_after_per_day": None, "survival_ratio": None, "users_after": 0, "new_users_after": 0,
             "days_alive_after": 0, "days_before": 0, "days_after": 0}
    if not usages or not repaired_day:
        return dict(empty, uses_before=len(usages), users_after=0)
    cut = (int(repaired_day) - 1) * tpd
    last = max([u["tick"] for u in usages] + ([int(last_tick)] if last_tick is not None else []))
    pre = [u for u in usages if u["tick"] < cut]
    post = [u for u in usages if u["tick"] >= cut]
    days_before = max(1, int(repaired_day) - 1)
    days_after = max(0, last // tpd + 1 - (int(repaired_day) - 1))
    rate = lambda n, d: round(n / d, 3) if d else None            # noqa: E731
    r_pre, r_post = rate(len(pre), days_before), rate(len(post), days_after)
    before_users = {u["speaker"] for u in pre}
    return {"repaired_day": repaired_day, "uses_before": len(pre), "uses_after": len(post),
            "days_before": days_before, "days_after": days_after,
            "rate_before_per_day": r_pre, "rate_after_per_day": r_post,
            "survival_ratio": round(r_post / r_pre, 3) if r_pre and r_post is not None else None,
            "users_after": len({u["speaker"] for u in post}),
            "new_users_after": len({u["speaker"] for u in post} - before_users - set(seeds)),
            "days_alive_after": len({u["tick"] // tpd for u in post})}


def registry_series(run_dir, tick: int | None = None, window: int = 4, rd=None) -> list[dict]:
    """One adoption curve, unique-user count and survival-after-repair figure per registry meme (6c).

    Registry memes are NOT discovered candidates: they are declared, their minority is known, and their
    surface form is known, so they are matched exactly rather than mined. Returns [] when the run declares
    no registry, so a run predating the mechanism is unchanged (R5)."""
    from backend.analysis.battery import registry as REG
    if rd is None:
        from backend.analysis.rundata import RunData
        rd = RunData(Path(run_dir))
    specs = REG.load_registry(rd.cfg, only_enabled=False)
    if not specs:
        return []
    tpd = rd.ticks_per_day
    t = int(tick) if tick is not None else max((u["tick"] for u in rd.utterances), default=0)
    window = max(1, int(window))
    W = t // window + 1
    totals = [0] * W
    for u in rd.utterances:
        if u["tick"] <= t:
            totals[min(u["tick"] // window, W - 1)] += 1
    n_active = len(rd.manifest.get("agents") or {}) or 1
    out = []
    for s in specs:
        us = REG.find_usages(rd, s, t)
        seeds = REG.seed_agents(rd, s)
        row = {"id": s.id, "phrase": s.phrase, "cell": s.cell, "grounding": s.grounding,
               "breadth": s.breadth, "repaired_day": s.repaired_day, "seeds": seeds, "k": len(seeds),
               "uses": [0] * W, "share": [0.0] * W, "unique_users": len({u["speaker"] for u in us}),
               "unique_adopters": len({u["speaker"] for u in us} - set(seeds)),
               "variants": sorted({u["variant"] for u in us}),
               "seed_graph": REG.seed_graph_stats(rd, s, seeds)}
        if us:
            ser = meme_series(us, W, window, totals, tpd, n_active, t, seeds=seeds)
            row.update({k: v for k, v in ser.items() if not k.startswith("_")})
        row["survival"] = survival_after_repair(us, s.repaired_day, tpd, seeds, last_tick=t)
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------ run-level API
def _judge(v: dict | None) -> dict | None:
    if not v:
        return None
    prov = v.get("model") or v.get("provider") or "judge"
    if not v.get("real"):
        prov = "mock"
    return {"provenance": prov, "is_convention": v.get("is_convention"), "gloss": v.get("gloss")}


def _mtime(p: Path):
    try:
        return p.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _build(run_dir: Path, tick, window: int, top: int):
    from backend.analysis import live
    sn = live.snapshot_context(run_dir, tick, top=max(top, 30))
    rd = sn.rd
    names = getattr(rd, "names", {}) or {}
    memes, extra = [], {}
    for r in sn.pool:
        memes.append({"id": r["id"], "phrase": r.get("display") or r["phrase"], "tier": r.get("tier"),
                      "status": r.get("status"), "judge": _judge(r.get("verdict")),
                      "first_use": {k: (r.get("first_use") or {}).get(k) for k in ("tick", "speaker", "speaker_name", "text")},
                      "usages": r["_usages"]})
        extra[r["id"]] = r
    utts = [{"tick": int(u.get("tick") or 0)} for u in rd.utterances]
    n_active = len(sn.roster.get("active") or []) or len(sn.population)
    res = compute(utts, memes, tick=sn.t, window=window, tpd=sn.tpd, n_active=n_active, man=sn.man,
                  competition=_competition(rd, sn.t, max(1, window), sn.t // max(1, window) + 1))
    res["memes"].sort(key=lambda m: (-sum(m["uses"]), m["id"]))
    # the injected cohort is listed separately from the mined pool: these are declared, not discovered,
    # and mixing them into a "top expressions" ranking would hide the meme that failed to spread
    res["registry"] = _registry(run_dir, sn.t, window, rd)
    return res, extra, names


def _registry(run_dir, t: int, window: int, rd) -> list:
    """Never fails a trends request: a run with no `memes` block simply has no registry panel."""
    try:
        return registry_series(run_dir, t, window, rd=rd)
    except Exception:
        return []


def _competition(rd, t: int, window: int, n_windows: int) -> list:
    """Rival names per referent for the competition block. Never fails a trends request: a run with no
    world map, or a resolver that cannot read it, simply leaves the panel empty."""
    try:
        from backend.analysis import places
        resolved = places.resolve_place_references(rd.dir, t, rd=rd)
        return places.competition_series(resolved, [w * window for w in range(max(1, n_windows))], window)
    except Exception:
        return []


def _cached(run_dir, tick, window, top):
    run_dir = Path(run_dir)
    key = (str(run_dir.resolve()), _mtime(run_dir / "trace.jsonl"), tick, int(window), int(top))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    val = _build(run_dir, tick, window, top)
    with _LOCK:
        _CACHE[key] = val
        while len(_CACHE) > 32:
            _CACHE.popitem(last=False)
    return val


def run_trends(run_dir, tick: int | None = None, window: int = 4, top: int = 20) -> dict:
    res, _extra, _names = _cached(run_dir, tick, window, top)
    out = dict(res)
    out["memes"] = [m for m in res["memes"]][:int(top)]
    out["registry"] = list(res.get("registry") or [])      # never truncated: the cohort is the study
    return out


def meme_detail(run_dir, meme_id: str, tick: int | None = None, window: int = 4) -> dict | None:
    res, extra, names = _cached(run_dir, tick, window, 20)
    m = next((x for x in res["memes"] if x["id"] == meme_id), None)
    if m is None:
        return None
    rec = extra[meme_id]
    us = [u for u in rec["_usages"] if u["tick"] <= res["tick"]]
    ad = adoptions(us)
    root = ad[0]["agent"] if ad else None
    return {"meme": m,
            "adoptions": [{"agent": a["agent"], "name": names.get(a["agent"], a["agent"]), "exposure_tick": a["exposure_tick"],
                           "adopt_tick": a["adopt_tick"], "source": a["source"]} for a in ad],
            "tree": {"root": root, "edges": [{"from": a["source"], "to": a["agent"], "tick": a["adopt_tick"]}
                                             for a in ad if a["source"]]},
            "contexts": [{"tick": u["tick"], "speaker": names.get(u["speaker"], u["speaker"]), "text": u.get("text")}
                         for u in us[:40]]}
