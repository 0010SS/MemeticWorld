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

Deferred (keys present, empty/null): competition per referent, neutral-copying baseline, group divergence,
copy fidelity, meaning drift.
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


def adoptions(usages: list[dict]) -> list[dict]:
    """[{agent, adopt_tick, exposure_tick, source, conversation_id}] in adoption order."""
    us = sorted(usages, key=lambda u: (u["tick"], u.get("idx", 0)))
    first, heard = {}, defaultdict(list)
    for u in us:
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


def exposure_stats(usages: list[dict]) -> tuple[int, int]:
    """(n_exposed, n_adopted): exposed before any own use (or never used); adopted = later secondary adoption."""
    ad = {r["agent"]: r for r in adoptions(usages)}
    exposed = set()
    for u in usages:
        for l in u.get("listeners") or []:
            if l == u["speaker"]:
                continue
            if l not in ad or u["tick"] < ad[l]["adopt_tick"]:
                exposed.add(l)
    n_ad = sum(1 for a in exposed if a in ad and ad[a]["source"] is not None)
    return len(exposed), n_ad


def meme_series(usages: list[dict], n_windows: int, window: int, totals: list[int], tpd: int,
                n_active: int, t: int) -> dict:
    W = n_windows
    wi = lambda tick: min(max(int(tick) // window, 0), W - 1)
    seen, uses = set(), [0] * W
    for u in usages:
        if u.get("utterance_id") in seen:
            continue
        seen.add(u.get("utterance_id"))
        uses[wi(u["tick"])] += 1
    share = [round(uses[i] / totals[i], 4) if totals[i] else 0.0 for i in range(W)]
    ad = adoptions(usages)
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
    n_exp, n_ado = exposure_stats(usages)
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
            man: dict | None = None) -> dict:
    """Pure core. memes: [{id, phrase, usages, ...extra fields copied through}]."""
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
    return {"tick": tick, "window_ticks": window, "windows": [w * window for w in range(W)],
            "labels": [_label(man, w * window, tpd) for w in range(W)],
            "day_bounds": [d * tpd for d in range(1, tick // tpd + 1)],
            "sufficiency": {"n_utterances": n_utt, "n_windows": W, "ok": not notes, "notes": notes},
            "population": {"alive": alive, "births": births, "deaths": deaths, "turnover": turnover,
                           "talk_volume": totals, "n_alive_now": alive[-1] if alive else 0,
                           "mean_R": round(sum(Rs) / len(Rs), 3) if Rs else None},
            "memes": out,
            "competition": [], "baseline": None, "divergence": None, "fidelity": None, "drift": None,
            "deferred": ["competition", "baseline", "divergence", "fidelity", "drift"]}


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
    res = compute(utts, memes, tick=sn.t, window=window, tpd=sn.tpd, n_active=n_active, man=sn.man)
    res["memes"].sort(key=lambda m: (-sum(m["uses"]), m["id"]))
    return res, extra, names


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
