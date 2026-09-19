"""Target expressions X / W / Y (ONTOLOGY_V3 §5.4). OBSERVER ONLY.

Selected once, on the TRUNK at C3, before any branch runs, and frozen in `probes/targets.json` (its sha goes to
the design's prereg log). Every branch of a seed probes the same X, W and Y.

X = the highest-scoring n-gram (2-3 words) over utterances AND binder entries up to the checkpoint that has
  - >= 3 tokens linked to K1 jobs (utterance provenance `referent_event_ids` or the entry's job) ,
  - >= 2 producers,
  - no occurrence in world text (surfaces, menu, oddities, panel code, tally/onboarding templates, binder stamps),
    the population lexicon or battery wording,
  - C0 production < 10% (share of C0 P descriptions containing it; unchecked if no C0 responses exist).
Y = the same rule for K2. Fallbacks X_desc / Y_desc: the most frequent K1 / K2 content bigram in the binder,
flagged descriptive. W = the panel code (world-anchored; flagged in_world_text, never emergence).

Score (MVP deviation from "v2 emergence score"): K-linked tokens x producers, ties broken by total uses, then
lexical order. The v2 emergence score can be substituted in `_score` once emergence.py counts binder reads.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from backend.analysis.battery.items import load_items
from backend.analysis.battery.nonce import make_nonce

_WORD = re.compile(r"[a-z0-9']+")
STOP = set("""a an the and or but if then so of to in on at for with by from as is are was were be been being it its
this that these those i you he she they we me him her them my your his their our do does did done have has had not
no yes just very really about into over after before up down out there here what which who when where why how all
any some can could would should will shall may might must also too than again it's that's i'm you're we're don't
didn't one two get got like think know yeah oh ok okay""".split())


def _toks(text: str) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def ngrams(text: str, ns=(2, 3)) -> set[str]:
    t = _toks(text)
    out = set()
    for n in ns:
        for i in range(len(t) - n + 1):
            g = t[i:i + n]
            if g[0] in STOP or g[-1] in STOP or sum(w not in STOP for w in g) < 2:
                continue
            out.add(" ".join(g))
    return out


def world_text(cfg: dict) -> list[str]:
    """Every world-provided wording the observer must never count as agent coinage (§5.4)."""
    pack = (cfg.get("workshop") or {}).get("content", "laser_alpha")
    out = []
    try:
        import importlib
        mod = importlib.import_module(f"backend.simulation.content.{pack}")
        for name in dir(mod):
            v = getattr(mod, name)
            if name.isupper():
                out += _strings(v)
    except Exception:   # noqa: BLE001 - pack absent: menu + panel code only
        pass
    from backend.analysis.battery.gt import MENU
    out += [t for _, t in MENU.get(pack, [])]
    out.append(str(((cfg.get("workshop") or {}).get("panel_code") or {}).get("text", "F4")))
    try:
        from backend.simulation import records as R
        out += [str(getattr(R, n)) for n in ("HEADER", "EMPTY_TEXT") if hasattr(R, n)]
    except Exception:   # noqa: BLE001
        pass
    return out


def _strings(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [s for x in v.values() for s in _strings(x)]
    if isinstance(v, (list, tuple)):
        return [s for x in v for s in _strings(x)]
    return []


def excluded_ngrams(cfg: dict, extra: list[str] = ()) -> set[str]:
    texts = world_text(cfg) + list(extra)
    pack = (cfg.get("workshop") or {}).get("content", "laser_alpha")
    for m in ("M1", "M2"):
        try:
            texts += [i["text"] for i in load_items(pack, m)]
        except NotImplementedError:
            pass
    out = set()
    for t in texts:
        out |= ngrams(t, (2, 3))
    try:
        from backend.simulation import lexicon as LX
        lex = LX.build(cfg) if hasattr(LX, "build") else None
        for b in (lex or {}).get("bigrams", []):
            out.add(b)
    except Exception:   # noqa: BLE001
        pass
    return out


def _job_classes(rd) -> dict[str, str]:
    """job id -> class under the regime active for that job (hidden job_truth trace)."""
    out = {}
    for r in rd.of("job_truth"):
        k = r.get("klass") or r.get("class")
        if isinstance(k, dict):
            k = k.get(r.get("regime") or "A")
        if r.get("job") and k:
            out[r["job"]] = str(k)
    return out


def _utterance_jobs(rd, jobs: dict) -> dict[str, set]:
    """utterance id -> job ids it is about (v2 provenance; jobs as events once provenance.py knows them)."""
    try:
        from backend.analysis.provenance import compute
        prov = compute(rd)
    except Exception:   # noqa: BLE001
        prov = {}
    out = {}
    for uid, p in prov.items():
        js = {e for e in p.get("referent_event_ids") or [] if e in jobs}
        if js:
            out[uid] = js
    return out


def occurrences(run_dir: Path, ckpt: str) -> tuple[list[dict], dict]:
    """[{text, producer, jobs, kind: utterance|binder}] up to the end of checkpoint day, and the job classes."""
    from backend.analysis.rundata import RunData
    rd = RunData(run_dir)
    day = int(ckpt[1:]) if ckpt[1:].isdigit() else int(rd.cfg.get("simulation_days", 1))
    tmax = day * int(rd.manifest.get("ticks_per_day", 60))
    jobs = _job_classes(rd)
    uj = _utterance_jobs(rd, jobs)
    occ = [{"text": u["text"], "producer": u.get("speaker"), "jobs": sorted(uj.get(u["id"], ())), "kind": "utterance"}
           for u in rd.utterances if int(u["tick"]) < tmax]
    bpath = Path(run_dir) / "checkpoints" / ckpt / "binder.json"
    binder = json.loads(bpath.read_text()) if bpath.exists() else {}
    pages = [binder] + list(binder.get("archived") or [])
    for b in pages:
        for e in b.get("log") or []:
            occ.append({"text": e.get("text", ""), "producer": e.get("author"), "jobs": [e["job_id"]] if e.get("job_id") else [],
                        "kind": "binder"})
        for r in b.get("front") or []:
            occ.append({"text": r.get("text", ""), "producer": r.get("author"), "jobs": [], "kind": "binder"})
    return occ, {"jobs": jobs, "cfg": rd.cfg}


def _stats(occ: list[dict], jobs: dict, cls: str, excl: set) -> dict[str, dict]:
    st: dict[str, dict] = {}
    for o in occ:
        linked = any(jobs.get(j) == cls for j in o["jobs"])
        for g in ngrams(o["text"]):
            if g in excl:
                continue
            s = st.setdefault(g, {"tokens": 0, "uses": 0, "producers": set(), "binder_tokens": 0})
            s["uses"] += 1
            if linked:
                s["tokens"] += 1
                s["producers"].add(o["producer"])
                s["binder_tokens"] += int(o["kind"] == "binder")
    return st


def _score(s: dict) -> float:
    return s["tokens"] * len(s["producers"])


def _pick(st: dict, prior, min_tokens: int, min_producers: int, max_prior: float) -> tuple[str | None, list]:
    ranked = sorted(st.items(), key=lambda kv: (-_score(kv[1]), -kv[1]["uses"], kv[0]))
    cands = []
    for g, s in ranked:
        if s["tokens"] < min_tokens or len(s["producers"]) < min_producers:
            continue
        p = prior(g)
        row = {"expr": g, "tokens": s["tokens"], "binder_tokens": s["binder_tokens"], "producers": len(s["producers"]),
               "uses": s["uses"], "score": _score(s), "c0_production": p}
        cands.append(row)
        if p is None or p < max_prior:
            return g, cands
    return None, cands


def _desc(occ: list[dict], jobs: dict, cls: str, excl: set) -> str | None:
    cnt: dict[str, int] = {}
    for o in occ:
        if o["kind"] == "binder" and any(jobs.get(j) == cls for j in o["jobs"]):
            for g in ngrams(o["text"], (2,)):
                if g not in excl:
                    cnt[g] = cnt.get(g, 0) + 1
    return min(cnt, key=lambda g: (-cnt[g], g)) if cnt else None


def select_targets(trunk_run_dir, ckpt: str = "C3", *, c0_responses: list[dict] | None = None,
                   out_path=None, min_tokens: int = 3, min_producers: int = 2, max_prior: float = 0.10) -> dict:
    """Pre-registered target selection on the trunk at `ckpt`; writes and returns targets.json."""
    trunk = Path(trunk_run_dir)
    occ, ctx = occurrences(trunk, ckpt)
    cfg, jobs = ctx["cfg"], ctx["jobs"]
    excl = excluded_ngrams(cfg)
    if c0_responses is None:
        p = trunk / "probes" / "C0" / "responses.jsonl"
        c0_responses = [json.loads(l) for l in open(p)] if p.exists() else None
    if c0_responses is not None:
        from backend.analysis.battery.fresh import prior_production
        prior = lambda g: round(prior_production(c0_responses, g), 4)  # noqa: E731
    else:
        prior = lambda g: None  # noqa: E731
    x, xc = _pick(_stats(occ, jobs, "K1", excl), prior, min_tokens, min_producers, max_prior)
    y, yc = _pick(_stats(occ, jobs, "K2", excl), prior, min_tokens, min_producers, max_prior)
    code = str(((cfg.get("workshop") or {}).get("panel_code") or {}).get("text", "F4"))
    x_desc, y_desc = _desc(occ, jobs, "K1", excl), _desc(occ, jobs, "K2", excl)
    out = {"trunk": str(trunk), "ckpt": ckpt, "X": x, "Y": y, "W": code, "W_in_world_text": True,
           "X_desc": x_desc, "Y_desc": y_desc, "X_kind": "coined" if x else ("descriptive" if x_desc else None),
           "Y_kind": "coined" if y else ("descriptive" if y_desc else None),
           "c0_checked": c0_responses is not None, "candidates": {"X": xc[:10], "Y": yc[:10]},
           "rule": {"min_tokens": min_tokens, "min_producers": min_producers, "max_prior": max_prior,
                    "score": "k_tokens*producers"}}
    out["NONCE"] = make_nonce(x or x_desc or code, cfg.get("seed", 0))
    out_path = Path(out_path) if out_path else trunk / "probes" / "targets.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(out, indent=1, sort_keys=True)
    out_path.write_text(body)
    out["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return out


def probe_targets(t: dict) -> dict:
    """The cue set a branch probes: X falls back to X_desc (descriptive), Y to Y_desc."""
    return {"X": t.get("X") or t.get("X_desc"), "W": t.get("W") or "F4", "Y": t.get("Y") or t.get("Y_desc"),
            "NONCE": t.get("NONCE")}
