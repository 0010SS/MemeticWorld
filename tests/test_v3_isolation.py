"""v3 step 8: probe battery isolation (ONTOLOGY_V3 §5.2-5.6). Mock backend only, no engine run: a synthetic
run directory with a C4 checkpoint in the checkpoint.py layout."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
import stat
from pathlib import Path

import pytest
import yaml

from backend.config import load_config

ROOT = Path(__file__).resolve().parents[1]
START = "2026-09-14T07:30:00"


def _cfg():
    return load_config("configs/default.yaml", {
        "seed": 13, "llm": {"backend": "mock"},
        "probe": {"backend": "mock", "model": "haiku", "workers": 2, "k": 6, "tau": 0.7},
        "records": {"enabled": True, "display": "current"},
        "regimes": {"mapping": "auto", "schedule": [{"day": 1, "regime": "A"}, {"day": 5, "regime": "B"}]},
        "retrieval": {"source_weights": {"seed": 0.5, "ambient": 0.5}},
    })


MEMS = [
    ("perception", ["j02.1"], "Dev's first sheet came out with the left side stuck and brown edges; Dev cleaned the lens."),
    ("conversation", [], "Maya said the lens needs wiping when the left side won't cut through."),
    ("seed", [], "The co-op's laser cutter is in the Makerspace."),
    ("record", [], "The binder by the laser says tea edges on the left means wipe the lens."),
    ("ambient", [], "The Makerspace was quiet in the afternoon."),
    ("perception", ["j03.2"], "Near the right edge the lines came out doubled; tightening the belt fixed it."),
]


def _write_ckpt(run: Path, cfg: dict, ids: list[str]):
    from backend.experiment.checkpoint import checksums
    from backend.llm.embeddings import make_embedder
    from backend.memory.store import MemoryMeta, MemoryStream
    embed = make_embedder(cfg.get("embedding"))
    ck = run / "checkpoints" / "C4"
    meta = {}
    for j, aid in enumerate(ids):
        ms = MemoryStream(aid)
        for i, (src, evs, text) in enumerate(MEMS):
            n = ms.add("event", dt.datetime(2026, 9, 15 + i % 3, 10, 0), aid, "saw", "laser", text, {"laser"},
                       3 + i % 4, embed(text))
            meta[n.node_id] = dataclasses.asdict(MemoryMeta(aid, src, originating_event_ids=list(evs)))
        ms.save_ga(ck / "agents" / aid / "associative_memory")
        ms.save_ga(run / "agents_final" / aid / "associative_memory")
    (ck / "memory_meta.json").write_text(json.dumps(meta, sort_keys=True))
    (run / "memory_meta.json").write_text(json.dumps(meta, sort_keys=True))
    (ck / "agent_state.json").write_text(json.dumps(
        {aid: {"cohort": "W1" if i == len(ids) - 1 else "founder", "arrival_day": 4 if i == len(ids) - 1 else 1}
         for i, aid in enumerate(ids)}, sort_keys=True))
    binder = {"binder_id": "binder-1", "created_tick": 0, "authority": "members",
              "front": [{"rev_id": "r1", "author": ids[0], "author_name": "Maya", "tick": 100,
                         "text": "Tea edges on the left: wipe the lens first.", "revision_of": None}],
              "log": [{"entry_id": f"e{i}", "author": ids[i % 2], "author_name": "X", "tick": 90 + i,
                       "text": t, "context": "job", "job_id": f"j0{2 + i}.1"}
                      for i, t in enumerate(["tea edges again, cleaned lens, fine", "saw tea edges on the left, lens",
                                             "tea edges and F4 -> lens"])],
              "archived": [], "receipts": {}, "seq": {"rev": 1, "entry": 3, "binder": 1}}
    (ck / "binder.json").write_text(json.dumps(binder, sort_keys=True))
    (ck / "roster.json").write_text(json.dumps({"active": ids}))
    (ck / "world_state.json").write_text(json.dumps(
        {"day": 4, "tick": 239, "time": "2026-09-17T22:15:00", "regime": "A", "mapping": "M1"}, sort_keys=True))
    sums = checksums(ck)
    (ck / "CHECKSUMS").write_text("".join(f"{h}  {p}\n" for p, h in sorted(sums.items())))
    for p in ck.rglob("*"):
        if p.is_file():
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


@pytest.fixture()
def run(tmp_path):
    from backend.analysis.rundata import simulated_profiles
    cfg = _cfg()
    run = tmp_path / "run"
    run.mkdir()
    ids = sorted(simulated_profiles(cfg))[:3]
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    (run / "manifest.json").write_text(json.dumps({"start": START, "ticks_per_day": 60, "tick_minutes": 15,
                                                   "agents": {}, "status": "finished"}))
    (run / "llm_calls.jsonl").write_text(json.dumps({"key": "d1t010:x|abc|0", "scope": "d1t010:x", "purpose": "react_decision",
                                                     "prompt": "p", "response": "r"}) + "\n")
    trace = [{"type": "job_truth", "tick": 90 + i, "job": f"j0{2 + i}.1", "klass": "K1", "regime": "A"} for i in range(3)]
    (run / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    _write_ckpt(run, cfg, ids)
    return run, ids


def _mock():
    from backend.llm.mock import MockBackend
    return MockBackend(seed=0)


TARGETS = {"X": "tea edges", "W": "F4", "Y": "doubled lines"}


def test_run_files_unchanged_and_separate_cache(run):
    from backend.analysis.battery.runner import run_battery, run_checksums
    rd, ids = run
    before = run_checksums(rd)
    s = run_battery(rd, "C4", plan="full", targets=TARGETS, backend=_mock())
    assert s["valid"] and s["n_errors"] == 0 and s["layout"] == "checkpoint"
    assert run_checksums(rd) == before                     # nothing outside probes/ changed
    assert not (rd / "probes" / "C4" / "work").exists()     # the copy is discarded
    calls = [json.loads(l) for l in open(rd / "probes" / "C4" / "llm_calls.jsonl")]
    assert calls and all(c["scope"].startswith("probe:C4:") for c in calls)
    sim = [json.loads(l) for l in open(rd / "llm_calls.jsonl")]
    assert not any(c["scope"].split(":")[0] in ("probe", "fresh", "dcf", "retest", "synth", "code") for c in sim)
    assert not any(c.get("purpose", "").startswith("probe_") for c in sim)
    resp = [json.loads(l) for l in open(rd / "probes" / "C4" / "responses.jsonl")]
    per = {}
    for r in resp:
        per[r["agent"]] = per.get(r["agent"], 0) + 1
    founders, newcomer = ids[:-1], ids[-1]
    for a in founders:
        assert per[a] == 56                                  # §5.3: full battery = 56 calls per agent
    assert per[newcomer] == 56 + 10                          # + source-ablated P for newcomers
    assert per["record_only"] == 9 + 15
    p = [r for r in resp if r["form"] == "P"]
    assert all(r["action"] in {"rerun", "lens", "dry", "belt", "slow", "stop"} for r in p)
    assert all(r["gt"] for r in p) and any(r["hit"] for r in p)
    assert {r["regime"] for r in resp} == {"A"} and {r["mapping"] for r in resp} == {"M1"}


def test_prompts_hide_hidden_state(run):
    from backend.analysis.battery.runner import run_battery
    rd, _ = run
    run_battery(rd, "C4", plan="full", targets=TARGETS, backend=_mock())
    bad = re.compile(r"\b(K[0-3]|LENS|DAMP|BELT|AIR|WARP|M1|M2|regime|mapping|trunk|wipe4|keep_shift|noshift)\b|j\d\d\.\d")
    for l in open(rd / "probes" / "C4" / "llm_calls.jsonl"):
        c = json.loads(l)
        assert not bad.search(c["prompt"]), bad.search(c["prompt"]).group(0)
    forbidden = re.compile(r"invent|coin|\bname|label|term|slang|meme|nickname|shorthand|rule|\btip|procedure|jot", re.I)
    for f in ("probe_act_v1.txt", "probe_apply_v1.txt", "probe_note_v1.txt"):
        assert not forbidden.search((ROOT / "backend" / "prompts" / f).read_text()), f


def test_cached_by_digest_and_agents_final_fallback(run):
    from backend.analysis.battery.runner import run_battery, run_checksums
    rd, ids = run
    s1 = run_battery(rd, "C4", targets=TARGETS, backend=_mock())
    n = sum(1 for _ in open(rd / "probes" / "C4" / "llm_calls.jsonl"))
    s2 = run_battery(rd, "C4", targets=TARGETS, backend=_mock())
    assert s2.get("cached") and s2["digest"] == s1["digest"]
    assert sum(1 for _ in open(rd / "probes" / "C4" / "llm_calls.jsonl")) == n
    before = run_checksums(rd)
    s3 = run_battery(rd, "final", modes=["memory_only"], backend=_mock())
    assert s3["layout"] == "agents_final" and s3["valid"] and s3["n_calls"] == 29 * len(ids)
    assert run_checksums(rd) == before


def test_source_weight_parity(run):
    """Probe copies score memories exactly like the simulation (D50 source weights via memory_meta)."""
    from backend.analysis.battery.runner import _src_files, build_agents, prepare_checkpoint
    from backend.analysis.rundata import RunData
    from backend.memory.retrieval import retrieve
    from backend.memory.store import MemoryMeta, SimMemoryMeta
    from backend.simulation.rngs import seed_rng
    from backend import ga_compat
    from backend.llm.embeddings import make_embedder
    rd, ids = run
    R = RunData(rd)
    ga_compat.load(None, make_embedder(R.cfg.get("embedding")))
    cd = prepare_checkpoint(rd, "C4", rd / "probes" / "C4")
    when = dt.datetime.fromisoformat("2026-09-17T22:15:00")
    a = build_agents(R.cfg, R.manifest, cd, when)[ids[0]]
    sim_meta = SimMemoryMeta()
    for nid, m in json.load(open(rd / "memory_meta.json")).items():
        sim_meta.set(nid, MemoryMeta(**m))
    focal = ["left side brown edges", "the co-op's laser cutter"]
    r1 = retrieve(a, focal, k=6, rng=seed_rng(1), touch=False)
    probe_scores = {f: r.scores for f, r in r1.items()}
    a.ctx.meta = sim_meta
    r2 = retrieve(a, focal, k=6, rng=seed_rng(1), touch=False)
    assert probe_scores == {f: r.scores for f, r in r2.items()}
    a.ctx.meta = None                                         # without meta the weights would not apply
    r3 = retrieve(a, focal, k=6, rng=seed_rng(1), touch=False)
    assert probe_scores != {f: r.scores for f, r in r3.items()}
    assert _src_files(rd, "C4")[0] == "checkpoint"


def test_import_guard():
    """Simulation-side code never imports the observer."""
    pat = re.compile(r"^\s*(from|import)\s+backend\.analysis", re.M)
    for sub in ("simulation", "agents", "memory", "modules", "llm"):
        for p in (ROOT / "backend" / sub).rglob("*.py"):
            assert not pat.search(p.read_text()), p


def test_gt_and_scoring():
    from backend.analysis.battery.gt import gt, score
    assert gt({"type": "K1c"}, "A", "M1") == "lens" and gt({"type": "K1c"}, "B", "M1") == "dry"
    assert gt({"type": "K1c"}, "A", "M2") == "dry" and gt({"type": "K1c"}, "B", "M2") == "lens"
    assert gt({"type": "K2"}, "B", "M1") == "belt" and gt({"type": "K3"}, "A", "M1") == "dry"
    assert gt({"type": "K0"}, "A", "M1") == "rerun"
    s = score("lens", {"type": "K1c"}, "B", "M1")
    assert s["category"] == "old" and s["correct"] is False
    assert score("dry", {"type": "K1c"}, "B", "M1")["category"] == "new"
    assert score("slow", {"type": "K1c"}, "B", "M1")["hedge"]
    assert score("lens", {"type": "CUE"}, "A", "M1")["correct"] is None


def test_gt_parity_with_world():
    W = pytest.importorskip("backend.simulation.workshop")
    from backend.analysis.battery import gt as GT
    for cause in ("LENS", "DAMP", "BELT", None):
        for act in ("rerun", "lens", "dry", "belt", "slow", "stop"):
            assert abs(GT.outcome_p(act, cause) - GT._local_p(act, cause)) < 1e-9, (act, cause)
    for cls in ("K0", "K1", "K1c", "K1a", "K2", "K3", "CUE"):
        for reg in ("A", "B"):
            for m in ("M1", "M2"):
                assert GT.local_gt(cls, reg, m) == W.gt(GT.base_class(cls), reg, m) == GT.gt(cls, reg, m)


def test_items_and_hygiene():
    from backend.analysis.battery.items import load_items, select
    for m in ("M1", "M2"):
        it = load_items("laser_alpha", m)
        assert len(it) == 21 and len(select(it, "A13")) == 13
        assert [len(select(it, t)) for t in ("K1c", "K1a", "K3", "K2", "K0", "CUE")] == [8, 2, 3, 4, 2, 2]
        assert not any("F4" in i["text"] or "panel" in i["text"].lower() for i in it)
    LA = pytest.importorskip("backend.simulation.content.laser_alpha")
    from backend.simulation.structures import content_bigrams
    world = [x for v in LA.SURFACES.values() for x in ([y for vv in v.values() for y in vv] if isinstance(v, dict) else v)]
    world += list(LA.ACTIONS.values()) + list(getattr(LA, "ODDITIES", []))
    wb = set().union(*(content_bigrams(t) for t in world))
    from backend.analysis.battery.items import heldout_texts
    for t in heldout_texts("laser_alpha"):
        assert not (content_bigrams(t) & wb), t


def test_nonce():
    from backend.analysis.battery.nonce import carrier, make_nonce, syllables
    n = make_nonce("tea edges", 13)
    assert n == make_nonce("tea edges", 13) and len(n.split()) == 2
    assert [syllables(w) for w in n.split()] == [syllables("tea"), syllables("edges")]
    assert carrier(None) == "Heads up about the laser today." and carrier("F4").startswith("F4 again")


def test_select_targets(run):
    from backend.analysis.battery.targets import select_targets
    rd, _ = run
    t = select_targets(rd, "C4", c0_responses=[{"form": "P", "describe": "the laser is broken"}] * 20)
    assert t["X"] == "tea edges" and t["W"] == "F4" and t["X_kind"] == "coined"
    assert (rd / "probes" / "targets.json").exists() and t["NONCE"]
    t2 = select_targets(rd, "C4", c0_responses=[{"form": "P", "describe": "tea edges"}] * 5,
                        out_path=rd / "probes" / "t2.json")
    assert t2["X"] is None and t2["X_desc"]                 # prior production >= 10% disqualifies


def test_fresh_baseline(tmp_path):
    from backend.analysis.battery.fresh import run_fresh
    from backend.analysis.rundata import simulated_profiles
    cfg = _cfg()
    profs = dict(list(sorted(simulated_profiles(cfg).items()))[:2])
    s = run_fresh(cfg, tmp_path / "probes", profiles=profs, backend=_mock())
    assert s["n_calls"] == 2 * (29 + 3 * 3) + 21 * 3 and s["n_errors"] == 0
    calls = [json.loads(l) for l in open(tmp_path / "probes" / "C0" / "llm_calls.jsonl")]
    assert all(c["scope"].startswith("fresh:C0:") for c in calls)


def test_responses_readable_by_metrics(run):
    """The observer metrics' normaliser (analysis/v3common.py) reads the runner's rows."""
    V = pytest.importorskip("backend.analysis.v3common")
    from backend.analysis.battery.runner import run_battery
    rd, _ = run
    run_battery(rd, "C4", plan="full", targets=TARGETS, backend=_mock())
    rows = V.probe_responses(rd, "C4")
    forms = {r["form"] for r in rows}
    assert {"P", "P-sit", "P-abl", "A", "N"} <= forms
    a = [r for r in rows if r["form"] == "A"]
    assert len(a) == 13 * 3 * 3 and all(r["fit"] in (0.0, 0.5, 1.0) for r in a)
    assert {r["cue"] for r in rows if r["form"] == "N"} == {"X", "W", "Y", "NONCE", "NONE"}
