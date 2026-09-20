"""The graded meme battery (Part 6): registry, isolation, boundary, foils, cohorts, cost.

No engine run and no real model: a synthetic run directory with a C4 checkpoint in the checkpoint.py
layout, and a fake backend that answers the two probe forms deterministically.
"""
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

REGISTRY = [
    {"id": "tray_washer", "phrase": "blue-tray", "grounding": "grounded", "breadth": "broad",
     "habit": 'says "blue-tray" about things that look fine and are not',
     "seeds": {"k": 2, "strategy": "spread"},
     "incident": {"location": "Dining Hall", "arena": "Main Floor", "days": [1, 2], "repaired_day": 3,
                  "prob_per_meal": 0.6},
     "probe_gradient": {"literal": "The trays come out of the washer looking clean but feel greasy.",
                        "near": "The cups on the rack look dry and leave a film on your hand.",
                        "mid": "A reading list looks complete and is missing half the chapters.",
                        "far": "Someone sounds certain and has not checked anything."},
     "foils": ["The trays come out visibly dirty and everyone can see it.",
               "The washer is loud but the trays are spotless."]},
    {"id": "hood_sash", "phrase": "hood-three", "grounding": "grounded", "breadth": "narrow",
     "habit": 'says "hood-three" about the third fume hood',
     "seeds": {"k": 2, "strategy": "spread"},
     "incident": {"location": "Lab", "arena": "Wet Bench", "days": [1, 2], "repaired_day": 3},
     "probe_gradient": {"literal": "The third hood reads closed and does not seal.",
                        "near": "Another hood on the same bench reads closed and does not seal.",
                        "mid": "A freezer door reads shut and is not.",
                        "far": "A form says submitted and was never sent."},
     "foils": ["The third hood is closed and seals properly.",
               "The bench light above the third hood is out."]},
    {"id": "running_warm", "phrase": "running warm", "grounding": "ungrounded", "breadth": "broad",
     "habit": 'says "running warm" about anything overcommitted',
     "seeds": {"k": 2, "strategy": "spread"},
     "probe_gradient": {"literal": "Someone has taken on more shifts than they can cover.",
                        "near": "A group project is two weeks behind schedule.",
                        "mid": "A friend keeps saying yes to things and cancelling.",
                        "far": "The bus timetable no longer matches when the buses come."},
     "foils": ["Someone finished early and has nothing left to do.",
               "The radiator in the lounge is genuinely hot."]},
]


def _cfg(registry=None, battery=None):
    return load_config("configs/default.yaml", {
        "seed": 13, "llm": {"backend": "mock"},
        "probe": {"backend": "mock", "model": "haiku", "workers": 2, "k": 6, "tau": 0.7},
        "analysis": {"judge": {"provider": "mock", "model": "mock", "prompt_version": "v1"}},
        "memes": {"enabled": True, "registry": REGISTRY if registry is None else registry,
                  "battery": {"enabled": True, "sample": {"per_cohort": 2, "max_agents": 4},
                              "foils_per_meme": 2, "workers": 2, **(battery or {})}},
    })


MEMS = [
    ("perception", "The trays looked clean but the tables were sticky at lunch."),
    ("conversation", "Priya said the washer had been off for two days."),
    ("ambient", "The dining hall was busy at noon."),
]


def _write_ckpt(run: Path, cfg: dict, ids: list[str]):
    from backend.experiment.checkpoint import checksums
    from backend.llm.embeddings import make_embedder
    from backend.memory.store import MemoryMeta, MemoryStream
    embed = make_embedder(cfg.get("embedding"))
    ck = run / "checkpoints" / "C4"
    meta = {}
    for aid in ids:
        ms = MemoryStream(aid)
        for i, (src, text) in enumerate(MEMS):
            n = ms.add("event", dt.datetime(2026, 9, 15 + i % 3, 10, 0), aid, "saw", "hall", text, {"hall"},
                       3 + i, embed(text))
            meta[n.node_id] = dataclasses.asdict(MemoryMeta(aid, src))
        ms.save_ga(ck / "agents" / aid / "associative_memory")
        ms.save_ga(run / "agents_final" / aid / "associative_memory")
    (ck / "memory_meta.json").write_text(json.dumps(meta, sort_keys=True))
    (run / "memory_meta.json").write_text(json.dumps(meta, sort_keys=True))
    (ck / "agent_state.json").write_text(json.dumps({aid: {"cohort": "founder"} for aid in ids}, sort_keys=True))
    (ck / "binder.json").write_text(json.dumps({"enabled": False}))
    (ck / "roster.json").write_text(json.dumps({"active": ids}))
    (ck / "world_state.json").write_text(json.dumps(
        {"day": 4, "tick": 239, "time": "2026-09-17T22:15:00"}, sort_keys=True))
    sums = checksums(ck)
    (ck / "CHECKSUMS").write_text("".join(f"{h}  {p}\n" for p, h in sorted(sums.items())))
    for p in ck.rglob("*"):
        if p.is_file():
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


# each meme gets its own disjoint minority (the engine refuses overlapping ones), plus one exposed
# adopter and one unexposed agent who never says anything
SEEDS = {"tray_washer": (0, 1), "hood_sash": (2, 3), "running_warm": (4, 5)}
ADOPTER, OUTSIDER = 6, 7
EXPOSED = (0, 1, 2, 6)                      # who was in the Dining Hall while the incident was running


def _trace(ids: list[str], tpd: int) -> list[dict]:
    """The tray seeds say their phrase and one exposed agent picks it up; the other two memes are said by a
    seed and by nobody else. Movement is what makes an agent exposed."""
    s0, s1 = ids[0], ids[1]
    rows = []
    for i, aid in enumerate(ids):
        rows.append({"type": "move", "tick": 4, "agent": aid, "frm": ["Dorm", "Room"],
                     "to": ["Dining Hall" if i in EXPOSED else "Library", "Main Floor"], "path": [],
                     "activity": "eating", "forced_by_event": None})
    # an utterance's own location counts as presence too, so the unexposed agents talk somewhere else
    say = lambda t, spk, ls, cid, uid, text, loc="Dining Hall": rows.append(
        {"type": "utterance", "tick": t, "id": uid, "conversation_id": cid, "idx": 0, "speaker": spk,
         "listeners": ls, "text": text, "location": loc, "arena": "Main Floor",
         "retrieved": [], "retrieved_event_ids": [], "source": "chat", "context": ""})
    ad, out = ids[ADOPTER], ids[OUTSIDER]
    say(5, s0, [ad, out], "cv1", "u1", "That tray was blue-tray if you ask me.")
    say(6, s1, [ad], "cv2", "u2", "Classic blue tray, the whole rack.")
    say(20, ad, [out], "cv3", "u3", "Honestly blue-trays, all of it.")       # a real adopter, variant form
    say(2 * tpd + 5, ad, [out], "cv4", "u4", "Still blue-tray around here.")  # after the repair day
    say(5, ids[2], [ad], "cv5", "u5", "hood-three again this morning.")
    say(5, ids[4], [out], "cv6", "u6", "Everyone is running warm this week.", "Library")
    return rows


@pytest.fixture()
def run(tmp_path):
    from backend.analysis.rundata import simulated_profiles
    cfg = _cfg()
    run = tmp_path / "run"
    run.mkdir()
    ids = sorted(simulated_profiles({**cfg, "memes": {"enabled": False, "registry": []}}))[:8]
    # explicit, DISJOINT seed ids, as the real study config uses them: matching minorities on graph
    # position (R6) is not something a sampling strategy can guarantee, and an agent carrying two phrases
    # would make the two memes compete inside one speaker
    cfg["memes"]["registry"] = [dict(e, seeds=dict(e["seeds"], agents=[ids[i] for i in SEEDS[e["id"]]]))
                                for e in cfg["memes"]["registry"]]
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    (run / "manifest.json").write_text(json.dumps({
        "start": START, "ticks_per_day": 60, "tick_minutes": 15, "status": "finished",
        "agents": {aid: {"name": f"Agent {i}", "habits": [], "routine": [], "relationships": {}}
                   for i, aid in enumerate(ids)},
        # the engine's record of what it seeded: two seeds per meme, matched k (R6)
        "memes": {m: {"seeds": [ids[i] for i in ix]} for m, ix in SEEDS.items()}}))
    (run / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _trace(ids, 60)))
    _write_ckpt(run, cfg, ids)
    return run, ids


class _Fake:
    """Answers the two graded forms as a model would; 'fits' shrinks with the gradient so the boundary is
    a known quantity, and foils are always rejected so the false-positive rate is a known 0."""
    name = "fake"
    model = "fake"

    def generate(self, prompt, system, max_tokens, temperature):
        from backend.llm.client import current_purpose
        if current_purpose() == "probe_apply":
            head = prompt.split("For every number")[0]
            nums = re.findall(r"^(\d+)\. (.+)$", head, re.M)
            out = {}
            for n, text in nums:
                fit = "fits"
                if "spotless" in text or "seals properly" in text or "visibly dirty" in text \
                        or "genuinely hot" in text or "nothing left to do" in text or "light above" in text:
                    fit = "doesn't fit"          # every foil
                elif "not checked anything" in text or "never sent" in text or "no longer matches" in text:
                    fit = "doesn't fit"          # the far level
                out[n] = fit
            return json.dumps(out)
        return json.dumps({"meaning": "something that looks fine and is not",
                           "use": "a warning before someone trusts it", "heard_before": "yes"})


def _client(path):
    from backend.llm.client import LLMClient
    return LLMClient(_Fake(), Path(path), mode="record")


# ------------------------------------------------------------------------------------------- registry
def test_registry_parses_and_validates():
    from backend.analysis.battery import registry as REG
    specs = REG.load_registry(_cfg())
    assert [s.id for s in specs] == ["tray_washer", "hood_sash", "running_warm"]
    assert REG.validate(specs) == []
    assert REG.cells(specs) == {"grounded_broad": ["tray_washer"], "grounded_narrow": ["hood_sash"],
                                "ungrounded_broad": ["running_warm"]}
    s = REG.by_id(specs)["tray_washer"]
    assert s.cell == "grounded_broad" and s.repaired_day == 3 and s.seed_k == 2
    assert s.distinctive_words == ("blue", "tray")
    assert REG.by_id(specs)["running_warm"].repaired_day is None   # nothing to repair: the comparison


def test_registry_is_disabled_by_default():
    from backend.analysis.battery import registry as REG
    assert REG.load_registry({}) == [] and REG.load_registry({"memes": {"registry": REGISTRY}}) == []
    assert len(REG.load_registry({"memes": {"registry": REGISTRY}}, only_enabled=False)) == 3


def test_validate_catches_design_breaks():
    from backend.analysis.battery import registry as REG
    bad = [dict(REGISTRY[0], foils=[], seeds={"k": 9}), dict(REGISTRY[2], seeds={"k": 2})]
    errs = " ".join(REG.validate([REG.parse_entry(b) for b in bad]))
    assert "no foils" in errs and "minority sizes differ" in errs


def test_explicit_seed_agents_are_the_minority_size_and_must_not_overlap():
    from backend.analysis.battery import registry as REG
    specs = [REG.parse_entry(dict(REGISTRY[0], seeds={"agents": ["aria", "benji"]})),
             REG.parse_entry(dict(REGISTRY[1], seeds={"agents": ["benji", "dev"]})),
             REG.parse_entry(dict(REGISTRY[2], seeds={"agents": ["ethan", "hana"]}))]
    assert [s.seed_k for s in specs] == [2, 2, 2]          # k absent: the list is the size
    errs = " ".join(REG.validate(specs))
    assert "more than one meme" in errs and "minority sizes differ" not in errs


def test_config_distinctive_words_extend_rather_than_replace():
    from backend.analysis.battery import registry as REG
    s = REG.parse_entry(dict(REGISTRY[0], distinctive_words=["greasy"]))
    assert s.distinctive_words == ("blue", "greasy", "tray")
    assert REG.world_contamination([s], "the greasy racks")[s.id]["words"] == {"greasy": 1}


def test_gradient_and_foils_never_contain_the_phrase():
    from backend.analysis.battery import registry as REG
    leak = REG.parse_entry(dict(REGISTRY[0], probe_gradient=dict(REGISTRY[0]["probe_gradient"],
                                                                 far="a blue tray of excuses")))
    assert any("answer its own question" in e for e in REG.validate([leak]))


def test_variant_matching_is_morphological_not_semantic():
    from backend.analysis.battery import registry as REG
    p = REG.usage_pattern("blue-tray")
    assert p.search("that blue tray") and p.search("blue-trays") and p.search("Blue Tray")
    assert not p.search("a blue plastic tray")
    assert REG.usage_pattern("the Tuesday list").search("check the tuesday list")


def test_world_contamination_check(run):
    from backend.analysis.battery import registry as REG
    from backend.analysis.rundata import RunData
    rd = RunData(run[0])
    specs = REG.load_registry(rd.cfg)
    c = REG.world_contamination(specs, "the washer ran cold and the trays came out greasy")
    assert c["tray_washer"]["phrase"] == 0 and c["tray_washer"]["words"] == {"tray": 1}   # R2: "tray" leaked
    assert REG.world_contamination(specs, "the washer ran cold")["tray_washer"]["words"] == {}


# --------------------------------------------------------------------------------- usages and cohorts
def test_seeds_are_read_from_wherever_the_engine_recorded_them(run):
    """The engine records the minority in more than one place (profile.apply_planted's `memes` list, the
    per-agent habits it actually applied); the observer must find it in any of them, and must prefer what
    the engine did over what the config asked for."""
    from backend.analysis.battery import registry as REG
    from backend.analysis.rundata import RunData
    rd, ids = RunData(run[0]), run[1]
    spec = REG.by_id(REG.load_registry(rd.cfg))["tray_washer"]
    assert REG.seed_agents(rd, spec) == sorted(ids[:2])                  # manifest["memes"]
    rd.manifest.pop("memes")
    rd.manifest["planted"] = {"memes": [{"id": "tray_washer", "seeds": [ids[OUTSIDER]]}]}
    assert REG.seed_agents(rd, spec) == [ids[OUTSIDER]]
    rd.manifest.pop("planted")
    rd.manifest["agents"][ids[ADOPTER]]["habits"] = ['says "blue-tray" about things that look fine']
    assert REG.seed_agents(rd, spec) == [ids[ADOPTER]]                    # the applied habit line
    rd.manifest["agents"][ids[ADOPTER]]["habits"] = []
    cfg_only = REG.parse_entry(dict(REGISTRY[0], seeds={"agents": ["someone"]}))
    assert REG.seed_agents(rd, cfg_only) == ["someone"]                   # config, only as a last resort


def test_find_usages_and_seed_cohorts(run):
    from backend.analysis.battery import graded as G
    from backend.analysis.battery import registry as REG
    from backend.analysis.rundata import RunData
    rd, ids = RunData(run[0]), run[1]
    specs = REG.load_registry(rd.cfg)
    tray = REG.by_id(specs)["tray_washer"]
    us = REG.find_usages(rd, tray)
    assert len(us) == 4 and sorted({u["variant"] for u in us}) == ["blue tray", "blue-tray", "blue-trays"]
    assert REG.seed_agents(rd, tray) == sorted(ids[:2])
    sites = G.exposure_sites(rd, specs)
    assert sites[ids[0]] == "exposed" and sites[ids[4]] == "unexposed"
    cm = G.cohort_map(rd, tray, sites)
    assert cm[ids[0]] == "seeded" and cm[ids[ADOPTER]] == "exposed_adopter" and cm[ids[4]] == "non_adopter"
    # a meme nobody picked up has no adopters at all - that IS the comparative result
    assert set(G.cohort_map(rd, REG.by_id(specs)["running_warm"], sites).values()) <= {"seeded", "non_adopter"}


def test_panel_is_deterministic_and_stratified(run):
    from backend.analysis.battery import graded as G
    from backend.analysis.battery import registry as REG
    from backend.analysis.rundata import RunData
    rd = RunData(run[0])
    specs = REG.load_registry(rd.cfg)
    p1, c1 = G.sample_panel(rd, specs, "C4", per_cohort=2, max_agents=4)
    p2, _ = G.sample_panel(rd, specs, "C4", per_cohort=2, max_agents=4)
    assert p1 == p2 and len(p1) == 4
    assert {c1["per_meme"]["tray_washer"][a] for a in p1} >= {"seeded", "non_adopter"}
    assert G.sample_panel(rd, specs, "C2", per_cohort=2, max_agents=4)[0] != p1 or len(p1) == 4


# ----------------------------------------------------------------------------------- the battery run
def test_graded_battery_isolated_and_scored(run):
    from backend.analysis.battery import boundary as B
    from backend.analysis.battery import graded as G
    from backend.analysis.battery import registry as REG
    from backend.analysis.battery.runner import run_checksums
    rd, ids = run
    before = run_checksums(rd)
    s = G.run_graded(rd, "C4", llm=_client(rd / "probes" / "graded" / "C4" / "llm_calls.jsonl"))
    assert s["valid"] and s["n_errors"] == 0
    assert run_checksums(rd) == before                      # nothing outside probes/ changed
    assert not (rd / "probes" / "graded" / "C4" / "work").exists()
    rows = G.read_responses(rd, "C4")
    # 3 memes x (4 gradient + 2 foils) rows + 3 explanations, per panel member
    n_panel = len(s["panel"])
    assert len(rows) == n_panel * (3 * 6 + 3) and n_panel == 4
    assert {r["form"] for r in rows} == {"G", "E"} and all(r["valid"] for r in rows)
    cells = B.checkpoint_cells(rows, REG.load_registry(yaml_cfg(rd)))
    t = cells["tray_washer"]["all"]
    assert t["levels"]["literal"]["applies"] == 1.0 and t["levels"]["far"]["applies"] == 0.0
    assert t["foil_false_positive"] == 0.0 and t["foil_invalid"] is False
    assert t["boundary_index"] == 3 and t["boundary_level"] == "mid"


def yaml_cfg(run_dir):
    return yaml.safe_load(open(Path(run_dir) / "config.resolved.yaml"))


def test_high_foil_rate_invalidates_the_boundary():
    from backend.analysis.battery import boundary as B
    rows = ([{"form": "G", "kind": "gradient", "level": l, "answer": "fits", "fit": 1.0, "valid": True,
              "agent": f"a{i}"} for l in B.LEVELS for i in range(4)]
            + [{"form": "G", "kind": "foil", "level": "foil", "answer": "fits", "fit": 1.0, "valid": True,
                "agent": f"a{i}"} for i in range(4)])
    c = B.cohort_cell(rows)
    assert c["foil_false_positive"] == 1.0 and c["foil_invalid"] is True
    assert c["boundary_raw"] == 4 and c["boundary_index"] is None    # agreeable, not broad


def test_boundary_needs_enough_answers():
    from backend.analysis.battery import boundary as B
    rows = [{"form": "G", "kind": "gradient", "level": "literal", "answer": "fits", "fit": 1.0,
             "valid": True, "agent": "a"}]
    assert B.cohort_cell(rows)["boundary_index"] is None              # n=1 is not a boundary


def test_repeat_run_is_cached_and_identical(run):
    from backend.analysis.battery import graded as G
    rd, _ = run
    p = rd / "probes" / "graded" / "C4" / "llm_calls.jsonl"
    s1 = G.run_graded(rd, "C4", llm=_client(p))
    first = (rd / "probes" / "graded" / "C4" / "responses.jsonl").read_text()
    s2 = G.run_graded(rd, "C4", llm=_client(p))
    assert s2.get("cached") and s1["digest"] == s2["digest"]
    assert (rd / "probes" / "graded" / "C4" / "responses.jsonl").read_text() == first


# ------------------------------------------------------------------------------ comparison and trends
def test_compare_puts_the_memes_side_by_side(run):
    from backend.analysis.battery import boundary as B
    from backend.analysis.battery import graded as G
    rd, _ = run
    G.run_graded(rd, "C4", llm=_client(rd / "probes" / "graded" / "C4" / "llm_calls.jsonl"))
    doc = B.compare(rd)
    assert (rd / "memes.json").exists()
    assert {r["meme"] for r in doc["table"]} == {"tray_washer", "hood_sash", "running_warm"}
    tray = doc["memes"]["tray_washer"]
    assert tray["usage"]["uses"] == 4 and tray["usage"]["unique_users"] == 3
    assert tray["usage"]["uses_after_repair"] == 1 and tray["usage"]["repaired_day"] == 3
    warm = doc["memes"]["running_warm"]
    assert warm["usage"]["survival_ratio"] is None      # ungrounded: no repair, by construction
    txt = B.render_table(doc)
    assert "grounded" in txt and "ungrounded" in txt and "tray_washer" in txt


def test_explanations_from_a_mock_judge_never_count(run):
    from backend.analysis.battery import boundary as B
    from backend.analysis.battery import graded as G
    rd, _ = run
    G.run_graded(rd, "C4", llm=_client(rd / "probes" / "graded" / "C4" / "llm_calls.jsonl"))
    d = B.classify_explanations(rd, "C4")
    assert d["placeholder"] is True and d["n_judge_calls"] >= 1
    fns = B.load_functions(rd, "C4")
    assert set(fns) == {"_provenance"} and "PLACEHOLDER" in fns["_provenance"]


def test_trends_registry_series_excludes_seeds_from_adopters(run):
    from backend.analysis import trends as T
    rd, ids = run
    rows = {r["id"]: r for r in T.registry_series(rd)}
    tray = rows["tray_washer"]
    assert tray["k"] == 2 and tray["unique_users"] == 3 and tray["unique_adopters"] == 1
    assert tray["speakers_cum"][-1] == 1                   # the two seeds are not adopters
    assert tray["survival"]["uses_after_repair" if "uses_after_repair" in tray["survival"] else "uses_after"] == 1
    assert rows["running_warm"]["survival"]["repaired_day"] is None
    assert rows["hood_sash"]["unique_adopters"] == 0        # narrow + grounded: nobody picked it up
    assert tray["seed_graph"]["k"] == 2 and tray["variants"]


def test_transmission_does_not_call_a_seed_an_inventor(run):
    from backend.analysis import transmission as TR
    from backend.analysis.battery import registry as REG
    from backend.analysis.rundata import RunData
    rd, ids = RunData(run[0]), run[1]
    spec = REG.by_id(REG.load_registry(rd.cfg))["tray_washer"]
    us = REG.find_usages(rd, spec)
    cand = {"id": "tray_washer", "variants": ["blue-tray", "blue tray"], "usages": us}
    plain = TR.analyze_transmission(cand, rd, {})
    seeded = TR.analyze_transmission(dict(cand, usages=list(us)), rd, {}, seeds=ids[:2])
    assert {i["agent"] for i in plain["inventors"]} >= set(ids[:2])
    assert not ({i["agent"] for i in seeded["inventors"]} & set(ids[:2]))
    assert seeded["seeds"] == sorted(ids[:2])


def test_cost_projection_is_computed_not_guessed(run):
    from backend.analysis.battery import graded as G
    from backend.analysis.battery import registry as REG
    from backend.analysis.rundata import RunData
    rd = RunData(run[0])
    specs = REG.load_registry(rd.cfg)
    c = G.project_cost(rd, specs, ["C2", "C4", "final"], REG.battery_cfg(rd.cfg))
    assert c["calls_per_agent_per_checkpoint"] == 6          # 3 memes x (grid + explanation)
    assert c["panel_size"] == 4 and c["probe_calls"] == 4 * 6 * 3
    assert c["situations_per_agent_per_checkpoint"] == 3 * 6
    assert c["all_agents_probe_calls"] == 8 * 6 * 3


def test_formal_variation_tracks_the_surface_forms(run):
    from backend.analysis import lineage as L
    from backend.analysis.battery import registry as REG
    from backend.analysis.rundata import RunData
    rd = RunData(run[0])
    spec = REG.by_id(REG.load_registry(rd.cfg))["tray_washer"]
    fv = L.formal_variation(REG.find_usages(rd, spec), spec.phrase)
    assert fv["n_variants"] == 3 and fv["canonical_share"] == 0.5
    assert {v["form"] for v in fv["variants"]} == {"blue-tray", "blue tray", "blue-trays"}


def test_probe_readiness_names_what_is_missing(run, tmp_path):
    from backend.analysis.probes import graded_probe_ready
    rd, _ = run
    r = graded_probe_ready(rd)
    assert r["ready"] and r["checkpoints_on_disk"] == ["C4"] and r["registry_errors"] == []
    bare = tmp_path / "bare"
    (bare / "agents_final").mkdir(parents=True)
    (bare / "config.resolved.yaml").write_text(yaml.safe_dump({}))
    r2 = graded_probe_ready(bare)
    assert not r2["ready"] and any("no memes.registry" in m for m in r2["missing"])
