"""v3 invariants (ontology v3 §8.1 step 10; hard constraints in the preamble). Mock backend only.

* Templates: no new (v3) agent-facing prompt template asks agents to invent, coin, name, label, ... anything.
* Structure: simulation code never imports backend.analysis and never opens probes/; agents hold no knowledge,
  procedure or meaning fields; the binder and its receipts live only in world state.
* API: demo mode strips every v3 hidden key (job truth, regime, mapping, K ids, causes, arm/branch names).
* CoopWorld: a one-day mock co-op run through the §4.7 hooks (a test harness wires CoopWorld into a Simulation
  subclass until engine.py does): 8 jobs start and end, decisions and outcomes happen, hidden records exist only
  as hidden trace types, no rendered prompt carries a hidden id, checkpoints are written.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import time
from pathlib import Path

import pytest

from tests.conftest import ROOT

PROMPTS = ROOT / "backend" / "prompts"
BANNED = re.compile(r"\b(invent\w*|coin(s|ed|ing)?|label\w*|terms?|slang|memes?|nicknames?|shorthand|rules?|tips?|"
                    r"procedures?|jot\w*)\b", re.I)
NAME_VERB = re.compile(r"\b(to|and|or|please|then|can|could|would|should)\s+name\b|\bname (it|them|this|that|these|"
                       r"those|the)\b", re.I)
# hidden ids that must never reach a rendered prompt (§8.1 step 10)
HIDDEN_IN_PROMPT = [re.compile(p) for p in (
    r"\bK[0-3]\b", r"\bLENS\b", r"\bDAMP\b", r"\bBELT\b", r"\bAIR\b", r"\bWARP\b", r"\bM[12]\b", r"\bj\d\d\.\d\b",
    r"\bwipe4\b", r"\bkeep_shift\b", r"\bwipe_shift\b", r"\bkeep_noshift\b", r"\bnoshift\b", r"\bplacebo\b",
    r"\btrunk\b", r"\bregime\b", r"\bmapping\b", r"\blaser_alpha\b", r"\blaser_beta\b", r"\bE[1-4]\b",
    r"\bu_attempt\b", r"\bp_fault\b")]
OBSERVER_ONLY = re.compile(r"^code_")          # Sonnet coder prompts: observer instruments, never shown to agents
SIM_PACKAGES = ("simulation", "agents", "memory", "modules")


def v3_prompt_files() -> list[Path]:
    """Prompt files added after the v2 freeze (git tag v2.0); every file if the tag is unavailable."""
    try:
        old = subprocess.run(["git", "ls-tree", "--name-only", "v2.0", "backend/prompts/"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10, check=True).stdout.split()
        old = {Path(p).name for p in old}
    except Exception:  # noqa: BLE001
        old = set()
    return sorted(p for p in PROMPTS.glob("*.txt") if p.name not in old)


def template_body(text: str) -> str:
    """The part of a template the model sees (GA templates: after the comment-block marker)."""
    return text.split("<commentblockmarker>###</commentblockmarker>", 1)[-1]


def audit_text(text: str) -> list[str]:
    return [p.pattern for p in HIDDEN_IN_PROMPT if p.search(text)]


# ------------------------------------------------------------------------------------------- templates
def test_new_templates_never_ask_for_coinage():
    files = [p for p in v3_prompt_files() if not OBSERVER_ONLY.match(p.name)]
    assert any(p.name == "job_decision_v1.txt" for p in files) or not (PROMPTS / "job_decision_v1.txt").exists()
    bad = {}
    for p in files:
        body = template_body(p.read_text())
        hits = sorted({m.group(0).lower() for m in BANNED.finditer(body)} |
                      {m.group(0).lower() for m in NAME_VERB.finditer(body)})
        if hits:
            bad[p.name] = hits
    assert not bad, f"v3 templates contain banned wording: {bad}"


def test_new_templates_carry_no_hidden_ids():
    bad = {p.name: audit_text(template_body(p.read_text())) for p in v3_prompt_files()
           if not OBSERVER_ONLY.match(p.name)}
    assert not {k: v for k, v in bad.items() if v}


# ------------------------------------------------------------------------------------------- structure
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def test_simulation_code_never_imports_the_observer():
    bad = []
    for pkg in SIM_PACKAGES:
        for f in sorted((ROOT / "backend" / pkg).rglob("*.py")):
            src = f.read_text()
            if any(i == "backend.analysis" or i.startswith("backend.analysis.") for i in _imports(f)) or \
                    re.search(r"import_module\(\s*['\"]backend\.analysis", src):
                bad.append(str(f.relative_to(ROOT)))
            if re.search(r"['\"/]probes/", src):
                bad.append(f"{f.relative_to(ROOT)} (opens probes/)")
    assert not bad, bad


def test_agents_hold_no_knowledge_fields():
    from backend.agents.agent import AgentState
    names = set(AgentState.__dataclass_fields__)
    forbidden = re.compile(r"knowledge|procedure|meaning|meme|belief|rule|lesson|binder|record|diagnos", re.I)
    assert not [n for n in names if forbidden.search(n)], names


# ------------------------------------------------------------------------------------------------ API
def test_api_strips_v3_hidden_fields():
    from backend.api.server import _strip, _strip_manifest
    man = {"config": {"regimes": {"mapping": "auto", "schedule": [{"day": 5, "regime": "B"}]},
                      "records": {"enabled": True, "transitions": [{"day": 4, "mode": "wipe"}]},
                      "workshop": {"enabled": True, "content": "laser_alpha", "causal": "scrambled",
                                   "class_weights": {"K1": 0.7, "K2": 0.3}},
                      "branch": {"parent": "trunk", "at_day": 5, "salt": None},
                      "turnover": {"enabled": True, "rotate": "by_world_seed"}},
           "branch": {"node": "wipe_shift", "parent": "trunk"}, "roster": {"schedule": []},
           "world_state": {"regime": "B"}}
    out = json.dumps(_strip_manifest(man))
    for s in ("regime", "mapping", "transitions", "laser_alpha", "scrambled", "K1", "wipe_shift", "trunk",
              "by_world_seed", "salt"):
        assert s not in out, s
    trace = [{"type": "job_truth", "tick": 6, "job": "j01.0", "class": "K1", "cause": "LENS"},
             {"type": "regime_active", "tick": 0, "day": 1, "regime": "A", "mapping": "M1"},
             {"type": "job_start", "tick": 6, "job": "j01.0", "operator": "dev", "project": "tube racks"},
             {"type": "job_end", "tick": 7, "job": "j01.0", "delivered": True, "klass": "K1"},
             {"type": "checkpoint", "tick": 47, "files": {"world_state.json": "abc"}}]
    got = _strip(trace)
    assert [r["type"] for r in got] == ["job_start", "job_end", "checkpoint"]
    s = json.dumps(got)
    for h in ("K1", "LENS", "M1", '"regime"', '"klass"'):
        assert h not in s, h
    assert got[0]["project"] == "tube racks" and got[1]["delivered"] is True


# ------------------------------------------------------------------------------------ CoopWorld (mock)
V3_DAY = {
    "population_size": 8,   # fixture written for the 8-agent campus (the file now has 10)
    "run_name": "coop_inv", "seed": 13, "simulation_days": 1, "day_end": "19:30",
    "llm": {"backend": "mock", "max_workers": 6},
    "latent_events": {"event_rate": 0.0},
    "workshop": {"enabled": True, "panel_code": {"enabled": True}},
    "records": {"enabled": True, "transitions": [{"day": 1, "mode": "keep"}]},
    "comm": {"clarify": {"enabled": True}, "handover": {"enabled": True},
             "meeting": {"enabled": True, "days": [1]}},
    "checkpoints": {"enabled": True},
    "regimes": {"cues": [{"day": 1, "time": "16:45", "arena": "Stockroom", "text_key": "new_supplier"}]},
}


def engine_wired() -> bool:
    return "coop_world" in (ROOT / "backend" / "simulation" / "engine.py").read_text()


def harness_sim_class():
    """A Simulation subclass that calls CoopWorld at the §4.7 places. It is the reference for the engine diff;
    once engine.py is wired the real engine is used instead."""
    import backend.simulation.engine as E
    from backend.agents import work as WK
    from backend.llm.client import llm_scope
    from backend.simulation.coop_world import CoopWorld
    from backend.simulation.rngs import seed_rng
    from backend.simulation.scheduler import plan_day

    class CoopSim(E.Simulation):
        def run(self):
            t_start = time.time()
            self.coop = CoopWorld(self)
            self.write_manifest("running")
            self._seed_memories()
            tpd = self.clock.ticks_per_day
            orig_render, orig_react = E.render_viewpoint, E.decide_reaction

            def render(a, o):                           # phase 4: episodes render only unrendered facts
                if getattr(o, "coop_episode", False):
                    return WK.render_pending(a, o)
                if o.source_type == "record":           # written text is read as written
                    return None
                return orig_render(a, o)

            def react(a, o, present, rng):
                if getattr(o, "no_react", False):
                    return {"action": "CONTINUE", "target": None, "utterance": None, "retrieved": []}
                return orig_react(a, o, present, rng)
            E.render_viewpoint, E.decide_reaction = render, react
            try:
                for tick in range(self.clock.total_ticks):
                    now = self.clock.time_of(tick)
                    self.tracer.tick, self.tracer.time = tick, now.isoformat()
                    self.tick_utts = []
                    if tick % tpd == 0:
                        day = self.clock.day_of(tick)
                        self.coop.day_start(tick)                                   # 0a
                        for aid in sorted(self.agents):
                            a = self.agents[aid]
                            a.day_plan = plan_day(a, self.clock, seed_rng(self.world_seed, "plan", aid, day))
                            a.state.talks_today = 0
                    for a in self.agents.values():
                        a.set_time(now)
                    self.ctx.mods.on_tick(tick, self.agents)
                    with llm_scope(f"t{tick:04d}:00world"):
                        self._release_events(tick)
                        beats = self._beats_now(tick) + self.coop.world(tick)       # 1
                        self._move(tick, beats)
                        obs = self._perceive(tick, beats)
                    obs = self.coop.perception_hook(tick, obs)                      # 3+
                    results = self._cognition(tick, obs)                            # 4
                    self.coop.after_cognition(tick, obs)                            # 4b
                    req = self.coop.apply(tick, results)                            # 4c
                    talks, remark_obs = self._apply_decisions(tick, results)        # 5
                    extra = self._conversations(tick, req.dyads + talks, remark_obs)  # 6
                    self._groups(tick, req.groups, extra)
                    self._encode_extra(tick, extra)
                    self._reflect(tick)
                    with llm_scope(f"t{tick:04d}:99frame"):
                        self._frame(tick, beats)
                    self.tracer.flush()
                    self.coop.day_end(tick)                                         # 8b
            finally:
                E.render_viewpoint, E.decide_reaction = orig_render, orig_react
            self.finish(time.time() - t_start)

        def _groups(self, tick, groups, extra):
            from backend.agents.group_conversation import run_group_conversation
            for j, (parts, topic, _mu) in enumerate(groups):
                parts = [p for p in parts if not self.agents[p].state.in_conversation]
                if len(parts) < 2:
                    continue
                cid = f"d{self.clock.day_of(tick)}t{tick:04d}m{j}"
                a = self.agents[parts[0]]
                by = [x for x in sorted(self.agents.values(), key=lambda z: z.id) if x.id not in parts and
                      (x.state.location, x.state.arena) == (a.state.location, a.state.arena)]
                with llm_scope(f"t{tick:04d}:05conv:{cid}"):
                    conv = run_group_conversation(cid, [self.agents[x] for x in parts], by,
                                                  seed_rng(self.cfg["seed"], "gconv", tick, *parts), topic=topic)
                for o in conv.pop("overheard"):
                    extra.setdefault(o.agent_id, []).append(o)
    return CoopSim


@pytest.fixture(scope="module")
def coop_run(tmp_path_factory):
    from backend.config import deep_merge, load_config
    from tests.conftest import ROOT as _R  # noqa: F401
    cfg = load_config("configs/baseline.yaml", deep_merge({}, V3_DAY))
    run_dir = tmp_path_factory.mktemp("coop") / "run"
    if engine_wired():
        from backend.simulation.engine import Simulation
        sim = Simulation(cfg, run_dir, progress=False)
    else:
        sim = harness_sim_class()(cfg, run_dir, progress=False)
    sim.run()
    trace = [json.loads(l) for l in open(run_dir / "trace.jsonl")]
    return sim, run_dir, trace


def _of(trace, t):
    return [r for r in trace if r["type"] == t]


def test_coop_day_jobs_run_to_the_end(coop_run):
    sim, run_dir, trace = coop_run
    starts, ends = _of(trace, "job_start"), _of(trace, "job_end")
    assert len(starts) == 8 and len(ends) == 8
    assert {r["job"] for r in starts} == {r["job"] for r in ends}
    truth = {r["job"]: r for r in _of(trace, "job_truth")}
    assert set(truth) == {r["job"] for r in starts}
    faulted = {j for j, r in truth.items() if r["class"] != "K0"}
    decided = {r["job"] for r in _of(trace, "job_decision")}
    assert faulted and faulted <= decided and not (decided - faulted)
    for e in ends:
        n = len(e["attempts"])
        assert n <= 2 and (e["job"] in faulted) == (n > 0)
        if e["delivered"] and n:
            assert e["attempts"][-1]["outcome"] == "success"
    for r in _of(trace, "job_decision"):                # at most one question per job, then a decision
        assert r["attempt"] in (1, 2)
    assert len(_of(trace, "regime_active")) == 1 and len(_of(trace, "tally")) == 1


def test_coop_day_episodes_reads_writes_talk(coop_run):
    sim, run_dir, trace = coop_run
    enc = [r.get("observation_id") or "" for r in _of(trace, "memory_encoded")]
    assert any(".ep:" in o for o in enc), "job episodes are encoded"
    assert any(o.startswith("rec:") for o in enc), "new binder items are encoded on read"
    reads = _of(trace, "record_read")
    assert {r["context"] for r in reads} >= {"decision", "tally"}
    assert _of(trace, "record_write") and _of(trace, "record_transition")
    assert _of(trace, "handover") and _of(trace, "meeting")
    topics = {(r.get("trigger") or {}).get("topic") for r in _of(trace, "conversation")}
    assert "meeting" in topics
    assert _of(trace, "cue_event")
    ck = _of(trace, "checkpoint")
    assert [c["id"] for c in ck] == ["C1"]
    ws = json.loads((run_dir / "checkpoints" / "C1" / "world_state.json").read_text())
    assert ws["coop"]["outcomes"] and ws["mapping"] in ("M1", "M2")
    binder = json.loads((run_dir / "checkpoints" / "C1" / "binder.json").read_text())
    assert "receipts" in binder


def test_coop_day_binder_is_world_state_only(coop_run):
    sim, _, _ = coop_run
    from backend.simulation.records import Binder
    for a in sim.agents.values():
        for k, v in vars(a).items():
            assert not isinstance(v, Binder), k
        for k, v in vars(a.state).items():
            assert not isinstance(v, Binder), k


def test_coop_day_prompts_carry_no_hidden_ids(coop_run):
    _, run_dir, _ = coop_run
    bad = []
    n = 0
    for line in open(run_dir / "llm_calls.jsonl"):
        r = json.loads(line)
        n += 1
        hits = audit_text(r.get("prompt") or "")
        if hits:
            bad.append((r.get("purpose"), hits, (r.get("prompt") or "")[:200]))
    assert n > 0
    assert not bad, bad[:5]


def test_coop_day_demo_api_hides_truth(coop_run):
    _, run_dir, trace = coop_run
    from backend.api.server import _strip
    s = json.dumps(_strip(trace))
    assert '"job_truth"' not in s and '"regime_active"' not in s
    for h in (r'"K[0-3]"', r'"(LENS|DAMP|BELT)"', r'"M[12]"', r'"regime"', r'"mapping"', r'"u_attempt"'):
        assert not re.search(h, s), h
