"""Probe batteries and their isolated runner. OBSERVER ONLY: nothing under backend/simulation,
backend/agents or backend/memory may import this package.

LB1, the v3 workshop battery (ONTOLOGY_V3 §5.2-5.6):
    run_battery(run_dir, ckpt, modes=("memory_only", "situated"), plan=None, targets=None, llm=None) -> summary
    select_targets(trunk_run_dir, "C3", c0_responses=None) -> targets.json dict
    run_fresh(cfg, out_root, mapping=None, targets=None) -> C0 summary
    run_record_only(session, items, cues, cue_keys, framings) -> records
    ground_truth(item, regime, mapping) (= gt.gt) / score(action, item, regime, mapping)

The graded meme battery (memo §3.3; the injected-meme cohort), sharing LB1's isolation protocol:
    registry.load_registry(cfg) -> [MemeSpec]        the config-declared cohort; nothing is hardcoded
    graded.run_graded(run_dir, ckpt) -> summary      one panel answers every meme on an isolated copy
    graded.project_cost(rd, specs, checkpoints, bcfg) -> the call count, before spending it
    boundary.compare(run_dir) -> the four memes side by side; boundary.render_table(doc) -> text

Stubs (MVP scope cuts): synthetic-memory calibration (synthetic.py), test-retest (retest.py).
"""
from backend.analysis.battery.gt import gt as ground_truth, score  # noqa: F401  (the `gt` name stays the submodule)
from backend.analysis.battery.items import heldout_texts, load_items  # noqa: F401
from backend.analysis.battery.nonce import make_nonce  # noqa: F401


def run_battery(*a, **k):
    from backend.analysis.battery.runner import run_battery as f
    return f(*a, **k)


def select_targets(*a, **k):
    from backend.analysis.battery.targets import select_targets as f
    return f(*a, **k)


def run_fresh(*a, **k):
    from backend.analysis.battery.fresh import run_fresh as f
    return f(*a, **k)


def run_graded(*a, **k):
    from backend.analysis.battery.graded import run_graded as f
    return f(*a, **k)


def compare_memes(*a, **k):
    from backend.analysis.battery.boundary import compare as f
    return f(*a, **k)
