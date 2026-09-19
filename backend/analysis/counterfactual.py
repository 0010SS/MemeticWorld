"""Decision counterfactuals (DCF; OBSERVER ONLY; ontology v3 §5.9). STUB (MVP scope cut).

Planned: re-query every logged post-D5 K1 first-attempt prompt of `keep_shift` and `wipe_shift` on a separate
observer client (scope `dcf:`; `llm.client.observer_client("dcf", run/"dcf/llm_calls.jsonl")`) in four versions
(unchanged, binder removed, front page -> last pre-D5 revision, front page -> latest post-D5 revision) and write
`dcf.jsonl`. It must never write into the run's simulation files or its llm_calls.jsonl.
"""
from __future__ import annotations

VERSIONS = ("unchanged", "binder_removed", "front_pre_change", "front_post_change")


def run_dcf(run_dir, versions=VERSIONS, backend=None):
    raise NotImplementedError("DCF counterfactuals are deferred (v3 MVP scope cut; §8.4 runs them after G2)")


def dcf_effects(run_dir) -> dict:
    """Summary for outcomes.json v3.dcf once dcf.jsonl exists; until then {"status": "not_run"}."""
    return {"status": "not_run"}
