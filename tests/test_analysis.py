"""Observer pipeline on a mock run + transmission logic on a synthetic candidate."""
from backend.analysis.pipeline import analyze
from backend.analysis.transmission import analyze_transmission


def test_pipeline_produces_complete_analysis(mock_run):
    out = analyze(mock_run.run_dir, probes=True, top_probe=1, verbose=False)
    assert out["candidates"], "expected at least one candidate"
    c = out["candidates"][0]
    for k in ("canonical_form", "variants", "first_occurrence", "speakers", "contexts", "usage_count"):
        assert k in c
    assert c["id"] in out["transmission"] and c["id"] in out["semantics"] and c["id"] in out["evaluation"]
    assert out["probes"]
    probed = next(iter(out["probes"].values()))
    assert len(probed["agents"]) == len(mock_run.agents)
    # probes must not touch agent memories on disk
    import json
    nodes = json.load(open(mock_run.run_dir / "agents_final" / "maya" / "associative_memory" / "nodes.json"))
    assert not any("expression" in n["description"] for n in nodes.values())


class _RD:
    manifest = {"ticks_per_day": 60, "tick_minutes": 15}
    groups = {"g1": ["a", "b"], "g2": ["c"]}

    def groups_of(self, x):
        return {g for g, m in self.groups.items() if x in m}


def test_transmission_edges_and_inventors():
    U = lambda uid, t, s, ls: {"utterance_id": uid, "tick": t, "speaker": s, "listeners": ls}
    cand = {"id": "m0", "variants": ["brody moment"], "usages": [
        U("u1", 1, "a", ["b"]), U("u2", 3, "b", ["c"]), U("u3", 5, "c", ["a"]), U("u4", 6, "d", [])]}
    t = analyze_transmission(cand, _RD(), {("c", "u2"): "C heard B say brody moment"})
    edges = {(e["source_agent"], e["target_agent"]): e for e in t["edges"]}
    assert ("a", "b") in edges and ("b", "c") in edges
    assert edges[("b", "c")]["cross_group"] is True
    assert {i["agent"] for i in t["inventors"]} == {"a", "d"}
    assert t["depth"] == 2
    assert edges[("b", "c")]["exposures"][0]["retained_in_memory"]


def test_transmission_needs_causal_exposure():
    """Same-tick remarks are decided in parallel and a tick-t remark reaches memory only after tick t's
    conversations: neither creates an edge. An earlier turn of the same conversation does."""
    R = lambda t, s, ls: {"utterance_id": f"t{t:04d}:remark:{s}:ev1.b0", "tick": t, "speaker": s, "listeners": ls,
                          "conversation_id": None}
    C = lambda cid, i, t, s, ls: {"utterance_id": f"{cid}.u{i}", "tick": t, "speaker": s, "listeners": ls,
                                  "conversation_id": cid, "idx": i}
    cand = {"id": "m0", "variants": ["chopsticks"], "usages": [
        R(7, "a", ["b", "c"]), R(7, "b", ["a", "c"]),         # simultaneous witness remarks
        C("k1", 0, 7, "c", ["d"]),                           # tick-7 conversation: cannot have heard tick-7 remarks
        C("k1", 1, 7, "d", ["c"])]}                          # heard c's turn just before: an edge c -> d
    t = analyze_transmission(cand, _RD(), {})
    assert {(e["source_agent"], e["target_agent"]) for e in t["edges"]} == {("c", "d")}
    assert {i["agent"] for i in t["inventors"]} == {"a", "b", "c"}
