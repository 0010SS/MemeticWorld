"""World-level contracts: causal effects, information boundaries, inheritance."""
import json

import pytest

from backend.simulation.commons import CommonsWorld


def world(**overrides):
    cfg = {"seed": 7, "change_day": 5, "turnover_day": 3, "newcomers": [], **overrides}
    w = CommonsWorld(cfg, {"a": "Ari Chen", "b": "Bo Morgan"}, ticks_per_day=24)
    w.advance(0)
    w.drain_events()
    return w


def act(w, aid="a", **intention):
    w.submit({aid: intention})
    events = w.drain_events()
    if aid not in w.operations:
        w.advance(w.tick + 1)
        events.extend(w.drain_events())
    while aid in w.operations:
        w.advance(w.tick + 1)
        events.extend(w.drain_events())
    return events


def prepare(w, kid="kit-01-2"):
    act(w, action="MOVE", destination="Research Lab")
    act(w, action="TAKE", kit=kid)
    act(w, action="ASSEMBLE", kit=kid)
    act(w, action="CALIBRATE", kit=kid, setting=w._bench_mode)


def test_operating_change_affects_field_but_not_bench_and_failure_is_recoverable():
    w = world(change_day=2)
    prepare(w)
    assert w.consumed["components"] == 3
    bench = act(w, action="TEST", kit="kit-01-2", test="bench")
    assert next(e for e in bench if e["kind"] == "test_result")["passed"]
    while w.tick < w.change_tick:
        w.advance(w.tick + 1)
    bench = act(w, action="TEST", kit="kit-01-2", test="bench")
    assert next(e for e in bench if e["kind"] == "test_result")["passed"]
    act(w, action="MOVE", destination="Quad")
    field = act(w, action="TEST", kit="kit-01-2", test="field")
    assert not next(e for e in field if e["kind"] == "test_result")["passed"]
    failed = act(w, action="DELIVER", kit="kit-01-2")
    assert not next(e for e in failed if e["kind"] == "delivery")["passed"]
    assert w.projects["project-01-2"].completed is None
    act(w, action="MOVE", destination="Research Lab")
    act(w, action="CALIBRATE", kit="kit-01-2", setting="B" if w._bench_mode == "A" else "A")
    act(w, action="MOVE", destination="Quad")
    delivered = act(w, action="DELIVER", kit="kit-01-2")
    assert next(e for e in delivered if e["kind"] == "delivery")["passed"]
    assert w.residents["a"].carrying is None
    assert w.projects["project-01-2"].attempts == 2


def test_simultaneous_station_contention_is_order_independent_and_charged_once():
    outcomes = []
    for order in (("a", "b"), ("b", "a")):
        w = world()
        for r in w.residents.values():
            r.location = "Research Lab"
        intents = {"a": {"action": "ASSEMBLE", "kit": "kit-01-1"},
                   "b": {"action": "ASSEMBLE", "kit": "kit-01-2"}}
        w.submit({aid: intents[aid] for aid in order})
        assert len(w.operations) == 1
        assert w.consumed["components"] == 2
        assert sum(e["kind"] == "action_rejected" for e in w.drain_events()) == 1
        outcomes.append(next(iter(w.operations)))
        w.advance(1)
        assert not any(k.assembled for k in w.kits.values())
        w.advance(2)
        assert sum(k.assembled for k in w.kits.values()) == 1
        assert not w.locks
    assert outcomes[0] == outcomes[1]


def test_travel_custody_and_private_measurements():
    w = world()
    prepare(w)
    w.residents["b"].location = "Research Lab"
    denied = act(w, "b", action="TEST", kit="kit-01-2", test="bench")
    assert denied[0]["kind"] == "action_rejected"
    events = act(w, action="TEST", kit="kit-01-2", test="bench")
    result = next(e for e in events if e["kind"] == "test_result")
    assert result["visible_to"] == ["a"]
    view = json.dumps(w.view("b"))
    for hidden in ("bench_mode", "change_enabled", "change_tick", "turnover_tick", "error", "passed"):
        assert hidden not in view
    w.submit({"a": {"action": "MOVE", "destination": "Gym"}})
    w.advance(w.tick + 1)
    assert w.residents["a"].location == "Research Lab"
    assert not any(p["id"] == "a" for p in w.view("b")["people"])
    w.advance(w.tick + 1)
    assert w.residents["a"].location == w.kits["kit-01-2"].location == "Gym"


def test_record_contents_require_a_read_and_versions_survive_revision():
    w = world()
    for r in w.residents.values():
        r.location = "Library"
    act(w, action="WRITE", title="Procedure", text="PRIVATE_CONTENT_SENTINEL")
    assert "PRIVATE_CONTENT_SENTINEL" not in json.dumps(w.view("b"))
    author_denied = act(w, action="REVISE", record="record-0001", version=1, title="Procedure", text="unchecked edit")
    assert author_denied[0]["kind"] == "action_rejected"
    denied = act(w, "b", action="REVISE", record="record-0001", version=1, title="Procedure", text="new")
    assert denied[0]["kind"] == "action_rejected"
    read = act(w, "b", action="READ", record="record-0001", version=1)
    assert any(e["kind"] == "record_read" and "PRIVATE_CONTENT_SENTINEL" in e["text"] and e["visible_to"] == ["b"] for e in read)
    act(w, "b", action="REVISE", record="record-0001", version=1, title="Procedure", text="REVISED_CONTENT")
    assert len(w.records["record-0001"].versions) == 2
    old = act(w, action="READ", record="record-0001", version=1)
    assert any("PRIVATE_CONTENT_SENTINEL" in e["text"] for e in old)
    conflict = act(w, action="REVISE", record="record-0001", version=1, title="Procedure", text="stale edit")
    assert conflict[0]["kind"] == "action_rejected"
    assert len(w.records["record-0001"].versions) == 2


def test_disabled_records_are_inaccessible():
    w = world(records_enabled=False)
    w.residents["a"].location = "Library"
    assert act(w, action="WRITE", title="x", text="x")[0]["kind"] == "action_rejected"
    assert not w.records
    assert not w.view("a")["archive_available"]


def test_departure_drops_objects_cancels_operations_and_keeps_records():
    w = world(turnover_day=2, newcomers=[{"id": "c", "name": "Casey Lee", "replaces": "a"}])
    w.residents["a"].location = "Library"
    act(w, action="WRITE", title="History", text="A report that can outlive its author.")
    prepare(w)
    while w.tick < w.turnover_tick - 1:
        w.advance(w.tick + 1)
    w.submit({"a": {"action": "TEST", "kit": "kit-01-2", "test": "bench"}})
    w.advance(w.tick + 1)
    assert not w.residents["a"].active and w.residents["c"].active
    assert w.kits["kit-01-2"].holder is None
    assert not w.operations and not w.locks
    assert w.read_versions["c"] == set()
    assert w.records["record-0001"].versions[0]["author"] == "a"
    assert "History" not in json.dumps(w.view("c"))  # newcomer is at the Quad


def test_exogenous_schedule_does_not_depend_on_agent_action_counts():
    drivers = []
    for records in (True, False):
        w = world(records_enabled=records, change_day=2)
        observed = []
        for tick in range(1, 50):
            if records and "a" not in w.operations:
                w.submit({"a": {"action": "MOVE", "destination": "Library" if w.residents["a"].location != "Library" else "Quad"}})
            w.advance(tick)
            for e in w.drain_events():
                if e["kind"] in ("project_released", "supply", "intervention"):
                    observed.append({k: v for k, v in e.items() if k not in ("event_key", "visible_to")})
        drivers.append(observed)
    assert drivers[0] == drivers[1]


@pytest.mark.parametrize("intention", [None, [], {"action": []}, {"action": "TEST", "kit": []},
                                      {"action": "MOVE", "destination": {}},
                                      {"action": "SAY", "text": "hello", "target": []}])
def test_malformed_intentions_leave_physical_state_unchanged(intention):
    w = world()
    before = w.snapshot()
    w.submit({"a": intention})
    assert w.snapshot() == before
    assert w.drain_events()[0]["kind"] == "action_rejected"
