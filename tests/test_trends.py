from backend.analysis.trends import compute, wilson


def U(tick, speaker, listeners, conv, uid):
    return {"tick": tick, "speaker": speaker, "listeners": listeners, "conversation_id": conv, "utterance_id": uid}


def _utts(n_per_tick, ticks):
    return [{"tick": t} for t in range(ticks) for _ in range(n_per_tick)]


def test_share_and_r_t():
    # A says it at t0 to B,C (conv c1); B carries it at t5 (c2) to D; C at t6 (c3); D at t9 (c4)
    us = [U(0, "A", ["B", "C"], "c1", "u1"), U(5, "B", ["D"], "c2", "u2"), U(6, "C", [], "c3", "u3"),
          U(9, "D", [], "c4", "u4")]
    r = compute(_utts(2, 12), [{"id": "x0", "phrase": "p", "usages": us}], tick=11, window=4, tpd=60, n_active=4)
    m = r["memes"][0]
    assert r["windows"] == [0, 4, 8] and r["population"]["talk_volume"] == [8, 8, 8]
    assert m["uses"] == [1, 2, 1] and m["share"] == [0.125, 0.25, 0.125]
    assert m["new_adopters"] == [1, 2, 1] and m["speakers_cum"] == [1, 3, 4]
    assert m["R_t"] == [2.0, 0.5, 0.0]           # A -> B,C ; B -> D, C -> nobody ; D -> nobody
    assert m["R"] == 0.75
    assert m["p_adopt"]["n_exposed"] == 3 and m["p_adopt"]["n_adopted"] == 3


def test_same_conversation_echo_is_not_secondary():
    us = [U(0, "A", ["B"], "c1", "u1"), U(1, "B", ["A"], "c1", "u2")]
    m = compute(_utts(1, 4), [{"id": "x", "phrase": "p", "usages": us}], tick=3, window=4, tpd=60, n_active=2)["memes"][0]
    assert m["R"] == 0 and m["p_adopt"]["n_adopted"] == 0


def test_half_life_and_stages():
    # used heavily in window 0, then once in window 2, nothing afterwards -> declining then extinct
    us = [U(t, "A", [], "c", f"u{t}") for t in range(4)] + [U(9, "B", [], "d", "v")]
    rows = [{"id": "x", "phrase": "p", "usages": us}]
    m = compute(_utts(4, 30), rows, tick=29, window=4, tpd=60, n_active=4)["memes"][0]
    assert m["peak"] == {"window": 0, "share": 0.25} and m["half_life_ticks"] == 4
    assert m["stage"] == "declining"
    m = compute(_utts(4, 60), rows, tick=59, window=4, tpd=60, n_active=4)["memes"][0]
    assert m["stage"] == "extinct"
    # new: one use in the latest window
    m = compute(_utts(4, 8), [{"id": "y", "phrase": "q", "usages": [U(7, "A", [], "c", "z")]}],
                tick=7, window=4, tpd=60, n_active=4)["memes"][0]
    assert m["stage"] == "spreading" and m["half_life_ticks"] is None


def test_established_across_days():
    us = [U(t, s, [], f"c{t}", f"u{t}") for t, s in [(1, "A"), (10, "B"), (61, "C"), (62, "A"), (63, "B")]]
    m = compute(_utts(1, 64), [{"id": "x", "phrase": "p", "usages": us}], tick=63, window=4, tpd=60, n_active=6)["memes"][0]
    assert m["stage"] == "established"


def test_wilson():
    lo, hi = wilson(3, 5)
    assert 0.2 < lo < 0.25 and 0.88 < hi < 0.89 and wilson(0, 0) == [None, None]


def test_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from tests.conftest import run_sim
    import backend.api.server as S
    run_sim(tmp_path, name="r1")
    monkeypatch.setattr(S, "RUNS", tmp_path.resolve())
    c = TestClient(S.app)
    j = c.get("/api/runs/r1/trends?window=4&top=5").json()
    for k in ("run_id", "tick", "window_ticks", "windows", "labels", "day_bounds", "sufficiency", "population", "memes"):
        assert k in j
    assert len(j["population"]["talk_volume"]) == len(j["windows"])
    j2 = c.get("/api/runs/r1/trends?tick=5&window=4").json()
    assert j2["tick"] <= 5
    if j["memes"]:
        mid = j["memes"][0]["id"]
        m = j["memes"][0]
        assert len(m["share"]) == len(j["windows"]) and m["stage"] in ("emerging", "spreading", "established", "declining", "extinct")
        d = c.get(f"/api/runs/r1/trends/meme/{mid}").json()
        assert d["meme"]["id"] == mid and "tree" in d and "adoptions" in d and "contexts" in d
    assert c.get("/api/runs/r1/trends/meme/nope").status_code == 404
