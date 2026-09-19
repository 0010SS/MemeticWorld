import time

from fastapi.testclient import TestClient

from app.main import app


def test_api_run_lifecycle():
    with TestClient(app) as client:
        assert client.get("/api/health").json()["llm_provider"] == "mock"
        world = client.get("/api/world").json()
        assert len(world["world"]["locations"]) == 7 and len(world["agents"]) == 8

        created = client.post("/api/runs", json={"days": 1, "seed": 11, "with_control": True}).json()
        full_id, control_id = created["run_ids"]
        for _ in range(200):
            runs = {r["id"]: r for r in client.get("/api/runs").json()}
            if runs[full_id]["status"] == "finished" and runs[control_id]["status"] == "finished":
                break
            time.sleep(0.05)
        assert runs[full_id]["status"] == "finished" and runs[control_id]["status"] == "finished"

        events = client.get(f"/api/runs/{full_id}/events").json()
        assert events and events[-1]["type"] == "run_end"
        with client.stream("GET", f"/api/runs/{full_id}/stream") as stream:
            body = "".join(stream.iter_text())
        assert "event: end" in body and body.count("data: {") > 10

        memes = client.get(f"/api/runs/{full_id}/memes").json()
        assert memes["control_run_id"] == control_id
        if memes["memes"]:
            detail = client.get(f"/api/runs/{full_id}/memes/{memes['memes'][0]['id']}").json()
            assert detail["nodes"] and detail["timeline"]
            glosses = client.post(f"/api/runs/{full_id}/memes/gloss?top=2").json()
            assert glosses
        compare = client.get(f"/api/compare?a={full_id}&b={control_id}").json()
        assert compare["a"]["condition"] == "full" and compare["b"]["condition"] == "no_speech_memory"

        agent = world["agents"][0]["id"]
        assert client.get(f"/api/runs/{full_id}/agents/{agent}/memories?max_tick=30").json()
        decisions = client.get(f"/api/runs/{full_id}/agents/{agent}/decisions").json()
        assert all("retrieved_memories" in d for d in decisions)
        assert client.delete(f"/api/runs/{control_id}").json()["ok"]
