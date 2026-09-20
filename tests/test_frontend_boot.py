import json
from fastapi.testclient import TestClient
from backend.api import server


def test_partial_recordings_do_not_break_frontend_lists(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    valid = tmp_path / 'valid'
    valid.mkdir()
    (valid / 'manifest.json').write_text(json.dumps({'config': {'llm': {'backend': 'mock'}, 'seed': 11},
        'world': {}, 'agents': {}, 'status': 'running'}))
    for i, content in enumerate(['{}', 'null', '[]', '{', '{"config": null}']):
        bad = tmp_path / str(i)
        bad.mkdir()
        (bad / 'manifest.json').write_text(content)
        (bad / 'analysis.json').write_text('{}')
    client = TestClient(server.app)
    response = client.get('/api/runs')
    assert response.status_code == 200
    assert [r['run_id'] for r in response.json()] == ['valid']
    assert client.get('/api/compare').status_code == 200
