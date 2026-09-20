import json
import socket
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


def test_killed_run_is_interrupted_without_rewriting_recording(tmp_path, monkeypatch):
    from backend.experiment import process
    monkeypatch.setattr(server, 'RUNS', tmp_path)
    monkeypatch.setattr(process, 'alive', lambda pid: pid == 123)
    manifests = {}
    for name, pid, host, status in [
        ('old', 456, socket.gethostname(), 'running'),
        ('live', 123, socket.gethostname(), 'running'),
        ('remote', 456, 'different-host', 'running'),
        ('done', 456, socket.gethostname(), 'finished'),
    ]:
        root = tmp_path / name
        root.mkdir()
        manifest = {'config': {'llm': {'backend': 'mock'}}, 'world': {}, 'agents': {}, 'status': status}
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        manifests[path] = path.read_bytes()
        (root / 'design_cell.json').write_text(json.dumps({'pid': pid, 'host': host}), encoding='utf-8')
    client = TestClient(server.app)
    assert {r['run_id']: r['status'] for r in client.get('/api/runs').json()} == {
        'old': 'interrupted', 'live': 'running', 'remote': 'running', 'done': 'finished'}
    assert client.get('/api/runs/old/manifest').json()['status'] == 'interrupted'
    for path, original in manifests.items():
        assert path.read_bytes() == original
